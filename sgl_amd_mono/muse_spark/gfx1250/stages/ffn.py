# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 wave32 K3 stages.

The implementation deliberately favors an explicit, auditable execution order:
one CTA computes one routed expert pick, one CTA finalizes one token and performs
the EP rank-order sum, and one CTA redundantly unpools and applies the boundary
for one token.  Native Quark MXFP4 tensors are decoded in software and multiplied
with the gfx1250 BF16 WMMA used by K1.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    hw_exp2,
    hw_rsq,
    rsrc,
)
from sgl_amd_mono.muse_spark.gfx1250.config import (
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOG2E,
    MOE_HIDDEN,
    POST_EPS,
    RMS_EPS,
    TOP_K,
)
THREADS = 256
WAVE_SIZE = 32
WAVES = 8
WMMA_K = 32
WMMA_ACC = 8
RED_FLOATS = WAVES * WAVE_SIZE * WMMA_ACC
ELEMS = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
W13_ROW_BYTES = MOE_HIDDEN // 2
W13_SCALE_COLS = MOE_HIDDEN // 32
W2_ROW_BYTES = INTER // 2
W2_SCALE_COLS = INTER // 32

assert INTER == MOE_HIDDEN == 4096
assert HIDDEN == 8192 and TOP_K == 8 and LOCAL_EXPERTS == 64


def _e8m0_f32(byte):
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def _mxfp4_scale(amax):
    bits = (amax.bitcast(fx.Int32) + fx.Int32(0x200000)) & fx.Int32(-0x800000)
    unbiased = fx.max(((bits >> 23) & 0xFF) - 129, fx.Int32(-127))
    return unbiased + 127, ((fx.Int32(127) - unbiased) << 23).bitcast(
        fx.Float32
    )


def _e2m1_code(x, qscale):
    bits = (x * qscale).bitcast(fx.Int32)
    negative = bits < 0
    absolute_bits = bits & fx.Int32(0x7FFFFFFF)
    absolute = absolute_bits.bitcast(fx.Float32)
    denormal = (
        (absolute + fx.Float32(4194304.0)).bitcast(fx.Int32)
        - fx.Int32(149 << 23)
    ) & 0xFF
    normal = (
        (
            absolute_bits
            + fx.Int32((-126 << 23) + 0x1FFFFF)
            + ((absolute_bits >> 22) & 1)
        )
        >> 22
    ) & 0xFF
    code = (absolute >= fx.Float32(6.0)).select(
        fx.Int32(7), (absolute < fx.Float32(1.0)).select(denormal, normal)
    )
    return code | negative.select(fx.Int32(8), fx.Int32(0))


def k3_lds():
    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        aux: fx.Array[fx.Float32, 64, 16]
        sums: fx.Array[fx.Float32, WAVES * 2, 16]

    return Lds


def _load_u8(resource, byte):
    word = fx.Int32(
        bo.buffer_load(resource, byte // 4, vec_width=1, dtype=T.i32)
    )
    return (word >> (8 * (byte % 4))) & 0xFF


def _decode_e2m1(code):
    mag = fx.Float32(E2M1[0])
    base = code & 7
    for i in range_constexpr(1, 8):
        mag = (base == i).select(fx.Float32(E2M1[i]), mag)
    return ((code & 8) != 0).select(-mag, mag)


def _decode_weight_fragment(r_w, r_s, row, k0, lane):
    """One native Quark row fragment as the WMMA v16bf16 A operand."""
    kh = (lane // 16) * 8
    vals = []
    for half in range_constexpr(2):
        for i in range_constexpr(8):
            k = k0 + half * 16 + kh + i
            byte = _load_u8(r_w, row * W13_ROW_BYTES + k // 2)
            code = (k & 1).select(byte >> 4, byte & 0xF)
            scale = _load_u8(r_s, row * W13_SCALE_COLS + k // 32)
            vals.append(_decode_e2m1(code) * _e8m0_f32(scale))
    return fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)


def _bf_fragment(resource, base, k0, lane):
    """One row-major BF16 vector, repeated in all WMMA B columns."""
    kh = (lane // 16) * 8
    lo = fx.Vector(
        bo.buffer_load(
            resource, (base + k0 + kh) // 2, vec_width=4, dtype=T.i32
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource, (base + k0 + 16 + kh) // 2, vec_width=4, dtype=T.i32
        )
    )
    return fx.Vector.from_elements(
        [lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32
    ).bitcast(fx.BFloat16)


def _wmma(a, b, acc):
    result = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(WMMA_ACC, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        reuseA=False,
        reuseB=False,
    ).result
    return fx.Vector(result)


def _tile_value(red, row):
    lane = ((row % 16) // 8) * 16
    elem = row % 8
    out = fx.Float32(0.0)
    for wave in range_constexpr(WAVES):
        out = out + fx.ptr_load(
            red + (wave * WAVE_SIZE + lane) * WMMA_ACC + elem
        )
    return out


@flyc.jit
def _mx_weight_gemv_tile(c, r_w, r_s, row0, x_addr, k):
    """Sixteen weight rows times one dynamically-quantized BF16 row."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    row = row0 + lane % 16
    r_x = rsrc(x_addr)
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range(0, k // (WAVES * WMMA_K)):
        k0 = (wave * (k // WAVES)) + step * WMMA_K
        acc = _wmma(
            _decode_weight_fragment(r_w, r_s, row, k0, lane),
            _bf_fragment(r_x, 0, k0, lane),
            acc,
        )
    fx.ptr_store(acc, red + (wave * WAVE_SIZE + lane) * WMMA_ACC)
    gpu.barrier()


@flyc.jit
def _bf_gemv_tile(c, r_w, row0, x_addr, k):
    """Sixteen BF16 rows times one BF16 row."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    row = row0 + lane % 16
    r_x = rsrc(x_addr)
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range(0, k // (WAVES * WMMA_K)):
        k0 = wave * (k // WAVES) + step * WMMA_K
        acc = _wmma(
            _bf_fragment(r_w, row * k, k0, lane),
            _bf_fragment(r_x, 0, k0, lane),
            acc,
        )
    fx.ptr_store(acc, red + (wave * WAVE_SIZE + lane) * WMMA_ACC)
    gpu.barrier()


@flyc.jit
def _quantize_row(c, src_addr, dst_addr, n):
    """Apply the exact Quark dynamic MXFP4 rule and store its BF16 dequantization."""
    tid, lane = c["tid"], c["lane"]
    r_src, r_dst = rsrc(src_addr), rsrc(dst_addr)
    for it in range_constexpr(n // THREADS):
        i = tid + it * THREADS
        x = fx.Float32(
            bo.buffer_load(r_src, i, vec_width=1, dtype=T.bf16)
        )
        amax = fx.coop.warp_reduce(abs(x), fx.ReductionOp.MAX, width=32)
        byte, qs = _mxfp4_scale(amax)
        value = _decode_e2m1(_e2m1_code(x, qs)) * _e8m0_f32(byte)
        bo.buffer_store(
            value.to(fx.BFloat16), r_dst, i, cache_modifier=16
        )
    gpu.barrier()


@flyc.jit
def _block_sum(c, value, slot):
    """CTA sum with wave32 reductions and one LDS scalar per wave."""
    lane, wave, sums = c["lane"], c["wave"], c["sums"]
    total = fx.coop.warp_reduce(value, fx.ReductionOp.ADD, width=32)
    if lane == 0:
        fx.ptr_store(total, sums + slot * WAVES + wave)
    gpu.barrier()
    out = fx.ptr_load(sums + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        out = out + fx.ptr_load(sums + slot * WAVES + w)
    gpu.barrier()
    return out


@flyc.jit
def stage_expert_pick(c):
    """Kernel 1: one CTA computes one ``(token, slot)`` local expert."""
    tid, pick = c["tid"], c["bid"]
    token, slot = pick // TOP_K, pick % TOP_K
    a = c["args"]
    expert = fx.Int32(
        bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
    )
    local = expert - a["local_start"]
    if (local >= 0) & (local < LOCAL_EXPERTS):
        qx = c["qx"].value + fx.Int64(pick * MOE_HIDDEN * 2)
        qmid = c["qmid"].value + fx.Int64(pick * INTER * 2)
        y = c["y"].value + fx.Int64(pick * MOE_HIDDEN * 2)
        xin = a["xin"] + fx.Int64(token * MOE_HIDDEN * 2)
        _quantize_row(c, xin, qx, MOE_HIDDEN)

        w13_base = fx.Int64(local) * (2 * INTER * W13_ROW_BYTES)
        s13_base = fx.Int64(local) * (2 * INTER * W13_SCALE_COLS)
        r13 = rsrc(a["w13"] + w13_base)
        rs13 = rsrc(a["s13"] + s13_base)
        # Two adjacent 16-row tiles form one MXFP4 activation block.
        for block in range(0, INTER // 32):
            for half in range_constexpr(2):
                row0 = block * 32 + half * 16
                _mx_weight_gemv_tile(c, r13, rs13, row0, qx, MOE_HIDDEN)
                if tid < 16:
                    fx.ptr_store(
                        _tile_value(c["red"], tid), c["aux"] + 32 + tid
                    )
                gpu.barrier()
                _mx_weight_gemv_tile(
                    c, r13, rs13, INTER + row0, qx, MOE_HIDDEN
                )
                if tid < 16:
                    gate = fx.ptr_load(c["aux"] + 32 + tid)
                    up = _tile_value(c["red"], tid)
                    mid = bf16_round(
                        gate
                        * (fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(-gate * LOG2E)))
                        * up
                    )
                    fx.ptr_store(mid, c["aux"] + half * 16 + tid)
                gpu.barrier()
            if tid < 32:
                v = fx.ptr_load(c["aux"] + tid)
                amax = fx.coop.warp_reduce(abs(v), fx.ReductionOp.MAX, width=32)
                byte, qs = _mxfp4_scale(amax)
                qv = _decode_e2m1(_e2m1_code(v, qs)) * _e8m0_f32(byte)
                bo.buffer_store(
                    qv.to(fx.BFloat16),
                    rsrc(qmid),
                    block * 32 + tid,
                    cache_modifier=16,
                )
            gpu.barrier()

        w2_base = fx.Int64(local) * (MOE_HIDDEN * W2_ROW_BYTES)
        s2_base = fx.Int64(local) * (MOE_HIDDEN * W2_SCALE_COLS)
        r2, rs2 = rsrc(a["w2"] + w2_base), rsrc(a["s2"] + s2_base)
        for tile in range(0, MOE_HIDDEN // 16):
            _mx_weight_gemv_tile(c, r2, rs2, tile * 16, qmid, INTER)
            if tid < 16:
                bo.buffer_store(
                    bf16_round(_tile_value(c["red"], tid)).to(fx.BFloat16),
                    rsrc(y),
                    tile * 16 + tid,
                    cache_modifier=16,
                )
            gpu.barrier()


@flyc.jit
def stage_finalize_ep(c):
    """Kernel 2: local slot finalize, peer publish, rank-order BF16 sum."""
    tid, token = c["tid"], c["bid"]
    a, put_bf, poll = c["args"], c["put_bf"], c["poll"]
    values = [fx.Float32(0.0) for _ in range(ELEMS)]
    col0 = tid * ELEMS
    pen = []
    for q in range_constexpr(ELEMS // 2):
        lo, hi = bf2_f32(
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["pen_w"]), col0 // 2 + q, vec_width=1, dtype=T.i32
                )
            )
        )
        pen += [lo, hi]
    for slot in range_constexpr(TOP_K):
        pick = token * TOP_K + slot
        expert = fx.Int32(
            bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
        )
        local = expert - a["local_start"]
        held = (local >= 0) & (local < LOCAL_EXPERTS)
        vv = []
        local_ss = fx.Float32(0.0)
        for q in range_constexpr(ELEMS // 2):
            word = fx.Int32(
                bo.buffer_load(
                    rsrc(c["y"].value),
                    pick * (MOE_HIDDEN // 2) + col0 // 2 + q,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            lo, hi = bf2_f32(word)
            v0 = held.select(bf16_round(lo * pen[2 * q]), fx.Float32(0.0))
            v1 = held.select(bf16_round(hi * pen[2 * q + 1]), fx.Float32(0.0))
            vv += [v0, v1]
            local_ss = local_ss + v0 * v0 + v1 * v1
        ss = _block_sum(c, local_ss, slot % 2)
        score = fx.Float32(
            bo.buffer_load(rsrc(a["scores"]), pick, vec_width=1, dtype=T.f32)
        )
        scale = held.select(
            score * hw_rsq(ss / float(MOE_HIDDEN) + POST_EPS),
            fx.Float32(0.0),
        )
        for i in range_constexpr(ELEMS):
            values[i] = values[i] + scale * vv[i]

    partial = [bf16_round(v) for v in values]
    # Keep peer stores at one payload/tag pair.  gfx1250 cannot select the
    # v16i32 system-scope store produced by publishing all 16 values at once.
    for pe in range_constexpr(c["W"]):
        for q in range_constexpr(ELEMS // 2):
            put_bf(
                c["peer_partial"][pe],
                (c["rank"] * c["S"] + token) * MOE_HIDDEN + col0 + 2 * q,
                [partial[2 * q], partial[2 * q + 1]],
            )
    for q in range_constexpr(ELEMS // 2):
        got = poll(
            [
                (
                    c["own_partial"],
                    (src * c["S"] + token) * (MOE_HIDDEN // 2) + col0 // 2 + q,
                    1,
                )
                for src in range(c["W"])
            ]
        )
        lo = fx.Float32(0.0)
        hi = fx.Float32(0.0)
        for src in range_constexpr(c["W"]):
            x0, x1 = bf2_f32(got[src][0])
            lo, hi = lo + x0, hi + x1
        bo.buffer_store(
            bf16_pair(lo, hi).bitcast(fx.Int32),
            rsrc(c["fbuf"].value),
            token * (MOE_HIDDEN // 2) + col0 // 2 + q,
            cache_modifier=16,
        )


@flyc.jit
def stage_unpool_boundary(c):
    """Kernel 3: replicated unpool GEMV and the complete FFN boundary."""
    tid, token = c["tid"], c["bid"]
    a = c["args"]
    frow = c["fbuf"].value + fx.Int64(token * MOE_HIDDEN * 2)
    branch = c["branch"].value + fx.Int64(token * HIDDEN * 2)
    r_unp = rsrc(a["w_unpool"])
    for tile in range(0, HIDDEN // 16):
        _bf_gemv_tile(c, r_unp, tile * 16, frow, MOE_HIDDEN)
        if tid < 16:
            bo.buffer_store(
                bf16_round(_tile_value(c["red"], tid)).to(fx.BFloat16),
                rsrc(branch),
                tile * 16 + tid,
                cache_modifier=16,
            )
        gpu.barrier()

    e0 = tid * 32
    post = []
    local_ss = fx.Float32(0.0)
    for q in range_constexpr(16):
        b0, b1 = bf2_f32(
            fx.Int32(
                bo.buffer_load(
                    rsrc(branch), e0 // 2 + q, vec_width=1, dtype=T.i32
                )
            )
        )
        w0, w1 = bf2_f32(
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["pfw"]), e0 // 2 + q, vec_width=1, dtype=T.i32
                )
            )
        )
        p0, p1 = bf16_round(b0 * w0), bf16_round(b1 * w1)
        post += [p0, p1]
        local_ss = local_ss + p0 * p0 + p1 * p1
    rinv = hw_rsq(_block_sum(c, local_ss, 0) / float(HIDDEN) + POST_EPS)
    new = []
    local_pre_ss = fx.Float32(0.0)
    is_last = a["last"] != 0
    for q in range_constexpr(8):
        al = fx.Vector(
            bo.buffer_load(rsrc(a["alpha"]), e0 + 4 * q, vec_width=4, dtype=T.f32)
        )
        be = fx.Vector(
            bo.buffer_load(rsrc(a["beta"]), e0 + 4 * q, vec_width=4, dtype=T.f32)
        )
        old = fx.Vector(
            bo.buffer_load(
                rsrc(a["res"]),
                token * HIDDEN + e0 + 4 * q,
                vec_width=4,
                dtype=T.f32,
            )
        )
        nv = []
        for j in range_constexpr(4):
            n = bf16_round(post[4 * q + j] * rinv)
            v = fx.Float32(al[j]) * fx.Float32(old[j]) + fx.Float32(be[j]) * n
            nv.append(v)
            pre = is_last.select(bf16_round(v), v)
            local_pre_ss = local_pre_ss + pre * pre
        new += nv
        bo.buffer_store(
            fx.Vector.from_elements(nv, fx.Float32),
            rsrc(a["res"]),
            token * HIDDEN + e0 + 4 * q,
        )
    pre_rinv = hw_rsq(
        _block_sum(c, local_pre_ss, 1) / float(HIDDEN) + RMS_EPS
    )
    for q in range_constexpr(16):
        nw0, nw1 = bf2_f32(
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["npw"]), e0 // 2 + q, vec_width=1, dtype=T.i32
                )
            )
        )
        i = 2 * q
        p0 = is_last.select(bf16_round(new[i]), new[i])
        p1 = is_last.select(bf16_round(new[i + 1]), new[i + 1])
        bo.buffer_store(
            bf16_pair(p0 * pre_rinv * nw0, p1 * pre_rinv * nw1)
            .bitcast(fx.Int32),
            rsrc(a["attn_out"]),
            token * (HIDDEN // 2) + e0 // 2 + q,
            cache_modifier=16,
        )
