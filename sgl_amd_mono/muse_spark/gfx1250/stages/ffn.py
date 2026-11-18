# SPDX-License-Identifier: Apache-2.0
"""gfx1250 wave32 K3 stages.

The implementation deliberately favors an explicit, auditable execution order:
one CTA computes one routed expert pick, one CTA finalizes one token and performs
the EP rank-order sum, and one CTA redundantly unpools and applies the boundary
for one token. Expert GEMMs consume packed MXFP8 activations and native Quark
MXFP4 weights with gfx1250 mixed scaled WMMA.
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
SCALED_WMMA_K = 128
WMMA_ACC = 8
RED_FLOATS = WAVES * WAVE_SIZE * WMMA_ACC
ELEMS = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_INV_MAX = 0x3B124925
W13_ROW_BYTES = MOE_HIDDEN // 2
W13_SCALE_COLS = MOE_HIDDEN // 32
W2_ROW_BYTES = INTER // 2
W2_SCALE_COLS = INTER // 32

assert INTER == MOE_HIDDEN == 4096
assert HIDDEN == 8192 and TOP_K == 8 and LOCAL_EXPERTS == 64


def _e8m0_f32(byte):
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def _mxfp8_quant(x, amax):
    """AITER per-1x32 MXFP8 quantization as E4M3 payload and E8M0 scale."""
    safe_max = fx.max(amax, fx.Float32(1e-10))
    working = (safe_max * fx.Int32(FP8_INV_MAX).bitcast(fx.Float32)).bitcast(
        fx.Int32
    )
    mantissa = working & fx.Int32(0x7FFFFF)
    exponent = (working >> 23) & fx.Int32(0xFF)
    e8m0 = (mantissa != 0).select(exponent + 1, exponent)
    e8m0 = fx.min(e8m0, fx.Int32(255))
    quant_scale = ((fx.Int32(254) - e8m0) << 23).bitcast(fx.Float32)
    dequant_scale = (e8m0 << 23).bitcast(fx.Float32)
    packed = rocdl.cvt_pk_fp8_f32(
        T.i32, x * quant_scale, fx.Float32(0.0), fx.Int32(0), 0
    )
    code = fx.Int32(packed) & 0xFF
    return code, e8m0


def _mxfp8_value(x, amax):
    """AITER MXFP8 quantization, returned as its dequantized f32."""
    code, e8m0 = _mxfp8_quant(x, amax)
    exponent, mantissa = (code >> 3) & 0xF, code & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    decoded = (exponent == 0).select(
        fx.Float32(mantissa) * fx.Float32(1.0 / 512.0),
        normal.bitcast(fx.Float32),
    )
    decoded = ((code & 0x80) != 0).select(-decoded, decoded)
    return decoded * dequant_scale


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


def _shuffled_weight_offset(row, k_byte):
    """AITER shuffle_weight((16, 16)) byte offset within one expert."""
    return (
        (((row // 16) * (W13_ROW_BYTES // 32) + k_byte // 32) * 2
          + (k_byte % 32) // 16)
        * 16
        + row % 16
    ) * 16 + k_byte % 16


def _shuffled_scale_offset(row, k_group):
    """AITER n32k4 scale byte offset within one expert."""
    return (
        (row // 32) * (W13_SCALE_COLS * 32)
        + (k_group // 4) * 128
        + (row % 32) * 4
        + k_group % 4
    )


def _decode_weight_fragment(r_w, r_s, row, k0, lane, shuffled):
    """One native-Quark or AITER-shuffled row fragment for WMMA."""
    kh = (lane // 16) * 8
    vals = []
    for half in range_constexpr(2):
        k_base = k0 + half * 16 + kh
        byte_base = k_base // 2
        weight_offset = (
            _shuffled_weight_offset(row, byte_base)
            if shuffled
            else row * W13_ROW_BYTES + byte_base
        )
        packed = fx.Int32(
            bo.buffer_load(r_w, weight_offset // 4, vec_width=1, dtype=T.i32)
        )
        scale_offset = (
            _shuffled_scale_offset(row, k_base // 32)
            if shuffled
            else row * W13_SCALE_COLS + k_base // 32
        )
        scale = _load_u8(r_s, scale_offset)
        for i in range_constexpr(8):
            byte = (packed >> (8 * (i // 2))) & 0xFF
            code = byte >> 4 if i & 1 else byte & 0xF
            vals.append(_decode_e2m1(code) * _e8m0_f32(scale))
    return fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)


def _packed_weight_fragment(r_w, row, k0, lane, shuffled):
    """One FP4 A operand for a 16x16x128 mixed scaled-WMMA."""
    lane_k = lane // 16
    bytes_ = [
        k0 // 2 + block * 32 + lane_k * 16
        for block in range(2)
    ]
    if shuffled:
        chunks = [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    _shuffled_weight_offset(row, byte) // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for byte in bytes_
        ]
        return fx.Vector.from_elements(
            [chunks[block][i] for block in range(2) for i in range(4)],
            fx.Int32,
        )
    chunks = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                (row * W13_ROW_BYTES + byte) // 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for byte in bytes_
    ]
    return fx.Vector.from_elements(
        [chunks[block][i] for block in range(2) for i in range(4)],
        fx.Int32,
    )


def _weight_scale_pack(r_s, row, k0, shuffled):
    group = k0 // 32
    byte = (
        _shuffled_scale_offset(row, group)
        if shuffled
        else row * W13_SCALE_COLS + group
    )
    return fx.Int32(bo.buffer_load(r_s, byte // 4, vec_width=1, dtype=T.i32))


def _packed_activation_fragment(resource, n, k0, lane):
    """One E4M3 B operand; the same row is broadcast over 16 columns."""
    lane_k = lane // 16
    chunks = [
        fx.Vector(
            bo.buffer_load(
                resource,
                (k0 + i * 32 + lane_k * 16) // 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for i in range(4)
    ]
    return fx.Vector.from_elements(
        [chunks[i][j] for i in range(4) for j in range(4)],
        fx.Int32,
    )


def _activation_scale_pack(resource, n, k0):
    return fx.Int32(
        bo.buffer_load(resource, (n + k0 // 32) // 4, vec_width=1, dtype=T.i32)
    )


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


def _bf_tokens_fragment(resource, k, k0, lane, tokens):
    """One BF16 B operand with one activation row per live WMMA column."""
    token = lane % 16
    kh = (lane // 16) * 8
    base = token * k
    lo = fx.Vector(
        bo.buffer_load(
            resource, (base + k0 + kh) // 2, vec_width=4, dtype=T.i32
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + 16 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
        )
    )
    valid = token < tokens
    words = [valid.select(lo[i], fx.Int32(0)) for i in range(4)]
    words += [valid.select(hi[i], fx.Int32(0)) for i in range(4)]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


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


def _wmma_scaled(a, b, acc, scale_a, scale_b):
    result = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        fx.Vector.make_type(WMMA_ACC, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        fx.as_ir_value(scale_a),
        fx.as_ir_value(scale_b),
        fmtA=4,
        fmtB=0,
        reuseA=False,
        reuseB=False,
    )
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


def _tile_token_value(red, token, row):
    lane = ((row % 16) // 8) * 16 + token
    elem = row % 8
    out = fx.Float32(0.0)
    for wave in range_constexpr(WAVES):
        out = out + fx.ptr_load(
            red + (wave * WAVE_SIZE + lane) * WMMA_ACC + elem
        )
    return out


@flyc.jit
def _mx_weight_gemv_tile(c, r_w, r_s, row0, x_addr, k, gate_up=0):
    """Sixteen weight rows times one dynamically-quantized BF16 row."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    row = row0 + lane % 16
    if c["shuffled"] and gate_up:
        # AITER interleaves stage-1 rows as g0,u0,g1,u1,...
        row = row * 2 + (gate_up - 1)
    r_x = rsrc(x_addr)
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range(0, k // (WAVES * SCALED_WMMA_K)):
        k0 = (wave * (k // WAVES)) + step * SCALED_WMMA_K
        acc = _wmma_scaled(
            _packed_weight_fragment(r_w, row, k0, lane, c["shuffled"]),
            _packed_activation_fragment(r_x, k, k0, lane),
            acc,
            _weight_scale_pack(r_s, row, k0, c["shuffled"]),
            _activation_scale_pack(r_x, k, k0),
        )
    fx.ptr_store(acc, red + (wave * WAVE_SIZE + lane) * WMMA_ACC)
    gpu.barrier()


@flyc.jit
def _mx_weight_gate_up_32(c, r_w, r_s, row0, x_addr, k):
    """Two 16-row Gate/Up tiles sharing one activation load per K step."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    r_x = rsrc(x_addr)
    gate_acc = [
        fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        for _ in range(2)
    ]
    up_acc = [
        fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        for _ in range(2)
    ]
    for step in range(0, k // (WAVES * SCALED_WMMA_K)):
        k0 = (wave * (k // WAVES)) + step * SCALED_WMMA_K
        activation = _packed_activation_fragment(r_x, k, k0, lane)
        activation_scale = _activation_scale_pack(r_x, k, k0)
        for half in range_constexpr(2):
            row = row0 + half * 16 + lane % 16
            gate_row = row * 2 if c["shuffled"] else row
            up_row = row * 2 + 1 if c["shuffled"] else INTER + row
            gate_acc[half] = _wmma_scaled(
                _packed_weight_fragment(
                    r_w, gate_row, k0, lane, c["shuffled"]
                ),
                activation,
                gate_acc[half],
                _weight_scale_pack(
                    r_s, gate_row, k0, c["shuffled"]
                ),
                activation_scale,
            )
            up_acc[half] = _wmma_scaled(
                _packed_weight_fragment(
                    r_w, up_row, k0, lane, c["shuffled"]
                ),
                activation,
                up_acc[half],
                _weight_scale_pack(
                    r_s, up_row, k0, c["shuffled"]
                ),
                activation_scale,
            )

    for half in range_constexpr(2):
        fx.ptr_store(
            gate_acc[half],
            red + (wave * WAVE_SIZE + lane) * WMMA_ACC,
        )
        gpu.barrier()
        if c["tid"] < 16:
            fx.ptr_store(
                _tile_value(red, c["tid"]), c["aux"] + 32 + c["tid"]
            )
        gpu.barrier()
        fx.ptr_store(
            up_acc[half],
            red + (wave * WAVE_SIZE + lane) * WMMA_ACC,
        )
        gpu.barrier()
        if c["tid"] < 16:
            gate = fx.ptr_load(c["aux"] + 32 + c["tid"])
            up = _tile_value(red, c["tid"])
            mid = bf16_round(
                gate
                * (
                    fx.Float32(1.0)
                    / (fx.Float32(1.0) + hw_exp2(-gate * LOG2E))
                )
                * up
            )
            fx.ptr_store(mid, c["aux"] + half * 16 + c["tid"])
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
def _bf_gemm_tokens_tile(c, r_w, row0, x_addr, k):
    """Sixteen BF16 weight rows times every live token in one WMMA tile."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    row = row0 + lane % 16
    r_x = rsrc(x_addr)
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range(0, k // (WAVES * WMMA_K)):
        k0 = wave * (k // WAVES) + step * WMMA_K
        acc = _wmma(
            _bf_fragment(r_w, row * k, k0, lane),
            _bf_tokens_fragment(r_x, k, k0, lane, c["S"]),
            acc,
        )
    fx.ptr_store(acc, red + (wave * WAVE_SIZE + lane) * WMMA_ACC)
    gpu.barrier()


@flyc.jit
def _quantize_row(c, src_addr, dst_addr, n):
    """Store packed E4M3 payload followed by one E8M0 byte per 32 values."""
    tid, lane = c["tid"], c["lane"]
    r_src, r_dst = rsrc(src_addr), rsrc(dst_addr)
    for it in range_constexpr(n // THREADS):
        i = tid + it * THREADS
        x = fx.Float32(
            bo.buffer_load(r_src, i, vec_width=1, dtype=T.bf16)
        )
        amax = fx.coop.warp_reduce(abs(x), fx.ReductionOp.MAX, width=32)
        code, e8m0 = _mxfp8_quant(x, amax)
        bo.buffer_store(fx.Int8(code), r_dst, i, cache_modifier=16)
        if lane == 0:
            bo.buffer_store(
                fx.Int8(e8m0),
                r_dst,
                n + i // 32,
                cache_modifier=16,
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
def stage_expert_quantize(c, pick):
    """Quantize one routed pick before CTA-parallel expert GEMMs."""
    token = pick // TOP_K
    a = c["args"]
    expert = fx.Int32(
        bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
    )
    local = expert - a["local_start"]
    if (local >= 0) & (local < LOCAL_EXPERTS):
        qx = c["qx"].value + fx.Int64(pick * MOE_HIDDEN * 2)
        xin = a["xin"] + fx.Int64(token * MOE_HIDDEN * 2)
        _quantize_row(c, xin, qx, MOE_HIDDEN)


@flyc.jit
def stage_expert_gate_up_block(c, task):
    """Compute one routed pick's 32-row Gate/Up/SiLU block."""
    blocks = INTER // 32
    pick, block = task // blocks, task % blocks
    a, tid = c["args"], c["tid"]
    expert = fx.Int32(
        bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
    )
    local = expert - a["local_start"]
    if (local >= 0) & (local < LOCAL_EXPERTS):
        qx = c["qx"].value + fx.Int64(pick * MOE_HIDDEN * 2)
        qmid = c["qmid"].value + fx.Int64(pick * INTER * 2)
        w13_base = fx.Int64(local) * (2 * INTER * W13_ROW_BYTES)
        s13_base = fx.Int64(local) * (2 * INTER * W13_SCALE_COLS)
        r13 = rsrc(a["w13"] + w13_base)
        rs13 = rsrc(a["s13"] + s13_base)
        _mx_weight_gate_up_32(
            c, r13, rs13, block * 32, qx, MOE_HIDDEN
        )
        if tid < 32:
            v = fx.ptr_load(c["aux"] + tid)
            amax = fx.coop.warp_reduce(abs(v), fx.ReductionOp.MAX, width=32)
            code, e8m0 = _mxfp8_quant(v, amax)
            bo.buffer_store(
                fx.Int8(code),
                rsrc(qmid),
                block * 32 + tid,
                cache_modifier=16,
            )
            if tid == 0:
                bo.buffer_store(
                    fx.Int8(e8m0),
                    rsrc(qmid),
                    INTER + block,
                    cache_modifier=16,
                )
        gpu.barrier()


@flyc.jit
def stage_expert_down_tile(c, task):
    """Compute one routed pick's 16-row Down tile."""
    tiles = MOE_HIDDEN // 16
    pick, tile = task // tiles, task % tiles
    a, tid = c["args"], c["tid"]
    expert = fx.Int32(
        bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
    )
    local = expert - a["local_start"]
    if (local >= 0) & (local < LOCAL_EXPERTS):
        qmid = c["qmid"].value + fx.Int64(pick * INTER * 2)
        y = c["y"].value + fx.Int64(pick * MOE_HIDDEN * 2)
        w2_base = fx.Int64(local) * (MOE_HIDDEN * W2_ROW_BYTES)
        s2_base = fx.Int64(local) * (MOE_HIDDEN * W2_SCALE_COLS)
        r2, rs2 = rsrc(a["w2"] + w2_base), rsrc(a["s2"] + s2_base)
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
def stage_expert_pick(c, task=None):
    """Kernel 1: one CTA computes one ``(token, slot)`` local expert."""
    tid, pick = c["tid"], c["bid"] if task is None else task
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
                _mx_weight_gemv_tile(
                    c, r13, rs13, row0, qx, MOE_HIDDEN, gate_up=1
                )
                if tid < 16:
                    fx.ptr_store(
                        _tile_value(c["red"], tid), c["aux"] + 32 + tid
                    )
                gpu.barrier()
                _mx_weight_gemv_tile(
                    c,
                    r13,
                    rs13,
                    row0 if c["shuffled"] else INTER + row0,
                    qx,
                    MOE_HIDDEN,
                    gate_up=2,
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
                code, e8m0 = _mxfp8_quant(v, amax)
                bo.buffer_store(
                    fx.Int8(code),
                    rsrc(qmid),
                    block * 32 + tid,
                    cache_modifier=16,
                )
                if tid == 0:
                    bo.buffer_store(
                        fx.Int8(e8m0),
                        rsrc(qmid),
                        INTER + block,
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
def stage_finalize_scale(c, pick):
    """Compute one routed row's post-RMS score scale."""
    tid, a = c["tid"], c["args"]
    col0 = tid * ELEMS
    expert = fx.Int32(
        bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
    )
    local = expert - a["local_start"]
    held = (local >= 0) & (local < LOCAL_EXPERTS)
    local_ss = fx.Float32(0.0)
    for q in range_constexpr(ELEMS // 2):
        yword = fx.Int32(
            bo.buffer_load(
                rsrc(c["y"].value),
                pick * (MOE_HIDDEN // 2) + col0 // 2 + q,
                vec_width=1,
                dtype=T.i32,
            )
        )
        pword = fx.Int32(
            bo.buffer_load(
                rsrc(a["pen_w"]),
                col0 // 2 + q,
                vec_width=1,
                dtype=T.i32,
            )
        )
        y0, y1 = bf2_f32(yword)
        p0, p1 = bf2_f32(pword)
        v0 = held.select(bf16_round(y0 * p0), fx.Float32(0.0))
        v1 = held.select(bf16_round(y1 * p1), fx.Float32(0.0))
        local_ss = local_ss + v0 * v0 + v1 * v1
    ss = _block_sum(c, local_ss, pick % 2)
    score = fx.Float32(
        bo.buffer_load(rsrc(a["scores"]), pick, vec_width=1, dtype=T.f32)
    )
    scale = held.select(
        score * hw_rsq(ss / float(MOE_HIDDEN) + POST_EPS),
        fx.Float32(0.0),
    )
    if tid == 0:
        bo.buffer_store(
            scale,
            rsrc(c["finalize_scale"].value),
            pick,
            cache_modifier=16,
        )


@flyc.jit
def stage_finalize_publish(c, task, chunks):
    """Combine one column chunk, exchange rank partials, and reduce in order."""
    tid, a = c["tid"], c["args"]
    token, chunk = task // chunks, task % chunks
    pairs = MOE_HIDDEN // 2
    pairs_per_chunk = (pairs + chunks - 1) // chunks
    pair = chunk * pairs_per_chunk + tid
    if (tid < pairs_per_chunk) & (pair < pairs):
        pword = fx.Int32(
            bo.buffer_load(
                rsrc(a["pen_w"]),
                pair,
                vec_width=1,
                dtype=T.i32,
            )
        )
        p0, p1 = bf2_f32(pword)
        value0, value1 = fx.Float32(0.0), fx.Float32(0.0)
        for slot in range_constexpr(TOP_K):
            pick = token * TOP_K + slot
            expert = fx.Int32(
                bo.buffer_load(rsrc(a["ids"]), pick, vec_width=1, dtype=T.i32)
            )
            local = expert - a["local_start"]
            held = (local >= 0) & (local < LOCAL_EXPERTS)
            yword = fx.Int32(
                bo.buffer_load(
                    rsrc(c["y"].value),
                    pick * pairs + pair,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            y0, y1 = bf2_f32(yword)
            v0 = held.select(bf16_round(y0 * p0), fx.Float32(0.0))
            v1 = held.select(bf16_round(y1 * p1), fx.Float32(0.0))
            scale = fx.Float32(
                bo.buffer_load(
                    rsrc(c["finalize_scale"].value),
                    pick,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            value0, value1 = value0 + scale * v0, value1 + scale * v1

        partial0, partial1 = bf16_round(value0), bf16_round(value1)
        for pe in range_constexpr(c["W"]):
            c["put_bf"](
                c["peer_partial"][pe],
                (c["rank"] * c["S"] + token) * MOE_HIDDEN + pair * 2,
                [partial0, partial1],
            )
        got = c["poll"](
            [
                (
                    c["own_partial"],
                    (src * c["S"] + token) * pairs + pair,
                    1,
                )
                for src in range(c["W"])
            ]
        )
        out0, out1 = fx.Float32(0.0), fx.Float32(0.0)
        for src in range_constexpr(c["W"]):
            x0, x1 = bf2_f32(got[src][0])
            out0, out1 = out0 + x0, out1 + x1
        bo.buffer_store(
            bf16_pair(out0, out1).bitcast(fx.Int32),
            rsrc(c["fbuf"].value),
            token * pairs + pair,
            cache_modifier=16,
        )


@flyc.jit
def stage_unpool_gemv(c, task):
    """Compute one 16-row unpool tile for one token."""
    tiles = HIDDEN // 16
    token, tile = task // tiles, task % tiles
    a = c["args"]
    frow = c["fbuf"].value + fx.Int64(token * MOE_HIDDEN * 2)
    branch = c["branch"].value + fx.Int64(token * HIDDEN * 2)
    r_unp = rsrc(a["w_unpool"])
    _bf_gemv_tile(c, r_unp, tile * 16, frow, MOE_HIDDEN)
    if c["tid"] < 16:
        bo.buffer_store(
            bf16_round(_tile_value(c["red"], c["tid"])).to(fx.BFloat16),
            rsrc(branch),
            tile * 16 + c["tid"],
            cache_modifier=16,
        )


@flyc.jit
def stage_unpool_tokens_gemv(c, tile):
    """Compute one 16-row unpool tile for every verify token."""
    a = c["args"]
    r_unp = rsrc(a["w_unpool"])
    _bf_gemm_tokens_tile(c, r_unp, tile * 16, c["fbuf"].value, MOE_HIDDEN)
    for store_round in range_constexpr((c["S"] * 16 + THREADS - 1) // THREADS):
        item = c["tid"] + store_round * THREADS
        if item < c["S"] * 16:
            token, row = item // 16, item % 16
            bo.buffer_store(
                bf16_round(
                    _tile_token_value(c["red"], token, row)
                ).to(fx.BFloat16),
                rsrc(c["branch"].value),
                token * HIDDEN + tile * 16 + row,
                cache_modifier=16,
            )
    gpu.barrier()


@flyc.jit
def stage_unpool_post(c, token=None):
    """Apply the complete FFN boundary after all unpool tiles are visible."""
    tid = c["tid"]
    token = c["bid"] if token is None else token
    a = c["args"]
    branch = c["branch"].value + fx.Int64(token * HIDDEN * 2)
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


@flyc.jit
def stage_unpool_boundary(c):
    """Diagnostic three-launch path: unpool GEMV then complete boundary."""
    token = c["bid"]
    for tile in range(0, HIDDEN // 16):
        stage_unpool_gemv(c, token * (HIDDEN // 16) + tile)
        gpu.barrier()
    stage_unpool_post(c, token)
