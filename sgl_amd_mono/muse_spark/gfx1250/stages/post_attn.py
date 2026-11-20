# SPDX-License-Identifier: Apache-2.0
"""Wave32-WMMA stages for the fixed-width gfx1250 Muse Spark K2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    rsrc,
    xshfl,
)
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)

THREADS = 256
WAVES = THREADS // 32
ROWS = 16
WMMA_K = 32
WMMA_ACC = 8
RED_FLOATS = WAVES * 32 * WMMA_ACC
RP_X_ROWS = 8
RP_X_TILE_K = 1024
RP_X_PAIRS = RP_X_ROWS * RP_X_TILE_K // 2
JOINT_ROWS = MOE_HIDDEN + 2 * EXPERTS
JOINT_WAVES = 4
JOINT_TASKS = JOINT_ROWS // (ROWS * JOINT_WAVES)
JOINT_TDM_TASKS = JOINT_ROWS // ROWS
JOINT_TDM_K = 512
JOINT_TDM_BUFFERS = 4
JOINT_TDM_PAD = 8
JOINT_TDM_STRIDE = JOINT_TDM_K + JOINT_TDM_PAD
JOINT_TDM_A_ROWS = ROWS
JOINT_TDM_B_ROWS = ROWS
JOINT_TDM_A_ELEMS = JOINT_TDM_A_ROWS * JOINT_TDM_STRIDE + JOINT_TDM_PAD
JOINT_TDM_B_ELEMS = JOINT_TDM_B_ROWS * JOINT_TDM_STRIDE + JOINT_TDM_PAD
JOINT_TDM_SLOT_ELEMS = JOINT_TDM_A_ELEMS + JOINT_TDM_B_ELEMS
JOINT_TDM_A_CHUNKS = JOINT_TDM_A_ROWS * JOINT_TDM_K // 8
JOINT_TDM_B_CHUNKS = JOINT_TDM_B_ROWS * JOINT_TDM_K // 8
JOINT_TDM_COPIES_PER_THREAD = (
    JOINT_TDM_A_CHUNKS + JOINT_TDM_B_CHUNKS
) // THREADS
JOINT_TDM_K_TILES = HIDDEN // JOINT_TDM_K


def joint_tdm_lds_bytes():
    """Four padded M16xK512 / N16xK512 BF16 slots."""
    return 2 * JOINT_TDM_BUFFERS * JOINT_TDM_SLOT_ELEMS


def joint_tdm_copy_coordinates(chunk):
    """CPU-visible mapping for one 16-byte async-copy chunk."""
    if not 0 <= chunk < JOINT_TDM_A_CHUNKS + JOINT_TDM_B_CHUNKS:
        raise ValueError(f"joint TDM chunk out of range: {chunk}")
    operand = "a"
    if chunk >= JOINT_TDM_A_CHUNKS:
        operand = "b"
        chunk -= JOINT_TDM_A_CHUNKS
    return operand, chunk // (JOINT_TDM_K // 8), (chunk % (JOINT_TDM_K // 8)) * 8


def _bf(ptr, element):
    word = fx.Int32(
        bo.buffer_load(rsrc(ptr), element // 2, vec_width=1, dtype=T.i32)
    )
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _f32(ptr, element):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), element, vec_width=1, dtype=T.f32)
    )


def _st_bf2(ptr, pair, lo, hi, cache_modifier=0):
    bo.buffer_store(
        bf16_pair(lo, hi).bitcast(fx.Int32),
        rsrc(ptr),
        pair,
        cache_modifier=cache_modifier,
    )


def _sigmoid(x):
    return fx.Float32(1.0) / (
        fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E))
    )


def _fragment(resource, base, k0, lane, cache_modifier=0):
    kh = (lane // 16) * 8
    lo = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + 16 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    return fx.Vector.from_elements(
        [lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32
    ).bitcast(fx.BFloat16)


def _lds_fragment(pairs, base, k0, lane):
    """One WMMA B fragment from an LDS row stored as packed bf16 pairs."""
    V4F = fx.Vector.make_type(4, fx.Float32)
    kh = (lane // 16) * 8
    lo = fx.Vector(
        fx.ptr_load(pairs + (base + k0 + kh) // 2, result_type=V4F)
    ).bitcast(fx.BFloat16)
    hi = fx.Vector(
        fx.ptr_load(pairs + (base + k0 + 16 + kh) // 2, result_type=V4F)
    ).bitcast(fx.BFloat16)
    return fx.Vector.from_elements(
        [lo[i] for i in range(8)] + [hi[i] for i in range(8)],
        fx.BFloat16,
    )


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


def _wave_value(red, wave, row, token):
    lane = ((row % ROWS) // 8) * 16 + token
    elem = row % 8
    return fx.ptr_load(red + (wave * 32 + lane) * WMMA_ACC + elem)


def _tile_value(red, row, token):
    value = fx.Float32(0.0)
    for wave in range_constexpr(WAVES):
        value = value + _wave_value(red, wave, row, token)
    return value


@flyc.jit
def _block_sum(value, lane, wave, red):
    value = butterfly(value, (16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(value, red + wave)
    gpu.barrier()
    total = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        total = total + fx.ptr_load(red + w)
    return total


@flyc.jit
def stage_norm_gate(c):
    """Per-head RMS norm and sigmoid gate, rounded once to bf16."""
    lane = c["lane"]
    head_task = c["bid"]
    token, head = head_task // (Q_K // HEAD_DIM), head_task % (Q_K // HEAD_DIM)
    base = token * Q_K + head * HEAD_DIM + 2 * lane
    o0, o1 = _bf(c["o"], base), _bf(c["o"], base + 1)
    g0, g1 = _bf(c["g"], base), _bf(c["g"], base + 1)
    ss = o0 * o0 + o1 * o1
    ss = butterfly(ss, (16, 8, 4, 2, 1))
    rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
    _st_bf2(
        c["og"],
        base // 2,
        o0 * rstd * _sigmoid(g0),
        o1 * rstd * _sigmoid(g1),
    )


@flyc.jit
def stage_norm_gate_lds(c):
    """Recompute all norm-gate rows into this CTA's LDS for O projection."""
    lane, wave = c["lane"], c["wave"]
    heads = Q_K // HEAD_DIM
    task = wave
    while task < c["S"] * heads:
        token, head = task // heads, task % heads
        base = token * Q_K + head * HEAD_DIM + 2 * lane
        o0, o1 = _bf(c["o"], base), _bf(c["o"], base + 1)
        g0, g1 = _bf(c["g"], base), _bf(c["g"], base + 1)
        ss = butterfly(o0 * o0 + o1 * o1, (16, 8, 4, 2, 1))
        rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
        fx.ptr_store(
            bf16_pair(
                o0 * rstd * _sigmoid(g0),
                o1 * rstd * _sigmoid(g1),
            ),
            c["og"] + base // 2,
        )
        task = task + WAVES
    gpu.barrier()


@flyc.jit
def stage_o_projection(c):
    """Eight full-K O tiles per CTA: one independent tile per wave."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["w_o"])
    tile = c["bid"] * WAVES + wave
    row0 = tile * ROWS
    row = row0 + lane % ROWS
    for token0 in range_constexpr((c["S"] + ROWS - 1) // ROWS):
        token = fx.min(token0 * ROWS + lane % ROWS, c["S"] - 1)
        acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        for step in range_constexpr(Q_K // WMMA_K):
            k0 = step * WMMA_K
            x = (
                _lds_fragment(c["og"], token * Q_K, k0, lane)
                if c["local_og"]
                else _fragment(rsrc(c["og"]), token * Q_K, k0, lane)
            )
            acc = _wmma(
                _fragment(r_w, row * Q_K, k0, lane, cache_modifier=2),
                x,
                acc,
            )
        fx.ptr_store(acc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
        gpu.barrier()
        batch = min(ROWS, c["S"] - token0 * ROWS)
        for p in range_constexpr(4):
            out = c["tid"] + p * THREADS
            if out < batch * WAVES * (ROWS // 2):
                tile_wave = out // (batch * (ROWS // 2))
                within = out % (batch * (ROWS // 2))
                out_token = token0 * ROWS + within // (ROWS // 2)
                dim = 2 * (within % (ROWS // 2))
                element = (c["bid"] * WAVES + tile_wave) * ROWS + dim
                a0 = bf16_round(_wave_value(c["red"], tile_wave, dim, within // (ROWS // 2)))
                a1 = bf16_round(
                    _wave_value(c["red"], tile_wave, dim + 1, within // (ROWS // 2))
                )
                peer_element = (
                    c["rank"] * MAX_TOKENS + out_token
                ) * HIDDEN + element
                for peer in range_constexpr(c["W"]):
                    c["put_bf"](
                        c["peer_attn"][peer], peer_element, [a0, a1]
                    )
        gpu.barrier()


@flyc.jit
def stage_boundary(c):
    """Rank-ordered partial sum, post norm/residual update, and pre-FFN norm."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    token = c["bid"]
    vals = []
    ss = fx.Float32(0.0)
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        pair = token * (HIDDEN // 2) + tid + j * THREADS
        got = c["poll"](
            [
                (
                    c["own_attn"],
                    src * (MAX_TOKENS * HIDDEN // 2) + pair,
                    1,
                )
                for src in range(c["W"])
            ]
        )
        v0, v1 = bf2_f32(got[0][0])
        for src in range_constexpr(1, c["W"]):
            p0, p1 = bf2_f32(got[src][0])
            v0, v1 = v0 + p0, v1 + p1
        v0, v1 = bf16_round(v0), bf16_round(v1)
        vals += [v0, v1]
        ss = ss + v0 * v0 + v1 * v1
    rstd = hw_rsq(_block_sum(ss, lane, wave, c["red"]) / float(HIDDEN) + POST_EPS)
    new = []
    ss2 = fx.Float32(0.0)
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        for q in range_constexpr(2):
            old = _f32(c["res"], token * HIDDEN + e + q)
            value = (
                _f32(c["alpha"], e + q) * old
                + _f32(c["beta"], e + q) * bf16_round(vals[2 * j + q] * rstd)
            )
            bo.buffer_store(value, rsrc(c["res"]), token * HIDDEN + e + q)
            new.append(value)
            ss2 = ss2 + value * value
    gpu.barrier()
    rstd2 = hw_rsq(
        _block_sum(ss2, lane, wave, c["red"]) / float(HIDDEN) + RMS_EPS
    )
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        _st_bf2(
            c["x"],
            (token * HIDDEN + e) // 2,
            new[2 * j] * rstd2 * _bf(c["pre_w"], e),
            new[2 * j + 1] * rstd2 * _bf(c["pre_w"], e + 1),
        )


@flyc.jit
def stage_router_pool(c):
    """Eight full-K router/pooler tiles per CTA, one per wave."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    r_x, r_w0 = rsrc(c["x"]), rsrc(c["w0"])
    r_w1 = rsrc(c["w1"]) if c["router"] else r_w0
    lds_x = c.get("lds_x", False)
    tile = c["bid"] * WAVES + wave
    row0 = tile * ROWS
    row = row0 + lane % ROWS
    for token0 in range_constexpr((c["S"] + ROWS - 1) // ROWS):
        vacc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        if lds_x:
            # All eight waves consume the same activation tile. Stage it once
            # per CTA and reuse it across the eight independent output tiles.
            for chunk in range_constexpr(HIDDEN // RP_X_TILE_K):
                for p in range_constexpr(RP_X_PAIRS // THREADS):
                    pair = tid + p * THREADS
                    local_token = pair // (RP_X_TILE_K // 2)
                    k_pair = pair % (RP_X_TILE_K // 2)
                    token = fx.min(
                        token0 * ROWS + local_token, c["S"] - 1
                    )
                    word = fx.Int32(
                        bo.buffer_load(
                            r_x,
                            (
                                token * HIDDEN
                                + chunk * RP_X_TILE_K
                            )
                            // 2
                            + k_pair,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    fx.ptr_store(
                        word.bitcast(fx.Float32), c["x_lds"] + pair
                    )
                gpu.barrier()
                for inner in range_constexpr(RP_X_TILE_K // WMMA_K):
                    k0 = chunk * RP_X_TILE_K + inner * WMMA_K
                    x = _lds_fragment(
                        c["x_lds"],
                        fx.min(lane % ROWS, c["S"] - 1) * RP_X_TILE_K,
                        inner * WMMA_K,
                        lane,
                    )
                    vacc = _wmma(
                        _fragment(
                            r_w0,
                            row * HIDDEN,
                            k0,
                            lane,
                            cache_modifier=2,
                        ),
                        x,
                        vacc,
                    )
                    if c["router"]:
                        vacc = _wmma(
                            _fragment(
                                r_w1,
                                row * HIDDEN,
                                k0,
                                lane,
                                cache_modifier=2,
                            ),
                            x,
                            vacc,
                        )
                gpu.barrier()
        else:
            token = fx.min(token0 * ROWS + lane % ROWS, c["S"] - 1)
            for step in range_constexpr(HIDDEN // WMMA_K):
                k0 = step * WMMA_K
                x = _fragment(r_x, token * HIDDEN, k0, lane)
                vacc = _wmma(
                    _fragment(
                        r_w0, row * HIDDEN, k0, lane, cache_modifier=2
                    ),
                    x,
                    vacc,
                )
                if c["router"]:
                    vacc = _wmma(
                        _fragment(
                            r_w1, row * HIDDEN, k0, lane, cache_modifier=2
                        ),
                        x,
                        vacc,
                    )
        fx.ptr_store(vacc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
        gpu.barrier()
        batch = min(ROWS, c["S"] - token0 * ROWS)
        if c["router"]:
            for p in range_constexpr(8):
                out = c["tid"] + p * THREADS
                if out < batch * WAVES * ROWS:
                    tile_wave = out // (batch * ROWS)
                    within = out % (batch * ROWS)
                    out_token, dim = token0 * ROWS + within // ROWS, within % ROWS
                    value = _wave_value(c["red"], tile_wave, dim, within // ROWS)
                    bo.buffer_store(
                        value,
                        rsrc(c["out"]),
                        out_token * EXPERTS
                        + (c["bid"] * WAVES + tile_wave) * ROWS
                        + dim,
                    )
        else:
            for p in range_constexpr(4):
                out = c["tid"] + p * THREADS
                if out < batch * WAVES * (ROWS // 2):
                    tile_wave = out // (batch * (ROWS // 2))
                    within = out % (batch * (ROWS // 2))
                    local_token = within // (ROWS // 2)
                    out_token = token0 * ROWS + local_token
                    dim = 2 * (within % (ROWS // 2))
                    output_row = (c["bid"] * WAVES + tile_wave) * ROWS + dim
                    _st_bf2(
                        c["out"],
                        (out_token * MOE_HIDDEN + output_row) // 2,
                        _wave_value(c["red"], tile_wave, dim, local_token),
                        _wave_value(c["red"], tile_wave, dim + 1, local_token),
                    )
        gpu.barrier()


@flyc.jit
def stage_router_pool_cooperative(c):
    """One output-row tile per CTA, with K split over its eight waves."""
    lane, wave = c["lane"], c["wave"]
    r_x, r_w0 = rsrc(c["x"]), rsrc(c["w0"])
    r_w1 = rsrc(c["w1"]) if c["router"] else r_w0
    row0 = c["task"] * ROWS
    row = row0 + lane % ROWS
    k_per_wave = HIDDEN // WAVES
    for token0 in range_constexpr((c["S"] + ROWS - 1) // ROWS):
        local_token = lane % ROWS
        token = fx.min(token0 * ROWS + local_token, c["S"] - 1)
        vacc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        for step in range_constexpr(k_per_wave // WMMA_K):
            k0 = wave * k_per_wave + step * WMMA_K
            x = _fragment(r_x, token * HIDDEN, k0, lane)
            vacc = _wmma(
                _fragment(r_w0, row * HIDDEN, k0, lane, cache_modifier=2),
                x,
                vacc,
            )
            if c["router"]:
                vacc = _wmma(
                    _fragment(r_w1, row * HIDDEN, k0, lane, cache_modifier=2),
                    x,
                    vacc,
                )
        fx.ptr_store(vacc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
        gpu.barrier()
        batch = min(ROWS, c["S"] - token0 * ROWS)
        if c["router"]:
            for p in range_constexpr(8):
                out = c["tid"] + p * THREADS
                if out < batch * ROWS:
                    out_token = token0 * ROWS + out // ROWS
                    dim = out % ROWS
                    bo.buffer_store(
                        _tile_value(c["red"], dim, out // ROWS),
                        rsrc(c["out"]),
                        out_token * EXPERTS + row0 + dim,
                    )
        else:
            for p in range_constexpr(4):
                out = c["tid"] + p * THREADS
                if out < batch * (ROWS // 2):
                    local_out_token = out // (ROWS // 2)
                    out_token = token0 * ROWS + local_out_token
                    dim = 2 * (out % (ROWS // 2))
                    _st_bf2(
                        c["out"],
                        (out_token * MOE_HIDDEN + row0 + dim) // 2,
                        _tile_value(c["red"], dim, local_out_token),
                        _tile_value(c["red"], dim + 1, local_out_token),
                    )
        gpu.barrier()


def _global_byte_ptr(address):
    return llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<1>"), fx.Int64(address).ir_value()
    )


def _joint_tdm_async_copy(global_address, lds_bf16):
    """Issue one native gfx1250 16-byte global-to-LDS copy."""
    global_ptr = _global_byte_ptr(global_address)
    lds_ptr = fx.recast_iter(fx.Int8, lds_bf16).llvm_ptr
    llvm.inline_asm(
        None,
        [],
        "s_wait_alu depctr_va_vdst(0)",
        "",
        has_side_effects=True,
    )
    fx.rocdl.global_load_async_to_lds_b128(global_ptr, lds_ptr, 0)


@flyc.jit
def _issue_joint_tdm_tile(c, k_tile, slot):
    """Fill one padded M16xK512 / N16xK512 ring slot cooperatively."""
    tid = c["tid"]
    slot_base = slot * JOINT_TDM_SLOT_ELEMS
    for issue in range_constexpr(JOINT_TDM_COPIES_PER_THREAD):
        chunk = tid + issue * THREADS
        if chunk < JOINT_TDM_A_CHUNKS:
            row = chunk // (JOINT_TDM_K // 8)
            col = (chunk % (JOINT_TDM_K // 8)) * 8
            source_row = fx.min(row, c["S"] - 1)
            source_elem = (
                source_row * HIDDEN + k_tile * JOINT_TDM_K + col
            )
            destination = (
                slot_base + row * JOINT_TDM_STRIDE + col
            )
            _joint_tdm_async_copy(
                fx.Int64(c["x"]) + fx.Int64(source_elem * 2),
                c["tdm"] + destination,
            )
        else:
            bchunk = chunk - JOINT_TDM_A_CHUNKS
            row = bchunk // (JOINT_TDM_K // 8)
            col = (bchunk % (JOINT_TDM_K // 8)) * 8
            source_row = c["task"] * JOINT_TDM_B_ROWS + row
            source_elem = (
                source_row * HIDDEN + k_tile * JOINT_TDM_K + col
            )
            destination = (
                slot_base
                + JOINT_TDM_A_ELEMS
                + row * JOINT_TDM_STRIDE
                + col
            )
            _joint_tdm_async_copy(
                fx.Int64(c["w_joint"]) + fx.Int64(source_elem * 2),
                c["tdm"] + destination,
            )
    rocdl.sched_barrier(0)


@flyc.jit
def stage_joint_router_pool(c):
    """Packed Pooler/Router GEMM; four N16 tiles per persistent CTA.

    This first joint-path implementation establishes the production packed
    weight/output ABI and spreads 72 tasks across the 96-CU persistent grid.
    Its global-load mainloop is intentionally isolated behind an opt-in key so
    it can be replaced by the six-buffer TDM mainloop without changing ABI.
    """
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    if wave < JOINT_WAVES:
        tile = c["task"] * JOINT_WAVES + wave
        row0 = tile * ROWS
        row = row0 + lane % ROWS
        token = fx.min(lane % ROWS, c["S"] - 1)
        acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
        r_x, r_w = rsrc(c["x"]), rsrc(c["w_joint"])
        for step in range_constexpr(HIDDEN // WMMA_K):
            k0 = step * WMMA_K
            acc = _wmma(
                _fragment(r_w, row * HIDDEN, k0, lane, cache_modifier=2),
                _fragment(r_x, token * HIDDEN, k0, lane),
                acc,
            )
        fx.ptr_store(acc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
    gpu.barrier()
    count = c["S"] * JOINT_WAVES * ROWS
    # Joint mode is currently build-time restricted to width 8: 512 values.
    for p in range_constexpr(2):
        out = tid + p * THREADS
        if out < count:
            tile_wave = out // (c["S"] * ROWS)
            within = out % (c["S"] * ROWS)
            token, dim = within // ROWS, within % ROWS
            row = (c["task"] * JOINT_WAVES + tile_wave) * ROWS + dim
            bo.buffer_store(
                _wave_value(c["red"], tile_wave, dim, token),
                rsrc(c["out"]),
                token * JOINT_ROWS + row,
            )
    gpu.barrier()


@flyc.jit
def stage_joint_router_pool_tdm(c):
    """Packed M16xN16xK512 GEMM with a four-slot async LDS ring.

    This is the rows=8 Baseline geometry. All resident waves prefetch while
    wave 0 owns the valid M8xN16 output; each persistent CTA executes three
    N16 tasks to cover all 4608 output rows over the 96-CTA grid.
    """
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)

    for tile in range_constexpr(JOINT_TDM_BUFFERS):
        _issue_joint_tdm_tile(c, tile, tile)
    rocdl.sched_barrier(0)

    for k_tile in range_constexpr(JOINT_TDM_K_TILES):
        remaining = (
            min(JOINT_TDM_BUFFERS - 1, JOINT_TDM_K_TILES - 1 - k_tile)
            * JOINT_TDM_COPIES_PER_THREAD
        )
        rocdl.s_wait_asynccnt(remaining)
        gpu.barrier()

        slot = k_tile % JOINT_TDM_BUFFERS
        slot_base = slot * JOINT_TDM_SLOT_ELEMS
        # Keep fragment construction outside a dynamic branch: loader-only
        # waves read valid clamped rows, while wave 0 owns the useful M8 tile.
        token = lane % ROWS
        row = lane % ROWS
        a_fragments = []
        b_fragments = []
        for inner in range_constexpr(JOINT_TDM_K // WMMA_K):
            a_fragments.append(
                _lds_fragment(
                    c["tdm_pairs"],
                    slot_base + token * JOINT_TDM_STRIDE,
                    inner * WMMA_K,
                    lane,
                )
            )
            b_fragments.append(
                _lds_fragment(
                    c["tdm_pairs"],
                    slot_base
                    + JOINT_TDM_A_ELEMS
                    + row * JOINT_TDM_STRIDE,
                    inner * WMMA_K,
                    lane,
                )
            )
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        next_tile = k_tile + JOINT_TDM_BUFFERS
        if next_tile < JOINT_TDM_K_TILES:
            _issue_joint_tdm_tile(c, next_tile, slot)
        rocdl.sched_barrier(0)

        if wave == 0:
            for inner in range_constexpr(JOINT_TDM_K // WMMA_K):
                acc = _wmma(b_fragments[inner], a_fragments[inner], acc)
        rocdl.sched_barrier(0)

    if wave == 0:
        fx.ptr_store(acc, c["red"] + lane * WMMA_ACC)
    gpu.barrier()
    out = tid
    if out < c["S"] * ROWS:
        token, dim = out // ROWS, out % ROWS
        row = c["task"] * ROWS + dim
        bo.buffer_store(
            _wave_value(c["red"], 0, dim, token),
            rsrc(c["out"]),
            token * JOINT_ROWS + row,
        )
    gpu.barrier()


@flyc.jit
def stage_joint_top8(c):
    """Barrier-free stable Top-8: one wave owns one token."""
    lane, wave, token = c["lane"], c["wave"], c["bid"]
    if wave == 0:
        joint_base = token * JOINT_ROWS + MOE_HIDDEN
        raw, biased = [], []
        for q in range_constexpr(EXPERTS // 32):
            expert = lane + q * 32
            logit = _f32(c["joint"], joint_base + expert) + _f32(
                c["joint"], joint_base + EXPERTS + expert
            )
            score = _sigmoid(logit * fx.Float32(ROUTER_MULT))
            raw.append(score)
            biased.append(score + _f32(c["bias"], expert))

        selected_ids, selected_scores = [], []
        for pick in range_constexpr(TOP_K):
            best = fx.Float32(-1.0e30)
            best_score = fx.Float32(0.0)
            best_id = fx.Int32(EXPERTS)
            for q in range_constexpr(EXPERTS // 32):
                expert = fx.Int32(lane + q * 32)
                candidate = biased[q]
                take = (candidate > best) | (
                    (candidate == best) & (expert < best_id)
                )
                best = take.select(candidate, best)
                best_score = take.select(raw[q], best_score)
                best_id = take.select(expert, best_id)
            for offset in (16, 8, 4, 2, 1):
                other = xshfl(best, offset)
                other_score = xshfl(best_score, offset)
                other_id = xshfl(best_id, offset)
                take = (other > best) | (
                    (other == best) & (other_id < best_id)
                )
                best = take.select(other, best)
                best_score = take.select(other_score, best_score)
                best_id = take.select(other_id, best_id)
            selected_ids.append(best_id)
            selected_scores.append(best_score)
            for q in range_constexpr(EXPERTS // 32):
                biased[q] = (fx.Int32(lane + q * 32) == best_id).select(
                    fx.Float32(-1.0e30), biased[q]
                )

        if lane == 0:
            denom = fx.Float32(0.0)
            for pick in range_constexpr(TOP_K):
                denom = denom + selected_scores[pick]
            for pick in range_constexpr(TOP_K):
                bo.buffer_store(
                    selected_ids[pick],
                    rsrc(c["ids"]),
                    token * TOP_K + pick,
                )
                bo.buffer_store(
                    selected_scores[pick]
                    / (denom + fx.Float32(ROUTE_EPS))
                    * fx.Float32(ROUTE_SCALE),
                    rsrc(c["scores"]),
                    token * TOP_K + pick,
                )


@flyc.jit
def stage_top8(c):
    """Stable parallel repeated argmax, preserving the lower-id tie rule."""
    tid, token = c["tid"], c["bid"]
    score = _sigmoid(
        _f32(c["logits"], token * EXPERTS + tid)
        * fx.Float32(ROUTER_MULT)
    )
    biased = score + _f32(c["bias"], tid)
    selected_base = 2 * EXPERTS
    for pick in range_constexpr(TOP_K):
        used = fx.Int32(0)
        for q in range_constexpr(pick):
            used = used | fx.Int32(
                tid == fx.Int32(fx.ptr_load(c["red"] + selected_base + q))
            )
        fx.ptr_store(
            (used == 0).select(biased, fx.Float32(-1.0e30)),
            c["red"] + tid,
        )
        fx.ptr_store(fx.Float32(tid), c["red"] + EXPERTS + tid)
        gpu.barrier()
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            if tid < stride:
                left = fx.ptr_load(c["red"] + tid)
                right = fx.ptr_load(c["red"] + tid + stride)
                left_id = fx.ptr_load(c["red"] + EXPERTS + tid)
                right_id = fx.ptr_load(c["red"] + EXPERTS + tid + stride)
                take_right = (right > left) | (
                    (right == left) & (right_id < left_id)
                )
                fx.ptr_store(
                    take_right.select(right, left), c["red"] + tid
                )
                fx.ptr_store(
                    take_right.select(right_id, left_id),
                    c["red"] + EXPERTS + tid,
                )
            gpu.barrier()
        if tid == 0:
            best_id = fx.Int32(fx.ptr_load(c["red"] + EXPERTS))
            fx.ptr_store(
                fx.Float32(best_id), c["red"] + selected_base + pick
            )
            fx.ptr_store(
                _sigmoid(
                    _f32(c["logits"], token * EXPERTS + best_id)
                    * fx.Float32(ROUTER_MULT)
                ),
                c["picked"] + pick,
            )
        gpu.barrier()

    if tid == 0:
        denom = fx.Float32(0.0)
        for pick in range_constexpr(TOP_K):
            denom = denom + fx.ptr_load(c["picked"] + pick)
        for pick in range_constexpr(TOP_K):
            best_id = fx.Int32(
                fx.ptr_load(c["red"] + selected_base + pick)
            )
            bo.buffer_store(
                best_id, rsrc(c["ids"]), token * TOP_K + pick
            )
            bo.buffer_store(
                fx.ptr_load(c["picked"] + pick)
                / (denom + fx.Float32(ROUTE_EPS))
                * fx.Float32(ROUTE_SCALE),
                rsrc(c["scores"]),
                token * TOP_K + pick,
            )


@flyc.jit
def stage_pooled_norm(c):
    """RMS norm the rounded pooler row and apply its bf16 weight."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    token = c["bid"]
    vals = []
    ss = fx.Float32(0.0)
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        v0 = _bf(c["pool"], token * MOE_HIDDEN + e)
        v1 = _bf(c["pool"], token * MOE_HIDDEN + e + 1)
        vals += [v0, v1]
        ss = ss + v0 * v0 + v1 * v1
    rstd = hw_rsq(
        _block_sum(ss, lane, wave, c["red"]) / float(MOE_HIDDEN) + RMS_EPS
    )
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        _st_bf2(
            c["xin"],
            (token * MOE_HIDDEN + e) // 2,
            vals[2 * j] * rstd * _bf(c["xn_w"], e),
            vals[2 * j + 1] * rstd * _bf(c["xn_w"], e + 1),
        )


@flyc.jit
def stage_joint_pooled_norm(c):
    """Round packed FP32 Pooler output once, then RMSNorm as Baseline does."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    token = c["bid"]
    vals = []
    ss = fx.Float32(0.0)
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        v0 = bf16_round(_f32(c["joint"], token * JOINT_ROWS + e))
        v1 = bf16_round(_f32(c["joint"], token * JOINT_ROWS + e + 1))
        vals += [v0, v1]
        ss = ss + v0 * v0 + v1 * v1
    rstd = hw_rsq(
        _block_sum(ss, lane, wave, c["red"]) / float(MOE_HIDDEN) + RMS_EPS
    )
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        _st_bf2(
            c["xin"],
            (token * MOE_HIDDEN + e) // 2,
            vals[2 * j] * rstd * _bf(c["xn_w"], e),
            vals[2 * j + 1] * rstd * _bf(c["xn_w"], e + 1),
        )
