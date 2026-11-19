# SPDX-License-Identifier: Apache-2.0
"""gfx1250 wave32 K1 for the TP4, eight-row Muse Spark V1.

The first correctness-oriented implementation keeps the tagged split-K handoff
used by gfx950, but replaces its wave64 MFMA mapping with a wave32 dot-product
mapping.  The 192 persistent CTAs execute all 512 projection tasks in
grid-stride order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    rsrc,
)
from sgl_amd_mono.muse_spark.gfx1250.device import clock
from sgl_amd_mono.muse_spark.gfx1250.config import (
    G_OFF,
    HEAD_DIM,
    HIDDEN,
    K_OFF,
    KV_K,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    V_OFF,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    GATE_TASK_BASE,
    GATE_TASKS,
    GEMV_ROW_TILES,
    GEMV_ROWS_PER_TASK_GROUP,
    GEMV_TASK_ROUNDS,
    GEMV_TASKS,
    HEAD_TASK_BASE,
    HEAD_TASKS,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEPS,
    RED_FLOATS,
    RESIDENT_BLOCKS,
    ROWS,
    THREADS,
    WAVES,
    WMMA_ACCUMULATORS,
    WMMA_K,
)

ELEMS = 4
TOKEN_LANES = HEAD_DIM // ELEMS


def _fragment(resource, base, kbase, lane, cache_modifier=0):
    """One lane's v16bf16 gfx1250 WMMA fragment, using two legal dwordx4 loads."""
    k_half = (lane // 16) * 8
    lo = fx.Vector(
        bo.buffer_load(
            resource,
            (base + kbase + k_half) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource,
            (base + kbase + 16 + k_half) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    return fx.Vector.from_elements(
        [lo[i] for i in range(4)] + [hi[i] for i in range(4)],
        fx.Int32,
    ).bitcast(fx.BFloat16)


def _wmma(a, b, acc):
    result = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(WMMA_ACCUMULATORS, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        reuseA=False,
        reuseB=False,
    ).result
    return fx.Vector(result)


def _row_sum_w32(red, row, token):
    # gfx1250 WMMA C layout: lane16 is the B/input row (token), lane_kgrp
    # selects an 8-column half of A/weight, and the v8 element is that column.
    lane = ((row % ROWS) // 8) * 16 + token
    elem = row % 8
    total = fx.Float32(0.0)
    for wave in range(WAVES):
        total = total + fx.ptr_load(
            red
            + (row // ROWS) * RED_FLOATS
            + (wave * 32 + lane) * WMMA_ACCUMULATORS
            + elem
        )
    return total


@flyc.jit
def stage_gemv(c, task):
    """Compute one row-group/K-part projection task with native wave32 WMMA."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part, args = c["red"], c["put"], c["part"], c["args"]
    row_group, kpart = task // K_PARTS, task % K_PARTS
    r_w, r_x = rsrc(args["w"], QKVG_ROWS * HIDDEN * 2), rsrc(args["x"])

    for token_tile in range_constexpr((S + ROWS - 1) // ROWS):
        tile_start = token_tile * ROWS
        tile_rows = min(ROWS, S - tile_start)
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        token = tile_start + token_local
        b_fragments = [
            _fragment(
                r_x,
                token * HIDDEN,
                kpart * K_PART + wave * K_PER_WAVE + step * WMMA_K,
                lane,
            )
            for step in range(K_STEPS)
        ]
        for tile in range_constexpr(GEMV_ROW_TILES):
            model_row = (
                row_group * GEMV_ROWS_PER_TASK_GROUP
                + tile * ROWS
                + lane % ROWS
            )
            acc = fx.Vector.filled(WMMA_ACCUMULATORS, 0.0, fx.Float32)
            for step in range_constexpr(K_STEPS):
                a = _fragment(
                    r_w,
                    model_row * HIDDEN,
                    kpart * K_PART + wave * K_PER_WAVE + step * WMMA_K,
                    lane,
                    cache_modifier=2,
                )
                acc = _wmma(a, b_fragments[step], acc)
            fx.ptr_store(
                acc,
                red
                + tile * RED_FLOATS
                + (wave * 32 + lane) * WMMA_ACCUMULATORS,
            )
        gpu.barrier()
        outputs = GEMV_ROWS_PER_TASK_GROUP * tile_rows
        for batch in range_constexpr((outputs + THREADS - 1) // THREADS):
            out = tid + batch * THREADS
            if out < outputs:
                row = out % GEMV_ROWS_PER_TASK_GROUP
                local_token = out // GEMV_ROWS_PER_TASK_GROUP
                value = _row_sum_w32(red, row, local_token)
                put(
                    part,
                    (kpart * MAX_TOKENS + tile_start + local_token)
                    * QKVG_ROWS
                    + row_group * GEMV_ROWS_PER_TASK_GROUP
                    + row,
                    value,
                )
        gpu.barrier()


def _four_partials(poll, part, token, row):
    got = poll(
        [
            (
                part,
                (kpart * MAX_TOKENS + token) * QKVG_ROWS + row + half * 2,
                2,
            )
            for kpart in range(K_PARTS)
            for half in range(2)
        ]
    )
    out = []
    for j in range_constexpr(ELEMS):
        value = got[j // 2][j % 2].bitcast(fx.Float32)
        for kpart in range_constexpr(1, K_PARTS):
            value = value + got[2 * kpart + j // 2][j % 2].bitcast(fx.Float32)
        out.append(bf16_round(value))
    return out


@flyc.jit
def stage_head(c, head, token_offset):
    """Reduce one q/k/v head, normalize q/k, apply RoPE, and write K/V cache."""
    S, tid = c["S"], c["tid"]
    poll, part, args = c["poll"], c["part"], c["args"]
    token = token_offset + tid // TOKEN_LANES
    lane16 = tid % TOKEN_LANES
    if token < S:
        is_q = head < LOCAL_Q_HEADS
        is_k = (head >= LOCAL_Q_HEADS) & (
            head < LOCAL_Q_HEADS + LOCAL_KV_HEADS
        )
        kv_head = is_k.select(
            head - LOCAL_Q_HEADS,
            head - LOCAL_Q_HEADS - LOCAL_KV_HEADS,
        )
        col = is_q.select(
            head * HEAD_DIM,
            is_k.select(K_OFF, V_OFF) + kv_head * HEAD_DIM,
        )
        dim = lane16 * ELEMS
        values = _four_partials(poll, part, token, col + dim)

        normed = is_q | is_k
        ss = fx.Float32(0.0)
        for j in range_constexpr(ELEMS):
            ss = ss + values[j] * values[j]
        ss = butterfly(ss, (8, 4, 2, 1))
        rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
        normalized = [bf16_round(values[j] * rstd) for j in range(ELEMS)]
        normalized = [
            is_q.select(bf16_round(value * QK_SCALE), value)
            for value in normalized
        ]

        pos = fx.Int32(
            bo.buffer_load(
                rsrc(args["positions"]), 2 * token, vec_width=1, dtype=T.i32
            )
        )
        cs = fx.Vector(
            bo.buffer_load(
                rsrc(args["cos_sin"]),
                pos * HEAD_DIM + dim // 2,
                vec_width=2,
                dtype=T.f32,
            )
        )
        sn = fx.Vector(
            bo.buffer_load(
                rsrc(args["cos_sin"]),
                pos * HEAD_DIM + HEAD_DIM // 2 + dim // 2,
                vec_width=2,
                dtype=T.f32,
            )
        )
        c0, c1 = bf16_round(cs[0]), bf16_round(cs[1])
        s0, s1 = bf16_round(sn[0]), bf16_round(sn[1])
        rotated = [
            bf16_round(normalized[0] * c0) - bf16_round(normalized[1] * s0),
            bf16_round(normalized[1] * c0) + bf16_round(normalized[0] * s0),
            bf16_round(normalized[2] * c1) - bf16_round(normalized[3] * s1),
            bf16_round(normalized[3] * c1) + bf16_round(normalized[2] * s1),
        ]
        roped = normed & (args["use_rope"] != 0)
        output = [
            normed.select(roped.select(bf16_round(rotated[j]), normalized[j]), values[j])
            for j in range(ELEMS)
        ]
        vec = fx.Vector.from_elements(output, fx.Float32).to(fx.BFloat16)

        slot = fx.Int32(
            bo.buffer_load(
                rsrc(args["slots"]),
                (args["slot64"] != 0).select(2 * token, token),
                vec_width=1,
                dtype=T.i32,
            )
        )
        if (args["use_swa"] != 0) & (slot >= 0):
            slot = fx.Int32(
                bo.buffer_load(
                    rsrc(args["swa_map"]),
                    (args["swa64"] != 0).select(2 * slot, slot),
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        if is_q:
            bo.buffer_store(
                vec, rsrc(args["q_out"]), token * Q_K + head * HEAD_DIM + dim
            )
        else:
            if is_k:
                bo.buffer_store(
                    vec,
                    rsrc(args["k_out"]),
                    token * KV_K + kv_head * HEAD_DIM + dim,
                )
            else:
                bo.buffer_store(
                    vec,
                    rsrc(args["v_out"]),
                    token * KV_K + kv_head * HEAD_DIM + dim,
                )
            if slot >= 0:
                packed = fp8_pack4(
                    clamp_fp8(output[0]),
                    clamp_fp8(output[1]),
                    clamp_fp8(output[2]),
                    clamp_fp8(output[3]),
                )
                index = (
                    (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + dim // 4
                )
                if is_k:
                    bo.buffer_store(packed, rsrc(args["k_cache"]), index)
                else:
                    bo.buffer_store(packed, rsrc(args["v_cache"]), index)


@flyc.jit
def stage_gate(c, gate, token_offset):
    """Reduce one 64-row output-gate head."""
    S, tid = c["S"], c["tid"]
    token = token_offset + tid // TOKEN_LANES
    lane16 = tid % TOKEN_LANES
    if token < S:
        dim = lane16 * ELEMS
        values = _four_partials(
            c["poll"], c["part"], token, G_OFF + gate * HEAD_DIM + dim
        )
        vec = fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)
        bo.buffer_store(
            vec,
            rsrc(c["args"]["g_out"]),
            token * Q_K + gate * HEAD_DIM + dim,
        )


@flyc.jit
def _stamp_k1(c, phase):
    if c.get("phase_trace", False):
        gpu.barrier()
        if c["tid"] == 0:
            bo.buffer_store(
                clock(),
                rsrc(c["phase_ticks"].value),
                c["bid"] * 3 + phase,
            )


@flyc.jit
def run_k1(c, block):
    """Run this CTA's GEMV tasks, then its optional head/gate reduction."""
    _stamp_k1(c, 0)
    for task_round in range_constexpr(GEMV_TASK_ROUNDS):
        task = block + task_round * RESIDENT_BLOCKS
        if task < GEMV_TASKS:
            stage_gemv(c, task)
    _stamp_k1(c, 1)
    tokens_per_round = THREADS // TOKEN_LANES
    for token_round in range_constexpr(
        (c["S"] + tokens_per_round - 1) // tokens_per_round
    ):
        token_offset = token_round * tokens_per_round
        if block < HEAD_TASKS:
            stage_head(c, block - HEAD_TASK_BASE, token_offset)
        if (block >= GATE_TASK_BASE) & (
            block < GATE_TASK_BASE + GATE_TASKS
        ):
            stage_gate(c, block - GATE_TASK_BASE, token_offset)
    _stamp_k1(c, 2)


def k1_lds():
    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, GEMV_ROW_TILES * RED_FLOATS, 16]

    return Lds


assert THREADS == 256
