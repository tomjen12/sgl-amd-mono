# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 wave32 Target attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_round,
    butterfly,
    hw_exp2,
    rsrc,
)
from sgl_amd_mono.muse_spark.gfx1250.device import clock, wait_stores
from sgl_amd_mono.muse_spark.gfx1250.config import (
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    LOG2E,
    Q_K,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    FUSED_BLOCK_STRIDE,
    FUSED_DIM_TILES,
    FUSED_GLOBAL_FLOATS,
    FUSED_KEY_BLOCK,
    FUSED_KEY_TILE,
    FUSED_PART_FLOATS,
    FUSED_ROWS,
    FUSED_SPLITS,
    FUSED_STAT_FLOATS,
    FUSED_TASKS,
    FUSED_TILES,
    FUSED_WMMA_ACC,
    FUSED_WMMA_K,
    GQA,
    KATTN_OUT_FLOATS,
    KATTN_WAVE_STATS,
    WAVES,
)

NEG = -1.0e30
FUSED_PHASES = 6
FUSED_TRACE_WORDS = 8


def _load_bf16x8(resource, element):
    words = fx.Vector(
        bo.buffer_load(resource, element // 2, vec_width=4, dtype=T.i32)
    )
    return words.bitcast(fx.BFloat16).to(fx.Float32)


def _load_fp8x8(resource, byte):
    words = fx.Vector(
        bo.buffer_load(resource, byte // 4, vec_width=2, dtype=T.i32)
    )
    values = []
    for i in range(8):
        raw = fx.Int32((words[i // 4] >> (8 * (i % 4))) & 0xFF)
        sign, exponent, mantissa = raw >> 7, (raw >> 3) & 0xF, raw & 0x7
        normal_bits = ((exponent + 120) << 23) | (mantissa << 20)
        normal = fx.Int32(normal_bits).bitcast(fx.Float32)
        subnormal = fx.Float32(mantissa) * (1.0 / 512.0)
        value = (exponent == 0).select(subnormal, normal)
        values.append((sign != 0).select(-value, value))
    return fx.Vector.from_elements(values, fx.Float32)


def _score(args, qv, request, key, kv_head, lane8, scale):
    page = fx.Int32(
        bo.buffer_load(
            rsrc(args["table"]),
            request * args["table_stride"] + key // 16,
            vec_width=1,
            dtype=T.i32,
        )
    )
    slot = page * 16 + key % 16
    byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane8 * 8
    kv = _load_fp8x8(rsrc(args["k_cache"]), byte)
    dot = fx.Float32(0.0)
    for i in range_constexpr(8):
        dot = dot + qv[i] * kv[i]
    return butterfly(dot, (4, 2, 1)) * scale, slot


@flyc.jit
def run_kattn(c):
    """One CTA computes one (verify token, local query head) attention row."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    args, stats, partial = c["args"], c["stats"], c["partial"]
    token = c["bid"] // LOCAL_Q_HEADS
    query_head = c["bid"] % LOCAL_Q_HEADS
    kv_head = query_head // GQA
    lane8 = lane % 8
    request = token // c["QL"]
    query_token = token % c["QL"]

    prefix = fx.Int32(
        bo.buffer_load(
            rsrc(args["prefix_lens"]), request, vec_width=1, dtype=T.i32
        )
    )
    end = prefix + query_token + 1
    begin = fx.Int32(0)
    if args["window"] > 0:
        lower = end - args["window"]
        begin = (lower > 0).select(lower, begin)
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    qv = _load_bf16x8(
        rsrc(args["q"]),
        token * Q_K + query_head * HEAD_DIM + lane8 * 8,
    )

    # Pass 1: a maximum per wave, then one block maximum.
    local_max = fx.Float32(NEG)
    key = begin + wave
    while key < end:
        score, _ = _score(args, qv, request, key, kv_head, lane8, scale)
        if lane == 0:
            local_max = (score > local_max).select(score, local_max)
        key = key + WAVES
    if lane == 0:
        fx.ptr_store(local_max, stats + wave)
    gpu.barrier()
    if tid == 0:
        block_max = fx.ptr_load(stats)
        for w in range_constexpr(1, WAVES):
            value = fx.ptr_load(stats + w)
            block_max = (value > block_max).select(value, block_max)
        fx.ptr_store(block_max, stats + 2 * WAVES)
    gpu.barrier()
    block_max = fx.ptr_load(stats + 2 * WAVES)

    # Pass 2: denominator per wave, then the block denominator.
    local_sum = fx.Float32(0.0)
    key = begin + wave
    while key < end:
        score, _ = _score(args, qv, request, key, kv_head, lane8, scale)
        if lane == 0:
            local_sum = local_sum + hw_exp2((score - block_max) * LOG2E)
        key = key + WAVES
    if lane == 0:
        fx.ptr_store(local_sum, stats + WAVES + wave)
    gpu.barrier()
    if tid == 0:
        block_sum = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            block_sum = block_sum + fx.ptr_load(stats + WAVES + w)
        fx.ptr_store(block_sum, stats + 2 * WAVES + 1)
    gpu.barrier()
    block_sum = fx.ptr_load(stats + 2 * WAVES + 1)

    # Pass 3: BF16 probabilities times V, one partial output per wave.
    output = fx.Vector.filled(8, 0.0, fx.Float32)
    key = begin + wave
    while key < end:
        score, slot = _score(args, qv, request, key, kv_head, lane8, scale)
        probability = bf16_round(hw_exp2((score - block_max) * LOG2E) / block_sum)
        byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane8 * 8
        value = _load_fp8x8(rsrc(args["v_cache"]), byte)
        output = output + value * probability
        key = key + WAVES
    if lane < 8:
        fx.ptr_store(output, partial + wave * HEAD_DIM + lane * 8)
    gpu.barrier()

    if tid < 8:
        total = fx.Vector.filled(8, 0.0, fx.Float32)
        for w in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                partial + w * HEAD_DIM + tid * 8,
                result_type=fx.Vector.make_type(8, fx.Float32),
            )
        bo.buffer_store(
            total.to(fx.BFloat16),
            rsrc(args["out"]),
            token * Q_K + query_head * HEAD_DIM + tid * 8,
        )


def kattn_lds():
    @fx.struct
    class Lds:
        stats: fx.Array[fx.Float32, KATTN_WAVE_STATS, 16]
        partial: fx.Array[fx.Float32, KATTN_OUT_FLOATS, 16]

    return Lds


# ------------------------------------------------------- fused cu96_tp4 Target
# The standalone scalar kernel above remains the correctness reference.  The
# fused layer uses all 96 resident CTAs as 16 row tiles x 6 key splits.


def _fused_fp8(address, byte):
    word = fx.Int32(
        bo.buffer_load(rsrc(address), byte // 4, vec_width=1, dtype=T.i32)
    )
    raw = (word >> (8 * (byte % 4))) & 0xFF
    sign, exponent, mantissa = raw >> 7, (raw >> 3) & 0xF, raw & 0x7
    normal = fx.Int32(
        ((exponent + 120) << 23) | (mantissa << 20)
    ).bitcast(fx.Float32)
    subnormal = fx.Float32(mantissa) * (1.0 / 512.0)
    value = (exponent == 0).select(subnormal, normal)
    return (sign != 0).select(-value, value)


def _fused_bf16(address, element):
    word = fx.Int32(
        bo.buffer_load(rsrc(address), element // 2, vec_width=1, dtype=T.i32)
    )
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _fused_page(args, request, key):
    return fx.Int32(
        bo.buffer_load(
            rsrc(args["table"]),
            request * args["table_stride"] + key // 16,
            vec_width=1,
            dtype=T.i32,
        )
    )


def _fused_slot(args, request, key):
    page = _fused_page(args, request, key)
    return page * 16 + key % 16


def fused_row_coords(tile, row, tokens):
    """Map one 16-row tile to its two tokens and eight GQA heads each."""
    kv_head = tile // (tokens // 2)
    token_pair = tile % (tokens // 2)
    logical = token_pair * FUSED_ROWS + row
    token = logical // GQA
    query_head = kv_head * GQA + logical % GQA
    return token, query_head, kv_head


def _fused_q_fragment(args, tile, tokens, k0, lane):
    token, query_head, _ = fused_row_coords(tile, lane % FUSED_ROWS, tokens)
    kh = (lane // FUSED_ROWS) * 8
    values = []
    for half in range_constexpr(2):
        for element in range_constexpr(8):
            dim = k0 + half * 16 + kh + element
            values.append(
                _fused_bf16(
                    args["q"],
                    token * Q_K + query_head * HEAD_DIM + dim,
                )
            )
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)


def _fused_k_fragment(args, kv_head, slot, k0, lane):
    kh = (lane // FUSED_ROWS) * 8
    values = []
    resource = rsrc(args["k_cache"])
    for half in range_constexpr(2):
        dim = k0 + half * 16 + kh
        byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + dim
        packed = _load_fp8x8(resource, byte)
        values += [packed[element] for element in range_constexpr(8)]
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)


def _fused_wmma(a, b, acc):
    result = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(FUSED_WMMA_ACC, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        reuseA=False,
        reuseB=False,
    ).result
    return fx.Vector(result)


def _fused_qk(args, tile, tokens, request, key0, keys, lane, scale):
    _, _, kv_head = fused_row_coords(tile, fx.Int32(0), tokens)
    key = fx.min(key0 + lane % FUSED_ROWS, keys - 1)
    slot = _fused_slot(args, request, key)
    acc = fx.Vector.filled(FUSED_WMMA_ACC, 0.0, fx.Float32)
    for step in range_constexpr(HEAD_DIM // FUSED_WMMA_K):
        k0 = step * FUSED_WMMA_K
        acc = _fused_wmma(
            _fused_q_fragment(args, tile, tokens, k0, lane),
            _fused_k_fragment(args, kv_head, slot, k0, lane),
            acc,
        )
    return acc * scale


def _fused_max(a, b):
    return fx.max(a, b)


def fused_stat_index(tile, split, row):
    return ((tile * FUSED_SPLITS + split) * FUSED_ROWS + row) * 2


def fused_global_index(tile, row):
    return (tile * FUSED_ROWS + row) * 2


def fused_part_index(tile, split, row, dim):
    return (
        ((tile * FUSED_SPLITS + split) * FUSED_ROWS + row) * HEAD_DIM
        + dim
    )


@flyc.jit
def _fused_publish(put, region, index, owner):
    wait_stores()
    gpu.barrier()
    # Split statistics are separate stores from the ready mailbox word.
    # Release them from every producer thread before the owner publishes the
    # tag, so the merge CTA cannot read stale max/sum/PV data.
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
    if owner:
        put(region, index, fx.Int32(1))


@flyc.jit
def _fused_poll_splits(mb, region, tile):
    """Poll six tagged split flags in batches no wider than four entries."""
    mb.poll(
        [
            (region, tile * FUSED_SPLITS + split, 1)
            for split in range(4)
        ],
        batch=4,
    )
    mb.poll(
        [
            (region, tile * FUSED_SPLITS + split, 1)
            for split in range(4, FUSED_SPLITS)
        ],
        batch=4,
    )
    # Pair with _fused_publish before loading the split statistics.
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)


def fused_attn_lds():
    @fx.struct
    class Lds:
        wave_stats: fx.Array[
            fx.Float32, WAVES * FUSED_ROWS * 2, 16
        ]
        row_stats: fx.Array[fx.Float32, FUSED_ROWS, 16]
        probabilities: fx.Array[
            fx.BFloat16, WAVES * FUSED_ROWS * FUSED_KEY_BLOCK, 16
        ]
        pv: fx.Array[
            fx.Float32, WAVES * 32 * FUSED_WMMA_ACC, 16
        ]

    return Lds


@flyc.jit
def _fused_stamp(c, phase):
    if c.get("phase_trace", False):
        if c["tid"] == 0:
            task = c["tile"] * FUSED_SPLITS + c["split"]
            bo.buffer_store(
                clock(),
                rsrc(c["phase_ticks"].value),
                task * FUSED_TRACE_WORDS + phase,
            )


@flyc.jit
def run_fused_attn(c):
    """Run one split-key CTA; split zero performs ordered merges and output."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, args, mb = c["tile"], c["split"], c["args"], c["mb"]
    tokens, q_len = c["S"], c["QL"]
    phase_trace = c.get("phase_trace", False)
    stats, glob, part = c["stats"], c["global"], c["part"]
    wave_stats, row_stats = c["wave_stats"], c["row_stats"]

    token0, _, _ = fused_row_coords(tile, fx.Int32(0), tokens)
    request = token0 // q_len
    prefix = fx.Int32(
        bo.buffer_load(
            rsrc(args["prefix_lens"]), request, vec_width=1, dtype=T.i32
        )
    )
    keys = prefix + q_len
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    _fused_stamp(c, 0)

    # Each wave owns block split*8 + wave + iteration*48.  QK, online softmax
    # and PV are completed in one pass over each 32-key block.  Every wave
    # retains a partial (max, sum, output), which is rescaled when waves and
    # splits merge.
    maxima = [fx.Float32(NEG) for _ in range(FUSED_WMMA_ACC)]
    sums = [fx.Float32(0.0) for _ in range(FUSED_WMMA_ACC)]
    pv_acc = [
        fx.Vector.filled(FUSED_WMMA_ACC, 0.0, fx.Float32)
        for _ in range(FUSED_DIM_TILES)
    ]
    probability_base = (
        c["probabilities"] + wave * FUSED_ROWS * FUSED_KEY_BLOCK
    )
    qk_probability_cycles = fx.Int64(0)
    pv_cycles = fx.Int64(0)
    iteration = fx.Int32(0)
    while iteration * (FUSED_BLOCK_STRIDE * FUSED_KEY_BLOCK) < keys:
        subphase_start = fx.Int64(0)
        subphase_middle = fx.Int64(0)
        if const_expr(phase_trace):
            subphase_start = clock()
        block = split * WAVES + wave + iteration * FUSED_BLOCK_STRIDE
        block0 = block * FUSED_KEY_BLOCK
        block_scores = []
        block_live = []
        for chunk in range_constexpr(2):
            key0 = block0 + chunk * FUSED_KEY_TILE
            scores = _fused_qk(
                args,
                tile,
                tokens,
                request,
                key0,
                keys,
                lane,
                scale,
            )
            lives = []
            key = key0 + lane % FUSED_ROWS
            for element in range_constexpr(FUSED_WMMA_ACC):
                row = (lane // FUSED_ROWS) * 8 + element
                token, _, _ = fused_row_coords(tile, row, tokens)
                end = prefix + token % q_len + 1
                begin = fx.Int32(0)
                if args["window"] > 0:
                    begin = fx.max(end - args["window"], fx.Int32(0))
                lives.append((key >= begin) & (key < end))
            block_scores.append(scores)
            block_live.append(lives)

        alphas = []
        for element in range_constexpr(FUSED_WMMA_ACC):
            local_max = fx.Float32(NEG)
            for chunk in range_constexpr(2):
                local_max = fx.max(
                    local_max,
                    block_live[chunk][element].select(
                        fx.Float32(block_scores[chunk][element]),
                        fx.Float32(NEG),
                    ),
                )
            block_max = butterfly(
                local_max, (8, 4, 2, 1), op=_fused_max
            )
            new_max = fx.max(maxima[element], block_max)
            alpha = hw_exp2((maxima[element] - new_max) * LOG2E)
            lane_sum = fx.Float32(0.0)
            row = (lane // FUSED_ROWS) * 8 + element
            for chunk in range_constexpr(2):
                probability = block_live[chunk][element].select(
                    hw_exp2(
                        (
                            fx.Float32(block_scores[chunk][element])
                            - new_max
                        )
                        * LOG2E
                    ),
                    fx.Float32(0.0),
                )
                lane_sum = lane_sum + probability
                fx.ptr_store(
                    fx.BFloat16(bf16_round(probability)),
                    probability_base
                    + row * FUSED_KEY_BLOCK
                    + chunk * FUSED_KEY_TILE
                    + lane % FUSED_ROWS,
                )
            sums[element] = sums[element] * alpha + butterfly(
                lane_sum, (8, 4, 2, 1)
            )
            alphas.append(alpha)
            maxima[element] = new_max
        gpu.barrier()
        if const_expr(phase_trace):
            subphase_middle = clock()
            qk_probability_cycles = (
                qk_probability_cycles + subphase_middle - subphase_start
            )

        alpha_vector = fx.Vector.from_elements(alphas, fx.Float32)
        kh = (lane // FUSED_ROWS) * 8
        probability_row = lane % FUSED_ROWS
        p_values = []
        for half in range_constexpr(2):
            p_values += [
                fx.ptr_load(
                    probability_base
                    + probability_row * FUSED_KEY_BLOCK
                    + half * 16
                    + kh
                    + element
                )
                for element in range_constexpr(8)
            ]
        p_fragment = fx.Vector.from_elements(p_values, fx.BFloat16)
        _, _, kv_head = fused_row_coords(tile, fx.Int32(0), tokens)
        v_slots = []
        v_live = []
        for half in range_constexpr(2):
            for element in range_constexpr(8):
                key = block0 + half * 16 + kh + element
                safe_key = fx.min(key, keys - 1)
                v_slots.append(_fused_slot(args, request, safe_key))
                v_live.append(key < keys)
        for dim_tile in range_constexpr(FUSED_DIM_TILES):
            v_values = []
            dim = dim_tile * FUSED_ROWS + lane % FUSED_ROWS
            for half in range_constexpr(2):
                for element in range_constexpr(8):
                    index = half * 8 + element
                    byte = (
                        (
                            v_slots[index] * LOCAL_KV_HEADS + kv_head
                        )
                        * HEAD_DIM
                        + dim
                    )
                    v_values.append(
                        v_live[index].select(
                            _fused_fp8(args["v_cache"], byte),
                            fx.Float32(0.0),
                        )
                    )
            pv_acc[dim_tile] = _fused_wmma(
                p_fragment,
                fx.Vector.from_elements(
                    v_values, fx.Float32
                ).to(fx.BFloat16),
                pv_acc[dim_tile] * alpha_vector,
            )
        if const_expr(phase_trace):
            subphase_end = clock()
            pv_cycles = pv_cycles + subphase_end - subphase_middle
        iteration = iteration + 1

    for element in range_constexpr(FUSED_WMMA_ACC):
        if lane % FUSED_ROWS == 0:
            row = (lane // FUSED_ROWS) * 8 + element
            fx.ptr_store(
                maxima[element],
                wave_stats + (wave * FUSED_ROWS + row) * 2,
            )
            fx.ptr_store(
                sums[element],
                wave_stats + (wave * FUSED_ROWS + row) * 2 + 1,
            )
    gpu.barrier()
    if tid < FUSED_ROWS:
        split_max = fx.ptr_load(wave_stats + tid * 2)
        for source_wave in range_constexpr(1, WAVES):
            split_max = fx.max(
                split_max,
                fx.ptr_load(
                    wave_stats
                    + (source_wave * FUSED_ROWS + tid) * 2
                ),
            )
        split_sum = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            wave_max = fx.ptr_load(
                wave_stats + (source_wave * FUSED_ROWS + tid) * 2
            )
            wave_sum = fx.ptr_load(
                wave_stats
                + (source_wave * FUSED_ROWS + tid) * 2
                + 1
            )
            split_sum = split_sum + wave_sum * hw_exp2(
                (wave_max - split_max) * LOG2E
            )
        bo.buffer_store(
            fx.Vector.from_elements(
                [split_max, split_sum], fx.Float32
            ),
            rsrc(stats.value),
            fused_stat_index(tile, split, tid),
            cache_modifier=2,
        )
    _fused_publish(
        mb.put,
        c["sflag"],
        tile * FUSED_SPLITS + split,
        tid == 0,
    )
    _fused_stamp(c, 1)

    if split == 0:
        _fused_poll_splits(mb, c["sflag"], tile)
        if tid < FUSED_ROWS:
            global_max = fx.Float32(NEG)
            for source in range_constexpr(FUSED_SPLITS):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        fused_stat_index(tile, source, tid),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                global_max = fx.max(global_max, fx.Float32(pair[0]))
            global_sum = fx.Float32(0.0)
            for source in range_constexpr(FUSED_SPLITS):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        fused_stat_index(tile, source, tid),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                global_sum = global_sum + fx.Float32(pair[1]) * hw_exp2(
                    (fx.Float32(pair[0]) - global_max) * LOG2E
                )
            bo.buffer_store(
                fx.Vector.from_elements(
                    [global_max, global_sum], fx.Float32
                ),
                rsrc(glob.value),
                fused_global_index(tile, tid),
                cache_modifier=2,
            )
        _fused_publish(mb.put, c["gflag"], tile, tid == 0)
    mb.poll([(c["gflag"], tile, 1)], batch=1)
    # The global denominator is stored separately from the ready flag.
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
    _fused_stamp(c, 2)
    _fused_stamp(c, 3)
    if const_expr(phase_trace):
        if tid == 0:
            task = tile * FUSED_SPLITS + split
            bo.buffer_store(
                qk_probability_cycles,
                rsrc(c["phase_ticks"].value),
                task * FUSED_TRACE_WORDS + 6,
            )
            bo.buffer_store(
                pv_cycles,
                rsrc(c["phase_ticks"].value),
                task * FUSED_TRACE_WORDS + 7,
            )

    # Reduce the eight wave partials into this split's FP32 output.
    for dim_tile in range_constexpr(FUSED_DIM_TILES):
        fx.ptr_store(
            pv_acc[dim_tile],
            c["pv"] + (wave * 32 + lane) * FUSED_WMMA_ACC,
        )
        gpu.barrier()
        row, col = tid // FUSED_ROWS, tid % FUSED_ROWS
        result_lane = (row // 8) * 16 + col
        split_pair = fx.Vector(
            bo.buffer_load(
                rsrc(stats.value),
                fused_stat_index(tile, split, row),
                vec_width=2,
                dtype=T.f32,
                cache_modifier=2,
            )
        )
        total = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            wave_max = fx.ptr_load(
                wave_stats
                + (source_wave * FUSED_ROWS + row) * 2
            )
            total = total + fx.ptr_load(
                c["pv"]
                + (source_wave * 32 + result_lane) * FUSED_WMMA_ACC
                + row % 8
            ) * hw_exp2(
                (wave_max - fx.Float32(split_pair[0])) * LOG2E
            )
        bo.buffer_store(
            total,
            rsrc(part.value),
            fused_part_index(
                tile, split, row, dim_tile * FUSED_ROWS + col
            ),
            cache_modifier=2,
        )
        gpu.barrier()
    _fused_publish(
        mb.put,
        c["pflag"],
        tile * FUSED_SPLITS + split,
        tid == 0,
    )
    _fused_stamp(c, 4)

    if split == 0:
        _fused_poll_splits(mb, c["pflag"], tile)
        for batch in range_constexpr(
            (FUSED_ROWS * HEAD_DIM) // 256
        ):
            item = tid + batch * 256
            row, dim = item // HEAD_DIM, item % HEAD_DIM
            global_pair = fx.Vector(
                bo.buffer_load(
                    rsrc(glob.value),
                    fused_global_index(tile, row),
                    vec_width=2,
                    dtype=T.f32,
                    cache_modifier=2,
                )
            )
            total = fx.Float32(0.0)
            for source in range_constexpr(FUSED_SPLITS):
                split_pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        fused_stat_index(tile, source, row),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                partial = fx.Float32(
                    bo.buffer_load(
                        rsrc(part.value),
                        fused_part_index(
                            tile, source, row, dim
                        ),
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                total = total + partial * hw_exp2(
                    (
                        fx.Float32(split_pair[0])
                        - fx.Float32(global_pair[0])
                    )
                    * LOG2E
                ) / fx.Float32(global_pair[1])
            token, query_head, _ = fused_row_coords(tile, row, tokens)
            bo.buffer_store(
                total.to(fx.BFloat16),
                rsrc(args["out"]),
                token * Q_K + query_head * HEAD_DIM + dim,
            )
    if const_expr(phase_trace):
        gpu.barrier()
    _fused_stamp(c, 5)
