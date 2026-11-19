# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 wave32 Target attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_scale_pk8_f32_fp8
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_round,
    butterfly,
    hw_exp2,
    lane_gather,
    rsrc,
    xshfl,
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
FP8_KEY_BLOCK = 64
FP8_LDS_ROW = 80
FP8_PROBABILITY_BIAS = 8.0


def _load_bf16x8(resource, element):
    words = fx.Vector(
        bo.buffer_load(resource, element // 2, vec_width=4, dtype=T.i32)
    )
    return words.bitcast(fx.BFloat16).to(fx.Float32)


def _decode_fp8x8(words):
    return fx.Vector(
        cvt_scale_pk8_f32_fp8(
            T.vec(8, T.f32),
            words.ir_value(),
            fx.Int32(127).ir_value(),
            0,
        )
    )


def _pack_fp8x32(values):
    words = []
    for word in range_constexpr(8):
        lo = _rocdl.cvt_pk_fp8_f32(
            T.i32,
            _to_raw(fx.Float32(values[word * 4])),
            _to_raw(fx.Float32(values[word * 4 + 1])),
            fx.Int32(0).ir_value(),
            False,
        )
        packed = _rocdl.cvt_pk_fp8_f32(
            T.i32,
            _to_raw(fx.Float32(values[word * 4 + 2])),
            _to_raw(fx.Float32(values[word * 4 + 3])),
            lo,
            True,
        )
        words.append(fx.Int32(packed))
    return fx.Vector.from_elements(words, fx.Int32)


def _wmma_fp8(a, b, acc):
    return fx.Vector(
        _rocdl.wmma_f32_16x16x64_fp8_fp8_(
            T.vec(8, T.f32),
            _to_raw(a),
            _to_raw(b),
            _to_raw(acc),
        )
    )


def _load_fp8x8(resource, byte):
    words = fx.Vector(
        bo.buffer_load(resource, byte // 4, vec_width=2, dtype=T.i32)
    )
    return _decode_fp8x8(words)


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


def _fused_lds_fp8(words, byte):
    word = fx.Int32(fx.ptr_load(words + byte // 4, result_type=T.i32))
    raw = (word >> ((byte % 4) * 8)) & 0xFF
    sign, exponent, mantissa = raw >> 7, (raw >> 3) & 0xF, raw & 0x7
    normal_bits = ((exponent + 120) << 23) | (mantissa << 20)
    normal = fx.Int32(normal_bits).bitcast(fx.Float32)
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


def _fused_lds_q_fragment(q_lds, k0, lane):
    row = lane % FUSED_ROWS
    kh = (lane // FUSED_ROWS) * 8
    values = []
    for half in range_constexpr(2):
        dim = k0 + half * 16 + kh
        packed = fx.Vector(
            fx.ptr_load(
                q_lds + (row * HEAD_DIM + dim) // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        values += [packed[element] for element in range_constexpr(8)]
    return fx.Vector.from_elements(values, fx.BFloat16)


def _fused_k_fragment(kv_lds, local_key, k0, lane):
    kh = (lane // FUSED_ROWS) * 8
    values = []
    for half in range_constexpr(2):
        dim = k0 + half * 16 + kh
        byte = local_key * HEAD_DIM + dim
        packed = _decode_fp8x8(
            fx.Vector(
                fx.ptr_load(
                    kv_lds + byte // 4,
                    result_type=T.vec(2, T.i32),
                )
            )
        )
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


def _fused_qk(
    args,
    tile,
    tokens,
    key0,
    keys,
    lane,
    scale,
    kv_lds,
    local_key0,
    q_lds,
    stage_q,
):
    key = fx.min(key0 + lane % FUSED_ROWS, keys - 1)
    local_key = local_key0 + lane % FUSED_ROWS
    acc = fx.Vector.filled(FUSED_WMMA_ACC, 0.0, fx.Float32)
    for step in range_constexpr(HEAD_DIM // FUSED_WMMA_K):
        k0 = step * FUSED_WMMA_K
        if const_expr(stage_q):
            q_fragment = _fused_lds_q_fragment(q_lds, k0, lane)
        else:
            q_fragment = _fused_q_fragment(
                args, tile, tokens, k0, lane
            )
        acc = _fused_wmma(
            q_fragment,
            _fused_k_fragment(kv_lds, local_key, k0, lane),
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
def _fused_poll_splits(mb, region, tile, splits):
    """Poll active tagged split flags in batches no wider than four entries."""
    if const_expr(splits <= 4):
        mb.poll(
            [
                (region, tile * FUSED_SPLITS + split, 1)
                for split in range_constexpr(splits)
            ],
            batch=4,
        )
    else:
        mb.poll(
            [
                (region, tile * FUSED_SPLITS + split, 1)
                for split in range_constexpr(4)
            ],
            batch=4,
        )
        mb.poll(
            [
                (region, tile * FUSED_SPLITS + split, 1)
                for split in range_constexpr(4, splits)
            ],
            batch=4,
        )
    # Pair with _fused_publish before loading the split statistics.
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)


def fused_attn_lds(register_probability=True, key_block=FUSED_KEY_BLOCK):
    if not register_probability:
        @fx.struct
        class LegacyLds:
            wave_stats: fx.Array[
                fx.Float32, WAVES * FUSED_ROWS * 2, 16
            ]
            row_stats: fx.Array[fx.Float32, FUSED_ROWS, 16]
            probabilities: fx.Array[
                fx.BFloat16, WAVES * FUSED_ROWS * key_block, 16
            ]
            pv: fx.Array[
                fx.Float32, WAVES * 32 * FUSED_WMMA_ACC, 16
            ]
            kv: fx.Array[
                fx.Int32,
                WAVES * 2 * key_block * HEAD_DIM // 4,
                16,
            ]

        return LegacyLds

    @fx.struct
    class Lds:
        wave_stats: fx.Array[
            fx.Float32, WAVES * FUSED_ROWS * 2, 16
        ]
        row_stats: fx.Array[fx.Float32, FUSED_ROWS, 16]
        q: fx.Array[fx.Int32, FUSED_ROWS * HEAD_DIM // 2, 16]
        pv: fx.Array[
            fx.Float32, WAVES * 32 * FUSED_WMMA_ACC, 16
        ]
        kv: fx.Array[
            fx.Int32,
            WAVES * 2 * key_block * HEAD_DIM // 4,
            16,
        ]

    return Lds


def fused_attn_fp8_lds():
    @fx.struct
    class Lds:
        wave_stats: fx.Array[
            fx.Float32, WAVES * FUSED_ROWS * 2, 16
        ]
        row_stats: fx.Array[fx.Float32, FUSED_ROWS, 16]
        probabilities: fx.Array[fx.Int32, 1, 16]
        pv: fx.Array[
            fx.Float32, WAVES * 32 * FUSED_WMMA_ACC, 16
        ]
        kv: fx.Array[
            fx.Int32,
            WAVES * 2 * FP8_KEY_BLOCK * FP8_LDS_ROW // 4,
            16,
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


def _fused_transpose_probabilities(words, lane):
    """Transpose packed two-half BF16 probabilities across one wave32."""
    values = words
    for bit in range_constexpr(3):
        offset = 1 << bit
        partners = [xshfl(value, offset) for value in values]
        transposed = []
        for element in range_constexpr(8):
            paired = element ^ offset
            transposed.append(
                ((lane & offset) != 0).select(
                    partners[paired] if element & offset == 0 else values[element],
                    values[element] if element & offset == 0 else partners[paired],
                )
            )
        values = transposed
    source_lane = (
        (lane & 7) | ((lane & 8) << 1) | ((lane & 16) >> 1)
    )
    return [lane_gather(value, source_lane) for value in values]


@flyc.jit
def run_fused_attn(c):
    """Run one split-key CTA; split zero performs ordered merges and output."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, args, mb = c["tile"], c["split"], c["args"], c["mb"]
    tokens, q_len = c["S"], c["QL"]
    splits = c.get("splits", FUSED_SPLITS)
    key_block = c.get("key_block", FUSED_KEY_BLOCK)
    load_bits = c.get("load_bits", 64)
    phase_trace = c.get("phase_trace", False)
    register_probability = c.get("register_probability", True)
    stage_q = c.get("stage_q", True)
    stats, glob, part = c["stats"], c["global"], c["part"]
    wave_stats, row_stats = c["wave_stats"], c["row_stats"]
    kv_lds = c["kv"]
    q_lds = c["q"]

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
    if const_expr(stage_q):
        q_row = tid // (HEAD_DIM // 4)
        q_dim = (tid % (HEAD_DIM // 4)) * 4
        q_token, q_head, _ = fused_row_coords(tile, q_row, tokens)
        packed_q = fx.Vector(
            bo.buffer_load(
                rsrc(args["q"]),
                (
                    q_token * Q_K
                    + q_head * HEAD_DIM
                    + q_dim
                )
                // 2,
                vec_width=2,
                dtype=T.i32,
            )
        )
        fx.ptr_store(packed_q, q_lds + tid * 2)
        gpu.barrier()

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
        c["probabilities"] + wave * FUSED_ROWS * key_block
    )
    qk_probability_cycles = fx.Int64(0)
    pv_cycles = fx.Int64(0)
    block_stride = splits * WAVES
    key_origin = fx.Int32(0)
    key_limit = keys
    if args["window"] > 0:
        token_last, _, _ = fused_row_coords(
            tile, fx.Int32(FUSED_ROWS - 1), tokens
        )
        key_begin = fx.max(
            prefix + token0 % q_len + 1 - args["window"],
            fx.Int32(0),
        )
        key_limit = fx.min(prefix + token_last % q_len + 1, keys)
        key_origin = (key_begin // key_block) * key_block
    iteration = fx.Int32(0)
    while (
        iteration * (block_stride * key_block)
        < key_limit - key_origin
    ):
        subphase_start = fx.Int64(0)
        subphase_middle = fx.Int64(0)
        if const_expr(phase_trace):
            subphase_start = clock()
        block = split * WAVES + wave + iteration * block_stride
        block0 = key_origin + block * key_block
        block_scores = []
        block_live = []
        pages = []
        for chunk in range_constexpr(key_block // FUSED_KEY_TILE):
            key0 = block0 + chunk * FUSED_KEY_TILE
            page_key = fx.min(key0, keys - 1)
            page = _fused_page(args, request, page_key)
            pages.append(page)
        gpu.barrier()
        lds_words = key_block * HEAD_DIM // 4
        load_bytes = load_bits // 8
        wave_kv_lds = kv_lds + wave * 2 * lds_words
        for load_iteration in range_constexpr(
            key_block * HEAD_DIM // (32 * load_bytes)
        ):
            local_byte = lane * load_bytes + load_iteration * 32 * load_bytes
            local_key = local_byte // HEAD_DIM
            key_dim = local_byte % HEAD_DIM
            logical_key = block0 + local_key
            safe_key = fx.min(logical_key, keys - 1)
            page = pages[0]
            for page_chunk in range_constexpr(
                1, key_block // FUSED_KEY_TILE
            ):
                if local_key >= page_chunk * FUSED_KEY_TILE:
                    page = pages[page_chunk]
            slot = page * 16 + safe_key % 16
            cache_byte = (
                (slot * LOCAL_KV_HEADS + fused_row_coords(
                    tile, fx.Int32(0), tokens
                )[2])
                * HEAD_DIM
                + key_dim
            )
            for cache_index, cache in enumerate(
                (args["k_cache"], args["v_cache"])
            ):
                packed = fx.Vector(
                    bo.buffer_load(
                        rsrc(cache),
                        cache_byte // 4,
                        vec_width=load_bits // 32,
                        dtype=T.i32,
                    )
                )
                fx.ptr_store(
                    packed,
                    wave_kv_lds
                    + cache_index * lds_words
                    + local_byte // 4,
                )
        gpu.barrier()
        for chunk in range_constexpr(key_block // FUSED_KEY_TILE):
            key0 = block0 + chunk * FUSED_KEY_TILE
            scores = _fused_qk(
                args,
                tile,
                tokens,
                key0,
                keys,
                lane,
                scale,
                wave_kv_lds,
                chunk * FUSED_KEY_TILE,
                q_lds,
                stage_q,
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
        probability_word_groups = [
            [] for _ in range((key_block + 31) // 32)
        ]
        for element in range_constexpr(FUSED_WMMA_ACC):
            local_max = fx.Float32(NEG)
            for chunk in range_constexpr(key_block // FUSED_KEY_TILE):
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
            probability_pair = []
            for chunk in range_constexpr(key_block // FUSED_KEY_TILE):
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
                rounded = fx.BFloat16(bf16_round(probability))
                if const_expr(register_probability):
                    probability_pair.append(rounded)
                else:
                    fx.ptr_store(
                        rounded,
                        probability_base
                        + row * key_block
                        + chunk * FUSED_KEY_TILE
                        + lane % FUSED_ROWS,
                    )
            if const_expr(register_probability):
                for group in range_constexpr((key_block + 31) // 32):
                    lo = probability_pair[group * 2]
                    hi = (
                        probability_pair[group * 2 + 1]
                        if group * 2 + 1 < key_block // FUSED_KEY_TILE
                        else fx.BFloat16(0.0)
                    )
                    probability_word_groups[group].append(
                        fx.Vector.from_elements(
                            [lo, hi], fx.BFloat16
                        ).bitcast(fx.Int32)[0]
                    )
            sums[element] = sums[element] * alpha + butterfly(
                lane_sum, (8, 4, 2, 1)
            )
            alphas.append(alpha)
            maxima[element] = new_max
        if const_expr(not register_probability):
            gpu.barrier()
        if const_expr(phase_trace):
            subphase_middle = clock()
            qk_probability_cycles = (
                qk_probability_cycles + subphase_middle - subphase_start
            )

        alpha_vector = fx.Vector.from_elements(alphas, fx.Float32)
        kh = (lane // FUSED_ROWS) * 8
        probability_row = lane % FUSED_ROWS
        for probability_group in range_constexpr((key_block + 31) // 32):
            p_values = []
            if const_expr(register_probability):
                gathered = [
                    fx.Vector.from_elements(
                        [word],
                        fx.Int32,
                    ).bitcast(fx.BFloat16)
                    for word in _fused_transpose_probabilities(
                        probability_word_groups[probability_group], lane
                    )
                ]
                for half in range_constexpr(2):
                    p_values += [
                        gathered[element][half]
                        for element in range_constexpr(8)
                    ]
            else:
                for half in range_constexpr(2):
                    p_values += [
                        fx.ptr_load(
                            probability_base
                            + probability_row * key_block
                            + probability_group * 32
                            + half * 16
                            + kh
                            + element
                        )
                        for element in range_constexpr(8)
                    ]
            p_fragment = fx.Vector.from_elements(p_values, fx.BFloat16)
            v_live = []
            for half in range_constexpr(2):
                for element in range_constexpr(8):
                    key = (
                        block0
                        + probability_group * 32
                        + half * 16
                        + kh
                        + element
                    )
                    v_live.append(
                        (key < keys) & (key < block0 + key_block)
                    )
            for dim_tile in range_constexpr(FUSED_DIM_TILES):
                v_values = []
                dim = dim_tile * FUSED_ROWS + lane % FUSED_ROWS
                for half in range_constexpr(2):
                    for element in range_constexpr(8):
                        index = half * 8 + element
                        local_key = (
                            probability_group * 32
                            + half * FUSED_KEY_TILE
                            + kh
                            + element
                        )
                        v_values.append(
                            v_live[index].select(
                                _fused_lds_fp8(
                                    wave_kv_lds + lds_words,
                                    local_key * HEAD_DIM + dim,
                                ),
                                fx.Float32(0.0),
                            )
                        )
                prior = pv_acc[dim_tile]
                if const_expr(probability_group == 0):
                    prior = prior * alpha_vector
                pv_acc[dim_tile] = _fused_wmma(
                    p_fragment,
                    fx.Vector.from_elements(
                        v_values, fx.Float32
                    ).to(fx.BFloat16),
                    prior,
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
        fx.ptr_store(split_max, row_stats + tid)
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
        _fused_poll_splits(mb, c["sflag"], tile, splits)
        if tid < FUSED_ROWS:
            global_max = fx.Float32(NEG)
            split_pairs = []
            for source in range_constexpr(splits):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        fused_stat_index(tile, source, tid),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                split_pairs.append(pair)
                global_max = fx.max(global_max, fx.Float32(pair[0]))
            global_sum = fx.Float32(0.0)
            for source in range_constexpr(splits):
                pair = split_pairs[source]
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

    row, col = tid // FUSED_ROWS, tid % FUSED_ROWS
    split_max = fx.ptr_load(
        row_stats + row,
        result_type=T.f32,
    )
    wave_weights = []
    for source_wave in range_constexpr(WAVES):
        wave_max = fx.ptr_load(
            wave_stats + (source_wave * FUSED_ROWS + row) * 2
        )
        wave_weights.append(
            hw_exp2((wave_max - split_max) * LOG2E)
        )
    for dim_tile in range_constexpr(FUSED_DIM_TILES):
        fx.ptr_store(
            pv_acc[dim_tile],
            c["pv"] + (wave * 32 + lane) * FUSED_WMMA_ACC,
        )
        gpu.barrier()
        result_lane = (row // 8) * 16 + col
        total = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                c["pv"]
                + (source_wave * 32 + result_lane) * FUSED_WMMA_ACC
                + row % 8
            ) * wave_weights[source_wave]
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

    if split < splits:
        _fused_poll_splits(mb, c["pflag"], tile, splits)
        global_lds = wave_stats
        split_lds = wave_stats + FUSED_ROWS * 2
        if tid < FUSED_ROWS * 2:
            fx.ptr_store(
                fx.Float32(
                    bo.buffer_load(
                        rsrc(glob.value),
                        tile * FUSED_ROWS * 2 + tid,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                ),
                global_lds + tid,
            )
        if tid < splits * FUSED_ROWS * 2:
            source = tid // (FUSED_ROWS * 2)
            item = tid % (FUSED_ROWS * 2)
            fx.ptr_store(
                fx.Float32(
                    bo.buffer_load(
                        rsrc(stats.value),
                        (tile * FUSED_SPLITS + source)
                        * FUSED_ROWS
                        * 2
                        + item,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                ),
                split_lds + tid,
            )
        gpu.barrier()
        for batch in range_constexpr(
            (
                FUSED_ROWS * HEAD_DIM
                + splits * 256
                - 1
            )
            // (splits * 256)
        ):
            item = tid + (split + batch * splits) * 256
            if item < FUSED_ROWS * HEAD_DIM:
                row, dim = item // HEAD_DIM, item % HEAD_DIM
                global_pair = fx.Vector.from_elements(
                    [
                        fx.ptr_load(global_lds + row * 2),
                        fx.ptr_load(global_lds + row * 2 + 1),
                    ],
                    fx.Float32,
                )
                total = fx.Float32(0.0)
                for source in range_constexpr(splits):
                    split_pair = fx.Vector.from_elements(
                        [
                            fx.ptr_load(
                                split_lds
                                + (source * FUSED_ROWS + row) * 2
                            ),
                            fx.ptr_load(
                                split_lds
                                + (source * FUSED_ROWS + row) * 2
                                + 1
                            ),
                        ],
                        fx.Float32,
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
                    )
                total = total / fx.Float32(global_pair[1])
                token, query_head, _ = fused_row_coords(tile, row, tokens)
                bo.buffer_store(
                    total.to(fx.BFloat16),
                    rsrc(args["out"]),
                    token * Q_K + query_head * HEAD_DIM + dim,
                )
    if const_expr(phase_trace):
        gpu.barrier()
    _fused_stamp(c, 5)


def _fused_q_fp8_fragment(args, tile, tokens, lane, scale):
    token, query_head, _ = fused_row_coords(
        tile, lane % FUSED_ROWS, tokens
    )
    half = lane // FUSED_ROWS
    words = []
    base = (
        token * Q_K
        + query_head * HEAD_DIM
        + half * (HEAD_DIM // 2)
    )
    for chunk in range_constexpr(4):
        loaded = fx.Vector(
            bo.buffer_load(
                rsrc(args["q"]),
                base // 2 + chunk * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        words += [loaded[index] for index in range_constexpr(4)]
    values = (
        fx.Vector.from_elements(words, fx.Int32)
        .bitcast(fx.BFloat16)
        .to(fx.Float32)
    )
    absmax = fx.Float32(1.0e-20)
    for element in range_constexpr(HEAD_DIM // 2):
        value = fx.Float32(values[element])
        absmax = fx.max(absmax, fx.max(value, -value))
    absmax = fx.max(
        absmax,
        absmax.shuffle_xor(fx.Int32(16), fx.Int32(32)),
    )
    inverse = fx.Float32(448.0) / absmax
    fragment = _pack_fp8x32(
        [
            fx.Float32(values[element]) * inverse
            for element in range_constexpr(HEAD_DIM // 2)
        ]
    )
    return fragment, absmax * fx.Float32(1.0 / 448.0) * scale


def _fused_k_fp8_fragment(k_lds, local_key, lane):
    half = lane // FUSED_ROWS
    byte = local_key * FP8_LDS_ROW + half * (HEAD_DIM // 2)
    lo = fx.Vector(
        fx.ptr_load(
            k_lds + byte // 4,
            result_type=T.vec(4, T.i32),
        )
    )
    hi = fx.Vector(
        fx.ptr_load(
            k_lds + byte // 4 + 4,
            result_type=T.vec(4, T.i32),
        )
    )
    return lo.shuffle(hi, list(range(8)))


def _fused_v_fp8_fragment(v_bytes, dim_tile, lane):
    tr_row = (
        (lane // FUSED_ROWS) * 8
        + lane % 4
        + ((lane % FUSED_ROWS) // 8) * 4
    )
    tr_col = (((lane % FUSED_ROWS) // 4) % 2) * 8
    raws = []
    for tile in range_constexpr(4):
        pointer = fx.to_llvm_ptr(
            v_bytes
            + tile * FUSED_ROWS * FP8_LDS_ROW
            + dim_tile * FUSED_ROWS
            + tr_row * FP8_LDS_ROW
            + tr_col
        )
        raws.append(
            fx.Vector(
                _rocdl.ds_load_tr8_b64(T.vec(2, T.i32), pointer)
            )
        )
    return (
        raws[0]
        .shuffle(raws[1], [0, 1, 2, 3])
        .shuffle(
            raws[2].shuffle(raws[3], [0, 1, 2, 3]),
            list(range(8)),
        )
    )


@flyc.jit
def run_fused_attn_fp8(c):
    """Experimental FP8-WMMA inner loop with the production split ABI."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, args, mb = c["tile"], c["split"], c["args"], c["mb"]
    tokens, q_len = c["S"], c["QL"]
    splits = c.get("splits", FUSED_SPLITS)
    phase_trace = c.get("phase_trace", False)
    stats, glob, part = c["stats"], c["global"], c["part"]
    wave_stats, row_stats = c["wave_stats"], c["row_stats"]
    kv_lds = c["kv"]

    token0, _, kv_head = fused_row_coords(tile, fx.Int32(0), tokens)
    request = token0 // q_len
    prefix = fx.Int32(
        bo.buffer_load(
            rsrc(args["prefix_lens"]), request, vec_width=1, dtype=T.i32
        )
    )
    keys = prefix + q_len
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    _fused_stamp(c, 0)

    q_fragment, q_scale = _fused_q_fp8_fragment(
        args, tile, tokens, lane, scale
    )
    maximum = fx.Float32(NEG)
    denominator = fx.Float32(0.0)
    pv_acc = [
        fx.Vector.filled(FUSED_WMMA_ACC, 0.0, fx.Float32)
        for _ in range(FUSED_DIM_TILES)
    ]
    qk_probability_cycles = fx.Int64(0)
    pv_cycles = fx.Int64(0)
    block_stride = splits * WAVES
    key_origin = fx.Int32(0)
    key_limit = keys
    if args["window"] > 0:
        token_last, _, _ = fused_row_coords(
            tile, fx.Int32(FUSED_ROWS - 1), tokens
        )
        key_begin = fx.max(
            prefix + token0 % q_len + 1 - args["window"],
            fx.Int32(0),
        )
        key_limit = fx.min(prefix + token_last % q_len + 1, keys)
        key_origin = (key_begin // FP8_KEY_BLOCK) * FP8_KEY_BLOCK

    lds_words = FP8_KEY_BLOCK * FP8_LDS_ROW // 4
    wave_kv_lds = kv_lds + wave * 2 * lds_words
    v_bytes = fx.recast_iter(fx.Int8, wave_kv_lds + lds_words)
    iteration = fx.Int32(0)
    while (
        iteration * (block_stride * FP8_KEY_BLOCK)
        < key_limit - key_origin
    ):
        subphase_start = fx.Int64(0)
        subphase_middle = fx.Int64(0)
        if const_expr(phase_trace):
            subphase_start = clock()
        block = split * WAVES + wave + iteration * block_stride
        block0 = key_origin + block * FP8_KEY_BLOCK
        pages = []
        for chunk in range_constexpr(4):
            page_key = fx.min(
                block0 + chunk * FUSED_ROWS,
                keys - 1,
            )
            pages.append(_fused_page(args, request, page_key))
        gpu.barrier()
        for load_iteration in range_constexpr(
            FP8_KEY_BLOCK * HEAD_DIM // (32 * 8)
        ):
            local_byte = lane * 8 + load_iteration * 32 * 8
            local_key = local_byte // HEAD_DIM
            key_dim = local_byte % HEAD_DIM
            logical_key = block0 + local_key
            safe_key = fx.min(logical_key, keys - 1)
            page = pages[0]
            if local_key >= FUSED_ROWS:
                page = pages[1]
            if local_key >= 2 * FUSED_ROWS:
                page = pages[2]
            if local_key >= 3 * FUSED_ROWS:
                page = pages[3]
            slot = page * 16 + safe_key % 16
            cache_byte = (
                (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM
                + key_dim
            )
            for cache_index, cache in enumerate(
                (args["k_cache"], args["v_cache"])
            ):
                packed = fx.Vector(
                    bo.buffer_load(
                        rsrc(cache),
                        cache_byte // 4,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                fx.ptr_store(
                    packed,
                    wave_kv_lds
                    + cache_index * lds_words
                    + (local_key * FP8_LDS_ROW + key_dim) // 4,
                )
        gpu.barrier()

        block_scores = []
        block_live = []
        for chunk in range_constexpr(4):
            key0 = block0 + chunk * FUSED_ROWS
            scores = _wmma_fp8(
                _fused_k_fp8_fragment(
                    wave_kv_lds,
                    chunk * FUSED_ROWS + lane % FUSED_ROWS,
                    lane,
                ),
                q_fragment,
                fx.Vector.filled(FUSED_WMMA_ACC, 0.0, fx.Float32),
            )
            lives = []
            key = key0 + (lane // FUSED_ROWS) * 8
            for element in range_constexpr(FUSED_WMMA_ACC):
                row = lane % FUSED_ROWS
                token, _, _ = fused_row_coords(tile, row, tokens)
                end = prefix + token % q_len + 1
                begin = fx.Int32(0)
                if args["window"] > 0:
                    begin = fx.max(end - args["window"], fx.Int32(0))
                logical_key = key + element
                lives.append(
                    (logical_key >= begin) & (logical_key < end)
                )
            block_scores.append(scores)
            block_live.append(lives)

        local_max = fx.Float32(NEG)
        for chunk in range_constexpr(4):
            for element in range_constexpr(FUSED_WMMA_ACC):
                natural_score = (
                    fx.Float32(block_scores[chunk][element]) * q_scale
                )
                local_max = fx.max(
                    local_max,
                    block_live[chunk][element].select(
                        natural_score,
                        fx.Float32(NEG),
                    ),
                )
        block_max = fx.max(
            local_max,
            local_max.shuffle_xor(fx.Int32(16), fx.Int32(32)),
        )
        new_max = fx.max(maximum, block_max)
        alpha = hw_exp2((maximum - new_max) * LOG2E)
        lane_sum = fx.Float32(0.0)
        probability_values = []
        for chunk in range_constexpr(4):
            chunk_probabilities = []
            for element in range_constexpr(FUSED_WMMA_ACC):
                probability = block_live[chunk][element].select(
                    hw_exp2(
                        (
                            fx.Float32(block_scores[chunk][element])
                            * q_scale
                            - new_max
                        )
                        * LOG2E
                        + FP8_PROBABILITY_BIAS
                    ),
                    fx.Float32(0.0),
                )
                lane_sum = lane_sum + probability
                chunk_probabilities.append(probability)
            probability_values.append(chunk_probabilities)
        block_sum = lane_sum + lane_sum.shuffle_xor(
            fx.Int32(16), fx.Int32(32)
        )
        denominator = denominator * alpha + block_sum
        maximum = new_max
        if const_expr(phase_trace):
            subphase_middle = clock()
            qk_probability_cycles = (
                qk_probability_cycles + subphase_middle - subphase_start
            )

        p_values = []
        for chunk in range_constexpr(4):
            for element in range_constexpr(FUSED_WMMA_ACC):
                p_values.append(probability_values[chunk][element])
        p_fragment = _pack_fp8x32(p_values)
        for dim_tile in range_constexpr(FUSED_DIM_TILES):
            pv_acc[dim_tile] = _wmma_fp8(
                _fused_v_fp8_fragment(v_bytes, dim_tile, lane),
                p_fragment,
                pv_acc[dim_tile] * alpha,
            )
        if const_expr(phase_trace):
            subphase_end = clock()
            pv_cycles = pv_cycles + subphase_end - subphase_middle
        iteration = iteration + 1

    if lane < FUSED_ROWS:
        fx.ptr_store(
            maximum,
            wave_stats + (wave * FUSED_ROWS + lane) * 2,
        )
        fx.ptr_store(
            denominator,
            wave_stats + (wave * FUSED_ROWS + lane) * 2 + 1,
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
        fx.ptr_store(split_max, row_stats + tid)
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
        _fused_poll_splits(mb, c["sflag"], tile, splits)
        if tid < FUSED_ROWS:
            global_max = fx.Float32(NEG)
            split_pairs = []
            for source in range_constexpr(splits):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        fused_stat_index(tile, source, tid),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                split_pairs.append(pair)
                global_max = fx.max(global_max, fx.Float32(pair[0]))
            global_sum = fx.Float32(0.0)
            for source in range_constexpr(splits):
                pair = split_pairs[source]
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

    row, col = tid // FUSED_ROWS, tid % FUSED_ROWS
    split_max = fx.ptr_load(row_stats + row, result_type=T.f32)
    wave_weights = []
    for source_wave in range_constexpr(WAVES):
        wave_max = fx.ptr_load(
            wave_stats + (source_wave * FUSED_ROWS + row) * 2
        )
        wave_weights.append(
            hw_exp2((wave_max - split_max) * LOG2E)
        )
    for dim_tile in range_constexpr(FUSED_DIM_TILES):
        fx.ptr_store(
            pv_acc[dim_tile],
            c["pv"] + (wave * 32 + lane) * FUSED_WMMA_ACC,
        )
        gpu.barrier()
        result_lane = row + (col // 8) * FUSED_ROWS
        total = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                c["pv"]
                + (source_wave * 32 + result_lane) * FUSED_WMMA_ACC
                + col % 8
            ) * wave_weights[source_wave]
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

    if split < splits:
        _fused_poll_splits(mb, c["pflag"], tile, splits)
        global_lds = wave_stats
        split_lds = wave_stats + FUSED_ROWS * 2
        if tid < FUSED_ROWS * 2:
            fx.ptr_store(
                fx.Float32(
                    bo.buffer_load(
                        rsrc(glob.value),
                        tile * FUSED_ROWS * 2 + tid,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                ),
                global_lds + tid,
            )
        if tid < splits * FUSED_ROWS * 2:
            source = tid // (FUSED_ROWS * 2)
            item = tid % (FUSED_ROWS * 2)
            fx.ptr_store(
                fx.Float32(
                    bo.buffer_load(
                        rsrc(stats.value),
                        (tile * FUSED_SPLITS + source)
                        * FUSED_ROWS
                        * 2
                        + item,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                ),
                split_lds + tid,
            )
        gpu.barrier()
        for batch in range_constexpr(
            (
                FUSED_ROWS * HEAD_DIM
                + splits * 256
                - 1
            )
            // (splits * 256)
        ):
            item = tid + (split + batch * splits) * 256
            if item < FUSED_ROWS * HEAD_DIM:
                row, dim = item // HEAD_DIM, item % HEAD_DIM
                global_pair = fx.Vector.from_elements(
                    [
                        fx.ptr_load(global_lds + row * 2),
                        fx.ptr_load(global_lds + row * 2 + 1),
                    ],
                    fx.Float32,
                )
                total = fx.Float32(0.0)
                for source in range_constexpr(splits):
                    split_pair = fx.Vector.from_elements(
                        [
                            fx.ptr_load(
                                split_lds
                                + (source * FUSED_ROWS + row) * 2
                            ),
                            fx.ptr_load(
                                split_lds
                                + (source * FUSED_ROWS + row) * 2
                                + 1
                            ),
                        ],
                        fx.Float32,
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
                    )
                total = total / fx.Float32(global_pair[1])
                token, query_head, _ = fused_row_coords(tile, row, tokens)
                bo.buffer_store(
                    total.to(fx.BFloat16),
                    rsrc(args["out"]),
                    token * Q_K + query_head * HEAD_DIM + dim,
                )
    if const_expr(phase_trace):
        gpu.barrier()
    _fused_stamp(c, 5)
