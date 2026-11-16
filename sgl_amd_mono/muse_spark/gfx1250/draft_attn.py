# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 DFlash2 TP4 draft attention.

One CTA computes one of the 8 x 16 (token, local query head) rows.  Its eight
wave32 waves split the keys and use three deliberately simple passes: maximum,
normalizer, and BF16-probability times V.  Every row attends the same noncausal
eight-row draft block plus the compact suffix of the prefix selected by
``window``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import bf16_round, bf2_f32, butterfly, hw_exp2, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.muse_spark.gfx1250.device import (
    mailbox_guard,
    step_tag,
    wait_stores,
)
from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
    BLOCK,
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    RESIDENT_BLOCKS,
    SUPPORTED_TOKENS,
)
from sgl_amd_mono.muse_spark.gfx1250.draft_layout import THREADS, WAVE_SIZE, WAVES

GQA = LOCAL_Q_HEADS // LOCAL_KV_HEADS
TASKS = BLOCK * LOCAL_Q_HEADS
LOG2E = 1.4426950408889634
NEG = -1.0e30
STATS_FLOATS = 2 * WAVES + 2
PARTIAL_FLOATS = WAVES * HEAD_DIM

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


def _load_bf16x8(resource, element):
    words = fx.Vector(
        bo.buffer_load(resource, element // 2, vec_width=4, dtype=T.i32)
    )
    return words.bitcast(fx.BFloat16).to(fx.Float32)


def _load_fp8x8(resource, byte):
    """Load and widen eight e4m3fn values without wave64-only conversion ops."""
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


def _i32_at(address, index, is_i64):
    word = (is_i64 != 0).select(2 * index, index)
    return fx.Int32(
        bo.buffer_load(rsrc(address), word, vec_width=1, dtype=T.i32)
    )


def _score(args, qv, request_row, key, kv_head, lane16, scale):
    slot = fx.Int32(
        bo.buffer_load(
            rsrc(args["req_to_token"], args["table_bytes"]),
            request_row * args["table_stride"] + key,
            vec_width=1,
            dtype=T.i32,
        )
    )
    byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane16 * 8
    kval = _load_fp8x8(rsrc(args["k_cache"]), byte)
    dot = fx.Float32(0.0)
    for i in range_constexpr(8):
        dot = dot + qv[i] * kval[i]
    return butterfly(dot, (8, 4, 2, 1)) * scale, slot


@flyc.jit
def run_dattn(c):
    """Compute one noncausal draft-block attention row."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    args, stats, partial = c["args"], c["stats"], c["partial"]
    token = c["task"] // LOCAL_Q_HEADS
    query_head = c["task"] % LOCAL_Q_HEADS
    kv_head = query_head // GQA
    lane16 = lane % 16

    prefix = _i32_at(args["seq_lens"], 0, args["lens64"])
    keys = prefix + BLOCK
    request_row = _i32_at(args["req_pool"], 0, args["rpi64"])
    begin = fx.Int32(0)
    if args["window"] > 0:
        lower = prefix - args["window"]
        begin = (lower > 0).select(lower, begin)
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    qv = _load_bf16x8(
        rsrc(args["q"]),
        token * Q_K + query_head * HEAD_DIM + lane16 * 8,
    )

    local_max = fx.Float32(NEG)
    key = begin + wave
    while key < keys:
        score, _ = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
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

    local_sum = fx.Float32(0.0)
    key = begin + wave
    while key < keys:
        score, _ = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
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

    output = fx.Vector.filled(8, 0.0, fx.Float32)
    key = begin + wave
    while key < keys:
        score, slot = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
        probability = bf16_round(
            hw_exp2((score - block_max) * LOG2E) / block_sum
        )
        byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane16 * 8
        value = _load_fp8x8(rsrc(args["v_cache"]), byte)
        output = output + value * probability
        key = key + WAVES
    if lane < HEAD_DIM // 8:
        fx.ptr_store(output, partial + wave * HEAD_DIM + lane * 8)
    gpu.barrier()

    if tid < HEAD_DIM // 8:
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


def dattn_lds():
    @fx.struct
    class Lds:
        stats: fx.Array[fx.Float32, STATS_FLOATS, 16]
        partial: fx.Array[fx.Float32, PARTIAL_FLOATS, 16]

    return Lds


class DAttnBuild(msgspec.Struct, frozen=True):
    tokens: int = BLOCK
    q_len: int = BLOCK
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


def build_dattn(key: DAttnBuild):
    """Build standalone gfx1250 D-attn for the initial width/q_len-8 profile."""
    if (key.tokens, key.q_len) != (MAX_TOKENS, BLOCK):
        raise ValueError(
            f"gfx1250 D-attn supports tokens=q_len={BLOCK}, "
            f"got {(key.tokens, key.q_len)}"
        )
    if key.resident_blocks not in (96, 192):
        raise ValueError(
            "gfx1250 D-attn resident_blocks must be 96 or 192, "
            f"got {key.resident_blocks}"
        )

    name = kernel_symbol(
        "muse_spark_gfx1250_dattn",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
        f=key.fault_site,
    )
    keyed = key_tuple(key, SOURCES)
    Lds = dattn_lds()
    rounds = (TASKS + key.resident_blocks - 1) // key.resident_blocks

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        epoch: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        args = {
            "q": q,
            "out": out,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "req_to_token": req_to_token,
            "table_stride": table_stride,
            "table_bytes": table_bytes,
            "req_pool": req_pool,
            "rpi64": rpi64,
            "seq_lens": seq_lens,
            "lens64": lens64,
            "window": window,
            "scale_bits": scale_bits,
        }
        active_guard = mailbox_guard(
            epoch,
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
        with guard.scope(active_guard):
            for task_round in range_constexpr(rounds):
                task = bid + task_round * key.resident_blocks
                if task < TASKS:
                    run_dattn(
                        {
                            "tid": tid,
                            "lane": tid % 32,
                            "wave": tid // 32,
                            "task": task,
                            "stats": lds.stats.ptr,
                            "partial": lds.partial.ptr,
                            "args": args,
                        }
                    )

    @flyc.jit
    def launch(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dattn(
            q,
            out,
            k_cache,
            v_cache,
            req_to_token,
            table_stride,
            table_bytes,
            req_pool,
            rpi64,
            seq_lens,
            lens64,
            window,
            scale_bits,
            epoch,
        ).launch(
            grid=(key.resident_blocks,),
            block=(THREADS,),
            stream=stream,
        )

    return launch


# ---------------------------------------------------------------------------
# cu96_tp4 split-key wave32 WMMA implementation.  The definitions below
# intentionally replace the correctness-first scalar builder above while
# retaining it as a readable fallback reference.

SPLIT_ROWS = 16
SPLIT_COUNT = 12
ROW_TILES = LOCAL_KV_HEADS * (BLOCK // 2)
SPLIT_TASKS = ROW_TILES * SPLIT_COUNT
MAX_REQUESTS = MAX_TOKENS // BLOCK
MAX_ROW_TILES = MAX_REQUESTS * ROW_TILES
MAX_SPLIT_TASKS = MAX_REQUESTS * SPLIT_TASKS
WMMA_K = 32
WMMA_ACC = 8
KEY_TILE = 16
PV_KEYS = 32
DIM_TILES = HEAD_DIM // SPLIT_ROWS

assert (LOCAL_Q_HEADS, LOCAL_KV_HEADS, GQA) == (16, 2, 8)
assert (ROW_TILES, SPLIT_TASKS) == (8, 96)
assert HEAD_DIM == 4 * WMMA_K


def _align(value, alignment=256):
    return (value + alignment - 1) // alignment * alignment


def dattn_scratch_layout(tokens: int = BLOCK, q_len: int = BLOCK) -> dict:
    """Plain payloads followed by tagged phase flags."""
    if tokens not in SUPPORTED_TOKENS or q_len != BLOCK:
        raise ValueError(
            f"gfx1250 D-attn supports rows {SUPPORTED_TOKENS}, q_len={BLOCK}; "
            f"got {(tokens, q_len)}"
        )
    requests = tokens // BLOCK
    row_tiles = requests * ROW_TILES
    split_tasks = requests * SPLIT_TASKS
    stat_floats = row_tiles * SPLIT_COUNT * SPLIT_ROWS * 2
    global_floats = row_tiles * SPLIT_ROWS * 2
    pv_part_floats = row_tiles * SPLIT_COUNT * SPLIT_ROWS * HEAD_DIM
    layout, offset = {}, 0
    for name, size in (
        ("dattn_stats", stat_floats * 4),
        ("dattn_global", global_floats * 4),
        ("dattn_part", pv_part_floats * 4),
        ("dattn_sflag", split_tasks * 8),
        ("dattn_gflag", row_tiles * 8),
        ("dattn_pflag", split_tasks * 8),
    ):
        offset = _align(offset)
        layout[name] = (offset, size)
        offset += size
    return layout


def dattn_scratch_bytes(tokens: int = MAX_TOKENS) -> int:
    return max(
        offset + size
        for offset, size in dattn_scratch_layout(tokens, BLOCK).values()
    )


def _split_fp8(address, byte):
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


def _split_bf16(address, element):
    word = fx.Int32(
        bo.buffer_load(rsrc(address), element // 2, vec_width=1, dtype=T.i32)
    )
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _split_slot(args, request_row, key):
    return fx.Int32(
        bo.buffer_load(
            rsrc(args["req_to_token"], args["table_bytes"]),
            request_row * args["table_stride"] + key,
            vec_width=1,
            dtype=T.i32,
        )
    )


def _split_row_coords(tile, row):
    """The required tile -> KV head, logical row -> token/query-head map."""
    request = tile // ROW_TILES
    request_tile = tile % ROW_TILES
    kv_head = request_tile // (BLOCK // 2)
    row_tile = request_tile % (BLOCK // 2)
    logical = row_tile * SPLIT_ROWS + row
    token = request * BLOCK + logical // GQA
    query_head = kv_head * GQA + logical % GQA
    return token, query_head, kv_head


def _split_q_fragment(args, tile, k0, lane):
    token, query_head, _ = _split_row_coords(tile, lane % SPLIT_ROWS)
    kh = (lane // SPLIT_ROWS) * 8
    values = []
    for half in range_constexpr(2):
        for element in range_constexpr(8):
            dim = k0 + half * 16 + kh + element
            values.append(
                _split_bf16(
                    args["q"],
                    token * Q_K + query_head * HEAD_DIM + dim,
                )
            )
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)


def _split_k_fragment(
    args, request_row, kv_head, key0, keys, k0, lane
):
    key = fx.min(key0 + lane % SPLIT_ROWS, keys - 1)
    slot = _split_slot(args, request_row, key)
    kh = (lane // SPLIT_ROWS) * 8
    values = []
    for half in range_constexpr(2):
        for element in range_constexpr(8):
            dim = k0 + half * 16 + kh + element
            byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + dim
            values.append(_split_fp8(args["k_cache"], byte))
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)


def _split_wmma(a, b, acc):
    result = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(WMMA_ACC, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        reuseA=False,
        reuseB=False,
    ).result
    return fx.Vector(result)


def _split_qk(args, tile, request_row, key0, keys, lane, scale):
    _, _, kv_head = _split_row_coords(tile, fx.Int32(0))
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range_constexpr(HEAD_DIM // WMMA_K):
        k0 = step * WMMA_K
        acc = _split_wmma(
            _split_q_fragment(args, tile, k0, lane),
            _split_k_fragment(
                args, request_row, kv_head, key0, keys, k0, lane
            ),
            acc,
        )
    return acc * scale


def _split_max(a, b):
    return fx.max(a, b)


def _split_stat_index(tile, split, row):
    return ((tile * SPLIT_COUNT + split) * SPLIT_ROWS + row) * 2


def _split_global_index(tile, row):
    return (tile * SPLIT_ROWS + row) * 2


def _split_part_index(tile, split, row, dim):
    return (
        ((tile * SPLIT_COUNT + split) * SPLIT_ROWS + row) * HEAD_DIM
        + dim
    )


@flyc.jit
def _split_publish(put, region, index, value, owner):
    wait_stores()
    gpu.barrier()
    if owner:
        put(region, index, value)


@flyc.jit
def _poll_twelve_flags(mb, region, base):
    """Three batches of four independent v2i32 tagged-pair loads."""
    for group in range_constexpr(3):
        mb.poll(
            [
                (region, base + group * 4 + element, 1)
                for element in range(4)
            ],
            batch=4,
        )


def split_dattn_lds():
    @fx.struct
    class Lds:
        wave_stats: fx.Array[fx.Float32, WAVES * SPLIT_ROWS, 16]
        row_stats: fx.Array[fx.Float32, SPLIT_ROWS, 16]
        probabilities: fx.Array[
            fx.BFloat16, WAVES * SPLIT_ROWS * PV_KEYS, 16
        ]
        pv: fx.Array[fx.Float32, WAVES * WAVE_SIZE * WMMA_ACC, 16]

    return Lds


@flyc.jit
def run_split_dattn(c):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split = c["tile"], c["split"]
    args, mb = c["args"], c["mb"]
    stats, glob, part = c["stats"], c["global"], c["part"]
    wave_stats, row_stats = c["wave_stats"], c["row_stats"]

    request = tile // ROW_TILES
    prefix = _i32_at(args["seq_lens"], request, args["lens64"])
    keys = prefix + BLOCK
    request_row = _i32_at(args["req_pool"], request, args["rpi64"])
    begin = fx.Int32(0)
    if args["window"] > 0:
        begin = fx.max(prefix - args["window"], fx.Int32(0))
    span = keys - begin
    split_keys = (span + SPLIT_COUNT - 1) // SPLIT_COUNT
    key_lo = fx.min(begin + split * split_keys, keys)
    key_hi = fx.min(key_lo + split_keys, keys)
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)

    # Split max: 16x16 QK, four K=32 WMMA steps.
    maxima = [fx.Float32(NEG) for _ in range(WMMA_ACC)]
    for key0 in range(
        key_lo + wave * KEY_TILE, key_hi, WAVES * KEY_TILE
    ):
        scores = _split_qk(
            args, tile, request_row, key0, keys, lane, scale
        )
        live = key0 + lane % SPLIT_ROWS < key_hi
        for element in range_constexpr(WMMA_ACC):
            score = live.select(
                fx.Float32(scores[element]), fx.Float32(NEG)
            )
            maxima[element] = fx.max(maxima[element], score)
    for element in range_constexpr(WMMA_ACC):
        maxima[element] = butterfly(
            maxima[element], (8, 4, 2, 1), op=_split_max
        )
        if lane % SPLIT_ROWS == 0:
            row = (lane // SPLIT_ROWS) * 8 + element
            fx.ptr_store(
                maxima[element], wave_stats + wave * SPLIT_ROWS + row
            )
    gpu.barrier()
    if tid < SPLIT_ROWS:
        value = fx.ptr_load(wave_stats + tid)
        for source_wave in range_constexpr(1, WAVES):
            value = fx.max(
                value,
                fx.ptr_load(
                    wave_stats + source_wave * SPLIT_ROWS + tid
                ),
            )
        fx.ptr_store(value, row_stats + tid)
    gpu.barrier()

    # Split denominator relative to the split max.
    sums = [fx.Float32(0.0) for _ in range(WMMA_ACC)]
    for key0 in range(
        key_lo + wave * KEY_TILE, key_hi, WAVES * KEY_TILE
    ):
        scores = _split_qk(
            args, tile, request_row, key0, keys, lane, scale
        )
        live = key0 + lane % SPLIT_ROWS < key_hi
        for element in range_constexpr(WMMA_ACC):
            row = (lane // SPLIT_ROWS) * 8 + element
            term = hw_exp2(
                (
                    fx.Float32(scores[element])
                    - fx.ptr_load(row_stats + row)
                )
                * LOG2E
            )
            sums[element] = sums[element] + live.select(
                term, fx.Float32(0.0)
            )
    for element in range_constexpr(WMMA_ACC):
        sums[element] = butterfly(sums[element], (8, 4, 2, 1))
        if lane % SPLIT_ROWS == 0:
            row = (lane // SPLIT_ROWS) * 8 + element
            fx.ptr_store(
                sums[element], wave_stats + wave * SPLIT_ROWS + row
            )
    gpu.barrier()
    if tid < SPLIT_ROWS:
        total = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                wave_stats + source_wave * SPLIT_ROWS + tid
            )
        bo.buffer_store(
            fx.Vector.from_elements(
                [fx.ptr_load(row_stats + tid), total], fx.Float32
            ),
            rsrc(stats.value),
            _split_stat_index(tile, split, tid),
            cache_modifier=2,
        )
    _split_publish(
        mb.put,
        c["sflag"],
        tile * SPLIT_COUNT + split,
        fx.Int32(1),
        tid == 0,
    )

    # Split zero merges in exact split order.  The three flag polls each load
    # only four tagged pairs (v8i32).
    if split == 0:
        _poll_twelve_flags(
            mb, c["sflag"], tile * SPLIT_COUNT
        )
        if tid < SPLIT_ROWS:
            global_max = fx.Float32(NEG)
            for source in range_constexpr(SPLIT_COUNT):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        _split_stat_index(tile, source, tid),
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
                global_max = fx.max(global_max, fx.Float32(pair[0]))
            global_sum = fx.Float32(0.0)
            for source in range_constexpr(SPLIT_COUNT):
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(stats.value),
                        _split_stat_index(tile, source, tid),
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
                _split_global_index(tile, tid),
                cache_modifier=2,
            )
        _split_publish(
            mb.put, c["gflag"], tile, fx.Int32(1), tid == 0
        )
    mb.poll([(c["gflag"], tile, 1)], batch=1)

    # Recompute QK after the final denominator, round P to BF16, and pair two
    # adjacent 16-key chunks into each K=32 PV WMMA.
    pair0 = key_lo + wave * PV_KEYS
    probability_base = (
        c["probabilities"] + wave * SPLIT_ROWS * PV_KEYS
    )
    for key_tile in range_constexpr(2):
        key0 = pair0 + key_tile * KEY_TILE
        scores = _split_qk(
            args, tile, request_row, key0, keys, lane, scale
        )
        live = key0 + lane % SPLIT_ROWS < key_hi
        for element in range_constexpr(WMMA_ACC):
            row = (lane // SPLIT_ROWS) * 8 + element
            global_stats = fx.Vector(
                bo.buffer_load(
                    rsrc(glob.value),
                    _split_global_index(tile, row),
                    vec_width=2,
                    dtype=T.f32,
                    cache_modifier=2,
                )
            )
            probability = bf16_round(
                hw_exp2(
                    (
                        fx.Float32(scores[element])
                        - fx.Float32(global_stats[0])
                    )
                    * LOG2E
                )
                / fx.Float32(global_stats[1])
            )
            fx.ptr_store(
                live.select(
                    fx.BFloat16(probability), fx.BFloat16(0.0)
                ),
                probability_base
                + row * PV_KEYS
                + key_tile * KEY_TILE
                + lane % SPLIT_ROWS,
            )
    gpu.barrier()

    kh = (lane // SPLIT_ROWS) * 8
    probability_row = lane % SPLIT_ROWS
    p_values = []
    for half in range_constexpr(2):
        p_values += [
            fx.ptr_load(
                probability_base
                + probability_row * PV_KEYS
                + half * 16
                + kh
                + element
            )
            for element in range_constexpr(8)
        ]
    p_fragment = fx.Vector.from_elements(p_values, fx.BFloat16)
    _, _, kv_head = _split_row_coords(tile, fx.Int32(0))
    for dim_tile in range_constexpr(DIM_TILES):
        v_values = []
        dim = dim_tile * SPLIT_ROWS + lane % SPLIT_ROWS
        for half in range_constexpr(2):
            for element in range_constexpr(8):
                key = pair0 + half * 16 + kh + element
                safe_key = fx.min(key, keys - 1)
                slot = _split_slot(args, request_row, safe_key)
                byte = (
                    (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + dim
                )
                v_values.append(
                    (key < key_hi).select(
                        _split_fp8(args["v_cache"], byte),
                        fx.Float32(0.0),
                    )
                )
        v_fragment = fx.Vector.from_elements(
            v_values, fx.Float32
        ).to(fx.BFloat16)
        acc = _split_wmma(
            p_fragment,
            v_fragment,
            fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32),
        )
        fx.ptr_store(
            acc,
            c["pv"] + (wave * WAVE_SIZE + lane) * WMMA_ACC,
        )
        gpu.barrier()
        row, col = tid // SPLIT_ROWS, tid % SPLIT_ROWS
        total = fx.Float32(0.0)
        result_lane = (row // 8) * 16 + col
        for source_wave in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                c["pv"]
                + (source_wave * WAVE_SIZE + result_lane) * WMMA_ACC
                + row % 8
            )
        bo.buffer_store(
            total,
            rsrc(part.value),
            _split_part_index(
                tile, split, row, dim_tile * SPLIT_ROWS + col
            ),
            cache_modifier=2,
        )
        gpu.barrier()
    _split_publish(
        mb.put,
        c["pflag"],
        tile * SPLIT_COUNT + split,
        fx.Int32(1),
        tid == 0,
    )

    # Fixed split-order FP32 merge; output converts to BF16 only at the end.
    if split == 0:
        _poll_twelve_flags(
            mb, c["pflag"], tile * SPLIT_COUNT
        )
        for batch in range_constexpr(
            (SPLIT_ROWS * HEAD_DIM) // THREADS
        ):
            item = tid + batch * THREADS
            row, dim = item // HEAD_DIM, item % HEAD_DIM
            total = fx.Float32(0.0)
            for source in range_constexpr(SPLIT_COUNT):
                total = total + fx.Float32(
                    bo.buffer_load(
                        rsrc(part.value),
                        _split_part_index(
                            tile, source, row, dim
                        ),
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=2,
                    )
                )
            token, query_head, _ = _split_row_coords(tile, row)
            bo.buffer_store(
                total.to(fx.BFloat16),
                rsrc(args["out"]),
                token * Q_K + query_head * HEAD_DIM + dim,
            )


class DAttnBuild(msgspec.Struct, frozen=True):
    tokens: int = BLOCK
    q_len: int = BLOCK
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


def build_dattn(key: DAttnBuild):
    """Build variable-request cu96 split-key WMMA attention."""
    if key.tokens not in SUPPORTED_TOKENS or key.q_len != BLOCK:
        raise ValueError(
            f"gfx1250 D-attn supports rows {SUPPORTED_TOKENS}, q_len={BLOCK}; "
            f"got {(key.tokens, key.q_len)}"
        )
    if key.resident_blocks != SPLIT_TASKS:
        raise ValueError(
            "gfx1250 split D-attn supports only cu96_tp4; "
            f"got resident_blocks={key.resident_blocks}"
        )

    name = kernel_symbol(
        "muse_spark_gfx1250_dattn_wmma",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
        f=key.fault_site,
    )
    keyed = key_tuple(key, SOURCES)
    layout = dattn_scratch_layout(key.tokens, key.q_len)
    Lds = split_dattn_lds()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        active_guard = mailbox_guard(
            epoch,
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
        with guard.scope(active_guard):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            mb = Mailbox(step_tag(epoch, layer), active_guard)
            regions = {
                name[6:]: sreg(scratch, layout[name][0], name)
                for name in (
                    "dattn_stats",
                    "dattn_global",
                    "dattn_part",
                    "dattn_sflag",
                    "dattn_gflag",
                    "dattn_pflag",
                )
            }
            args = {
                "q": q,
                "out": out,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "req_to_token": req_to_token,
                "table_stride": table_stride,
                "table_bytes": table_bytes,
                "req_pool": req_pool,
                "rpi64": rpi64,
                "seq_lens": seq_lens,
                "lens64": lens64,
                "window": window,
                "scale_bits": scale_bits,
            }
            task = bid
            while task < (key.tokens // BLOCK) * SPLIT_TASKS:
                local_task = task % SPLIT_TASKS
                request = task // SPLIT_TASKS
                run_split_dattn(
                    {
                        "tid": tid,
                        "lane": tid % WAVE_SIZE,
                        "wave": tid // WAVE_SIZE,
                        "tile": request * ROW_TILES
                        + local_task // SPLIT_COUNT,
                        "split": local_task % SPLIT_COUNT,
                        "mb": mb,
                        "wave_stats": lds.wave_stats.ptr,
                        "row_stats": lds.row_stats.ptr,
                        "probabilities": lds.probabilities.ptr,
                        "pv": lds.pv.ptr,
                        **regions,
                        "args": args,
                    }
                )
                task = task + SPLIT_TASKS

    @flyc.jit
    def launch(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dattn(
            q,
            out,
            k_cache,
            v_cache,
            req_to_token,
            table_stride,
            table_bytes,
            req_pool,
            rpi64,
            seq_lens,
            lens64,
            window,
            scale_bits,
            scratch,
            epoch,
            layer,
        ).launch(
            grid=(SPLIT_TASKS,),
            block=(THREADS,),
            stream=stream,
        )

    return launch
