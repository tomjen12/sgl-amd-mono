# SPDX-License-Identifier: Apache-2.0
"""Standalone two-wave BM16xBN64 split K-attention compute prototype."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import bf16_round, butterfly, hw_exp2, rsrc
from sgl_amd_mono.muse_spark.gfx1250.config import (
    DRAFT_TOKENS,
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOG2E,
    Q_K,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import attention

THREADS = 64
WAVES = 2
ROWS = 16
BN = 64
WAVE_BN = BN // WAVES
DIM_TILES = HEAD_DIM // ROWS
ACC = 8
NEG = -1.0e30
_STREAM = fx.Stream(None)


def output_elements(splits: int) -> tuple[int, int]:
    """Return ``(partial, statistic)`` FP32 element counts for width eight."""
    tiles = LOCAL_KV_HEADS * (DRAFT_TOKENS // 2)
    return tiles * splits * ROWS * HEAD_DIM, tiles * splits * ROWS


def _part_index(tile, split, splits, row, dim):
    return ((tile * splits + split) * ROWS + row) * HEAD_DIM + dim


def _stat_index(tile, split, splits, row):
    return (tile * splits + split) * ROWS + row


def bn64_2warp_lds():
    """LDS contract reusable by a future persistent launcher."""

    @fx.struct
    class Lds:
        q: fx.Array[fx.Int32, ROWS * HEAD_DIM // 2, 16]
        wave_stats: fx.Array[fx.Float32, WAVES * ROWS * 2, 16]
        row_max: fx.Array[fx.Float32, ROWS, 16]
        pv: fx.Array[fx.Float32, WAVES * 32 * ACC, 16]
        kv: fx.Array[
            fx.Int32,
            WAVES * 2 * WAVE_BN * HEAD_DIM // 4,
            16,
        ]

    return Lds


@flyc.jit
def run_bn64_2warp_compute(c):
    """Compute one split partial without mailboxes, merge, norm, or output."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, splits, args = c["tile"], c["split"], c["splits"], c["args"]
    q_lds, wave_stats, row_max = c["q"], c["wave_stats"], c["row_max"]
    pv_lds, kv_lds = c["pv"], c["kv"]

    token0, _, kv_head = attention.fused_row_coords(
        tile, fx.Int32(0), DRAFT_TOKENS
    )
    request = token0 // DRAFT_TOKENS
    prefix = fx.Int32(
        bo.buffer_load(
            rsrc(args["prefix_lens"]), request, vec_width=1, dtype=T.i32
        )
    )
    keys = prefix + DRAFT_TOKENS
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)

    # Four cooperative rounds stage the 16x64 BF16 Q tile.
    for load_round in range_constexpr(4):
        item = tid * 4 + load_round * THREADS * 4
        row, dim = item // HEAD_DIM, item % HEAD_DIM
        token, query_head, _ = attention.fused_row_coords(
            tile, row, DRAFT_TOKENS
        )
        packed = fx.Vector(
            bo.buffer_load(
                rsrc(args["q"]),
                (token * Q_K + query_head * HEAD_DIM + dim) // 2,
                vec_width=2,
                dtype=T.i32,
            )
        )
        fx.ptr_store(packed, q_lds + item // 2)
    gpu.barrier()

    maximum = [fx.Float32(NEG) for _ in range(ACC)]
    denominator = [fx.Float32(0.0) for _ in range(ACC)]
    pv_acc = [
        fx.Vector.filled(ACC, 0.0, fx.Float32)
        for _ in range(DIM_TILES)
    ]

    key_origin = fx.Int32(0)
    key_limit = keys
    if args["window"] > 0:
        token_last, _, _ = attention.fused_row_coords(
            tile, fx.Int32(ROWS - 1), DRAFT_TOKENS
        )
        key_begin = fx.max(
            prefix + token0 % DRAFT_TOKENS + 1 - args["window"],
            fx.Int32(0),
        )
        key_limit = fx.min(
            prefix + token_last % DRAFT_TOKENS + 1,
            keys,
        )
        key_origin = (key_begin // BN) * BN

    lds_words = WAVE_BN * HEAD_DIM // 4
    wave_kv = kv_lds + wave * 2 * lds_words
    iteration = fx.Int32(0)
    while iteration * (splits * BN) < key_limit - key_origin:
        block0 = key_origin + (split + iteration * splits) * BN + wave * WAVE_BN
        pages = []
        for chunk in range_constexpr(WAVE_BN // ROWS):
            page_key = fx.min(block0 + chunk * ROWS, keys - 1)
            pages.append(attention._fused_page(args, request, page_key))

        for load_round in range_constexpr(
            WAVE_BN * HEAD_DIM // (32 * 8)
        ):
            local_byte = lane * 8 + load_round * 32 * 8
            local_key, key_dim = local_byte // HEAD_DIM, local_byte % HEAD_DIM
            logical_key = block0 + local_key
            safe_key = fx.min(logical_key, keys - 1)
            page = pages[0]
            if local_key >= ROWS:
                page = pages[1]
            slot = page * 16 + safe_key % 16
            cache_byte = (
                (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + key_dim
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
                    wave_kv + cache_index * lds_words + local_byte // 4,
                )
        gpu.barrier()

        scores_by_chunk = []
        live_by_chunk = []
        for chunk in range_constexpr(WAVE_BN // ROWS):
            key0 = block0 + chunk * ROWS
            scores_by_chunk.append(
                attention._fused_qk(
                    args,
                    tile,
                    DRAFT_TOKENS,
                    key0,
                    keys,
                    lane,
                    scale,
                    wave_kv,
                    chunk * ROWS,
                    q_lds,
                    True,
                )
            )
            lives = []
            key = key0 + lane % ROWS
            for element in range_constexpr(ACC):
                row = (lane // ROWS) * ACC + element
                token, _, _ = attention.fused_row_coords(
                    tile, row, DRAFT_TOKENS
                )
                end = prefix + token % DRAFT_TOKENS + 1
                begin = fx.Int32(0)
                if args["window"] > 0:
                    begin = fx.max(end - args["window"], fx.Int32(0))
                lives.append((key >= begin) & (key < end))
            live_by_chunk.append(lives)

        alphas = []
        probability_words = []
        for element in range_constexpr(ACC):
            local_max = fx.Float32(NEG)
            for chunk in range_constexpr(WAVE_BN // ROWS):
                local_max = fx.max(
                    local_max,
                    live_by_chunk[chunk][element].select(
                        fx.Float32(scores_by_chunk[chunk][element]),
                        fx.Float32(NEG),
                    ),
                )
            block_max = butterfly(local_max, (8, 4, 2, 1), op=fx.max)
            new_max = fx.max(maximum[element], block_max)
            alpha = hw_exp2((maximum[element] - new_max) * LOG2E)
            lane_sum = fx.Float32(0.0)
            probabilities = []
            for chunk in range_constexpr(WAVE_BN // ROWS):
                probability = live_by_chunk[chunk][element].select(
                    hw_exp2(
                        (
                            fx.Float32(scores_by_chunk[chunk][element])
                            - new_max
                        )
                        * LOG2E
                    ),
                    fx.Float32(0.0),
                )
                lane_sum = lane_sum + probability
                probabilities.append(fx.BFloat16(bf16_round(probability)))
            probability_words.append(
                fx.Vector.from_elements(
                    probabilities, fx.BFloat16
                ).bitcast(fx.Int32)[0]
            )
            denominator[element] = (
                denominator[element] * alpha
                + butterfly(lane_sum, (8, 4, 2, 1))
            )
            maximum[element] = new_max
            alphas.append(alpha)

        gathered = [
            fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)
            for word in attention._fused_transpose_probabilities(
                probability_words, lane
            )
        ]
        p_values = []
        for half in range_constexpr(2):
            p_values += [
                gathered[element][half] for element in range_constexpr(ACC)
            ]
        p_fragment = fx.Vector.from_elements(p_values, fx.BFloat16)
        alpha_vector = fx.Vector.from_elements(alphas, fx.Float32)
        kh = (lane // ROWS) * ACC
        for dim_tile in range_constexpr(DIM_TILES):
            v_values = []
            dim = dim_tile * ROWS + lane % ROWS
            for half in range_constexpr(2):
                for element in range_constexpr(ACC):
                    local_key = half * ROWS + kh + element
                    logical_key = block0 + local_key
                    v_values.append(
                        (logical_key < keys).select(
                            attention._fused_lds_fp8(
                                wave_kv + lds_words,
                                local_key * HEAD_DIM + dim,
                            ),
                            fx.Float32(0.0),
                        )
                    )
            pv_acc[dim_tile] = attention._fused_wmma(
                p_fragment,
                fx.Vector.from_elements(v_values, fx.Float32).to(
                    fx.BFloat16
                ),
                pv_acc[dim_tile] * alpha_vector,
            )
        iteration = iteration + 1

    for element in range_constexpr(ACC):
        if lane % ROWS == 0:
            row = (lane // ROWS) * ACC + element
            fx.ptr_store(
                maximum[element],
                wave_stats + (wave * ROWS + row) * 2,
            )
            fx.ptr_store(
                denominator[element],
                wave_stats + (wave * ROWS + row) * 2 + 1,
            )
    gpu.barrier()

    if tid < ROWS:
        split_max = fx.max(
            fx.ptr_load(wave_stats + tid * 2),
            fx.ptr_load(wave_stats + (ROWS + tid) * 2),
        )
        split_sum = fx.Float32(0.0)
        for source_wave in range_constexpr(WAVES):
            wave_max = fx.ptr_load(
                wave_stats + (source_wave * ROWS + tid) * 2
            )
            wave_sum = fx.ptr_load(
                wave_stats + (source_wave * ROWS + tid) * 2 + 1
            )
            split_sum = split_sum + wave_sum * hw_exp2(
                (wave_max - split_max) * LOG2E
            )
        fx.ptr_store(split_max, row_max + tid)
        index = _stat_index(tile, split, splits, tid)
        bo.buffer_store(split_max, rsrc(c["maximum"]), index)
        bo.buffer_store(split_sum, rsrc(c["denominator"]), index)
    gpu.barrier()

    for dim_tile in range_constexpr(DIM_TILES):
        fx.ptr_store(
            pv_acc[dim_tile],
            pv_lds + (wave * 32 + lane) * ACC,
        )
        gpu.barrier()
        for row_batch in range_constexpr(ROWS // (THREADS // ROWS)):
            item = tid + row_batch * THREADS
            row, col = item // ROWS, item % ROWS
            split_max = fx.ptr_load(row_max + row)
            result_lane = (row // ACC) * ROWS + col
            total = fx.Float32(0.0)
            for source_wave in range_constexpr(WAVES):
                wave_max = fx.ptr_load(
                    wave_stats + (source_wave * ROWS + row) * 2
                )
                total = total + fx.ptr_load(
                    pv_lds
                    + (source_wave * 32 + result_lane) * ACC
                    + row % ACC
                ) * hw_exp2((wave_max - split_max) * LOG2E)
            bo.buffer_store(
                total,
                rsrc(c["partial"]),
                _part_index(
                    tile,
                    split,
                    splits,
                    row,
                    dim_tile * ROWS + col,
                ),
            )
        gpu.barrier()


def build_bn64_2warp(splits: int):
    """Build a width-8 launcher with an exact ``16 * splits`` CTA grid."""
    if splits not in (6, 8):
        raise ValueError(f"prototype splits must be 6 or 8, got {splits}")
    tiles = LOCAL_KV_HEADS * (DRAFT_TOKENS // 2)
    tasks = tiles * splits
    name = kernel_symbol("gfx1250_kattn_bn64_2warp_proto", p=splits)
    Lds = bn64_2warp_lds()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        q: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        prefix_lens: Int64,
        window: Int32,
        scale_bits: Int32,
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
    ):
        tid = fx.thread_idx.x
        task = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        run_bn64_2warp_compute(
            {
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "tile": task // splits,
                "split": task % splits,
                "splits": splits,
                "q": lds.q.ptr,
                "wave_stats": lds.wave_stats.ptr,
                "row_max": lds.row_max.ptr,
                "pv": lds.pv.ptr,
                "kv": lds.kv.ptr,
                "partial": partial,
                "maximum": maximum,
                "denominator": denominator,
                "args": {
                    "q": q,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "table": table,
                    "table_stride": table_stride,
                    "prefix_lens": prefix_lens,
                    "window": window,
                    "scale_bits": scale_bits,
                },
            }
        )

    @flyc.jit
    def launch(
        q: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        prefix_lens: Int64,
        window: Int32,
        scale_bits: Int32,
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(
            q,
            k_cache,
            v_cache,
            table,
            table_stride,
            prefix_lens,
            window,
            scale_bits,
            partial,
            maximum,
            denominator,
        ).launch(grid=(tasks,), block=(THREADS,), stream=stream)

    return launch
