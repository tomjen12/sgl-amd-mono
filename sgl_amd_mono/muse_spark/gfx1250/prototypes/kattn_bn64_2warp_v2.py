# SPDX-License-Identifier: Apache-2.0
"""Pipelined standalone two-wave BM16xBN64 K-attention prototype.

This is deliberately separate from ``kattn_bn64_2warp`` (v1). Each wave owns
two K/V LDS slots. The next physical-paged FP8 tile is copied directly from
global memory into the alternate LDS slot while the current slot is consumed.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
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
from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_bn64_2warp import (
    ACC,
    BN,
    DIM_TILES,
    NEG,
    ROWS,
    THREADS,
    WAVES,
    WAVE_BN,
    _part_index,
    _stat_index,
    output_elements,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import attention

_STREAM = fx.Stream(None)
PAGE_KEYS = 16
DMA_BYTES = 16
PAGE_BYTES = PAGE_KEYS * HEAD_DIM
PAGE_WORDS = PAGE_BYTES // 4
CACHE_WORDS = WAVE_BN * HEAD_DIM // 4
SLOT_WORDS = 2 * CACHE_WORDS
RING_SLOTS = 2


def pipeline_metadata(splits: int) -> dict[str, object]:
    """Static metadata emitted by the focused launcher."""
    if splits not in (6, 8):
        raise ValueError(f"prototype splits must be 6 or 8, got {splits}")
    return {
        "version": "v2",
        "symbol": kernel_symbol("gfx1250_kattn_bn64_2warp_proto_v2", p=splits),
        "splits": splits,
        "ctas": LOCAL_KV_HEADS * (DRAFT_TOKENS // 2) * splits,
        "threads": THREADS,
        "waves": WAVES,
        "bm": ROWS,
        "bn": BN,
        "stages": RING_SLOTS,
        "q_staged_once": True,
        "kv_transfer": "global_load_async_to_lds_b128",
        "direct_to_lds": True,
        "async_copies_per_wave_tile": (
            2
            * (WAVE_BN // PAGE_KEYS)
            * (HEAD_DIM // (2 * DMA_BYTES))
        ),
        "wait": "s_wait_asynccnt 0",
        "lds_bytes": (
            ROWS * HEAD_DIM * 2
            + WAVES * ROWS * 2 * 4
            + ROWS * 4
            + WAVES * 32 * ACC * 4
            + WAVES * RING_SLOTS * SLOT_WORDS * 4
        ),
    }


def bn64_2warp_v2_lds():
    """LDS contract with two K/V slots per wave."""

    @fx.struct
    class Lds:
        q: fx.Array[fx.Int32, ROWS * HEAD_DIM // 2, 16]
        wave_stats: fx.Array[fx.Float32, WAVES * ROWS * 2, 16]
        row_max: fx.Array[fx.Float32, ROWS, 16]
        pv: fx.Array[fx.Float32, WAVES * 32 * ACC, 16]
        kv: fx.Array[
            fx.Int32,
            WAVES * RING_SLOTS * SLOT_WORDS,
            16,
        ]

    return Lds


def _global_byte_ptr(address):
    """Build the addrspace(1) byte pointer required by gfx1250 async loads."""
    return llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<1>"),
        fx.Int64(address).ir_value(),
    )


def _wait_async_copies():
    """Wait until this wave's direct global-to-LDS copies have landed."""
    rocdl.s_wait_asynccnt(0)


def _async_copy_16(global_address, lds_words):
    """Issue one native gfx1250 16-byte global-to-LDS copy."""
    rocdl.sched_barrier(0)
    global_ptr = _global_byte_ptr(global_address)
    lds_ptr = fx.recast_iter(fx.Int8, lds_words).llvm_ptr
    rocdl.sched_barrier(0)
    llvm.inline_asm(
        None,
        [],
        "s_wait_alu depctr_va_vdst(0)",
        "",
        has_side_effects=True,
    )
    fx.rocdl.global_load_async_to_lds_b128(global_ptr, lds_ptr, 0)
    rocdl.sched_barrier(0)


@flyc.jit
def _issue_kv_tile(args, request, block0, keys, kv_head, lane, slot):
    """Issue one wave's 32-key K/V tile directly into an LDS ring slot."""
    lane_key = lane % PAGE_KEYS
    lane_chunk = lane // PAGE_KEYS
    for page_index in range_constexpr(WAVE_BN // PAGE_KEYS):
        page_key = fx.min(block0 + page_index * PAGE_KEYS, keys - 1)
        physical_page = attention._fused_page(args, request, page_key)
        for issue in range_constexpr(HEAD_DIM // (2 * DMA_BYTES)):
            dim = issue * 2 * DMA_BYTES + lane_chunk * DMA_BYTES
            source_byte = (
                (
                    (physical_page * PAGE_KEYS + lane_key) * LOCAL_KV_HEADS
                    + kv_head
                )
                * HEAD_DIM
                + dim
            )
            destination = (
                slot
                + page_index * PAGE_WORDS
                + issue * 512 // 4
                + lane * DMA_BYTES // 4
            )
            _async_copy_16(
                fx.Int64(args["k_cache"]) + fx.Int64(source_byte),
                destination,
            )
            _async_copy_16(
                fx.Int64(args["v_cache"]) + fx.Int64(source_byte),
                destination + CACHE_WORDS,
            )


def _dma_byte(local_key, dim):
    """Map key-major coordinates into the DMA's chunk-major LDS layout."""
    page = local_key // PAGE_KEYS
    key = local_key % PAGE_KEYS
    chunk = dim // DMA_BYTES
    return page * PAGE_BYTES + chunk * (PAGE_KEYS * DMA_BYTES) + key * DMA_BYTES + dim % DMA_BYTES


def _dma_fp8(slot, cache_words, local_key, dim):
    byte = _dma_byte(local_key, dim)
    return attention._fused_lds_fp8(slot + cache_words, byte)


def _dma_k_fragment(slot, local_key, k0, lane):
    kh = (lane // ROWS) * ACC
    values = []
    for half in range_constexpr(2):
        dim = k0 + half * ROWS + kh
        byte = _dma_byte(local_key, dim)
        packed = attention._decode_fp8x8(
            fx.Vector(
                fx.ptr_load(
                    slot + byte // 4,
                    result_type=T.vec(2, T.i32),
                )
            )
        )
        values += [packed[element] for element in range_constexpr(ACC)]
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)


def _dma_qk(tile, key0, keys, lane, scale, slot, local_key0, q_lds):
    del keys
    local_key = local_key0 + lane % ROWS
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for step in range_constexpr(HEAD_DIM // 32):
        k0 = step * 32
        acc = attention._fused_wmma(
            attention._fused_lds_q_fragment(q_lds, k0, lane),
            _dma_k_fragment(slot, local_key, k0, lane),
            acc,
        )
    return acc * scale


@flyc.jit
def run_bn64_2warp_v2_compute(c):
    """Compute one split partial with a two-stage K/V software pipeline."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, splits, args = c["tile"], c["split"], c["splits"], c["args"]
    q_lds, wave_stats, row_max = c["q"], c["wave_stats"], c["row_max"]
    pv_lds, kv_ring = c["pv"], c["kv"]

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

    # Q is staged once and retained for the complete split.
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

    wave_ring = kv_ring + wave * RING_SLOTS * SLOT_WORDS
    iteration = fx.Int32(0)
    live = iteration * (splits * BN) < key_limit - key_origin
    if live:
        first_block = key_origin + split * BN + wave * WAVE_BN
        _issue_kv_tile(
            args,
            request,
            first_block,
            keys,
            kv_head,
            lane,
            wave_ring,
        )
        _wait_async_copies()
        gpu.barrier()

    while iteration * (splits * BN) < key_limit - key_origin:
        block0 = key_origin + (split + iteration * splits) * BN + wave * WAVE_BN
        slot_index = iteration % RING_SLOTS
        slot = wave_ring + slot_index * SLOT_WORDS
        next_iteration = iteration + 1
        next_block = (
            key_origin
            + (split + next_iteration * splits) * BN
            + wave * WAVE_BN
        )
        # The tail intentionally repeats a clamped physical-page prefetch.  A
        # fixed issue count avoids divergent pipeline state and matches the
        # ring discipline used by the gfx950 attention implementation.
        _issue_kv_tile(
            args,
            request,
            next_block,
            keys,
            kv_head,
            lane,
            wave_ring + (next_iteration % RING_SLOTS) * SLOT_WORDS,
        )

        scores_by_chunk = []
        live_by_chunk = []
        for chunk in range_constexpr(WAVE_BN // ROWS):
            key0 = block0 + chunk * ROWS
            scores_by_chunk.append(
                _dma_qk(
                    tile,
                    key0,
                    keys,
                    lane,
                    scale,
                    slot,
                    chunk * ROWS,
                    q_lds,
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
                            _dma_fp8(
                                slot,
                                CACHE_WORDS,
                                local_key,
                                dim,
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

        # Copies into the alternate slot were issued before QK/softmax/PV.
        # Wait before that slot becomes current; the CTA barrier also prevents
        # either wave from reusing a slot while another lane still reads it.
        _wait_async_copies()
        gpu.barrier()
        iteration = next_iteration

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


def build_bn64_2warp_v2(splits: int):
    """Build the isolated v2 launcher; production defaults are untouched."""
    if splits not in (6, 8):
        raise ValueError(f"prototype splits must be 6 or 8, got {splits}")
    tiles = LOCAL_KV_HEADS * (DRAFT_TOKENS // 2)
    tasks = tiles * splits
    name = kernel_symbol("gfx1250_kattn_bn64_2warp_proto_v2", p=splits)
    Lds = bn64_2warp_v2_lds()

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
        run_bn64_2warp_v2_compute(
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
