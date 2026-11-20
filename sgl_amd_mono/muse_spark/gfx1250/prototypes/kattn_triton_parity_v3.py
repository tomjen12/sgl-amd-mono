# SPDX-License-Identifier: Apache-2.0
"""CTA-layout gfx1250 BM16xBN64 split K-attention prototype.

This prototype follows the compiled Triton ``_swa_verify_partials`` ownership
change: QK waves partition query rows, while PV waves partition output
head-dimension.  The only layout exchange is a BF16 16x64 probability tile;
the 16x64 FP32 output accumulator remains register resident.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import bf16_round, hw_exp2, lane_gather, rsrc
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
    NEG,
    ROWS,
    THREADS,
    WAVES,
    _part_index,
    _stat_index,
    output_elements,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import attention

_STREAM = fx.Stream(None)
RING_SLOTS = 2
PAGE_KEYS = 16
DMA_BYTES = 16
KV_PAD_GROUP_BYTES = 256
KV_PAD_BYTES = 16
KV_LOGICAL_SLOT_BYTES = BN * HEAD_DIM
KV_SLOT_BYTES = (
    KV_LOGICAL_SLOT_BYTES
    + KV_LOGICAL_SLOT_BYTES // KV_PAD_GROUP_BYTES * KV_PAD_BYTES
)
KV_SLOT_WORDS = KV_SLOT_BYTES // 4
LEGACY_K_ROW_BYTES = HEAD_DIM + 16
LEGACY_V_ROW_BYTES = HEAD_DIM + 32
LEGACY_K_SLOT_WORDS = BN * LEGACY_K_ROW_BYTES // 4
LEGACY_V_SLOT_WORDS = BN * LEGACY_V_ROW_BYTES // 4
Q_BYTES = ROWS * HEAD_DIM * 2
P_ELEMENTS = ROWS * BN
PAGE_SCRATCH_ELEMENTS = BN
DIM_TILES_PER_WAVE = HEAD_DIM // (ROWS * WAVES)


def q_lds_byte(row: int, dim: int) -> int:
    """AITER QManager16bV1's 4x4 b128 XOR swizzle."""
    tile = dim // 32
    col = dim % 32
    chunk = col // 8
    return tile * 1024 + row * 64 + (chunk ^ (row // 4)) * 16


def kv_lds_byte(row: int, dim: int) -> int:
    """Triton padded_shared<[256:+16]> physical byte offset."""
    # HEAD_DIM is 64, so each 256-byte pad boundary is exactly four rows.
    # Keeping the pad term row-only avoids a dependent add+shift in each
    # async destination and fragment address.
    return (
        row * HEAD_DIM
        + (row // (KV_PAD_GROUP_BYTES // HEAD_DIM)) * KV_PAD_BYTES
        + dim
    )


def k_lds_byte(row: int, dim: int, compact: bool = True) -> int:
    if compact:
        return kv_lds_byte(row, dim)
    return row * LEGACY_K_ROW_BYTES + dim


def v_lds_byte(row: int, dim: int, compact: bool = True) -> int:
    if compact:
        return kv_lds_byte(row, dim)
    return row * LEGACY_V_ROW_BYTES + dim


def qk_owner(row: int) -> int:
    return row // (ROWS // WAVES)


def pv_dimensions(wave: int) -> range:
    return range(wave * (HEAD_DIM // WAVES), (wave + 1) * (HEAD_DIM // WAVES))


def probability_fragment_keys(lane: int, contraction: int) -> list[int]:
    """Keys held by one lane for the PV WMMA A fragment."""
    lane_half = lane // ROWS
    return [
        contraction * 32 + half * ROWS + lane_half * ACC + element
        for half in range(2)
        for element in range(ACC)
    ]


def contiguous_split_tiles(
    low: int, high: int, split: int, splits: int
) -> tuple[int, int]:
    """Triton's contiguous ceil-div tile assignment, in 64-key tile units."""
    tile_start = low // BN
    tile_stop = (high + BN - 1) // BN
    tiles = max(tile_stop - tile_start, 0)
    tiles_per_split = (tiles + splits - 1) // splits
    begin = tile_start + split * tiles_per_split
    return begin, min(tile_stop, begin + tiles_per_split)


def pipeline_metadata(splits: int) -> dict[str, object]:
    if splits not in (6, 8):
        raise ValueError(f"prototype splits must be 6 or 8, got {splits}")
    return {
        "version": "v3",
        "symbol": kernel_symbol("gfx1250_kattn_triton_parity_v3", p=splits),
        "splits": splits,
        "threads": THREADS,
        "waves": WAVES,
        "bm": ROWS,
        "bn": BN,
        "k": HEAD_DIM,
        "stages": RING_SLOTS,
        "split_assignment": "contiguous",
        "q_layout": "qmanager16bv1_xor_swizzle",
        "qk_warp_partition": "rows",
        "pv_warp_partition": "head_dim",
        "kv_transfer": "global_load_async_to_lds_b128",
        "kv_lds_layout": "padded_shared_[256:+16]",
        "kv_slot_bytes": KV_SLOT_BYTES,
        "probability_exchange_bytes": P_ELEMENTS * 2,
        "fp32_pv_spill_bytes": 0,
        "output_accumulator": "register_resident_16x64_fp32",
        "lds_bytes": (
            Q_BYTES
            + P_ELEMENTS * 2
            + ROWS * 3 * 4
            + RING_SLOTS * KV_SLOT_BYTES * 2
        ),
    }


def triton_parity_v3_lds(*, compact_kv: bool = True):
    k_slot_words = KV_SLOT_WORDS if compact_kv else LEGACY_K_SLOT_WORDS
    v_slot_words = KV_SLOT_WORDS if compact_kv else LEGACY_V_SLOT_WORDS

    @fx.struct
    class Lds:
        q: fx.Array[fx.Int32, Q_BYTES // 4, 16]
        probabilities: fx.Array[fx.BFloat16, P_ELEMENTS, 16]
        maximum: fx.Array[fx.Float32, ROWS, 16]
        denominator: fx.Array[fx.Float32, ROWS, 16]
        alpha: fx.Array[fx.Float32, ROWS, 16]
        page_indices: fx.Array[fx.Int32, PAGE_SCRATCH_ELEMENTS, 16]
        k: fx.Array[fx.Int32, RING_SLOTS * k_slot_words, 16]
        v: fx.Array[fx.Int32, RING_SLOTS * v_slot_words, 16]

    return Lds


def _global_byte_ptr(address):
    return llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<1>"), fx.Int64(address).ir_value()
    )


def _async_copy_16(global_address, lds_bytes):
    global_ptr = _global_byte_ptr(global_address)
    lds_ptr = fx.recast_iter(fx.Int8, lds_bytes).llvm_ptr
    llvm.inline_asm(
        None,
        [],
        "s_wait_alu depctr_va_vdst(0)",
        "",
        has_side_effects=True,
    )
    fx.rocdl.global_load_async_to_lds_b128(global_ptr, lds_ptr, 0)


def _ds_load_4addr_b32(page_indices, local_key):
    """Load four page indices 16 elements apart with two DS instructions."""
    address = fx.Int32(fx.ptrtoint(page_indices + local_key))
    result_type = ir.Type.parse(
        "!llvm.struct<(vector<2xi32>, vector<2xi32>)>"
    )
    result = llvm.inline_asm(
        result_type,
        [address.ir_value()],
        "\n".join(
            [
                "ds_load_2addr_b32 $0, $2 offset1:16",
                "ds_load_2addr_b32 $1, $2 offset0:32 offset1:48",
                "s_wait_dscnt 0",
            ]
        ),
        "=v,=v,v,~{memory}",
        has_side_effects=True,
    )
    pairs = [
        fx.Vector(
            llvm.extractvalue(T.vec(2, T.i32), result, [pair])
        )
        for pair in range_constexpr(2)
    ]
    return pairs[0].shuffle(pairs[1], list(range(4)))


@flyc.jit
def _issue_kv_tile(
    args,
    request,
    block0,
    keys,
    kv_head,
    tid,
    k_slot,
    v_slot,
    dedup_page,
    compact_kv,
    page_indices,
    stage_pages,
    paired_page_load,
):
    """Cooperatively fill one padded CTA-wide 64x64 K tile and V tile."""
    if const_expr(stage_pages):
        page_key = fx.min(block0 + tid, keys - 1)
        fx.ptr_store(
            attention._fused_page(args, request, page_key),
            page_indices + tid,
        )
        gpu.barrier()
    page_pairs = []
    if const_expr(stage_pages and paired_page_load):
        lane_key = tid // (HEAD_DIM // DMA_BYTES)
        page_pairs = _ds_load_4addr_b32(page_indices, lane_key)
    for issue in range_constexpr(BN * HEAD_DIM // (THREADS * DMA_BYTES)):
        chunk = tid + issue * THREADS
        local_key = chunk // (HEAD_DIM // DMA_BYTES)
        dim = (chunk % (HEAD_DIM // DMA_BYTES)) * DMA_BYTES
        logical_key = block0 + local_key
        safe_key = fx.min(logical_key, keys - 1)
        if const_expr(stage_pages):
            if const_expr(paired_page_load):
                physical_page = fx.Int32(
                    page_pairs[issue]
                )
            else:
                physical_page = fx.Int32(
                    fx.ptr_load(page_indices + local_key)
                )
        elif const_expr(dedup_page):
            physical_page = fx.Int32(0)
            if tid % 4 == 0:
                physical_page = attention._fused_page(
                    args, request, safe_key
                )
            physical_page = lane_gather(
                physical_page, (tid % 32) & fx.Int32(-4)
            )
        else:
            physical_page = attention._fused_page(
                args, request, safe_key
            )
        source_byte = (
            (
                (physical_page * PAGE_KEYS + safe_key % PAGE_KEYS)
                * LOCAL_KV_HEADS
                + kv_head
            )
            * HEAD_DIM
            + dim
        )
        _async_copy_16(
            fx.Int64(args["k_cache"]) + fx.Int64(source_byte),
            k_slot + k_lds_byte(local_key, dim, compact_kv) // 4,
        )
        _async_copy_16(
            fx.Int64(args["v_cache"]) + fx.Int64(source_byte),
            v_slot + v_lds_byte(local_key, dim, compact_kv) // 4,
        )


def _fp8x8_to_bf16(words):
    return fx.Vector(
        _rocdl.cvt_scale_pk8_bf16_fp8(
            T.vec(8, T.bf16),
            words.ir_value(),
            fx.Int32(127).ir_value(),
            0,
        )
    )


def _q_fragment(q_words, k0, lane):
    row = lane % ROWS
    kh = (lane // ROWS) * ACC
    halves = []
    q_bytes = fx.recast_iter(fx.Int8, q_words)
    for half in range_constexpr(2):
        dim = k0 + half * ROWS + kh
        halves.append(
            fx.Vector(
                fx.ptr_load(
                    q_bytes + q_lds_byte(row, dim),
                    result_type=T.vec(4, T.i32),
                )
            ).bitcast(fx.BFloat16)
        )
    return halves[0].shuffle(halves[1], list(range(2 * ACC)))


def _k_fragment(k_slot, local_key, k0, lane, compact_kv):
    kh = (lane // ROWS) * ACC
    halves = []
    for half in range_constexpr(2):
        dim = k0 + half * ROWS + kh
        words = fx.Vector(
            fx.ptr_load(
                k_slot + k_lds_byte(local_key, dim, compact_kv) // 4,
                result_type=T.vec(2, T.i32),
            )
        )
        halves.append(_fp8x8_to_bf16(words))
    return halves[0].shuffle(halves[1], list(range(2 * ACC)))


def _p_fragment(probabilities, contraction, lane):
    row = lane % ROWS
    kh = (lane // ROWS) * ACC
    halves = []
    for half in range_constexpr(2):
        key = contraction * 32 + half * ROWS + kh
        halves.append(
            fx.Vector(
                fx.ptr_load(
                    probabilities + row * BN + key,
                    result_type=T.vec(8, T.bf16),
                )
            )
        )
    return halves[0].shuffle(halves[1], list(range(2 * ACC)))


def _v_fragment(v_slot, contraction, dim_tile, lane, compact_kv):
    """AITER VManager16bV1's FP8 ``ds_load_tr8_b64`` plan."""
    tr_row = (
        (lane // ROWS) * 8
        + lane % 4
        + ((lane % ROWS) // 8) * 4
    )
    tr_col = (((lane % ROWS) // 4) % 2) * 8
    halves = []
    v_bytes = fx.recast_iter(fx.Int8, v_slot)
    for half in range_constexpr(2):
        byte = v_lds_byte(
            contraction * 32 + half * ROWS + tr_row,
            dim_tile * ROWS + tr_col,
            compact_kv,
        )
        raw = fx.Vector(
            _rocdl.ds_load_tr8_b64(
                T.vec(2, T.i32), fx.to_llvm_ptr(v_bytes + byte)
            )
        )
        halves.append(_fp8x8_to_bf16(raw))
    return halves[0].shuffle(halves[1], list(range(2 * ACC)))


def _wmma(a, b, acc):
    return fx.Vector(
        rocdl.wmma_f32_16x16x32_bf16(
            T.vec(8, T.f32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(acc),
            reuseA=False,
            reuseB=False,
        ).result
    )


def _wmma_pair(a0, b0, a1, b1, acc):
    """Keep one K64 two-WMMA dependency chain opaque to FlyDSL lowering."""
    return fx.Vector(
        llvm.inline_asm(
            T.vec(8, T.f32),
            [
                fx.as_ir_value(a0),
                fx.as_ir_value(b0),
                fx.as_ir_value(a1),
                fx.as_ir_value(b1),
                fx.as_ir_value(acc),
            ],
            "\n".join(
                [
                    "v_wmma_f32_16x16x32_bf16 $0, $1, $2, $5",
                    "v_wmma_f32_16x16x32_bf16 $0, $3, $4, $0",
                ]
            ),
            "=v,v,v,v,v,v",
            has_side_effects=True,
        )
    )


def _tree_add(values):
    level = values
    while len(level) > 1:
        level = [
            level[index] + level[index + 1]
            for index in range(0, len(level), 2)
        ]
    return level[0]


@flyc.jit
def run_triton_parity_v3_compute(c):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    tile, split, splits, args = c["tile"], c["split"], c["splits"], c["args"]
    q_lds, probabilities = c["q"], c["probabilities"]
    maximum_lds, denominator_lds, alpha_lds = (
        c["maximum"],
        c["denominator"],
        c["alpha"],
    )
    k_ring, v_ring = c["k"], c["v"]
    hoist_q = c.get("hoist_q", False)
    register_probability = c.get("register_probability", False)
    dedup_page = c.get("dedup_page", False)
    compact_kv = c.get("compact_kv", True)
    vector_epilogue = c.get("vector_epilogue", False)
    stage_pages = c.get("stage_pages", False)
    interleave_pv_pack = c.get("interleave_pv_pack", False)
    paired_page_load = c.get("paired_page_load", False)
    inline_wmma_pair = c.get("inline_wmma_pair", False)
    interleaved_stats = c.get("interleaved_stats", False)
    k_slot_words = (
        KV_SLOT_WORDS if compact_kv else LEGACY_K_SLOT_WORDS
    )
    v_slot_words = (
        KV_SLOT_WORDS if compact_kv else LEGACY_V_SLOT_WORDS
    )

    token0, _, kv_head = attention.fused_row_coords(
        tile, fx.Int32(0), DRAFT_TOKENS
    )
    request = token0 // DRAFT_TOKENS
    prefix = attention._seq_len(args, request)
    keys = prefix + DRAFT_TOKENS
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)

    # Stage Q once in the same 4x4 b128 XOR swizzle used by AITER QManager16bV1.
    q_bytes = fx.recast_iter(fx.Int8, q_lds)
    for load_round in range_constexpr(Q_BYTES // (THREADS * 16)):
        chunk = tid + load_round * THREADS
        row = chunk // (HEAD_DIM // 8)
        dim = (chunk % (HEAD_DIM // 8)) * 8
        token, query_head, _ = attention.fused_row_coords(
            tile, row, DRAFT_TOKENS
        )
        packed = fx.Vector(
            bo.buffer_load(
                rsrc(args["q"]),
                (token * Q_K + query_head * HEAD_DIM + dim) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        fx.ptr_store(packed, q_bytes + q_lds_byte(row, dim))
    gpu.barrier()
    q_fragments = []

    maximum = fx.Float32(NEG)
    denominator = fx.Float32(0.0)
    output = [
        fx.Vector.filled(ACC, 0.0, fx.Float32)
        for _ in range(DIM_TILES_PER_WAVE)
    ]

    token_last, _, _ = attention.fused_row_coords(
        tile, fx.Int32(ROWS - 1), DRAFT_TOKENS
    )
    low = fx.Int32(0)
    high = keys
    if args["window"] > 0:
        low = fx.max(
            prefix + token0 % DRAFT_TOKENS + 1 - args["window"],
            fx.Int32(0),
        )
        high = fx.min(
            prefix + token_last % DRAFT_TOKENS + 1,
            keys,
        )
    tile_start = low // BN
    tile_stop = (high + BN - 1) // BN
    tile_count = fx.max(tile_stop - tile_start, fx.Int32(0))
    tiles_per_split = (tile_count + splits - 1) // splits
    begin = tile_start + split * tiles_per_split
    end_tile = fx.min(tile_stop, begin + tiles_per_split)

    block_tile = begin
    if block_tile < end_tile:
        _issue_kv_tile(
            args,
            request,
            block_tile * BN,
            keys,
            kv_head,
            tid,
            k_ring + (block_tile % RING_SLOTS) * k_slot_words,
            v_ring + (block_tile % RING_SLOTS) * v_slot_words,
            dedup_page,
            compact_kv,
            c["page_indices"],
            stage_pages,
            paired_page_load,
        )
        rocdl.s_wait_asynccnt(0)
        gpu.barrier()
    if const_expr(hoist_q):
        q_fragments = [
            _q_fragment(q_lds, step * 32, lane)
            for step in range_constexpr(HEAD_DIM // 32)
        ]

    while block_tile < end_tile:
        slot_index = block_tile % RING_SLOTS
        k_slot = k_ring + slot_index * k_slot_words
        v_slot = v_ring + slot_index * v_slot_words
        next_tile = block_tile + 1
        if next_tile < end_tile:
            _issue_kv_tile(
                args,
                request,
                next_tile * BN,
                keys,
                kv_head,
                tid,
                k_ring + (next_tile % RING_SLOTS) * k_slot_words,
                v_ring + (next_tile % RING_SLOTS) * v_slot_words,
                dedup_page,
                compact_kv,
                c["page_indices"],
                stage_pages,
                paired_page_load,
            )

        block0 = block_tile * BN
        scores = []
        lives = []
        row = lane % ROWS
        row_owner = row // (ROWS // WAVES)
        token, _, _ = attention.fused_row_coords(
            tile, row, DRAFT_TOKENS
        )
        row_end = prefix + token % DRAFT_TOKENS + 1
        row_begin = fx.Int32(0)
        if args["window"] > 0:
            row_begin = fx.max(
                row_end - args["window"], fx.Int32(0)
            )
        for chunk in range_constexpr(BN // ROWS):
            acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
            local_key = chunk * ROWS + lane % ROWS
            if const_expr(inline_wmma_pair):
                q0 = (
                    q_fragments[0]
                    if hoist_q
                    else _q_fragment(q_lds, 0, lane)
                )
                q1 = (
                    q_fragments[1]
                    if hoist_q
                    else _q_fragment(q_lds, 32, lane)
                )
                acc = _wmma_pair(
                    _k_fragment(k_slot, local_key, 0, lane, compact_kv),
                    q0,
                    _k_fragment(k_slot, local_key, 32, lane, compact_kv),
                    q1,
                    acc,
                )
            else:
                for step in range_constexpr(HEAD_DIM // 32):
                    if const_expr(hoist_q):
                        q_fragment = q_fragments[step]
                    else:
                        q_fragment = _q_fragment(q_lds, step * 32, lane)
                    acc = _wmma(
                        _k_fragment(
                            k_slot, local_key, step * 32, lane, compact_kv
                        ),
                        q_fragment,
                        acc,
                    )
            scores.append(acc * scale)
            chunk_lives = []
            key0 = (
                block0
                + chunk * ROWS
                + (lane // ROWS) * ACC
            )
            for element in range_constexpr(ACC):
                key = key0 + element
                chunk_lives.append(
                    (key >= row_begin) & (key < row_end)
                )
            lives.append(chunk_lives)

        local_max = fx.Float32(NEG)
        for chunk in range_constexpr(BN // ROWS):
            for element in range_constexpr(ACC):
                local_max = fx.max(
                    local_max,
                    lives[chunk][element].select(
                        fx.Float32(scores[chunk][element]),
                        fx.Float32(NEG),
                    ),
                )
        block_max = fx.max(
            local_max,
            local_max.shuffle_xor(fx.Int32(16), fx.Int32(32)),
        )
        new_max = fx.max(maximum, block_max)
        alpha = hw_exp2((maximum - new_max) * LOG2E)
        probability_values = []
        probability_flat = []
        lane_sum = fx.Float32(0.0)
        for chunk in range_constexpr(BN // ROWS):
            chunk_values = []
            for element in range_constexpr(ACC):
                probability = lives[chunk][element].select(
                    hw_exp2(
                        (
                            fx.Float32(scores[chunk][element])
                            - new_max
                        )
                        * LOG2E
                    ),
                    fx.Float32(0.0),
                )
                chunk_values.append(probability)
                probability_flat.append(probability)
                if not const_expr(register_probability):
                    lane_sum = lane_sum + probability
            probability_values.append(chunk_values)
        if const_expr(register_probability):
            lane_sum = _tree_add(probability_flat)
        block_sum = lane_sum + lane_sum.shuffle_xor(
            fx.Int32(16), fx.Int32(32)
        )
        maximum = new_max
        denominator = denominator * alpha + block_sum

        register_p = []
        if const_expr(
            register_probability and not interleave_pv_pack
        ):
            for contraction in range_constexpr(BN // 32):
                values = []
                for half in range_constexpr(2):
                    chunk = contraction * 2 + half
                    values += [
                        fx.BFloat16(
                            bf16_round(probability_values[chunk][element])
                        )
                        for element in range_constexpr(ACC)
                    ]
                register_p.append(
                    fx.Vector.from_elements(values, fx.BFloat16)
                )
        else:
            if row_owner == wave:
                for chunk in range_constexpr(BN // ROWS):
                    key0 = chunk * ROWS + (lane // ROWS) * ACC
                    for element in range_constexpr(ACC):
                        fx.ptr_store(
                            fx.BFloat16(
                                bf16_round(
                                    probability_values[chunk][element]
                                )
                            ),
                            probabilities
                            + row * BN
                            + key0
                            + element,
                        )
                if lane < ROWS:
                    fx.ptr_store(alpha, alpha_lds + row)
            gpu.barrier()

        if const_expr(register_probability):
            alpha_value = alpha
        else:
            alpha_value = fx.Float32(
                fx.ptr_load(alpha_lds + lane % ROWS)
            )
        alpha_vector = fx.Vector.filled(ACC, alpha_value, fx.Float32)
        if const_expr(register_probability and interleave_pv_pack):
            for local_dim_tile in range_constexpr(DIM_TILES_PER_WAVE):
                output[local_dim_tile] = (
                    output[local_dim_tile] * alpha_vector
                )
            for contraction in range_constexpr(BN // 32):
                values = []
                for half in range_constexpr(2):
                    chunk = contraction * 2 + half
                    values += [
                        fx.BFloat16(
                            bf16_round(probability_values[chunk][element])
                        )
                        for element in range_constexpr(ACC)
                    ]
                p_fragment = fx.Vector.from_elements(
                    values, fx.BFloat16
                )
                for local_dim_tile in range_constexpr(
                    DIM_TILES_PER_WAVE
                ):
                    dim_tile = (
                        wave * DIM_TILES_PER_WAVE + local_dim_tile
                    )
                    output[local_dim_tile] = _wmma(
                        _v_fragment(
                            v_slot,
                            contraction,
                            dim_tile,
                            lane,
                            compact_kv,
                        ),
                        p_fragment,
                        output[local_dim_tile],
                    )
        else:
            for local_dim_tile in range_constexpr(DIM_TILES_PER_WAVE):
                dim_tile = wave * DIM_TILES_PER_WAVE + local_dim_tile
                acc = output[local_dim_tile] * alpha_vector
                if const_expr(inline_wmma_pair):
                    if const_expr(register_probability):
                        p0, p1 = register_p[0], register_p[1]
                    else:
                        p0 = _p_fragment(probabilities, 0, lane)
                        p1 = _p_fragment(probabilities, 1, lane)
                    acc = _wmma_pair(
                        _v_fragment(
                            v_slot, 0, dim_tile, lane, compact_kv
                        ),
                        p0,
                        _v_fragment(
                            v_slot, 1, dim_tile, lane, compact_kv
                        ),
                        p1,
                        acc,
                    )
                else:
                    for contraction in range_constexpr(BN // 32):
                        if const_expr(register_probability):
                            p_fragment = register_p[contraction]
                        else:
                            p_fragment = _p_fragment(
                                probabilities, contraction, lane
                            )
                        acc = _wmma(
                            _v_fragment(
                                v_slot,
                                contraction,
                                dim_tile,
                                lane,
                                compact_kv,
                            ),
                            p_fragment,
                            acc,
                        )
                output[local_dim_tile] = acc
        gpu.barrier()

        if next_tile < end_tile:
            rocdl.s_wait_asynccnt(0)
            gpu.barrier()
        block_tile = next_tile

    row = lane % ROWS
    if row // (ROWS // WAVES) == wave and lane < ROWS:
        fx.ptr_store(maximum, maximum_lds + row)
        fx.ptr_store(denominator, denominator_lds + row)
        if const_expr(interleaved_stats):
            bo.buffer_store(
                fx.Vector.from_elements(
                    [maximum, denominator], fx.Float32
                ),
                rsrc(c["maximum_out"]),
                _stat_index(tile, split, splits, row) * 2,
            )
        else:
            bo.buffer_store(
                maximum,
                rsrc(c["maximum_out"]),
                _stat_index(tile, split, splits, row),
            )
            bo.buffer_store(
                denominator,
                rsrc(c["denominator_out"]),
                _stat_index(tile, split, splits, row),
            )

    for local_dim_tile in range_constexpr(DIM_TILES_PER_WAVE):
        dim_base = (
            wave * (HEAD_DIM // WAVES)
            + local_dim_tile * ROWS
            + (lane // ROWS) * ACC
        )
        row = lane % ROWS
        if const_expr(vector_epilogue):
            for half in range_constexpr(2):
                values = fx.Vector.from_elements(
                    [
                        fx.Float32(
                            output[local_dim_tile][half * 4 + element]
                        )
                        for element in range_constexpr(4)
                    ],
                    fx.Float32,
                )
                bo.buffer_store(
                    values,
                    rsrc(c["partial"]),
                    _part_index(
                        tile,
                        split,
                        splits,
                        row,
                        dim_base + half * 4,
                    ),
                )
        else:
            for element in range_constexpr(ACC):
                bo.buffer_store(
                    fx.Float32(output[local_dim_tile][element]),
                    rsrc(c["partial"]),
                    _part_index(
                        tile,
                        split,
                        splits,
                        row,
                        dim_base + element,
                    ),
                )


def build_triton_parity_v3(
    splits: int,
    *,
    hoist_q: bool = False,
    register_probability: bool = False,
    dedup_page: bool = False,
    compact_kv: bool = True,
    vector_epilogue: bool = False,
    stage_pages: bool = False,
    interleave_pv_pack: bool = False,
    paired_page_load: bool = False,
    baseline_llvm_attrs: bool = False,
    inline_wmma_pair: bool = False,
):
    """Build the isolated v3 helper; production launch paths remain untouched."""
    if splits not in (6, 8):
        raise ValueError(f"prototype splits must be 6 or 8, got {splits}")
    tasks = LOCAL_KV_HEADS * (DRAFT_TOKENS // 2) * splits
    variant = ""
    if hoist_q:
        variant += "hq"
    if register_probability:
        variant += "rp"
    if dedup_page:
        variant += "dp"
    if not compact_kv:
        variant += "lk"
    if vector_epilogue:
        variant += "ve"
    if stage_pages:
        variant += "sp"
    if interleave_pv_pack:
        variant += "ip"
    if paired_page_load:
        variant += "pp"
    if baseline_llvm_attrs:
        variant += "ba"
    if inline_wmma_pair:
        variant += "iw"
    stem = "gfx1250_kattn_triton_parity_v3"
    if variant:
        stem += f"_{variant}"
    name = kernel_symbol(stem, p=splits)
    Lds = triton_parity_v3_lds(compact_kv=compact_kv)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        q: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
    ):
        tid = fx.thread_idx.x
        task = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        run_triton_parity_v3_compute(
            {
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "tile": task // splits,
                "split": task % splits,
                "splits": splits,
                "q": lds.q.ptr,
                "probabilities": lds.probabilities.ptr,
                "maximum": lds.maximum.ptr,
                "denominator": lds.denominator.ptr,
                "alpha": lds.alpha.ptr,
                "page_indices": lds.page_indices.ptr,
                "k": lds.k.ptr,
                "v": lds.v.ptr,
                "hoist_q": hoist_q,
                "register_probability": register_probability,
                "dedup_page": dedup_page,
                "compact_kv": compact_kv,
                "vector_epilogue": vector_epilogue,
                "stage_pages": stage_pages,
                "interleave_pv_pack": interleave_pv_pack,
                "paired_page_load": paired_page_load,
                "inline_wmma_pair": inline_wmma_pair,
                "partial": partial,
                "maximum_out": maximum,
                "denominator_out": denominator,
                "args": {
                    "q": q,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "table": table,
                    "table_stride": table_stride,
                    "seq_lens": seq_lens,
                    "lens64": lens64,
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
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
        stream: fx.Stream = _STREAM,
    ):
        arguments = (
            q,
            k_cache,
            v_cache,
            table,
            table_stride,
            seq_lens,
            lens64,
            window,
            scale_bits,
            partial,
            maximum,
            denominator,
        )
        if baseline_llvm_attrs:
            kernel(
                *arguments,
                value_attrs={
                    "llvm.func_attrs": ir.DictAttr.get(
                        {
                            "amdgpu-sched-strategy": ir.StringAttr.get(
                                "coexec"
                            ),
                            "denormal-fp-math-f32": ir.StringAttr.get(
                                "ieee"
                            ),
                        }
                    ),
                },
            ).launch(grid=(tasks,), block=(THREADS,), stream=stream)
        else:
            kernel(*arguments).launch(
                grid=(tasks,), block=(THREADS,), stream=stream
            )

    if baseline_llvm_attrs:
        launch.compile_hints["llvm_options"] = {
            "amdgpu-sched-strategy": "coexec"
        }
    return launch
