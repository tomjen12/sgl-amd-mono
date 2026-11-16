# SPDX-License-Identifier: Apache-2.0
"""CPU-only task planning for the gfx1250 TP4 Muse Spark V1.

The first slice deliberately models scheduling without choosing MFMA, thread or
LDS layouts.  Those are wave32-specific and must not inherit gfx950's wave64
arithmetic accidentally.
"""

from sgl_amd_mono.muse_spark.gfx1250.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QKVG_ROWS,
    RESIDENT_BLOCKS,
)
from sgl_amd_mono.common.plan import pair_layout

# Keep the proven gfx950 split-K decomposition initially.  TP4 produces 512
# projection tasks.  The 192 resident CTAs consume them in grid-stride order
# before any CTA enters its optional polling head/gate stage.
THREADS = 256
WAVE_SIZE = 32
WAVES = THREADS // WAVE_SIZE
ROWS = 16
TILE = 16
GEMV_ROWS_PER_TASK_GROUP = 144
GEMV_ROW_TILES = GEMV_ROWS_PER_TASK_GROUP // ROWS
K_PARTS = 16
K_PART = HIDDEN // K_PARTS
K_PER_WAVE = K_PART // WAVES
WMMA_K = 32
K_STEPS = K_PER_WAVE // WMMA_K
WMMA_ACCUMULATORS = 8
RED_FLOATS = WAVES * WAVE_SIZE * WMMA_ACCUMULATORS
GEMV_GROUPS = QKVG_ROWS // GEMV_ROWS_PER_TASK_GROUP
GEMV_TASKS = GEMV_GROUPS * K_PARTS
GEMV_TASK_ROUNDS = (GEMV_TASKS + RESIDENT_BLOCKS - 1) // RESIDENT_BLOCKS

HEAD_TASKS = LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS
GATE_TASKS = Q_K // HEAD_DIM
HEAD_TASK_BASE = 0
GATE_TASK_BASE = HEAD_TASK_BASE + HEAD_TASKS

assert QKVG_ROWS % GEMV_ROWS_PER_TASK_GROUP == 0
assert HIDDEN % K_PARTS == 0
assert GEMV_ROWS_PER_TASK_GROUP % ROWS == 0
assert K_PER_WAVE % WMMA_K == 0
assert GATE_TASK_BASE + GATE_TASKS <= RESIDENT_BLOCKS


def k1_gemv_tasks(block_id: int) -> tuple[int, ...]:
    """Return the GEMV task ids run by one persistent CTA."""
    if not 0 <= block_id < RESIDENT_BLOCKS:
        raise ValueError(f"block id {block_id}, want [0, {RESIDENT_BLOCKS})")
    return tuple(
        block_id + task_round * RESIDENT_BLOCKS
        for task_round in range(GEMV_TASK_ROUNDS)
        if block_id + task_round * RESIDENT_BLOCKS < GEMV_TASKS
    )


def gemv_task_shape(task_id: int) -> tuple[int, int]:
    """Decode a GEMV task into ``(projection row group, K part)``."""
    if not 0 <= task_id < GEMV_TASKS:
        raise ValueError(f"task id {task_id}, want [0, {GEMV_TASKS})")
    return task_id // K_PARTS, task_id % K_PARTS


def k1_post_tasks(block_id: int) -> tuple[str, int] | None:
    """Return the head or gate reduction assigned after this CTA's GEMV tasks."""
    if not 0 <= block_id < RESIDENT_BLOCKS:
        raise ValueError(f"block id {block_id}, want [0, {RESIDENT_BLOCKS})")
    if HEAD_TASK_BASE <= block_id < HEAD_TASK_BASE + HEAD_TASKS:
        return "head", block_id - HEAD_TASK_BASE
    if GATE_TASK_BASE <= block_id < GATE_TASK_BASE + GATE_TASKS:
        return "gate", block_id - GATE_TASK_BASE
    return None


def k1_scratch_layout() -> dict:
    """Tagged fp32 partials for every K part, token and projection row."""
    return pair_layout(
        [("k1_part", K_PARTS * MAX_TOKENS * QKVG_ROWS)],
        align=256,
    )


def scratch_bytes(layout: dict) -> int:
    return max(offset + size for offset, size in layout.values())


def k1_lds_bytes() -> int:
    """Nine WMMA accumulator row tiles, reduced across eight wave32s."""
    return GEMV_ROW_TILES * RED_FLOATS * 4


# ---------------------------------------------------------------- K-attn V1
GQA = LOCAL_Q_HEADS // LOCAL_KV_HEADS
KATTN_TASKS = MAX_TOKENS * LOCAL_Q_HEADS
KATTN_TASK_ROUNDS = (KATTN_TASKS + RESIDENT_BLOCKS - 1) // RESIDENT_BLOCKS
KATTN_WAVE_STATS = 2 * WAVES + 2  # per-wave max/sum and block max/sum
KATTN_OUT_FLOATS = WAVES * HEAD_DIM

assert GQA == 8
assert KATTN_TASKS > RESIDENT_BLOCKS


def kattn_lds_bytes() -> int:
    return (KATTN_WAVE_STATS + KATTN_OUT_FLOATS) * 4


# The fused Target attention layout is kept here so CPU-only geometry tests do
# not need to import FlyDSL/AITER and probe for a GPU.
FUSED_ROWS = 16
FUSED_SPLITS = 6
FUSED_TILES = LOCAL_KV_HEADS * (MAX_TOKENS // 2)
FUSED_TASKS = FUSED_TILES * FUSED_SPLITS
FUSED_KEY_TILE = 16
FUSED_KEY_BLOCK = 32
FUSED_BLOCK_STRIDE = FUSED_SPLITS * WAVES
FUSED_WMMA_K = 32
FUSED_WMMA_ACC = 8
FUSED_DIM_TILES = HEAD_DIM // FUSED_ROWS
FUSED_STAT_FLOATS = FUSED_TASKS * FUSED_ROWS * 2
FUSED_GLOBAL_FLOATS = FUSED_TILES * FUSED_ROWS * 2
FUSED_PART_FLOATS = FUSED_TASKS * FUSED_ROWS * HEAD_DIM

assert (LOCAL_Q_HEADS, LOCAL_KV_HEADS, GQA) == (32, 4, 8)
assert (FUSED_TILES, FUSED_TASKS) == (16, 96)
assert HEAD_DIM == 2 * FUSED_WMMA_K
