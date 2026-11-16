"""CPU-only checks for the active gfx1250 TP4 CU profile and task maps."""

import pytest

from sgl_amd_mono.muse_spark.gfx1250 import config as C
from sgl_amd_mono.muse_spark.gfx1250 import layout as L


def test_tp4_profile_shapes():
    assert (C.TP, C.EP, C.WIDTHS) == (4, 4, (8,))
    assert (C.LOCAL_Q_HEADS, C.LOCAL_KV_HEADS, C.LOCAL_EXPERTS) == (32, 4, 64)
    assert (C.Q_K, C.KV_K, C.G_OFF, C.QKVG_ROWS) == (2048, 256, 2560, 4608)
    assert C.PROFILE in {"cu96_tp4", "cu192_tp4"}
    assert C.RESIDENT_BLOCKS in {96, 192}
    assert (C.WAVE_SIZE, C.RESIDENT_BLOCKS) == (32, C.CU_COUNT)


def test_k1_maps_512_gemv_tasks_to_profile_blocks():
    assigned = [
        task
        for block_id in range(C.RESIDENT_BLOCKS)
        for task in L.k1_gemv_tasks(block_id)
    ]
    assert L.GEMV_TASKS == 512
    assert sorted(assigned) == list(range(L.GEMV_TASKS))
    assert len(set(assigned)) == L.GEMV_TASKS
    counts = [len(L.k1_gemv_tasks(block_id)) for block_id in range(C.RESIDENT_BLOCKS)]
    rounds, extra = divmod(L.GEMV_TASKS, C.RESIDENT_BLOCKS)
    assert counts[:extra] == [rounds + 1] * extra
    assert counts[extra:] == [rounds] * (C.RESIDENT_BLOCKS - extra)

    shapes = {L.gemv_task_shape(task) for task in assigned}
    assert shapes == {
        (row_group, k_part)
        for row_group in range(L.GEMV_GROUPS)
        for k_part in range(L.K_PARTS)
    }


def test_k1_uses_wave32_and_scratch_covers_every_partial():
    assert (L.THREADS, L.WAVE_SIZE, L.WAVES) == (256, 32, 8)
    assert (L.K_PER_WAVE, L.K_STEPS) == (64, 2)
    assert L.k1_lds_bytes() == 72 * 1024
    assert L.k1_lds_bytes() <= C.LDS_CAPACITY_BYTES
    layout = L.k1_scratch_layout()
    assert layout["k1_part"][1] == L.K_PARTS * C.MAX_TOKENS * C.QKVG_ROWS * 8
    assert L.scratch_bytes(layout) == layout["k1_part"][1]


def test_k1_post_tasks_cover_tp4_heads_and_gates_once():
    tasks = [
        task
        for block_id in range(C.RESIDENT_BLOCKS)
        if (task := L.k1_post_tasks(block_id)) is not None
    ]
    assert tasks == [
        *(("head", i) for i in range(L.HEAD_TASKS)),
        *(("gate", i) for i in range(L.GATE_TASKS)),
    ]


def test_kattn_maps_all_bs1_query_heads_over_profile_ctas():
    assert L.GQA == 8
    assert L.KATTN_TASKS == C.MAX_TOKENS * C.LOCAL_Q_HEADS == 256
    assigned = [
        block_id + task_round * C.RESIDENT_BLOCKS
        for block_id in range(C.RESIDENT_BLOCKS)
        for task_round in range(L.KATTN_TASK_ROUNDS)
        if block_id + task_round * C.RESIDENT_BLOCKS < L.KATTN_TASKS
    ]
    assert sorted(assigned) == list(range(L.KATTN_TASKS))
    assert len(set(assigned)) == L.KATTN_TASKS
    assert L.kattn_lds_bytes() < 4 * 1024


def test_fused_target_attention_cu96_geometry_and_scratch():
    assert (L.FUSED_TILES, L.FUSED_SPLITS, L.FUSED_TASKS) == (16, 6, 96)
    assert (L.FUSED_ROWS, L.FUSED_KEY_BLOCK, L.FUSED_BLOCK_STRIDE) == (
        16,
        32,
        48,
    )
    assert L.FUSED_DIM_TILES == 4
    assert L.FUSED_STAT_FLOATS * 4 == 12_288
    assert L.FUSED_GLOBAL_FLOATS * 4 == 2_048
    assert L.FUSED_PART_FLOATS * 4 == 393_216
    assert L.FUSED_TASKS * 8 == 768
    assert L.FUSED_TILES * 8 == 128


@pytest.mark.parametrize("block_id", [-1, C.RESIDENT_BLOCKS])
def test_invalid_block_is_rejected(block_id):
    with pytest.raises(ValueError):
        L.k1_gemv_tasks(block_id)
    with pytest.raises(ValueError):
        L.k1_post_tasks(block_id)
