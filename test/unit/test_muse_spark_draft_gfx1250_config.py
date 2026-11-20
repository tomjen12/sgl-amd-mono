"""Checkpoint-shape checks for the gfx1250 DFlash2 TP4 draft profile."""

import torch

from sgl_amd_mono.muse_spark import draft as D
from sgl_amd_mono.muse_spark.gfx1250 import draft_config as C
from sgl_amd_mono.muse_spark.gfx1250 import draft_layout as L


def test_dflash2_checkpoint_shapes():
    assert (C.LAYERS, C.HIDDEN, C.INTERMEDIATE) == (5, 8192, 28672)
    assert (C.BLOCK, C.WINDOW, C.PAGE) == (8, 2048, 16)
    assert C.MAX_BATCH_TOKENS == C.MAX_TOKENS == 64
    assert C.SUPPORTED_TOKENS == (8, 16, 32, 64)
    assert (C.TOTAL_Q_HEADS, C.TOTAL_KV_HEADS, C.HEAD_DIM) == (64, 8, 128)
    assert (C.CONV_GROUP, C.CONV_TAPS, C.CONV_SIDES) == (16, 2, 2)


def test_tp4_local_shapes():
    assert C.TP == 4
    assert (C.LOCAL_Q_HEADS, C.LOCAL_KV_HEADS) == (16, 2)
    assert (C.Q_K, C.KV_K, C.QKV_ROWS) == (2048, 256, 2560)
    assert (C.LOCAL_INTERMEDIATE, C.GATE_UP_ROWS) == (7168, 14336)
    assert C.COEF_ROWS == 2048


def test_active_cu_profile_is_supported():
    assert C.RESIDENT_BLOCKS in (96, 192)


def test_tp4_projection_task_shapes():
    assert (L.COEF_TASKS, L.GATE_UP_TASKS, L.DOWN_TASKS) == (256, 512, 448)
    assert (L.QKV_TASKS, L.O_TASKS, L.ATTENTION_TASKS) == (256, 256, 1024)
    for tasks in (
        L.COEF_TASKS,
        L.GATE_UP_TASKS,
        L.DOWN_TASKS,
        L.QKV_TASKS,
        L.O_TASKS,
        L.ATTENTION_TASKS,
    ):
        assigned = [
            task
            for block in range(C.RESIDENT_BLOCKS)
            for task in L.tasks_for_block(tasks, block)
        ]
        assert sorted(assigned) == list(range(tasks))
        assert len(set(assigned)) == tasks


def test_variable_request_attention_geometry():
    for tokens in C.SUPPORTED_TOKENS:
        requests = tokens // C.BLOCK
        assert L.request_count(tokens) == requests
        assert L.attention_tasks(tokens) == requests * 96


def test_graph_table_width_uses_host_planning_bound():
    lens = torch.tensor([2048], dtype=torch.int32)
    lens_cpu = torch.tensor([2064], dtype=torch.int32)

    assert D._mono_table_width(8192, lens, lens_cpu) == 2072
    assert D._mono_table_width(2064, lens, lens_cpu) == 2064
