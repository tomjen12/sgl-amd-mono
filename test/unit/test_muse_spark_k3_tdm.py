import pytest

from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
from sgl_amd_mono.muse_spark.gfx1250.config import (
    LDS_CAPACITY_BYTES,
)
from sgl_amd_mono.muse_spark.gfx1250.k3_tdm import (
    DOWN_N,
    GATE_N,
    TILE_M,
    TILE_N,
    build_k3_tdm_tasks,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import RESIDENT_BLOCKS


def test_k3_tdm_contract_matches_persistent_k3():
    tasks = build_k3_tdm_tasks(
        threads=256,
        persistent_workers=RESIDENT_BLOCKS,
        shuffled=True,
    )
    contract = tasks.contract

    assert contract.block == 256
    assert contract.arena_bytes == 81_408
    assert contract.arena_bytes < LDS_CAPACITY_BYTES
    assert contract.gate_tiles_per_pick == GATE_N // TILE_N == 32
    assert contract.down_tiles_per_pick == DOWN_N // TILE_N == 16
    assert "t64x256x128_w2x4_b3" in contract.gate_name
    assert "_act1_q1r2_" in contract.gate_name
    assert "t64x256x128_w2x4_b3" in contract.down_name
    assert "_act" not in contract.down_name
    assert TILE_M == 64


def test_k3_tdm_rejects_native_quark_weight_layout():
    with pytest.raises(
        ValueError, match="requires AITER-preshuffled MXFP4 weights"
    ):
        build_k3_tdm_tasks(
            threads=256,
            persistent_workers=RESIDENT_BLOCKS,
            shuffled=False,
        )


def test_k3_tdm_rejects_parent_block_mismatch():
    with pytest.raises(ValueError, match="block mismatch"):
        build_k3_tdm_tasks(
            threads=128,
            persistent_workers=RESIDENT_BLOCKS,
            shuffled=True,
        )


def test_k3_tdm_is_opt_in_until_gpu_gate_passes():
    assert K3Build().tdm_expert is False
    assert K3Build(shuffled=True, tdm_expert=True).tdm_expert is True


def test_k3_tdm_builds_launch_contract_without_gpu():
    launch = build_k3(K3Build(shuffled=True, tdm_expert=True))
    assert callable(launch)
