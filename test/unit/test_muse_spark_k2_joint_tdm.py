import pytest

from sgl_amd_mono.muse_spark.gfx1250 import build_k2
from sgl_amd_mono.muse_spark.gfx1250.stages import post_attn


def test_joint_tdm_geometry_matches_packed_baseline_tile():
    assert post_attn.JOINT_TASKS == 72
    assert (
        post_attn.JOINT_TDM_A_ROWS,
        post_attn.JOINT_TDM_B_ROWS,
        post_attn.JOINT_TDM_K,
        post_attn.JOINT_TDM_BUFFERS,
    ) == (16, 64, 256, 6)
    assert post_attn.JOINT_TDM_COPIES_PER_THREAD == 10
    assert post_attn.JOINT_TDM_K_TILES == 32
    assert post_attn.joint_tdm_lds_bytes() == 253_632
    assert build_k2.JOINT_TDM_ARENA_ELEMS * 2 == 253_632


@pytest.mark.parametrize(
    "chunk,expected",
    [
        (0, ("a", 0, 0)),
        (31, ("a", 0, 248)),
        (511, ("a", 15, 248)),
        (512, ("b", 0, 0)),
        (2559, ("b", 63, 248)),
    ],
)
def test_joint_tdm_async_copy_mapping(chunk, expected):
    assert post_attn.joint_tdm_copy_coordinates(chunk) == expected


@pytest.mark.parametrize("chunk", [-1, 2560])
def test_joint_tdm_async_copy_mapping_rejects_invalid_chunks(chunk):
    with pytest.raises(ValueError):
        post_attn.joint_tdm_copy_coordinates(chunk)


def test_joint_tdm_requires_packed_joint_path():
    with pytest.raises(ValueError, match="requires the packed joint"):
        build_k2.build_k2(
            build_k2.K2Build(tokens=8, joint_rp=False, joint_rp_tdm=True)
        )
