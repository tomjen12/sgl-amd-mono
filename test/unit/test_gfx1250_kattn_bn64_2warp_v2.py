# SPDX-License-Identifier: Apache-2.0

from sgl_amd_mono.muse_spark.gfx1250.prototypes import (
    kattn_bn64_2warp_v2 as v2,
)


def test_v2_dma_layout_is_bijective():
    offsets = {
        v2._dma_byte(key, dim)
        for key in range(v2.WAVE_BN)
        for dim in range(v2.HEAD_DIM)
    }

    assert offsets == set(range(v2.WAVE_BN * v2.HEAD_DIM))


def test_v2_pipeline_metadata_is_isolated():
    metadata = v2.pipeline_metadata(8)

    assert metadata["version"] == "v2"
    assert metadata["waves"] == 2
    assert metadata["bm"] == 16
    assert metadata["bn"] == 64
    assert metadata["stages"] == 2
    assert metadata["q_staged_once"] is True
    assert metadata["kv_transfer"] == "global_load_async_to_lds_b128"
    assert metadata["direct_to_lds"] is True
    assert metadata["async_copies_per_wave_tile"] == 8
    assert metadata["wait"] == "s_wait_asynccnt 0"
    assert "_v2_" in metadata["symbol"]
    assert metadata["lds_bytes"] == 20800


def test_v2_rejects_unsupported_split_count():
    try:
        v2.pipeline_metadata(7)
    except ValueError as error:
        assert "splits" in str(error)
    else:
        raise AssertionError("unsupported split count was accepted")
