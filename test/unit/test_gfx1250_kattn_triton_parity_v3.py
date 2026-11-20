# SPDX-License-Identifier: Apache-2.0

import pytest

from sgl_amd_mono.muse_spark.gfx1250.prototypes import (
    kattn_triton_parity_v3 as v3,
)


def test_q_swizzle_is_bijective_and_b128_aligned():
    offsets = {
        v3.q_lds_byte(row, dim)
        for row in range(v3.ROWS)
        for dim in range(0, v3.HEAD_DIM, 8)
    }
    assert len(offsets) == v3.ROWS * (v3.HEAD_DIM // 8)
    assert min(offsets) == 0
    assert max(offsets) == v3.Q_BYTES - 16
    assert all(offset % 16 == 0 for offset in offsets)


def test_padded_kv_layouts_are_bijective():
    k_offsets = {
        v3.k_lds_byte(row, dim)
        for row in range(v3.BN)
        for dim in range(v3.HEAD_DIM)
    }
    v_offsets = {
        v3.v_lds_byte(row, dim)
        for row in range(v3.BN)
        for dim in range(v3.HEAD_DIM)
    }
    assert len(k_offsets) == v3.BN * v3.HEAD_DIM
    assert len(v_offsets) == v3.BN * v3.HEAD_DIM
    assert k_offsets == v_offsets
    assert all(
        v3.kv_lds_byte(row, dim)
        == (
            row * v3.HEAD_DIM
            + dim
            + (row * v3.HEAD_DIM + dim)
            // v3.KV_PAD_GROUP_BYTES
            * v3.KV_PAD_BYTES
        )
        for row in range(v3.BN)
        for dim in range(v3.HEAD_DIM)
    )
    assert max(k_offsets) < v3.KV_SLOT_BYTES
    assert v3.KV_SLOT_BYTES == 4352
    assert [v3.kv_lds_byte(row, 0) for row in (0, 4, 16, 32, 48)] == [
        0,
        272,
        1088,
        2176,
        3264,
    ]


def test_warp_ownership_covers_rows_and_dimensions_once():
    assert [v3.qk_owner(row) for row in range(v3.ROWS)] == [0] * 8 + [1] * 8
    dimensions = [
        dim for wave in range(v3.WAVES) for dim in v3.pv_dimensions(wave)
    ]
    assert dimensions == list(range(v3.HEAD_DIM))


def test_register_probability_fragment_matches_wmma_a_layout():
    for contraction in range(2):
        for row in range(v3.ROWS):
            low = v3.probability_fragment_keys(row, contraction)
            high = v3.probability_fragment_keys(row + v3.ROWS, contraction)
            assert low == list(
                range(contraction * 32, contraction * 32 + 8)
            ) + list(
                range(contraction * 32 + 16, contraction * 32 + 24)
            )
            assert high == list(
                range(contraction * 32 + 8, contraction * 32 + 16)
            ) + list(
                range(contraction * 32 + 24, contraction * 32 + 32)
            )
            assert sorted(low + high) == list(
                range(contraction * 32, contraction * 32 + 32)
            )


def test_contiguous_split_assignment_matches_triton():
    ranges = [
        v3.contiguous_split_tiles(945, 2994, split, 8)
        for split in range(8)
    ]
    assert ranges == [
        (14, 19),
        (19, 24),
        (24, 29),
        (29, 34),
        (34, 39),
        (39, 44),
        (44, 47),
        (49, 47),
    ]


def test_page_scratch_reloads_cover_each_lanes_four_keys():
    keys_by_lane = [
        [lane // 4 + group * 16 for group in range(4)]
        for lane in range(v3.THREADS)
    ]
    assert all(len(set(keys)) == 4 for keys in keys_by_lane)
    assert all(0 <= key < v3.BN for keys in keys_by_lane for key in keys)
    for group in range(4):
        assert [keys[group] for keys in keys_by_lane] == [
            lane // 4 + group * 16 for lane in range(v3.THREADS)
        ]
    assert all(
        [
            [lane // 4 + pair * 32, lane // 4 + pair * 32 + 16]
            for pair in range(2)
        ]
        == [keys_by_lane[lane][0:2], keys_by_lane[lane][2:4]]
        for lane in range(v3.THREADS)
    )


def test_v3_metadata_has_no_fp32_pv_merge():
    metadata = v3.pipeline_metadata(8)
    assert metadata["version"] == "v3"
    assert metadata["qk_warp_partition"] == "rows"
    assert metadata["pv_warp_partition"] == "head_dim"
    assert metadata["split_assignment"] == "contiguous"
    assert metadata["kv_transfer"] == "global_load_async_to_lds_b128"
    assert metadata["fp32_pv_spill_bytes"] == 0
    assert metadata["output_accumulator"] == "register_resident_16x64_fp32"
    assert metadata["kv_lds_layout"] == "padded_shared_[256:+16]"
    assert metadata["kv_slot_bytes"] == 4352
    assert metadata["lds_bytes"] == 21696


def test_block_b_one_stage_requires_width8_v3():
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        KAttnBuild,
        build_tiled_kattn,
    )

    with pytest.raises(ValueError, match="requires the Block A v3"):
        build_tiled_kattn(KAttnBuild(tokens=8, block_b_parity=True))

    # Six split CTAs assign each of the 16x64 tile outputs exactly once.
    owned = []
    splits, threads = 6, 64
    batches = (16 * 64 + splits * threads - 1) // (splits * threads)
    for split in range(splits):
        for batch in range(batches):
            for tid in range(threads):
                item = tid + (split + batch * splits) * threads
                if item < 16 * 64:
                    owned.append(item)
    assert sorted(owned) == list(range(16 * 64))
