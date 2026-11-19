# SPDX-License-Identifier: Apache-2.0

import pytest

from sgl_amd_mono.muse_spark.gfx1250.layout import (
    FUSED_KEY_BLOCK_CHOICES,
    FUSED_SPLITS,
    FUSED_SPLIT_CHOICES,
    RESIDENT_BLOCKS,
    fused_key_block_plan,
    fused_scratch_task,
    fused_tasks,
    fused_tiles,
    resident_rounds,
    validate_fused_splits,
)


@pytest.mark.parametrize("splits", FUSED_SPLIT_CHOICES)
@pytest.mark.parametrize("keys", (8, 32, 33, 4096))
@pytest.mark.parametrize("key_block", FUSED_KEY_BLOCK_CHOICES)
def test_split_plan_covers_each_key_block_once(keys, splits, key_block):
    plan = fused_key_block_plan(keys, splits, key_block=key_block)
    assigned = [block for split_blocks in plan for block in split_blocks]
    expected = list(range((keys + key_block - 1) // key_block))

    assert len(plan) == splits
    assert sorted(assigned) == expected
    assert len(assigned) == len(set(assigned))


@pytest.mark.parametrize("splits", FUSED_SPLIT_CHOICES)
@pytest.mark.parametrize("key_block", FUSED_KEY_BLOCK_CHOICES)
@pytest.mark.parametrize(
    ("keys", "begin", "end"),
    (
        (8192, 6145, 8192),
        (8192, 6138, 8185),
        (3008, 954, 3001),
        (64, 17, 33),
    ),
)
def test_window_plan_skips_only_blocks_outside_live_union(
    keys, begin, end, splits, key_block
):
    plan = fused_key_block_plan(
        keys, splits, begin, end, key_block=key_block
    )
    assigned = [block for split_blocks in plan for block in split_blocks]

    assert sorted(assigned) == list(
        range(
            begin // key_block,
            (end + key_block - 1) // key_block,
        )
    )
    assert len(assigned) == len(set(assigned))


@pytest.mark.parametrize("splits", FUSED_SPLIT_CHOICES)
@pytest.mark.parametrize("tokens", (8, 16, 32, 64))
def test_active_task_count_and_resident_rounds(tokens, splits):
    tasks = fused_tasks(tokens, splits)

    assert tasks == fused_tiles(tokens) * splits
    assert resident_rounds(tasks) == (
        tasks + RESIDENT_BLOCKS - 1
    ) // RESIDENT_BLOCKS


@pytest.mark.parametrize("splits", FUSED_SPLIT_CHOICES)
def test_scratch_slots_keep_max_six_stride(splits):
    active = [
        fused_scratch_task(tile, split)
        for tile in range(4)
        for split in range(splits)
    ]

    assert len(active) == len(set(active))
    assert active[-1] == 3 * FUSED_SPLITS + splits - 1
    if splits < FUSED_SPLITS:
        assert fused_scratch_task(1, 0) - fused_scratch_task(0, 0) == FUSED_SPLITS


@pytest.mark.parametrize("splits", (0, 1, 5, 7))
def test_invalid_split_count_is_rejected(splits):
    with pytest.raises(ValueError, match="splits"):
        validate_fused_splits(splits)
