import sys
from dataclasses import asdict
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_synthetic_contract import (  # noqa: E402
    CONTRACT,
    ContractError,
    deterministic_ep8_routing,
    parse_widths,
    rank_manifest,
    routing_histogram,
    validate_manifests,
    validate_rank,
    validate_static_contract,
    virtual_expert_ranges,
)


def _constants(**changes):
    values = {
        "HIDDEN": 8192,
        "HEAD_DIM": 64,
        "LOCAL_Q_HEADS": 16,
        "LOCAL_KV_HEADS": 2,
        "Q_K": 1024,
        "KV_K": 128,
        "QKVG_ROWS": 2304,
        "EXPERTS": 256,
        "LOCAL_EXPERTS": 32,
        "TOP_K": 8,
        "MOE_HIDDEN": 4096,
        "INTER": 4096,
        "MAX_TOKENS": 64,
        "BLOCKS": 96,
    }
    values.update(changes)
    return values


def _manifest(rank):
    return rank_manifest(
        rank=rank,
        active_world_size=4,
        widths=(8, 16, 32, 64),
        production_constants=_constants(),
        device={"gcn_arch": "gfx1250"},
    )


def test_literal_contract_is_tp8_local_on_four_active_ranks():
    assert asdict(CONTRACT) == {
        "global_hidden": 8192,
        "global_q_heads": 128,
        "global_kv_heads": 16,
        "head_dim": 64,
        "logical_tp": 8,
        "logical_ep": 8,
        "active_npes": 4,
        "local_q_heads": 16,
        "local_kv_heads": 2,
        "local_q_width": 1024,
        "local_kv_width": 128,
        "local_qkvg_rows": 2304,
        "global_experts": 256,
        "local_experts": 32,
        "top_k": 8,
        "moe_hidden": 4096,
        "intermediate": 4096,
        "q_len": 8,
        "resident_blocks": 96,
        "widths": (8, 16, 32, 64),
    }


def test_widths_are_exactly_the_supported_verify_widths():
    assert parse_widths("8,16,32,64") == (8, 16, 32, 64)
    for bad in ("", "1", "8,8", "8,24", "8,no"):
        with pytest.raises(ContractError):
            parse_widths(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_world_size", 8),
        ("logical_tp", 4),
        ("logical_ep", 4),
        ("local_q_heads", 32),
        ("local_kv_heads", 4),
        ("local_experts", 64),
        ("resident_blocks", 95),
    ],
)
def test_rank_contract_rejects_non_synthetic_or_tp4_values(field, value):
    args = {
        "rank": 0,
        "active_world_size": 4,
        "logical_tp": 8,
        "logical_ep": 8,
        "local_q_heads": 16,
        "local_kv_heads": 2,
        "local_experts": 32,
        "resident_blocks": 96,
        "widths": (8, 16, 32, 64),
    }
    args[field] = value
    with pytest.raises(ContractError):
        validate_rank(**args)


@pytest.mark.parametrize(
    "change",
    [
        {"LOCAL_Q_HEADS": 32},
        {"LOCAL_KV_HEADS": 4},
        {"Q_K": 2048},
        {"KV_K": 256},
        {"QKVG_ROWS": 4608},
        {"LOCAL_EXPERTS": 64},
        {"BLOCKS": 128},
    ],
)
def test_static_contract_rejects_tp4_or_wrong_build_values(change):
    with pytest.raises(ContractError):
        validate_static_contract(_constants(**change))


def test_four_rank_manifests_cover_only_the_active_ep8_half():
    manifests = [_manifest(rank) for rank in range(4)]
    validate_manifests(manifests)
    assert [item["expert_range"] for item in manifests] == [
        [0, 32],
        [32, 64],
        [64, 96],
        [96, 128],
    ]


def test_four_active_ranks_cover_each_logical_expert_once():
    assert [virtual_expert_ranges(rank) for rank in range(4)] == [
        ((0, 32), (128, 160)),
        ((32, 64), (160, 192)),
        ((64, 96), (192, 224)),
        ((96, 128), (224, 256)),
    ]
    experts = [
        expert
        for rank in range(4)
        for start, end in virtual_expert_ranges(rank)
        for expert in range(start, end)
    ]
    assert sorted(experts) == list(range(CONTRACT.global_experts))
    assert len(experts) == len(set(experts))


@pytest.mark.parametrize("rows", [8, 16])
def test_deterministic_routing_has_one_pick_per_logical_rank(rows):
    ids, scores = deterministic_ep8_routing(rows)
    assert ids == deterministic_ep8_routing(rows)[0]
    assert all(len(row) == CONTRACT.top_k for row in ids)
    assert all(sum(row) == pytest.approx(1.0) for row in scores)
    per_rank, per_expert = routing_histogram(ids)
    assert per_rank == (rows,) * CONTRACT.logical_ep
    assert sum(per_expert[: 4 * CONTRACT.local_experts]) == rows * 4
    assert sum(per_expert[4 * CONTRACT.local_experts :]) == rows * 4


def test_manifest_validation_rejects_a_rank_with_wrong_expert_ownership():
    manifests = [_manifest(rank) for rank in range(4)]
    manifests[2]["expert_range"] = [128, 160]
    with pytest.raises(ContractError):
        validate_manifests(manifests)
