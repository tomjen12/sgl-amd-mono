import sys
from dataclasses import asdict
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_synthetic_contract import (  # noqa: E402
    CONTRACT,
    WORKLOAD,
    ContractError,
    deterministic_ep8_routing,
    ep8_expert_coverage_suite,
    local_expert_range,
    parse_widths,
    rank_manifest,
    routing_histogram,
    runtime_workload_manifest,
    specialization_attestation,
    validate_manifests,
    validate_rank,
    validate_runtime_workload_parity,
    validate_static_contract,
    validate_kernel_symbol,
    validate_ep8_expert_coverage,
    validate_workload_manifests,
    virtual_expert_ranges,
    workload_manifest,
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


@pytest.mark.parametrize("active", [1, 2, 4])
def test_rank_manifests_support_active_one_two_or_four(active):
    manifests = [
        rank_manifest(
            rank=rank,
            active_world_size=active,
            widths=(8, 16),
            production_constants=_constants(),
            device={"gcn_arch": "gfx1250"},
        )
        for rank in range(active)
    ]
    validate_manifests(manifests, active_npes=active)
    assert all(item["contract"]["active_npes"] == active for item in manifests)


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


def test_workload_contract_is_w8_with_one_router_slice_per_active_rank():
    assert asdict(WORKLOAD) == {
        "logical_npes": 8,
        "active_npes": 4,
        "peer_slots": 8,
        "router_slices_per_active_rank": 1,
        "router_experts_per_active_rank": 32,
        "local_experts": 32,
        "pooler_mode": "sharded",
        "resident_blocks": 96,
    }
    manifests = [
        workload_manifest(
            rank=rank,
            kernel_npes=8,
            peer_slots=8,
            router_ranges=(local_expert_range(rank),),
            pooler_mode="sharded",
            resident_blocks=96,
        )
        for rank in range(4)
    ]
    validate_workload_manifests(manifests)
    assert [item["router_experts"] for item in manifests] == [32] * 4


@pytest.mark.parametrize("active", [1, 2, 4])
def test_workload_manifests_keep_w8_for_supported_active_counts(active):
    manifests = [
        workload_manifest(
            rank=rank,
            kernel_npes=8,
            peer_slots=8,
            router_ranges=(local_expert_range(rank),),
            pooler_mode="sharded",
            resident_blocks=96,
            active_npes=active,
        )
        for rank in range(active)
    ]
    validate_workload_manifests(manifests, active_npes=active)
    assert all(item["logical_npes"] == 8 for item in manifests)
    assert all(item["active_npes"] == active for item in manifests)


@pytest.mark.parametrize(
    "change",
    [
        {"kernel_npes": 4},
        {"peer_slots": 4},
        {"router_ranges": ((0, 32), (128, 160))},
        {"pooler_mode": "replicated"},
        {"resident_blocks": 128},
    ],
)
def test_workload_contract_rejects_functional_bringup_values(change):
    values = {
        "rank": 0,
        "kernel_npes": 8,
        "peer_slots": 8,
        "router_ranges": ((0, 32),),
        "pooler_mode": "sharded",
        "resident_blocks": 96,
    }
    values.update(change)
    with pytest.raises(ContractError):
        workload_manifest(**values)


@pytest.mark.parametrize("stage", ["k2", "k3", "layer_v1", "layer_v2"])
@pytest.mark.parametrize("rows", [8, 16])
def test_specialization_attestation_requires_w8(stage, rows):
    attestation = specialization_attestation(stage, rows)
    assert attestation["kernel_npes"] == 8
    assert attestation["peer_slots"] == 8
    assert attestation["resident_ctas"] == 96
    if stage != "k3":
        assert attestation["router_tasks_per_active_rank"] == 1
        assert attestation["pooler_rows_per_active_rank"] == 512
        assert "ps1" in attestation["expected_symbol_fragments"]
    if stage != "k2":
        assert attestation["local_experts_per_active_rank"] == 32
        assert attestation["unpool_rows_per_active_rank"] == 1024


def test_kernel_symbol_attestation_rejects_w4_or_replicated_pooler():
    attestation = specialization_attestation("k2", 8)
    validate_kernel_symbol(attestation, "muse_spark_k2_s8_w8_ps1")
    for bad in ("muse_spark_k2_s8_w4_ps0", "muse_spark_k2_s8_w8_ps0"):
        with pytest.raises(ContractError):
            validate_kernel_symbol(attestation, bad)


@pytest.mark.parametrize("active", [1, 2, 4])
def test_runtime_workload_parity_accepts_same_active_tp8_contract(active):
    baseline = [
        runtime_workload_manifest(
            implementation="baseline",
            stage="k3",
            width=8,
            rank=rank,
            active_npes=active,
        )
        for rank in range(active)
    ]
    mono = [
        runtime_workload_manifest(
            implementation="mono",
            stage="k3",
            width=8,
            rank=rank,
            active_npes=active,
        )
        for rank in range(active)
    ]
    evidence = validate_runtime_workload_parity(baseline, mono)
    assert evidence["status"] == "passed"
    assert evidence["case_count"] == active


def test_runtime_workload_parity_rejects_different_active_counts():
    baseline = [
        runtime_workload_manifest(
            implementation="baseline",
            stage="k3",
            width=8,
            rank=0,
            active_npes=1,
        )
    ]
    mono = [
        runtime_workload_manifest(
            implementation="mono",
            stage="k3",
            width=8,
            rank=rank,
            active_npes=2,
        )
        for rank in range(2)
    ]
    with pytest.raises(ContractError):
        validate_runtime_workload_parity(baseline, mono)


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


@pytest.mark.parametrize("rows,case_count", [(8, 4), (16, 2)])
def test_ep8_suite_covers_all_32_experts_per_logical_rank(rows, case_count):
    cases = ep8_expert_coverage_suite(rows)
    assert len(cases) == case_count
    evidence = validate_ep8_expert_coverage(cases)
    assert evidence["experts_covered_per_logical_rank"] == [32] * 8
    assert evidence["missing_logical_ranks"] == [4, 5, 6, 7]
    assert evidence["missing_expert_output"] == "tagged-zero"


def test_manifest_validation_rejects_a_rank_with_wrong_expert_ownership():
    manifests = [_manifest(rank) for rank in range(4)]
    manifests[2]["expert_range"] = [128, 160]
    with pytest.raises(ContractError):
        validate_manifests(manifests)
