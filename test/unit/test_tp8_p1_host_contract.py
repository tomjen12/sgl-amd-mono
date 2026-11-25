# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for TP8-on-active4 P1 harness contracts."""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_p1_host_contract import (  # noqa: E402
    ACTIVE_WORLD,
    BF16_EXPERT_LAYERS,
    NEXT_K1_INTERMEDIATES,
    REFERENCE_INTERMEDIATES,
    REQUIRED_ENV,
    SWA_WINDOW,
    ExecutionEvent,
    HarnessContract,
    KVContract,
    P1ContractError,
    deterministic_routing_cases,
    bf16_boundary_dispatch,
    expected_execution,
    layer_variant_matrix,
    logical_rank_batches,
    repeat_fixture_plan,
    routing_histogram,
    validate_execution,
    validate_bf16_dispatch,
    validate_launch_contract,
    validate_logical_rank_observations,
    validate_reference_manifest,
    validate_repeat_observations,
    whole_layer_reference_schema,
)


def _variant(attention="full", version="v1", width=8):
    return next(
        item
        for item in layer_variant_matrix()
        if item.attention == attention
        and item.layer_version == version
        and item.width == width
    )


@pytest.mark.parametrize("rows", [8, 16])
def test_required_routing_cases_are_deterministic_and_valid(rows):
    cases = deterministic_routing_cases(rows)
    assert tuple(case.name for case in cases) == (
        "hot_expert",
        "zero_local_pick",
        "expert_boundary",
        "score_ties",
        "imbalance",
    )
    assert cases == deterministic_routing_cases(rows)
    assert all(0 in row for row in cases[0].ids)
    assert routing_histogram(cases[1])[2] == 0
    boundary_ids = {expert for row in cases[2].ids for expert in row}
    assert {31, 32, 127, 128}.issubset(boundary_ids)
    assert all(len(set(scores)) == 1 for scores in cases[3].scores)
    imbalance = routing_histogram(cases[4])
    assert imbalance[0] > max(imbalance[2:])


def test_routing_cases_reject_non_gate_widths():
    for rows in (1, 24, 32, 64):
        with pytest.raises(P1ContractError, match="expected one of"):
            deterministic_routing_cases(rows)


@pytest.mark.parametrize(
    ("active_gpus", "expected"),
    [
        (1, tuple((rank,) for rank in range(8))),
        (2, ((0, 1), (2, 3), (4, 5), (6, 7))),
    ],
)
def test_active_one_two_batch_schedule_covers_logical_ranks(active_gpus, expected):
    assert logical_rank_batches(active_gpus) == expected
    observations = [
        {
            "logical_ranks": ranks,
            "active_gpus": active_gpus,
            "runtime_mapping_hash": f"mapping-{index}",
            "status": "passed",
        }
        for index, ranks in enumerate(expected)
    ]
    validate_logical_rank_observations(active_gpus, observations)


def test_logical_rank_coverage_rejects_missing_reordered_or_unmapped_batch():
    observations = [
        {
            "logical_ranks": ranks,
            "active_gpus": 2,
            "runtime_mapping_hash": f"mapping-{index}",
            "status": "passed",
        }
        for index, ranks in enumerate(logical_rank_batches(2))
    ]
    for field, bad in (
        ("logical_ranks", (3, 2)),
        ("active_gpus", 1),
        ("runtime_mapping_hash", ""),
        ("status", "failed"),
    ):
        broken = [dict(item) for item in observations]
        broken[1][field] = bad
        with pytest.raises(P1ContractError):
            validate_logical_rank_observations(2, broken)


def test_full_swa_v1_v2_width_matrix_is_complete():
    variants = layer_variant_matrix()
    assert len(variants) == 8
    assert len({variant.case_id for variant in variants}) == 8
    assert {
        (variant.width, variant.attention, variant.layer_version)
        for variant in variants
    } == {
        (width, attention, version)
        for width in (8, 16)
        for attention in ("full", "swa")
        for version in ("v1", "v2")
    }
    for variant in variants:
        assert variant.next_k1 == (variant.layer_version == "v2")
        assert variant.window == (SWA_WINDOW if variant.attention == "swa" else None)
        assert variant.kv_pool == variant.attention


@pytest.mark.parametrize("attention", ["full", "swa"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_production_execution_order_for_layer_variants(attention, version):
    variant = _variant(attention, version)
    events = expected_execution(layer=9, variant=variant, bootstrap=True)
    assert [event.stage for event in events] == [
        "k1_bootstrap",
        "kattn",
        "k2",
        "k3",
    ] + (["k1_next"] if version == "v2" else [])
    assert events[1].attention == attention
    assert events[1].window == (SWA_WINDOW if attention == "swa" else None)
    assert sum(event.stage in ("k1_bootstrap", "k1_next") for event in events) == (
        2 if version == "v2" else 1
    )
    validate_execution(events, events)


def test_execution_order_covers_transition_and_last_layer_rules():
    assert BF16_EXPERT_LAYERS == {0, 61}
    first = expected_execution(layer=0, variant=_variant(version="v2"), bootstrap=True)
    assert next(event for event in first if event.stage == "k3").expert_family == "bf16"

    transition = expected_execution(
        layer=60,
        variant=_variant(attention="full", version="v2"),
        next_attention="swa",
    )
    assert [event.stage for event in transition] == ["kattn", "k2", "k3", "k1_next"]
    assert transition[-1] == ExecutionEvent(
        "k1_next",
        61,
        attention="swa",
        window=SWA_WINDOW,
        kv_pool="swa",
    )

    last = expected_execution(layer=61, variant=_variant(version="v1"))
    assert next(event for event in last if event.stage == "k3").expert_family == "bf16"
    with pytest.raises(P1ContractError, match="last layer"):
        expected_execution(layer=61, variant=_variant(version="v2"))


@pytest.mark.parametrize("layer", [0, 61])
@pytest.mark.parametrize("attention", ["full", "swa"])
def test_bf16_boundary_dispatch_uses_sglang_experts_not_mxfp4_k3(
    layer, attention
):
    next_k1 = layer == 0
    observed = bf16_boundary_dispatch(
        layer=layer, attention=attention, next_k1=next_k1
    )
    observed["operator_sequence"] = (
        "mono_k1",
        "mono_kattn",
        "mono_k2",
        "sglang_tpx_bf16",
    ) + (("mono_k1_next",) if next_k1 else ())
    validate_bf16_dispatch(
        observed,
        layer=layer,
        attention=attention,
        next_k1=next_k1,
    )


def test_bf16_boundary_dispatch_rejects_mxfp4_and_bad_last_next_k1():
    observed = bf16_boundary_dispatch(
        layer=0, attention="full", next_k1=True
    )
    observed["operator_sequence"] = (
        "mono_k1",
        "mono_kattn",
        "mono_k2",
        "mono_mxfp4_k3",
        "mono_k1_next",
    )
    with pytest.raises(P1ContractError, match="expert operator"):
        validate_bf16_dispatch(
            observed, layer=0, attention="full", next_k1=True
        )
    with pytest.raises(P1ContractError, match="last"):
        bf16_boundary_dispatch(layer=61, attention="swa", next_k1=True)


def test_execution_order_rejects_duplicate_current_k1_and_wrong_variant():
    expected = expected_execution(layer=9, variant=_variant(attention="swa", version="v2"))
    duplicate = (ExecutionEvent("k1_bootstrap", 9),) + expected
    with pytest.raises(P1ContractError, match="execution mismatch"):
        validate_execution(duplicate, expected)

    wrong_window = list(expected)
    wrong_window[0] = replace(wrong_window[0], window=None)
    with pytest.raises(P1ContractError, match="execution mismatch"):
        validate_execution(wrong_window, expected)


def _valid_launch():
    return asdict(HarnessContract())


def test_valid_launch_contract_is_logical_tp8_on_active_four():
    validate_launch_contract(
        _valid_launch(),
        env=REQUIRED_ENV,
        kv=KVContract(),
    )
    assert HarnessContract().active_world == ACTIVE_WORLD


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("logical_tp", 4),
        ("logical_ep", 4),
        ("active_world", 8),
        ("peer_slots", 4),
        ("local_q_heads", 32),
        ("local_kv_heads", 4),
        ("local_experts", 64),
        ("pooler", "replicated"),
    ],
)
def test_launch_contract_rejects_tp4_wrong_world_and_pooler(field, bad):
    declaration = _valid_launch()
    declaration[field] = bad
    with pytest.raises(P1ContractError, match=field):
        validate_launch_contract(declaration, env=REQUIRED_ENV, kv=KVContract())


@pytest.mark.parametrize(
    "kv",
    [
        replace(KVContract(), layout="HND"),
        replace(KVContract(), page_size=32),
        replace(KVContract(), local_kv_heads=4),
        replace(KVContract(), dtype="bf16"),
        replace(KVContract(), swa_pool="full"),
    ],
)
def test_launch_contract_rejects_wrong_kv_contract(kv):
    with pytest.raises(P1ContractError, match="KV contract"):
        validate_launch_contract(_valid_launch(), env=REQUIRED_ENV, kv=kv)


@pytest.mark.parametrize("name", tuple(REQUIRED_ENV))
def test_launch_contract_rejects_missing_or_wrong_environment(name):
    env = dict(REQUIRED_ENV)
    env[name] = "wrong"
    with pytest.raises(P1ContractError, match=name):
        validate_launch_contract(_valid_launch(), env=env, kv=KVContract())
    del env[name]
    with pytest.raises(P1ContractError, match=name):
        validate_launch_contract(_valid_launch(), env=env, kv=KVContract())


def _observations(plan):
    return [
        {
            "fixture_id": plan.fixture_id,
            "scratch_id": plan.scratch_id,
            "shadow_id": plan.shadow_id,
            "seed": plan.seed,
            "epoch": epoch,
            "poisoned": True,
            "poison_digest": f"poison-{index}",
            "initialized_regions": plan.initialized_regions,
            "semantic_digest": "same-logical-result",
        }
        for index, epoch in enumerate(plan.epochs)
    ]


def test_repeat_fixture_reuses_allocations_and_ignores_stale_state():
    plan = repeat_fixture_plan(3)
    observations = _observations(plan)
    validate_repeat_observations(plan, observations)
    assert len({item["scratch_id"] for item in observations}) == 1
    assert len({item["shadow_id"] for item in observations}) == 1
    assert len({item["poison_digest"] for item in observations}) == 3
    assert len({item["semantic_digest"] for item in observations}) == 1


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("fixture_id", "rebuilt"),
        ("scratch_id", "new-scratch"),
        ("shadow_id", "new-shadow"),
        ("seed", 99),
        ("epoch", 999),
        ("poisoned", False),
        ("initialized_regions", ("scratch",)),
        ("semantic_digest", "drift"),
    ],
)
def test_repeat_fixture_rejects_rebuild_stale_init_or_nondeterminism(field, bad):
    plan = repeat_fixture_plan(3)
    observations = _observations(plan)
    observations[1][field] = bad
    with pytest.raises(P1ContractError):
        validate_repeat_observations(plan, observations)


def test_repeat_fixture_requires_distinct_poison_each_time():
    plan = repeat_fixture_plan(3)
    observations = _observations(plan)
    for item in observations:
        item["poison_digest"] = "same-stale-bytes"
    with pytest.raises(P1ContractError, match="distinguishable"):
        validate_repeat_observations(plan, observations)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_independent_whole_layer_reference_schema(version):
    variant = _variant(attention="swa", version=version)
    schema = whole_layer_reference_schema(variant)
    expected_intermediates = set(REFERENCE_INTERMEDIATES)
    if version == "v2":
        expected_intermediates.update(NEXT_K1_INTERMEDIATES)
    assert set(schema["required_intermediates"]) == expected_intermediates
    assert schema["producer_kind"] == "independent_reference"
    assert schema["forbidden_dependencies"] == ["mono_fused", "mono_split"]

    manifest = {
        key: schema[key]
        for key in (
            "schema",
            "version",
            "producer_kind",
            "variant",
            "logical_tp",
            "active_world",
        )
    }
    manifest.update(
        dependencies=["host_reference_math"],
        inputs=schema["required_inputs"],
        intermediates=schema["required_intermediates"],
        tolerances={"rtol": 1.0 / 128.0, "atol": 1.0 / 128.0},
    )
    validate_reference_manifest(manifest, variant)


@pytest.mark.parametrize("dependency", ["mono_fused", "mono_split"])
def test_reference_manifest_rejects_shared_implementation(dependency):
    variant = _variant(version="v2")
    schema = whole_layer_reference_schema(variant)
    manifest = {
        "schema": schema["schema"],
        "version": schema["version"],
        "producer_kind": schema["producer_kind"],
        "variant": schema["variant"],
        "logical_tp": schema["logical_tp"],
        "active_world": schema["active_world"],
        "dependencies": ["host_reference_math", dependency],
        "inputs": schema["required_inputs"],
        "intermediates": schema["required_intermediates"],
        "tolerances": {"rtol": 0.01, "atol": 0.01},
    }
    with pytest.raises(P1ContractError, match="shares Mono"):
        validate_reference_manifest(manifest, variant)


def test_reference_manifest_requires_all_inputs_intermediates_and_tolerances():
    variant = _variant()
    schema = whole_layer_reference_schema(variant)
    base = {
        "schema": schema["schema"],
        "version": schema["version"],
        "producer_kind": schema["producer_kind"],
        "variant": schema["variant"],
        "logical_tp": schema["logical_tp"],
        "active_world": schema["active_world"],
        "dependencies": ["host_reference_math"],
        "inputs": schema["required_inputs"],
        "intermediates": schema["required_intermediates"],
        "tolerances": {"rtol": 0.01, "atol": 0.01},
    }
    for key in ("inputs", "intermediates"):
        broken = dict(base)
        broken[key] = broken[key][:-1]
        with pytest.raises(P1ContractError, match=key[:-1]):
            validate_reference_manifest(broken, variant)
    broken = dict(base)
    broken["tolerances"] = {"rtol": -1, "atol": 0}
    with pytest.raises(P1ContractError, match="tolerances"):
        validate_reference_manifest(broken, variant)
