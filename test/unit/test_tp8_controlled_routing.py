# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for controlled-routing contract v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_controlled_routing import (  # noqa: E402
    GLOBAL_EXPERTS,
    MOE_HIDDEN,
    ControlledRoutingError,
    build_package,
    canonical_global_ids,
    load_package,
    map_routing,
    normalized_scores,
    validate_package,
    validate_runtime_mapping,
)


def _xin(width: int):
    return tuple(
        tuple(((token * 17 + column * 3) % 257 - 128) / 128.0 for column in range(MOE_HIDDEN))
        for token in range(width)
    )


def _package(width: int = 8):
    return build_package(_xin(width), normalized_scores(width), width=width)


def _mapping(world: int, *, permuted: bool = False):
    per_rank = GLOBAL_EXPERTS // world
    entries = []
    for global_expert in range(GLOBAL_EXPERTS):
        owner = global_expert // per_rank
        local = global_expert % per_rank
        if permuted:
            local = (local * 17) % per_rank
        entries.append(
            {
                "global_expert": global_expert,
                "owner": owner,
                "local_expert": local,
            }
        )
    return entries


@pytest.mark.parametrize("width", [8, 16])
def test_package_round_trip_and_canonical_formula(tmp_path, width):
    package = _package(width)
    assert package.ids == canonical_global_ids(width)
    path = tmp_path / "gb300-routing.json"
    path.write_text(json.dumps(package.as_dict()))
    loaded = load_package(path)
    assert loaded == package
    assert set(loaded.hashes) == {
        "token_order_hash",
        "logical_xin_hash",
        "global_ids_hash",
        "scores_hash",
    }


def test_missing_package_is_blocked_without_local_fallback(tmp_path):
    with pytest.raises(ControlledRoutingError, match="unavailable"):
        load_package(tmp_path / "missing-gb300-package.json")


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("schema", "wrong", "schema"),
        ("version", 2, "version"),
        ("producer", "mi455", "producer"),
        ("width", 4, "width"),
        ("top_k", 4, "Top-K"),
        ("global_experts", 128, "expert count"),
        ("moe_hidden", 2048, "hidden"),
        ("rotation", 1, "rotation"),
    ],
)
def test_package_rejects_wrong_contract_fields(field, bad, message):
    document = _package().as_dict()
    document[field] = bad
    with pytest.raises(ControlledRoutingError, match=message):
        validate_package(document)


def test_package_rejects_remapped_ids_scores_xin_and_hashes():
    for mutate, message in (
        (lambda doc: doc["global_ids"][0].__setitem__(0, 1), "canonical"),
        (lambda doc: doc["scores"][0].__setitem__(0, float("nan")), "finite"),
        (lambda doc: doc["logical_xin"][0].pop(), "shape"),
        (lambda doc: doc["canonical_hashes"].__setitem__("scores_hash", "0" * 64), "hash"),
    ):
        document = json.loads(json.dumps(_package().as_dict()))
        mutate(document)
        with pytest.raises(ControlledRoutingError, match=message):
            validate_package(document)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("permuted", [False, True])
def test_platform_mapping_is_runtime_derived(world, permuted):
    package = _package()
    result = map_routing(
        package,
        _mapping(world, permuted=permuted),
        world_size=world,
        packed_metadata=b"production-packer-output",
    )
    expected_picks = package.width * 8 // world
    assert result["per_rank_picks"] == (expected_picks,) * world
    assert result["active_experts"] == (expected_picks,) * world
    assert len(result["runtime_mapping_hash"]) == 64
    assert len(result["local_ids_hash"]) == 64
    assert len(result["packed_metadata_hash"]) == 64


def test_mapping_rejects_missing_duplicate_and_wrong_ownership():
    valid = _mapping(8)
    with pytest.raises(ControlledRoutingError, match="cover"):
        validate_runtime_mapping(valid[:-1], world_size=8)
    duplicate = [dict(item) for item in valid]
    duplicate[-1]["global_expert"] = 0
    with pytest.raises(ControlledRoutingError, match="bijection"):
        validate_runtime_mapping(duplicate, world_size=8)
    wrong_count = [dict(item) for item in valid]
    wrong_count[0]["owner"] = 1
    with pytest.raises(ControlledRoutingError, match="wrong expert count"):
        validate_runtime_mapping(wrong_count, world_size=8)
