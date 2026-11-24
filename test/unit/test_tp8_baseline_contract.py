# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

import tp8_baseline_contract as contract  # noqa: E402
from tp8_baseline_production import BaselineContractError, PRODUCTION_ENV  # noqa: E402


def _recipe():
    env = dict(PRODUCTION_ENV)
    env.pop("SGLANG_AMD_MONO_BLOCKS")
    return {
        "local_sources": {
            "repositories": {
                name: {"commit": revision}
                for name, revision in contract.PINNED_REVISIONS.items()
            }
        },
        "server": {
            "env": env,
            "args": dict(contract.REQUIRED_SERVER_ARGS),
        },
    }


def _model_source():
    return "\n".join(
        token
        for entry in contract.CALL_GRAPH.values()
        for token in entry["source_tokens"]
    )


def test_runtime_manifest_is_json_ready_and_explicitly_not_runtime_evidence():
    manifest = contract.build_runtime_baseline_contract(
        _recipe(), model_source=_model_source()
    )
    json.dumps(manifest)
    assert manifest["status"] == "static_contract_only"
    assert manifest["runtime_observed"] is False
    assert manifest["synthetic_target"]["performance_evidence"] is False
    assert manifest["execution_order_contract"]["standalone"]["semantic_order"] == [
        "k1",
        "kattn",
        "k2",
        "k3",
    ]


def test_checked_in_recipe_and_model_build_the_pinned_contract():
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[5]
    manifest = contract.load_runtime_baseline_contract(
        root / "recipe.mi45x.pr50707.tp4.yaml",
        root
        / "sources/mi45x/sglang/python/sglang/srt/rocm_muse_spark/model.py",
    )
    assert manifest["pinned_revisions"] == contract.PINNED_REVISIONS
    assert manifest["recipe"]["topology"] == {"tp_size": 4, "ep_size": 4}


@pytest.mark.parametrize("drift", ["revision", "environment", "argument", "call_graph"])
def test_runtime_manifest_fails_closed_on_pinned_input_drift(drift):
    recipe = _recipe()
    source = _model_source()
    if drift == "revision":
        recipe["local_sources"]["repositories"]["sglang"]["commit"] = "untrusted"
    elif drift == "environment":
        recipe["server"]["env"]["SGLANG_USE_AITER"] = "0"
    elif drift == "argument":
        recipe["server"]["args"]["tp-size"] = 8
    else:
        source = source.replace("run_gfx1250_a8w4", "fallback_moe")
    with pytest.raises(BaselineContractError):
        contract.build_runtime_baseline_contract(recipe, model_source=source)


def test_tp4_schema_is_structural_only_and_forbids_performance_claims():
    schema = contract.TP4_RUNTIME_DISPATCH_SCHEMA
    assert schema["scope"] == "tp4_production_structural_only"
    assert "latency" in schema["prohibited_comparisons"]
    assert "tp8_performance" in schema["prohibited_comparisons"]


def test_execution_order_accepts_stage_and_whole_kernel_contracts():
    standalone = contract.validate_execution_order(
        "standalone", ["k1", "kattn", "k2", "k3"], 4
    )
    layer = contract.validate_execution_order(
        "layer_v2", ["kattn", "k2", "k3", "k1_next"], 1
    )
    assert standalone["status"] == "passed"
    assert layer["launch_count"] == 1
    assert "whole-kernel" in layer["semantics"]


@pytest.mark.parametrize(
    "variant,order,launches",
    [
        ("standalone", ["k1", "k2", "kattn", "k3"], 4),
        ("layer_v1", ["kattn", "k2", "k3"], 3),
    ],
)
def test_execution_order_fails_closed(variant, order, launches):
    with pytest.raises(BaselineContractError):
        contract.validate_execution_order(variant, order, launches)


def test_writer_refuses_a_runtime_claim_from_cpu_manifest(tmp_path):
    manifest = contract.build_runtime_baseline_contract(
        _recipe(), model_source=_model_source()
    )
    manifest["runtime_observed"] = True
    with pytest.raises(BaselineContractError, match="must not claim"):
        contract.write_runtime_baseline_contract(tmp_path / "contract.json", manifest)
