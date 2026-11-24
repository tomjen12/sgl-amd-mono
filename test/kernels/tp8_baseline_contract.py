# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the TP8 synthetic production Baseline.

The manifest built here is static attestation from the pinned recipe and model
call graph.  It is deliberately not runtime dispatch evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp8_baseline_dispatch import expected_manifest
from tp8_baseline_production import (
    BaselineContractError,
    PRODUCTION_ENV,
    load_recipe_environment,
)
from tp8_synthetic_contract import CONTRACT


PINNED_REVISIONS = {
    "sglang": "db52385b8a86d24db693a794ed980bd759313a3b",
    "sgl_amd_mono": "a716d4d4d2f58001a221b7497cbd25d27dcaa9f1",
    "aiter": "afc300d7fee1b38cd805da517414c42114b35968",
}

REQUIRED_SERVER_ARGS = {
    "tp-size": 4,
    "ep-size": 4,
    "attention-backend": "aiter",
    "page-size": 16,
    "kv-cache-dtype": "fp8_e4m3",
}

CALL_GRAPH = {
    "k1": {
        "callsite": "MuseSpark1xAttention.forward",
        "sequence": [
            "qkv_proj_norm_rope_kv_write",
            "qkv_proj_qk_norm_rope_cache",
        ],
        "source_tokens": ["qkv_proj_norm_rope_kv_write"],
    },
    "kattn": {
        "callsite": "MuseSpark1xAttention.forward",
        "sequence": [
            "attention.attn",
            "aiter target-verify",
            "VERIFY_REDUCE_DEFERRAL.take",
            "forward_verify_partials",
        ],
        "source_tokens": ["VERIFY_REDUCE_DEFERRAL.take", "forward_verify_partials"],
    },
    "k2": {
        "callsite": "MuseSpark1xDecoderLayer.forward",
        "sequence": [
            "self_attn.o_proj",
            "tensor_model_parallel_all_reduce",
            "_fused_attn_boundary.forward",
            "route_with_shared_pooler",
        ],
        "source_tokens": [
            "tensor_model_parallel_all_reduce",
            "_fused_attn_boundary.forward",
            "route_with_shared_pooler",
        ],
    },
    "k3": {
        "callsite": "MuseSpark1xDecoderLayer.forward -> MuseSpark1xMoE.forward",
        "sequence": [
            "run_gfx1250_a8w4",
            "tensor_model_parallel_all_reduce",
            "unpool_gemm_norm_residual",
        ],
        "source_tokens": [
            "run_gfx1250_a8w4",
            "tensor_model_parallel_all_reduce",
            "unpool_gemm_norm_residual",
        ],
    },
}

EXECUTION_ORDER_CONTRACT = {
    "standalone": {
        "semantic_order": ["k1", "kattn", "k2", "k3"],
        "launch_semantics": "one validated production stage at a time",
    },
    "layer_v1": {
        "semantic_order": ["kattn", "k2", "k3"],
        "launch_semantics": "one whole-kernel launch; K1 was produced by bootstrap/previous layer",
        "whole_kernel_launch_count": 1,
        "next_k1": False,
    },
    "layer_v2": {
        "semantic_order": ["kattn", "k2", "k3", "k1_next"],
        "launch_semantics": "one whole-kernel launch including exactly one next-layer K1",
        "whole_kernel_launch_count": 1,
        "next_k1": True,
    },
}

TP4_RUNTIME_DISPATCH_SCHEMA = {
    "schema_version": 1,
    "evidence_kind": "runtime_profile",
    "scope": "tp4_production_structural_only",
    "warning": "structural-only; not TP8 performance or TP8 runtime evidence",
    "permitted_comparisons": [
        "stage_order",
        "attention_variant",
        "operator_backend_family",
        "kernel_semantic_order",
        "launch_count",
        "fallback_absence",
    ],
    "prohibited_comparisons": [
        "latency",
        "throughput",
        "shape-specific_kernel_name",
        "tp8_correctness",
        "tp8_performance",
    ],
    "required_fields": [
        "status",
        "runtime_observed",
        "recipe_topology",
        "dispatches",
    ],
}


def _tensor_schema() -> dict[str, list[dict[str, Any]]]:
    rows = [8, 16]
    common = {
        "rows": rows,
        "layout": "row_major",
        "stride": "contiguous unless explicitly marked",
        "alignment": "production allocation; runtime value must be recorded",
    }
    return {
        "k1": [
            {**common, "name": "x", "shape": ["M", 8192], "dtype": "bfloat16"},
            {
                **common,
                "name": "weight",
                "shape": [2304, 8192],
                "dtype": "bfloat16",
                "packing": "TP8-local Q/K/V/O-gate rows",
            },
            {
                **common,
                "name": "kv_cache",
                "shape": ["slots", 2, 64],
                "dtype": "float8_e4m3fn",
                "cache_format": "page_size=16; separate K/V tensors",
                "quantization": "per-cache K/V descale",
            },
        ],
        "kattn": [
            {**common, "name": "q", "shape": ["M", 1024], "dtype": "bfloat16"},
            {**common, "name": "k/v", "shape": ["M", 128], "dtype": "bfloat16"},
            {
                **common,
                "name": "paged_kv",
                "shape": [2, "blocks", 16, 2, 64],
                "dtype": "float8_e4m3fn",
                "cache_format": "AITER paged KV; Full or SWA selected explicitly",
                "quantization": "K/V descale tensors",
            },
        ],
        "k2": [
            {
                **common,
                "name": "attention_output",
                "shape": ["M", 1024],
                "dtype": "bfloat16",
            },
            {
                **common,
                "name": "stream/projected/ffn_input",
                "shape": ["M", 8192],
                "dtype": "bfloat16",
                "packing": "joint router/pooler production packed weight",
            },
        ],
        "k3": [
            {**common, "name": "expert_input", "shape": ["M", 4096], "dtype": "bfloat16"},
            {**common, "name": "ids", "shape": ["M", 8], "dtype": "int32"},
            {**common, "name": "scores", "shape": ["M", 8], "dtype": "float32"},
            {
                **common,
                "name": "expert_weights",
                "shape": [32, "production packed"],
                "dtype": "MXFP4",
                "packing": "AITER gfx1250 a8w4 TDM",
                "quantization": "MXFP4 weights with production scale tensors",
            },
        ],
    }


def validate_execution_order(variant: str, actual: Sequence[str], launch_count: int) -> dict:
    if variant not in EXECUTION_ORDER_CONTRACT:
        raise BaselineContractError(f"unknown execution variant {variant!r}")
    expected = EXECUTION_ORDER_CONTRACT[variant]
    order = list(actual)
    if order != expected["semantic_order"]:
        raise BaselineContractError(
            f"{variant}: execution order {order}, expected {expected['semantic_order']}"
        )
    expected_launches = expected.get("whole_kernel_launch_count", len(order))
    if launch_count != expected_launches:
        raise BaselineContractError(
            f"{variant}: launch count {launch_count}, expected {expected_launches}"
        )
    return {
        "status": "passed",
        "variant": variant,
        "semantic_order": order,
        "launch_count": launch_count,
        "semantics": expected["launch_semantics"],
    }


def _recipe_revisions(recipe: Mapping[str, Any]) -> dict[str, str]:
    try:
        repositories = recipe["local_sources"]["repositories"]
        return {
            name: str(repositories[name]["commit"])
            for name in PINNED_REVISIONS
        }
    except (KeyError, TypeError) as exc:
        raise BaselineContractError("recipe is missing pinned repository commits") from exc


def build_runtime_baseline_contract(
    recipe: Mapping[str, Any],
    *,
    model_source: str,
) -> dict[str, Any]:
    """Build a JSON-ready static manifest, failing closed on pin/call-graph drift."""
    revisions = _recipe_revisions(recipe)
    if revisions != PINNED_REVISIONS:
        raise BaselineContractError(
            f"recipe revision drift: {revisions}, expected {PINNED_REVISIONS}"
        )
    try:
        env = {str(k): str(v) for k, v in recipe["server"]["env"].items()}
        args = recipe["server"]["args"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise BaselineContractError("recipe is missing server env/args") from exc
    for name, value in PRODUCTION_ENV.items():
        if name == "SGLANG_AMD_MONO_BLOCKS":
            continue
        if env.get(name) != value:
            raise BaselineContractError(
                f"recipe {name}={env.get(name)!r}, expected {value!r}"
            )
    for name, value in REQUIRED_SERVER_ARGS.items():
        if args.get(name) != value:
            raise BaselineContractError(
                f"recipe server arg {name}={args.get(name)!r}, expected {value!r}"
            )
    for stage, entry in CALL_GRAPH.items():
        missing = [token for token in entry["source_tokens"] if token not in model_source]
        if missing:
            raise BaselineContractError(
                f"production call graph {stage} is missing source tokens {missing}"
            )
    return {
        "schema_version": 1,
        "status": "static_contract_only",
        "runtime_observed": False,
        "warning": "CPU-generated manifest; actual dispatch requires a GPU/runtime profile",
        "pinned_revisions": revisions,
        "recipe": {
            "topology": {"tp_size": 4, "ep_size": 4},
            "server_args": dict(args),
            "dispatch_sensitive_server_args": {
                name: args[name] for name in REQUIRED_SERVER_ARGS
            },
            "dispatch_sensitive_environment": env,
        },
        "synthetic_target": {
            "topology": {"logical_tp": 8, "logical_ep": 8, "active_npes": 4},
            "shape_contract": asdict(CONTRACT),
            "performance_evidence": False,
        },
        "production_call_graph": CALL_GRAPH,
        "dispatch_contract": expected_manifest(),
        "tensor_contract": _tensor_schema(),
        "execution_order_contract": EXECUTION_ORDER_CONTRACT,
        "tp4_runtime_dispatch_schema": TP4_RUNTIME_DISPATCH_SCHEMA,
    }


def load_runtime_baseline_contract(
    recipe_path: str | Path, model_path: str | Path
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - production image has PyYAML
        raise BaselineContractError("PyYAML is required to load the pinned recipe") from exc
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    if not isinstance(recipe, Mapping):
        raise BaselineContractError("pinned recipe is not a mapping")
    # Exercise the same environment loader used by the production adapter.
    load_recipe_environment(recipe_path)
    return build_runtime_baseline_contract(
        recipe, model_source=Path(model_path).read_text()
    )


def write_runtime_baseline_contract(path: str | Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "static_contract_only":
        raise BaselineContractError("refusing to write an invalid baseline contract")
    if manifest.get("runtime_observed") is not False:
        raise BaselineContractError("CPU contract must not claim runtime observation")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
