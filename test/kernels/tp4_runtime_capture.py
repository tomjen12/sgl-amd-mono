# SPDX-License-Identifier: Apache-2.0
"""Fail-closed structural capture contracts for the Mono-off TP4 runtime."""

from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tp8_baseline_contract import CALL_GRAPH
from tp8_baseline_production import BaselineContractError

SCHEMA = "muse-tp4-production-runtime-dispatch"
STAGE_ORDER = ("k1", "kattn", "k2", "k3")
ALLOWED_ATTENTION = {"full", "swa"}
ALLOWED_KV_BACKENDS = {"aiter"}


class RuntimeCaptureError(BaselineContractError):
    pass


class StructuralCallSpy:
    """Tiny test-only callable wrapper; it records no tensors or timings."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def wrap(
        self,
        function: Callable[..., Any],
        *,
        stage: str,
        operator: str,
        critical_args: Callable[[tuple[Any, ...], Mapping[str, Any]], Mapping[str, Any]],
    ) -> Callable[..., Any]:
        if stage not in STAGE_ORDER:
            raise RuntimeCaptureError(f"unknown stage {stage!r}")

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            fields = dict(critical_args(args, kwargs))
            self.events.append(
                {
                    "ordinal": len(self.events),
                    "stage": stage,
                    "operator": operator,
                    "critical_args": fields,
                }
            )
            return function(*args, **kwargs)

        return wrapped


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeCaptureError(message)


def compare_baseline_sequence(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare observed per-layer stage order with the Baseline adapter boundary."""
    per_layer: dict[str, list[str]] = {}
    for event in events:
        critical = event.get("critical_args", {})
        layer = critical.get("layer")
        if not isinstance(layer, int):
            return {
                "status": "blocked",
                "reason": "observed event is missing integer layer",
                "per_layer": {},
            }
        rank = event.get("rank")
        key = f"r{rank}.l{layer}" if isinstance(rank, int) else str(layer)
        per_layer.setdefault(key, []).append(str(event.get("stage")))
    mismatches = {
        key: stages
        for key, stages in sorted(per_layer.items())
        if tuple(stages) != STAGE_ORDER
    }
    return {
        "status": "passed" if per_layer and not mismatches else "blocked",
        "expected_per_layer": list(STAGE_ORDER),
        "per_layer": dict(sorted(per_layer.items())),
        "mismatches": mismatches,
    }


def validate_runtime_dispatch(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema") == SCHEMA, "runtime dispatch schema mismatch")
    _require(manifest.get("schema_version") == 1, "runtime dispatch version mismatch")
    _require(manifest.get("scope") == "tp4_production_structural_only", "scope mismatch")
    _require(manifest.get("performance_evidence") is False, "TP4 timing cannot be evidence")
    topology = manifest.get("recipe_topology", {})
    _require(topology == {"tp_size": 4, "ep_size": 4}, "runtime topology mismatch")
    _require(manifest.get("monokernel_enabled") is False, "capture must be Mono-off")
    status = manifest.get("status")
    _require(status in {"passed", "blocked"}, "invalid runtime capture status")
    events = manifest.get("dispatches", [])
    _require(isinstance(events, list), "dispatches must be a list")
    if status == "passed":
        _require(manifest.get("runtime_observed") is True, "passed capture is not runtime-observed")
        _require(bool(events), "passed capture has no observed dispatches")
        _require(manifest.get("call_spy_observed") is True, "passed capture lacks call-spy evidence")
        _require(manifest.get("fallbacks") == [], "production fallback was observed")
        _require(
            manifest.get("baseline_sequence_comparison", {}).get("status") == "passed",
            "Baseline adapter stage sequence mismatch",
        )
        _require(
            manifest.get("fallback_observation_complete") is True,
            "fallback observation is incomplete",
        )
        _require(
            manifest.get("kernel_inventory_observed") is True,
            "kernel inventory was not observed",
        )
        stages = [event.get("stage") for event in events]
        _require(all(stage in STAGE_ORDER for stage in stages), "unknown observed stage")
        for event in events:
            args = event.get("critical_args", {})
            if event["stage"] == "kattn":
                _require(args.get("attention_variant") in ALLOWED_ATTENTION, "attention variant missing")
                _require(args.get("kv_backend") in ALLOWED_KV_BACKENDS, "KV backend mismatch")
                _require(args.get("kv_cache_dtype") == "fp8_e4m3", "KV dtype mismatch")
                _require(args.get("page_size") == 16, "KV page size mismatch")
        _require({"full", "swa"} <= set(manifest.get("attention_variants", [])), "Full/SWA coverage missing")
        _require(manifest.get("first_k1_observed") is True, "first-K1 evidence missing")
        _require(manifest.get("next_k1_observed") is True, "next-K1 evidence missing")
    else:
        _require(bool(manifest.get("blockers")), "blocked capture requires blockers")


def build_runtime_dispatch(
    *,
    events: Sequence[Mapping[str, Any]],
    runtime_observed: bool,
    call_spy_observed: bool,
    attention_variants: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
    fallback_observation_complete: bool = False,
    kernel_inventory_observed: bool = False,
    first_k1_observed: bool = False,
    next_k1_observed: bool = False,
    blockers: Sequence[str] = (),
    runtime_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_rows = [dict(event) for event in events]
    sequence_comparison = compare_baseline_sequence(event_rows)
    complete = (
        runtime_observed
        and call_spy_observed
        and bool(event_rows)
        and sequence_comparison["status"] == "passed"
        and fallback_observation_complete
        and kernel_inventory_observed
        and not fallbacks
        and {"full", "swa"} <= set(attention_variants)
        and first_k1_observed
        and next_k1_observed
    )
    effective_blockers = list(blockers)
    if not complete and not effective_blockers:
        effective_blockers.append("runtime call-spy evidence is incomplete")
    operators = Counter(str(event.get("operator")) for event in event_rows)
    stages = Counter(str(event.get("stage")) for event in event_rows)
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "passed" if complete else "blocked",
        "runtime_observed": runtime_observed,
        "call_spy_observed": call_spy_observed,
        "scope": "tp4_production_structural_only",
        "warning": "TP4 structural evidence only; not TP8 correctness or performance evidence",
        "performance_evidence": False,
        "recipe_topology": {"tp_size": 4, "ep_size": 4},
        "monokernel_enabled": False,
        "dispatches": event_rows,
        "operator_inventory": dict(sorted(operators.items())),
        "semantic_stage_inventory": dict(sorted(stages.items())),
        "baseline_adapter_call_graph": CALL_GRAPH,
        "baseline_sequence_comparison": sequence_comparison,
        "attention_variants": sorted(set(attention_variants)),
        "fallbacks": list(fallbacks),
        "fallback_observation_complete": fallback_observation_complete,
        "kernel_inventory_observed": kernel_inventory_observed,
        "first_k1_observed": first_k1_observed,
        "next_k1_observed": next_k1_observed,
        "blockers": effective_blockers,
        "runtime_evidence": dict(runtime_evidence or {}),
    }
    manifest["dispatch_hash"] = hashlib.sha256(
        json.dumps(event_rows, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    validate_runtime_dispatch(manifest)
    return manifest


def inspect_replay_bundle(root: str | Path) -> dict[str, Any]:
    """Validate the four-rank eager replay metadata emitted by production."""
    bundle = Path(root)
    rank_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, int]] = []
    full_layers: set[int] = set()
    swa_layers: set[int] = set()
    for rank in range(4):
        rank_dir = bundle / f"rank_{rank:04d}"
        try:
            manifest = json.loads((rank_dir / "manifest.json").read_text())
            verify = json.loads((rank_dir / "target_verify.json").read_text())
            layers = json.loads((rank_dir / "target_layers.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeCaptureError(f"rank {rank}: replay metadata missing or invalid") from exc
        _require(manifest.get("complete") is True, f"rank {rank}: replay capture incomplete")
        _require(
            manifest.get("metadata", {}).get("execution") == "eager",
            f"rank {rank}: replay capture was not eager",
        )
        cache = verify.get("cache", {})
        for cache_name, destination in (("kv.full", full_layers), ("kv.swa", swa_layers)):
            item = cache.get(cache_name)
            _require(isinstance(item, Mapping), f"rank {rank}: {cache_name} metadata missing")
            _require(item.get("page_size") == 16, f"rank {rank}: {cache_name} page size mismatch")
            _require(item.get("layout") == "nhd", f"rank {rank}: {cache_name} layout mismatch")
            for layer, values in item.get("layers", {}).items():
                destination.add(int(layer))
                for kind in ("k", "v"):
                    descriptor = values[kind]
                    _require(
                        descriptor.get("dtype") == "torch.float8_e4m3fn",
                        f"rank {rank}: {cache_name} {kind} dtype mismatch",
                    )
        starts = set()
        local_counts = set()
        observed_layers = []
        for key, value in layers.items():
            if not key.endswith(".routing"):
                continue
            layer = int(key.split("_", 1)[1].split(".", 1)[0])
            observed_layers.append(layer)
            starts.add(int(value["local_expert_start"]))
            local_counts.add(int(value["local_experts"]))
            _require(value.get("num_experts") == 256, f"rank {rank}: global expert count mismatch")
            _require(value.get("top_k") == 8, f"rank {rank}: top-k mismatch")
        _require(starts == {rank * 64}, f"rank {rank}: TP4 ownership start mismatch")
        _require(local_counts == {64}, f"rank {rank}: TP4 local expert count mismatch")
        mapping_rows.extend(
            {
                "global_expert": expert,
                "owner": rank,
                "local_expert": expert - rank * 64,
            }
            for expert in range(rank * 64, (rank + 1) * 64)
        )
        rank_rows.append(
            {
                "rank": rank,
                "capture_complete": True,
                "execution": "eager",
                "captured_target_layers": sorted(observed_layers),
                "local_expert_start": rank * 64,
                "local_experts": 64,
            }
        )
    mapping_hash = hashlib.sha256(
        json.dumps(mapping_rows, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "replay_bundle": str(bundle),
        "ranks": rank_rows,
        "attention_variants": {
            "full_layers": sorted(full_layers),
            "swa_layers": sorted(swa_layers),
        },
        "kv_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3",
        "kv_layout": "nhd",
        "page_size": 16,
        "tp4_runtime_mapping": mapping_rows,
        "tp4_runtime_mapping_hash": mapping_hash,
        "warning": "TP4 ownership is not a TP8 ownership manifest",
    }


def inspect_call_spy(root: str | Path) -> dict[str, Any]:
    """Load per-worker semantic events and verify a complete 62-layer TP4 call graph."""
    directory = Path(root)
    files = sorted(directory.glob("pid_*.jsonl"))
    _require(bool(files), "structural call-spy produced no event files")
    events: list[dict[str, Any]] = []
    rank_files: dict[int, str] = {}
    unexpected_fallbacks: list[str] = []
    for path in files:
        process_events = []
        try:
            for line in path.read_text().splitlines():
                if line:
                    process_events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeCaptureError(f"{path.name}: invalid call-spy JSONL") from exc
        if not process_events:
            continue
        first_target = next(
            (index for index, event in enumerate(process_events) if event.get("stage") == "k1"),
            None,
        )
        _require(first_target is not None, f"{path.name}: target K1 was not observed")
        # RadixAttention is shared with the draft model; draft-only events can
        # precede the first target K1 while draft metadata capture is active.
        process_events = process_events[first_target:]
        starts = {
            event.get("critical_args", {}).get("local_expert_start")
            for event in process_events
            if event.get("stage") in {"k2", "k3"}
        }
        _require(len(starts) == 1, f"{path.name}: inconsistent ownership observations")
        start = starts.pop()
        _require(
            isinstance(start, int) and start in {0, 64, 128, 192},
            f"{path.name}: invalid TP4 ownership start",
        )
        rank = start // 64
        _require(rank not in rank_files, f"rank {rank}: duplicate call-spy worker")
        rank_files[rank] = path.name
        for event in process_events:
            event = dict(event)
            event["rank"] = rank
            critical = dict(event.get("critical_args", {}))
            event["critical_args"] = critical
            if event.get("stage") == "kattn":
                backend = str(critical.get("kv_backend", "")).lower()
                critical["kv_backend_class"] = critical.get("kv_backend")
                critical["kv_backend"] = "aiter" if "aiter" in backend else backend
            if event.get("stage") == "k3" and event.get("operator") == "production_expert_fallback":
                layer = critical.get("layer")
                if layer in (0, 61):
                    event["operator"] = "tpx_bf16_or_native_expert"
                    critical["expected_bf16_boundary"] = True
                else:
                    unexpected_fallbacks.append(f"rank {rank} layer {layer}: expert fallback")
            if event.get("stage") == "k1" and critical.get("fused_kv_write") is not True:
                unexpected_fallbacks.append(
                    f"rank {rank} layer {critical.get('layer')}: fused K1 fallback"
                )
            events.append(event)
    _require(set(rank_files) == {0, 1, 2, 3}, "call-spy did not cover all four ranks")
    comparison = compare_baseline_sequence(events)
    _require(comparison["status"] == "passed", "call-spy stage sequence mismatch")
    layers_by_rank = {
        rank: {
            int(event["critical_args"]["layer"])
            for event in events
            if event["rank"] == rank
        }
        for rank in range(4)
    }
    _require(
        all(layers == set(range(62)) for layers in layers_by_rank.values()),
        "call-spy did not cover all 62 layers",
    )
    variants = {
        event["critical_args"]["attention_variant"]
        for event in events
        if event["stage"] == "kattn"
    }
    _require(variants == ALLOWED_ATTENTION, "call-spy Full/SWA coverage mismatch")
    packed_rows = [
        event
        for event in events
        if event["stage"] == "k3" and event["operator"] == "run_gfx1250_a8w4"
    ]
    _require(len(packed_rows) == 4 * 60, "packed MXFP4 layer coverage mismatch")
    packed_shapes: dict[str, set[tuple[int, ...]]] = {
        name: set() for name in ("w13", "w2", "s13", "s2")
    }
    for event in packed_rows:
        metadata = event["critical_args"].get("packed_metadata", {})
        _require(
            set(metadata) == set(packed_shapes),
            "packed MXFP4 tensor metadata missing",
        )
        for name, item in metadata.items():
            _require(item.get("address_mod_256") == 0, f"{name}: packed address alignment mismatch")
            packed_shapes[name].add(tuple(item.get("shape", ())))
            if name in ("w13", "w2"):
                _require(
                    item.get("dtype") == "torch.float4_e2m1fn_x2",
                    f"{name}: packed dtype mismatch",
                )
                _require(item.get("is_shuffled") is True, f"{name}: shuffled marker missing")
            else:
                _require(item.get("dtype") == "torch.uint8", f"{name}: scale dtype mismatch")
    return {
        "events": events,
        "rank_files": rank_files,
        "sequence_comparison": comparison,
        "unexpected_fallbacks": unexpected_fallbacks,
        "attention_variants": sorted(variants),
        "first_k1_observed": all(
            next(
                event for event in events if event["rank"] == rank
            )["stage"]
            == "k1"
            for rank in range(4)
        ),
        "next_k1_observed": all(
            any(
                event["stage"] == "k1"
                and event["critical_args"]["layer"] == 61
                for event in events
                if event["rank"] == rank
            )
            for rank in range(4)
        ),
        "packed_metadata_summary": {
            "observations": len(packed_rows),
            "all_addresses_256_aligned": True,
            "shapes": {
                name: [list(shape) for shape in sorted(shapes)]
                for name, shapes in packed_shapes.items()
            },
            "weight_dtype": "torch.float4_e2m1fn_x2",
            "scale_dtype": "torch.uint8",
            "weights_shuffled": True,
        },
    }


def write_runtime_dispatch(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_runtime_dispatch(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
