#!/usr/bin/env python3
"""Report Target Baseline, split Mono, and fused Mono GPU kernel gaps."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from trace_fingerprint import (
    CPU_CATEGORIES,
    KERNEL_CATEGORIES,
    RUNTIME_CATEGORIES,
    _arg,
    _category,
    _complete_events,
    _correlation_id,
    _duration,
    _external_id,
    _inside,
    _load,
    _ordered,
    normalize_name,
)

STAGES = ("K1", "K-attn", "K2", "K3")
SPLIT_STAGES = ("K1", "K-attn-full", "K-attn-window", "K2", "K3")
FAMILIES = ("K1", "Layer V2", "Layer V1")
TARGET_LAYERS = 10
ELIGIBLE_LAYERS = range(1, TARGET_LAYERS)
FULL_ATTENTION_LAYERS = frozenset((1, 5, 9))

BASELINE_LAYOUT = {
    "K1": ("_dense_partial.kd", "_qkv_epilogue.kd"),
    "K-attn": (("_vv2_partials.kd", "_swa_verify_partials.kd"), "_reduce_norm_gate.kd"),
    "K2": (
        "_gemm_a16w16_gfx1250_bandwidth_bound_kernel_",
        "all_reduce_1shot_push_kernel",
        "_fused_norm_residual_triton_kernel.kd",
        "kernel_gemm_a16w16_0.kd",
        "_joint_epilogue.kd",
    ),
    "K3": (
        "_local_expert_ids_kernel.kd",
        "moe_route_psum_fused.kd",
        "moe_fused_quant_preshuffle_routeks_",
        "_act1_q1r4_mg10.kd",
        "_K4096_e64_mg10.kd",
        "_finalize_rows_kernel.kd",
        "all_reduce_1shot_push_kernel",
        "_dense_partial.kd",
        "_unpool_reduce_boundary.kd",
    ),
}

MONO_NEEDLES = {
    "K1": "muse_spark_gfx1250_k1_",
    "K-attn": "muse_spark_gfx1250_kattn_",
    "K2": "muse_spark_gfx1250_k2_",
    "K3": "muse_spark_gfx1250_k3_",
    "Layer": "muse_spark_gfx1250_layer_",
}


def _scope_kernels(trace: dict[str, Any], pattern: str) -> list[list[dict[str, Any]]]:
    """Return kernels correlated to each matching profiler scope."""
    events = _complete_events(trace)
    scope_re = re.compile(pattern)
    scopes = _ordered(
        event
        for event in events
        if _category(event) in CPU_CATEGORIES
        and scope_re.search(str(event.get("name", "")))
    )
    result = []
    for scope in scopes:
        parents: dict[str, list[str]] = defaultdict(list)
        for event in events:
            if (
                event is scope
                or _category(event) not in CPU_CATEGORIES
                or not _inside(event, scope)
            ):
                continue
            external_id = _external_id(event)
            if external_id is not None:
                parents[external_id].append(normalize_name(event.get("name", "")))

        launches: dict[str, str | None] = {}
        for event in events:
            if _category(event) not in RUNTIME_CATEGORIES or not _inside(event, scope):
                continue
            correlation = _correlation_id(event)
            if correlation is None or "launch" not in str(event.get("name", "")).lower():
                continue
            cpu_id = _arg(event.get("args") or {}, "cid", "external id")
            launches[correlation] = (
                parents.get(str(cpu_id), [])[-1]
                if cpu_id is not None and str(cpu_id) in parents
                else None
            )

        result.append(
            _ordered(
                event
                for event in events
                if _category(event) in KERNEL_CATEGORIES
                and (
                    (_correlation_id(event) in launches if launches else _inside(event, scope))
                    or (
                        _external_id(event) is not None
                        and _external_id(event) in parents
                    )
                )
            )
        )
    return result


def _matches(name: str, expected: str | tuple[str, ...]) -> bool:
    if isinstance(expected, tuple):
        return any(needle in name for needle in expected)
    return expected in name


def _baseline_relevant(kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep Target layer kernels when graph replay has no Python layer scopes."""
    needles = tuple(
        needle
        for expected in BASELINE_LAYOUT.values()
        for needle in expected
    )
    return [
        event
        for event in kernels
        if any(
            _matches(normalize_name(event.get("name", "")), needle)
            for needle in needles
        )
    ]


def _baseline_layer(kernels: list[dict[str, Any]], layer: int) -> dict[str, float]:
    if len(kernels) != 18:
        raise ValueError(f"baseline layer {layer}: expected 18 launches, got {len(kernels)}")
    buckets: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    dense_seen = all_reduce_seen = 0
    for event in kernels:
        name = normalize_name(event.get("name", ""))
        if "_dense_partial.kd" in name:
            stage = "K1" if dense_seen == 0 else "K3"
            dense_seen += 1
        elif "all_reduce_1shot_push_kernel" in name:
            stage = "K2" if all_reduce_seen == 0 else "K3"
            all_reduce_seen += 1
        else:
            stage = next(
                (
                    candidate
                    for candidate, expected in BASELINE_LAYOUT.items()
                    if any(
                        _matches(name, needle)
                        for needle in expected
                        if needle
                        not in ("_dense_partial.kd", "all_reduce_1shot_push_kernel")
                    )
                ),
                None,
            )
            if stage is None:
                raise ValueError(f"baseline layer {layer}: unclassified kernel {name}")
        buckets[stage].append(event)

    expected_counts = {stage: len(BASELINE_LAYOUT[stage]) for stage in STAGES}
    got_counts = {stage: len(events) for stage, events in buckets.items()}
    if got_counts != expected_counts:
        raise ValueError(
            f"baseline layer {layer}: bucket counts {got_counts}, "
            f"expected {expected_counts}"
        )
    return {
        stage: sum(_duration(event) for event in buckets[stage]) for stage in STAGES
    }


def _mono_relevant(kernels: list[dict[str, Any]]) -> list[tuple[str, float, str]]:
    relevant = []
    for event in kernels:
        name = normalize_name(event.get("name", ""))
        family = next(
            (label for label, needle in MONO_NEEDLES.items() if needle in name),
            None,
        )
        if family is not None:
            relevant.append((family, _duration(event), name))
    return relevant


def _split_iteration(kernels: list[dict[str, Any]]) -> list[dict[str, float]]:
    relevant = _mono_relevant(kernels)
    expected = ["K1", "K-attn", "K2", "K3"] * 9
    got = [family for family, _, _ in relevant]
    if got != expected:
        raise ValueError(
            f"split launch map mismatch: got counts={_counts(got)}, "
            f"expected counts={_counts(expected)}"
        )
    layers = []
    cursor = 0
    for _ in ELIGIBLE_LAYERS:
        values = {}
        for stage, count in (("K1", 1), ("K-attn", 1), ("K2", 1), ("K3", 1)):
            values[stage] = sum(value for _, value, _ in relevant[cursor : cursor + count])
            cursor += count
        layers.append(values)
    return layers


def _fused_iteration(kernels: list[dict[str, Any]]) -> dict[str, list[float]]:
    relevant = _mono_relevant(kernels)
    got = [family for family, _, _ in relevant]
    expected = ["K1", *(["Layer"] * 9)]
    if got != expected:
        raise ValueError(f"fused launch map mismatch: got {got}, expected {expected}")
    layer_names = [name for _, _, name in relevant[1:]]
    if "_n1_" not in layer_names[0] or any("_n1_" not in name for name in layer_names[:8]):
        raise ValueError("fused Layer V2 names do not carry next-K1 (_n1_)")
    if "_n0_" not in layer_names[-1]:
        raise ValueError("fused Layer V1 name does not carry no-next-K1 (_n0_)")
    return {
        "K1": [relevant[0][1]],
        "Layer V2": [value for _, value, _ in relevant[1:9]],
        "Layer V1": [relevant[9][1]],
    }


def _counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def _trace_paths(directory: Path) -> list[Path]:
    paths = sorted((directory / "traces").glob("target.rank_*.trace.json.gz"))
    if len(paths) != 4:
        raise ValueError(f"{directory}: expected four rank traces, found {len(paths)}")
    return paths


def _rank(path: Path) -> int:
    match = re.search(r"rank_(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot read rank from {path}")
    return int(match.group(1))


def _baseline_samples(directory: Path):
    labels = (*STAGES, "K-attn-full", "K-attn-window")
    by_rank = {rank: {stage: [] for stage in labels} for rank in range(4)}
    per_iteration = {rank: [] for rank in range(4)}
    iterations = None
    for path in _trace_paths(directory):
        rank = _rank(path)
        trace = _load(str(path))
        scopes = _scope_kernels(trace, r"^muse_replay/target/layer_[0-9]+$")
        graph_scopes = []
        if not scopes:
            graph_scopes = _scope_kernels(trace, r"^muse_replay/target/verify$")
            for index, kernels in enumerate(graph_scopes):
                relevant = _baseline_relevant(kernels)
                expected = 17 + len(ELIGIBLE_LAYERS) * 18
                if len(relevant) != expected:
                    raise ValueError(
                        f"{path}: graph iteration {index} has {len(relevant)} "
                        f"Target layer kernels, expected {expected}"
                    )
                scopes.extend(
                    [
                        relevant[:17],
                        *[
                            relevant[17 + offset * 18 : 17 + (offset + 1) * 18]
                            for offset in range(len(ELIGIBLE_LAYERS))
                        ],
                    ]
                )
        if len(scopes) % TARGET_LAYERS:
            raise ValueError(
                f"{path}: {len(scopes)} layer scopes is not divisible by 10"
            )
        current_iterations = (
            len(graph_scopes) if graph_scopes else len(scopes) // TARGET_LAYERS
        )
        iterations = current_iterations if iterations is None else iterations
        if current_iterations != iterations:
            raise ValueError(f"{path}: iteration mismatch")
        for iteration in range(current_iterations):
            layers = scopes[
                iteration * TARGET_LAYERS : (iteration + 1) * TARGET_LAYERS
            ]
            if len(layers[0]) != 17:
                raise ValueError(
                    f"baseline layer 0: expected 17 fallback launches, got {len(layers[0])}"
                )
            total = 0.0
            for layer in ELIGIBLE_LAYERS:
                values = _baseline_layer(layers[layer], layer)
                for stage, duration in values.items():
                    by_rank[rank][stage].append(duration)
                    if stage == "K-attn":
                        attention_stage = (
                            "K-attn-full"
                            if layer in FULL_ATTENTION_LAYERS
                            else "K-attn-window"
                        )
                        by_rank[rank][attention_stage].append(duration)
                    total += duration
            per_iteration[rank].append(total)
    return by_rank, per_iteration, int(iterations or 0)


def _mono_samples(directory: Path, mode: str):
    labels = (*STAGES, "K-attn-full", "K-attn-window") if mode == "split" else FAMILIES
    by_rank = {rank: {label: [] for label in labels} for rank in range(4)}
    per_iteration = {rank: [] for rank in range(4)}
    iterations = None
    parser: Callable[[list[dict[str, Any]]], Any] = (
        _split_iteration if mode == "split" else _fused_iteration
    )
    for path in _trace_paths(directory):
        rank = _rank(path)
        scopes = _scope_kernels(_load(str(path)), r"^muse_replay/target/verify$")
        current_iterations = len(scopes)
        iterations = current_iterations if iterations is None else iterations
        if current_iterations != iterations:
            raise ValueError(f"{path}: iteration mismatch")
        for kernels in scopes:
            parsed = parser(kernels)
            total = 0.0
            if mode == "split":
                for layer_id, layer in zip(ELIGIBLE_LAYERS, parsed):
                    for stage, duration in layer.items():
                        by_rank[rank][stage].append(duration)
                        if stage == "K-attn":
                            attention_stage = (
                                "K-attn-full"
                                if layer_id in FULL_ATTENTION_LAYERS
                                else "K-attn-window"
                            )
                            by_rank[rank][attention_stage].append(duration)
                        total += duration
            else:
                for family, durations in parsed.items():
                    by_rank[rank][family].extend(durations)
                    total += sum(durations)
            per_iteration[rank].append(total)
    return by_rank, per_iteration, int(iterations or 0)


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def _critical(by_rank: dict[int, dict[str, list[float]]], label: str) -> list[float]:
    samples = len(by_rank[0][label])
    if any(len(by_rank[rank][label]) != samples for rank in range(4)):
        raise ValueError(f"{label}: rank sample count mismatch")
    return [max(by_rank[rank][label][sample] for rank in range(4)) for sample in range(samples)]


def _selected(
    by_rank: dict[int, dict[str, list[float]]], label: str, rank: int | None
) -> list[float]:
    return _critical(by_rank, label) if rank is None else by_rank[rank][label]


def _comparison(
    baseline: dict[int, dict[str, list[float]]],
    mono: dict[int, dict[str, list[float]]],
    labels: tuple[str, ...],
    rank: int | None,
) -> dict[str, Any]:
    result = {}
    for label in labels:
        baseline_values = _selected(baseline, label, rank)
        mono_values = _selected(mono, label, rank)
        baseline_stats, mono_stats = _stats(baseline_values), _stats(mono_values)
        result[label] = {
            "baseline": baseline_stats,
            "mono": mono_stats,
            "gap_mean_us": mono_stats["mean_us"] - baseline_stats["mean_us"],
            "speedup": baseline_stats["mean_us"] / mono_stats["mean_us"],
        }
    return result


def _fused_baseline(
    baseline: dict[int, dict[str, list[float]]], iterations: int
) -> dict[int, dict[str, list[float]]]:
    result = {rank: {family: [] for family in FAMILIES} for rank in range(4)}
    per_layer = len(ELIGIBLE_LAYERS)
    for rank in range(4):
        for iteration in range(iterations):
            offset = iteration * per_layer
            result[rank]["K1"].append(baseline[rank]["K1"][offset])
            for layer in range(8):
                current = offset + layer
                value = sum(
                    baseline[rank][stage][current]
                    for stage in ("K-attn", "K2", "K3")
                )
                value += baseline[rank]["K1"][current + 1]
                result[rank]["Layer V2"].append(value)
            final = offset + 8
            result[rank]["Layer V1"].append(
                sum(
                    baseline[rank][stage][final]
                    for stage in ("K-attn", "K2", "K3")
                )
            )
    return result


def _total_comparison(
    baseline: dict[int, list[float]],
    split: dict[int, list[float]],
    fused: dict[int, list[float]],
    rank: int | None,
) -> dict[str, Any]:
    selected = {
        name: (
            [
                max(values[current][sample] for current in range(4))
                for sample in range(len(values[0]))
            ]
            if rank is None
            else values[rank]
        )
        for name, values in (
            ("baseline", baseline),
            ("split", split),
            ("fused", fused),
        )
    }
    stats = {name: _stats(values) for name, values in selected.items()}
    return {
        **stats,
        "split_gap_mean_us": stats["split"]["mean_us"] - stats["baseline"]["mean_us"],
        "split_speedup": stats["baseline"]["mean_us"] / stats["split"]["mean_us"],
        "fused_gap_mean_us": stats["fused"]["mean_us"] - stats["baseline"]["mean_us"],
        "fused_speedup": stats["baseline"]["mean_us"] / stats["fused"]["mean_us"],
    }


def report(
    baseline_dir: Path,
    split_dir: Path,
    fused_dir: Path | None,
    rank: int | None = None,
) -> dict[str, Any]:
    baseline, baseline_total, baseline_iterations = _baseline_samples(baseline_dir)
    split, split_total, split_iterations = _mono_samples(split_dir, "split")
    split_comparison = _comparison(baseline, split, SPLIT_STAGES, rank)
    result = {
        "schema_version": 1,
        "workload": "Muse Spark synthetic Target layers 0-9 width8 8K replay",
        "iterations": baseline_iterations,
        "iteration_counts": {
            "baseline": baseline_iterations,
            "split": split_iterations,
        },
        "ranks": 4,
        "selected_rank": rank,
        "selection": "critical-rank" if rank is None else f"rank-{rank}",
        "excluded": "layer 0 BF16 fallback",
        "launches_per_rank_iteration": {
            "baseline_eligible": 162,
            "split_mono": 36,
        },
        "split_stages": split_comparison,
        "largest_absolute_gap_first": {
            "split": sorted(
                SPLIT_STAGES,
                key=lambda label: abs(split_comparison[label]["gap_mean_us"]),
                reverse=True,
            ),
        },
        "directories": {
            "baseline": str(baseline_dir),
            "split": str(split_dir),
        },
    }
    if fused_dir is not None:
        fused, fused_total, fused_iterations = _mono_samples(fused_dir, "fused")
        fused_baseline = _fused_baseline(baseline, baseline_iterations)
        fused_comparison = _comparison(fused_baseline, fused, FAMILIES, rank)
        result["iteration_counts"]["fused"] = fused_iterations
        result["launches_per_rank_iteration"]["fused_mono"] = 10
        result["fused_families"] = fused_comparison
        result["largest_absolute_gap_first"]["fused"] = sorted(
            FAMILIES,
            key=lambda label: abs(fused_comparison[label]["gap_mean_us"]),
            reverse=True,
        )
        result["eligible_total"] = _total_comparison(
            baseline_total, split_total, fused_total, rank
        )
        result["directories"]["fused"] = str(fused_dir)
    return result


def _print(result: dict[str, Any]) -> None:
    print(
        f"Split Mono semantic stages ({result['selection']}, average per layer)"
    )
    print("stage      baseline_us  split_us   gap_us  speedup")
    for stage in SPLIT_STAGES:
        item = result["split_stages"][stage]
        print(
            f"{stage:<14} {item['baseline']['mean_us']:>11.3f} "
            f"{item['mono']['mean_us']:>9.3f} "
            f"{item['gap_mean_us']:>8.3f} {item['speedup']:>7.3f}x"
        )
    if "fused_families" in result:
        print(f"\nFused Mono launch families ({result['selection']})")
        print("family     baseline_us  fused_us   gap_us  speedup")
        for family in FAMILIES:
            item = result["fused_families"][family]
            print(
                f"{family:<10} {item['baseline']['mean_us']:>11.3f} "
                f"{item['mono']['mean_us']:>9.3f} "
                f"{item['gap_mean_us']:>8.3f} {item['speedup']:>7.3f}x"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("fused_dir", type=Path, nargs="?")
    parser.add_argument(
        "--rank",
        type=int,
        choices=range(4),
        help="report one rank instead of the cross-rank critical path",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.baseline_dir, args.split_dir, args.fused_dir, args.rank)
    _print(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
