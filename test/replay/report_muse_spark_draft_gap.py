#!/usr/bin/env python3
"""Report per-layer Draft baseline-to-Mono GPU kernel gaps from PyTorch traces."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

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

STAGES = ("D-pre", "D-attn", "D-post", "D-MLP")

BASELINE_ALIASES = (
    ("rms", "_rms_norm_kernel_"),
    ("row", "_rowmajor_partial.kd"),
    ("prep", "_conv_prepare_kernel.kd"),
    ("sum", "_sum_partials.kd"),
    ("qk", "_table_qk_norm_rope_kernel.kd"),
    ("attn_part", "_partials.kd"),
    ("attn_reduce", "_reduce.kd"),
    ("o_gemm", "_gemm_a16w16_gfx1250_bandwidth_bound_kernel_"),
    ("all_reduce", "all_reduce_1shot_push_kernel"),
    ("finish", "_conv_finish_add_rmsnorm_kernel.kd"),
    ("silu_mul", "act_and_mul_kernel"),
)

MONO_ALIASES = (
    ("D-pre", "muse_spark_gfx1250_dpre"),
    ("D-attn", "muse_spark_gfx1250_dattn"),
    ("D-post", "muse_spark_gfx1250_dpost"),
    ("D-MLP", "muse_spark_gfx1250_dmlp"),
)

FIRST_LAYER = {
    "D-pre": ("rms", "row", "prep", "row", "sum", "qk"),
    "D-attn": ("attn_part", "attn_reduce"),
    "D-post": ("o_gemm", "all_reduce", "finish"),
    "D-MLP": ("row", "prep", "row", "silu_mul", "row", "all_reduce", "finish"),
}
LATER_LAYER = {
    **FIRST_LAYER,
    "D-pre": ("row", "prep", "row", "sum", "qk"),
}


def _scope_kernels(trace: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """GPU kernel events associated with each Draft forward scope."""
    events = _complete_events(trace)
    scopes = _ordered(
        event
        for event in events
        if _category(event) in CPU_CATEGORIES
        and str(event.get("name", "")) == "muse_replay/draft/forward"
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

        kernels = [
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
        ]
        result.append(_ordered(kernels))
    return result


def _alias(name: str, aliases: tuple[tuple[str, str], ...]) -> str | None:
    return next((label for label, needle in aliases if needle in name), None)


def _baseline_layers(kernels: list[dict[str, Any]]) -> list[dict[str, float]]:
    relevant = [
        (alias, _duration(event))
        for event in kernels
        if (alias := _alias(normalize_name(event.get("name", "")), BASELINE_ALIASES))
    ]
    expected = []
    layouts = [FIRST_LAYER, *([LATER_LAYER] * 4)]
    for layout in layouts:
        for stage in STAGES:
            expected.extend(layout[stage])
    got = [alias for alias, _ in relevant]
    if got != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(got, expected)) if pair[0] != pair[1]),
            min(len(got), len(expected)),
        )
        raise ValueError(
            "baseline stage map mismatch at relevant kernel "
            f"{mismatch}: got {got[mismatch:mismatch + 6]}, "
            f"expected {expected[mismatch:mismatch + 6]}; "
            f"counts got={len(got)} expected={len(expected)}"
        )

    layers, cursor = [], 0
    for layout in layouts:
        values = {}
        for stage in STAGES:
            count = len(layout[stage])
            values[stage] = sum(duration for _, duration in relevant[cursor : cursor + count])
            cursor += count
        layers.append(values)
    return layers


def _mono_layers(kernels: list[dict[str, Any]]) -> list[dict[str, float]]:
    relevant = [
        (alias, _duration(event))
        for event in kernels
        if (alias := _alias(normalize_name(event.get("name", "")), MONO_ALIASES))
    ]
    expected = list(STAGES) * 5
    got = [alias for alias, _ in relevant]
    if got != expected:
        raise ValueError(
            f"Mono stage map mismatch: got {got}, expected {expected}"
        )
    return [
        {stage: relevant[layer * 4 + offset][1] for offset, stage in enumerate(STAGES)}
        for layer in range(5)
    ]


def _trace_paths(directory: Path) -> list[Path]:
    paths = sorted((directory / "traces").glob("draft.rank_*.trace.json.gz"))
    if len(paths) != 4:
        raise ValueError(f"{directory}: expected four rank traces, found {len(paths)}")
    return paths


def _rank(path: Path) -> int:
    match = re.search(r"rank_(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot read rank from {path}")
    return int(match.group(1))


def _collect(directory: Path, mode: str):
    samples = {stage: [] for stage in STAGES}
    rank_samples = {rank: {stage: [] for stage in STAGES} for rank in range(4)}
    iterations = None
    parser = _baseline_layers if mode == "baseline" else _mono_layers
    for path in _trace_paths(directory):
        rank = _rank(path)
        scopes = _scope_kernels(_load(str(path)))
        if iterations is None:
            iterations = len(scopes)
        elif len(scopes) != iterations:
            raise ValueError(f"{path}: {len(scopes)} iterations, expected {iterations}")
        for kernels in scopes:
            for layer in parser(kernels):
                for stage, duration in layer.items():
                    samples[stage].append(duration)
                    rank_samples[rank][stage].append(duration)
    return samples, rank_samples, int(iterations or 0)


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def report(baseline_dir: Path, mono_dir: Path) -> dict[str, Any]:
    baseline, baseline_ranks, baseline_iterations = _collect(
        baseline_dir, "baseline"
    )
    mono, mono_ranks, mono_iterations = _collect(mono_dir, "mono")
    if baseline_iterations != mono_iterations:
        raise ValueError(
            f"iteration mismatch: baseline={baseline_iterations}, mono={mono_iterations}"
        )

    stages = {}
    for stage in STAGES:
        baseline_stats, mono_stats = _stats(baseline[stage]), _stats(mono[stage])
        per_rank_samples = len(baseline_ranks[0][stage])
        baseline_critical = [
            max(baseline_ranks[rank][stage][sample] for rank in range(4))
            for sample in range(per_rank_samples)
        ]
        mono_critical = [
            max(mono_ranks[rank][stage][sample] for rank in range(4))
            for sample in range(per_rank_samples)
        ]
        baseline_critical_stats = _stats(baseline_critical)
        mono_critical_stats = _stats(mono_critical)
        per_rank = {}
        for rank in range(4):
            b = statistics.fmean(baseline_ranks[rank][stage])
            m = statistics.fmean(mono_ranks[rank][stage])
            per_rank[str(rank)] = {
                "baseline_mean_us": b,
                "mono_mean_us": m,
                "gap_us": m - b,
                "speedup": b / m,
            }
        stages[stage] = {
            "baseline": baseline_stats,
            "mono": mono_stats,
            "gap_mean_us": mono_stats["mean_us"] - baseline_stats["mean_us"],
            "speedup": baseline_stats["mean_us"] / mono_stats["mean_us"],
            "critical_path": {
                "baseline": baseline_critical_stats,
                "mono": mono_critical_stats,
                "gap_mean_us": (
                    mono_critical_stats["mean_us"]
                    - baseline_critical_stats["mean_us"]
                ),
                "speedup": (
                    baseline_critical_stats["mean_us"]
                    / mono_critical_stats["mean_us"]
                ),
            },
            "per_rank": per_rank,
            "slowest_mono_rank": max(
                per_rank, key=lambda rank: per_rank[rank]["mono_mean_us"]
            ),
        }
    return {
        "schema_version": 1,
        "workload": "Muse Spark DFlash2 width8 8K replay",
        "iterations": baseline_iterations,
        "layers_per_iteration": 5,
        "ranks": 4,
        "baseline_dir": str(baseline_dir),
        "mono_dir": str(mono_dir),
        "stages": stages,
    }


def _print(result: dict[str, Any]) -> None:
    print(
        "stage   baseline_critical_us  mono_critical_us  gap_us  speedup  "
        "all_rank_baseline_mean  all_rank_mono_mean"
    )
    for stage in STAGES:
        item = result["stages"][stage]
        critical = item["critical_path"]
        print(
            f"{stage:<7} "
            f"{critical['baseline']['mean_us']:>20.3f} "
            f"{critical['mono']['mean_us']:>17.3f} "
            f"{critical['gap_mean_us']:>7.3f} "
            f"{critical['speedup']:>7.3f}x "
            f"{item['baseline']['mean_us']:>22.3f} "
            f"{item['mono']['mean_us']:>18.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("mono_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.baseline_dir, args.mono_dir)
    _print(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
