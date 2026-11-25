# SPDX-License-Identifier: Apache-2.0
"""Reports for four-active-rank, logical-TP8 synthetic benchmarks."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


WORKLOAD = "tp8-synthetic-4gpu-logical-tp8"


@dataclass(frozen=True)
class Sample:
    run_id: str
    stage: str
    width: int
    rank: int
    implementation: str
    iteration: int
    launch_count: int
    total_us: float
    execution: str = "eager"


def _with_fair_stage_groups(samples: list[Sample]) -> list[Sample]:
    """Add K-attn+K2 groups whose semantic boundary matches across backends."""
    values = list(samples)
    indexed = {
        (
            sample.stage,
            sample.width,
            sample.rank,
            sample.implementation,
            sample.execution,
            sample.iteration,
        ): sample
        for sample in values
    }
    for attention, combined in (
        ("kattn_full", "kattn_k2_full"),
        ("kattn_window", "kattn_k2_window"),
    ):
        keys = {
            (
                sample.width,
                sample.rank,
                sample.implementation,
                sample.execution,
                sample.iteration,
            )
            for sample in values
            if sample.stage == attention
        }
        for width, rank, implementation, execution, iteration in sorted(keys):
            combined_key = (
                combined,
                width,
                rank,
                implementation,
                execution,
                iteration,
            )
            if combined_key in indexed:
                continue
            attention_sample = indexed.get(
                (attention, width, rank, implementation, execution, iteration)
            )
            k2_sample = indexed.get(
                ("k2", width, rank, implementation, execution, iteration)
            )
            if attention_sample is None or k2_sample is None:
                continue
            derived = Sample(
                run_id=attention_sample.run_id,
                stage=combined,
                width=width,
                rank=rank,
                implementation=implementation,
                iteration=iteration,
                launch_count=(
                    attention_sample.launch_count + k2_sample.launch_count
                ),
                total_us=attention_sample.total_us + k2_sample.total_us,
                execution=execution,
            )
            values.append(derived)
            indexed[combined_key] = derived
    return values


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "total_us": sum(values),
        "mean_us": statistics.fmean(values),
        "p50_us": _percentile(values, 0.50),
        "p90_us": _percentile(values, 0.90),
        "min_us": min(values),
        "max_us": max(values),
    }


def _rank0_samples(
    samples: Iterable[Sample],
) -> dict[tuple[str, int, str, str], list[float]]:
    selected: dict[
        tuple[str, int, str, str], list[tuple[int, float]]
    ] = defaultdict(list)
    for sample in samples:
        if sample.rank == 0:
            selected[
                (
                    sample.stage,
                    sample.width,
                    sample.implementation,
                    sample.execution,
                )
            ].append((sample.iteration, sample.total_us))
    return {
        key: [value for _iteration, value in sorted(values)]
        for key, values in selected.items()
    }


def summarize(samples: Iterable[Sample]) -> tuple[list[dict], list[dict]]:
    """Return per-rank rows and paired rank-0 gap rows."""
    values = list(samples)
    grouped: dict[tuple[str, int, int, str, str], list[Sample]] = defaultdict(list)
    for sample in values:
        grouped[
            (
                sample.stage,
                sample.width,
                sample.rank,
                sample.implementation,
                sample.execution,
            )
        ].append(sample)

    per_rank: list[dict] = []
    for (stage, width, rank, implementation, execution), group in sorted(
        grouped.items()
    ):
        launches = {sample.launch_count for sample in group}
        if len(launches) != 1:
            raise ValueError(
                f"{stage} width {width} rank {rank} {implementation}: "
                f"inconsistent launch counts {sorted(launches)}"
            )
        per_rank.append(
            {
                "scope": "rank",
                "stage": stage,
                "width": width,
                "rank": rank,
                "implementation": implementation,
                "execution": execution,
                "launch_count": launches.pop(),
                **_stats([sample.total_us for sample in group]),
            }
        )

    selected = _rank0_samples(values)
    paired: list[dict] = []
    stage_widths = sorted(
        {(sample.stage, sample.width, sample.execution) for sample in values}
    )
    for stage, width, execution in stage_widths:
        baseline = selected.get((stage, width, "baseline", execution))
        mono = selected.get((stage, width, "mono", execution))
        if baseline is None or mono is None:
            paired.append(
                {
                    "scope": "rank0",
                    "stage": stage,
                    "width": width,
                    "execution": execution,
                    "status": "INVALID",
                    "reason": "missing baseline or mono samples",
                }
            )
            continue
        if len(baseline) != len(mono):
            raise ValueError(
                f"{stage} width {width}: baseline/mono sample count differs"
            )
        baseline_stats = _stats(baseline)
        mono_stats = _stats(mono)
        baseline_launches = {
            sample.launch_count
            for sample in values
            if sample.rank == 0
            and sample.stage == stage
            and sample.width == width
            and sample.implementation == "baseline"
            and sample.execution == execution
        }
        mono_launches = {
            sample.launch_count
            for sample in values
            if sample.rank == 0
            and sample.stage == stage
            and sample.width == width
            and sample.implementation == "mono"
            and sample.execution == execution
        }
        if len(baseline_launches) != 1 or len(mono_launches) != 1:
            raise ValueError(f"{stage} width {width}: inconsistent launch counts")
        baseline_mean = float(baseline_stats["mean_us"])
        mono_mean = float(mono_stats["mean_us"])
        paired.append(
            {
                "scope": "rank0",
                "stage": stage,
                "width": width,
                "execution": execution,
                "status": "PASS",
                "samples": len(baseline),
                "baseline_launch_count": baseline_launches.pop(),
                "mono_launch_count": mono_launches.pop(),
                "baseline_mean_us": baseline_mean,
                "baseline_p50_us": baseline_stats["p50_us"],
                "baseline_p90_us": baseline_stats["p90_us"],
                "mono_mean_us": mono_mean,
                "mono_p50_us": mono_stats["p50_us"],
                "mono_p90_us": mono_stats["p90_us"],
                "gap_us": mono_mean - baseline_mean,
                "speedup": baseline_mean / mono_mean,
            }
        )
    return per_rank, paired


def write_reports(
    output_dir: Path,
    samples: Iterable[Sample],
    *,
    metadata: dict,
    dispatch_valid: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = _with_fair_stage_groups(list(samples))
    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="") as handle:
        fields = [
            "run_id",
            "workload",
            "stage",
            "width",
            "rank",
            "implementation",
            "execution",
            "iteration",
            "launch_count",
            "total_us",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in sorted(
            values,
            key=lambda item: (
                item.stage,
                item.width,
                item.implementation,
                item.rank,
                item.iteration,
            ),
        ):
            writer.writerow({**asdict(sample), "workload": WORKLOAD})

    per_rank, paired = summarize(values)
    if not dispatch_valid:
        for row in paired:
            row.update(
                status="INVALID",
                reason="production Baseline dispatch validation failed",
            )

    summary_path = output_dir / "summary.csv"
    summary_fields = [
        "scope",
        "stage",
        "width",
        "rank",
        "implementation",
        "execution",
        "launch_count",
        "samples",
        "total_us",
        "mean_us",
        "p50_us",
        "p90_us",
        "min_us",
        "max_us",
        "status",
        "baseline_launch_count",
        "mono_launch_count",
        "baseline_mean_us",
        "baseline_p50_us",
        "baseline_p90_us",
        "mono_mean_us",
        "mono_p50_us",
        "mono_p90_us",
        "gap_us",
        "speedup",
        "reason",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in [*per_rank, *paired]:
            writer.writerow(row)

    result = {
        "workload": WORKLOAD,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "dispatch_valid": dispatch_valid,
        "per_rank": per_rank,
        "critical_gap": paired,
    }
    (output_dir / "gap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def render_performance_log(report: dict, artifact_path: str) -> str:
    metadata = report["metadata"]
    stage_order = {
        stage: index
        for index, stage in enumerate(
            (
                "k1",
                "kattn_full",
                "kattn_window",
                "k2",
                "kattn_k2_full",
                "kattn_k2_window",
                "k3",
                "layer_v1_full",
                "layer_v1_swa",
                "layer_v2_full",
                "layer_v2_swa",
            )
        )
    }
    rows = sorted(
        report["critical_gap"],
        key=lambda row: (
            row["width"],
            row.get("execution", "eager"),
            stage_order.get(row["stage"], len(stage_order)),
            row["stage"],
        ),
    )
    header = (
        f"{'execution':<9} {'stage':<20} {'width':>5} | "
        f"{'baseline avg us':>15} {'launches':>8} | "
        f"{'mono avg us':>12} {'launches':>8} | "
        f"{'gap us':>11} {'speedup':>8} {'status':>7}"
    )
    direct_rows = [
        row for row in rows if not row["stage"].startswith("kattn_k2_")
    ]
    derived_rows = [
        row for row in rows if row["stage"].startswith("kattn_k2_")
    ]
    graph_direct = [
        row
        for row in direct_rows
        if row.get("execution") == "graph" and row.get("status") == "PASS"
    ]
    sample_count = graph_direct[0]["samples"] if graph_direct else 0
    official_samples = len(graph_direct) * 2 * sample_count
    runtime_parity = metadata.get(
        "runtime_graph_dispatch_parity", "NOT_RECORDED"
    )
    unexpected_fallback = metadata.get("unexpected_fallback", "unknown")
    if isinstance(unexpected_fallback, bool):
        unexpected_fallback = str(unexpected_fallback).lower()

    lines = [
        "TP8 Synthetic Baseline vs Mono (4 active GPUs, logical TP8/EP8)",
        f"Updated: {report['created_at']}",
        "",
        "CONFIGURATION",
        "=============",
        f"workload={report['workload']}",
        f"widths={metadata.get('widths')} warmup={metadata.get('warmup')} "
        f"iterations={metadata.get('iterations')} profile={metadata.get('profile')}",
        "active_ranks=4 logical_tp=8 logical_ep=8 resident_blocks=96",
        "pooler=sharded",
        "",
        "This is a four-active-rank synthetic result. It is not real eight-GPU",
        "TP8/EP8 latency and is not comparable to TP4 runtime performance.",
        "",
        "VALIDATION",
        "==========",
        f"runtime_graph_dispatch_parity={runtime_parity}",
        f"unexpected_fallback={unexpected_fallback}",
        f"graph_baseline_cases={len(graph_direct)}",
        f"graph_mono_cases={len(graph_direct)}",
        f"samples_per_case={sample_count}",
        f"official_rank0_samples={official_samples}",
        "",
        "DIRECT STAGE RESULTS (rank 0 latency)",
        "=====================================",
        "Average is the sum of matching GPU kernel durations per iteration,",
        "averaged over all profiled iterations. CPU time and bubbles are excluded.",
        "",
        header,
        "-" * len(header),
    ]

    def append_rows(selected: list[dict]) -> None:
        for row in selected:
            if row["status"] != "PASS":
                lines.append(
                    f"{row.get('execution', 'eager'):<9} "
                    f"{row['stage']:<20} {row['width']:>5} | "
                    f"{'-':>15} {'-':>8} | {'-':>12} {'-':>8} | "
                    f"{'-':>11} {'-':>8} {'INVALID':>7}"
                )
                continue
            lines.append(
                f"{row.get('execution', 'eager'):<9} "
                f"{row['stage']:<20} {row['width']:>5} | "
                f"{row['baseline_mean_us']:>15.3f} "
                f"{row['baseline_launch_count']:>8} | "
                f"{row['mono_mean_us']:>12.3f} "
                f"{row['mono_launch_count']:>8} | "
                f"{row['gap_us']:>+11.3f} "
                f"{row['speedup']:>7.3f}x {'PASS':>7}"
            )

    append_rows(direct_rows)
    if derived_rows:
        lines.extend(
            (
                "",
                "DERIVED SEMANTIC-BOUNDARY RESULTS",
                "=================================",
                header,
                "-" * len(header),
            )
        )
        append_rows(derived_rows)

    by_stage = {
        row["stage"]: row
        for row in graph_direct
        if row.get("status") == "PASS"
    }
    priority_stages = (
        "layer_v1_swa",
        "k2",
        "k3",
        "kattn_window",
    )
    if all(stage in by_stage for stage in priority_stages):
        lines.extend(
            (
                "",
                "INITIAL OPTIMIZATION PRIORITY",
                "=============================",
                "1. Layer V1 SWA: recover the "
                f"{by_stage['layer_v1_swa']['gap_us']:.3f} us regression "
                "without regressing V2.",
                "2. K2 diagnostic path: investigate the "
                f"{by_stage['k2']['gap_us']:.3f} us standalone gap.",
                "3. K3 diagnostic path: investigate the "
                f"{by_stage['k3']['gap_us']:.3f} us standalone gap.",
                "4. K-attn SWA: investigate the "
                f"{by_stage['kattn_window']['gap_us']:.3f} us standalone gap.",
            )
        )
        if "layer_v2_full" in by_stage and "layer_v2_swa" in by_stage:
            lines.extend(
                (
                    "",
                    "Layer V2 speedup is "
                    f"{by_stage['layer_v2_full']['speedup']:.3f}x Full and "
                    f"{by_stage['layer_v2_swa']['speedup']:.3f}x SWA. "
                    "Standalone K2/K3 are diagnostic",
                    "proxies; they must not be added to predict fused Layer latency.",
                )
            )

    lines.extend(("", f"ARTIFACTS: {artifact_path}"))
    runtime_parity_path = metadata.get("runtime_parity_path")
    if runtime_parity_path:
        lines.append(f"RUNTIME PARITY: {runtime_parity_path}")
    lines.append("")
    return "\n".join(lines)
