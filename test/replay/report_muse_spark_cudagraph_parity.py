#!/usr/bin/env python3
"""Report CUDA Graph replay envelopes separately from stage dispatch sums."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import report_muse_spark_draft_gap as draft
import report_muse_spark_target_gap as target
from trace_fingerprint import _duration, _load, _timestamp


def _stats(values):
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def _summary(directory: Path) -> dict:
    candidates = (directory / "summary.json", directory / "replay" / "summary.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"{directory}: summary.json not found")
    value = json.loads(path.read_text())
    if value.get("execution") != "cuda-graph":
        raise ValueError(f"{path}: expected execution=cuda-graph")
    ranks = value.get("ranks", ())
    if len(ranks) != 4:
        raise ValueError(f"{path}: expected four ranks, got {len(ranks)}")
    return value


def _trace_paths(directory: Path, scope: str):
    trace_dir = directory / "traces"
    if not trace_dir.is_dir():
        trace_dir = directory / "replay" / "traces"
    paths = sorted(trace_dir.glob(f"{scope}.rank_*.trace.json.gz"))
    if len(paths) != 4:
        raise ValueError(f"{directory}: expected four {scope} traces, got {len(paths)}")
    return paths


def _envelopes(directory: Path, scope: str):
    by_rank = {}
    counts = {}
    for rank, path in enumerate(_trace_paths(directory, scope)):
        trace = _load(str(path))
        scopes = (
            draft._scope_kernels(trace)
            if scope == "draft"
            else target._scope_kernels(trace, r"^muse_replay/target/verify$")
        )
        rank_envelopes = []
        rank_sums = []
        rank_counts = []
        for kernels in scopes:
            if not kernels:
                raise ValueError(f"{path}: empty graph replay scope")
            starts = [_timestamp(event) for event in kernels]
            ends = [
                _timestamp(event) + _duration(event)
                for event in kernels
            ]
            rank_envelopes.append(max(ends) - min(starts))
            rank_sums.append(sum(_duration(event) for event in kernels))
            rank_counts.append(len(kernels))
        by_rank[rank] = {
            "gpu_graph_envelope": _stats(rank_envelopes),
            "all_gpu_dispatch_sum": _stats(rank_sums),
        }
        counts[rank] = rank_counts
    if any(len(counts[rank]) != len(counts[0]) for rank in range(1, 4)):
        raise ValueError(f"{directory}: replay iteration count differs across ranks")
    return by_rank, counts


def _identity(draft_base, draft_mono, target_base, target_mono):
    if draft_base["descriptor"] != draft_mono["descriptor"]:
        raise ValueError("Draft Baseline/Mono descriptors differ")
    baseline_inputs = [rank["input"] for rank in target_base["ranks"]]
    mono_inputs = [rank["input"] for rank in target_mono["ranks"]]
    if baseline_inputs != mono_inputs:
        raise ValueError("Target Baseline/Mono inputs differ")
    return {
        "draft_descriptor_equal": True,
        "target_inputs_equal": True,
        "draft_seed": draft_base["descriptor"]["seed"],
        "target_input": baseline_inputs[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_baseline", type=Path)
    parser.add_argument("draft_mono", type=Path)
    parser.add_argument("target_baseline", type=Path)
    parser.add_argument("target_mono", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        "draft_baseline": _summary(args.draft_baseline),
        "draft_mono": _summary(args.draft_mono),
        "target_baseline": _summary(args.target_baseline),
        "target_mono": _summary(args.target_mono),
    }
    directories = {
        "draft_baseline": (args.draft_baseline, "draft"),
        "draft_mono": (args.draft_mono, "draft"),
        "target_baseline": (args.target_baseline, "target"),
        "target_mono": (args.target_mono, "target"),
    }
    runs = {}
    for name, (directory, scope) in directories.items():
        ranks, counts = _envelopes(directory, scope)
        summary_rank0 = summaries[name]["ranks"][0]
        runs[name] = {
            "rank0": ranks[0],
            "rank0_kernel_counts": counts[0],
            "iterations_per_rank": [len(counts[rank]) for rank in range(4)],
            "summary_iterations_per_rank": [
                rank["iterations"] for rank in summaries[name]["ranks"]
            ],
            "rank0_synchronized_wall_seconds": summary_rank0.get(
                "latency_seconds"
            ),
        }
    result = {
        "schema_version": 1,
        "contract": {
            "gpu_graph_envelope": "first-to-last GPU kernel span per replay iteration",
            "all_gpu_dispatch_sum": "sum of GPU kernel durations per replay iteration",
            "warning": "Envelope and dispatch sums are separate metrics and must not be mixed.",
        },
        "workload_identity": _identity(
            summaries["draft_baseline"],
            summaries["draft_mono"],
            summaries["target_baseline"],
            summaries["target_mono"],
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
