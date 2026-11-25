# SPDX-License-Identifier: Apache-2.0
"""Merge isolated TP8 stage profiles into one Baseline-vs-Mono report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tp8_synthetic_report import Sample, render_performance_log, write_reports


BASE_STAGES = ("k1", "kattn_full", "kattn_window", "k2", "k3")
LAYER_PARTS = {
    "layer_v1_full": ("kattn_full", "k2", "k3"),
    "layer_v1_swa": ("kattn_window", "k2", "k3"),
    "layer_v2_full": ("kattn_full", "k2", "k3", "k1"),
    "layer_v2_swa": ("kattn_window", "k2", "k3", "k1"),
}


def _read_samples(directory: Path) -> list[Sample]:
    with (directory / "raw.csv").open(newline="") as handle:
        return [
            Sample(
                run_id=row["run_id"],
                stage=row["stage"],
                width=int(row["width"]),
                rank=int(row["rank"]),
                implementation=row["implementation"],
                iteration=int(row["iteration"]),
                launch_count=int(row["launch_count"]),
                total_us=float(row["total_us"]),
            )
            for row in csv.DictReader(handle)
            if not row["stage"].startswith("kattn_k2_")
        ]


def _derive_baseline_layers(samples: list[Sample]) -> list[Sample]:
    indexed = {
        (sample.stage, sample.width, sample.iteration): sample
        for sample in samples
        if sample.rank == 0 and sample.implementation == "baseline"
    }
    derived: list[Sample] = []
    widths = sorted({sample.width for sample in samples})
    iterations = sorted({sample.iteration for sample in samples})
    for width in widths:
        for iteration in iterations:
            for stage, parts in LAYER_PARTS.items():
                items = [indexed[(part, width, iteration)] for part in parts]
                derived.append(
                    Sample(
                        run_id="tp8-synthetic",
                        stage=stage,
                        width=width,
                        rank=0,
                        implementation="baseline",
                        iteration=iteration,
                        launch_count=sum(item.launch_count for item in items),
                        total_us=sum(item.total_us for item in items),
                    )
                )
    return derived


def _validate(samples: list[Sample], iterations: int) -> None:
    expected = {
        (implementation, stage)
        for implementation in ("baseline", "mono")
        for stage in (*BASE_STAGES, *LAYER_PARTS)
    }
    actual = {(sample.implementation, sample.stage) for sample in samples}
    if actual != expected:
        raise ValueError(
            f"profile coverage mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for implementation, stage in sorted(expected):
        group = [
            sample
            for sample in samples
            if sample.implementation == implementation and sample.stage == stage
        ]
        if len(group) != iterations or {sample.rank for sample in group} != {0}:
            raise ValueError(
                f"{implementation} {stage}: expected {iterations} rank-0 samples, "
                f"got {len(group)}"
            )


def _merge_dispatch(
    directories: list[Path],
    name: str,
    output: Path,
) -> bool:
    dispatches: list[dict] = []
    for directory in directories:
        path = directory / name
        document = json.loads(path.read_text())
        if document.get("status") != "passed":
            return False
        dispatches.extend(document.get("dispatches", ()))
    (output / name).write_text(
        json.dumps({"status": "passed", "dispatches": dispatches}, indent=2)
        + "\n"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--mono", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    baseline = [
        sample
        for directory in args.baseline
        for sample in _read_samples(directory)
    ]
    mono = [
        sample for directory in args.mono for sample in _read_samples(directory)
    ]
    samples = [*baseline, *_derive_baseline_layers(baseline), *mono]
    _validate(samples, args.iterations)

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_valid = _merge_dispatch(
        args.baseline, "baseline_dispatch.json", args.output
    )
    mono_valid = _merge_dispatch(args.mono, "mono_dispatch.json", args.output)
    report = write_reports(
        args.output,
        samples,
        metadata={
            "widths": [8],
            "warmup": 50,
            "iterations": args.iterations,
            "profile": "torch",
            "rank": 0,
            "execution": "stage-isolated",
            "baseline_sources": [str(path) for path in args.baseline],
            "mono_sources": [str(path) for path in args.mono],
        },
        dispatch_valid=baseline_valid and mono_valid,
    )
    log = render_performance_log(report, str(args.output))
    (args.output / "performance.log").write_text(log + "\n")
    repository_root = Path(__file__).resolve().parents[5]
    (repository_root / "TP8_4GPU_SYNTHETIC_PERFORMANCE.log").write_text(
        log + "\n"
    )
    print(log)


if __name__ == "__main__":
    main()
