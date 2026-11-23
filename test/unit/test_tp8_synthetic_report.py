import csv
import json
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_synthetic_report import (  # noqa: E402
    Sample,
    render_performance_log,
    summarize,
    write_reports,
)


def _samples():
    values = []
    for implementation, base in (("baseline", 10.0), ("mono", 8.0)):
        for iteration in range(2):
            for rank in range(4):
                values.append(
                    Sample(
                        run_id="test",
                        stage="k1",
                        width=8,
                        rank=rank,
                        implementation=implementation,
                        iteration=iteration,
                        launch_count=2 if implementation == "baseline" else 1,
                        total_us=base + rank + iteration,
                    )
                )
    return values


def test_summary_uses_rank0_latency():
    per_rank, paired = summarize(_samples())
    assert len(per_rank) == 8
    assert len(paired) == 1
    row = paired[0]
    assert (row["scope"], row["stage"], row["width"], row["status"]) == (
        "rank0",
        "k1",
        8,
        "PASS",
    )
    assert row["samples"] == 2
    assert row["baseline_mean_us"] == 10.5
    assert row["baseline_p50_us"] == 10.5
    assert row["baseline_p90_us"] == pytest.approx(10.9)
    assert row["mono_mean_us"] == 8.5
    assert row["mono_p50_us"] == 8.5
    assert row["mono_p90_us"] == pytest.approx(8.9)
    assert row["gap_us"] == -2.0
    assert row["speedup"] == pytest.approx(10.5 / 8.5)


def test_inconsistent_launch_count_is_rejected():
    samples = _samples()
    samples[0] = Sample(**{**samples[0].__dict__, "launch_count": 99})
    with pytest.raises(ValueError, match="inconsistent launch counts"):
        summarize(samples)


def test_reports_and_log_are_written(tmp_path):
    report = write_reports(
        tmp_path,
        _samples(),
        metadata={
            "widths": [8],
            "warmup": 10,
            "iterations": 2,
            "profile": "torch",
        },
        dispatch_valid=True,
    )
    with (tmp_path / "raw.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 16
    with (tmp_path / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 9
    assert json.loads((tmp_path / "gap.json").read_text())["dispatch_valid"]
    rendered = render_performance_log(report, "artifacts/example")
    assert "logical TP8/EP8" in rendered
    assert "k1 8 10.500 8.500 -2.000" in rendered


def test_failed_dispatch_invalidates_gap(tmp_path):
    report = write_reports(
        tmp_path,
        _samples(),
        metadata={},
        dispatch_valid=False,
    )
    assert report["critical_gap"][0]["status"] == "INVALID"
    assert "dispatch validation failed" in report["critical_gap"][0]["reason"]
