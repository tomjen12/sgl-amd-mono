#!/usr/bin/env python3
"""Validate and summarize versioned Muse Spark internal phase traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sgl_amd_mono.muse_spark.gfx1250.phase_trace import (
    ABI_VERSION,
    MAGIC,
    phase_ids,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def validate_record(record: dict[str, Any]) -> list[int]:
    if record.get("magic") != MAGIC:
        raise ValueError("phase trace magic mismatch")
    if record.get("abi_version") != ABI_VERSION:
        raise ValueError("phase trace ABI version mismatch")
    points = phase_ids(record["family"])
    ticks = record.get("ticks")
    if not isinstance(ticks, list) or len(ticks) != len(points):
        raise ValueError(
            f"{record['family']} task {record.get('task')}: "
            f"expected {len(points)} ticks"
        )
    if any(not isinstance(value, int) for value in ticks):
        raise ValueError("phase ticks must be integers")
    deltas = [right - left for left, right in zip(ticks, ticks[1:])]
    if any(value < 0 for value in deltas):
        raise ValueError("phase ticks are unordered")
    supplied_points = tuple(record.get("phase_ids", points))
    if supplied_points != points:
        raise ValueError(
            f"{record['family']} phase IDs do not match ABI: "
            f"{supplied_points!r} != {points!r}"
        )
    return deltas


def summarize(
    artifacts: list[Path],
    *,
    reconciliation_tolerance: float = 0.20,
    required_families: tuple[str, ...] = (),
    required_ranks: int | None = None,
    required_layers: int | None = None,
    required_layer_start: int = 0,
) -> dict[str, Any]:
    records = []
    keys = set()
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    reconciliations = []
    for artifact in artifacts:
        payload = json.loads(artifact.read_text())
        current = payload if isinstance(payload, list) else payload.get("records", [])
        if not current:
            raise ValueError(f"{artifact}: no phase records")
        for record in current:
            key = (
                record["family"],
                int(record["rank"]),
                int(record["layer"]),
                int(record["task"]),
            )
            if key in keys:
                raise ValueError(f"duplicate phase record {key}")
            keys.add(key)
            deltas = validate_record(record)
            clock_hz = float(record.get("clock_hz", 1_545_000_000.0))
            delta_us = [value / clock_hz * 1_000_000.0 for value in deltas]
            for index, value in enumerate(delta_us):
                values[(record["family"], index)].append(value)
            enclosing = record.get("enclosing_us")
            if enclosing is not None:
                span = sum(delta_us)
                relative = abs(span - float(enclosing)) / max(float(enclosing), 1e-9)
                if relative > reconciliation_tolerance:
                    raise ValueError(
                        f"{key}: phase span {span:.3f} us does not reconcile "
                        f"with {enclosing:.3f} us"
                    )
                reconciliations.append(relative)
            records.append(record)
    observed_families = {record["family"] for record in records}
    missing_families = sorted(set(required_families) - observed_families)
    if missing_families:
        raise ValueError(f"missing phase families: {missing_families}")
    if required_ranks is not None:
        observed_ranks = {int(record["rank"]) for record in records}
        expected_ranks = set(range(required_ranks))
        if observed_ranks != expected_ranks:
            raise ValueError(
                f"phase rank coverage {sorted(observed_ranks)}, "
                f"expected {sorted(expected_ranks)}"
            )
    if required_layers is not None:
        expected_layers = set(
            range(required_layer_start, required_layer_start + required_layers)
        )
        for family in required_families or sorted(observed_families):
            observed_layers = {
                int(record["layer"])
                for record in records
                if record["family"] == family
            }
            if observed_layers != expected_layers:
                raise ValueError(
                    f"{family} layer coverage {sorted(observed_layers)}, "
                    f"expected {sorted(expected_layers)}"
                )
    families = {}
    for (family, index), samples in values.items():
        family_stats = families.setdefault(
            family, {"points": list(phase_ids(family)), "intervals": {}}
        )
        name = f"{phase_ids(family)[index]}->{phase_ids(family)[index + 1]}"
        family_stats["intervals"][name] = {
            "samples": len(samples),
            "mean_us": statistics.fmean(samples),
            "median_us": statistics.median(samples),
            "min_us": min(samples),
            "max_us": max(samples),
            "p95_us": _percentile(samples, 0.95),
        }
    return {
        "status": "PASS",
        "abi_version": ABI_VERSION,
        "records": len(records),
        "families": families,
        "reconciliation": {
            "samples": len(reconciliations),
            "max_relative_error": max(reconciliations, default=0.0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconciliation-tolerance", type=float, default=0.20)
    parser.add_argument("--require-family", action="append", default=[])
    parser.add_argument("--require-ranks", type=int)
    parser.add_argument("--require-layers", type=int)
    parser.add_argument("--require-layer-start", type=int, default=0)
    args = parser.parse_args()
    result = summarize(
        args.artifacts,
        reconciliation_tolerance=args.reconciliation_tolerance,
        required_families=tuple(args.require_family),
        required_ranks=args.require_ranks,
        required_layers=args.require_layers,
        required_layer_start=args.require_layer_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
