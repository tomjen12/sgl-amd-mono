#!/usr/bin/env python3
"""Compare four-rank Muse Spark Target baseline and Mono replay snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from validation_contract import COSINE_THRESHOLD

LAYERS = 10
DEFAULT_COSINE = COSINE_THRESHOLD


def _minimum_row_cosine(got: torch.Tensor, expected: torch.Tensor) -> float:
    if got.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {got.shape} != {expected.shape}")
    got = got.float().reshape(got.shape[0], -1)
    expected = expected.float().reshape(expected.shape[0], -1)
    return float(F.cosine_similarity(got, expected, dim=-1).min().item())


def _require_cosine(checks, name, got, expected, threshold):
    value = _minimum_row_cosine(got, expected)
    checks[name] = value
    if value <= threshold:
        raise AssertionError(f"{name}: cosine {value:.6f} <= {threshold:.6f}")


def compare_rank(baseline: dict, mono: dict, threshold: float) -> dict:
    rank = baseline["rank"]
    if rank != mono["rank"]:
        raise AssertionError(f"rank mismatch: {rank} != {mono['rank']}")
    checks: dict[str, float] = {}
    _require_cosine(
        checks, "output", mono["output"], baseline["output"], threshold
    )
    for layer in range(LAYERS):
        expected_ids = baseline["routing_ids"][layer]
        if not torch.equal(mono["routing_ids"][layer], expected_ids):
            raise AssertionError(f"rank {rank} layer {layer}: routing override differs")
        actual_ids = mono["layers"][layer]["routing_ids"]
        if not torch.equal(actual_ids, expected_ids):
            raise AssertionError(f"rank {rank} layer {layer}: Mono routing differs")
        for name in ("hidden", "residual"):
            _require_cosine(
                checks,
                f"layer_{layer}.{name}",
                mono["layers"][layer][name],
                baseline["layers"][layer][name],
                threshold,
            )
        for name in ("key", "value"):
            got = mono["kv_cache"][layer][name]
            expected = baseline["kv_cache"][layer][name]
            if not torch.equal(got, expected):
                raise AssertionError(
                    f"rank {rank} layer {layer}: current KV {name} differs"
                )
    return {
        "rank": rank,
        "minimum_cosine": min(checks.values()),
        "checks": checks,
        "routing_ids_exact": True,
        "current_kv_exact": True,
    }


def compare_directories(
    baseline_dir: Path,
    mono_dir: Path,
    threshold: float = DEFAULT_COSINE,
) -> dict:
    baseline_paths = sorted(baseline_dir.glob("rank_*.snapshot.pt"))
    mono_paths = sorted(mono_dir.glob("rank_*.snapshot.pt"))
    if len(baseline_paths) != 4 or len(mono_paths) != 4:
        raise AssertionError("Target Gate 2 requires four snapshots per mode")
    ranks = []
    for baseline_path, mono_path in zip(baseline_paths, mono_paths):
        if baseline_path.name != mono_path.name:
            raise AssertionError(
                f"snapshot mismatch: {baseline_path.name} != {mono_path.name}"
            )
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        mono = torch.load(mono_path, map_location="cpu", weights_only=False)
        ranks.append(compare_rank(baseline, mono, threshold))
    return {
        "status": "passed",
        "threshold": threshold,
        "ranks": ranks,
        "minimum_cosine": min(item["minimum_cosine"] for item in ranks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("mono_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_directories(args.baseline_dir, args.mono_dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
