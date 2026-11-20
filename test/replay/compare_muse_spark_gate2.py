#!/usr/bin/env python3
"""Compare four-rank Muse Spark Draft baseline and Mono replay snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from validation_contract import COSINE_THRESHOLD

LAYERS = 5
DEFAULT_COSINE = COSINE_THRESHOLD


def _cosine_min(got: torch.Tensor, expected: torch.Tensor) -> float:
    if got.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {got.shape} != {expected.shape}")
    got = got.float().reshape(got.shape[0], -1)
    expected = expected.float().reshape(expected.shape[0], -1)
    return float(F.cosine_similarity(got, expected, dim=-1).min().item())


def _require_cosine(
    checks: dict[str, float],
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    threshold: float,
) -> None:
    value = _cosine_min(got, expected)
    checks[name] = value
    if value <= threshold:
        raise AssertionError(f"{name}: cosine {value:.6f} <= {threshold:.6f}")


def compare_rank(
    baseline: dict, mono: dict, *, threshold: float = DEFAULT_COSINE
) -> dict:
    if baseline["rank"] != mono["rank"]:
        raise AssertionError(
            f"rank mismatch: {baseline['rank']} != {mono['rank']}"
        )
    checks: dict[str, float] = {}
    _require_cosine(
        checks, "output", mono["output"], baseline["output"], threshold
    )
    for layer in range(LAYERS):
        native = baseline["layers"][layer]
        fused = mono["layers"][layer]
        _require_cosine(
            checks,
            f"layer_{layer}.hidden",
            fused["hidden"],
            native["hidden"],
            threshold,
        )
        _require_cosine(
            checks,
            f"layer_{layer}.residual",
            fused["residual"],
            native["residual"],
            threshold,
        )
        _require_cosine(
            checks,
            f"layer_{layer}.same_input_attention",
            fused["attention_o"],
            fused["same_input_reference_attention_o"],
            threshold,
        )
        _require_cosine(
            checks,
            f"layer_{layer}.mono_hidden_reference",
            fused["hidden"],
            fused["reference_hidden"],
            threshold,
        )
        _require_cosine(
            checks,
            f"layer_{layer}.convolution_carry",
            mono["convolution_carry"][layer],
            mono["reference_convolution_carry"][layer],
            threshold,
        )
        for name in ("key", "value"):
            got = fused[f"kv_{name}"]
            expected = fused[f"expected_kv_{name}"]
            _require_cosine(
                checks,
                f"layer_{layer}.current_kv_{name}",
                got,
                expected,
                threshold,
            )
    return {
        "rank": baseline["rank"],
        "minimum_cosine": min(checks.values()),
        "checks": checks,
        "current_kv_gate": f"fused-source cosine > {threshold:.6f}",
        "persistent_cache_comparison": "not-applicable-save_kv_cache_false",
    }


def compare_directories(
    baseline_dir: Path,
    mono_dir: Path,
    *,
    threshold: float = DEFAULT_COSINE,
) -> dict:
    baseline_paths = sorted(baseline_dir.glob("rank_*.snapshot.pt"))
    mono_paths = sorted(mono_dir.glob("rank_*.snapshot.pt"))
    if not baseline_paths or len(baseline_paths) != len(mono_paths):
        raise AssertionError(
            "baseline and Mono snapshot counts must match and be non-zero"
        )
    ranks = []
    for baseline_path, mono_path in zip(baseline_paths, mono_paths):
        if baseline_path.name != mono_path.name:
            raise AssertionError(
                f"snapshot mismatch: {baseline_path.name} != {mono_path.name}"
            )
        baseline = torch.load(
            baseline_path, map_location="cpu", weights_only=False
        )
        mono = torch.load(mono_path, map_location="cpu", weights_only=False)
        ranks.append(compare_rank(baseline, mono, threshold=threshold))
    return {
        "passed": True,
        "threshold": threshold,
        "rank_count": len(ranks),
        "minimum_cosine": min(rank["minimum_cosine"] for rank in ranks),
        "ranks": ranks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("mono_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_directories(args.baseline_dir, args.mono_dir)
        exit_code = 0
    except AssertionError as error:
        result = {
            "passed": False,
            "threshold": DEFAULT_COSINE,
            "error": str(error),
        }
        exit_code = 1
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
