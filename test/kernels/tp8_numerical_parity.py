# SPDX-License-Identifier: Apache-2.0
"""Direct numerical comparisons for shared Baseline/Mono logical fixtures."""

from __future__ import annotations

from typing import Mapping

import torch


def compare_outputs(
    baseline: Mapping[str, torch.Tensor],
    mono: Mapping[str, torch.Tensor],
) -> dict:
    if set(baseline) != set(mono):
        raise RuntimeError(
            f"output names differ: baseline={sorted(baseline)} "
            f"mono={sorted(mono)}"
        )
    checks = []
    for name in baseline:
        reference = baseline[name]
        actual = mono[name]
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise RuntimeError(
                f"{name}: Mono {tuple(actual.shape)} {actual.dtype}, Baseline "
                f"{tuple(reference.shape)} {reference.dtype}"
            )
        if not actual.is_floating_point():
            equal = bool(torch.equal(actual, reference))
            check = {
                "name": name,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "exact": equal,
            }
            checks.append(check)
            if not equal:
                mismatch = (actual != reference).nonzero()
                index = tuple(int(value) for value in mismatch[0].tolist())
                raise RuntimeError(
                    f"{name}: integer outputs differ at {index}: "
                    f"Mono={int(actual[index])} Baseline={int(reference[index])}"
                )
            continue
        got = actual.float()
        ref = reference.float()
        error = (got - ref).abs()
        limit = ref.abs() / 32 + 2e-2
        bad_fraction = float((error > limit).float().mean().item())
        cosine = float(
            torch.nn.functional.cosine_similarity(got, ref, dim=-1)
            .min()
            .item()
        )
        check = {
            "name": name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_error": float(error.max().item()),
            "cosine_min": cosine,
            "bad_fraction": bad_fraction,
        }
        checks.append(check)
        if bad_fraction != 0.0 or cosine < 0.996:
            raise RuntimeError(
                f"{name}: cosine={cosine:.9g}, "
                f"bad_fraction={bad_fraction:.9g}, "
                f"max_error={check['max_error']:.9g}"
            )
    return {"status": "passed", "checks": checks}
