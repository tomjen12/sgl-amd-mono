# SPDX-License-Identifier: Apache-2.0
"""Profiler evidence extraction for production TP8 Baseline dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tp8_baseline_dispatch import DispatchContractError, validate_dispatch


_GPU_CATEGORIES = {"kernel", "gpu_kernel", "cuda_kernel", "hip_kernel"}


def _trace_events(source: str | Path | Mapping[str, Any] | Iterable[Any]) -> list[Any]:
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text())
    if isinstance(source, Mapping):
        events = source.get("traceEvents")
        if not isinstance(events, list):
            raise DispatchContractError("profiler trace has no traceEvents list")
        return events
    return list(source)


def _mapping_kernel_name(event: Mapping[str, Any]) -> str | None:
    category = str(event.get("cat", "")).lower()
    args = event.get("args")
    args = args if isinstance(args, Mapping) else {}
    name = args.get("Kernel Name") or args.get("kernel_name")
    if name is None and category in _GPU_CATEGORIES:
        name = event.get("name")
    if name is None or category not in _GPU_CATEGORIES:
        return None
    return str(name)


def _object_kernel_name(event: Any) -> str | None:
    device_type = getattr(event, "device_type", None)
    device_name = str(device_type).lower()
    # torch.profiler uses DeviceType.CUDA for both CUDA and HIP activities.
    if not any(token in device_name for token in ("cuda", "hip", "gpu")):
        return None
    name = getattr(event, "name", None) or getattr(event, "key", None)
    return None if name is None else str(name)


def extract_kernel_names(
    source: str | Path | Mapping[str, Any] | Iterable[Any],
) -> list[str]:
    """Return GPU kernel launches, preserving launch order and duplicates."""
    names = []
    for event in _trace_events(source):
        name = (
            _mapping_kernel_name(event)
            if isinstance(event, Mapping)
            else _object_kernel_name(event)
        )
        if name:
            names.append(name)
    if not names:
        raise DispatchContractError(
            "profiler produced no GPU kernel launches; dispatch is unproven"
        )
    return names


def dispatch_evidence_from_profile(
    stage: str,
    *,
    operator: str,
    backend: str,
    profile: str | Path | Mapping[str, Any] | Iterable[Any],
) -> dict[str, Any]:
    """Extract measured launches and validate them before evidence is emitted."""
    names = extract_kernel_names(profile)
    evidence = validate_dispatch(
        stage,
        operator=operator,
        backend=backend,
        kernel_names=names,
    )
    evidence["actual"]["evidence_source"] = "profiler"
    return evidence


def write_validated_dispatch(path: str | Path, evidence: Mapping[str, Any]) -> None:
    """Write only evidence already accepted by the fail-closed validator."""
    if evidence.get("status") != "passed":
        raise DispatchContractError("refusing to write unvalidated dispatch evidence")
    actual = evidence.get("actual")
    if not isinstance(actual, Mapping) or actual.get("evidence_source") != "profiler":
        raise DispatchContractError("dispatch evidence is not profiler-derived")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
