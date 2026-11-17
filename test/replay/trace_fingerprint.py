#!/usr/bin/env python3
"""Normalize scoped PyTorch Chrome traces for replay parity checks.

The comparison intentionally ignores timestamps, addresses, process/thread IDs,
correlation IDs and generated compile hashes.  It keeps operator/kernel order,
launch geometry, shape/dtype metadata and CPU-to-GPU correlation structure.
Checkpoint-only FP4 storage reinterpretations are also ignored because the
config-only Target replay starts from the equivalent packed uint8 weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CPU_CATEGORIES = frozenset(
    {"cpu_op", "python_function", "user_annotation", "operator"}
)
KERNEL_CATEGORIES = frozenset({"kernel", "gpu_kernel", "cuda_kernel", "hip_kernel"})
RUNTIME_CATEGORIES = frozenset({"cuda_runtime", "hip_runtime"})
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_HASH = re.compile(r"(?<=[_.-])[0-9a-fA-F]{12,64}(?=($|[_.-]))")
_HIP_CLONE = re.compile(r"\s*\[clone\s+\.kd\]\s*$")
_SPACE = re.compile(r"\s+")


def normalize_name(name: object) -> str:
    """Remove run-specific text while preserving the executable/operator name."""
    value = _ADDRESS.sub("0x*", str(name))
    value = _HASH.sub("<hash>", value)
    value = _HIP_CLONE.sub("", value)
    return _SPACE.sub(" ", value).strip()


def _category(event: dict[str, Any]) -> str:
    return str(event.get("cat", "")).lower()


def _duration(event: dict[str, Any]) -> float:
    try:
        return float(event.get("dur", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _timestamp(event: dict[str, Any]) -> float:
    try:
        return float(event.get("ts", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _complete_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in trace.get("traceEvents", ())
        if isinstance(event, dict) and event.get("ph") == "X"
    ]


def _arg(args: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower().replace("_", " "): value for key, value in args.items()}
    for name in names:
        key = name.lower().replace("_", " ")
        if key in lowered:
            return lowered[key]
    return None


def _external_id(event: dict[str, Any]) -> str | None:
    value = _arg(event.get("args") or {}, "external id", "external_id")
    return None if value is None else str(value)


def _correlation_id(event: dict[str, Any]) -> str | None:
    value = _arg(event.get("args") or {}, "correlation", "correlation id")
    return None if value is None else str(value)


def _selected_cpu_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args") or {}
    selected = {}
    for output, aliases in {
        "input_dims": ("input dims", "input shapes", "shapes"),
        "input_types": ("input type", "input types", "dtypes"),
        "concrete_inputs": ("concrete inputs",),
        "collective_name": ("collective name",),
        "collective_dtype": ("dtype",),
        "input_elements": (
            "in msg nelems",
            "input elements",
            "input size",
        ),
        "output_elements": (
            "out msg nelems",
            "output elements",
            "output size",
        ),
        "group_size": ("group size", "world size"),
    }.items():
        value = _arg(args, *aliases)
        if value is not None:
            selected[output] = value
    return selected


def _is_checkpoint_fp4_storage_view(
    name: str, selected_args: dict[str, Any]
) -> bool:
    """Ignore a storage-only view absent from config-only packed dummy weights."""
    if name != "aten::view":
        return False
    input_types = selected_args.get("input_types") or ()
    concrete_inputs = selected_args.get("concrete_inputs") or ()
    return (
        bool(input_types)
        and str(input_types[0]).startswith("c10::Float4_")
        and len(concrete_inputs) > 1
        and str(concrete_inputs[1]) == "0"
    )


def _selected_kernel_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args") or {}
    selected = {}
    for output, aliases in {
        "grid": ("grid", "grid size"),
        "block": ("block", "block size"),
        "shared_memory": (
            "shared memory",
            "shared memory usage",
            "static shared memory",
        ),
    }.items():
        value = _arg(args, *aliases)
        if value is not None:
            selected[output] = value
    return selected


def _inside(event: dict[str, Any], scope: dict[str, Any]) -> bool:
    start, end = _timestamp(scope), _timestamp(scope) + _duration(scope)
    event_start = _timestamp(event)
    return start <= event_start < end


def _ordered(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Chrome event order breaks ties that share the same timestamp.
    return sorted(events, key=_timestamp)


def fingerprint_trace(
    trace: dict[str, Any],
    *,
    scope_pattern: str = r"^muse_replay/",
    scope_occurrence: int | None = None,
) -> dict[str, Any]:
    """Return deterministic fingerprints for matching user-annotation scopes."""
    if scope_occurrence is not None and scope_occurrence < 0:
        raise ValueError("scope_occurrence must be non-negative")
    events = _complete_events(trace)
    scope_re = re.compile(scope_pattern)
    scopes = [
        event
        for event in events
        if scope_re.search(str(event.get("name", "")))
        and _category(event) in CPU_CATEGORIES
    ]
    if scope_occurrence is not None:
        occurrences: dict[str, int] = defaultdict(int)
        selected = []
        for scope in _ordered(scopes):
            name = normalize_name(scope.get("name", ""))
            occurrence = occurrences[name]
            occurrences[name] += 1
            if occurrence == scope_occurrence:
                selected.append(scope)
        scopes = selected
    result: list[dict[str, Any]] = []
    for scope in _ordered(scopes):
        cpu_events = [
            event
            for event in events
            if event is not scope
            and _category(event) in CPU_CATEGORIES
            and _inside(event, scope)
        ]
        cpu_events = _ordered(cpu_events)
        parents: dict[str, list[str]] = defaultdict(list)
        cpu_sequence = []
        for event in cpu_events:
            name = normalize_name(event.get("name", ""))
            selected_args = _selected_cpu_args(event)
            if _is_checkpoint_fp4_storage_view(name, selected_args):
                continue
            external_id = _external_id(event)
            if external_id is not None:
                parents[external_id].append(name)
            cpu_sequence.append(
                {
                    "name": name,
                    "args": selected_args,
                }
            )

        # ROCm kernels can execute well after the CPU scope closes. Associate
        # them through the runtime launch's correlation id; use the interval
        # fallback only for traces that do not expose launch correlations.
        launch_parents: dict[str, str | None] = {}
        for event in events:
            if _category(event) not in RUNTIME_CATEGORIES or not _inside(event, scope):
                continue
            correlation = _correlation_id(event)
            if correlation is None or "launch" not in str(event.get("name", "")).lower():
                continue
            cpu_id = _arg(event.get("args") or {}, "cid", "external id")
            launch_parents[correlation] = (
                parents.get(str(cpu_id), [])[-1]
                if cpu_id is not None and str(cpu_id) in parents
                else None
            )

        kernel_events = [
            event
            for event in events
            if _category(event) in KERNEL_CATEGORIES
            and (
                (
                    _correlation_id(event) in launch_parents
                    if launch_parents
                    else _inside(event, scope)
                )
                or (
                    _external_id(event) is not None
                    and _external_id(event) in parents
                )
            )
        ]
        kernel_sequence = []
        ordered_kernels = _ordered(kernel_events)
        for event in ordered_kernels:
            external_id = _external_id(event)
            correlation = _correlation_id(event)
            kernel_sequence.append(
                {
                    "name": normalize_name(event.get("name", "")),
                    "args": _selected_kernel_args(event),
                    "cpu_parent": parents.get(external_id, [])[-1]
                    if external_id in parents
                    else launch_parents.get(correlation),
                }
            )
        result.append(
            {
                "scope": normalize_name(scope.get("name", "")),
                "cpu_ops": cpu_sequence,
                "gpu_kernels": kernel_sequence,
                "metrics": {
                    "gpu_kernel_duration_us": sum(
                        _duration(event) for event in ordered_kernels
                    ),
                    "gpu_span_us": (
                        max(_timestamp(event) + _duration(event) for event in ordered_kernels)
                        - min(_timestamp(event) for event in ordered_kernels)
                        if ordered_kernels
                        else 0.0
                    ),
                },
            }
        )
    return {
        "schema_version": 1,
        "scope_pattern": scope_pattern,
        "scopes": result,
    }


def compare_fingerprints(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    gpu_time_tolerance: float | None = None,
) -> list[str]:
    """Return human-readable parity failures; an empty list means exact parity."""
    failures: list[str] = []
    expected_scopes = expected.get("scopes", [])
    actual_scopes = actual.get("scopes", [])

    def events_match(want_event: dict[str, Any], got_event: dict[str, Any]) -> bool:
        if want_event.get("name") != got_event.get("name"):
            return False
        if want_event.get("args", {}) != got_event.get("args", {}):
            return False
        # Some ROCm profiler runs omit the runtime launch's CPU linkage even
        # when kernel order and geometry are intact. Compare the parent when
        # both traces report one; absence on one side is missing telemetry.
        want_parent = want_event.get("cpu_parent")
        got_parent = got_event.get("cpu_parent")
        return not (want_parent and got_parent and want_parent != got_parent)

    if len(expected_scopes) != len(actual_scopes):
        failures.append(
            f"scope count: expected {len(expected_scopes)}, got {len(actual_scopes)}"
        )
    for index, (want, got) in enumerate(zip(expected_scopes, actual_scopes)):
        if want.get("scope") != got.get("scope"):
            failures.append(
                f"scope {index}: expected {want.get('scope')!r}, got {got.get('scope')!r}"
            )
            continue
        for field in ("cpu_ops", "gpu_kernels"):
            want_events, got_events = want.get(field, []), got.get(field, [])
            if len(want_events) != len(got_events):
                failures.append(
                    f"{want['scope']} {field} count: "
                    f"expected {len(want_events)}, got {len(got_events)}"
                )
            for event_index, (want_event, got_event) in enumerate(
                zip(want_events, got_events)
            ):
                if not events_match(want_event, got_event):
                    failures.append(
                        f"{want['scope']} {field}[{event_index}]: "
                        f"expected {want_event!r}, got {got_event!r}"
                    )
                    break
        if gpu_time_tolerance is not None:
            want_time = float(want.get("metrics", {}).get("gpu_span_us", 0.0))
            got_time = float(got.get("metrics", {}).get("gpu_span_us", 0.0))
            if want_time > 0:
                relative_error = abs(got_time - want_time) / want_time
                if relative_error > gpu_time_tolerance:
                    failures.append(
                        f"{want['scope']} gpu_span_us: expected {want_time:.3f}, "
                        f"got {got_time:.3f} ({relative_error:.1%} difference; "
                        f"limit {gpu_time_tolerance:.1%})"
                    )
    return failures


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    handle = (
        gzip.open(source, mode="rt", encoding="utf-8")
        if source.suffix == ".gz"
        else source.open(mode="r", encoding="utf-8")
    )
    with handle:
        return json.load(handle)


def _write(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("trace")
    extract.add_argument("output")
    extract.add_argument("--scope-pattern", default=r"^muse_replay/")
    extract.add_argument("--scope-occurrence", type=int)
    compare = subparsers.add_parser("compare")
    compare.add_argument("expected")
    compare.add_argument("actual")
    compare.add_argument(
        "--gpu-time-tolerance",
        type=float,
        default=None,
        help="Optional relative tolerance, e.g. 0.10 for graph latency parity.",
    )
    args = parser.parse_args()

    if args.command == "extract":
        _write(
            args.output,
            fingerprint_trace(
                _load(args.trace),
                scope_pattern=args.scope_pattern,
                scope_occurrence=args.scope_occurrence,
            ),
        )
        return 0
    failures = compare_fingerprints(
        _load(args.expected),
        _load(args.actual),
        gpu_time_tolerance=args.gpu_time_tolerance,
    )
    if failures:
        print("\n".join(failures))
        return 1
    print("Trace fingerprints match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
