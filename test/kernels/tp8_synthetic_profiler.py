# SPDX-License-Identifier: Apache-2.0
"""Timing and Torch-profiler helpers for the TP8 synthetic harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


def benchmark_us(
    launch: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    before_each: Callable[[], None] | None = None,
) -> list[float]:
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        launch()
        # Persistent collectives reuse mailbox/epoch storage.  Do not enqueue
        # the next logical model step before every rank has completed this one.
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        if before_each is not None:
            before_each()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) * 1000.0)
    return samples


def capture_dispatch(
    launch: Callable[[], None],
    trace_path: Path,
    *,
    before_each: Callable[[], None] | None = None,
    record: bool = True,
) -> list[str]:
    """Run two dispatches on every rank, but trace only the reporting rank."""
    if before_each is not None:
        before_each()
    launch()
    torch.cuda.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    if before_each is not None:
        before_each()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as profile:
        with torch.profiler.record_function("tp8_synthetic/measured_dispatch"):
            launch()
        torch.cuda.synchronize()
    if not record:
        return []
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(trace_path))
    names: list[str] = []
    for event in profile.events():
        if (
            event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.startswith("tp8_synthetic/")
        ):
            names.append(event.name)
    return names


def profile_kernel_iterations(
    launch: Callable[[], None],
    trace_path: Path,
    *,
    kernel_names: list[str],
    warmup: int,
    iterations: int,
    before_each: Callable[[], None] | None = None,
    record: bool = True,
) -> list[float]:
    """Run every rank but record kernel durations only on the reporting rank."""
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        launch()
        # Match sequential runtime steps and prevent a later launch from
        # reusing persistent-collective protocol state that is still live.
        torch.cuda.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as profile:
        for iteration in range(iterations):
            if before_each is not None:
                before_each()
            with torch.profiler.record_function(
                f"tp8_synthetic/iteration/{iteration}"
            ):
                launch()
                # This CPU-side wait is intentionally outside the reported
                # metric: only allowlisted GPU kernel durations are summed.
                torch.cuda.synchronize()
    if not record:
        return []
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text()).get("traceEvents", [])
    allowed = set(kernel_names)
    measured = [
        (str(event.get("name")), float(event.get("dur", 0.0)))
        for event in events
        if event.get("cat") == "kernel" and str(event.get("name")) in allowed
    ]
    launches = len(kernel_names)
    expected = iterations * launches
    if len(measured) != expected:
        raise RuntimeError(
            f"profiler selected {len(measured)} launches, expected {expected}; "
            f"allowlist={kernel_names}"
        )
    values: list[float] = []
    expected_order = list(kernel_names)
    for iteration in range(iterations):
        chunk = measured[iteration * launches : (iteration + 1) * launches]
        actual_order = [name for name, _duration in chunk]
        if actual_order != expected_order:
            raise RuntimeError(
                f"iteration {iteration}: kernel order {actual_order}, "
                f"expected {expected_order}"
            )
        values.append(sum(duration for _name, duration in chunk))
    return values
