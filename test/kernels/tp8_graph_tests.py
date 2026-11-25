# SPDX-License-Identifier: Apache-2.0
"""Width-8/16 CUDA Graph capture/replay tests for the TP8 synthetic harness."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

from tp8_numerical_parity import compare_outputs
from tp8_synthetic_contract import CONTRACT, ContractError


STAGES = (
    "k1",
    "kattn_full",
    "kattn_window",
    "k2",
    "k3",
    "layer_v1_full",
    "layer_v1_swa",
    "layer_v2_full",
    "layer_v2_swa",
)


@dataclass
class GraphFixture:
    implementation: str
    stage: str
    width: int
    launch: Callable[[], None]
    before_step: Callable[[int], None]
    outputs: Callable[[], dict[str, torch.Tensor]]
    addresses: Callable[[], dict[str, int]]
    close: Callable[[], None]
    launch_count: int
    operator: str
    backend: str
    result_boxes: list[dict[str, Any]]
    validate_reference: Callable[[dict[str, torch.Tensor]], dict[str, Any]] | None


def _tensor_addresses(value: Any, prefix: str = "", depth: int = 0) -> dict[str, int]:
    if depth > 3:
        return {}
    if isinstance(value, torch.Tensor):
        return {prefix or "tensor": int(value.data_ptr())}
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for name, item in value.items():
            out.update(_tensor_addresses(item, f"{prefix}.{name}".strip("."), depth + 1))
        return out
    if is_dataclass(value):
        out = {}
        for field in fields(value):
            out.update(
                _tensor_addresses(
                    getattr(value, field.name),
                    f"{prefix}.{field.name}".strip("."),
                    depth + 1,
                )
            )
        return out
    if isinstance(value, (tuple, list)):
        out = {}
        for index, item in enumerate(value):
            out.update(
                _tensor_addresses(item, f"{prefix}.{index}".strip("."), depth + 1)
            )
        return out
    return {}


def _baseline_outputs(stage: str, value: Any) -> dict[str, torch.Tensor]:
    if stage == "k1":
        return {
            "q": value.value[0],
            "k": value.value[1],
            "v": value.value[2],
            "gate": value.value[3],
        }
    if stage in ("kattn_full", "kattn_window"):
        return {"output": value.value}
    if stage == "k2":
        return {
            "stream": value.value[0],
            "scores": value.value[1],
            "ids": value.value[2],
            "expert_input": value.value[3],
        }
    if stage == "k3":
        return {
            "attention_input": value.value[0],
            "stream": value.value[1],
        }
    raise ValueError(stage)


def _dynamic_tensors(stage: str, fixture: Any) -> dict[str, torch.Tensor]:
    if stage == "k1":
        return {
            "x": fixture.x,
            "positions": fixture.positions,
            "cache_k": fixture.cache_k,
            "cache_v": fixture.cache_v,
        }
    if stage in ("kattn_full", "kattn_window"):
        return {
            "q": fixture.q,
            "gate": fixture.gate,
            "cache_k": fixture.cache_k,
            "cache_v": fixture.cache_v,
        }
    if stage == "k2":
        return {
            "attention_output": fixture.attention_output,
            "stream": fixture.stream,
        }
    if stage == "k3":
        return {
            "xin": fixture.xin,
            "stream": fixture.stream,
        }
    raise ValueError(stage)


def _make_updater(stage: str, fixture: Any) -> tuple[Callable[[int], None], dict]:
    dynamic = _dynamic_tensors(stage, fixture)
    base = {name: value.clone() for name, value in dynamic.items()}

    def update(step: int) -> None:
        for name, value in dynamic.items():
            value.copy_(base[name])
        delta = step * 2.0**-8
        if stage == "k1":
            fixture.x.add_(delta)
            fixture.positions.add_(step)
            fixture.cache_k.zero_()
            fixture.cache_v.zero_()
        elif stage in ("kattn_full", "kattn_window"):
            fixture.q.add_(delta)
            fixture.gate.sub_(delta)
            # Change graph-static KV state without reallocating the 8K fixture.
            fixture.cache_k.reshape(-1)[:256].copy_(
                base["cache_k"].reshape(-1)[:256]
            )
            fixture.cache_v.reshape(-1)[:256].copy_(
                base["cache_v"].reshape(-1)[:256]
            )
            if step:
                fixture.cache_k.reshape(-1)[:256].copy_(
                    base["cache_k"].reshape(-1)[256:512]
                )
                fixture.cache_v.reshape(-1)[:256].copy_(
                    base["cache_v"].reshape(-1)[256:512]
                )
        elif stage == "k2":
            fixture.attention_output.add_(delta)
            fixture.stream.sub_(delta)
        elif stage == "k3":
            fixture.xin.add_(delta)
            fixture.stream.sub_(delta)

    return update, base


def _reference_validator(
    implementation: str,
    stage: str,
    width: int,
    fixture: Any,
) -> Callable[[dict[str, torch.Tensor]], dict[str, Any]] | None:
    if stage == "k1":
        from test_muse_spark_gfx1250_k1 import (
            _close,
            _fp8,
            _one_ulp_fp8,
            _reference,
        )

        def validate_k1(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
            q, k, v, gate = _reference(
                fixture.x,
                fixture.weight,
                fixture.cos_sin,
                fixture.positions,
                True,
            )
            _close(outputs["q"], q, 1 / 64, "q")
            _close(outputs["k"], k, 1 / 64, "k")
            _close(outputs["v"], v, 1 / 128, "v")
            _close(outputs["gate"], gate, 1 / 128, "gate")
            _one_ulp_fp8(
                outputs["cache_k"].reshape(width, -1), _fp8(k), "K cache"
            )
            _one_ulp_fp8(
                outputs["cache_v"].reshape(width, -1), _fp8(v), "V cache"
            )
            return {"status": "passed", "kind": "strict-pytorch-k1"}

        return validate_k1
    if stage in ("kattn_full", "kattn_window"):
        from tp8_logical_case import kattn_post_norm_gate, kattn_reference

        def validate_kattn(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
            reference = kattn_reference(
                fixture,
                fp8_qp=(
                    implementation == "baseline" and stage == "kattn_full"
                ),
            )
            if implementation == "baseline":
                reference = kattn_post_norm_gate(reference, fixture.gate)
            comparison = compare_outputs({"output": reference}, outputs)
            comparison["kind"] = "strict-pytorch-kattn-addressed-8k"
            return comparison

        return validate_kattn
    return None


def _prepare_baseline_stage(
    stage: str,
    width: int,
    rank: int,
    world: int,
    controlled_package=None,
) -> GraphFixture:
    from tp8_baseline_fixtures import prepare_k1, prepare_k2, prepare_k3, prepare_kattn

    device = torch.device("cuda", rank)
    prepared = (
        prepare_k1(width, rank, device)
        if stage == "k1"
        else prepare_kattn(
            width,
            rank,
            device,
            window=2047 if stage == "kattn_window" else 0,
        )
        if stage in ("kattn_full", "kattn_window")
        else prepare_k2(width, rank, device, active_npes=world)
        if stage == "k2"
        else prepare_k3(
            width,
            rank,
            device,
            active_npes=world,
            controlled_package=controlled_package,
        )
    )
    if prepared.fixture is None:
        raise ContractError(f"{stage} Baseline fixture is missing")
    update, _base = _make_updater(stage, prepared.fixture)
    box: dict[str, Any] = {}

    def launch() -> None:
        box["result"] = prepared.launch()

    def outputs() -> dict[str, torch.Tensor]:
        if "result" not in box:
            raise ContractError(f"{stage} Baseline graph produced no result")
        out = _baseline_outputs(stage, box["result"])
        if stage == "k1":
            out.update(
                cache_k=prepared.fixture.cache_k[prepared.fixture.locations],
                cache_v=prepared.fixture.cache_v[prepared.fixture.locations],
            )
        return {name: value.clone() for name, value in out.items()}

    return GraphFixture(
        implementation="baseline",
        stage=stage,
        width=width,
        launch=launch,
        before_step=update,
        outputs=outputs,
        addresses=lambda: _tensor_addresses(prepared.fixture, "fixture"),
        close=lambda: None,
        launch_count=prepared.launch_count,
        operator=prepared.operator,
        backend=prepared.backend,
        result_boxes=[box],
        validate_reference=_reference_validator(
            "baseline", stage, width, prepared.fixture
        ),
    )


def _prepare_mono_stage(
    stage: str,
    width: int,
    rank: int,
    world: int,
    peer_group,
    controlled_package=None,
) -> GraphFixture:
    from tp8_logical_case import (
        build_k1_case,
        build_k2_case,
        build_k3_case,
        build_kattn_case,
    )
    from tp8_mono_fixtures import prepare_mono

    device = torch.device("cuda", rank)
    case = (
        build_k1_case(width, rank, device)
        if stage == "k1"
        else build_kattn_case(
            width,
            rank,
            2047 if stage == "kattn_window" else 0,
            device,
        )
        if stage in ("kattn_full", "kattn_window")
        else build_k2_case(width, rank, world, device)
        if stage == "k2"
        else build_k3_case(width, rank, world, device)
    )
    if controlled_package is not None:
        if stage != "k3":
            raise ContractError("controlled package is legal only for K3")
        case.xin.copy_(
            torch.tensor(controlled_package.xin, dtype=case.xin.dtype, device=device)
        )
        case.ids.copy_(
            torch.tensor(controlled_package.ids, dtype=case.ids.dtype, device=device)
        )
        case.scores.copy_(
            torch.tensor(
                controlled_package.scores,
                dtype=case.scores.dtype,
                device=device,
            )
        )
    prepared = prepare_mono(
        stage,
        width,
        rank,
        window=stage == "kattn_window",
        peer_group=peer_group,
        active_npes=world,
        logical_case=case,
    )
    update_case, _base = _make_updater(stage, case)
    if prepared.protocol is None or prepared.outputs is None:
        prepared.close()
        raise ContractError(f"{stage} Mono graph fixture is incomplete")
    buffers = prepared.protocol["buffers"]

    def before_step(step: int) -> None:
        update_case(step)
        if stage in ("kattn_full", "kattn_window"):
            buffers.q[:width].copy_(case.q)
            buffers.g[:width].copy_(case.gate)
        elif stage == "k2":
            buffers.o[:width].copy_(case.o)
            buffers.g[:width].copy_(case.gate)
        elif stage == "k3":
            buffers.xin[:width].copy_(case.xin)
            buffers.ids[:width].copy_(case.ids)
            buffers.scores[:width].copy_(case.scores)
        prepared.before_each()

    def addresses() -> dict[str, int]:
        values = _tensor_addresses(case, "fixture")
        values.update(_tensor_addresses(prepared.protocol["epoch"], "epoch"))
        values.update(_tensor_addresses(buffers.__dict__, "buffers"))
        values.update(
            {
                f"shadow.{index}": int(value.data_ptr())
                for index, value in enumerate(buffers.peers.shadow_bytes)
            }
        )
        return values

    return GraphFixture(
        implementation="mono",
        stage=stage,
        width=width,
        launch=prepared.launch,
        before_step=before_step,
        outputs=prepared.outputs,
        addresses=addresses,
        close=prepared.close,
        launch_count=1,
        operator="muse_spark_monokernel",
        backend="sgl_amd_mono_gfx1250",
        result_boxes=[],
        validate_reference=_reference_validator("mono", stage, width, case),
    )


def _layer_stage_parts(stage: str) -> tuple[str, ...]:
    attention = "kattn_window" if stage.endswith("_swa") else "kattn_full"
    parts = [attention, "k2", "k3"]
    if stage.startswith("layer_v2"):
        parts.append("k1")
    return tuple(parts)


def _prepare_baseline_layer(
    stage: str,
    width: int,
    rank: int,
    world: int,
) -> GraphFixture:
    parts = [
        _prepare_baseline_stage(name, width, rank, world)
        for name in _layer_stage_parts(stage)
    ]

    def launch() -> None:
        for part in parts:
            part.launch()

    def before_step(step: int) -> None:
        for part in parts:
            part.before_step(step)

    def outputs() -> dict[str, torch.Tensor]:
        out = {}
        for part in parts:
            out.update(
                {
                    f"{part.stage}.{name}": value
                    for name, value in part.outputs().items()
                }
            )
        return out

    def addresses() -> dict[str, int]:
        out = {}
        for part in parts:
            out.update(
                {
                    f"{part.stage}.{name}": address
                    for name, address in part.addresses().items()
                }
            )
        return out

    return GraphFixture(
        implementation="baseline",
        stage=stage,
        width=width,
        launch=launch,
        before_step=before_step,
        outputs=outputs,
        addresses=addresses,
        close=lambda: [part.close() for part in parts],
        launch_count=sum(part.launch_count for part in parts),
        operator="+".join(part.operator for part in parts),
        backend="+".join(part.backend for part in parts),
        result_boxes=[box for part in parts for box in part.result_boxes],
        validate_reference=None,
    )


def _prepare_mono_layer(
    stage: str,
    width: int,
    rank: int,
    world: int,
    peer_group,
) -> GraphFixture:
    from tp8_mono_fixtures import prepare_mono

    fixture_stage = stage.rsplit("_", 1)[0]
    prepared = prepare_mono(
        fixture_stage,
        width,
        rank,
        window=stage.endswith("_swa"),
        peer_group=peer_group,
        active_npes=world,
    )
    if prepared.protocol is None or prepared.outputs is None:
        prepared.close()
        raise ContractError(f"{stage} Mono graph fixture is incomplete")
    buffers = prepared.protocol["buffers"]
    layer = prepared.protocol["layer"]
    dynamic = {
        "q": buffers.q[:width],
        "k": buffers.k[:width],
        "v": buffers.v[:width],
        "gate": buffers.g[:width],
        "residual": buffers.res[:width],
        "cache_k": layer.k_cache,
        "cache_v": layer.v_cache,
    }
    base = {name: value.clone() for name, value in dynamic.items()}

    def before_step(step: int) -> None:
        for name, value in dynamic.items():
            value.copy_(base[name])
        delta = step * 2.0**-8
        buffers.q[:width].add_(delta)
        buffers.g[:width].sub_(delta)
        buffers.res[:width].add_(delta)
        if step:
            layer.k_cache.reshape(-1)[:256].copy_(
                base["cache_k"].reshape(-1)[256:512]
            )
            layer.v_cache.reshape(-1)[:256].copy_(
                base["cache_v"].reshape(-1)[256:512]
            )
        prepared.before_each()

    def addresses() -> dict[str, int]:
        out = _tensor_addresses(buffers.__dict__, "buffers")
        out.update(_tensor_addresses(dynamic, "dynamic"))
        out.update(_tensor_addresses(prepared.protocol["epoch"], "epoch"))
        return out

    return GraphFixture(
        implementation="mono",
        stage=stage,
        width=width,
        launch=prepared.launch,
        before_step=before_step,
        outputs=prepared.outputs,
        addresses=addresses,
        close=prepared.close,
        launch_count=1,
        operator="muse_spark_fused_layer",
        backend="sgl_amd_mono_gfx1250",
        result_boxes=[],
        validate_reference=None,
    )


def prepare_graph_fixture(
    implementation: str,
    stage: str,
    width: int,
    rank: int,
    world: int,
    peer_group,
    controlled_package=None,
) -> GraphFixture:
    if stage not in STAGES:
        raise ContractError(f"unsupported graph stage {stage}")
    if stage.startswith("layer_"):
        return (
            _prepare_baseline_layer(stage, width, rank, world)
            if implementation == "baseline"
            else _prepare_mono_layer(stage, width, rank, world, peer_group)
        )
    return (
        _prepare_baseline_stage(
            stage, width, rank, world, controlled_package=controlled_package
        )
        if implementation == "baseline"
        else _prepare_mono_stage(
            stage,
            width,
            rank,
            world,
            peer_group,
            controlled_package=controlled_package,
        )
    )


def _profile_kernels(call: Callable[[], None]) -> tuple[list[str], float]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as profile:
        call()
        torch.cuda.synchronize()
    events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    names = [event.name for event in events]
    pure_us = sum(float(event.device_time_total) for event in events)
    return names, pure_us


def _capture(fixture: GraphFixture, barrier: Callable[[], None]) -> torch.cuda.CUDAGraph:
    fixture.before_step(0)
    barrier()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    # Warm up on the same side stream used for capture. Besides being the
    # documented graph pattern, this keeps stream-scoped allocator/kernel
    # setup out of the captured production stage.
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        fixture.launch()
    capture_stream.synchronize()
    barrier()
    with torch.cuda.graph(graph, stream=capture_stream):
        fixture.launch()
    torch.cuda.synchronize()
    barrier()
    return graph


def run_graph_case(
    *,
    implementation: str,
    stage: str,
    width: int,
    rank: int,
    world: int,
    peer_group,
    replays: int = 3,
    controlled_package=None,
) -> dict[str, Any]:
    graph_fixture = prepare_graph_fixture(
        implementation,
        stage,
        width,
        rank,
        world,
        peer_group,
        controlled_package,
    )
    eager_fixture = prepare_graph_fixture(
        implementation,
        stage,
        width,
        rank,
        world,
        peer_group,
        controlled_package,
    )

    def barrier() -> None:
        dist.barrier(group=peer_group)

    def validate_reference_collectively(
        outputs: dict[str, torch.Tensor],
        label: str,
    ) -> dict[str, Any]:
        if graph_fixture.validate_reference is None:
            return {
                "status": "not_available",
                "kind": (
                    "existing stage gate only; independent layer reference "
                    "is Phase C"
                    if stage.startswith("layer_")
                    else "existing distributed PyTorch stage gate"
                ),
            }
        result = None
        local_error = None
        try:
            result = graph_fixture.validate_reference(outputs)
        except Exception as exc:
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * world
        dist.all_gather_object(errors, local_error, group=peer_group)
        failures = [error for error in errors if error]
        if failures:
            return {
                "status": "failed",
                "kind": "strict-pytorch-reference",
                "error": f"{label}: " + " | ".join(failures),
            }
        assert result is not None
        return result

    try:
        # Materialize lazy fixture views/result storage before address
        # attestation; capture/replay must not create or replace them.
        graph_fixture.before_step(0)
        barrier()
        graph_fixture.launch()
        torch.cuda.synchronize()
        initial_reference = validate_reference_collectively(
            graph_fixture.outputs(), "initial"
        )
        initial_addresses = graph_fixture.addresses()
        graph = _capture(graph_fixture, barrier)
        capture_addresses = graph_fixture.addresses()
        if capture_addresses != initial_addresses:
            raise ContractError(f"{implementation} {stage}: capture changed addresses")

        steps = []
        for step in range(1, replays + 1):
            eager_fixture.before_step(step)
            barrier()
            eager_fixture.launch()
            torch.cuda.synchronize()
            expected = eager_fixture.outputs()

            graph_fixture.before_step(step)
            barrier()
            graph.replay()
            torch.cuda.synchronize()
            actual = graph_fixture.outputs()
            comparison = compare_outputs(expected, actual)
            validate_this_step = (
                graph_fixture.validate_reference is not None
                and (
                    stage not in ("kattn_full", "kattn_window")
                    or step == replays
                )
            )
            pytorch_reference = (
                validate_reference_collectively(
                    actual, f"replay {step}"
                )
                if validate_this_step
                else {
                    "status": "not_available",
                    "kind": (
                        "8K reference reserved for initial and final replay"
                        if (
                            graph_fixture.validate_reference is not None
                            and stage in ("kattn_full", "kattn_window")
                        )
                        else (
                            "existing stage gate only; independent layer reference "
                            "is Phase C"
                            if stage.startswith("layer_")
                            else "existing distributed PyTorch stage gate"
                        )
                    ),
                }
            )
            addresses = graph_fixture.addresses()
            if addresses != initial_addresses:
                raise ContractError(
                    f"{implementation} {stage} step {step}: addresses changed"
                )
            steps.append(
                {
                    "step": step,
                    "comparison": comparison,
                    "pytorch_reference": pytorch_reference,
                    "addresses_stable": True,
                }
            )

        eager_fixture.before_step(replays + 1)
        barrier()
        eager_names, _eager_us = _profile_kernels(eager_fixture.launch)
        graph_fixture.before_step(replays + 1)
        barrier()
        graph_names, pure_us = _profile_kernels(graph.replay)
        if eager_names != graph_names:
            raise ContractError(
                f"{implementation} {stage}: graph kernels {graph_names}, "
                f"expected eager {eager_names}"
            )
        if len(graph_names) != graph_fixture.launch_count:
            raise ContractError(
                f"{implementation} {stage}: graph launches {len(graph_names)}, "
                f"expected {graph_fixture.launch_count}"
            )

        graph_fixture.before_step(replays + 2)
        barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        envelope_us = float(start.elapsed_time(stop)) * 1000.0
        final_addresses = graph_fixture.addresses()
        if final_addresses != initial_addresses:
            raise ContractError(f"{implementation} {stage}: final addresses changed")
        reference_failures = []
        if initial_reference["status"] == "failed":
            reference_failures.append(initial_reference["error"])
        reference_failures.extend(
            step["pytorch_reference"]["error"]
            for step in steps
            if step["pytorch_reference"]["status"] == "failed"
        )
        return {
            "status": "failed" if reference_failures else "passed",
            "graph_status": "passed",
            **(
                {"error": " | ".join(reference_failures)}
                if reference_failures
                else {}
            ),
            "implementation": implementation,
            "stage": stage,
            "width": width,
            "rank": rank,
            "active_npes": world,
            "logical_npes": CONTRACT.logical_tp,
            "replays": replays,
            "initial_pytorch_reference": initial_reference,
            "operator": graph_fixture.operator,
            "kernel_names": graph_names,
            "launch_count": len(graph_names),
            "expected_launch_count": graph_fixture.launch_count,
            "addresses": initial_addresses,
            "addresses_stable": True,
            "steps": steps,
            "timing_support": {
                "graph_envelope_us": envelope_us,
                "pure_stage_kernel_us": pure_us,
                "profiling_kind": "single-replay-support-smoke",
                "final_profile": False,
            },
            "reference_kind": (
                "strict graph-vs-eager same-fixture; stage PyTorch gates separate"
            ),
        }
    finally:
        graph_fixture.close()
        eager_fixture.close()
