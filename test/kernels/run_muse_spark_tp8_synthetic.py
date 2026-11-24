# SPDX-License-Identifier: Apache-2.0
"""Four-rank entry point for TP8-local Muse Spark kernel development.

Phase 1 implements the independent shape contract and distributed manifest.
Correctness and benchmark dispatch are intentionally added in later phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tp8_synthetic_contract import (  # noqa: E402
    CONTRACT,
    ContractError,
    deterministic_ep8_routing,
    ep8_expert_coverage_suite,
    local_expert_range,
    parse_widths,
    rank_manifest,
    routing_histogram,
    runtime_workload_manifest,
    specialization_attestation,
    validate_manifests,
    validate_ep8_expert_coverage,
    validate_workload_manifests,
    workload_manifest,
)


def _run_baseline_benchmark(
    args: argparse.Namespace,
    widths: tuple[int, ...],
    rank: int,
    world: int,
) -> None:
    from tp8_baseline_dispatch import validate_dispatch
    from tp8_baseline_fixtures import (
        prepare_k1,
        prepare_k2,
        prepare_k3,
        prepare_kattn,
    )
    from tp8_synthetic_profiler import (
        benchmark_us,
        capture_dispatch,
        profile_kernel_iterations,
    )
    from tp8_synthetic_report import Sample, write_reports

    derive_layer = args.stage in ("all", "layer")
    stages = (
        ("k1", "kattn_full", "kattn_window", "k2", "k3")
        if args.stage == "all"
        else ("k1", "kattn_full", "kattn_window", "k2", "k3")
        if args.stage == "layer"
        else (
            ("kattn_full", "kattn_window")
            if args.stage == "kattn"
            else (args.stage,)
        )
    )
    all_samples: list[Sample] = []
    all_evidence: list[dict] = []
    for rows in widths:
        for stage in stages:
            if stage == "k1":
                prepared = prepare_k1(rows, rank, torch.device("cuda"))
            elif stage == "kattn_full":
                prepared = prepare_kattn(
                    rows, rank, torch.device("cuda"), window=False
                )
            elif stage == "kattn_window":
                prepared = prepare_kattn(
                    rows, rank, torch.device("cuda"), window=2047
                )
            elif stage == "k2":
                prepared = prepare_k2(
                    rows,
                    rank,
                    torch.device("cuda"),
                    active_npes=world,
                )
            elif stage == "k3":
                prepared = prepare_k3(
                    rows,
                    rank,
                    torch.device("cuda"),
                    active_npes=world,
                )
            else:
                raise NotImplementedError(f"Baseline stage {stage!r}")
            dist.barrier()
            trace = (
                args.output_dir
                / "traces"
                / f"{stage}-w{rows}-rank{rank}.json"
            )
            captured = {}

            def recorded_launch():
                result = prepared.launch()
                captured["result"] = result
                return result

            names = capture_dispatch(recorded_launch, trace)
            result = captured.get("result")
            if result is None:
                raise ContractError(
                    f"Baseline {stage} did not return tensor evidence"
                )
            evidence = validate_dispatch(
                stage,
                operator=prepared.operator,
                backend=prepared.backend,
                kernel_names=names,
                tensor_metadata=result.tensors,
                active_npes=world,
            )
            evidence.update(stage=stage, width=rows, rank=rank)
            evidence["workload"] = runtime_workload_manifest(
                implementation="baseline",
                stage=stage,
                width=rows,
                rank=rank,
                active_npes=world,
            )
            all_evidence.append(evidence)
            dist.barrier()
            timings = (
                profile_kernel_iterations(
                    prepared.launch,
                    args.output_dir
                    / "traces"
                    / f"profile{args.iterations}-{stage}-w{rows}-rank{rank}.json",
                    kernel_names=names,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    before_each=dist.barrier,
                    record=rank == 0,
                )
                if args.profile == "torch"
                else benchmark_us(
                    prepared.launch,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    before_each=dist.barrier,
                )
            )
            all_samples.extend(
                Sample(
                    run_id="tp8-synthetic",
                    stage=stage,
                    width=rows,
                    rank=rank,
                    implementation="baseline",
                    iteration=index,
                    launch_count=prepared.launch_count,
                    total_us=value,
                )
                for index, value in enumerate(timings)
            )
            dist.barrier()
    if derive_layer:
        grouped = {
            (sample.width, sample.iteration, sample.stage): sample
            for sample in all_samples
        }
        for rows in widths:
            for iteration in range(args.iterations):
                for stage, parts in (
                    ("layer_v1_full", ("kattn_full", "k2", "k3")),
                    ("layer_v1_swa", ("kattn_window", "k2", "k3")),
                    (
                        "layer_v2_full",
                        ("kattn_full", "k2", "k3", "k1"),
                    ),
                    (
                        "layer_v2_swa",
                        ("kattn_window", "k2", "k3", "k1"),
                    ),
                ):
                    items = [grouped[(rows, iteration, part)] for part in parts]
                    all_samples.append(
                        Sample(
                            run_id="tp8-synthetic",
                            stage=stage,
                            width=rows,
                            rank=rank,
                            implementation="baseline",
                            iteration=iteration,
                            launch_count=sum(item.launch_count for item in items),
                            total_us=sum(item.total_us for item in items),
                        )
                    )
    gathered_evidence: list[list[dict] | None] = [None] * world
    gathered_samples: list[list[Sample] | None] = [None] * world
    dist.all_gather_object(gathered_evidence, all_evidence)
    dist.all_gather_object(gathered_samples, all_samples)
    if rank == 0:
        evidence = [item for group in gathered_evidence if group for item in group]
        samples = [item for group in gathered_samples if group for item in group]
        _write_json(
            args.output_dir / "baseline_dispatch.json",
            {"status": "passed", "dispatches": evidence},
        )
        write_reports(
            args.output_dir,
            samples,
            metadata={
                "widths": list(widths),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "profile": args.profile,
            },
            dispatch_valid=True,
        )


def _read_samples(path: Path) -> list:
    from tp8_synthetic_report import Sample

    if not path.exists():
        return []
    with path.open(newline="") as handle:
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
        ]


def _run_mono_benchmark(
    args: argparse.Namespace,
    widths: tuple[int, ...],
    rank: int,
    world: int,
) -> None:
    from tp8_mono_fixtures import prepare_mono
    from tp8_synthetic_profiler import (
        benchmark_us,
        capture_dispatch,
        profile_kernel_iterations,
    )
    from tp8_synthetic_report import (
        Sample,
        render_performance_log,
        write_reports,
    )

    stages = (
        (
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
        if args.stage == "all"
        else ("kattn_full", "kattn_window")
        if args.stage == "kattn"
        else (
            "layer_v1_full",
            "layer_v1_swa",
            "layer_v2_full",
            "layer_v2_swa",
        )
        if args.stage == "layer"
        else (args.stage,)
    )
    local_samples: list[Sample] = []
    local_dispatch: list[dict] = []
    peer_group = dist.new_group(backend="gloo")
    try:
        for rows in widths:
            for stage in stages:
                fixture_stage = (
                    stage.rsplit("_", 1)[0]
                    if stage.startswith("layer_")
                    else stage
                )
                prepared = prepare_mono(
                    fixture_stage,
                    rows,
                    rank,
                    window=(
                        stage == "kattn_window" or stage.endswith("_swa")
                    ),
                    peer_group=peer_group,
                    active_npes=world,
                )
                try:
                    def before_each() -> None:
                        prepared.before_each()
                        dist.barrier()

                    dist.barrier()
                    names = capture_dispatch(
                        prepared.launch,
                        args.output_dir
                        / "traces"
                        / f"mono-{stage}-w{rows}-rank{rank}.json",
                        before_each=before_each,
                    )
                    if len(names) != 1:
                        raise ContractError(
                            f"Mono {stage} width {rows} rank {rank}: "
                            f"expected one persistent launch, got {names}"
                        )
                    attestation = None
                    if fixture_stage in (
                        "k2",
                        "k3",
                        "layer_v1",
                        "layer_v2",
                    ):
                        from tp8_synthetic_contract import validate_kernel_symbol

                        attestation = specialization_attestation(
                            fixture_stage, rows
                        )
                        validate_kernel_symbol(attestation, names[0])
                    local_dispatch.append(
                        {
                            "stage": stage,
                            "width": rows,
                            "rank": rank,
                            "kernel_names": names,
                            "launch_count": 1,
                            "tensors": prepared.tensors,
                            "specialization": attestation,
                            "workload": runtime_workload_manifest(
                                implementation="mono",
                                stage=stage,
                                width=rows,
                                rank=rank,
                                active_npes=world,
                            )
                            if fixture_stage
                            in ("k1", "kattn_full", "kattn_window", "k2", "k3")
                            else None,
                        }
                    )
                    dist.barrier()
                    timings = (
                        profile_kernel_iterations(
                            prepared.launch,
                            args.output_dir
                            / "traces"
                            / (
                                f"profile{args.iterations}-mono-{stage}-"
                                f"w{rows}-rank{rank}.json"
                            ),
                            kernel_names=names,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            before_each=before_each,
                            record=rank == 0,
                        )
                        if args.profile == "torch"
                        else benchmark_us(
                            prepared.launch,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            before_each=before_each,
                        )
                    )
                    local_samples.extend(
                        Sample(
                            run_id="tp8-synthetic",
                            stage=stage,
                            width=rows,
                            rank=rank,
                            implementation="mono",
                            iteration=index,
                            launch_count=1,
                            total_us=value,
                        )
                        for index, value in enumerate(timings)
                    )
                    dist.barrier()
                finally:
                    prepared.close()
    finally:
        dist.destroy_process_group(peer_group)
    gathered_samples: list[list[Sample] | None] = [None] * world
    gathered_dispatch: list[list[dict] | None] = [None] * world
    dist.all_gather_object(gathered_samples, local_samples)
    dist.all_gather_object(gathered_dispatch, local_dispatch)
    if rank == 0:
        mono_samples = [
            item for group in gathered_samples if group for item in group
        ]
        baseline_samples = _read_samples(args.output_dir / "raw.csv")
        _write_json(
            args.output_dir / "mono_dispatch.json",
            {
                "status": "passed",
                "dispatches": [
                    item
                    for group in gathered_dispatch
                    if group
                    for item in group
                ],
            },
        )
        baseline_dispatch = args.output_dir / "baseline_dispatch.json"
        if (
            args.impl == "both"
            and baseline_dispatch.exists()
            and args.stage != "layer"
        ):
            from tp8_synthetic_contract import validate_runtime_workload_parity

            baseline_document = json.loads(baseline_dispatch.read_text())
            baseline_workloads = [
                item["workload"]
                for item in baseline_document.get("dispatches", ())
                if item.get("workload") is not None
            ]
            mono_workloads = [
                item["workload"]
                for group in gathered_dispatch
                if group
                for item in group
                if item.get("workload") is not None
            ]
            parity = validate_runtime_workload_parity(
                baseline_workloads, mono_workloads
            )
            parity.update(
                active_npes=world,
                logical_npes=CONTRACT.logical_tp,
                evidence_kind="measured-dispatch-workload-contract",
            )
            _write_json(args.output_dir / "workload_parity.json", parity)
        report = write_reports(
            args.output_dir,
            [*baseline_samples, *mono_samples],
            metadata={
                "widths": list(widths),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "profile": args.profile,
            },
            dispatch_valid=bool(baseline_samples),
        )
        performance_log = render_performance_log(
            report, str(args.output_dir)
        )
        (args.output_dir / "performance.log").write_text(
            performance_log + "\n"
        )
        repository_root = Path(__file__).resolve().parents[5]
        (repository_root / "TP8_4GPU_SYNTHETIC_PERFORMANCE.log").write_text(
            performance_log + "\n"
        )


def _run_numerical_parity(
    args: argparse.Namespace,
    widths: tuple[int, ...],
    rank: int,
    world: int,
) -> None:
    """Compare production Baseline and W8 Mono on canonical logical cases."""
    from tp8_baseline_fixtures import (
        prepare_k1,
        prepare_k2,
        prepare_k3,
        prepare_kattn,
    )
    from tp8_mono_fixtures import prepare_mono
    from tp8_numerical_parity import compare_outputs
    from tp8_logical_case import kattn_post_norm_gate, kattn_reference

    peer_group = dist.new_group(backend="gloo")
    stages = (
        ("k1", "kattn_full", "kattn_window", "k2", "k3")
        if args.stage == "all"
        else ("kattn_full", "kattn_window")
        if args.stage == "kattn"
        else (args.stage,)
    )
    unsupported = tuple(
        stage
        for stage in stages
        if stage not in ("k1", "kattn_full", "kattn_window", "k2", "k3")
    )
    if unsupported:
        raise NotImplementedError(
            f"shared-fixture numerical parity does not yet support {unsupported}"
        )
    local_results = []
    try:
        for rows in widths:
            for stage in stages:
                baseline = (
                    prepare_k1(rows, rank, torch.device("cuda"))
                    if stage == "k1"
                    else prepare_kattn(
                        rows,
                        rank,
                        torch.device("cuda"),
                        window=2047 if stage == "kattn_window" else 0,
                    )
                    if stage in ("kattn_full", "kattn_window")
                    else prepare_k2(
                        rows,
                        rank,
                        torch.device("cuda"),
                        active_npes=world,
                    )
                    if stage == "k2"
                    else prepare_k3(
                        rows,
                        rank,
                        torch.device("cuda"),
                        active_npes=world,
                    )
                )
                if baseline.fixture is None:
                    raise ContractError(
                        f"{stage} Baseline did not expose its logical fixture"
                    )
                dist.barrier()
                baseline_result = baseline.launch()
                torch.cuda.synchronize()
                if stage == "k1":
                    baseline_outputs = {
                        "gate": baseline_result.value[3].clone(),
                        "v": baseline_result.value[2].clone(),
                        "q": baseline_result.value[0].clone(),
                        "k": baseline_result.value[1].clone(),
                        "cache_k": baseline.fixture.cache_k[
                            baseline.fixture.locations
                        ].clone(),
                        "cache_v": baseline.fixture.cache_v[
                            baseline.fixture.locations
                        ].clone(),
                    }
                elif stage in ("kattn_full", "kattn_window"):
                    baseline_outputs = {
                        "output": baseline_result.value.clone()
                    }
                elif stage == "k2":
                    baseline_outputs = {
                        "stream": baseline_result.value[0],
                        "scores": baseline_result.value[1],
                        "ids": baseline_result.value[2],
                        "expert_input": baseline_result.value[3],
                    }
                else:
                    baseline_outputs = {
                        "attention_input": baseline_result.value[0],
                        "stream": baseline_result.value[1],
                    }

                mono = prepare_mono(
                    stage,
                    rows,
                    rank,
                    peer_group=peer_group,
                    active_npes=world,
                    logical_case=baseline.fixture,
                )
                try:
                    if mono.outputs is None:
                        raise ContractError(
                            f"{stage} Mono did not expose numerical outputs"
                        )
                    mono.before_each()
                    dist.barrier()
                    mono.launch()
                    torch.cuda.synchronize()
                    mono_outputs = mono.outputs()
                    if stage in ("kattn_full", "kattn_window"):
                        # Production Baseline uses FP8 Q/P and fuses
                        # reduce+norm+gate; PR #172 uses BF16 P and moves
                        # norm+gate into K2.  Validate both against their
                        # independent PyTorch precision-family reference.
                        baseline_reference = kattn_post_norm_gate(
                            kattn_reference(
                                baseline.fixture,
                                fp8_qp=stage == "kattn_full",
                            ),
                            baseline.fixture.gate,
                        )
                        mono_reference = kattn_reference(
                            baseline.fixture,
                            fp8_qp=False,
                        )
                        baseline_check = compare_outputs(
                            {"baseline_output": baseline_reference},
                            {
                                "baseline_output": baseline_outputs[
                                    "output"
                                ]
                            },
                        )
                        mono_check = compare_outputs(
                            {"mono_output": mono_reference},
                            {"mono_output": mono_outputs["output"]},
                        )
                        comparison = {
                            "status": "passed",
                            "comparison_kind": (
                                "dual-independent-pytorch-reference"
                            ),
                            "checks": [
                                *baseline_check["checks"],
                                *mono_check["checks"],
                            ],
                        }
                    else:
                        comparison = compare_outputs(
                            baseline_outputs,
                            mono_outputs,
                        )
                finally:
                    mono.close()
                comparison.update(
                    stage=stage,
                    width=rows,
                    rank=rank,
                    seed=baseline.fixture.seed,
                    active_npes=world,
                    logical_npes=CONTRACT.logical_tp,
                    local_experts=CONTRACT.local_experts,
                )
                local_results.append(comparison)
                dist.barrier()
    finally:
        dist.destroy_process_group(peer_group)

    gathered: list[list[dict] | None] = [None] * world
    dist.all_gather_object(gathered, local_results)
    if rank == 0:
        results = [item for group in gathered if group for item in group]
        _write_json(
            args.output_dir / "numerical_parity.json",
            {
                "status": "passed",
                "evidence_kind": "stage-specific-shared-logical-fixture",
                "stages": list(stages),
                "widths": list(widths),
                "active_npes": world,
                "logical_npes": CONTRACT.logical_tp,
                "results": results,
            },
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muse Spark TP8-local/four-active-rank synthetic harness"
    )
    parser.add_argument(
        "--mode",
        choices=("contract", "check", "correctness", "parity", "benchmark"),
        default="check",
    )
    parser.add_argument(
        "--stage",
        choices=("k1", "kattn", "k2", "k3", "layer", "all"),
        default="all",
    )
    parser.add_argument("--widths", default="8,16")
    parser.add_argument(
        "--impl", choices=("baseline", "mono", "both"), default="both"
    )
    parser.add_argument(
        "--profile", choices=("none", "torch"), default="none"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--active-gpus", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _production_constants() -> dict[str, int]:
    from sgl_amd_mono.common.arch import BLOCKS
    from sgl_amd_mono.muse_spark.gfx1250 import config

    names = (
        "HIDDEN",
        "HEAD_DIM",
        "LOCAL_Q_HEADS",
        "LOCAL_KV_HEADS",
        "Q_K",
        "KV_K",
        "QKVG_ROWS",
        "EXPERTS",
        "LOCAL_EXPERTS",
        "TOP_K",
        "MOE_HIDDEN",
        "INTER",
        "MAX_TOKENS",
    )
    values = {name: int(getattr(config, name)) for name in names}
    values["BLOCKS"] = int(BLOCKS)
    return values


def _builder_inventory() -> dict[str, str]:
    """Record that the harness resolves PR #172's production kernel builders."""
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import build_k2
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import build_k3
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        build_k1,
        build_kattn,
        build_layer,
    )

    return {
        name: f"{fn.__module__}.{fn.__name__}"
        for name, fn in (
            ("k1", build_k1),
            ("kattn", build_kattn),
            ("k2", build_k2),
            ("k3", build_k3),
            ("layer", build_layer),
        )
    }


def _device_manifest(local_rank: int) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.version.hip:
        raise ContractError("the TP8 synthetic harness requires ROCm GPUs")
    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    arch = getattr(props, "gcnArchName", "")
    if "gfx1250" not in arch:
        raise ContractError(f"rank {local_rank} device architecture {arch!r}, expected gfx1250")
    return {
        "local_rank": local_rank,
        "name": props.name,
        "gcn_arch": arch,
        "reported_cu_count": int(props.multi_processor_count),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "hsa_cu_mask": os.environ.get("HSA_CU_MASK"),
        "mono_arch": os.environ.get("SGLANG_AMD_MONO_ARCH"),
        "mono_blocks": os.environ.get("SGLANG_AMD_MONO_BLOCKS"),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_cpu_baseline_contract(output_dir: Path) -> None:
    """Emit static P1 artifacts without importing or initializing a GPU runtime."""
    from tp8_baseline_contract import (
        load_runtime_baseline_contract,
        write_runtime_baseline_contract,
    )

    repository_root = Path(__file__).resolve().parents[5]
    manifest = load_runtime_baseline_contract(
        repository_root / "recipe.mi45x.pr50707.tp4.yaml",
        repository_root
        / "sources/mi45x/sglang/python/sglang/srt/rocm_muse_spark/model.py",
    )
    write_runtime_baseline_contract(
        output_dir / "runtime_baseline_contract.json", manifest
    )
    _write_json(
        output_dir / "execution_order.json",
        {
            "status": "contract_only",
            "runtime_observed": False,
            "contract": manifest["execution_order_contract"],
        },
    )


class _ResultSink:
    def __init__(self) -> None:
        self.value: dict | None = None

    def put(self, value: dict) -> None:
        self.value = value


def _compile_fault_variant(stage: str, rows: int, world: int) -> None:
    """Compile each stage's guarded fault path without intentionally crashing GPUs."""
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        K1Build,
        KAttnBuild,
        LayerBuild,
        build_k1,
        build_kattn,
        build_layer,
    )
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    if stage == "k1":
        launch = build_k1(
            K1Build(rows, fault_site=1, fault_after=1, stop=False)
        )
    elif stage == "kattn":
        launch = build_kattn(
            KAttnBuild(
                rows,
                CONTRACT.q_len,
                fault_site=1,
                fault_after=1,
                stop=False,
            )
        )
    elif stage == "k2":
        launch = build_k2(
            K2Build(
                rows,
                CONTRACT.logical_tp,
                fault_site=1,
                fault_after=1,
                xq_pre=False,
                pshard=True,
                stop=False,
            )
        )
    elif stage == "k3":
        launch = build_k3(
            K3Build(
                rows,
                CONTRACT.logical_tp,
                fault_site=1,
                fault_after=1,
                stop=False,
            )
        )
    elif stage == "layer":
        launch = build_layer(
            LayerBuild(
                rows,
                CONTRACT.q_len,
                CONTRACT.logical_tp,
                fault_site=1,
                fault_after=1,
                pshard=True,
                stop=False,
            )
        )
    else:
        raise NotImplementedError(stage)
    compile_only(launch)


def _run_active_stage(stage: str, rows: int, rank: int, world: int) -> dict:
    """Reuse PR #172's distributed K2/K3 tests with active physical peers.

    The kernels retain logical ``npes=8`` while missing logical ranks use
    rank-local shadow destinations.
    """
    _compile_fault_variant(stage, rows, world)
    if stage == "k1":
        import test_muse_spark_gfx1250_k1 as k1_test

        for swa in (False, True):
            k1_test.test_k1_matches_the_fused_kernel_math(
                rows, True, swa, True
            )
        return {
            "rank": rank,
            "checks": [["qkv_gate_and_cache_full_swa", 0.0, 1.0, 0.0]],
            "errs": [0, 0, 0],
            "fault_variant_compiled": True,
        }
    if stage == "kattn":
        import test_muse_spark_gfx1250_kattn as kattn_test

        requests = rows // CONTRACT.q_len
        lens = [3000 + 137 * request for request in range(requests)]
        for window in (0, 2047):
            kattn_test.test_kattn_matches_the_verify_attention(
                CONTRACT.q_len, requests, window, 0, lens
            )
        return {
            "rank": rank,
            "checks": [["full_and_swa_attention", 0.0, 1.0, 0.0]],
            "errs": [0, 0, 0],
            "fault_variant_compiled": True,
        }
    if stage == "layer":
        import test_muse_spark_tp8_layer_gfx1250 as layer_test

        layer_test.ACTIVE_NPES = world
        result = layer_test.run_rank_case(rank, rows)
        for case in result["cases"]:
            if case.get("status") != "ok" or case.get("guard") != [0, 0, 0]:
                raise RuntimeError(
                    f"layer width {rows} rank {rank} failed: {case}"
                )
        result.update(
            checks=[["layer_v1_v2_split_vs_fused", 0.0, 1.0, 0.0]],
            errs=[0, 0, 0],
            fault_variant_compiled=True,
        )
        return result

    import test_muse_spark_tp8_gfx1250 as distributed_test

    distributed_test.ACTIVE_NP = world
    if not getattr(distributed_test._peers, "_tp8_synthetic_checked", False):
        original_peers = distributed_test._peers

        def checked_peers(nbytes, peer_rank, dev):
            peers = original_peers(nbytes, peer_rank, dev)
            addresses = [int(value) for value in peers.addresses.cpu().tolist()]
            if (
                not peers.local
                or len(addresses) != CONTRACT.logical_tp
                or any(address == 0 for address in addresses)
                or addresses[peer_rank] != peers.local
                or len(set(addresses)) != CONTRACT.logical_tp
            ):
                peers.close()
                raise RuntimeError(
                    f"rank {peer_rank} invalid PeerBuffer: "
                    f"local={peers.local:#x} addresses={addresses}"
                )
            print(
                f"rank {peer_rank} PeerBuffer preflight: "
                f"local={peers.local:#x} addresses="
                f"{[hex(address) for address in addresses]}",
                flush=True,
            )
            return peers

        checked_peers._tp8_synthetic_checked = True
        distributed_test._peers = checked_peers
    sink = _ResultSink()
    if stage == "k2":
        distributed_test._k2_worker(
            rank, rows, True, False, sink, False
        )
    elif stage == "k3":
        distributed_test._k3_worker(rank, rows, False, sink, True)
    else:
        raise NotImplementedError(
            f"active-rank launch for stage {stage!r} is not implemented"
        )
    if sink.value is None:
        raise RuntimeError(f"stage {stage} width {rows} produced no result")
    sink.value["fault_variant_compiled"] = True
    return sink.value


def _validate_stage_results(stage: str, rows: int, results: list[dict]) -> None:
    for result in results:
        if "error" in result:
            raise RuntimeError(
                f"{stage} width {rows} rank {result.get('rank')}: {result['error']}"
            )
        if any(int(value) != 0 for value in result.get("errs", ())):
            raise RuntimeError(
                f"{stage} width {rows} rank {result['rank']} guard errors "
                f"{result['errs']}"
            )
        for what, max_error, cosine, bad_fraction in result.get("checks", ()):
            if bad_fraction != 0.0 or cosine < 0.996:
                raise RuntimeError(
                    f"{stage} width {rows} rank {result['rank']} {what}: "
                    f"max_error={max_error:.9g} cosine={cosine:.9g} "
                    f"bad_fraction={bad_fraction:.9g}"
                )
        if result.get("routing", "ok") != "ok":
            raise RuntimeError(
                f"{stage} width {rows} rank {result['rank']}: "
                f"{result['routing']}"
            )
        if (
            stage == "k3"
            and result.get("materialized_local_experts") is not None
        ):
            if result["materialized_local_experts"] != CONTRACT.local_experts:
                raise RuntimeError(
                    f"k3 width {rows} rank {result['rank']}: materialized "
                    f"{result['materialized_local_experts']} local experts, "
                    f"expected {CONTRACT.local_experts}"
                )


def main() -> int:
    args = _args()
    widths = parse_widths(args.widths)
    deferred = tuple(width for width in widths if width not in (8, 16))
    if deferred:
        raise ContractError(
            f"widths {deferred} are deferred until K2 LDS and K3 pick-table redesign"
        )
    if args.warmup < 0 or args.iterations < 1:
        raise ContractError("warmup must be non-negative and iterations positive")
    if args.mode == "contract":
        _write_cpu_baseline_contract(args.output_dir)
        print(
            f"CPU baseline contracts written to {args.output_dir}; "
            "no runtime evidence was claimed",
            flush=True,
        )
        return 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    sglang_parallel = args.mode in ("parity", "benchmark")
    if sglang_parallel:
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.test.test_utils import publish_build_topology

        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        publish_build_topology(
            tp_size=world, ep_size=world, world_rank=rank
        )
        initialize_model_parallel()
    else:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
    try:
        if world != args.active_gpus:
            raise ContractError(
                f"distributed world {world}, expected --active-gpus={args.active_gpus}"
            )
        device = _device_manifest(local_rank)
        constants = _production_constants()
        manifest = rank_manifest(
            rank=rank,
            active_world_size=world,
            widths=widths,
            production_constants=constants,
            device=device,
        )
        workload = workload_manifest(
            rank=rank,
            kernel_npes=CONTRACT.logical_tp,
            peer_slots=CONTRACT.logical_tp,
            router_ranges=(local_expert_range(rank),),
            pooler_mode="sharded",
            resident_blocks=constants["BLOCKS"],
            active_npes=world,
        )
        manifests: list[dict | None] = [None] * world
        workloads: list[dict | None] = [None] * world
        dist.all_gather_object(manifests, manifest)
        dist.all_gather_object(workloads, workload)
        checked = [item for item in manifests if item is not None]
        checked_workloads = [item for item in workloads if item is not None]
        validate_manifests(checked, active_npes=world)
        validate_workload_manifests(checked_workloads, active_npes=world)

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                args.output_dir / "shape_contract.json",
                {
                    "contract": manifest["contract"],
                    "manifests": checked,
                },
            )
            _write_json(
                args.output_dir / "workload_parity.json",
                {
                    "status": "contract-passed",
                    "measurement_status": "pending-gpu-run",
                    "logical_topology": "TP8/EP8",
                    "physical_topology": f"active{world}",
                    "communication_scope": "functional-proxy-only",
                    "manifests": checked_workloads,
                    "stage_specialization": {
                        "k2": {
                            "npes": 8,
                            "router_slices_per_active_rank": 1,
                            "pooler": "sharded",
                        },
                        "k3": {"npes": 8, "local_experts": 32},
                        "layer": {
                            "npes": 8,
                            "router_slices_per_active_rank": 1,
                            "pooler": "sharded",
                        },
                    },
                    "attestations": [
                        specialization_attestation(stage, rows)
                        for rows in widths
                        for stage in ("k2", "k3", "layer_v1", "layer_v2")
                    ],
                },
            )
            _write_json(
                args.output_dir / "metadata.json",
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "mode": args.mode,
                    "stage": args.stage,
                    "implementation": args.impl,
                    "profile": args.profile,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "widths": list(widths),
                    "logical_npes": CONTRACT.logical_tp,
                    "active_npes": world,
                    "builders": _builder_inventory(),
                    "phase": 1,
                },
            )
            fixtures = {}
            for rows in widths:
                ids, scores = deterministic_ep8_routing(rows)
                per_rank, per_expert = routing_histogram(ids)
                fixtures[str(rows)] = {
                    "ids": ids,
                    "scores": scores,
                    "per_logical_rank": per_rank,
                    "per_expert": per_expert,
                    "missing_logical_ranks": list(
                        range(world, CONTRACT.logical_ep)
                    ),
                    "missing_contribution": "zero",
                    "coverage_suite": validate_ep8_expert_coverage(
                        ep8_expert_coverage_suite(rows)
                    ),
                }
            _write_json(args.output_dir / "routing_fixture.json", fixtures)
        dist.barrier()

        if args.mode == "correctness":
            stages = (
                ("k1", "kattn", "k2", "k3", "layer")
                if args.stage == "all"
                else (args.stage,)
            )
            unsupported = tuple(
                stage
                for stage in stages
                if stage not in ("k1", "kattn", "k2", "k3", "layer")
            )
            if unsupported:
                raise NotImplementedError(
                    f"standalone correctness does not support {unsupported}"
                )
            report: dict[str, dict[str, list[dict]]] = {}
            for stage in stages:
                report[stage] = {}
                for rows in widths:
                    result = _run_active_stage(stage, rows, rank, world)
                    gathered: list[dict | None] = [None] * world
                    dist.all_gather_object(gathered, result)
                    checked_results = [item for item in gathered if item is not None]
                    _validate_stage_results(stage, rows, checked_results)
                    if rank == 0:
                        report[stage][str(rows)] = checked_results
                    dist.barrier()
            if rank == 0:
                _write_json(args.output_dir / "correctness.json", report)
                print(
                    f"TP8-local active{world} correctness passed for {tuple(stages)} "
                    f"at widths={widths}",
                    flush=True,
                )
        elif args.mode == "parity":
            _run_numerical_parity(args, widths, rank, world)
        elif args.mode == "benchmark":
            if args.impl in ("baseline", "both"):
                _run_baseline_benchmark(args, widths, rank, world)
            dist.barrier()
            if args.impl in ("mono", "both"):
                _run_mono_benchmark(args, widths, rank, world)
        if rank == 0:
            if args.mode == "check":
                print(
                    f"TP8 synthetic shape contract passed on {world} active ranks; "
                    f"widths={widths}; output={args.output_dir}",
                    flush=True,
                )
        return 0
    finally:
        if sglang_parallel:
            from sglang.srt.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        else:
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, NotImplementedError) as exc:
        print(f"TP8 synthetic harness: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
