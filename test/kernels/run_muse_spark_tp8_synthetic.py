# SPDX-License-Identifier: Apache-2.0
"""Four-rank entry point for TP8-local Muse Spark kernel development.

Phase 1 implements the independent shape contract and distributed manifest.
Correctness and benchmark dispatch are intentionally added in later phases.
"""

from __future__ import annotations

import argparse
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
    parse_widths,
    rank_manifest,
    routing_histogram,
    validate_manifests,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muse Spark TP8-local/four-active-rank synthetic harness"
    )
    parser.add_argument(
        "--mode",
        choices=("check", "correctness", "benchmark"),
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
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
        build_k1,
        build_kattn,
    )
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    if stage == "k1":
        launch = build_k1(K1Build(rows, fault_site=1, fault_after=1))
    elif stage == "kattn":
        launch = build_kattn(
            KAttnBuild(rows, CONTRACT.q_len, fault_site=1, fault_after=1)
        )
    elif stage == "k2":
        launch = build_k2(
            K2Build(rows, world, fault_site=1, fault_after=1, xq_pre=False)
        )
    elif stage == "k3":
        launch = build_k3(K3Build(rows, world, fault_site=1, fault_after=1))
    else:
        raise NotImplementedError(stage)
    compile_only(launch)


def _run_active4_stage(stage: str, rows: int, rank: int, world: int) -> dict:
    """Reuse PR #172's distributed K2/K3 tests with four active peers.

    This Phase-2 path proves the production builders can launch with TP8 local
    constants and active ``npes=4``.  Its references currently cover the active
    half only; the logical-ranks-4..7 zero semantics are added in Phase 3.
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

    import test_muse_spark_tp8_gfx1250 as distributed_test

    distributed_test.NP = world
    if not getattr(distributed_test._peers, "_tp8_synthetic_checked", False):
        original_peers = distributed_test._peers

        def checked_peers(nbytes, peer_rank, dev):
            peers = original_peers(nbytes, peer_rank, dev)
            addresses = [int(value) for value in peers.addresses.cpu().tolist()]
            if (
                not peers.local
                or len(addresses) != world
                or any(address == 0 for address in addresses)
                or addresses[peer_rank] != peers.local
                or len(set(addresses)) != world
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
            rank, rows, False, False, sink, False
        )
    elif stage == "k3":
        distributed_test._k3_worker(rank, rows, False, sink, True)
    else:
        raise NotImplementedError(
            f"four-rank launch for stage {stage!r} is not implemented in Phase 2"
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
        for what, _max_error, cosine, bad_fraction in result.get("checks", ()):
            if bad_fraction != 0.0 or cosine < 0.996:
                raise RuntimeError(
                    f"{stage} width {rows} rank {result['rank']} {what}: "
                    f"cosine={cosine:.6f} bad={bad_fraction:.2%}"
                )
        if result.get("routing", "ok") != "ok":
            raise RuntimeError(
                f"{stage} width {rows} rank {result['rank']}: "
                f"{result['routing']}"
            )


def main() -> int:
    args = _args()
    widths = parse_widths(args.widths)
    if args.warmup < 0 or args.iterations < 1:
        raise ContractError("warmup must be non-negative and iterations positive")

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    try:
        device = _device_manifest(local_rank)
        constants = _production_constants()
        manifest = rank_manifest(
            rank=rank,
            active_world_size=world,
            widths=widths,
            production_constants=constants,
            device=device,
        )
        manifests: list[dict | None] = [None] * world
        dist.all_gather_object(manifests, manifest)
        checked = [item for item in manifests if item is not None]
        validate_manifests(checked)

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
                    "missing_logical_ranks": [4, 5, 6, 7],
                    "missing_contribution": "zero",
                }
            _write_json(args.output_dir / "routing_fixture.json", fixtures)
        dist.barrier()

        if args.mode == "correctness":
            stages = ("k1", "kattn", "k2", "k3") if args.stage == "all" else (args.stage,)
            unsupported = tuple(
                stage for stage in stages if stage not in ("k1", "kattn", "k2", "k3")
            )
            if unsupported:
                raise NotImplementedError(
                    f"standalone correctness does not support {unsupported}"
                )
            report: dict[str, dict[str, list[dict]]] = {}
            for stage in stages:
                report[stage] = {}
                for rows in widths:
                    result = _run_active4_stage(stage, rows, rank, world)
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
                    f"TP8-local active4 correctness passed for {tuple(stages)} "
                    f"at widths={widths}",
                    flush=True,
                )
        elif args.mode == "benchmark":
            raise NotImplementedError("benchmark mode is implemented in Phase 6")
        if rank == 0:
            if args.mode == "check":
                print(
                    f"TP8 synthetic shape contract passed on {world} active ranks; "
                    f"widths={widths}; output={args.output_dir}",
                    flush=True,
                )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, NotImplementedError) as exc:
        print(f"TP8 synthetic harness: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
