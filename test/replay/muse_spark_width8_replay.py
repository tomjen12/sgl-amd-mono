#!/usr/bin/env python3
"""Width-8 Muse Spark replay validation and baseline preflight entry point.

The lightweight capture stores geometry and routing workload, not runtime
tensor values. Replay inputs and cache contents are regenerated
deterministically from the captured seed and descriptors.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
from pathlib import Path
from typing import Any

from muse_spark_synthetic_target import (
    TARGET_LAYERS,
    build_config_directory,
    deterministic_seed,
    replay_scope_policy,
)
from replay_bundle import ReplayBundle
from trace_fingerprint import compare_fingerprints, fingerprint_trace

DEFAULT_DRAFT_CHECKPOINT = Path("/data/models/Muse-Spark-Open-DFlash2")
DEFAULT_CU_MASK = "0-3:0-95"
DEFAULT_PROFILE = "cu96_tp4"
REQUIRED_DRAFT_TAPS = (1, 16, 30, 44, 59)


class ReplayExecutionBlocked(RuntimeError):
    """Raised when preflight proves the capture cannot be replayed faithfully."""


@dataclasses.dataclass(frozen=True)
class ScopePlan:
    name: str
    checkpoint: str
    load_format: str
    layers: tuple[int, ...]
    warmup: int
    iterations: int
    trace_files: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SessionPlan:
    mode: str
    bundle: str
    step: int
    width: int
    tp_size: int
    ep_size: int
    cu_mask: str
    hardware_profile: str
    environment: dict[str, str]
    synthetic_seed: int
    rank_weight_seeds: tuple[int, ...]
    scopes: tuple[ScopePlan, ...]
    full_cycle: bool
    blocker: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, mode="rt", encoding="utf-8") as handle:
            return json.load(handle)
    with source.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_config(directory: Path, label: str) -> dict[str, Any]:
    path = directory / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{label} config does not exist: {path}")
    return _load_json(path)


def _validate_run_config(
    bundle: ReplayBundle,
    draft_checkpoint: Path,
    target_checkpoint: Path,
) -> int:
    draft = _read_config(draft_checkpoint, "Draft checkpoint")
    target = _read_config(target_checkpoint, "Synthetic Target")
    policy_path = target_checkpoint / "replay_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Synthetic Target replay policy does not exist: {policy_path}"
        )
    policy = _load_json(policy_path)

    if draft.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Draft checkpoint is not the DFlash2 model.")
    if int(draft.get("num_hidden_layers", -1)) != 5:
        raise ValueError("DFlash2 replay requires exactly five Draft layers.")
    taps = tuple(
        int(value)
        for value in draft.get("dflash_config", {}).get("target_layer_ids", ())
    )
    if taps != REQUIRED_DRAFT_TAPS:
        raise ValueError(
            f"DFlash2 taps must be {list(REQUIRED_DRAFT_TAPS)}, got {list(taps)}."
        )
    if int(draft.get("dflash_config", {}).get("block_size", -1)) != 8:
        raise ValueError("DFlash2 checkpoint block_size must be 8.")

    text = target.get("text_config")
    if not isinstance(text, dict) or int(text.get("num_hidden_layers", -1)) != 10:
        raise ValueError("Synthetic Target must contain config-only layers 0-9.")
    if list(policy.get("target_layers", ())) != list(range(TARGET_LAYERS)):
        raise ValueError("Synthetic Target replay policy does not select layers 0-9.")
    expected_policy = replay_scope_policy(draft, target_layers=TARGET_LAYERS)
    if policy != expected_policy:
        raise ValueError("Synthetic Target replay policy does not match DFlash2.")
    shards = sorted(target_checkpoint.glob("*.safetensors"))
    if shards:
        raise ValueError(
            f"Synthetic Target must be config-only; found shards: {shards}"
        )

    cycles = [rank.metadata("cycle") for rank in bundle.ranks]
    seed = int(cycles[0].get("synthetic_seed", -1))
    if seed < 0 or any(int(cycle.get("synthetic_seed", -1)) != seed for cycle in cycles):
        raise ValueError("Capture ranks do not share a valid synthetic_seed.")
    if any(int(cycle.get("batch_size", -1)) != 1 for cycle in cycles):
        raise ValueError("Replay requires the formal concurrency-1 client cycle.")

    draft_input = bundle.ranks[0].metadata("draft_forward_input")
    target_input = bundle.ranks[0].metadata("target_verify")
    if draft_input.get("forward_mode") != "TARGET_VERIFY":
        raise ValueError("Captured Draft scope is not a TARGET_VERIFY forward.")
    if target_input.get("forward_mode") != "TARGET_VERIFY":
        raise ValueError("Captured Target scope is not a TARGET_VERIFY forward.")
    draft_shape = draft_input.get("tensors", {}).get("input_embeds", {}).get("shape")
    target_shape = target_input.get("tensors", {}).get("input_ids", {}).get("shape")
    if draft_shape != [8, 8192] or target_shape != [8]:
        raise ValueError(
            "Capture is not the expected width-8 Muse Spark Draft/Target workload: "
            f"input_embeds={draft_shape}, target input_ids={target_shape}."
        )
    if int(target_input.get("seq_lens_sum", 0)) < 8192:
        raise ValueError("Captured Target scope is not an 8K client cycle.")
    for rank in bundle.ranks:
        draft_layers = rank.metadata("draft_layers")
        target_layers = rank.metadata("target_layers")
        for layer in range(5):
            if f"layer_{layer}.input.hidden" not in draft_layers:
                raise ValueError(
                    f"rank {rank.rank}: Draft layer {layer} descriptor is missing."
                )
        for layer in range(10):
            required = (
                f"layer_{layer}.input.stream",
                f"layer_{layer}.routing",
            )
            missing = [name for name in required if name not in target_layers]
            if missing:
                raise ValueError(
                    f"rank {rank.rank}: Target layer {layer} metadata is missing "
                    f"{missing}."
                )
    return seed


def build_session_plan(
    *,
    bundle_path: str | Path,
    target_checkpoint: str | Path,
    draft_checkpoint: str | Path = DEFAULT_DRAFT_CHECKPOINT,
    trace_dir: str | Path,
    step: int = 0,
    warmup: int = 2,
    iterations: int = 5,
    mode: str = "baseline",
) -> SessionPlan:
    if mode != "baseline":
        raise ValueError(
            "Mono replay is not implemented. Run baseline first and validate its "
            "trace fingerprint before enabling mono."
        )
    if warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if iterations <= 0:
        raise ValueError("--iterations must be positive.")

    bundle = ReplayBundle.load(bundle_path, step=step, expected_ranks=4)
    if len(bundle.ranks) != 4:
        raise ValueError("Muse Spark replay requires exactly four ranks.")
    seed = _validate_run_config(
        bundle, Path(draft_checkpoint), Path(target_checkpoint)
    )
    trace_dir = Path(trace_dir)

    def traces(scope: str) -> tuple[str, ...]:
        return tuple(
            str(trace_dir / f"{scope}.rank_{rank:04d}.trace.json.gz")
            for rank in range(4)
        )

    blocker = (
        "The deterministic tensor/cache rehydration and SGLang ModelRunner "
        "execution adapter are not implemented yet. Captured runtime tensor "
        "values are intentionally not required."
    )
    return SessionPlan(
        mode=mode,
        bundle=str(Path(bundle_path)),
        step=step,
        width=8,
        tp_size=4,
        ep_size=4,
        cu_mask=DEFAULT_CU_MASK,
        hardware_profile=DEFAULT_PROFILE,
        environment={
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "HSA_CU_MASK": DEFAULT_CU_MASK,
            "SGLANG_MUSE_MONO_GFX1250_CONFIG": DEFAULT_PROFILE,
            "SGLANG_ENABLE_AMD_MONOKERNEL": "0",
        },
        synthetic_seed=seed,
        rank_weight_seeds=tuple(
            deterministic_seed(seed, rank, "target") for rank in range(4)
        ),
        scopes=(
            ScopePlan(
                name="draft",
                checkpoint=str(Path(draft_checkpoint)),
                load_format="checkpoint",
                layers=tuple(range(5)),
                warmup=warmup,
                iterations=iterations,
                trace_files=traces("draft"),
            ),
            ScopePlan(
                name="target",
                checkpoint=str(Path(target_checkpoint)),
                load_format="dummy",
                layers=tuple(range(10)),
                warmup=warmup,
                iterations=iterations,
                trace_files=traces("target"),
            ),
        ),
        full_cycle=False,
        blocker=blocker,
    )


def execute_session(plan: SessionPlan) -> None:
    """Refuse a substituted workload until deterministic rehydration is wired."""
    if os.environ.get("SGLANG_ENABLE_AMD_MONOKERNEL") not in (None, "", "0"):
        raise ReplayExecutionBlocked(
            "Mono is disabled for baseline replay; unset "
            "SGLANG_ENABLE_AMD_MONOKERNEL."
        )
    raise ReplayExecutionBlocked(
        f"{plan.blocker} No GPU work was launched. Implement the two planned "
        "scopes independently through benchmark.one_batch bootstrap/ModelRunner."
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("bundle")
    inspect.add_argument("--step", type=int, default=0)

    build_config = subparsers.add_parser("build-config")
    build_config.add_argument("target_checkpoint")
    build_config.add_argument("draft_checkpoint")
    build_config.add_argument("output")
    build_config.add_argument("--layers", type=int, default=10)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("trace")
    fingerprint.add_argument("output")
    fingerprint.add_argument(
        "--scope-pattern", default=r"^muse_replay/(draft|target|spec_cycle)"
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("expected")
    compare.add_argument("actual")
    compare.add_argument("--gpu-time-tolerance", type=float, default=None)

    run = subparsers.add_parser("run")
    run.add_argument("bundle")
    run.add_argument("target_checkpoint")
    run.add_argument("trace_dir")
    run.add_argument(
        "--draft-checkpoint", default=str(DEFAULT_DRAFT_CHECKPOINT)
    )
    run.add_argument("--step", type=int, default=0)
    run.add_argument("--warmup", type=int, default=2)
    run.add_argument("--iterations", type=int, default=5)
    run.add_argument("--mode", choices=("baseline", "mono"), default="baseline")
    run.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the independent per-rank session plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    if args.command == "inspect":
        bundle = ReplayBundle.load(args.bundle, step=args.step)
        print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "build-config":
        policy = build_config_directory(
            args.target_checkpoint,
            args.draft_checkpoint,
            args.output,
            layers=args.layers,
        )
        print(json.dumps(policy, indent=2, sort_keys=True))
        return 0
    if args.command == "fingerprint":
        value = fingerprint_trace(
            _load_json(args.trace), scope_pattern=args.scope_pattern
        )
        _write_json(args.output, value)
        return 0
    if args.command == "run":
        try:
            plan = build_session_plan(
                bundle_path=args.bundle,
                target_checkpoint=args.target_checkpoint,
                draft_checkpoint=args.draft_checkpoint,
                trace_dir=args.trace_dir,
                step=args.step,
                warmup=args.warmup,
                iterations=args.iterations,
                mode=args.mode,
            )
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            if not args.plan_only:
                execute_session(plan)
            return 0
        except (ValueError, FileNotFoundError, ReplayExecutionBlocked) as error:
            parser = create_parser()
            parser.error(str(error))

    failures = compare_fingerprints(
        _load_json(args.expected),
        _load_json(args.actual),
        gpu_time_tolerance=args.gpu_time_tolerance,
    )
    if failures:
        print("\n".join(failures))
        return 1
    print("Replay trace fingerprints match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
