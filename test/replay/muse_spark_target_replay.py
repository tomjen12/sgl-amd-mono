#!/usr/bin/env python3
"""Executable synthetic Muse Spark Target layers 0-9 width-8 replay.

This deliberately runs only the Target scope.  DFlash2 requires Target taps
outside layers 0-9, so pretending that the config-only model can execute a
complete speculative cycle would be a different workload.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import multiprocessing
import os
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

WIDTH = 8
WORLD_SIZE = 4
TARGET_LAYERS = 10
DEFAULT_SEED = 300959315
ROUTING_BLOCKER = (
    "Seeded synthetic router weights did not reproduce the captured per-expert "
    "token counts. The row/top-k workload is preserved, but exact grouped-GEMM "
    "shapes require a focused Target router-output override hook."
)


class ReplayInputError(ValueError):
    """The supplied capture is not the requested width-8 Target workload."""


@dataclasses.dataclass(frozen=True)
class RankDescriptor:
    rank: int
    seed: int
    prefix_len: int
    seq_lens_sum: int
    vocab_size: int
    captured_routing: tuple[tuple[int, ...], ...]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def deterministic_seed(base_seed: int, rank: int, name: str) -> int:
    payload = f"{base_seed}:{rank}:{name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def deterministic_values(
    *, seed: int, rank: int, name: str, count: int, modulus: int
) -> list[int]:
    """Generate stable values without depending on torch's RNG implementation."""
    if count < 0 or modulus <= 0:
        raise ValueError("count must be non-negative and modulus must be positive")
    rng = random.Random(deterministic_seed(seed, rank, name))
    return [rng.randrange(modulus) for _ in range(count)]


def _rank_directory(bundle: Path, step: int, rank: int) -> Path:
    return bundle / f"step_{step:06d}" / f"rank_{rank:04d}"


def load_rank_descriptor(
    bundle: str | Path, target_config: str | Path, *, step: int, rank: int
) -> RankDescriptor:
    bundle = Path(bundle)
    rank_dir = _rank_directory(bundle, step, rank)
    cycle = _read_json(rank_dir / "cycle.json")
    verify = _read_json(rank_dir / "target_verify.json")
    layers = _read_json(rank_dir / "target_layers.json")
    config = _read_json(Path(target_config) / "config.json")
    text = config.get("text_config", {})

    if int(cycle.get("width", -1)) != WIDTH or int(cycle.get("batch_size", -1)) != 1:
        raise ReplayInputError(f"rank {rank}: capture is not width-8/batch-1")
    if int(cycle.get("draft_tp_size", -1)) != WORLD_SIZE:
        raise ReplayInputError(f"rank {rank}: capture is not TP4")
    if verify.get("forward_mode") != "TARGET_VERIFY":
        raise ReplayInputError(f"rank {rank}: forward mode is not TARGET_VERIFY")
    if verify.get("tensors", {}).get("input_ids", {}).get("shape") != [WIDTH]:
        raise ReplayInputError(f"rank {rank}: Target input is not shape [{WIDTH}]")
    if int(text.get("num_hidden_layers", -1)) != TARGET_LAYERS:
        raise ReplayInputError("config-only Target must contain exactly layers 0-9")
    if list(Path(target_config).glob("*.safetensors")):
        raise ReplayInputError("Target directory must be config-only (no checkpoint)")

    lower_bound = int(verify.get("attention_kv_len_lower_bound") or 0)
    if lower_bound < 8192:
        raise ReplayInputError(f"rank {rank}: captured prefix is shorter than 8192")
    routing: list[tuple[int, ...]] = []
    for layer in range(TARGET_LAYERS):
        entry = layers.get(f"layer_{layer}.routing")
        stream = layers.get(f"layer_{layer}.input.stream")
        if (
            not isinstance(entry, dict)
            or entry.get("rows") != WIDTH
            or entry.get("top_k") != WIDTH
            or stream is None
            or stream.get("shape") != [WIDTH, 8192]
        ):
            raise ReplayInputError(
                f"rank {rank}: layer {layer} lacks width-8 routing/stream geometry"
            )
        counts = tuple(int(value) for value in entry.get("global_token_counts", ()))
        if len(counts) != 256 or sum(counts) != WIDTH * WIDTH:
            raise ReplayInputError(f"rank {rank}: layer {layer} routing is malformed")
        routing.append(counts)

    return RankDescriptor(
        rank=rank,
        seed=int(cycle.get("synthetic_seed", -1)),
        prefix_len=lower_bound,
        seq_lens_sum=int(verify.get("seq_lens_sum", lower_bound + WIDTH)),
        vocab_size=int(text.get("vocab_size", 0)),
        captured_routing=tuple(routing),
    )


def validate_inputs(
    bundle: str | Path, target_config: str | Path, *, step: int = 0
) -> tuple[RankDescriptor, ...]:
    descriptors = tuple(
        load_rank_descriptor(bundle, target_config, step=step, rank=rank)
        for rank in range(WORLD_SIZE)
    )
    seeds = {item.seed for item in descriptors}
    prefixes = {item.prefix_len for item in descriptors}
    if len(seeds) != 1 or min(seeds) < 0:
        raise ReplayInputError("all four ranks must share a valid synthetic seed")
    if len(prefixes) != 1:
        raise ReplayInputError("all four ranks must share the captured prefix length")
    if any(item.vocab_size <= 0 for item in descriptors):
        raise ReplayInputError("Target config has no valid vocabulary")
    return descriptors


def compare_routing(
    expected: Iterable[Iterable[int]], observed: dict[str, Any]
) -> dict[str, Any]:
    expected = tuple(tuple(int(v) for v in row) for row in expected)
    mismatched: list[int] = []
    shape_preserved = True
    for layer, counts in enumerate(expected):
        entry = observed.get(f"layer_{layer}.routing")
        if not isinstance(entry, dict):
            shape_preserved = False
            mismatched.append(layer)
            continue
        if entry.get("rows") != WIDTH or entry.get("top_k") != WIDTH:
            shape_preserved = False
        actual = tuple(int(v) for v in entry.get("global_token_counts", ()))
        if actual != counts:
            mismatched.append(layer)
    exact = not mismatched
    return {
        "shape_preserved": shape_preserved,
        "exact_counts": exact,
        "mismatched_layers": mismatched,
        "blocker": None if exact else ROUTING_BLOCKER,
    }


def _tensor_checksum(tensor) -> dict[str, Any]:
    import torch

    value = tensor.detach()
    if value.numel() == 0:
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sum": 0.0}
    finite = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sum": float(finite.sum(dtype=torch.float64).cpu()),
        "abs_sum": float(finite.abs().sum(dtype=torch.float64).cpu()),
    }


def _output_checksum(output) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    logits = getattr(output, "logits_output", None)
    for owner_name, owner in (("output", output), ("logits", logits)):
        if owner is None:
            continue
        for name in ("hidden_states", "next_token_logits", "logits"):
            value = getattr(owner, name, None)
            if hasattr(value, "detach"):
                tensors[f"{owner_name}.{name}"] = _tensor_checksum(value)
    return tensors


def _kv_state_summary(runner, layer_ids: Iterable[int], out_cache_loc) -> dict[str, Any]:
    state: dict[str, Any] = {"write_ids": [int(v) for v in out_cache_loc.cpu().tolist()]}
    for layer in layer_ids:
        try:
            key = runner.token_to_kv_pool.get_key_buffer(layer)
            value = runner.token_to_kv_pool.get_value_buffer(layer)
            ids = out_cache_loc.to(device=key.device, dtype=torch.int64)
            state[f"layer_{layer}.k"] = _tensor_checksum(key[ids])
            state[f"layer_{layer}.v"] = _tensor_checksum(value[ids])
        except (AttributeError, KeyError, TypeError, RuntimeError) as error:
            state[f"layer_{layer}.unavailable"] = str(error)
    return state


class _CaptureWriter:
    def __init__(self):
        self.values: dict[str, dict[str, Any]] = {}

    def write_metadata(self, name: str, value: dict[str, Any]) -> None:
        self.values[name] = value


def _build_server_args(args):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=str(Path(args.target_config)),
        tokenizer_path=str(Path(args.target_config)),
        load_format="dummy",
        trust_remote_code=True,
        tp_size=WORLD_SIZE,
        ep_size=WORLD_SIZE,
        mem_fraction_static=args.mem_fraction_static,
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )


def _load_runner(server_args, port_args, gpu_id: int, rank: int, weight_seed: int):
    import torch
    import torch.distributed as dist
    from sglang.benchmark.one_batch import _init_process_global_configs
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_device

    _init_process_global_configs()
    torch.manual_seed(weight_seed)
    torch.cuda.manual_seed_all(weight_seed)
    cfg = resolving_view(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    bootstrap.init_parallel_runtime(
        server_args=server_args,
        device=get_device().device,
        dist_port=port_args.nccl_port,
    )
    bootstrap.init_layer_runtime(model_config=model_config)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=cfg.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    if cfg.tp_size > 1:
        dist.barrier()
    return runner


def _prepare_verify_batch(runner, descriptor: RankDescriptor):
    import torch
    from sglang.benchmark.one_batch import (
        extend,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.allocation import alloc_for_decode
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    tokens = deterministic_values(
        seed=descriptor.seed,
        rank=0,
        name="target-prefix",
        count=descriptor.prefix_len,
        modulus=descriptor.vocab_size,
    )
    reqs = prepare_synthetic_inputs_for_latency_test(
        1, descriptor.prefix_len, [tokens]
    )
    _, _, batch = extend(reqs, runner)
    if not isinstance(batch, ScheduleBatch):
        raise RuntimeError("one_batch prefill did not return a ScheduleBatch")

    draft_tokens = torch.tensor(
        deterministic_values(
            seed=descriptor.seed,
            rank=0,
            name="target-width8-input",
            count=WIDTH,
            modulus=descriptor.vocab_size,
        ),
        dtype=torch.int64,
        device=runner.device,
    )
    prefix_lens = torch.full(
        (1,), descriptor.prefix_len, dtype=torch.int64, device=runner.device
    )
    positions = torch.arange(
        descriptor.prefix_len,
        descriptor.prefix_len + WIDTH,
        dtype=torch.int64,
        device=runner.device,
    )
    out_cache_loc = alloc_for_decode(batch, token_per_req=WIDTH)
    plan = runner.runner_ids.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=prefix_lens,
        seq_lens_cpu=torch.tensor([descriptor.prefix_len], dtype=torch.int64),
        write_virtual=out_cache_loc,
        read_extent=WIDTH,
    )
    spec = DFlashVerifyInput(
        draft_token=draft_tokens,
        positions=positions,
        draft_token_num=WIDTH,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        iteration_ids=plan,
    )
    batch.out_cache_loc = out_cache_loc
    # Preserve the captured host-side planner bound exactly; GPU seq_lens stays
    # at the committed 8192-token prefix as in DFlashWorkerV2.
    batch.seq_lens_cpu = torch.tensor([descriptor.seq_lens_sum], dtype=torch.int64)
    batch.seq_lens_sum = descriptor.seq_lens_sum
    forward_batch, can_graph = spec.prepare_for_verify(
        batch, SimpleNamespace(model_runner=runner)
    )
    if can_graph:
        raise RuntimeError("Target replay must use the eager baseline path")
    return forward_batch, out_cache_loc, positions


def _trace_context(enabled: bool):
    if not enabled:
        return None
    import torch

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )
    profiler.start()
    return profiler


def _worker(args, server_args, port_args, gpu_id: int, rank: int, result_path: str):
    result: dict[str, Any] = {"rank": rank, "status": "error"}
    try:
        os.environ.update(
            HIP_VISIBLE_DEVICES="0,1,2,3",
            HSA_CU_MASK="0-3:0-95",
            SGLANG_ENABLE_AMD_MONOKERNEL="0",
            SGLANG_MUSE_MONO_GFX1250_CONFIG="cu96_tp4",
        )
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.rocm_muse_spark.replay_capture import (
            runtime_metadata_capture,
        )
        from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank

        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0),
                gpu_id=gpu_id,
            ),
        )
        descriptor = load_rank_descriptor(
            args.bundle, args.target_config, step=args.step, rank=rank
        )
        weight_seed = deterministic_seed(descriptor.seed, rank, "target")
        os.environ["SGLANG_DUMMY_WEIGHT_SEED"] = str(weight_seed)
        runner = _load_runner(server_args, port_args, gpu_id, rank, weight_seed)
        forward_batch, out_cache_loc, positions = _prepare_verify_batch(
            runner, descriptor
        )

        for _ in range(args.warmup):
            runner.forward(forward_batch)
        torch.cuda.synchronize()

        writer = _CaptureWriter()
        profiler = _trace_context(args.trace_dir is not None)
        checksums = []
        elapsed = []
        with runtime_metadata_capture(writer, "target_layers"):
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = runner.forward(forward_batch)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
                checksums.append(_output_checksum(output))
        if profiler is not None:
            profiler.stop()
            trace = Path(args.trace_dir) / f"target.rank_{rank:04d}.trace.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            raw = trace.with_suffix("")
            profiler.export_chrome_trace(str(raw))
            with raw.open("rb") as source, gzip.open(trace, "wb") as destination:
                destination.write(source.read())
            raw.unlink()

        observed = writer.values.get("target_layers", {})
        routing = compare_routing(descriptor.captured_routing, observed)
        result = {
            "rank": rank,
            "status": "ok",
            "tp_size": WORLD_SIZE,
            "ep_size": WORLD_SIZE,
            "weight_seed": weight_seed,
            "input": {
                "prefix_len": descriptor.prefix_len,
                "width": WIDTH,
                "positions": [int(v) for v in positions.cpu().tolist()],
                "seq_lens_sum": descriptor.seq_lens_sum,
            },
            "iterations": args.iterations,
            "latency_seconds": elapsed,
            "output_checksums": checksums,
            "state": _kv_state_summary(
                runner, range(TARGET_LAYERS), out_cache_loc
            ),
            "routing": routing,
        }
        if dist.is_initialized():
            dist.barrier()
            destroy_model_parallel()
            destroy_distributed_environment()
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    Path(result_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def execute(args) -> tuple[int, dict[str, Any]]:
    from sglang.srt.arg_groups.overrides import declare_resolution
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs

    descriptors = validate_inputs(args.bundle, args.target_config, step=args.step)
    server_args = _build_server_args(args)
    declare_resolution(server_args, "muse.target.replay")
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    with tempfile.TemporaryDirectory(prefix="muse-target-replay-") as directory:
        workers = []
        paths = []
        for rank in range(WORLD_SIZE):
            path = str(Path(directory) / f"rank_{rank}.json")
            paths.append(path)
            process = multiprocessing.Process(
                target=_worker,
                args=(args, server_args, port_args, rank, rank, path),
            )
            process.start()
            workers.append(process)
        for process in workers:
            process.join()

        results = []
        for rank, (process, path) in enumerate(zip(workers, paths)):
            if Path(path).is_file():
                results.append(_read_json(path))
            else:
                results.append(
                    {
                        "rank": rank,
                        "status": "error",
                        "error": f"worker exited {process.exitcode} without a result",
                    }
                )

    failures = [item for item in results if item.get("status") != "ok"]
    routing_blockers = [
        item["rank"]
        for item in results
        if item.get("routing", {}).get("blocker") is not None
    ]
    summary = {
        "workload": "synthetic-target-layers-0-9-width8",
        "tp_size": WORLD_SIZE,
        "ep_size": WORLD_SIZE,
        "processes": WORLD_SIZE,
        "synthetic_seed": descriptors[0].seed,
        "ranks": results,
        "routing_override_blocked_ranks": routing_blockers,
    }
    if failures:
        return 1, summary
    if routing_blockers and not args.allow_routing_mismatch:
        return 2, summary
    return 0, summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("target_config")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trace-dir")
    parser.add_argument("--output")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument(
        "--allow-routing-mismatch",
        action="store_true",
        help=(
            "Explicitly accept natural seeded routing when captured per-expert "
            "counts differ. The result still reports the mismatch."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be non-negative and --iterations positive")
    try:
        code, result = execute(args)
    except (ReplayInputError, FileNotFoundError, ValueError) as error:
        create_parser().error(str(error))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
