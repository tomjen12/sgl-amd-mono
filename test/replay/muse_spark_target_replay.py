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
import inspect
import json
import logging
import multiprocessing
import os
import pickle
import random
import signal
import subprocess
import sys
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
KV_GUARD_ROWS = 16
DEFAULT_SYNTHETIC_DRAFT_CONFIG = (
    Path(__file__).resolve().parents[5] / "synthetic-target-0-9" / "dflash"
)
BASELINE_ENVIRONMENT = {
    "HIP_VISIBLE_DEVICES": "0,1,2,3",
    "HSA_CU_MASK": "0-3:0-95",
    "SGLANG_ENABLE_AMD_MONOKERNEL": "0",
    "SGLANG_MUSE_MONO_GFX1250_CONFIG": "cu96_tp4",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_GROUPED_DEBUG": "0",
    "SGLANG_SIMULATE_ACC_LEN": "4.05",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "MUSE_MOE_BACKEND": "triton",
    "NCCL_CUMEM_ENABLE": "1",
    "AITER_CUSTOM_AR_FORCE_VMM": "1",
    "AITER_CUSTOM_AR_MAX_SIZE": "1048576",
    "SGLANG_USE_AITER_AG": "0",
    "SGLANG_ROCM_MUSE_FINALIZE_AR_PUSH": "0",
    "SGLANG_ROCM_MUSE_A4W4_MOE": "0",
    "SGLANG_ROCM_MUSE_COMPACT_A4_VERIFY": "0",
    "SGLANG_ROCM_MUSE_GFX1250_TRY_GFX95": "1",
    "SGLANG_ROCM_MUSE_EXTEND_GLUON": "0",
    "SGLANG_ROCM_MUSE_EXTEND_SWA_GLUON": "0",
    "SGLANG_ROCM_MUSE_EXTEND_FLYDSL": "0",
    "SGLANG_ROCM_MUSE_VERIFY_FLYDSL": "0",
    "SGLANG_ROCM_MUSE_VERIFY_BALANCED": "0",
    "SGLANG_ROCM_MUSE_VERIFY_SWA_FLYDSL": "0",
    "HIP_FORCE_DEV_KERNARG": "1",
    "SGLANG_ROCM_MUSE_GFX1250_EXTEND": "1",
    "SGLANG_ROCM_MUSE_PUSH_AR": "0",
    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "1",
    "SGLANG_OPT_SCHEDULER_SHM_SYNC": "1",
}
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
    full_kv_physical_rows: int
    swa_kv_physical_rows: int
    captured_routing: tuple[tuple[int, ...], ...]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_output(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


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

    cache = verify.get("cache", {})
    cache_rows: dict[str, int] = {}
    for cache_name in ("kv.full", "kv.swa"):
        cache_entry = cache.get(cache_name, {})
        layers_entry = cache_entry.get("layers", {})
        layer_entry = next(iter(layers_entry.values()), {})
        shape = layer_entry.get("k", {}).get("shape", ())
        if not shape or int(shape[0]) <= KV_GUARD_ROWS:
            raise ReplayInputError(
                f"rank {rank}: capture has no valid {cache_name} physical extent"
            )
        cache_rows[cache_name] = int(shape[0])

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
        full_kv_physical_rows=cache_rows["kv.full"],
        swa_kv_physical_rows=cache_rows["kv.swa"],
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
    pool_extents = {
        (item.full_kv_physical_rows, item.swa_kv_physical_rows)
        for item in descriptors
    }
    if len(seeds) != 1 or min(seeds) < 0:
        raise ReplayInputError("all four ranks must share a valid synthetic seed")
    if len(prefixes) != 1:
        raise ReplayInputError("all four ranks must share the captured prefix length")
    if len(pool_extents) != 1:
        raise ReplayInputError("all four ranks must share captured KV pool extents")
    if any(item.vocab_size <= 0 for item in descriptors):
        raise ReplayInputError("Target config has no valid vocabulary")
    return descriptors


def routing_ids_from_counts(counts: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Build a deterministic [8, 8] routing table with unique ids per row."""
    counts = tuple(int(value) for value in counts)
    if len(counts) != 256 or sum(counts) != WIDTH * WIDTH:
        raise ReplayInputError("routing counts must contain 64 assignments")
    if any(value < 0 or value > WIDTH for value in counts):
        raise ReplayInputError("an expert may appear at most once per routing row")

    rows: list[list[int]] = [[] for _ in range(WIDTH)]
    for expert, count in sorted(
        enumerate(counts), key=lambda item: (-item[1], item[0])
    ):
        candidates = sorted(range(WIDTH), key=lambda row: (len(rows[row]), row))
        for row in candidates[:count]:
            rows[row].append(expert)
    if any(len(row) != WIDTH or len(set(row)) != WIDTH for row in rows):
        raise ReplayInputError("captured routing counts cannot form an [8, 8] table")
    return tuple(tuple(sorted(row)) for row in rows)


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


def _output_hidden(output):
    logits = getattr(output, "logits_output", None)
    for owner in (output, logits):
        value = getattr(owner, "hidden_states", None)
        if hasattr(value, "detach"):
            return value
    raise RuntimeError("Target replay output has no hidden states tensor.")


def _kv_state_summary(runner, layer_ids: Iterable[int], out_cache_loc) -> dict[str, Any]:
    import torch

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


def _cpu_snapshot(tensor):
    return tensor.detach().to(device="cpu").clone()


class _Restore:
    def __init__(self, owner, name: str, original):
        self.owner = owner
        self.name = name
        self.original = original

    def remove(self) -> None:
        setattr(self.owner, self.name, self.original)


def _target_model(model):
    return model.model.language_model


def _installed_mono(model):
    try:
        built = inspect.getclosurevars(model.forward).nonlocals["built"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Unable to inspect installed Target Mono runner.") from error
    if not hasattr(built, "runner") or built.runner is None:
        raise RuntimeError("Installed Target Mono forward has no runner.")
    return built


def _install_baseline_state_observers(model, states: dict[int, dict[str, Any]]):
    import torch

    handles = []
    for layer_id, layer in enumerate(_target_model(model).layers):
        original = layer.forward

        def observe(*args, layer_id=layer_id, original=original, **kwargs):
            output = original(*args, **kwargs)
            # Snapshot collection is correctness-only.  The layer outputs may
            # be backed by buffers reused by the next asynchronous layer, so
            # finish this layer before cloning them.
            torch.cuda.current_stream().synchronize()
            states[layer_id] = {
                "hidden": output[0].detach().clone(),
                "residual": output[1].detach().clone(),
            }
            return output

        layer.forward = observe
        handles.append(_Restore(layer, "forward", original))
    return handles


def _install_mono_state_observer(model, states: dict[int, dict[str, Any]]):
    import torch

    built = _installed_mono(model)
    runner = built.runner
    original = runner.launch

    def observe(*args, **kwargs):
        layer_id = int(kwargs["i"])
        output = original(*args, **kwargs)
        # runner.ids and the state tensors are reused across layers.  Without
        # this correctness-only synchronization, an early layer's asynchronous
        # clone can observe a later layer (layer 0 previously captured layer 9).
        torch.cuda.current_stream().synchronize()
        states[layer_id] = {
            "hidden": output[0].detach().clone(),
            "residual": output[1].detach().clone(),
            "routing_ids": runner.ids[: kwargs["rows"]].detach().clone(),
        }
        return output

    runner.launch = observe
    return [_Restore(runner, "launch", original)]


def _snapshot_kv(runner, out_cache_loc) -> dict[int, dict[str, Any]]:
    import torch

    snapshots = {}
    for layer_id in range(TARGET_LAYERS):
        key = runner.token_to_kv_pool.get_key_buffer(layer_id)
        value = runner.token_to_kv_pool.get_value_buffer(layer_id)
        ids = out_cache_loc.to(device=key.device, dtype=torch.int64)
        snapshots[layer_id] = {
            "key": _cpu_snapshot(key[ids]),
            "value": _cpu_snapshot(value[ids]),
        }
    return snapshots


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
        max_running_requests=2,
        max_total_tokens=16384,
        context_length=16384,
        attention_backend="aiter",
        page_size=16,
        kv_cache_dtype="fp8_e4m3",
        disable_radix_cache=True,
        enable_amd_monokernel=args.mode == "mono",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=str(Path(args.draft_checkpoint)),
        speculative_draft_attention_backend="triton",
        speculative_num_draft_tokens=WIDTH,
        speculative_draft_window_size=2048,
        cuda_graph_backend_decode="full" if args.cuda_graph else "disabled",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=1 if args.cuda_graph else None,
    )


def _load_runner(
    server_args,
    port_args,
    gpu_id: int,
    rank: int,
    weight_seed: int,
    descriptor: RankDescriptor,
):
    import torch
    import torch.distributed as dist
    from sglang.benchmark.one_batch import _init_process_global_configs
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
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
    runner.alloc_memory_pool(
        memory_pool_config=MemoryPoolConfig(
            max_total_num_tokens=(
                descriptor.full_kv_physical_rows - KV_GUARD_ROWS
            ),
            full_max_total_num_tokens=(
                descriptor.full_kv_physical_rows - KV_GUARD_ROWS
            ),
            swa_max_total_num_tokens=(
                descriptor.swa_kv_physical_rows - KV_GUARD_ROWS
            ),
            max_running_requests=128,
        )
    )
    runner.init_attention_backends()
    if cfg.tp_size > 1:
        dist.barrier()
    return runner


def _prepare_verify_batch(
    runner, descriptor: RankDescriptor, *, cuda_graph: bool = False
):
    import torch
    from sglang.benchmark.one_batch import (
        extend,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.allocation import (
        alloc_paged_token_slots_decode,
        assign_req_to_token_pool_func,
    )
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
    last_loc = batch.req_to_token_pool.req_to_token[
        batch.req_pool_indices, batch.seq_lens - 1
    ]
    out_cache_loc = alloc_paged_token_slots_decode(
        tree_cache=batch.tree_cache,
        seq_lens=batch.seq_lens + WIDTH,
        seq_lens_cpu=batch.seq_lens_cpu + WIDTH,
        last_loc=last_loc,
        token_per_req=WIDTH,
        req_pool_indices=batch.req_pool_indices,
        batch=batch,
    )
    assign_req_to_token_pool_func(
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        prefix_lens,
        prefix_lens + WIDTH,
        out_cache_loc,
        batch_size=1,
    )
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
    batch.can_run_decode_cuda_graph = cuda_graph
    # Preserve the captured host-side planner bound exactly; GPU seq_lens stays
    # at the committed 8192-token prefix as in DFlashWorkerV2.
    batch.seq_lens_cpu = torch.tensor([descriptor.seq_lens_sum], dtype=torch.int64)
    batch.seq_lens_sum = descriptor.seq_lens_sum
    forward_batch, can_graph = spec.prepare_for_verify(
        batch, SimpleNamespace(model_runner=runner)
    )
    if bool(can_graph) != bool(runner.decode_cuda_graph_captured):
        raise RuntimeError(
            "Target replay CUDA Graph eligibility disagrees with the configured "
            f"runner: eligible={bool(can_graph)}, "
            f"captured={bool(runner.decode_cuda_graph_captured)}"
        )
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
        os.environ.update(BASELINE_ENVIRONMENT)
        os.environ["AITER_CONFIG_GROUPED_FMOE"] = str(
            Path(__file__).resolve().parents[3]
            / "sglang/docker/muse_gfx1250/"
            "muse_gfx1250_tp8_ep8_tuned_grouped_fmoe.csv"
        )
        os.environ["SGLANG_MUSE_REPLAY_TARGET_LAYERS"] = str(TARGET_LAYERS)
        os.environ["SGLANG_ENABLE_AMD_MONOKERNEL"] = (
            "1" if args.mode == "mono" else "0"
        )
        os.environ["SGLANG_AMD_MONO_MUSE_KERNELS"] = "1"
        os.environ["SGLANG_AMD_MONO_MUSE_LAYER"] = (
            "1" if args.target_kernel_mode == "layer" else "0"
        )
        os.environ["SGLANG_AMD_MONO_CHECK"] = "1" if args.check else "0"
        os.environ["MUSE_PHASE_TRACE"] = "1" if args.phase_dir else "0"
        if args.phase_dir:
            os.environ["MUSE_PHASE_TRACE_OUTPUT"] = str(Path(args.phase_dir).resolve())
        if args.attn_dump_dir is not None:
            os.environ["SGLANG_AMD_MONO_ATTN_DUMP"] = args.attn_dump_dir
        if args.check:
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("sgl_amd_mono.muse_spark.spec").setLevel(logging.INFO)
        os.environ["SGLANG_MUSE_REPLAY_MARKERS"] = (
            "1" if args.trace_dir is not None else "0"
        )
        if args.mode == "mono":
            import sgl_amd_mono
            from sglang.srt.kernel_plugins.amd import binding as mono_binding

            sgl_amd_mono.register_specs()
            # This tool replays one captured Target Verify scope independently;
            # enabling the full DFLASH server would incorrectly require taps
            # outside synthetic Target layers 0-9.
            mono_binding._server_step = lambda: ("target_verify", WIDTH)
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.rocm_muse_spark.replay_capture import (
            replay_scope,
            routing_override,
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
        runner = _load_runner(
            server_args,
            port_args,
            gpu_id,
            rank,
            weight_seed,
            descriptor,
        )
        mono_attached = hasattr(runner.model, "_monokernel_unread_metadata")
        if args.mode == "mono" and not mono_attached:
            from sglang.srt.kernel_plugins.amd.binding import _selection

            forward = getattr(runner.model.forward, "__func__", runner.model.forward)
            raise RuntimeError(
                "Target Mono mode requested but plugin did not attach: "
                f"selection={_selection()!r}, "
                f"model={type(runner.model).__module__}.{type(runner.model).__name__}, "
                f"forward={getattr(forward, '__module__', None)}."
                f"{getattr(forward, '__qualname__', type(forward).__name__)}, "
                f"attrs={sorted(runner.model.__dict__)}"
            )
        route_ids = {
            layer: torch.tensor(
                routing_ids_from_counts(counts),
                dtype=torch.int32,
                device=runner.device,
            )
            for layer, counts in enumerate(descriptor.captured_routing)
        }
        route_scores = None
        if args.routing_snapshot_dir is not None:
            routing_snapshot = torch.load(
                Path(args.routing_snapshot_dir)
                / f"rank_{rank:04d}.snapshot.pt",
                map_location="cpu",
                weights_only=False,
            )
            route_ids = {
                layer: routing_snapshot["routing_ids"][layer]
                .to(device=runner.device, dtype=torch.int32)
                .contiguous()
                for layer in range(TARGET_LAYERS)
            }
            route_scores = {
                layer: routing_snapshot["routing_scores"][layer]
                .to(device=runner.device, dtype=torch.float32)
                .contiguous()
                for layer in range(TARGET_LAYERS)
            }

        # Capture under the same deterministic routing override used by replay.
        # The graph runner owns static input buffers, while the override fixes
        # the per-layer expert workload captured by the Mono wrappers.
        with routing_override(route_ids, route_scores):
            runner.init_cuda_graphs()
        forward_batch, out_cache_loc, positions = _prepare_verify_batch(
            runner, descriptor, cuda_graph=args.cuda_graph
        )

        def run_once():
            with replay_scope("target/verify"):
                return runner.forward(forward_batch)

        with routing_override(route_ids, route_scores):
            for _ in range(args.warmup):
                run_once()
            torch.cuda.synchronize()

            writer = _CaptureWriter()
            if args.cuda_graph:
                from sglang.srt.rocm_muse_spark.replay_capture import (
                    force_eager_model_runner,
                )

                # Python metadata hooks are bypassed by graph replay. Collect
                # one unprofiled eager pass while preserving the captured graph.
                runner.attn_backend.init_forward_metadata(forward_batch)
                with force_eager_model_runner(runner):
                    with runtime_metadata_capture(writer, "target_layers"):
                        run_once()
            else:
                with runtime_metadata_capture(writer, "target_layers"):
                    run_once()
            torch.cuda.synchronize()

            profiler = _trace_context(args.trace_dir is not None)
            checksums = []
            elapsed = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = run_once()
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
                checksums.append(_output_checksum(output))

            snapshot_path = None
            if args.snapshot_dir is not None:
                layer_states: dict[int, dict[str, Any]] = {}
                handles = (
                    _install_mono_state_observer(runner.model, layer_states)
                    if args.mode == "mono"
                    else _install_baseline_state_observers(
                        runner.model, layer_states
                    )
                )
                output = run_once()
                torch.cuda.synchronize()
                for handle in handles:
                    handle.remove()
                if set(layer_states) != set(range(TARGET_LAYERS)):
                    raise RuntimeError(
                        f"Target {args.mode} layer snapshots are incomplete: "
                        f"{sorted(layer_states)}"
                    )
                snapshot = {
                    "rank": rank,
                    "mode": args.mode,
                    "output": _cpu_snapshot(_output_hidden(output)),
                    "layers": {
                        layer: {
                            name: _cpu_snapshot(value)
                            for name, value in layer_states[layer].items()
                        }
                        for layer in range(TARGET_LAYERS)
                    },
                    "routing_ids": {
                        layer: (
                            _cpu_snapshot(route_ids[layer])
                            if args.mode == "mono"
                            else torch.tensor(
                                writer.values["target_layers"][
                                    f"layer_{layer}.routing"
                                ]["topk_ids"],
                                dtype=torch.int32,
                            )
                        )
                        for layer in range(TARGET_LAYERS)
                    },
                    "routing_scores": {
                        layer: torch.tensor(
                            writer.values["target_layers"][
                                f"layer_{layer}.routing"
                            ]["topk_scores"],
                            dtype=torch.float32,
                        )
                        for layer in range(TARGET_LAYERS)
                    },
                    "kv_cache": _snapshot_kv(runner, out_cache_loc),
                }
                snapshot_path = (
                    Path(args.snapshot_dir) / f"rank_{rank:04d}.snapshot.pt"
                )
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, snapshot_path)
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
            "mode": args.mode,
            "mono_attached": mono_attached,
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
            "snapshot": str(snapshot_path) if snapshot_path is not None else None,
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


def _profile_sibling_workers(args, server_args, port_args, directory: str):
    """Run four peer ranks while rocprof-compute wraps only one worker.

    The profiler process is a sibling of the three unprofiled workers, not
    their parent. This preserves the symmetric TP4 process-group lifecycle and
    prevents rocprofiler-sdk from following every distributed child.
    """
    root = Path(directory)
    state_path = root / "worker-state.pkl"
    with state_path.open("wb") as stream:
        pickle.dump((args, server_args, port_args), stream)

    paths = [str(root / f"rank_{rank}.json") for rank in range(WORLD_SIZE)]
    clean_env = os.environ.copy()
    for name in tuple(clean_env):
        if (
            name == "LD_PRELOAD"
            or name.startswith("ROCPROF")
            or name.startswith("ROCP_TOOL")
        ):
            clean_env.pop(name)

    processes: dict[int, subprocess.Popen] = {}
    for rank in range(WORLD_SIZE):
        worker_env = clean_env.copy()
        worker = [
            sys.executable,
            str(Path(__file__).resolve()),
            args.bundle,
            args.target_config,
            "--worker-state",
            str(state_path),
            "--worker-rank",
            str(rank),
            "--worker-result",
            paths[rank],
        ]
        if rank == args.profile_worker_rank:
            worker_env["LD_PRELOAD"] = args.profile_preload
            command = [
                args.profile_rocprof,
                "profile",
                "--output-directory",
                args.profile_output,
                "--set",
                args.profile_set,
                "--no-roof",
                "--kernel",
                args.profile_kernel,
                "--overwrite",
                "--",
                *worker,
            ]
        else:
            command = worker
        processes[rank] = subprocess.Popen(
            command,
            env=worker_env,
            start_new_session=rank == args.profile_worker_rank,
        )

    deadline = time.monotonic() + args.profile_timeout
    profile_ready_since = None
    forced_profile_completion = False
    while any(process.poll() is None for process in processes.values()):
        profile_process = processes[args.profile_worker_rank]
        counters_ready = any(
            Path(args.profile_output).rglob("*_native_counter_collection.csv.gz")
        )
        results_ready = all(Path(path).is_file() for path in paths)
        if profile_process.poll() is None and counters_ready and results_ready:
            if profile_ready_since is None:
                profile_ready_since = time.monotonic()
            elif (
                time.monotonic() - profile_ready_since
                >= args.profile_finalize_grace
            ):
                os.killpg(profile_process.pid, signal.SIGTERM)
                try:
                    profile_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(profile_process.pid, signal.SIGKILL)
                    profile_process.wait()
                forced_profile_completion = True
        failures = [
            (rank, process.returncode)
            for rank, process in processes.items()
            if process.returncode not in (None, 0)
            and not (rank == args.profile_worker_rank and forced_profile_completion)
        ]
        if failures or time.monotonic() >= deadline:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            if failures:
                raise RuntimeError(f"profile worker failure: {failures}")
            raise TimeoutError(
                f"four-rank profiling exceeded {args.profile_timeout} seconds"
            )
        time.sleep(0.2)

    results = []
    for rank, path in enumerate(paths):
        if Path(path).is_file():
            results.append(_read_json(path))
        else:
            results.append(
                {
                    "rank": rank,
                    "status": "error",
                    "error": (
                        f"profile worker exited {processes[rank].returncode} "
                        "without a result"
                    ),
                }
            )
    return results


def execute(args) -> tuple[int, dict[str, Any]]:
    os.environ.update(BASELINE_ENVIRONMENT)
    os.environ["SGLANG_MUSE_REPLAY_TARGET_LAYERS"] = str(TARGET_LAYERS)
    os.environ["SGLANG_ENABLE_AMD_MONOKERNEL"] = (
        "1" if args.mode == "mono" else "0"
    )
    os.environ["SGLANG_AMD_MONO_MUSE_LAYER"] = (
        "1" if args.target_kernel_mode == "layer" else "0"
    )
    if args.mode == "mono":
        import sgl_amd_mono

        sgl_amd_mono.register_specs()

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
        if args.profile_worker_rank is not None:
            results = _profile_sibling_workers(
                args, server_args, port_args, directory
            )
        else:
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
                            "error": (
                                f"worker exited {process.exitcode} without a result"
                            ),
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
        "mode": args.mode,
        "execution": "cuda-graph" if args.cuda_graph else "eager",
        "target_kernel_mode": args.target_kernel_mode,
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
    parser.add_argument(
        "--draft-checkpoint",
        default=str(DEFAULT_SYNTHETIC_DRAFT_CONFIG),
        help="DFlash2 config used to resolve width-8 Target verify graph shape.",
    )
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trace-dir")
    parser.add_argument(
        "--phase-dir",
        help="Write versioned K1/K-attn/K2/K3 phase artifacts for Mono split replay.",
    )
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--routing-snapshot-dir")
    parser.add_argument("--output")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--mode", choices=("baseline", "mono"), default="baseline")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Profile full decode CUDA Graph replay. Graph capture and warmup "
            "complete before the trace starts."
        ),
    )
    parser.add_argument(
        "--target-kernel-mode",
        choices=("layer", "split"),
        default="layer",
        help="Run fused Target Layer kernels or standalone K1/K-attn/K2/K3.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--attn-dump-dir")
    parser.add_argument("--profile-worker-rank", type=int, choices=range(WORLD_SIZE))
    parser.add_argument("--profile-rocprof")
    parser.add_argument("--profile-preload")
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-kernel")
    parser.add_argument(
        "--profile-set",
        choices=("launch_stats", "mem_thruput", "compute_thruput_flops"),
    )
    parser.add_argument("--profile-timeout", type=float, default=300.0)
    parser.add_argument("--profile-finalize-grace", type=float, default=15.0)
    parser.add_argument("--worker-state", help=argparse.SUPPRESS)
    parser.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
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
    if args.worker_state is not None:
        if args.worker_rank is None or args.worker_result is None:
            raise SystemExit(
                "--worker-state requires --worker-rank and --worker-result"
            )
        with Path(args.worker_state).open("rb") as stream:
            worker_args, server_args, port_args = pickle.load(stream)
        _worker(
            worker_args,
            server_args,
            port_args,
            args.worker_rank,
            args.worker_rank,
            args.worker_result,
        )
        # PyTorch may lazily start multiprocessing.resource_tracker even though
        # this worker does not create child ranks. Stop it before process exit;
        # otherwise rocprofiler-sdk finalization waits for that helper while
        # Python's atexit cleanup waits for profiler finalization.
        from multiprocessing import resource_tracker

        tracker = resource_tracker._resource_tracker
        if tracker._pid is not None:
            tracker._stop()
        result = _read_json(args.worker_result)
        return 0 if result.get("status") == "ok" else 1
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be non-negative and --iterations positive")
    profile_values = (
        args.profile_rocprof,
        args.profile_preload,
        args.profile_output,
        args.profile_kernel,
        args.profile_set,
    )
    if args.profile_worker_rank is not None and any(
        value is None for value in profile_values
    ):
        raise SystemExit(
            "--profile-worker-rank requires --profile-rocprof, "
            "--profile-preload, --profile-output, --profile-kernel, and --profile-set"
        )
    if args.profile_worker_rank is None and any(
        value is not None for value in profile_values
    ):
        raise SystemExit(
            "--profile-rocprof/--profile-preload/--profile-output/"
            "--profile-kernel/--profile-set require --profile-worker-rank"
        )
    if args.profile_timeout <= 0:
        raise SystemExit("--profile-timeout must be positive")
    if args.profile_finalize_grace <= 0:
        raise SystemExit("--profile-finalize-grace must be positive")
    if args.mode == "mono" and args.routing_snapshot_dir is None:
        raise SystemExit(
            "Target Mono replay requires --routing-snapshot-dir from baseline."
        )
    if args.mode != "mono" and args.target_kernel_mode != "layer":
        raise SystemExit("--target-kernel-mode split requires --mode mono.")
    if args.phase_dir and (
        args.mode != "mono" or args.target_kernel_mode != "split"
    ):
        raise SystemExit("--phase-dir requires --mode mono --target-kernel-mode split.")
    if args.cuda_graph and (
        args.snapshot_dir is not None
        or args.check
        or args.attn_dump_dir is not None
    ):
        raise SystemExit(
            "--cuda-graph is profiling-only and cannot be combined with "
            "--snapshot-dir, --check, or --attn-dump-dir."
        )
    if args.cuda_graph and not (Path(args.draft_checkpoint) / "config.json").is_file():
        raise SystemExit(
            "--cuda-graph requires a valid DFlash2 --draft-checkpoint config."
        )
    try:
        code, result = execute(args)
    except (ReplayInputError, FileNotFoundError, ValueError) as error:
        create_parser().error(str(error))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_output(args.output, text)
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
