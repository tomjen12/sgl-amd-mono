#!/usr/bin/env python3
"""Executable TP4 replay of the real five-layer Muse Spark DFlash2 draft.

This is deliberately a ModelRunner replay, not a direct monokernel benchmark.
The lightweight capture bundle contains descriptors and a seed rather than
tensor values, so all values are regenerated deterministically.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import multiprocessing
import os
import socket
import traceback
from pathlib import Path
from typing import Any, Iterable

WORLD_SIZE = 4
WIDTH = 8
LAYERS = 5
HIDDEN_SIZE = 8192
WINDOW = 2048
PAGE_SIZE = 16
KV_HEADS_PER_RANK = 2
HEAD_DIM = 128
DEFAULT_BUNDLE = Path(
    "results/20261117_034841_mono0_profile0/replay_bundle"
)
DEFAULT_CHECKPOINT = Path("/data/models/Muse-Spark-Open-DFlash2")


class ReplayInputError(RuntimeError):
    """The requested replay cannot be constructed faithfully."""


@dataclasses.dataclass(frozen=True)
class ReplayDescriptor:
    bundle: Path
    checkpoint: Path
    step: int
    seed: int
    prefix_position: int
    seq_len: int
    seq_len_bound: int
    cache_rows: int
    input_shape: tuple[int, int]
    position_shape: tuple[int, ...]
    kv_shape: tuple[int, int, int]
    conv_shape: tuple[int, int, int]

    def summary(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["bundle"] = str(self.bundle)
        value["checkpoint"] = str(self.checkpoint)
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ReplayInputError(f"Missing replay artifact: {path}") from error
    if not isinstance(value, dict):
        raise ReplayInputError(f"Replay artifact is not a JSON object: {path}")
    return value


def _artifact(rank_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    matches = [
        item
        for item in manifest.get("artifacts", ())
        if item.get("name") == name and item.get("kind") == "metadata"
    ]
    if len(matches) != 1:
        raise ReplayInputError(
            f"{rank_dir}: expected one metadata artifact {name!r}, found "
            f"{len(matches)}"
        )
    return rank_dir / str(matches[0]["path"])


def _shape(mapping: dict[str, Any], name: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in mapping[name]["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayInputError(f"Missing or invalid tensor descriptor {name!r}") from error


def _checkpoint_shards(checkpoint: Path) -> tuple[Path, ...]:
    direct = tuple(sorted(checkpoint.glob("*.safetensors")))
    if direct:
        return direct
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map", {})
        return tuple(
            checkpoint / name for name in sorted(set(weight_map.values()))
        )
    return ()


def validate_replay_inputs(
    bundle: str | Path,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    step: int = 0,
    require_weights: bool = True,
) -> ReplayDescriptor:
    """Validate four-rank descriptors and return the fixed Draft geometry."""
    bundle = Path(bundle)
    checkpoint = Path(checkpoint)
    config = _read_json(checkpoint / "config.json")
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise ReplayInputError("Checkpoint architecture is not DFlash2DraftModel.")
    if int(config.get("num_hidden_layers", -1)) != LAYERS:
        raise ReplayInputError("Draft checkpoint must contain exactly five layers.")
    if int(config.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise ReplayInputError("Draft checkpoint hidden_size must be 8192.")
    draft_config = config.get("dflash_config", {})
    if int(draft_config.get("block_size", -1)) != WIDTH:
        raise ReplayInputError("Draft checkpoint block_size must be 8.")
    if int(config.get("sliding_window", -1)) != WINDOW:
        raise ReplayInputError("Draft checkpoint sliding_window must be 2048.")
    shards = _checkpoint_shards(checkpoint)
    missing_shards = [path for path in shards if not path.is_file()]
    if require_weights and (not shards or missing_shards):
        detail = (
            f"missing indexed shards: {[str(path) for path in missing_shards]}"
            if missing_shards
            else "no *.safetensors or model.safetensors.index.json was found"
        )
        raise ReplayInputError(
            f"Real DFlash2 checkpoint weights are unavailable in {checkpoint}: {detail}."
        )

    common: dict[str, Any] | None = None
    forward_zero: dict[str, Any] | None = None
    for rank in range(WORLD_SIZE):
        rank_dir = bundle / f"step_{step:06d}" / f"rank_{rank:04d}"
        manifest = _read_json(rank_dir / "manifest.json")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("rank") != rank
            or manifest.get("step") != step
            or manifest.get("complete") is not True
        ):
            raise ReplayInputError(f"Invalid or incomplete rank manifest: {rank_dir}")
        cycle = _read_json(_artifact(rank_dir, manifest, "cycle"))
        forward = _read_json(_artifact(rank_dir, manifest, "draft_forward_input"))
        layers = _read_json(_artifact(rank_dir, manifest, "draft_layers"))
        current = {
            "seed": int(cycle.get("synthetic_seed", -1)),
            "width": int(cycle.get("width", -1)),
            "batch_size": int(cycle.get("batch_size", -1)),
            "tp": int(cycle.get("draft_tp_size", -1)),
            "input_shape": _shape(forward.get("tensors", {}), "input_embeds"),
            "position_shape": _shape(forward.get("tensors", {}), "positions"),
            "seq_lens_shape": _shape(forward.get("tensors", {}), "seq_lens"),
            "seq_lens_sum": int(forward.get("seq_lens_sum", -1)),
        }
        if common is None:
            common = current
            forward_zero = forward
        elif current != common:
            raise ReplayInputError(
                f"rank {rank} Draft geometry differs from rank 0: "
                f"{current!r} != {common!r}"
            )
        for layer in range(LAYERS):
            if _shape(layers, f"layer_{layer}.input.hidden") != (
                WIDTH,
                HIDDEN_SIZE,
            ):
                raise ReplayInputError(
                    f"rank {rank} layer {layer} hidden descriptor is not [8, 8192]."
                )
            if _shape(layers, f"layer_{layer}.output.conv_coefficients") != (
                WIDTH,
                2,
                HIDDEN_SIZE // 16,
            ):
                raise ReplayInputError(
                    f"rank {rank} layer {layer} convolution carry descriptor is invalid."
                )

    assert common is not None and forward_zero is not None
    expected = {
        "width": WIDTH,
        "batch_size": 1,
        "tp": WORLD_SIZE,
        "input_shape": (WIDTH, HIDDEN_SIZE),
        "position_shape": (WIDTH,),
        "seq_lens_shape": (1,),
    }
    for key, value in expected.items():
        if common[key] != value:
            raise ReplayInputError(
                f"Captured Draft {key} must be {value!r}, got {common[key]!r}."
            )
    if forward_zero.get("forward_mode") != "TARGET_VERIFY":
        raise ReplayInputError("Captured Draft forward mode is not TARGET_VERIFY.")
    if forward_zero.get("can_run_decode_cuda_graph") is not False:
        raise ReplayInputError("Validated Draft baseline must be eager.")

    cache = forward_zero.get("cache", {}).get("kv.full", {})
    if int(cache.get("active_length", -1)) != WINDOW + WIDTH:
        raise ReplayInputError("Captured Draft cache active length is not 2056.")
    if int(cache.get("page_size", -1)) != 16:
        raise ReplayInputError("Captured Draft cache page size is not 16.")
    kv_shape = _shape(cache.get("layers", {}).get("0", {}), "k")
    if kv_shape[1:] != (KV_HEADS_PER_RANK, HEAD_DIM):
        raise ReplayInputError(f"Captured rank-local KV shape is invalid: {kv_shape}.")

    # The selected client cycle starts at target position 8192. The Draft's
    # compact table stores the last 2048 committed positions and eight queries.
    return ReplayDescriptor(
        bundle=bundle,
        checkpoint=checkpoint,
        step=step,
        seed=int(common["seed"]),
        prefix_position=8192,
        seq_len=WINDOW,
        seq_len_bound=int(common["seq_lens_sum"]),
        cache_rows=WINDOW + WIDTH,
        input_shape=(WIDTH, HIDDEN_SIZE),
        position_shape=(WIDTH,),
        kv_shape=kv_shape,
        conv_shape=(WIDTH, 2, HIDDEN_SIZE // 16),
    )


def component_seed(seed: int, rank: int, component: str) -> int:
    payload = f"muse-draft-replay-v1:{seed}:{rank}:{component}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def deterministic_values(
    shape: Iterable[int],
    *,
    seed: int,
    rank: int,
    component: str,
    scale: float,
):
    """Generate values on CPU so results do not depend on GPU RNG versions."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(component_seed(seed, rank, component))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).mul_(
        scale
    )


def deterministic_positions(descriptor: ReplayDescriptor):
    import torch

    return torch.arange(
        descriptor.prefix_position,
        descriptor.prefix_position + WIDTH,
        dtype=torch.int64,
    )


def _tensor_summary(tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous()
    raw = value.numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_args(descriptor: ReplayDescriptor, mode: str = "baseline"):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=str(descriptor.checkpoint),
        tokenizer_path=str(descriptor.checkpoint),
        device="cuda",
        tp_size=WORLD_SIZE,
        mem_fraction_static=0.75,
        max_running_requests=2,
        max_total_tokens=4096,
        context_length=16384,
        page_size=PAGE_SIZE,
        kv_cache_dtype="fp8_e4m3",
        attention_backend="triton",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=str(descriptor.checkpoint),
        speculative_draft_attention_backend="triton",
        speculative_num_draft_tokens=WIDTH,
        speculative_draft_window_size=WINDOW,
        enable_amd_monokernel=mode == "mono",
        disable_cuda_graph=True,
    )


def _install_conv_observers(model, captures: dict[int, Any]) -> list[Any]:
    class _Restore:
        def __init__(self, module, original):
            self.module = module
            self.original = original

        def remove(self):
            self.module.prepare = self.original

    handles = []
    for layer_id, layer in enumerate(model.layers):
        conv = layer.mlp_conv
        original = conv.prepare

        def observe(hidden_states, layer_id=layer_id, original=original):
            output, carry = original(hidden_states)
            captures[layer_id] = carry.detach().clone()
            return output, carry

        conv.prepare = observe
        handles.append(_Restore(conv, original))
    return handles


def _install_baseline_state_observers(
    model, states: dict[int, dict[str, Any]]
) -> list[Any]:
    """Capture each native layer's hidden/residual without changing its result."""

    class _Restore:
        def __init__(self, module, original):
            self.module = module
            self.original = original

        def remove(self):
            self.module.forward = self.original

    handles = []
    for layer_id, layer in enumerate(model.layers):
        original = layer.forward

        def observe(*args, layer_id=layer_id, original=original, **kwargs):
            output = original(*args, **kwargs)
            hidden = output[0]
            # Native DFlash carries the MLP output convolution into the next
            # layer. Mono applies it inside D-MLP, so materialize the same
            # logical boundary before comparing layer hidden states.
            if len(output) > 2 and output[2] is not None:
                conv, coefficient = output[2]
                hidden = conv.finish(hidden, coefficient)
            from sglang.srt.model_executor.forward_context import get_attn_backend

            forward_batch = kwargs.get("forward_batch", args[2])
            backend = get_attn_backend()
            rail = backend._window_rail_of(layer.self_attn.attn)
            window_locs = backend._window_write_locs()
            slots = rail.out_cache_loc if rail is not None else None
            if slots is None and window_locs:
                slots = next(iter(window_locs.values()))
            if slots is None:
                slots = forward_batch.out_cache_loc
            key, value = backend.token_to_kv_pool.get_kv_buffer(
                layer.self_attn.attn.layer_id
            )
            states[layer_id] = {
                "hidden": hidden.detach().clone(),
                "raw_hidden": output[0].detach().clone(),
                "residual": output[1].detach().clone(),
                "kv_key": key[slots.long()].float().clone(),
                "kv_value": value[slots.long()].float().clone(),
            }
            return output

        layer.forward = observe
        handles.append(_Restore(layer, original))
    return handles


def _installed_mono(model):
    """Return the object captured by SGLang's installed Mono forward closure."""
    try:
        built = inspect.getclosurevars(model.forward).nonlocals["built"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Unable to inspect the installed Draft Mono runner.") from error
    if not hasattr(built, "runner") or built.runner is None:
        raise RuntimeError("Installed Draft Mono forward has no runner.")
    return built


def _install_mono_state_observer(
    model,
    states: dict[int, dict[str, Any]],
    conv_captures: dict[int, Any],
    *,
    use_reference_attn: bool = False,
    use_reference_mlp: bool = False,
):
    """Capture D-MLP outputs and its output-convolution coefficients.

    The fused D-MLP stores all four projected coefficient groups as tagged
    ``(float32 value, launch tag)`` mailbox pairs. The last two groups are the
    carry consumed by the output convolution and match ``mlp_conv.prepare``.
    """
    import torch

    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp

    built = _installed_mono(model)
    runner = built.runner
    original_attn = runner.launch_attn
    original_mlp = runner.launch
    offset, nbytes = draft_mlp.dm_scratch_layout()["dm_coef"]

    def observe_attn(
        i, rows, hidden, residual, first, positions, forward_batch, stream, trace=False
    ):
        layer = built.model.layers[i]
        reference_input = hidden.clone()
        if first:
            reference_residual = reference_input
            reference_input = layer.input_layernorm(reference_input)
        else:
            reference_input, reference_residual = layer.input_layernorm(
                reference_input, residual.clone()
            )
        reference_pre, reference_coefficient = layer.attention_conv.prepare(
            reference_input
        )
        attention_op = layer.self_attn.attn
        original_attention_forward = attention_op.forward
        reference_raw = None
        reference_qkv = None
        reference_call = None

        def capture_attention(*args, **kwargs):
            nonlocal reference_raw, reference_qkv, reference_call
            reference_qkv = tuple(value.detach().clone() for value in args[:3])
            reference_call = (args[3:], kwargs)
            output = original_attention_forward(*args, **kwargs)
            reference_raw = output.detach().clone()
            return output

        attention_op.forward = capture_attention
        try:
            reference_o = layer.self_attn(
                positions=positions,
                hidden_states=reference_pre,
                forward_batch=forward_batch,
            )
        finally:
            attention_op.forward = original_attention_forward
        if reference_raw is None or reference_qkv is None:
            raise RuntimeError(f"Draft layer {i} did not execute native attention.")
        reference_attention = layer.attention_conv.finish(
            reference_o, reference_coefficient
        )
        backend = runner._backend()
        rail = backend._window_rail_of(layer.self_attn.attn)
        window_locs = backend._window_write_locs()
        slots = rail.out_cache_loc if rail is not None else None
        if slots is None and window_locs:
            slots = next(iter(window_locs.values()))
        if slots is None:
            slots = forward_batch.out_cache_loc
        slots = slots[:rows].long()
        reference_k, reference_v = backend.token_to_kv_pool.get_kv_buffer(
            layer.self_attn.attn.layer_id
        )
        reference_k = reference_k[slots].float().clone()
        reference_v = reference_v[slots].float().clone()
        output = original_attn(
            i,
            rows,
            hidden,
            residual,
            first,
            positions,
            forward_batch,
            stream,
            trace=trace,
        )
        if reference_call is None:
            raise RuntimeError(f"Draft layer {i} did not capture native attention call.")
        reference_tail, reference_kwargs = reference_call
        same_input_reference = original_attention_forward(
            runner.q[:rows],
            runner.k[:rows],
            runner.v[:rows],
            *reference_tail,
            **reference_kwargs,
        )
        mono_k, mono_v = backend.token_to_kv_pool.get_kv_buffer(
            layer.self_attn.attn.layer_id
        )
        states.setdefault(i, {}).update(
            {
                "attention": output[0].detach().clone(),
                "attention_residual": output[1].detach().clone(),
                "reference_attention": reference_attention.detach().clone(),
                "reference_attention_residual": reference_residual.detach().clone(),
                "attention_pre": runner.q[:rows].detach().clone(),
                "reference_q": reference_qkv[0],
                "reference_k": reference_qkv[1],
                "reference_v": reference_qkv[2],
                "attention_o": runner.o[:rows].detach().clone(),
                "reference_attention_o": reference_raw.detach().clone(),
                "same_input_reference_attention_o": same_input_reference.detach().clone(),
                "attention_coefficient": runner.coef[:rows, 2 * draft_mlp.GROUPS :]
                .reshape(rows, 2, draft_mlp.GROUPS)
                .detach()
                .clone(),
                "reference_attention_coefficient": reference_coefficient.detach().clone(),
                "kv_key": mono_k[slots].float().clone(),
                "kv_value": mono_v[slots].float().clone(),
                "reference_kv_key": reference_k,
                "reference_kv_value": reference_v,
            }
        )
        if i == 0:
            prefix_len = int(forward_batch.seq_lens[0].item())
            request = int(forward_batch.req_pool_indices[0].item())
            prefix_slots = backend.req_to_token_pool.req_to_token[
                request, :prefix_len
            ].long()
            states[i].update(
                {
                    "prefix_key": mono_k[prefix_slots].float().clone(),
                    "prefix_value": mono_v[prefix_slots].float().clone(),
                }
            )
        if use_reference_attn:
            return reference_attention, reference_residual
        return output

    def observe_mlp(i, rows, attn_out, residual, stream):
        reference = built._reference(
            built.model.layers[i], attn_out, residual
        )
        output = original_mlp(i, rows, attn_out, residual, stream)
        pairs = runner.scratch[offset : offset + nbytes].view(torch.int32).view(-1, 2)
        coefficient = (
            pairs[:, 0]
            .contiguous()
            .view(torch.float32)
            .view(draft_mlp.MAX_TOKENS, draft_mlp.COEF_ROWS)[:rows]
        )
        groups = draft_mlp.GROUPS
        conv_captures[i] = coefficient[:, 2 * groups :].reshape(
            rows, 2, groups
        ).clone()
        states.setdefault(i, {}).update(
            {
                "hidden": output[0].detach().clone(),
                "residual": output[1].detach().clone(),
                "reference_hidden": reference[0].detach().clone(),
                "reference_residual": reference[1].detach().clone(),
            }
        )
        return reference if use_reference_mlp else output

    runner.launch_attn = observe_attn
    runner.launch = observe_mlp

    class _Restore:
        def remove(self):
            runner.launch_attn = original_attn
            runner.launch = original_mlp

    return _Restore()


def _cpu_snapshot(tensor):
    return tensor.detach().float().cpu().contiguous()


def _allocate_standalone_draft_pool(runner, pool_rows: int) -> None:
    """Allocate a private pool for a Draft-only ModelRunner.

    DFlashWorkerV2 normally passes the already-created Target allocator into
    its Draft runner. This replay intentionally has no Target runner, so the
    Draft must size an equivalent private pool. The flag is changed only while
    ModelRunner performs pool sizing; model loading and every forward retain
    normal Draft-worker semantics.
    """
    if not runner.is_draft_worker or runner.token_to_kv_pool_allocator is not None:
        raise RuntimeError("Standalone Draft pool bootstrap received an invalid runner.")
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    runner.alloc_memory_pool(
        memory_pool_config=MemoryPoolConfig(
            # The paged allocator contributes one guard page to the physical
            # tensor; the capture descriptor records that physical extent.
            max_total_num_tokens=pool_rows - PAGE_SIZE,
            max_running_requests=128,
        )
    )


def _make_forward_batch(runner, descriptor: ReplayDescriptor, rank: int):
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
    from sglang.srt.speculative.draft_worker_common import (
        make_draft_block_spec_info,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    device = torch.device("cuda", rank)
    slots = torch.arange(
        1, descriptor.cache_rows + 1, dtype=torch.int64, device=device
    )
    runner.req_to_token_pool.req_to_token[0, : descriptor.cache_rows].copy_(
        slots.to(torch.int32)
    )
    for layer_id in range(LAYERS):
        key, value = runner.token_to_kv_pool.get_kv_buffer(layer_id)
        key_values = deterministic_values(
            (descriptor.seq_len, KV_HEADS_PER_RANK, HEAD_DIM),
            seed=descriptor.seed,
            rank=rank,
            component=f"layer-{layer_id}-key",
            scale=0.7,
        ).to(dtype=torch.float8_e4m3fn, device=device)
        value_values = deterministic_values(
            (descriptor.seq_len, KV_HEADS_PER_RANK, HEAD_DIM),
            seed=descriptor.seed,
            rank=rank,
            component=f"layer-{layer_id}-value",
            scale=0.7,
        ).to(dtype=torch.float8_e4m3fn, device=device)
        key[slots[: descriptor.seq_len]] = key_values
        value[slots[: descriptor.seq_len]] = value_values

    positions = deterministic_positions(descriptor).to(device)
    input_embeds = deterministic_values(
        descriptor.input_shape,
        seed=descriptor.seed,
        # The Target-produced Draft embedding is replicated after TP
        # collectives. KV heads are rank-local, but this input must be identical.
        rank=0,
        component="input-embeddings",
        scale=0.3,
    ).to(dtype=torch.bfloat16, device=device)
    batch = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=1,
        input_ids=torch.full(
            (WIDTH,), 200002, dtype=torch.int64, device=device
        ),
        req_pool_indices=torch.zeros((1,), dtype=torch.int64, device=device),
        seq_lens=torch.tensor(
            [descriptor.seq_len], dtype=torch.int32, device=device
        ),
        out_cache_loc=slots[descriptor.seq_len :],
        seq_lens_sum=descriptor.seq_len_bound,
        seq_lens_cpu=torch.tensor(
            [descriptor.seq_len_bound], dtype=torch.int32
        ),
        positions=positions,
        input_embeds=input_embeds,
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        spec_info=make_draft_block_spec_info(
            draft_token_num=WIDTH, device=device
        ),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        global_num_token_non_padded_cpu=WIDTH,
        can_run_decode_cuda_graph=False,
    )
    runner.runner_ids.bind_own_plan(batch)
    return batch


def _rank_worker(
    rank: int,
    descriptor: ReplayDescriptor,
    port: int,
    warmup: int,
    iterations: int,
    trace_dir: str | None,
    record_shapes: bool,
    mode: str,
    output_dir: str,
    queue,
) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            HIP_VISIBLE_DEVICES="0,1,2,3",
            HSA_CU_MASK="0-3:0-95",
            SGLANG_ENABLE_AMD_MONOKERNEL="1" if mode == "mono" else "0",
            SGLANG_AMD_MONO_DRAFT_KERNELS="1",
            SGLANG_AMD_MONO_DRAFT_STAGE="2",
            SGLANG_AMD_MONO_CHECK=os.environ.get("SGLANG_AMD_MONO_CHECK", "0"),
            SGLANG_MUSE_MONO_GFX1250_CONFIG="cu96_tp4",
            SGLANG_MUSE_REPLAY_MARKERS="1" if trace_dir is not None else "0",
            SGLANG_USE_AITER="1",
            SGLANG_USE_AITER_UNIFIED_ATTN="1",
            AITER_CUSTOM_AR_FORCE_VMM="1",
            AITER_CUSTOM_AR_MAX_SIZE="1048576",
            SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2="1",
            SGLANG_OPT_SCHEDULER_SHM_SYNC="1",
        )
        if mode == "mono":
            import sgl_amd_mono

            sgl_amd_mono.register_specs()
        import torch
        import torch.distributed as dist

        from sglang.benchmark.one_batch import _init_process_global_configs
        from sglang.srt.arg_groups.model_override_base import resolving_view
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import bootstrap
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.rocm_muse_spark.replay_capture import replay_scope
        from sglang.srt.runtime_context import (
            SpawnRanks,
            get_device,
            publish,
            spawn_world_rank,
        )

        torch.cuda.set_device(rank)
        server_args = _server_args(descriptor, mode)
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0),
                gpu_id=rank,
            ),
        )
        _init_process_global_configs()
        config = ModelConfig.from_server_args(server_args)
        bootstrap.init_parallel_runtime(
            server_args=server_args,
            device=get_device().device,
            dist_port=port,
        )
        bootstrap.init_layer_runtime(model_config=config)
        cfg = resolving_view(server_args)
        runner = ModelRunner(
            model_config=config,
            mem_fraction_static=cfg.mem_fraction_static,
            gpu_id=rank,
            nccl_port=port,
            server_args=server_args,
            is_draft_worker=True,
            draft_attention_backend="triton",
        )
        _allocate_standalone_draft_pool(runner, descriptor.kv_shape[0])
        runner.init_attention_backends()
        # This initializes the eager runner too; graphs remain disabled by the
        # ServerArgs and every replay batch explicitly rejects graph dispatch.
        runner.init_cuda_graphs()
        mono_attached = hasattr(runner.model, "_monokernel_unread_metadata")
        if mode == "mono" and not mono_attached:
            raise RuntimeError("Draft Mono mode requested but plugin did not attach.")
        batch = _make_forward_batch(runner, descriptor, rank)

        def run_once():
            with replay_scope("draft/forward"):
                output = runner.forward(batch).logits_output.hidden_states
            if output is None:
                raise RuntimeError("DFlash2 Draft ModelRunner returned no hidden states.")
            return output

        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()

        profiler = None
        if trace_dir is not None:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=record_shapes,
                with_stack=False,
            )
            profiler.start()
        output = None
        for _ in range(iterations):
            output = run_once()
        torch.cuda.synchronize()
        trace_file = None
        if profiler is not None:
            profiler.stop()
            trace_path = Path(trace_dir) / f"draft.rank_{rank:04d}.trace.json.gz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
            trace_file = str(trace_path)
        assert output is not None

        # Run one unprofiled snapshot iteration. Capturing clones must not alter
        # the trace whose kernel sequence is compared separately in Gate 1.
        layer_states: dict[int, dict[str, Any]] = {}
        conv_captures: dict[int, Any] = {}
        reference_conv_captures: dict[int, Any] = {}
        handles = (
            [
                _install_mono_state_observer(
                    runner.model,
                    layer_states,
                    conv_captures,
                    use_reference_attn=(
                        os.environ.get("SGLANG_MUSE_REPLAY_REFERENCE_ATTN") == "1"
                    ),
                    use_reference_mlp=(
                        os.environ.get("SGLANG_MUSE_REPLAY_REFERENCE_MLP") == "1"
                    ),
                ),
                *_install_conv_observers(
                    runner.model, reference_conv_captures
                ),
            ]
            if mode == "mono"
            else _install_baseline_state_observers(runner.model, layer_states)
            + _install_conv_observers(runner.model, conv_captures)
        )
        output = run_once()
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        if set(layer_states) != set(range(LAYERS)):
            raise RuntimeError(
                f"Draft {mode} layer snapshots are incomplete: {sorted(layer_states)}"
            )
        if set(conv_captures) != set(range(LAYERS)):
            raise RuntimeError(
                f"Draft {mode} convolution snapshots are incomplete: "
                f"{sorted(conv_captures)}"
            )
        if mode == "mono" and set(reference_conv_captures) != set(range(LAYERS)):
            raise RuntimeError(
                "Draft Mono reference convolution snapshots are incomplete: "
                f"{sorted(reference_conv_captures)}"
            )

        cache_state = {}
        snapshot = {
            "rank": rank,
            "mode": mode,
            "output": _cpu_snapshot(output),
            "layers": {
                layer: {
                    name: _cpu_snapshot(value)
                    for name, value in layer_states[layer].items()
                }
                for layer in range(LAYERS)
            },
            "convolution_carry": {
                layer: _cpu_snapshot(conv_captures[layer])
                for layer in range(LAYERS)
            },
            "reference_convolution_carry": {
                layer: _cpu_snapshot(reference_conv_captures[layer])
                for layer in sorted(reference_conv_captures)
            },
            "kv_cache": {},
        }
        for layer_id in range(LAYERS):
            written_key = layer_states[layer_id]["kv_key"]
            written_value = layer_states[layer_id]["kv_value"]
            snapshot["kv_cache"][layer_id] = {
                "written_key": _cpu_snapshot(written_key),
                "written_value": _cpu_snapshot(written_value),
            }
            cache_state[str(layer_id)] = {
                "written_key": _tensor_summary(written_key),
                "written_value": _tensor_summary(written_value),
            }
        snapshot_path = Path(output_dir) / f"rank_{rank:04d}.snapshot.pt"
        torch.save(snapshot, snapshot_path)
        result = {
            "rank": rank,
            "mode": mode,
            "mono_attached": mono_attached,
            "workload": "sglang-dflash2-draft-modelrunner-eager",
            "tp_size": WORLD_SIZE,
            "warmup": warmup,
            "iterations": iterations,
            "output": _tensor_summary(output),
            "input_embeddings": _tensor_summary(batch.input_embeds),
            "positions": batch.positions.cpu().tolist(),
            "kv_cache": cache_state,
            "convolution_carry": {
                str(layer): _tensor_summary(conv_captures[layer])
                for layer in range(LAYERS)
            },
            "snapshot": str(snapshot_path),
            "trace": trace_file,
        }
        if dist.is_initialized():
            dist.barrier()
        destroy_model_parallel()
        destroy_distributed_environment()
        queue.put((rank, result, None))
    except BaseException:
        queue.put((rank, None, traceback.format_exc()))


def run_replay(
    descriptor: ReplayDescriptor,
    *,
    warmup: int,
    iterations: int,
    output_dir: str | Path,
    trace: bool,
    record_shapes: bool,
    timeout: int,
    mode: str = "baseline",
) -> dict[str, Any]:
    if warmup < 0:
        raise ReplayInputError("--warmup must be non-negative.")
    if iterations <= 0:
        raise ReplayInputError("--iterations must be positive.")
    if mode not in {"baseline", "mono"}:
        raise ReplayInputError("--mode must be baseline or mono.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = str(output_dir / "traces") if trace else None
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_rank_worker,
            args=(
                rank,
                descriptor,
                port,
                warmup,
                iterations,
                trace_dir,
                record_shapes,
                mode,
                str(output_dir),
                queue,
            ),
        )
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    received = [queue.get(timeout=timeout) for _ in processes]
    for process in processes:
        process.join(timeout=60)
    errors = [f"rank {rank}:\n{error}" for rank, _, error in received if error]
    errors.extend(
        f"process {index} exited with code {process.exitcode}"
        for index, process in enumerate(processes)
        if process.exitcode != 0
    )
    if errors:
        raise RuntimeError("Four-rank Draft replay failed:\n" + "\n".join(errors))

    ranks = [result for _, result, _ in sorted(received) if result is not None]
    summary = {
        "descriptor": descriptor.summary(),
        "workload": "sglang-dflash2-draft-modelrunner-eager",
        "mode": mode,
        "world_size": WORLD_SIZE,
        "ranks": ranks,
    }
    destination = output_dir / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="draft-replay-output")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "mono"), default="baseline")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the real checkpoint and captured descriptors without GPUs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        descriptor = validate_replay_inputs(
            args.bundle, args.checkpoint, step=args.step, require_weights=True
        )
        if args.preflight_only:
            print(json.dumps(descriptor.summary(), indent=2, sort_keys=True))
            return 0
        summary = run_replay(
            descriptor,
            warmup=args.warmup,
            iterations=args.iterations,
            output_dir=args.output_dir,
            trace=args.trace,
            record_shapes=args.record_shapes,
            timeout=args.timeout,
            mode=args.mode,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (ReplayInputError, RuntimeError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
