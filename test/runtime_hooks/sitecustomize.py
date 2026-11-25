# SPDX-License-Identifier: Apache-2.0
"""Opt-in production structural call spy loaded only by run.sh test captures."""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = os.getenv("MUSE_STRUCTURAL_CALL_SPY_DIR")
_LOCK = threading.Lock()
_ORDINAL = 0


def _capture_active() -> bool:
    module = sys.modules.get("sglang.srt.rocm_muse_spark.replay_capture")
    active = getattr(module, "_ACTIVE_METADATA", None)
    return active is not None and active.get() is not None


def _emit(stage: str, operator: str, critical_args: dict) -> None:
    global _ORDINAL
    if not _ROOT or not _capture_active():
        return
    with _LOCK:
        ordinal = _ORDINAL
        _ORDINAL += 1
        row = {
            "pid": os.getpid(),
            "ordinal": ordinal,
            "stage": stage,
            "operator": operator,
            "critical_args": critical_args,
        }
        root = Path(_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"pid_{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _wrap_function(module, name: str, factory) -> None:
    original = getattr(module, name)
    if getattr(original, "_muse_structural_spy", False):
        return
    wrapped = factory(original)
    wrapped._muse_structural_spy = True
    setattr(module, name, wrapped)


def _install_model_hooks(model) -> None:
    def qkv_factory(original):
        @functools.wraps(original)
        def wrapped(attention, positions, hidden_states, forward_batch):
            output = original(attention, positions, hidden_states, forward_batch)
            _emit(
                "k1",
                "qkv_proj_norm_rope_kv_write",
                {
                    "layer": int(attention.attn.layer_id),
                    "rows": int(hidden_states.shape[0]),
                    "fused_kv_write": output is not None,
                },
            )
            return output

        return wrapped

    _wrap_function(model, "qkv_proj_norm_rope_kv_write", qkv_factory)

    route = model.MuseSpark1xMoE.route
    if not getattr(route, "_muse_structural_spy", False):

        @functools.wraps(route)
        def route_wrapped(moe, hidden_states, *args, **kwargs):
            output = route(moe, hidden_states, *args, **kwargs)
            _emit(
                "k2",
                "MuseSpark1xMoE.route",
                {
                    "layer": int(moe.layer_idx),
                    "rows": int(hidden_states.shape[0]),
                    "top_k": int(moe.top_k),
                    "local_expert_start": int(moe.local_expert_start),
                    "local_experts": int(moe.num_local_experts),
                },
            )
            return output

        route_wrapped._muse_structural_spy = True
        model.MuseSpark1xMoE.route = route_wrapped

    run_experts = model.MuseSpark1xMoE.run_experts
    if not getattr(run_experts, "_muse_structural_spy", False):

        @functools.wraps(run_experts)
        def experts_wrapped(moe, expert_input, topk_scores, topk_ids, *args, **kwargs):
            output = run_experts(
                moe, expert_input, topk_scores, topk_ids, *args, **kwargs
            )
            use_a4w4 = bool(args[1] if len(args) > 1 else kwargs.get("use_a4w4"))
            packed = getattr(moe.experts, "gfx1250_a8w4", None)
            gfx1250_ready = packed is not None
            packed_metadata = {}
            if use_a4w4 and gfx1250_ready:
                for name, tensor in zip(("w13", "w2", "s13", "s2"), packed[:4]):
                    packed_metadata[name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "stride": list(tensor.stride()),
                        "address": int(tensor.data_ptr()),
                        "address_mod_256": int(tensor.data_ptr()) % 256,
                        "is_shuffled": bool(getattr(tensor, "is_shuffled", False)),
                    }
            _emit(
                "k3",
                (
                    "run_gfx1250_a8w4"
                    if use_a4w4 and gfx1250_ready
                    else "production_expert_fallback"
                ),
                {
                    "layer": int(moe.layer_idx),
                    "rows": int(topk_ids.shape[0]),
                    "use_a4w4": use_a4w4,
                    "gfx1250_weights_ready": gfx1250_ready,
                    "packed_metadata": packed_metadata,
                    "local_expert_start": int(moe.local_expert_start),
                    "local_experts": int(moe.num_local_experts),
                },
            )
            return output

        experts_wrapped._muse_structural_spy = True
        model.MuseSpark1xMoE.run_experts = experts_wrapped


def _install_attention_hook(radix) -> None:
    forward = radix.RadixAttention.forward
    if getattr(forward, "_muse_structural_spy", False):
        return

    @functools.wraps(forward)
    def wrapped(attention, q, k, v, forward_batch, *args, **kwargs):
        output = forward(attention, q, k, v, forward_batch, *args, **kwargs)
        try:
            backend = radix.get_attn_backend()
            backend_name = type(backend).__name__
        except Exception:
            backend_name = "unknown"
        _emit(
            "kattn",
            "RadixAttention.forward",
            {
                "layer": int(attention.layer_id),
                "rows": int(q.shape[0]),
                "attention_variant": (
                    "swa" if int(attention.sliding_window_size) > 0 else "full"
                ),
                "kv_backend": backend_name,
                "kv_cache_dtype": "fp8_e4m3",
                "page_size": 16,
            },
        )
        return output

    wrapped._muse_structural_spy = True
    radix.RadixAttention.forward = wrapped


def _installer() -> None:
    deadline = time.monotonic() + 120
    model_done = False
    attention_done = False
    while time.monotonic() < deadline and not (model_done and attention_done):
        try:
            model = sys.modules.get("sglang.srt.rocm_muse_spark.model")
            if model is not None and hasattr(model, "MuseSpark1xMoE"):
                _install_model_hooks(model)
                model_done = True
            radix = sys.modules.get("sglang.srt.layers.radix_attention")
            if radix is not None and hasattr(radix, "RadixAttention"):
                _install_attention_hook(radix)
                attention_done = True
        except Exception as exc:
            if _ROOT:
                root = Path(_ROOT)
                root.mkdir(parents=True, exist_ok=True)
                (root / f"install_error_{os.getpid()}.txt").write_text(repr(exc))
            return
        time.sleep(0.005)


if _ROOT:
    threading.Thread(target=_installer, name="muse-structural-call-spy", daemon=True).start()
