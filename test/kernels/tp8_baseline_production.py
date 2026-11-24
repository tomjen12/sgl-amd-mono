# SPDX-License-Identifier: Apache-2.0
"""Production-only Baseline adapters for the TP8 active-four harness.

This module deliberately contains no mathematical fallback.  Every stage calls
the same SGLang/AITER entrypoint used by the Muse production model; an import or
dispatch failure is therefore a failed Baseline, not permission to use ATen.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from tp8_synthetic_contract import CONTRACT, ContractError, validate_tensor


class BaselineContractError(ContractError):
    """The requested run is not the recipe's production Baseline."""


# Values which affect production operator selection.  SGLANG_AMD_MONO_BLOCKS is
# synthetic-harness-only, but is validated here because a wrong resident grid
# invalidates Baseline/Mono comparisons.
PRODUCTION_ENV = {
    "HIP_VISIBLE_DEVICES": "0,1,2,3",
    "HSA_CU_MASK": "0-3:0-95",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "MUSE_MOE_BACKEND": "triton",
    "AITER_CUSTOM_AR_FORCE_VMM": "1",
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
    "SGLANG_AMD_MONO_BLOCKS": "96",
}


def _string_env(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def validate_production_environment(
    environ: Mapping[str, Any] | None = None,
    *,
    recipe_env: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate dispatch-sensitive process values against the production recipe.

    ``recipe_env`` is accepted explicitly so unit tests and recipe loaders can
    prove that the checked values have not drifted from the pinned recipe.
    """
    actual = _string_env(os.environ if environ is None else environ)
    recipe = _string_env(recipe_env or {})
    expected = dict(PRODUCTION_ENV)
    for name in tuple(expected):
        if name == "SGLANG_AMD_MONO_BLOCKS":
            continue
        if recipe and name not in recipe:
            raise BaselineContractError(f"production recipe is missing env {name}")
        if name in recipe and recipe[name] != expected[name]:
            raise BaselineContractError(
                f"production recipe {name}={recipe[name]!r}, expected {expected[name]!r}"
            )
    errors = [
        f"{name}={actual.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if errors:
        raise BaselineContractError(
            "non-production Baseline environment: " + "; ".join(errors)
        )
    return {name: actual[name] for name in expected}


def load_recipe_environment(path: str | Path) -> dict[str, str]:
    """Load ``server.env`` from the pinned YAML recipe."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - production image has PyYAML
        raise BaselineContractError("PyYAML is required to validate the recipe") from exc
    value = yaml.safe_load(Path(path).read_text())
    try:
        env = value["server"]["env"]
    except (KeyError, TypeError) as exc:
        raise BaselineContractError(f"{path}: missing server.env") from exc
    if not isinstance(env, Mapping):
        raise BaselineContractError(f"{path}: server.env is not a mapping")
    return _string_env(env)


def validate_recipe_environment(
    path: str | Path, environ: Mapping[str, Any] | None = None
) -> dict[str, str]:
    return validate_production_environment(
        environ, recipe_env=load_recipe_environment(path)
    )


def _import_attr(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise BaselineContractError(
            f"production entrypoint {module}.{name} is unavailable"
        ) from exc


def _tensor(
    evidence: dict[str, dict[str, Any]],
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    dtype: Any | None = None,
    contiguous: bool = True,
) -> None:
    try:
        descriptor = validate_tensor(
            name, value, shape=shape, dtype=dtype, contiguous=contiguous
        )
        descriptor["layout"] = (
            "contiguous_row_major"
            if descriptor["contiguous"]
            else "production_noncontiguous"
        )
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr):
            address = int(data_ptr())
            alignment = 1
            while alignment < 256 and address % (alignment * 2) == 0:
                alignment *= 2
            descriptor["data_ptr"] = address
            descriptor["alignment_bytes"] = alignment
        else:
            descriptor["data_ptr"] = None
            descriptor["alignment_bytes"] = None
        evidence[name] = descriptor
    except ContractError as exc:
        raise BaselineContractError(str(exc)) from exc


@dataclass(frozen=True)
class BaselineResult:
    value: Any
    tensors: dict[str, dict[str, Any]]
    operator: str
    backend: str


class K1BaselineAdapter:
    operator = "qkv_proj_qk_norm_rope_cache"
    backend = "sglang_rocm_fused_qkv_cache"

    def __call__(self, **kw: Any) -> BaselineResult:
        x, weight = kw["x"], kw["weight"]
        rows, bf16 = int(x.shape[0]), x.dtype
        if rows not in (8, 16):
            raise BaselineContractError(f"k1 width {rows}, expected 8 or 16")
        evidence: dict[str, dict[str, Any]] = {}
        _tensor(evidence, "k1.x", x, (rows, CONTRACT.global_hidden), dtype=bf16)
        _tensor(
            evidence,
            "k1.weight",
            weight,
            (CONTRACT.local_qkvg_rows, CONTRACT.global_hidden),
            dtype=bf16,
        )
        _tensor(evidence, "k1.positions", kw["positions"], (rows,))
        _tensor(evidence, "k1.locations", kw["locations"], (rows,))
        cache_shape = tuple(int(v) for v in kw["cache_k"].shape)
        if cache_shape[1:] != (
            CONTRACT.local_kv_heads,
            CONTRACT.head_dim,
        ):
            raise BaselineContractError(f"k1 cache local shape {cache_shape}")
        _tensor(evidence, "k1.cache_k", kw["cache_k"], cache_shape)
        _tensor(evidence, "k1.cache_v", kw["cache_v"], cache_shape)
        fn = _import_attr(
            "sglang.kernels.ops.attention.qk_norm_rope_cache",
            self.operator,
        )
        return BaselineResult(fn(**kw), evidence, self.operator, self.backend)


class KAttnBaselineAdapter:
    operator = "aiter_target_verify+verify_reduce_norm_gate"
    backend = "aiter_unified_attention+sglang_rocm"

    def __call__(
        self,
        *,
        attention: Any,
        q: Any,
        k: Any,
        v: Any,
        forward_batch: Any,
        gate: Any,
        save_kv_cache: bool = False,
    ) -> BaselineResult:
        rows, bf16 = int(q.shape[0]), q.dtype
        if rows not in (8, 16):
            raise BaselineContractError(f"kattn width {rows}, expected 8 or 16")
        evidence: dict[str, dict[str, Any]] = {}
        _tensor(evidence, "kattn.q", q, (rows, CONTRACT.local_q_width), dtype=bf16)
        _tensor(evidence, "kattn.k", k, (rows, CONTRACT.local_kv_width), dtype=bf16)
        _tensor(evidence, "kattn.v", v, (rows, CONTRACT.local_kv_width), dtype=bf16)
        _tensor(evidence, "kattn.gate", gate, (rows, CONTRACT.local_q_width), dtype=bf16)

        # This is the configured attention layer's production backend (AITER in
        # the recipe), not a direct attention reimplementation.
        deferral = _import_attr(
            "sglang.srt.rocm_muse_spark.model", "VERIFY_REDUCE_DEFERRAL"
        )
        deferral.requested = True
        try:
            output = attention.attn(
                q, k, v, forward_batch, save_kv_cache=save_kv_cache
            )
        finally:
            deferral.requested = False
        state = deferral.take()
        if state is None:
            raise BaselineContractError(
                "AITER target-verify did not return deferred partials"
            )
        output = attention._fused_o_norm_gate.forward_verify_partials(
            state, output, gate
        )
        return BaselineResult(output, evidence, self.operator, self.backend)


class KAttnFullBaselineAdapter(KAttnBaselineAdapter):
    """Production AITER target-verify adapter for a full-attention layer."""

    def __call__(self, *, attention: Any, **kw: Any) -> BaselineResult:
        window = getattr(attention.attn, "sliding_window_size", -1)
        if window not in (-1, 0, None):
            raise BaselineContractError(
                f"full attention adapter received sliding window {window}"
            )
        return super().__call__(attention=attention, **kw)


class KAttnSWABaselineAdapter(KAttnBaselineAdapter):
    """Production AITER target-verify adapter for a sliding-attention layer."""

    def __call__(self, *, attention: Any, **kw: Any) -> BaselineResult:
        window = getattr(attention.attn, "sliding_window_size", None)
        if not isinstance(window, int) or window <= 0:
            raise BaselineContractError(
                f"SWA adapter requires a positive sliding window, got {window!r}"
            )
        return super().__call__(attention=attention, **kw)


class K2BaselineAdapter:
    operator = "attention_boundary+route_with_shared_pooler"
    backend = "sglang_rocm+aiter"

    def __call__(
        self, *, layer: Any, stream: Any, attention_output: Any
    ) -> BaselineResult:
        rows, bf16 = int(attention_output.shape[0]), attention_output.dtype
        if rows not in (8, 16):
            raise BaselineContractError(f"k2 width {rows}, expected 8 or 16")
        evidence: dict[str, dict[str, Any]] = {}
        _tensor(
            evidence,
            "k2.attention_output",
            attention_output,
            (rows, CONTRACT.local_q_width),
            dtype=bf16,
        )
        _tensor(
            evidence,
            "k2.stream",
            stream,
            (rows, CONTRACT.global_hidden),
            contiguous=False,
        )
        projected = layer.self_attn.o_proj(attention_output)
        projected = projected[0] if isinstance(projected, tuple) else projected
        _tensor(
            evidence,
            "k2.projected",
            projected,
            (rows, CONTRACT.global_hidden),
            dtype=bf16,
        )
        ffn_input, next_stream, _ = layer._fused_attn_boundary.forward(
            stream, projected, branch_dtype=bf16
        )
        route = _import_attr(
            "sglang.srt.rocm_muse_spark.router_pooler",
            "route_with_shared_pooler",
        )
        routed = route(layer.mlp, ffn_input)
        _tensor(
            evidence,
            "k2.ffn_input",
            ffn_input,
            (rows, CONTRACT.global_hidden),
            dtype=bf16,
        )
        return BaselineResult(
            (next_stream, *routed), evidence, self.operator, self.backend
        )


class ActiveLogicalTP8:
    """Active-rank collective whose absent logical TP8 ranks contribute zeros."""

    logical_world_size = CONTRACT.logical_tp

    def __init__(self, active_world_size: int = CONTRACT.active_npes, group: Any = None):
        if active_world_size not in (1, 2, 4):
            raise BaselineContractError(
                f"active world size {active_world_size}, expected 1, 2, or 4"
            )
        self.active_world_size = active_world_size
        self.group = group

    def validate(self) -> tuple[int, int]:
        dist = _import_attr("torch", "distributed")
        if not dist.is_initialized():
            raise BaselineContractError("active-four distributed group is not initialized")
        world, rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        if world != self.active_world_size:
            raise BaselineContractError(
                f"active world size {world}, expected {self.active_world_size}"
            )
        return rank, world

    def all_reduce(self, value: Any) -> Any:
        self.validate()
        all_reduce = _import_attr(
            "sglang.srt.distributed.communication_op",
            "tensor_model_parallel_all_reduce",
        )
        return all_reduce(value)

    @contextmanager
    def patch_model_collective(self) -> Iterator[None]:
        """Patch the symbol imported by Muse's model for an isolated adapter call."""
        model = importlib.import_module("sglang.srt.rocm_muse_spark.model")
        old = model.tensor_model_parallel_all_reduce
        model.tensor_model_parallel_all_reduce = self.all_reduce
        try:
            yield
        finally:
            model.tensor_model_parallel_all_reduce = old


Active4LogicalTP8 = ActiveLogicalTP8


class K3BaselineAdapter:
    operator = "run_gfx1250_a8w4+finalize_allreduce+unpool_boundary"
    backend = "aiter_a8w4_tdm+sglang_rocm"

    def __init__(self, distributed: ActiveLogicalTP8):
        self.distributed = distributed

    def __call__(
        self,
        *,
        x: Any,
        scores: Any,
        ids: Any,
        experts: Any,
        local_start: int,
        norm_weight: Any,
        eps: float,
        gain: float,
        unpool: Callable[[Any], Any],
        unpool_args: tuple[Any, ...] = (),
        unpool_kwargs: Mapping[str, Any] | None = None,
        quantized: Any = None,
    ) -> BaselineResult:
        rows, bf16 = int(x.shape[0]), x.dtype
        if rows not in (8, 16):
            raise BaselineContractError(f"k3 width {rows}, expected 8 or 16")
        evidence: dict[str, dict[str, Any]] = {}
        _tensor(
            evidence,
            "k3.x",
            x,
            (rows, CONTRACT.moe_hidden),
            dtype=bf16,
        )
        _tensor(evidence, "k3.ids", ids, (rows, CONTRACT.top_k))
        _tensor(evidence, "k3.scores", scores, (rows, CONTRACT.top_k))
        run = _import_attr(
            "sglang.srt.rocm_muse_spark.mxfp4_moe", "run_gfx1250_a8w4"
        )
        local = run(
            x,
            scores,
            ids,
            experts,
            local_start,
            norm_weight,
            eps,
            gain,
            quantized=quantized,
        )
        reduced = self.distributed.all_reduce(local)
        if not callable(unpool) or not getattr(unpool, "__module__", "").startswith(
            "sglang."
        ):
            raise BaselineContractError(
                "k3 unpool must be a production SGLang entrypoint"
            )
        output = unpool(
            reduced,
            *unpool_args,
            **({} if unpool_kwargs is None else dict(unpool_kwargs)),
        )
        return BaselineResult(output, evidence, self.operator, self.backend)
