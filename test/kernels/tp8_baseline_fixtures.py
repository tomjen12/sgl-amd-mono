# SPDX-License-Identifier: Apache-2.0
"""Deterministic TP8-local fixtures for production Baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch

from tp8_baseline_production import (
    BaselineContractError,
    BaselineResult,
    K1BaselineAdapter,
    KAttnFullBaselineAdapter,
    KAttnSWABaselineAdapter,
    K2BaselineAdapter,
    K3BaselineAdapter,
    ActiveLogicalTP8,
)
from tp8_synthetic_contract import CONTRACT


def _validated_launch(
    launch: Callable[[], BaselineResult],
    *,
    active_npes: int | None = None,
) -> Callable[[], BaselineResult]:
    def checked() -> BaselineResult:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise BaselineContractError(
                "Baseline launch requires an initialized process group"
            )
        world = dist.get_world_size()
        expected = world if active_npes is None else active_npes
        if world not in (1, 2, 4) or world != expected:
            raise BaselineContractError(
                f"active world size {world}, expected {expected} in (1, 2, 4)"
            )
        return launch()

    return checked


@dataclass
class PreparedBaseline:
    stage: str
    operator: str
    backend: str
    launch_count: int
    launch: Callable[[], BaselineResult]
    fixture: Any | None = None


def _cos_sin(device: torch.device, positions: int = 16384) -> torch.Tensor:
    half = CONTRACT.head_dim // 2
    inv = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(positions, device=device, dtype=torch.float32)[:, None]
    freqs = pos * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)


def prepare_k1(rows: int, rank: int, device: torch.device) -> PreparedBaseline:
    from sgl_amd_mono.muse_spark.gfx1250.config import QK_SCALE
    from tp8_logical_case import build_k1_case

    case = build_k1_case(rows, rank, device)
    adapter = K1BaselineAdapter()
    kwargs: dict[str, Any] = {
        "x": case.x,
        "weight": case.weight,
        "positions": case.positions,
        "cos_sin": case.cos_sin,
        "cache_k": case.cache_k,
        "cache_v": case.cache_v,
        "locations": case.locations,
        "mapping": None,
        "eps": 1e-5,
        "q_scale": QK_SCALE,
        "k_scale": case.k_scale,
        "v_scale": case.v_scale,
        "num_heads": CONTRACT.local_q_heads,
        "num_kv_heads": CONTRACT.local_kv_heads,
    }
    return PreparedBaseline(
        stage="k1",
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=2,
        launch=_validated_launch(lambda: adapter(**kwargs)),
        fixture=case,
    )


def prepare_kattn(
    rows: int,
    rank: int,
    device: torch.device,
    *,
    window: int,
) -> PreparedBaseline:
    from sglang.kernels.ops.attention.verify_reduce_norm_gate import (
        verify_reduce_norm_gate,
    )
    from sglang.srt.layers.attention.aiter_backend import _muse_verify_attention
    from tp8_logical_case import build_kattn_case

    requests = rows // CONTRACT.q_len
    if rows not in (8, 16) or requests * CONTRACT.q_len != rows:
        raise BaselineContractError(f"unsupported K-attn width {rows}")
    case = build_kattn_case(rows, rank, window, device)
    output = torch.empty_like(case.q)
    eps = 1e-5
    norm_weight = torch.ones(
        CONTRACT.head_dim, device=device, dtype=torch.bfloat16
    )
    sliding_window = (window - 1, 0) if window else (-1, -1)
    kernels = {"swa_splitkv"} if window else {"full_v2", "full_v2_fp8"}

    class BackendCall:
        sliding_window_size = window

        def __call__(self, q_flat, _k, _v, _batch, *, save_kv_cache=False):
            del save_kv_cache
            handled = _muse_verify_attention(
                q_flat.reshape(rows, CONTRACT.local_q_heads, CONTRACT.head_dim),
                case.cache_k,
                case.cache_v,
                output.reshape(
                    rows, CONTRACT.local_q_heads, CONTRACT.head_dim
                ),
                case.cu_seqlens_q,
                CONTRACT.q_len,
                case.sequence_lengths,
                case.sequence_lengths,
                case.block_table,
                CONTRACT.head_dim**-0.5,
                sliding_window,
                0.0,
                case.k_descale,
                case.v_descale,
                None,
                kernels,
            )
            if not handled:
                raise BaselineContractError("production AITER verify rejected fixture")
            return output

    class FusedNormGate:
        @staticmethod
        def forward_verify_partials(state, out, gate_flat):
            return verify_reduce_norm_gate(
                state,
                out.shape,
                out.dtype,
                gate_flat.reshape(rows, CONTRACT.local_kv_heads, -1),
                eps,
                norm_weight,
            )

    attention = type(
        "SyntheticProductionAttention",
        (),
        {"attn": BackendCall(), "_fused_o_norm_gate": FusedNormGate()},
    )()
    adapter = (
        KAttnSWABaselineAdapter() if window else KAttnFullBaselineAdapter()
    )
    stage = "kattn_window" if window else "kattn_full"
    return PreparedBaseline(
        stage=stage,
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=2,
        launch=_validated_launch(
            lambda: adapter(
                attention=attention,
                q=case.q,
                k=case.k_new,
                v=case.v_new,
                forward_batch=None,
                gate=case.gate,
                save_kv_cache=False,
            )
        ),
        fixture=case,
    )


def prepare_k2(
    rows: int,
    rank: int,
    device: torch.device,
    *,
    active_npes: int = CONTRACT.active_npes,
) -> PreparedBaseline:
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from sglang.kernels.ops.moe.router_pooler_topk import (
        pack_joint_router_pooler_weight,
    )
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        FusedPostNormResidualPreNorm,
    )
    from sglang.srt.rocm_muse_spark.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )
    from tp8_logical_case import build_k2_case
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        POST_EPS,
        RMS_EPS,
        ROUTER_MULT,
        ROUTE_SCALE,
    )

    case = build_k2_case(rows, rank, active_npes, device)
    bf16 = torch.bfloat16
    attention_output = case.attention_output
    stream = case.stream
    w_o = case.w_o

    class OProj:
        def __call__(self, value):
            projected = gemm_a16w16(value, w_o, dtype=bf16)
            return tensor_model_parallel_all_reduce(projected), None

    class Norm:
        hidden_size = CONTRACT.global_hidden
        has_weight = False

        def __init__(self, weight=None, eps=RMS_EPS):
            self.weight = weight
            self.has_weight = weight is not None
            self.eps = eps

        def _get_effective_weight(self):
            return self.weight

    residual = MuseSpark1xNormPreservingResidualConnection(
        CONTRACT.global_hidden
    ).to(device)
    residual._alpha = case.weights["alpha"]
    residual._beta = case.weights["beta"]
    fused = FusedPostNormResidualPreNorm(
        Norm(eps=POST_EPS),
        residual,
        Norm(case.weights["pre_w"], eps=RMS_EPS),
    )
    boundary = SimpleNamespace(forward=fused.forward_triton)
    router = case.weights["router"]
    pooler = case.weights["w_pool"]
    packed = pack_joint_router_pooler_weight(router, pooler)

    class EffectiveNorm:
        eps = 1e-5

        def __init__(self):
            self.weight = torch.ones(
                CONTRACT.moe_hidden, device=device, dtype=bf16
            )

        def _get_effective_weight(self):
            return self.weight

    pre_expert_norm = EffectiveNorm()
    pre_expert_norm.weight.copy_(case.weights["xn_w"])
    pre_expert_proj = SimpleNamespace(weight=packed[: CONTRACT.moe_hidden])
    mlp = SimpleNamespace(
        layer_idx=9,
        router_pooler_packed_weight=packed,
        pre_expert_proj=pre_expert_proj,
        expert_bias=case.weights["bias"],
        pre_expert_norm=pre_expert_norm,
        router_multiplier=ROUTER_MULT,
        route_scale=ROUTE_SCALE,
        router=SimpleNamespace(weight=router),
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=OProj()),
        _fused_attn_boundary=boundary,
        mlp=mlp,
    )
    adapter = K2BaselineAdapter()
    return PreparedBaseline(
        stage="k2",
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=4 if active_npes == 1 else 5,
        launch=_validated_launch(
            lambda: adapter(
                layer=layer,
                stream=stream,
                attention_output=attention_output,
            ),
            active_npes=active_npes,
        ),
        fixture=case,
    )


def prepare_k3(
    rows: int,
    rank: int,
    device: torch.device,
    *,
    active_npes: int = CONTRACT.active_npes,
    controlled_package=None,
) -> PreparedBaseline:
    from types import SimpleNamespace

    from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
        unpool_gemm_norm_residual,
    )
    from tp8_logical_case import build_k3_case

    case = build_k3_case(rows, rank, active_npes, device)
    if controlled_package is not None:
        if controlled_package.width != rows:
            raise ValueError(
                f"controlled routing width {controlled_package.width}, expected {rows}"
            )
        case.xin.copy_(
            torch.tensor(
                controlled_package.xin,
                dtype=case.xin.dtype,
                device=device,
            )
        )
        case.ids.copy_(
            torch.tensor(
                controlled_package.ids,
                dtype=case.ids.dtype,
                device=device,
            )
        )
        case.scores.copy_(
            torch.tensor(
                controlled_package.scores,
                dtype=case.scores.dtype,
                device=device,
            )
        )
    mask = torch.zeros(
        CONTRACT.global_experts, device=device, dtype=torch.int32
    )
    mask[case.local_start : case.local_start + CONTRACT.local_experts] = 1
    experts = SimpleNamespace(
        num_local_experts=CONTRACT.local_experts,
        gfx1250_a8w4=(case.w13, case.w2, case.s13, case.s2, mask),
        expert_hidden_size=CONTRACT.moe_hidden,
    )
    weights = case.weights

    adapter = K3BaselineAdapter(ActiveLogicalTP8(active_npes))
    return PreparedBaseline(
        stage="k3",
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=8 if active_npes == 1 else 9,
        launch=_validated_launch(
            lambda: adapter(
                x=case.xin,
                scores=case.scores,
                ids=case.ids,
                experts=experts,
                local_start=case.local_start,
                norm_weight=weights["pen_w"],
                eps=1e-5,
                gain=0.0,
                unpool=unpool_gemm_norm_residual,
                unpool_args=(
                    weights["w_unpool"],
                    case.stream,
                    weights["pfw"],
                    weights["npw"],
                    weights["alpha"],
                    weights["beta"],
                    1e-5,
                    1e-5,
                ),
            ),
            active_npes=active_npes,
        ),
        fixture=case,
    )
