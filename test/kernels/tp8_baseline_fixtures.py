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
    Active4LogicalTP8,
)
from tp8_synthetic_contract import CONTRACT, deterministic_ep8_routing


@dataclass
class PreparedBaseline:
    stage: str
    operator: str
    backend: str
    launch_count: int
    launch: Callable[[], BaselineResult]


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
    torch.manual_seed(1000 + 17 * rows + rank)
    bf16 = torch.bfloat16
    x = torch.randn(
        rows, CONTRACT.global_hidden, device=device, dtype=torch.float32
    ).to(bf16)
    weight = (
        torch.randn(
            CONTRACT.local_qkvg_rows,
            CONTRACT.global_hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).to(bf16)
    cos_sin = _cos_sin(device)
    positions = torch.arange(rows, device=device, dtype=torch.int64) + 8192
    locations = torch.arange(rows, device=device, dtype=torch.int64)
    cache_k = torch.zeros(
        64,
        CONTRACT.local_kv_heads,
        CONTRACT.head_dim,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    cache_v = torch.zeros_like(cache_k)
    k_scale = torch.ones(1, device=device, dtype=torch.float32)
    v_scale = torch.ones(1, device=device, dtype=torch.float32)
    adapter = K1BaselineAdapter()
    kwargs: dict[str, Any] = {
        "x": x,
        "weight": weight,
        "positions": positions,
        "cos_sin": cos_sin,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "locations": locations,
        "mapping": None,
        "eps": 1e-5,
        "q_scale": 1.0,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "num_heads": CONTRACT.local_q_heads,
        "num_kv_heads": CONTRACT.local_kv_heads,
    }
    return PreparedBaseline(
        stage="k1",
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=2,
        launch=lambda: adapter(**kwargs),
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

    requests = rows // CONTRACT.q_len
    if rows not in (8, 16) or requests * CONTRACT.q_len != rows:
        raise BaselineContractError(f"unsupported K-attn width {rows}")
    torch.manual_seed(2000 + 17 * rows + rank + window)
    q = torch.randn(
        rows, CONTRACT.local_q_width, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    k_new = torch.zeros(
        rows, CONTRACT.local_kv_width, device=device, dtype=torch.bfloat16
    )
    v_new = torch.zeros_like(k_new)
    gate = torch.randn_like(q)
    sequence_lengths = torch.tensor(
        [8192 + 137 * request for request in range(requests)],
        device=device,
        dtype=torch.int32,
    )
    blocks_per_request = 520
    total_blocks = requests * blocks_per_request
    cache = torch.randn(
        2,
        total_blocks,
        16,
        CONTRACT.local_kv_heads,
        CONTRACT.head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    cache_k, cache_v = cache[0], cache[1]
    block_table = torch.arange(
        total_blocks, device=device, dtype=torch.int32
    ).reshape(requests, blocks_per_request)
    cu_seqlens_q = torch.arange(
        requests + 1, device=device, dtype=torch.int32
    ) * CONTRACT.q_len
    output = torch.empty_like(q)
    k_descale = torch.ones(1, device=device, dtype=torch.float32)
    v_descale = torch.ones(1, device=device, dtype=torch.float32)
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
                cache_k,
                cache_v,
                output.reshape(
                    rows, CONTRACT.local_q_heads, CONTRACT.head_dim
                ),
                cu_seqlens_q,
                CONTRACT.q_len,
                sequence_lengths,
                sequence_lengths,
                block_table,
                CONTRACT.head_dim**-0.5,
                sliding_window,
                0.0,
                k_descale,
                v_descale,
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
        launch=lambda: adapter(
            attention=attention,
            q=q,
            k=k_new,
            v=v_new,
            forward_batch=None,
            gate=gate,
            save_kv_cache=False,
        ),
    )


def prepare_k2(rows: int, rank: int, device: torch.device) -> PreparedBaseline:
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

    torch.manual_seed(3000 + 17 * rows + rank)
    bf16 = torch.bfloat16
    attention_output = torch.randn(
        rows, CONTRACT.local_q_width, device=device, dtype=torch.float32
    ).to(bf16)
    stream = torch.randn(
        rows, CONTRACT.global_hidden, device=device, dtype=torch.float32
    )
    w_o = (
        torch.randn(
            CONTRACT.global_hidden,
            CONTRACT.local_q_width,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).to(bf16)

    class OProj:
        def __call__(self, value):
            projected = gemm_a16w16(value, w_o, dtype=bf16)
            return tensor_model_parallel_all_reduce(projected), None

    class Norm:
        hidden_size = CONTRACT.global_hidden
        eps = 1e-5
        has_weight = True

        def __init__(self):
            self.weight = torch.ones(
                self.hidden_size, device=device, dtype=bf16
            )

        def _get_effective_weight(self):
            return self.weight

    fused = FusedPostNormResidualPreNorm(Norm(), object(), Norm())
    boundary = SimpleNamespace(forward=fused.forward_triton)
    router = torch.randn(
        CONTRACT.global_hidden,
        CONTRACT.global_experts,
        device=device,
        dtype=torch.float32,
    )
    pooler = (
        torch.randn(
            CONTRACT.moe_hidden,
            CONTRACT.global_hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).to(bf16)
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
    pre_expert_proj = SimpleNamespace(weight=packed[: CONTRACT.moe_hidden])
    mlp = SimpleNamespace(
        layer_idx=9,
        router_pooler_packed_weight=packed,
        pre_expert_proj=pre_expert_proj,
        expert_bias=torch.zeros(
            CONTRACT.global_experts, device=device, dtype=torch.float32
        ),
        pre_expert_norm=pre_expert_norm,
        router_multiplier=1.0,
        route_scale=1.0,
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
        launch_count=5,
        launch=lambda: adapter(
            layer=layer,
            stream=stream,
            attention_output=attention_output,
        ),
    )


def prepare_k3(rows: int, rank: int, device: torch.device) -> PreparedBaseline:
    from types import SimpleNamespace

    import test_muse_spark_k3_gfx1250 as k3_test
    from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
        unpool_gemm_norm_residual,
    )

    ids_list, scores_list = deterministic_ep8_routing(rows)
    ids = torch.tensor(ids_list, device=device, dtype=torch.int32)
    scores = torch.tensor(scores_list, device=device, dtype=torch.float32)
    case = k3_test.k3_case(
        rows,
        False,
        4000 + 17 * rows + rank,
        experts=CONTRACT.local_experts,
        ids=ids,
    )
    case.scores.copy_(scores)
    local_start = rank * CONTRACT.local_experts
    mask = torch.zeros(
        CONTRACT.global_experts, device=device, dtype=torch.int32
    )
    mask[local_start : local_start + CONTRACT.local_experts] = 1
    experts = SimpleNamespace(
        num_local_experts=CONTRACT.local_experts,
        gfx1250_a8w4=(case.w13, case.w2, case.s13, case.s2, mask),
        expert_hidden_size=CONTRACT.moe_hidden,
    )
    weights = case.w

    adapter = K3BaselineAdapter(Active4LogicalTP8())
    return PreparedBaseline(
        stage="k3",
        operator=adapter.operator,
        backend=adapter.backend,
        launch_count=9,
        launch=lambda: adapter(
            x=case.xin,
            scores=case.scores,
            ids=case.ids,
            experts=experts,
            local_start=local_start,
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
    )
