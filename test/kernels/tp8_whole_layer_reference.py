# SPDX-License-Identifier: Apache-2.0
"""Independent eager PyTorch reference for a Width-8 middle MXFP4 layer."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist

from tp8_logical_case import (
    k3_local_reference,
    k3_reduce_unpool_reference,
    kattn_reference,
)
from tp8_synthetic_contract import CONTRACT


def _bf(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def _rms(
    value: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    result = value.float() * torch.rsqrt(
        value.float().pow(2).mean(-1, keepdim=True) + eps
    )
    if weight is not None:
        result *= weight.float()
    return result.to(torch.bfloat16)


def _gather_rank_order(
    value: torch.Tensor,
    world: int,
    peer_group,
) -> tuple[torch.Tensor, ...]:
    gathered: list[torch.Tensor | None] = [None] * world
    dist.all_gather_object(gathered, value.cpu(), group=peer_group)
    if any(item is None for item in gathered):
        raise RuntimeError("independent layer reference gather incomplete")
    return tuple(item for item in gathered if item is not None)


def whole_layer_reference(
    *,
    layer,
    buffers,
    rank: int,
    world: int,
    peer_group,
    window: int,
    next_k1: bool,
) -> dict[str, torch.Tensor]:
    """Compose K-attn, K2, decoded-expert K3, and optional next K1."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        POST_EPS,
        RMS_EPS,
        ROUTE_EPS,
        ROUTE_SCALE,
        ROUTER_MULT,
    )

    rows = 8
    q = buffers.q[:rows].clone()
    gate = buffers.g[:rows].clone()
    residual = buffers.res[:rows].clone()
    cache_k = layer.k_cache.reshape(
        -1, 16, CONTRACT.local_kv_heads, HEAD_DIM
    )
    cache_v = layer.v_cache.reshape_as(cache_k)
    attention_case = SimpleNamespace(
        q=q,
        sequence_lengths=layer.seq_lens,
        cache_k=cache_k,
        cache_v=cache_v,
        block_table=layer.table,
        window=window,
    )
    attention = kattn_reference(attention_case, fp8_qp=False)
    heads = attention.float().reshape(rows, -1, HEAD_DIM)
    gated = (
        heads
        * torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        * torch.sigmoid(gate.float().reshape_as(heads))
    ).to(torch.bfloat16).reshape(rows, -1)
    local_o = (
        gated.float() @ layer.w2k["w_o"].float().T
    ).to(torch.bfloat16)
    o_parts = _gather_rank_order(local_o, world, peer_group)
    reduced_o = torch.zeros_like(residual)
    for part in o_parts:
        reduced_o += part.to(
            device=residual.device, dtype=torch.bfloat16
        ).float()
    reduced_o = reduced_o.to(torch.bfloat16)

    post = _rms(reduced_o, POST_EPS).float()
    stream_after_k2 = (
        layer.w2k["alpha"] * residual
        + layer.w2k["beta"] * post
    )
    router_input = (
        stream_after_k2
        * torch.rsqrt(
            stream_after_k2.pow(2).mean(-1, keepdim=True) + RMS_EPS
        )
        * layer.w2k["pre_w"].float()
    )
    logits = router_input @ layer.w2k["w_router"]
    sigmoid_scores = torch.sigmoid(logits * ROUTER_MULT)
    biased = sigmoid_scores + layer.w2k["bias"]
    order = torch.argsort(biased, dim=-1, descending=True, stable=True)
    ids = order[:, : CONTRACT.top_k].to(torch.int32)
    selected = sigmoid_scores.gather(1, ids.long())
    scores = (
        selected
        / (selected.sum(-1, keepdim=True) + ROUTE_EPS)
        * ROUTE_SCALE
    )

    pool_rows = CONTRACT.moe_hidden // CONTRACT.logical_tp
    local_pool = (
        _rms(stream_after_k2, RMS_EPS, layer.w2k["pre_w"]).float()
        @ layer.w2k["w_pool"][
            rank * pool_rows : (rank + 1) * pool_rows
        ].float().T
    ).to(torch.bfloat16)
    pool_parts = _gather_rank_order(local_pool, world, peer_group)
    pooled = torch.zeros(
        rows,
        CONTRACT.moe_hidden,
        device=residual.device,
        dtype=torch.bfloat16,
    )
    for physical_rank, part in enumerate(pool_parts):
        pooled[
            :, physical_rank * pool_rows : (physical_rank + 1) * pool_rows
        ] = part.to(device=residual.device, dtype=torch.bfloat16)
    expert_input = _rms(
        pooled, RMS_EPS, layer.w2k["xn_w"]
    )

    unpool = layer.w3k["w_unpool"].clone()
    unpool[world * (CONTRACT.global_hidden // CONTRACT.logical_tp) :].zero_()
    k3_case = SimpleNamespace(
        xin=expert_input,
        ids=ids,
        scores=scores,
        stream=stream_after_k2,
        weights={**layer.w3k, "w_unpool": unpool},
        dec13=layer.dec13,
        dec2=layer.dec2,
        local_start=rank * CONTRACT.local_experts,
        active_npes=world,
    )
    local_k3 = k3_local_reference(k3_case)
    k3_parts = _gather_rank_order(local_k3, world, peer_group)
    result = k3_reduce_unpool_reference(k3_case, k3_parts)
    output = {
        "residual": result["stream"],
        "ids": ids,
        "scores": scores,
        "expert_input": expert_input,
        "attention_input": result["attention_input"],
        "cache_k": layer.k_cache[layer.slots].clone(),
        "cache_v": layer.v_cache[layer.slots].clone(),
    }
    if next_k1:
        from test_muse_spark_gfx1250_k1 import _reference as k1_reference

        next_q, next_k, next_v, next_gate = k1_reference(
            result["attention_input"],
            layer.w_qkvg1,
            layer.cos_sin,
            layer.positions,
            True,
        )
        output.update(
            next_q=next_q,
            next_k=next_k,
            next_v=next_v,
            next_gate=next_gate,
            next_cache_k=next_k.reshape(
                rows, CONTRACT.local_kv_heads, CONTRACT.head_dim
            ).to(layer.k_cache1.dtype),
            next_cache_v=next_v.reshape(
                rows, CONTRACT.local_kv_heads, CONTRACT.head_dim
            ).to(layer.v_cache1.dtype),
        )
    return output
