# SPDX-License-Identifier: Apache-2.0
"""Canonical logical fixtures shared by production Baseline and Mono tests."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from tp8_synthetic_contract import CONTRACT, deterministic_ep8_routing


@dataclass
class LogicalK1Case:
    x: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    cos_sin: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    locations: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    seed: int


@dataclass
class LogicalKAttnCase:
    q: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    gate: torch.Tensor
    sequence_lengths: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor
    window: int
    seed: int


@dataclass
class LogicalK3Case:
    xin: torch.Tensor
    ids: torch.Tensor
    scores: torch.Tensor
    stream: torch.Tensor
    weights: dict[str, torch.Tensor]
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    dec13: torch.Tensor
    dec2: torch.Tensor
    local_start: int
    active_npes: int
    seed: int


def kattn_reference(
    case: LogicalKAttnCase,
    *,
    fp8_qp: bool,
) -> torch.Tensor:
    """PyTorch reference for the production FP8-Q/P or Mono BF16-P family."""
    from sgl_amd_mono.muse_spark.gfx1250.config import SOFTMAX_SCALE

    rows = case.q.shape[0]
    requests = rows // CONTRACT.q_len
    output = torch.zeros_like(case.q, dtype=torch.float32)
    page = case.cache_k.shape[1]
    for request in range(requests):
        keys = int(case.sequence_lengths[request].item())
        context = keys - CONTRACT.q_len
        slots = (
            case.block_table[
                request, : math.ceil(keys / page)
            ].long()[:, None]
            * page
            + torch.arange(page, device=case.q.device)[None, :]
        ).reshape(-1)[:keys]
        key = case.cache_k.reshape(
            -1, CONTRACT.local_kv_heads, CONTRACT.head_dim
        )[slots].float()
        value = case.cache_v.reshape(
            -1, CONTRACT.local_kv_heads, CONTRACT.head_dim
        )[slots].float()
        key_index = torch.arange(keys, device=case.q.device)
        for kv_head in range(CONTRACT.local_kv_heads):
            for query_position in range(CONTRACT.q_len):
                token = request * CONTRACT.q_len + query_position
                query_heads = case.q[token].reshape(
                    CONTRACT.local_q_heads, CONTRACT.head_dim
                )[kv_head * 8 : (kv_head + 1) * 8].float()
                if fp8_qp:
                    q_scale = (
                        query_heads.abs().amax(-1, keepdim=True).clamp_min(1e-20)
                        / 448.0
                    )
                    query_heads = (
                        (query_heads / q_scale)
                        .to(torch.float8_e4m3fn)
                        .float()
                        * q_scale
                    )
                scores = SOFTMAX_SCALE * (
                    query_heads @ key[:, kv_head].T
                )
                allowed = key_index < context + query_position + 1
                if case.window:
                    allowed &= (
                        context + query_position - key_index
                    ) < case.window
                scores = torch.where(
                    allowed[None, :],
                    scores,
                    torch.full_like(scores, float("-inf")),
                )
                if fp8_qp:
                    # Production full-attention v2 forms 32 split-KV partials.
                    # Each split quantizes its unnormalized exp probabilities
                    # to FP8 before P.V, then the deferred reducer merges the
                    # local maxima/denominators in FP32.
                    tiles = math.ceil(keys / 64)
                    split_count = 32
                    tiles_per_split = math.ceil(tiles / split_count)
                    split_keys = tiles_per_split * 64
                    padded_keys = split_count * split_keys
                    padded_scores = torch.full(
                        (query_heads.shape[0], padded_keys),
                        float("-inf"),
                        device=scores.device,
                    )
                    padded_scores[:, :keys] = scores
                    split_scores = padded_scores.reshape(
                        query_heads.shape[0], split_count, split_keys
                    )
                    maxima_tensor = split_scores.amax(-1)
                    valid = torch.isfinite(maxima_tensor)
                    safe_max = torch.where(
                        valid,
                        maxima_tensor,
                        torch.zeros_like(maxima_tensor),
                    )
                    probability = torch.where(
                        valid[:, :, None],
                        torch.exp(split_scores - safe_max[:, :, None]),
                        torch.zeros_like(split_scores),
                    )
                    denom_tensor = probability.sum(-1)
                    padded_value = torch.zeros(
                        padded_keys,
                        CONTRACT.head_dim,
                        device=value.device,
                    )
                    padded_value[:keys] = value[:, kv_head]
                    split_value = padded_value.reshape(
                        split_count, split_keys, CONTRACT.head_dim
                    )
                    partials_tensor = torch.einsum(
                        "hsk,skd->hsd",
                        probability.to(torch.float8_e4m3fn).float(),
                        split_value,
                    )
                    global_max = maxima_tensor.amax(1)
                    merge_weight = torch.where(
                        denom_tensor > 0,
                        torch.exp(maxima_tensor - global_max[:, None]),
                        torch.zeros_like(denom_tensor),
                    )
                    attended = (
                        partials_tensor * merge_weight[:, :, None]
                    ).sum(1) / (
                        denom_tensor * merge_weight
                    ).sum(1)[:, None]
                else:
                    probability = (
                        torch.softmax(scores, dim=-1)
                        .to(torch.bfloat16)
                        .float()
                    )
                    attended = probability @ value[:, kv_head]
                start = kv_head * 8 * CONTRACT.head_dim
                output[
                    token,
                    start : start + 8 * CONTRACT.head_dim,
                ] = attended.reshape(-1)
    return output.to(torch.bfloat16)


def kattn_post_norm_gate(
    raw: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    values = raw.float().reshape(
        raw.shape[0], CONTRACT.local_q_heads, CONTRACT.head_dim
    )
    normalized = values * torch.rsqrt(
        values.pow(2).mean(-1, keepdim=True) + 1e-5
    )
    return (
        normalized * torch.sigmoid(gate.float().reshape_as(values))
    ).to(torch.bfloat16).reshape(raw.shape[0], CONTRACT.local_q_width)


def build_kattn_case(
    rows: int,
    rank: int,
    window: int,
    device: torch.device,
) -> LogicalKAttnCase:
    if rows not in (8, 16) or rows % CONTRACT.q_len:
        raise ValueError(f"K-attn logical fixture width {rows}")
    from test_muse_spark_gfx1250_layer import MAX_BLOCKS
    from sgl_amd_mono.muse_spark.gfx1250.config import QK_SCALE

    requests = rows // CONTRACT.q_len
    seed = 2000 + 17 * rows + rank + window
    torch.manual_seed(seed)
    q = (
        torch.randn(
            rows, CONTRACT.local_q_width, device=device, dtype=torch.float32
        )
        * QK_SCALE
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
    page = 16
    blocks_per_request = math.ceil(
        int(sequence_lengths.max().item()) / page
    )
    total_blocks = requests * blocks_per_request
    cache = torch.randn(
        2,
        total_blocks,
        page,
        CONTRACT.local_kv_heads,
        CONTRACT.head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    block_table = torch.zeros(
        requests, MAX_BLOCKS, device=device, dtype=torch.int32
    )
    block_ids = torch.arange(
        total_blocks, device=device, dtype=torch.int32
    ).reshape(requests, blocks_per_request)
    block_table[:, :blocks_per_request].copy_(block_ids)
    cu_seqlens_q = torch.arange(
        requests + 1, device=device, dtype=torch.int32
    ) * CONTRACT.q_len
    return LogicalKAttnCase(
        q=q,
        k_new=k_new,
        v_new=v_new,
        gate=gate,
        sequence_lengths=sequence_lengths,
        cache_k=cache[0],
        cache_v=cache[1],
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=torch.ones(1, device=device, dtype=torch.float32),
        v_descale=torch.ones(1, device=device, dtype=torch.float32),
        window=window,
        seed=seed,
    )


def build_k1_case(
    rows: int,
    rank: int,
    device: torch.device,
) -> LogicalK1Case:
    if rows not in (8, 16):
        raise ValueError(f"K1 logical fixture width {rows}, expected 8 or 16")
    seed = 1000 + 17 * rows + rank
    torch.manual_seed(seed)
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
    inverse = 1.0 / (
        500_000.0
        ** (
            torch.arange(
                0,
                CONTRACT.head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / CONTRACT.head_dim
        )
    )
    positions_table = torch.arange(
        16384, device=device, dtype=torch.float32
    )[:, None]
    frequencies = positions_table * inverse[None, :]
    cos_sin = torch.cat(
        [frequencies.cos(), frequencies.sin()], dim=-1
    )
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
    return LogicalK1Case(
        x=x,
        weight=weight,
        positions=positions,
        cos_sin=cos_sin,
        cache_k=cache_k,
        cache_v=cache_v,
        locations=locations,
        k_scale=torch.ones(1, device=device, dtype=torch.float32),
        v_scale=torch.ones(1, device=device, dtype=torch.float32),
        seed=seed,
    )


@dataclass
class LogicalK2Case:
    attention_output: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor
    stream: torch.Tensor
    w_o: torch.Tensor
    weights: dict[str, torch.Tensor]
    missing_router_partials: torch.Tensor
    active_npes: int
    seed: int


def build_k2_case(
    rows: int,
    rank: int,
    active_npes: int,
    device: torch.device,
) -> LogicalK2Case:
    """Build the same post-attention K2 problem for Baseline and W8 Mono."""
    if rows not in (8, 16):
        raise ValueError(f"K2 logical fixture width {rows}, expected 8 or 16")
    if active_npes not in (1, 2, 4) or not 0 <= rank < active_npes:
        raise ValueError(f"K2 active topology rank={rank} world={active_npes}")

    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HEAD_DIM,
        HIDDEN,
        MOE_HIDDEN,
        POST_EPS,
        Q_K,
        RMS_EPS,
    )
    from tp8_layer_contract import deterministic_router_bias

    seed = 100 + rows
    torch.manual_seed(seed)
    bf16 = torch.bfloat16
    outputs = [
        torch.randn(rows, Q_K, device=device).to(bf16)
        for _ in range(CONTRACT.logical_tp)
    ]
    gates = [
        torch.randn(rows, Q_K, device=device).to(bf16)
        for _ in range(CONTRACT.logical_tp)
    ]
    projections = [
        (torch.randn(HIDDEN, Q_K, device=device) * 0.02).to(bf16)
        for _ in range(CONTRACT.logical_tp)
    ]
    stream = torch.randn(rows, HIDDEN, device=device)
    # The direct cross-backend gate checks math/layout parity, not top-k tie
    # sensitivity.  Zero logits plus a deterministic separated bias keep ids
    # stable across different reduction/rounding orders; randomized routing is
    # covered independently by the Mono-vs-PyTorch stress cases.
    router = torch.zeros(HIDDEN, EXPERTS, device=device)
    pooler = (torch.randn(MOE_HIDDEN, HIDDEN, device=device) * 0.02).to(bf16)
    if active_npes < CONTRACT.logical_tp:
        pooler[
            active_npes * (MOE_HIDDEN // CONTRACT.logical_tp) :
        ].zero_()
    weights = {
        "alpha": torch.rand(HIDDEN, device=device) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=device) * 0.5,
        "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=device)).to(bf16),
        "router": router.contiguous(),
        # Keep the top-k boundary separated so backend rounding differences in
        # router logits cannot turn this direct numerical gate into a tie test.
        "bias": torch.tensor(
            deterministic_router_bias(),
            device=device,
            dtype=torch.float32,
        ),
        "w_pool": pooler,
        "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=device)).to(bf16),
    }

    gated_outputs = []
    for output, gate in zip(outputs, gates):
        values = output.float().reshape(rows, -1, HEAD_DIM)
        normalized = values * torch.rsqrt(
            values.pow(2).mean(-1, keepdim=True) + RMS_EPS
        )
        gated_outputs.append(
            (normalized * torch.sigmoid(gate.float().reshape_as(values)))
            .to(bf16)
            .reshape(rows, Q_K)
        )

    reduced = torch.zeros(rows, HIDDEN, device=device)
    for physical_rank in range(active_npes):
        reduced += (
            gated_outputs[physical_rank].float()
            @ projections[physical_rank].float().T
        ).to(bf16).float()
    reduced = reduced.to(bf16)
    normalized_branch = (
        reduced.float()
        * torch.rsqrt(
            reduced.float().pow(2).mean(-1, keepdim=True) + POST_EPS
        )
    ).to(bf16)
    new_stream = (
        weights["alpha"] * stream
        + weights["beta"] * normalized_branch.float()
    )
    router_input = (
        new_stream
        * torch.rsqrt(
            new_stream.pow(2).mean(-1, keepdim=True) + RMS_EPS
        )
        * weights["pre_w"].float()
    )
    part_width = HIDDEN // CONTRACT.logical_tp
    missing_start = active_npes * CONTRACT.local_experts
    missing_router_partials = torch.stack(
        [
            router_input[
                :, part * part_width : (part + 1) * part_width
            ]
            @ weights["router"][
                part * part_width : (part + 1) * part_width,
                missing_start:,
            ]
            for part in range(CONTRACT.logical_tp)
        ]
    )
    return LogicalK2Case(
        attention_output=gated_outputs[rank],
        o=outputs[rank],
        gate=gates[rank],
        stream=stream,
        w_o=projections[rank],
        weights=weights,
        missing_router_partials=missing_router_partials,
        active_npes=active_npes,
        seed=seed,
    )


def build_k3_case(
    rows: int,
    rank: int,
    active_npes: int,
    device: torch.device,
) -> LogicalK3Case:
    """Build one TP8-local K3 problem, independently packable by both paths."""
    if rows not in (8, 16):
        raise ValueError(f"K3 logical fixture width {rows}, expected 8 or 16")
    if active_npes not in (1, 2, 4) or not 0 <= rank < active_npes:
        raise ValueError(f"K3 active topology rank={rank} world={active_npes}")

    import test_muse_spark_k3_gfx1250 as k3_test
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        INTER,
        MOE_HIDDEN,
    )

    seed = 300 + rows
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    bf16 = torch.bfloat16
    local_experts = CONTRACT.local_experts
    local_start = rank * local_experts

    raw_w13, raw_s13, dec13 = k3_test._fp4_weight(
        local_experts, 2 * INTER, MOE_HIDDEN, device, generator
    )
    raw_w2, raw_s2, dec2 = k3_test._fp4_weight(
        local_experts, MOE_HIDDEN, INTER, device, generator
    )
    w13, w2, s13, s2 = k3_test.gfx1250_experts(
        raw_w13, raw_s13, raw_w2, raw_s2, local_start
    )
    xin = (torch.randn(rows, MOE_HIDDEN, device=device) * 0.5).to(bf16)
    fixture_ids, fixture_scores = deterministic_ep8_routing(
        rows, local_experts
    )
    ids = torch.tensor(fixture_ids, dtype=torch.int32, device=device)
    scores = torch.tensor(fixture_scores, dtype=torch.float32, device=device)
    stream = torch.randn(rows, HIDDEN, device=device)
    weights = {
        "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=device)).to(bf16),
        "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=device)).to(bf16),
        "alpha": torch.rand(HIDDEN, device=device) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=device) * 0.5,
        "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=device)).to(bf16),
        "w_unpool": (
            torch.randn(HIDDEN, MOE_HIDDEN, device=device) * 0.02
        ).to(bf16),
    }
    if active_npes < CONTRACT.logical_tp:
        weights["w_unpool"][
            active_npes * (HIDDEN // CONTRACT.logical_tp) :
        ].zero_()
    return LogicalK3Case(
        xin=xin,
        ids=ids,
        scores=scores,
        stream=stream,
        weights=weights,
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        dec13=dec13,
        dec2=dec2,
        local_start=local_start,
        active_npes=active_npes,
        seed=seed,
    )
