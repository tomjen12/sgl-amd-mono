# SPDX-License-Identifier: Apache-2.0
"""Host-side mailbox fixture for logical-W8 kernels on four active ranks.

The production kernels still compile with ``npes=8``.  Active ranks 0..3 push
their real contributions.  Before each launch this module publishes tagged
zero contributions for logical ranks 4..7 into the active rank's own symmetric
buffer.  Shadow destination buffers remain valid write targets but are never
counted as physical peers.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch

from tp8_synthetic_contract import (
    CONTRACT,
    ContractError,
    validate_active_npes,
)


def step_tag(step: int, layer: int) -> int:
    from sgl_amd_mono.muse_spark.gfx1250.config import LAYER_SLOTS

    if step < 0 or layer < 0:
        raise ContractError(f"invalid mailbox step/layer {step}/{layer}")
    # Device tags are int32 and equality remains valid across signed rollover.
    value = (step * LAYER_SLOTS + layer + 1) & 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _pairs(
    buffer: torch.Tensor, region: tuple[int, int], name: str
) -> torch.Tensor:
    offset, nbytes = region
    if buffer.dtype != torch.uint8 or buffer.ndim != 1:
        raise ContractError(
            f"{name}: peer buffer must be flat uint8, got "
            f"{buffer.dtype} {tuple(buffer.shape)}"
        )
    if offset < 0 or nbytes % 8 or offset + nbytes > buffer.numel():
        raise ContractError(
            f"{name}: invalid pair region offset={offset} bytes={nbytes} "
            f"for buffer bytes={buffer.numel()}"
        )
    return buffer[offset : offset + nbytes].view(torch.int32).reshape(-1, 2)


def _publish_zero(
    buffer: torch.Tensor,
    region: tuple[int, int],
    pair_indices: torch.Tensor | slice | Iterable[int],
    tag: int,
    name: str,
) -> int:
    pairs = _pairs(buffer, region, name)
    pairs[pair_indices, 0] = 0
    pairs[pair_indices, 1] = tag
    if isinstance(pair_indices, slice):
        start, stop, stride = pair_indices.indices(pairs.shape[0])
        return len(range(start, stop, stride))
    if isinstance(pair_indices, torch.Tensor):
        return int(pair_indices.numel())
    return len(tuple(pair_indices))


def _publish_values(
    buffer: torch.Tensor,
    region: tuple[int, int],
    pair_indices: torch.Tensor | Iterable[int],
    values: torch.Tensor,
    tag: int,
    name: str,
) -> int:
    pairs = _pairs(buffer, region, name)
    indices = (
        pair_indices.to(device=buffer.device, dtype=torch.int64)
        if isinstance(pair_indices, torch.Tensor)
        else torch.tensor(tuple(pair_indices), device=buffer.device, dtype=torch.int64)
    )
    flat = values.to(device=buffer.device, dtype=torch.float32).contiguous().reshape(-1)
    if indices.numel() != flat.numel():
        raise ContractError(
            f"{name}: {indices.numel()} mailbox slots for {flat.numel()} values"
        )
    pairs[indices, 0] = flat.view(torch.int32)
    pairs[indices, 1] = tag
    return int(indices.numel())


def _rect_indices(
    *,
    outer: int,
    pitch: int,
    start: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if outer < 0 or pitch <= 0 or start < 0 or width < 0 or start + width > pitch:
        raise ContractError(
            f"invalid mailbox rectangle outer={outer} pitch={pitch} "
            f"start={start} width={width}"
        )
    base = torch.arange(outer, dtype=torch.int64, device=device) * pitch + start
    cols = torch.arange(width, dtype=torch.int64, device=device)
    return (base[:, None] + cols[None, :]).reshape(-1)


def seed_missing_rank_mailboxes(
    peer_bytes: torch.Tensor,
    *,
    rows: int,
    step: int,
    layer: int,
    missing_router_partials: torch.Tensor | None = None,
    active_npes: int = CONTRACT.active_npes,
) -> dict[str, int]:
    """Publish the missing ranks' zero inputs for one K2/K3/Layer launch."""
    if rows not in (8, 16):
        raise ContractError(f"shadow protocol supports widths 8/16, got {rows}")

    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
    )
    from sgl_amd_mono.muse_spark.gfx1250.device import ATTN_COLS
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import k3_sym_layout
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import POOL_PARTS

    logical = CONTRACT.logical_tp
    active = validate_active_npes(active_npes)
    layout = k3_sym_layout(logical)
    tag = step_tag(step, layer)
    device = peer_bytes.device
    counts: dict[str, int] = {}
    if os.environ.get("SGLANG_AMD_MONO_RSAG", "0") not in ("", "0") or os.environ.get(
        "SGLANG_AMD_MONO_RSAG3", "0"
    ) not in ("", "0"):
        raise ContractError(
            "active4/logical8 shadow fixture does not support RSAG ared/fred paths"
        )

    # Attention all-reduce: source-major bf16 pairs.
    attn_source_pairs = MAX_TOKENS * ATTN_COLS // 2
    counts["attn"] = _publish_zero(
        peer_bytes,
        layout["attn"],
        slice(active * attn_source_pairs, logical * attn_source_pairs),
        tag,
        "attn",
    )

    # Router logits: [K part][token][global expert], fp32 tagged pairs.
    missing_expert = active * CONTRACT.local_experts
    lgx_idx: list[int] = []
    for part in range(POOL_PARTS):
        part_base = part * MAX_TOKENS * EXPERTS
        for token in range(rows):
            begin = part_base + token * EXPERTS + missing_expert
            lgx_idx.extend(range(begin, begin + EXPERTS - missing_expert))
    if missing_router_partials is None:
        counts["lgx"] = _publish_zero(
            peer_bytes, layout["lgx"], lgx_idx, tag, "lgx"
        )
    else:
        expected = (
            POOL_PARTS,
            rows,
            EXPERTS - missing_expert,
        )
        if tuple(missing_router_partials.shape) != expected:
            raise ContractError(
                f"lgx fixture shape {tuple(missing_router_partials.shape)}, "
                f"expected {expected}"
            )
        counts["lgx"] = _publish_values(
            peer_bytes,
            layout["lgx"],
            lgx_idx,
            missing_router_partials,
            tag,
            "lgx",
        )

    # Sharded pooler: missing ranks own the upper half of the output rows.
    pool_idx = _rect_indices(
        outer=rows,
        pitch=MOE_HIDDEN,
        start=active * (MOE_HIDDEN // logical),
        width=(logical - active) * (MOE_HIDDEN // logical),
        device=device,
    )
    counts["poolr"] = _publish_zero(
        peer_bytes, layout["poolr"], pool_idx, tag, "poolr"
    )

    # Expert all-reduce: source-major bf16 pairs.
    ffn_source_pairs = MAX_TOKENS * MOE_HIDDEN // 2
    counts["ffn"] = _publish_zero(
        peer_bytes,
        layout["ffn"],
        slice(active * ffn_source_pairs, logical * ffn_source_pairs),
        tag,
        "ffn",
    )

    # Sharded unpooler: missing ranks own the upper half of output columns.
    unp_idx = _rect_indices(
        outer=rows,
        pitch=HIDDEN // 2,
        start=active * (HIDDEN // logical) // 2,
        width=(logical - active) * (HIDDEN // logical) // 2,
        device=device,
    )
    counts["unp"] = _publish_zero(
        peer_bytes, layout["unp"], unp_idx, tag, "unp"
    )
    counts["tag"] = tag
    return counts
