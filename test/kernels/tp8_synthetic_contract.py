# SPDX-License-Identifier: Apache-2.0
"""Independent TP8-shape contract for the four-GPU Muse Spark test harness.

The expected values are deliberately literal.  Do not derive them from the
production gfx1250 config: this checker must reject an accidental TP4 build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """A synthetic run is not the agreed TP8-local/four-active-rank setup."""


@dataclass(frozen=True)
class TP8ShapeContract:
    global_hidden: int = 8192
    global_q_heads: int = 128
    global_kv_heads: int = 16
    head_dim: int = 64
    logical_tp: int = 8
    logical_ep: int = 8
    active_npes: int = 4
    local_q_heads: int = 16
    local_kv_heads: int = 2
    local_q_width: int = 1024
    local_kv_width: int = 128
    local_qkvg_rows: int = 2304
    global_experts: int = 256
    local_experts: int = 32
    top_k: int = 8
    moe_hidden: int = 4096
    intermediate: int = 4096
    q_len: int = 8
    resident_blocks: int = 96
    widths: tuple[int, ...] = (8, 16, 32, 64)


CONTRACT = TP8ShapeContract()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def parse_widths(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            widths = tuple(int(part) for part in value.split(",") if part)
        except ValueError as exc:
            raise ContractError(f"invalid widths {value!r}") from exc
    else:
        widths = tuple(int(width) for width in value)
    _require(bool(widths), "at least one width is required")
    _require(len(set(widths)) == len(widths), f"duplicate widths: {widths}")
    invalid = tuple(width for width in widths if width not in CONTRACT.widths)
    _require(not invalid, f"widths {invalid} are not TP8 verify widths {CONTRACT.widths}")
    _require(
        all(width % CONTRACT.q_len == 0 for width in widths),
        f"every width must be divisible by q_len={CONTRACT.q_len}: {widths}",
    )
    return widths


def virtual_expert_ranges(
    rank: int, active_npes: int = CONTRACT.active_npes
) -> tuple[tuple[int, int], ...]:
    """Logical EP8 router slices produced by one active rank."""
    _require(
        CONTRACT.logical_ep % active_npes == 0,
        f"logical EP {CONTRACT.logical_ep} is not divisible by {active_npes}",
    )
    _require(0 <= rank < active_npes, f"rank {rank} outside active world")
    return tuple(
        (
            (rank + virtual_rank * active_npes) * CONTRACT.local_experts,
            (rank + virtual_rank * active_npes + 1) * CONTRACT.local_experts,
        )
        for virtual_rank in range(CONTRACT.logical_ep // active_npes)
    )


def deterministic_ep8_routing(
    rows: int, experts_used_per_rank: int = 4
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """Fixed Top-8 fixture with one pick in every logical EP8 shard."""
    _require(rows in CONTRACT.widths, f"unsupported routing width {rows}")
    _require(
        1 <= experts_used_per_rank <= CONTRACT.local_experts,
        f"invalid experts_used_per_rank={experts_used_per_rank}",
    )
    ids = tuple(
        tuple(
            logical_rank * CONTRACT.local_experts
            + (token + 3 * logical_rank) % experts_used_per_rank
            for logical_rank in range(CONTRACT.logical_ep)
        )
        for token in range(rows)
    )
    weights = tuple(float(CONTRACT.top_k - slot) for slot in range(CONTRACT.top_k))
    total = sum(weights)
    scores = tuple(tuple(weight / total for weight in weights) for _ in range(rows))
    return ids, scores


def routing_histogram(
    ids: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Counts by logical rank and by global expert for a routing fixture."""
    per_rank = [0] * CONTRACT.logical_ep
    per_expert = [0] * CONTRACT.global_experts
    for row in ids:
        _require(len(row) == CONTRACT.top_k, f"routing row has {len(row)} picks")
        for expert in row:
            _require(
                0 <= expert < CONTRACT.global_experts,
                f"routing expert {expert} outside global table",
            )
            per_rank[expert // CONTRACT.local_experts] += 1
            per_expert[expert] += 1
    return tuple(per_rank), tuple(per_expert)


def validate_static_contract(values: Mapping[str, Any]) -> None:
    """Validate production constants against independent literal expectations."""
    expected = {
        "HIDDEN": CONTRACT.global_hidden,
        "HEAD_DIM": CONTRACT.head_dim,
        "LOCAL_Q_HEADS": CONTRACT.local_q_heads,
        "LOCAL_KV_HEADS": CONTRACT.local_kv_heads,
        "Q_K": CONTRACT.local_q_width,
        "KV_K": CONTRACT.local_kv_width,
        "QKVG_ROWS": CONTRACT.local_qkvg_rows,
        "EXPERTS": CONTRACT.global_experts,
        "LOCAL_EXPERTS": CONTRACT.local_experts,
        "TOP_K": CONTRACT.top_k,
        "MOE_HIDDEN": CONTRACT.moe_hidden,
        "INTER": CONTRACT.intermediate,
        "MAX_TOKENS": max(CONTRACT.widths),
        "BLOCKS": CONTRACT.resident_blocks,
    }
    for name, want in expected.items():
        got = values.get(name)
        _require(got == want, f"{name}={got!r}, expected TP8 contract value {want!r}")


def validate_rank(
    *,
    rank: int,
    active_world_size: int,
    logical_tp: int,
    logical_ep: int,
    local_q_heads: int,
    local_kv_heads: int,
    local_experts: int,
    resident_blocks: int,
    widths: str | Iterable[int],
) -> tuple[int, ...]:
    checked_widths = parse_widths(widths)
    _require(
        active_world_size == CONTRACT.active_npes,
        f"active world size {active_world_size}, expected {CONTRACT.active_npes}",
    )
    _require(0 <= rank < active_world_size, f"rank {rank} outside active world")
    _require(logical_tp == CONTRACT.logical_tp, f"logical TP {logical_tp}, expected 8")
    _require(logical_ep == CONTRACT.logical_ep, f"logical EP {logical_ep}, expected 8")
    _require(
        local_q_heads == CONTRACT.local_q_heads,
        f"local Q heads {local_q_heads}, expected {CONTRACT.local_q_heads}",
    )
    _require(
        local_kv_heads == CONTRACT.local_kv_heads,
        f"local KV heads {local_kv_heads}, expected {CONTRACT.local_kv_heads}",
    )
    _require(
        local_experts == CONTRACT.local_experts,
        f"local experts {local_experts}, expected {CONTRACT.local_experts}",
    )
    _require(
        resident_blocks == CONTRACT.resident_blocks,
        f"resident blocks {resident_blocks}, expected {CONTRACT.resident_blocks}",
    )
    return checked_widths


def tensor_descriptor(tensor: Any) -> dict[str, Any]:
    """The tensor fields that must match between Baseline and Mono fixtures."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
    }


def validate_tensor(
    name: str,
    tensor: Any,
    *,
    shape: Sequence[int],
    dtype: Any | None = None,
    contiguous: bool = True,
) -> dict[str, Any]:
    got_shape = tuple(int(value) for value in tensor.shape)
    want_shape = tuple(int(value) for value in shape)
    _require(got_shape == want_shape, f"{name} shape {got_shape}, expected {want_shape}")
    if dtype is not None:
        _require(tensor.dtype == dtype, f"{name} dtype {tensor.dtype}, expected {dtype}")
    if contiguous:
        _require(tensor.is_contiguous(), f"{name} must be contiguous")
    return tensor_descriptor(tensor)


def rank_manifest(
    *,
    rank: int,
    active_world_size: int,
    widths: Iterable[int],
    production_constants: Mapping[str, Any],
    device: Mapping[str, Any],
    tensors: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    checked_widths = validate_rank(
        rank=rank,
        active_world_size=active_world_size,
        logical_tp=CONTRACT.logical_tp,
        logical_ep=CONTRACT.logical_ep,
        local_q_heads=int(production_constants["LOCAL_Q_HEADS"]),
        local_kv_heads=int(production_constants["LOCAL_KV_HEADS"]),
        local_experts=int(production_constants["LOCAL_EXPERTS"]),
        resident_blocks=int(production_constants["BLOCKS"]),
        widths=widths,
    )
    validate_static_contract(production_constants)
    expert_start = rank * CONTRACT.local_experts
    return {
        "contract": asdict(CONTRACT),
        "rank": rank,
        "active_world_size": active_world_size,
        "logical_tp": CONTRACT.logical_tp,
        "logical_ep": CONTRACT.logical_ep,
        "widths": list(checked_widths),
        "expert_range": [expert_start, expert_start + CONTRACT.local_experts],
        "device": dict(device),
        "production_constants": dict(production_constants),
        "tensors": dict(tensors or {}),
    }


def validate_manifests(manifests: Sequence[Mapping[str, Any]]) -> None:
    _require(
        len(manifests) == CONTRACT.active_npes,
        f"received {len(manifests)} manifests, expected {CONTRACT.active_npes}",
    )
    baseline = manifests[0]
    for rank, manifest in enumerate(manifests):
        _require(manifest["rank"] == rank, f"manifest {rank} reports rank {manifest['rank']}")
        _require(
            manifest["active_world_size"] == CONTRACT.active_npes,
            f"rank {rank} active world mismatch",
        )
        _require(manifest["logical_tp"] == 8 and manifest["logical_ep"] == 8, f"rank {rank} is not logical TP8/EP8")
        _require(manifest["contract"] == baseline["contract"], f"rank {rank} contract differs")
        _require(manifest["widths"] == baseline["widths"], f"rank {rank} widths differ")
        want = [rank * CONTRACT.local_experts, (rank + 1) * CONTRACT.local_experts]
        _require(manifest["expert_range"] == want, f"rank {rank} experts {manifest['expert_range']}, expected {want}")
