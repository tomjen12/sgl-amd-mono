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


@dataclass(frozen=True)
class TP8WorkloadContract:
    logical_npes: int = 8
    active_npes: int = 4
    peer_slots: int = 8
    router_slices_per_active_rank: int = 1
    router_experts_per_active_rank: int = 32
    local_experts: int = 32
    pooler_mode: str = "sharded"
    resident_blocks: int = 96


WORKLOAD = TP8WorkloadContract()
SUPPORTED_ACTIVE_NPES = (1, 2, 4)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_active_npes(active_npes: int) -> int:
    active = int(active_npes)
    _require(
        active in SUPPORTED_ACTIVE_NPES,
        f"active npes {active}, expected one of {SUPPORTED_ACTIVE_NPES}",
    )
    return active


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


def local_expert_range(rank: int) -> tuple[int, int]:
    """The one logical EP8 slice an active rank owns in workload-parity mode."""
    _require(0 <= rank < CONTRACT.active_npes, f"rank {rank} outside active world")
    start = rank * CONTRACT.local_experts
    return start, start + CONTRACT.local_experts


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


def ep8_expert_coverage_suite(
    rows: int,
) -> tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]], ...]:
    """Deterministic cases that cover every one of the 32 experts per EP8 rank."""
    _require(rows in (8, 16), f"coverage width {rows}, expected 8 or 16")
    cases = []
    weights = tuple(float(CONTRACT.top_k - slot) for slot in range(CONTRACT.top_k))
    total = sum(weights)
    scores = tuple(tuple(value / total for value in weights) for _ in range(rows))
    for offset in range(0, CONTRACT.local_experts, rows):
        ids = tuple(
            tuple(
                logical_rank * CONTRACT.local_experts
                + (offset + token) % CONTRACT.local_experts
                for logical_rank in range(CONTRACT.logical_ep)
            )
            for token in range(rows)
        )
        cases.append((ids, scores))
    return tuple(cases)


def validate_ep8_expert_coverage(
    cases: Sequence[
        tuple[Sequence[Sequence[int]], Sequence[Sequence[float]]]
    ],
) -> dict[str, Any]:
    covered = [set() for _ in range(CONTRACT.logical_ep)]
    picks = [0] * CONTRACT.logical_ep
    for ids, scores in cases:
        _require(len(ids) == len(scores), "routing ids/scores row count differs")
        for id_row, score_row in zip(ids, scores):
            _require(len(id_row) == CONTRACT.top_k, "routing row is not Top-8")
            _require(len(score_row) == CONTRACT.top_k, "score row is not Top-8")
            _require(abs(sum(score_row) - 1.0) < 1e-6, "routing scores are not normalized")
            for slot, expert in enumerate(id_row):
                logical_rank = expert // CONTRACT.local_experts
                _require(
                    logical_rank == slot,
                    f"slot {slot} expert {expert} does not cover logical rank {slot}",
                )
                covered[logical_rank].add(expert % CONTRACT.local_experts)
                picks[logical_rank] += 1
    expected = set(range(CONTRACT.local_experts))
    for rank, actual in enumerate(covered):
        _require(actual == expected, f"logical rank {rank} covers experts {sorted(actual)}")
    return {
        "logical_rank_pick_counts": picks,
        "experts_covered_per_logical_rank": [len(item) for item in covered],
        "missing_logical_ranks": list(range(CONTRACT.active_npes, CONTRACT.logical_ep)),
        "missing_expert_output": "tagged-zero",
    }


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
    active_world_size = validate_active_npes(active_world_size)
    _require(
        0 <= rank < active_world_size,
        f"rank {rank} outside active world {active_world_size}",
    )
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


def workload_manifest(
    *,
    rank: int,
    kernel_npes: int,
    peer_slots: int,
    router_ranges: Sequence[Sequence[int]],
    pooler_mode: str,
    resident_blocks: int,
    active_npes: int = WORKLOAD.active_npes,
) -> dict[str, Any]:
    """Validate and describe one rank's actual logical-TP8 workload."""
    active_npes = validate_active_npes(active_npes)
    _require(0 <= rank < active_npes, f"rank {rank} outside active world")
    _require(
        kernel_npes == WORKLOAD.logical_npes,
        f"kernel npes {kernel_npes}, expected logical W8",
    )
    _require(
        peer_slots == WORKLOAD.peer_slots,
        f"peer slots {peer_slots}, expected {WORKLOAD.peer_slots}",
    )
    ranges = tuple(tuple(int(v) for v in item) for item in router_ranges)
    _require(
        ranges == (local_expert_range(rank),),
        f"rank {rank} router ranges {ranges}, expected one local slice "
        f"{local_expert_range(rank)}",
    )
    _require(
        pooler_mode == WORKLOAD.pooler_mode,
        f"pooler mode {pooler_mode!r}, expected {WORKLOAD.pooler_mode!r}",
    )
    _require(
        resident_blocks == WORKLOAD.resident_blocks,
        f"resident blocks {resident_blocks}, expected {WORKLOAD.resident_blocks}",
    )
    return {
        "rank": rank,
        "active_npes": active_npes,
        "logical_npes": kernel_npes,
        "peer_slots": peer_slots,
        "router_ranges": [list(item) for item in ranges],
        "router_experts": sum(end - start for start, end in ranges),
        "local_experts": WORKLOAD.local_experts,
        "pooler_mode": pooler_mode,
        "resident_blocks": resident_blocks,
    }


def validate_workload_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    active_npes: int = WORKLOAD.active_npes,
) -> None:
    """Validate all four active ranks' W8 workload attestations."""
    active_npes = validate_active_npes(active_npes)
    _require(
        len(manifests) == active_npes,
        f"received {len(manifests)} workload manifests, expected "
        f"{active_npes}",
    )
    for rank, manifest in enumerate(manifests):
        _require(manifest["rank"] == rank, f"manifest {rank} rank mismatch")
        _require(
            manifest["active_npes"] == active_npes,
            f"rank {rank} active npes mismatch",
        )
        _require(
            manifest["logical_npes"] == WORKLOAD.logical_npes,
            f"rank {rank} is not W8",
        )
        _require(
            manifest["peer_slots"] == WORKLOAD.peer_slots,
            f"rank {rank} peer slot mismatch",
        )
        _require(
            manifest["router_ranges"] == [list(local_expert_range(rank))],
            f"rank {rank} router slice mismatch",
        )
        _require(
            manifest["router_experts"]
            == WORKLOAD.router_experts_per_active_rank,
            f"rank {rank} router workload mismatch",
        )
        _require(
            manifest["local_experts"] == WORKLOAD.local_experts,
            f"rank {rank} local experts mismatch",
        )
        _require(
            manifest["pooler_mode"] == WORKLOAD.pooler_mode,
            f"rank {rank} pooler is not sharded",
        )


def specialization_attestation(stage: str, rows: int) -> dict[str, Any]:
    """Expected build-key and per-rank work for a logical-W8 launch.

    This is host-side evidence.  The measured kernel symbol is checked against
    these fields separately when profiler dispatch is captured.
    """
    _require(rows in (8, 16), f"attestation width {rows}, expected 8 or 16")
    _require(
        stage in ("k2", "k3", "layer_v1", "layer_v2"),
        f"stage {stage!r} has no W8 specialization contract",
    )
    value: dict[str, Any] = {
        "stage": stage,
        "width": rows,
        "kernel_npes": WORKLOAD.logical_npes,
        "active_npes": WORKLOAD.active_npes,
        "peer_slots": WORKLOAD.peer_slots,
        "resident_ctas": WORKLOAD.resident_blocks,
        "expected_symbol_fragments": [f"s{rows}", "w8"],
    }
    if stage in ("k2", "layer_v1", "layer_v2"):
        value.update(
            router_tasks_per_active_rank=1,
            router_experts_per_active_rank=WORKLOAD.router_experts_per_active_rank,
            pooler_mode="sharded",
            pooler_rows_per_active_rank=CONTRACT.moe_hidden // CONTRACT.logical_tp,
        )
        value["expected_symbol_fragments"].append("ps1")
    if stage in ("k3", "layer_v1", "layer_v2"):
        value.update(
            local_experts_per_active_rank=CONTRACT.local_experts,
            unpool_rows_per_active_rank=CONTRACT.global_hidden
            // CONTRACT.logical_tp,
        )
    return value


def validate_kernel_symbol(attestation: Mapping[str, Any], symbol: str) -> None:
    _require(bool(symbol), "kernel symbol is empty")
    missing = [
        fragment
        for fragment in attestation["expected_symbol_fragments"]
        if fragment not in symbol
    ]
    _require(
        not missing,
        f"{attestation['stage']} symbol {symbol!r} misses W8 fragments {missing}",
    )


def runtime_workload_manifest(
    *,
    implementation: str,
    stage: str,
    width: int,
    rank: int,
    active_npes: int,
) -> dict[str, Any]:
    """Comparable runtime workload fields shared by Baseline and Mono."""
    _require(implementation in ("baseline", "mono"), "invalid implementation")
    active_npes = validate_active_npes(active_npes)
    _require(0 <= rank < active_npes, f"rank {rank} outside active world")
    _require(width in (8, 16), f"runtime workload width {width}")
    _require(
        stage in ("k1", "kattn_full", "kattn_window", "k2", "k3"),
        f"unsupported parity stage {stage}",
    )
    return {
        "implementation": implementation,
        "stage": stage,
        "width": width,
        "rank": rank,
        "active_npes": active_npes,
        "logical_tp": CONTRACT.logical_tp,
        "logical_ep": CONTRACT.logical_ep,
        "local_q_heads": CONTRACT.local_q_heads,
        "local_kv_heads": CONTRACT.local_kv_heads,
        "local_q_width": CONTRACT.local_q_width,
        "local_kv_width": CONTRACT.local_kv_width,
        "local_experts": CONTRACT.local_experts,
        "moe_hidden": CONTRACT.moe_hidden,
        "q_len": CONTRACT.q_len,
        "resident_blocks": CONTRACT.resident_blocks,
        "missing_logical_ranks": list(range(active_npes, CONTRACT.logical_tp)),
    }


def validate_runtime_workload_parity(
    baseline: Sequence[Mapping[str, Any]],
    mono: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail if measured Baseline and Mono cases do not share workload fields."""
    ignored = {"implementation"}
    bmap = {
        (item["stage"], item["width"], item["rank"]): item for item in baseline
    }
    mmap = {(item["stage"], item["width"], item["rank"]): item for item in mono}
    _require(bmap.keys() == mmap.keys(), "Baseline/Mono workload cases differ")
    for key in sorted(bmap):
        b = {name: value for name, value in bmap[key].items() if name not in ignored}
        m = {name: value for name, value in mmap[key].items() if name not in ignored}
        _require(b == m, f"Baseline/Mono workload mismatch for {key}: {b} != {m}")
    return {
        "status": "passed",
        "case_count": len(bmap),
        "cases": [
            {"stage": stage, "width": width, "rank": rank}
            for stage, width, rank in sorted(bmap)
        ],
    }


def tensor_descriptor(tensor: Any) -> dict[str, Any]:
    """The tensor fields that must match between Baseline and Mono fixtures."""
    address = int(tensor.data_ptr()) if callable(getattr(tensor, "data_ptr", None)) else 0
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "data_ptr": address or None,
    }


def validate_tensor(
    name: str,
    tensor: Any,
    *,
    shape: Sequence[int],
    dtype: Any | None = None,
    contiguous: bool = True,
    stride: Sequence[int] | None = None,
    alignment_bytes: int | None = None,
) -> dict[str, Any]:
    got_shape = tuple(int(value) for value in tensor.shape)
    want_shape = tuple(int(value) for value in shape)
    _require(got_shape == want_shape, f"{name} shape {got_shape}, expected {want_shape}")
    if dtype is not None:
        _require(tensor.dtype == dtype, f"{name} dtype {tensor.dtype}, expected {dtype}")
    if contiguous:
        _require(tensor.is_contiguous(), f"{name} must be contiguous")
    if stride is not None:
        got_stride = tuple(int(value) for value in tensor.stride())
        want_stride = tuple(int(value) for value in stride)
        _require(
            got_stride == want_stride,
            f"{name} stride {got_stride}, expected {want_stride}",
        )
    descriptor = tensor_descriptor(tensor)
    if alignment_bytes is not None:
        _require(
            alignment_bytes > 0
            and alignment_bytes & (alignment_bytes - 1) == 0,
            f"{name} alignment must be a positive power of two",
        )
        address = descriptor["data_ptr"]
        _require(address is not None, f"{name} does not expose a data pointer")
        _require(
            address % alignment_bytes == 0,
            f"{name} address {address:#x} is not {alignment_bytes}-byte aligned",
        )
        descriptor["required_alignment_bytes"] = alignment_bytes
    return descriptor


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
    contract = asdict(CONTRACT)
    contract["active_npes"] = active_world_size
    return {
        "contract": contract,
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


def validate_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    active_npes: int = CONTRACT.active_npes,
) -> None:
    active_npes = validate_active_npes(active_npes)
    _require(
        len(manifests) == active_npes,
        f"received {len(manifests)} manifests, expected {active_npes}",
    )
    baseline = manifests[0]
    for rank, manifest in enumerate(manifests):
        _require(manifest["rank"] == rank, f"manifest {rank} reports rank {manifest['rank']}")
        _require(
            manifest["active_world_size"] == active_npes,
            f"rank {rank} active world mismatch",
        )
        _require(manifest["logical_tp"] == 8 and manifest["logical_ep"] == 8, f"rank {rank} is not logical TP8/EP8")
        _require(manifest["contract"] == baseline["contract"], f"rank {rank} contract differs")
        _require(manifest["widths"] == baseline["widths"], f"rank {rank} widths differ")
        want = [rank * CONTRACT.local_experts, (rank + 1) * CONTRACT.local_experts]
        _require(manifest["expert_range"] == want, f"rank {rank} experts {manifest['expert_range']}, expected {want}")
