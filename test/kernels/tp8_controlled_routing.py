# SPDX-License-Identifier: Apache-2.0
"""Controlled-routing contract v1 shared by the GB300 producer and MI consumer.

This is deliberately a logical-data contract.  Backend-local expert numbers and
packed metadata are derived by each platform from its runtime ownership map and
are therefore recorded separately from the four cross-platform canonical hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "muse-controlled-routing"
VERSION = 1
WIDTHS = (8, 16)
TOP_K = 8
GLOBAL_EXPERTS = 256
MOE_HIDDEN = 4096
ROUTE_EPSILON = 1e-20


class ControlledRoutingError(ValueError):
    """A controlled-routing package or runtime mapping is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlledRoutingError(message)


def canonical_global_ids(width: int, rotation: int = 0) -> tuple[tuple[int, ...], ...]:
    _require(width in WIDTHS, f"width {width}, expected one of {WIDTHS}")
    _require(rotation == 0, "controlled timing/correctness requires rotation=0")
    return tuple(
        tuple(slot * 32 + ((token + 3 * slot + rotation) % 32) for slot in range(TOP_K))
        for token in range(width)
    )


def normalized_scores(width: int) -> tuple[tuple[float, ...], ...]:
    """Deterministic production-semantics score fixture for package producers."""
    _require(width in WIDTHS, f"width {width}, expected one of {WIDTHS}")
    sigmoid = tuple(1.0 / (1.0 + math.exp(-float(TOP_K - slot))) for slot in range(TOP_K))
    total = sum(sigmoid) + ROUTE_EPSILON
    row = tuple(value / total for value in sigmoid)
    return tuple(row for _ in range(width))


def _hash(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return digest.hexdigest()


def _i32_bytes(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def canonical_hashes(
    width: int,
    xin: Sequence[Sequence[float]],
    ids: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
) -> dict[str, str]:
    tokens = tuple(range(width))
    flat_xin = tuple(float(value) for row in xin for value in row)
    flat_ids = tuple(int(value) for row in ids for value in row)
    flat_scores = tuple(float(value) for row in scores for value in row)
    return {
        "token_order_hash": _hash((_i32_bytes(tokens),)),
        "logical_xin_hash": _hash((_i32_bytes((width, MOE_HIDDEN)), _f32_bytes(flat_xin))),
        "global_ids_hash": _hash((_i32_bytes((width, TOP_K)), _i32_bytes(flat_ids))),
        "scores_hash": _hash((_i32_bytes((width, TOP_K)), _f32_bytes(flat_scores))),
    }


@dataclass(frozen=True)
class ControlledRoutingPackage:
    width: int
    xin: tuple[tuple[float, ...], ...]
    ids: tuple[tuple[int, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    hashes: Mapping[str, str]
    producer: str = "gb300"
    rotation: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "producer": self.producer,
            "width": self.width,
            "top_k": TOP_K,
            "global_experts": GLOBAL_EXPERTS,
            "moe_hidden": MOE_HIDDEN,
            "rotation": self.rotation,
            "token_order": list(range(self.width)),
            "logical_xin": self.xin,
            "global_ids": self.ids,
            "scores": self.scores,
            "canonical_hashes": dict(self.hashes),
        }


def validate_package(document: Mapping[str, Any]) -> ControlledRoutingPackage:
    _require(document.get("schema") == SCHEMA, "controlled-routing schema mismatch")
    _require(document.get("version") == VERSION, "controlled-routing version mismatch")
    _require(document.get("producer") == "gb300", "package producer must be gb300")
    width = document.get("width")
    _require(type(width) is int and width in WIDTHS, f"unsupported package width {width!r}")
    _require(document.get("top_k") == TOP_K, "package Top-K mismatch")
    _require(document.get("global_experts") == GLOBAL_EXPERTS, "global expert count mismatch")
    _require(document.get("moe_hidden") == MOE_HIDDEN, "MoE hidden width mismatch")
    _require(document.get("rotation") == 0, "package rotation must be zero")
    _require(document.get("token_order") == list(range(width)), "token order mismatch")

    xin = tuple(tuple(float(value) for value in row) for row in document.get("logical_xin", ()))
    ids = tuple(tuple(int(value) for value in row) for row in document.get("global_ids", ()))
    scores = tuple(tuple(float(value) for value in row) for row in document.get("scores", ()))
    _require(len(xin) == width, "logical xin row count mismatch")
    _require(all(len(row) == MOE_HIDDEN for row in xin), "logical xin shape mismatch")
    _require(all(math.isfinite(value) for row in xin for value in row), "logical xin is non-finite")
    _require(ids == canonical_global_ids(width), "global IDs are not canonical rotation-zero IDs")
    _require(len(scores) == width and all(len(row) == TOP_K for row in scores), "score shape mismatch")
    _require(
        all(math.isfinite(value) and value >= 0 for row in scores for value in row),
        "scores must be finite and nonnegative",
    )
    hashes = canonical_hashes(width, xin, ids, scores)
    _require(document.get("canonical_hashes") == hashes, "canonical hash mismatch")
    return ControlledRoutingPackage(width, xin, ids, scores, hashes)


def load_package(path: Path) -> ControlledRoutingPackage:
    _require(path.is_file(), f"routing package is unavailable: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlledRoutingError(f"cannot read routing package {path}: {exc}") from exc
    _require(isinstance(document, Mapping), "routing package root must be an object")
    return validate_package(document)


def build_package(
    xin: Sequence[Sequence[float]],
    scores: Sequence[Sequence[float]],
    *,
    width: int,
) -> ControlledRoutingPackage:
    """GB300 producer helper; MI tests consume files and never call this as fallback."""
    ids = canonical_global_ids(width)
    provisional = {
        "schema": SCHEMA,
        "version": VERSION,
        "producer": "gb300",
        "width": width,
        "top_k": TOP_K,
        "global_experts": GLOBAL_EXPERTS,
        "moe_hidden": MOE_HIDDEN,
        "rotation": 0,
        "token_order": list(range(width)),
        "logical_xin": xin,
        "global_ids": ids,
        "scores": scores,
    }
    provisional["canonical_hashes"] = canonical_hashes(width, xin, ids, scores)
    return validate_package(provisional)


@dataclass(frozen=True)
class ExpertLocation:
    global_expert: int
    owner: int
    local_expert: int


def validate_runtime_mapping(
    entries: Sequence[Mapping[str, Any]], *, world_size: int
) -> tuple[ExpertLocation, ...]:
    _require(world_size in (4, 8), f"runtime EP world {world_size}, expected 4 or 8")
    _require(len(entries) == GLOBAL_EXPERTS, "runtime mapping must cover 256 experts")
    locations = tuple(
        ExpertLocation(
            int(item["global_expert"]), int(item["owner"]), int(item["local_expert"])
        )
        for item in entries
    )
    _require(
        {item.global_expert for item in locations} == set(range(GLOBAL_EXPERTS)),
        "runtime mapping global IDs are not a bijection",
    )
    for item in locations:
        _require(0 <= item.owner < world_size, f"invalid owner for expert {item.global_expert}")
        _require(item.local_expert >= 0, f"negative local ID for expert {item.global_expert}")
    for owner in range(world_size):
        local = [item.local_expert for item in locations if item.owner == owner]
        _require(len(local) == GLOBAL_EXPERTS // world_size, f"owner {owner} has wrong expert count")
        _require(len(set(local)) == len(local), f"owner {owner} has duplicate local IDs")
    return tuple(sorted(locations, key=lambda item: item.global_expert))


def map_routing(
    package: ControlledRoutingPackage,
    entries: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    packed_metadata: bytes = b"",
) -> dict[str, Any]:
    locations = validate_runtime_mapping(entries, world_size=world_size)
    owners = tuple(tuple(locations[expert].owner for expert in row) for row in package.ids)
    local_ids = tuple(tuple(locations[expert].local_expert for expert in row) for row in package.ids)
    per_rank_picks = tuple(
        sum(owner == rank for row in owners for owner in row) for rank in range(world_size)
    )
    active_experts = tuple(
        len(
            {
                local_ids[token][slot]
                for token in range(package.width)
                for slot in range(TOP_K)
                if owners[token][slot] == rank
            }
        )
        for rank in range(world_size)
    )
    expected_picks = package.width * TOP_K // world_size
    _require(per_rank_picks == (expected_picks,) * world_size, "controlled per-rank picks mismatch")
    expected_active = min(GLOBAL_EXPERTS // world_size, expected_picks)
    _require(active_experts == (expected_active,) * world_size, "controlled active experts mismatch")
    mapping_bytes = b"".join(
        _i32_bytes((item.global_expert, item.owner, item.local_expert)) for item in locations
    )
    return {
        "runtime_mapping_hash": _hash((mapping_bytes,)),
        "local_ids_hash": _hash(
            (_i32_bytes((package.width, TOP_K)), _i32_bytes(tuple(value for row in local_ids for value in row)))
        ),
        "packed_metadata_hash": _hash((packed_metadata,)),
        "per_rank_picks": per_rank_picks,
        "active_experts": active_experts,
        "owners": owners,
        "local_ids": local_ids,
    }
