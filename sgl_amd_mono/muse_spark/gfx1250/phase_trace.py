"""Versioned phase-trace ABI shared by gfx1250 Muse Spark kernels."""

from __future__ import annotations

from enum import IntEnum

ABI_VERSION = 1
MAGIC = 0x4D535048


class PhaseFamily(IntEnum):
    K1 = 0x01
    K_ATTN = 0x02
    K2 = 0x03
    K3 = 0x04


PHASE_POINTS = {
    "K1": ("start", "projection", "head_gate_rope_kv"),
    "K-attn": (
        "start",
        "split_stats",
        "global_merge",
        "probability",
        "partial_publish",
        "output",
    ),
    "K2": (
        "start",
        "norm_oproj",
        "boundary",
        "router",
        "pool",
        "top8",
        "pooled_norm",
    ),
    "K3": (
        "start",
        "route_quant",
        "gate_up",
        "down",
        "finalize_prepare",
        "finalize_ep",
        "unpool_boundary",
    ),
}


def phase_ids(family: str) -> tuple[str, ...]:
    try:
        return PHASE_POINTS[family]
    except KeyError as error:
        raise ValueError(f"unknown phase family {family!r}") from error
