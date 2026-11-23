# SPDX-License-Identifier: Apache-2.0
"""Small, CPU-testable contract helpers for the active4/logical8 layer test."""

from __future__ import annotations

from typing import Mapping, Sequence

ACTIVE_NPES = 4
LOGICAL_NPES = 8
LOCAL_EXPERTS = 32
TOP_K = 8
WIDTHS = (8, 16)


def deterministic_routing() -> tuple[tuple[int, ...], tuple[float, ...]]:
    """One ordered pick from every logical EP8 shard.

    A zero router makes every un-biased sigmoid score 0.5.  The bias only fixes
    Top-K selection and order, so the normalized scores are exactly one eighth.
    """
    ids = tuple(
        logical_rank * LOCAL_EXPERTS + (3 * logical_rank) % 4
        for logical_rank in range(LOGICAL_NPES)
    )
    return ids, (1.0 / TOP_K,) * TOP_K


def deterministic_router_bias() -> tuple[float, ...]:
    """Bias vector that selects :func:`deterministic_routing` in order."""
    ids, _ = deterministic_routing()
    bias = [-1.0] * (LOGICAL_NPES * LOCAL_EXPERTS)
    for order, expert in enumerate(ids):
        bias[expert] = 0.08 - 0.01 * order
    return tuple(bias)


def active_expert_ranges(rank: int) -> tuple[tuple[int, int], ...]:
    """Logical expert slices whose router logits physical ``rank`` computes."""
    if not 0 <= rank < ACTIVE_NPES:
        raise ValueError(f"rank {rank} outside active world")
    return tuple(
        (
            (rank + virtual * ACTIVE_NPES) * LOCAL_EXPERTS,
            (rank + virtual * ACTIVE_NPES + 1) * LOCAL_EXPERTS,
        )
        for virtual in range(LOGICAL_NPES // ACTIVE_NPES)
    )


def validate_case_result(result: Mapping[str, object]) -> None:
    """Validate the compact result returned by a distributed GPU worker."""
    rank = int(result["rank"])
    if not 0 <= rank < ACTIVE_NPES:
        raise AssertionError(f"invalid rank {rank}")
    cases = result.get("cases")
    if not isinstance(cases, Sequence) or len(cases) != len(WIDTHS) * 2:
        raise AssertionError(f"rank {rank}: incomplete cases {cases!r}")
    for case in cases:
        if not isinstance(case, Mapping):
            raise AssertionError(f"rank {rank}: malformed case {case!r}")
        if case.get("status") != "ok":
            raise AssertionError(
                f"rank {rank} width {case.get('width')} "
                f"next_k1={case.get('next_k1')}: {case.get('status')}"
            )
        guard = case.get("guard")
        if guard != [0, 0, 0]:
            raise AssertionError(
                f"rank {rank} width {case.get('width')}: guard errors {guard}"
            )
