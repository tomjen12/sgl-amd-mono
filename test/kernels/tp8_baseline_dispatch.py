# SPDX-License-Identifier: Apache-2.0
"""Fail-closed production Baseline dispatch contract for TP8 synthetic runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


class DispatchContractError(RuntimeError):
    """The Baseline did not use the recipe's production dispatch family."""


@dataclass(frozen=True)
class KernelExpectation:
    semantic: str
    pattern: str
    count: int = 1
    classification: str = "local_compute"


@dataclass(frozen=True)
class StageExpectation:
    operator: str
    backend: str
    kernels: tuple[KernelExpectation, ...]


EXPECTED: dict[str, StageExpectation] = {
    "k1": StageExpectation(
        operator="qkv_proj_qk_norm_rope_cache",
        backend="sglang_rocm_fused_qkv_cache",
        kernels=(
            KernelExpectation("T_QKV_GEMM", r"_dense_partial(?:\.kd)?"),
            KernelExpectation("T_QKV_EPILOGUE", r"_qkv_epilogue(?:\.kd)?"),
        ),
    ),
    "kattn_full": StageExpectation(
        operator="aiter_target_verify+verify_reduce_norm_gate",
        backend="aiter_unified_attention+sglang_rocm",
        kernels=(
            KernelExpectation("T_FULL_ATTN", r"_vv2_partials(?:\.kd)?"),
            KernelExpectation(
                "T_ATTN_REDUCE_NORM_GATE", r"_reduce_norm_gate(?:\.kd)?"
            ),
        ),
    ),
    "kattn_window": StageExpectation(
        operator="aiter_target_verify+verify_reduce_norm_gate",
        backend="aiter_unified_attention+sglang_rocm",
        kernels=(
            KernelExpectation("T_SWA_ATTN", r"_swa_verify_partials(?:\.kd)?"),
            KernelExpectation(
                "T_ATTN_REDUCE_NORM_GATE", r"_reduce_norm_gate(?:\.kd)?"
            ),
        ),
    ),
    "k2": StageExpectation(
        operator="attention_boundary+route_with_shared_pooler",
        backend="sglang_rocm+aiter",
        kernels=(
            KernelExpectation(
                "BF16_O_GEMM",
                r"_gemm_a16w16_gfx1250_bandwidth_bound_kernel_.*(?:\.kd)?",
            ),
            KernelExpectation(
                "TP_AR",
                r"(?:all_reduce|custom_all_reduce).*",
                classification="collective",
            ),
            KernelExpectation(
                "T_ATTN_BOUNDARY", r"_fused_norm_residual_triton_kernel(?:\.kd)?"
            ),
            KernelExpectation(
                "T_ROUTER_POOLER", r"kernel_gemm_a16w16_0(?:\.kd)?"
            ),
            KernelExpectation(
                "T_ROUTER_POOLER_EPILOGUE", r"_joint_epilogue(?:\.kd)?"
            ),
        ),
    ),
    "k3": StageExpectation(
        operator="run_gfx1250_a8w4+finalize_allreduce+unpool_boundary",
        backend="aiter_a8w4_tdm+sglang_rocm",
        kernels=(
            KernelExpectation("T_LOCAL_EXPERT_IDS", r"_local_expert_ids_kernel(?:\.kd)?"),
            KernelExpectation("T_ROUTE_PSUM", r"moe_route_psum_fused(?:\.kd)?"),
            KernelExpectation(
                "T_ROUTE_QUANT",
                r"moe_fused_quant_preshuffle_routeks_.*(?:\.kd)?",
            ),
            KernelExpectation(
                "T_MOE_GATE_UP", r"mxfp4_tdm_fp8_.*K4096_.*act1_.*(?:\.kd)?"
            ),
            KernelExpectation(
                "T_MOE_DOWN", r"mxfp4_tdm_fp8_.*K4096_(?!.*act1).*(?:\.kd)?"
            ),
            KernelExpectation("T_MOE_FINALIZE", r"_finalize_rows_kernel(?:\.kd)?"),
            KernelExpectation(
                "TP_AR",
                r"(?:all_reduce|custom_all_reduce).*",
                classification="collective",
            ),
            KernelExpectation("T_UNPOOL_GEMM", r"(?:_dense_partial|Cijk_).*(?:\.kd)?"),
            KernelExpectation(
                "T_UNPOOL_BOUNDARY", r"_unpool_reduce_boundary(?:\.kd)?"
            ),
        ),
    ),
}


def expected_manifest() -> dict[str, dict]:
    return {stage: asdict(expectation) for stage, expectation in EXPECTED.items()}


def validate_dispatch(
    stage: str,
    *,
    operator: str,
    backend: str,
    kernel_names: Iterable[str],
    tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    active_npes: int = 4,
) -> dict:
    """Validate one measured stage and return its JSON-ready evidence."""
    if stage not in EXPECTED:
        raise DispatchContractError(f"unknown Baseline stage {stage!r}")
    expected = EXPECTED[stage]
    if operator != expected.operator:
        raise DispatchContractError(
            f"{stage}: operator {operator!r}, expected {expected.operator!r}"
        )
    if backend != expected.backend:
        raise DispatchContractError(
            f"{stage}: backend {backend!r}, expected {expected.backend!r}"
        )
    names = tuple(str(name) for name in kernel_names)
    matches: dict[str, list[str]] = {}
    launches: list[dict[str, Any]] = []
    consumed: set[int] = set()
    next_index = 0
    for item in expected.kernels:
        count = (
            0
            if item.classification == "collective" and active_npes == 1
            else item.count
        )
        indices = [
            index
            for index, name in enumerate(names)
            if re.search(item.pattern, name, flags=re.IGNORECASE)
        ]
        if len(indices) != count:
            raise DispatchContractError(
                f"{stage}/{item.semantic}: matched {len(indices)} kernels, "
                f"expected {count}; actual={names}"
            )
        if indices != list(range(next_index, next_index + count)):
            raise DispatchContractError(
                f"{stage}/{item.semantic}: kernel order {indices}, "
                f"expected {list(range(next_index, next_index + count))}; "
                f"actual={names}"
            )
        matches[item.semantic] = [names[index] for index in indices]
        launches.extend(
            {
                "launch_index": index,
                "kernel_name": names[index],
                "semantic": item.semantic,
                "classification": item.classification,
            }
            for index in indices
        )
        consumed.update(indices)
        next_index += count
    unexpected = [name for index, name in enumerate(names) if index not in consumed]
    if unexpected:
        raise DispatchContractError(
            f"{stage}: unexpected GPU kernels (possible fallback): {unexpected}"
        )
    return {
        "status": "passed",
        "expected": asdict(expected),
        "actual": {
            "operator": operator,
            "backend": backend,
            "kernel_names": list(names),
            "launch_count": len(names),
            "active_npes": active_npes,
            "ordered_launches": launches,
            "semantic_matches": matches,
            "tensor_metadata": {
                str(name): dict(metadata)
                for name, metadata in (tensor_metadata or {}).items()
            },
        },
    }
