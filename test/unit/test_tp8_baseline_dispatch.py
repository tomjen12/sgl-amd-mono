# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_baseline_dispatch import (  # noqa: E402
    DispatchContractError,
    EXPECTED,
    validate_dispatch,
)


def _matching_names(stage: str) -> list[str]:
    # These canonical names are deliberately minimal; measured names may carry
    # compiler-generated prefixes/suffixes but must match the same allowlist.
    return {
        "k1": ["_dense_partial.kd", "_qkv_epilogue.kd"],
        "kattn_full": ["_vv2_partials.kd", "_reduce_norm_gate.kd"],
        "kattn_window": ["_swa_verify_partials.kd", "_reduce_norm_gate.kd"],
        "k2": [
            "_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16.kd",
            "custom_all_reduce",
            "_fused_norm_residual_triton_kernel.kd",
            "kernel_gemm_a16w16_0.kd",
            "_joint_epilogue.kd",
        ],
        "k3": [
            "_local_expert_ids_kernel.kd",
            "moe_route_psum_fused.kd",
            "moe_fused_quant_preshuffle_routeks_fd4096.kd",
            "mxfp4_tdm_fp8_x_K4096_e64_act1_q1r4_mg10.kd",
            "mxfp4_tdm_fp8_x_K4096_e64_mg10.kd",
            "_finalize_rows_kernel.kd",
            "custom_all_reduce",
            "_dense_partial.kd",
            "_unpool_reduce_boundary.kd",
        ],
    }[stage]


@pytest.mark.parametrize("stage", EXPECTED)
def test_expected_dispatch_accepts_canonical_production_kernels(stage):
    expected = EXPECTED[stage]
    evidence = validate_dispatch(
        stage,
        operator=expected.operator,
        backend=expected.backend,
        kernel_names=_matching_names(stage),
    )
    assert evidence["status"] == "passed"
    assert evidence["actual"]["launch_count"] == len(expected.kernels)


@pytest.mark.parametrize("field", ["operator", "backend"])
def test_dispatch_rejects_wrong_operator_or_backend(field):
    kwargs = {
        "operator": EXPECTED["k1"].operator,
        "backend": EXPECTED["k1"].backend,
        "kernel_names": _matching_names("k1"),
    }
    kwargs[field] = "unexpected_fallback"
    with pytest.raises(DispatchContractError, match=field):
        validate_dispatch("k1", **kwargs)


def test_dispatch_rejects_missing_kernel():
    with pytest.raises(DispatchContractError, match="T_QKV_EPILOGUE"):
        validate_dispatch(
            "k1",
            operator=EXPECTED["k1"].operator,
            backend=EXPECTED["k1"].backend,
            kernel_names=["_dense_partial.kd"],
        )


def test_dispatch_rejects_unexpected_kernel():
    with pytest.raises(DispatchContractError, match="possible fallback"):
        validate_dispatch(
            "k1",
            operator=EXPECTED["k1"].operator,
            backend=EXPECTED["k1"].backend,
            kernel_names=[*_matching_names("k1"), "aten_matmul_fallback"],
        )
