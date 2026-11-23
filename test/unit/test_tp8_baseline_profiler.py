# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_baseline_dispatch import DispatchContractError, EXPECTED  # noqa: E402
from tp8_baseline_profiler import (  # noqa: E402
    dispatch_evidence_from_profile,
    extract_kernel_names,
    write_validated_dispatch,
)


def _event(name, category="kernel"):
    return {"cat": category, "name": name, "args": {"Kernel Name": name}}


def test_profiler_extracts_only_gpu_launches_in_order():
    trace = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "aten::empty", "args": {}},
            _event("_dense_partial.kd"),
            _event("_qkv_epilogue.kd", "hip_kernel"),
        ]
    }
    assert extract_kernel_names(trace) == [
        "_dense_partial.kd",
        "_qkv_epilogue.kd",
    ]


def test_profile_is_validated_and_writable_only_after_dispatch_passes(tmp_path):
    expected = EXPECTED["k1"]
    evidence = dispatch_evidence_from_profile(
        "k1",
        operator=expected.operator,
        backend=expected.backend,
        profile={
            "traceEvents": [
                _event("_dense_partial.kd"),
                _event("_qkv_epilogue.kd"),
            ]
        },
    )
    path = tmp_path / "baseline_dispatch.json"
    write_validated_dispatch(path, evidence)
    written = json.loads(path.read_text())
    assert written["actual"]["evidence_source"] == "profiler"
    assert written["actual"]["launch_count"] == 2


@pytest.mark.parametrize(
    "trace,match",
    [
        ({"traceEvents": []}, "no GPU kernel"),
        (
            {
                "traceEvents": [
                    _event("_dense_partial.kd"),
                    _event("_qkv_epilogue.kd"),
                    _event("aten_matmul_fallback"),
                ]
            },
            "possible fallback",
        ),
    ],
)
def test_profiler_evidence_fails_closed(trace, match):
    expected = EXPECTED["k1"]
    with pytest.raises(DispatchContractError, match=match):
        dispatch_evidence_from_profile(
            "k1",
            operator=expected.operator,
            backend=expected.backend,
            profile=trace,
        )


def test_unvalidated_or_synthetic_evidence_is_not_written(tmp_path):
    with pytest.raises(DispatchContractError, match="not profiler-derived"):
        write_validated_dispatch(
            tmp_path / "baseline_dispatch.json",
            {"status": "passed", "actual": {}},
        )
