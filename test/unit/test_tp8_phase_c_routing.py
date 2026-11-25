# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from run_tp8_phase_c_routing import (  # noqa: E402
    CASES,
    classify_result,
    run_bounded_subprocess,
    run_case_sequence,
)
from tp8_logical_case import (  # noqa: E402
    k3_local_reference,
    k3_reduce_unpool_reference,
)
from tp8_whole_layer_reference import whole_layer_reference  # noqa: E402


def _toy_case(*, selected: bool):
    generator = torch.Generator().manual_seed(9)
    rows = 2
    expert_width = 32
    intermediate = 32
    hidden = 64
    ids = torch.full((rows, 8), 32, dtype=torch.int32)
    scores = torch.zeros(rows, 8)
    if selected:
        ids[:, 0] = 0
        scores[:, 0] = 1.0
    return SimpleNamespace(
        xin=(torch.randn(rows, expert_width, generator=generator) * 0.25).to(
            torch.bfloat16
        ),
        ids=ids,
        scores=scores,
        stream=torch.randn(rows, hidden, generator=generator),
        weights={
            "pen_w": torch.ones(expert_width, dtype=torch.bfloat16),
            "pfw": torch.ones(hidden, dtype=torch.bfloat16),
            "alpha": torch.full((hidden,), 0.75),
            "beta": torch.full((hidden,), 0.25),
            "npw": torch.ones(hidden, dtype=torch.bfloat16),
            "w_unpool": (
                torch.randn(hidden, expert_width, generator=generator) * 0.02
            ).to(torch.bfloat16),
        },
        dec13=torch.randn(
            1, 2 * intermediate, expert_width, generator=generator
        )
        * 0.05,
        dec2=torch.randn(
            1, expert_width, intermediate, generator=generator
        )
        * 0.05,
        local_start=0,
        active_npes=1,
    )


def test_decoded_weight_reference_has_no_baseline_or_mono_dependency():
    source = inspect.getsource(k3_local_reference) + inspect.getsource(
        k3_reduce_unpool_reference
    )
    assert "prepare_mono" not in source
    assert "prepare_k3" not in source
    assert "baseline." not in source.lower()


def test_whole_layer_reference_composes_math_without_split_outputs():
    source = inspect.getsource(whole_layer_reference)
    assert "prepare_mono" not in source
    assert "prepare_k" not in source
    assert "buffers.o" not in source
    assert "buffers.xin" not in source
    for required in (
        "kattn_reference",
        "k3_local_reference",
        "k3_reduce_unpool_reference",
        "k1_reference",
    ):
        assert required in source


def test_k3_reference_zero_local_pick_matches_boundary_math_exactly():
    case = _toy_case(selected=False)
    partial = k3_local_reference(case)
    assert partial.dtype == torch.bfloat16
    assert torch.count_nonzero(partial) == 0
    output = k3_reduce_unpool_reference(case, (partial,))
    expected_stream = case.weights["alpha"] * case.stream
    expected_attention = (
        expected_stream
        * torch.rsqrt(
            expected_stream.pow(2).mean(-1, keepdim=True) + 1e-5
        )
    ).to(torch.bfloat16)
    torch.testing.assert_close(output["stream"], expected_stream, rtol=0, atol=0)
    torch.testing.assert_close(
        output["attention_input"], expected_attention, rtol=0, atol=0
    )


def test_k3_reference_selected_expert_is_deterministic_and_finite():
    case = _toy_case(selected=True)
    first = k3_local_reference(case)
    second = k3_local_reference(case)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first) > 0
    output = k3_reduce_unpool_reference(case, (first,))
    assert all(torch.isfinite(value).all() for value in output.values())


def test_case_order_places_known_hot_expert_fault_last():
    assert CASES == (
        "zero_local_pick",
        "expert_boundary",
        "score_ties",
        "imbalance",
        "hot_expert",
    )


@pytest.mark.parametrize(
    ("returncode", "output", "status"),
    [
        (0, "passed", "passed"),
        (1, "assertion", "failed"),
        (124, "bounded", "timeout"),
        (-6, "Memory access fault by GPU node-5", "hsa_fault"),
    ],
)
def test_subprocess_result_classification(returncode, output, status):
    assert classify_result(returncode, output) == status


def test_isolated_sequence_continues_normal_failures_but_stops_hsa_fault():
    launched = []

    def fake(case):
        launched.append(case)
        if case == "expert_boundary":
            return 1, "strict mismatch"
        if case == "imbalance":
            return -6, "HSA_STATUS_ERROR_MEMORY_FAULT"
        return 0, "passed"

    results = run_case_sequence(fake)
    assert launched == list(CASES[:4])
    assert [item["status"] for item in results] == [
        "passed",
        "failed",
        "passed",
        "hsa_fault",
    ]


def test_bounded_subprocess_times_out_and_terminates_process_group(tmp_path):
    returncode, output = run_bounded_subprocess(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(10)",
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=0.05,
        terminate_grace_seconds=0.2,
    )
    assert returncode == 124
    assert output == "started\n"
