# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_numerical_parity import compare_outputs  # noqa: E402


def test_numerical_parity_accepts_matching_declared_outputs():
    value = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    result = compare_outputs({"stream": value}, {"stream": value.clone()})
    assert result["status"] == "passed"
    assert result["checks"][0]["bad_fraction"] == 0.0


def test_numerical_parity_rejects_output_contract_mismatch():
    value = torch.zeros(2, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="output names differ"):
        compare_outputs({"stream": value}, {"attention_input": value})
    with pytest.raises(RuntimeError, match="Mono"):
        compare_outputs(
            {"stream": value},
            {"stream": value.to(torch.float32)},
        )


def test_numerical_parity_rejects_numerical_mismatch():
    reference = torch.ones(2, 8, dtype=torch.bfloat16)
    actual = torch.zeros_like(reference)
    with pytest.raises(RuntimeError, match="bad_fraction"):
        compare_outputs({"stream": reference}, {"stream": actual})
