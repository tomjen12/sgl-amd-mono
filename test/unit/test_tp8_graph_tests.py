from pathlib import Path
import sys

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_graph_tests import STAGES, _layer_stage_parts  # noqa: E402


def test_phase_b_width8_stage_order_is_explicit():
    assert STAGES == (
        "k1",
        "kattn_full",
        "kattn_window",
        "k2",
        "k3",
        "layer_v1_full",
        "layer_v1_swa",
        "layer_v2_full",
        "layer_v2_swa",
    )


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("layer_v1_full", ("kattn_full", "k2", "k3")),
        ("layer_v1_swa", ("kattn_window", "k2", "k3")),
        ("layer_v2_full", ("kattn_full", "k2", "k3", "k1")),
        ("layer_v2_swa", ("kattn_window", "k2", "k3", "k1")),
    ],
)
def test_baseline_layer_graph_uses_documented_stage_boundary(stage, expected):
    assert _layer_stage_parts(stage) == expected
