import pytest

from report_muse_spark_target_gap import (
    BASELINE_LAYOUT,
    _baseline_layer,
    _fused_iteration,
    _split_iteration,
)


def _event(name, duration=1.0):
    return {"name": name, "dur": duration}


def _baseline_events():
    events = []
    for expected in BASELINE_LAYOUT.values():
        for needle in expected:
            name = needle[0] if isinstance(needle, tuple) else needle
            events.append(_event(f"prefix{name}suffix"))
    return events


def test_baseline_layer_validates_and_buckets_18_launches():
    values = _baseline_layer(_baseline_events(), 1)

    assert values == {"K1": 2.0, "K-attn": 2.0, "K2": 5.0, "K3": 9.0}


def test_baseline_layer_accepts_independent_stream_reordering():
    events = _baseline_events()
    events[2], events[4] = events[4], events[2]

    assert _baseline_layer(events, 1) == {
        "K1": 2.0,
        "K-attn": 2.0,
        "K2": 5.0,
        "K3": 9.0,
    }


def test_baseline_layer_rejects_unknown_kernel():
    events = _baseline_events()
    events[2] = _event("unknown")

    with pytest.raises(ValueError, match="unclassified"):
        _baseline_layer(events, 1)


def test_split_iteration_requires_nine_complete_standalone_chains():
    events = []
    for _ in range(9):
        events.extend(
            [
                _event("muse_spark_gfx1250_k1_s8"),
                _event("muse_spark_gfx1250_kattn_s8"),
                _event("muse_spark_gfx1250_k2_s8"),
                _event("muse_spark_gfx1250_k3_s8"),
            ]
        )

    layers = _split_iteration(events)

    assert len(layers) == 9
    assert layers[0] == {"K1": 1.0, "K-attn": 1.0, "K2": 1.0, "K3": 1.0}


def test_fused_iteration_distinguishes_v2_and_final_v1():
    events = [_event("muse_spark_gfx1250_k1_s8")]
    events.extend(
        _event("muse_spark_gfx1250_layer_s8_n1_l0") for _ in range(8)
    )
    events.append(_event("muse_spark_gfx1250_layer_s8_n0_l1"))

    values = _fused_iteration(events)

    assert values == {
        "K1": [1.0],
        "Layer V2": [1.0] * 8,
        "Layer V1": [1.0],
    }
