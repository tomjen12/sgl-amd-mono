from muse_spark_synthetic_target import (
    deterministic_seed,
    replay_scope_policy,
    truncated_target_config,
)


def _target_config():
    return {
        "text_config": {
            "num_hidden_layers": 12,
            "layer_types": [f"attn-{index}" for index in range(12)],
            "mlp_layer_types": [f"mlp-{index}" for index in range(12)],
            "layer_rope_theta": list(range(12)),
        }
    }


def test_truncation_preserves_first_ten_layer_schedule():
    source = _target_config()

    result = truncated_target_config(source)

    assert result["text_config"]["num_hidden_layers"] == 10
    assert result["text_config"]["layer_types"] == [
        f"attn-{index}" for index in range(10)
    ]
    assert len(source["text_config"]["layer_types"]) == 12


def test_real_dflash_taps_require_segmented_replay():
    policy = replay_scope_policy(
        {"dflash_config": {"target_layer_ids": [1, 16, 30, 44, 59]}}
    )

    assert policy["full_cycle_supported"] is False
    assert policy["out_of_range_taps"] == [16, 30, 44, 59]
    assert policy["validation_scopes"] == [
        "draft/layer_0..4",
        "target/layer_0..9",
    ]


def test_parameter_seed_is_stable_and_rank_specific():
    assert deterministic_seed(42, 0, "layer.1.weight") == deterministic_seed(
        42, 0, "layer.1.weight"
    )
    assert deterministic_seed(42, 0, "layer.1.weight") != deterministic_seed(
        42, 1, "layer.1.weight"
    )
