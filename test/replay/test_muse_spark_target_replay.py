import json

import pytest

from muse_spark_target_replay import (
    ROUTING_BLOCKER,
    WIDTH,
    ReplayInputError,
    compare_routing,
    create_parser,
    deterministic_seed,
    deterministic_values,
    validate_inputs,
)


def _write_rank(bundle, rank, *, seed=42, prefix=8192):
    directory = bundle / "step_000000" / f"rank_{rank:04d}"
    directory.mkdir(parents=True)
    (directory / "cycle.json").write_text(
        json.dumps(
            {
                "width": WIDTH,
                "batch_size": 1,
                "draft_tp_size": 4,
                "synthetic_seed": seed,
            }
        )
    )
    (directory / "target_verify.json").write_text(
        json.dumps(
            {
                "forward_mode": "TARGET_VERIFY",
                "attention_kv_len_lower_bound": prefix,
                "seq_lens_sum": prefix + 48,
                "tensors": {"input_ids": {"shape": [WIDTH]}},
            }
        )
    )
    layers = {}
    for layer in range(10):
        counts = [0] * 256
        counts[layer] = WIDTH * WIDTH
        layers[f"layer_{layer}.routing"] = {
            "rows": WIDTH,
            "top_k": WIDTH,
            "global_token_counts": counts,
        }
        layers[f"layer_{layer}.input.stream"] = {"shape": [WIDTH, 8192]}
    (directory / "target_layers.json").write_text(json.dumps(layers))


def _make_inputs(tmp_path):
    bundle = tmp_path / "bundle"
    for rank in range(4):
        _write_rank(bundle, rank)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 10,
                    "vocab_size": 202048,
                }
            }
        )
    )
    return bundle, target


def test_deterministic_inputs_and_rank_weight_seeds():
    first = deterministic_values(
        seed=42, rank=0, name="target", count=8, modulus=100
    )

    assert first == deterministic_values(
        seed=42, rank=0, name="target", count=8, modulus=100
    )
    assert first != deterministic_values(
        seed=42, rank=1, name="target", count=8, modulus=100
    )
    assert deterministic_seed(42, 0, "target") != deterministic_seed(
        42, 1, "target"
    )


def test_validate_real_geometry_contract(tmp_path):
    bundle, target = _make_inputs(tmp_path)

    descriptors = validate_inputs(bundle, target)

    assert len(descriptors) == 4
    assert descriptors[0].prefix_len == 8192
    assert descriptors[0].seq_lens_sum == 8240
    assert len(descriptors[3].captured_routing) == 10


def test_validation_rejects_silent_workload_substitution(tmp_path):
    bundle, target = _make_inputs(tmp_path)
    verify = bundle / "step_000000" / "rank_0002" / "target_verify.json"
    value = json.loads(verify.read_text())
    value["tensors"]["input_ids"]["shape"] = [4]
    verify.write_text(json.dumps(value))

    with pytest.raises(ReplayInputError, match="not shape"):
        validate_inputs(bundle, target)


def test_validation_rejects_target_checkpoint(tmp_path):
    bundle, target = _make_inputs(tmp_path)
    (target / "model.safetensors").write_bytes(b"forbidden")

    with pytest.raises(ReplayInputError, match="config-only"):
        validate_inputs(bundle, target)


def test_routing_match_and_explicit_override_blocker():
    expected = []
    observed = {}
    for layer in range(10):
        counts = [0] * 256
        counts[layer] = 64
        expected.append(counts)
        observed[f"layer_{layer}.routing"] = {
            "rows": 8,
            "top_k": 8,
            "global_token_counts": counts[:],
        }

    matched = compare_routing(expected, observed)
    assert matched == {
        "shape_preserved": True,
        "exact_counts": True,
        "mismatched_layers": [],
        "blocker": None,
    }

    observed["layer_4.routing"]["global_token_counts"][4] = 63
    observed["layer_4.routing"]["global_token_counts"][5] = 1
    mismatch = compare_routing(expected, observed)
    assert mismatch["shape_preserved"] is True
    assert mismatch["mismatched_layers"] == [4]
    assert mismatch["blocker"] == ROUTING_BLOCKER


def test_cli_requires_explicit_routing_mismatch_opt_in():
    parser = create_parser()
    defaults = parser.parse_args(["bundle", "target"])
    opted_in = parser.parse_args(
        ["bundle", "target", "--allow-routing-mismatch", "--warmup", "3"]
    )

    assert defaults.allow_routing_mismatch is False
    assert defaults.iterations == 5
    assert opted_in.allow_routing_mismatch is True
    assert opted_in.warmup == 3
