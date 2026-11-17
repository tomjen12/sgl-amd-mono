import json

import pytest
import torch

from muse_spark_draft_replay import (
    HIDDEN_SIZE,
    LAYERS,
    ReplayInputError,
    component_seed,
    create_parser,
    deterministic_positions,
    deterministic_values,
    validate_replay_inputs,
)


def _write_bundle(root, *, changed_rank=None):
    for rank in range(4):
        directory = root / "step_000000" / f"rank_{rank:04d}"
        directory.mkdir(parents=True)
        cycle = {
            "batch_size": 1,
            "draft_tp_size": 4,
            "synthetic_seed": 300959315,
            "target_tp_rank": rank,
            "width": 8,
        }
        forward = {
            "batch_size": 1,
            "cache": {
                "kv.full": {
                    "active_length": 2056,
                    "page_size": 16,
                    "layers": {
                        str(layer): {
                            "k": {
                                "shape": [2757328, 2, 128],
                                "dtype": "torch.float8_e4m3fn",
                            },
                            "v": {
                                "shape": [2757328, 2, 128],
                                "dtype": "torch.float8_e4m3fn",
                            },
                        }
                        for layer in range(LAYERS)
                    },
                }
            },
            "can_run_decode_cuda_graph": False,
            "forward_mode": "TARGET_VERIFY",
            "seq_lens_sum": 2064,
            "tensors": {
                "input_embeds": {"shape": [8, HIDDEN_SIZE]},
                "positions": {"shape": [8]},
                "seq_lens": {"shape": [1]},
            },
        }
        if changed_rank == rank:
            forward["tensors"]["input_embeds"]["shape"] = [16, HIDDEN_SIZE]
        layers = {}
        for layer in range(LAYERS):
            layers[f"layer_{layer}.input.hidden"] = {
                "shape": [8, HIDDEN_SIZE]
            }
            layers[f"layer_{layer}.output.conv_coefficients"] = {
                "shape": [8, 2, 512]
            }
        artifacts = []
        for name, value in (
            ("cycle", cycle),
            ("draft_forward_input", forward),
            ("draft_layers", layers),
        ):
            path = f"{name}.json"
            (directory / path).write_text(json.dumps(value))
            artifacts.append({"name": name, "kind": "metadata", "path": path})
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "rank": rank,
                    "step": 0,
                    "complete": True,
                    "artifacts": artifacts,
                }
            )
        )


def _write_checkpoint(root, *, with_weights=False):
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "num_hidden_layers": 5,
                "hidden_size": 8192,
                "sliding_window": 2048,
                "dflash_config": {"block_size": 8},
            }
        )
    )
    if with_weights:
        (root / "model.safetensors").write_bytes(b"test placeholder")


def test_validates_exact_four_rank_draft_geometry(tmp_path):
    bundle = tmp_path / "bundle"
    checkpoint = tmp_path / "checkpoint"
    _write_bundle(bundle)
    _write_checkpoint(checkpoint)

    descriptor = validate_replay_inputs(
        bundle, checkpoint, require_weights=False
    )

    assert descriptor.seed == 300959315
    assert descriptor.input_shape == (8, 8192)
    assert descriptor.seq_len == 2048
    assert descriptor.seq_len_bound == 2064
    assert descriptor.cache_rows == 2056
    assert descriptor.conv_shape == (8, 2, 512)


def test_deterministic_rehydration_is_component_and_rank_stable(tmp_path):
    bundle = tmp_path / "bundle"
    checkpoint = tmp_path / "checkpoint"
    _write_bundle(bundle)
    _write_checkpoint(checkpoint)
    descriptor = validate_replay_inputs(
        bundle, checkpoint, require_weights=False
    )

    first = deterministic_values(
        (8, 16),
        seed=descriptor.seed,
        rank=2,
        component="input-embeddings",
        scale=0.3,
    )
    second = deterministic_values(
        (8, 16),
        seed=descriptor.seed,
        rank=2,
        component="input-embeddings",
        scale=0.3,
    )
    other_rank = deterministic_values(
        (8, 16),
        seed=descriptor.seed,
        rank=3,
        component="input-embeddings",
        scale=0.3,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, other_rank)
    assert component_seed(descriptor.seed, 2, "key") != component_seed(
        descriptor.seed, 2, "value"
    )
    assert deterministic_positions(descriptor).tolist() == list(range(8192, 8200))


def test_rejects_cross_rank_descriptor_drift(tmp_path):
    bundle = tmp_path / "bundle"
    checkpoint = tmp_path / "checkpoint"
    _write_bundle(bundle, changed_rank=3)
    _write_checkpoint(checkpoint)

    with pytest.raises(ReplayInputError, match="rank 3 Draft geometry differs"):
        validate_replay_inputs(bundle, checkpoint, require_weights=False)


def test_real_execution_requires_checkpoint_weights(tmp_path):
    bundle = tmp_path / "bundle"
    checkpoint = tmp_path / "checkpoint"
    _write_bundle(bundle)
    _write_checkpoint(checkpoint)

    with pytest.raises(ReplayInputError, match="checkpoint weights are unavailable"):
        validate_replay_inputs(bundle, checkpoint, require_weights=True)


def test_cli_is_eager_baseline_with_explicit_iteration_controls():
    args = create_parser().parse_args(
        [
            "bundle",
            "--warmup",
            "3",
            "--iterations",
            "7",
            "--trace",
            "--record-shapes",
        ]
    )

    assert args.warmup == 3
    assert args.iterations == 7
    assert args.trace is True
    assert args.record_shapes is True
    assert not hasattr(args, "monokernel")
