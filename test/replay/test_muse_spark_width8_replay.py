import json

import pytest

from muse_spark_synthetic_target import deterministic_seed, replay_scope_policy
from muse_spark_width8_replay import (
    DEFAULT_CU_MASK,
    ReplayExecutionBlocked,
    build_session_plan,
    create_parser,
    execute_session,
)
from test_replay_bundle import _make_bundle


def _write_metadata(bundle, rank, name, value):
    directory = bundle / "step_000000" / f"rank_{rank:04d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    artifact = next(item for item in manifest["artifacts"] if item["name"] == name)
    (directory / artifact["path"]).write_text(json.dumps(value))


def _make_inputs(tmp_path):
    bundle = tmp_path / "bundle"
    _make_bundle(bundle)
    for rank in range(4):
        _write_metadata(
            bundle,
            rank,
            "draft_forward_input",
            {
                "forward_mode": "TARGET_VERIFY",
                "tensors": {"input_embeds": {"shape": [8, 8192]}},
            },
        )
        _write_metadata(
            bundle,
            rank,
            "target_verify",
            {
                "forward_mode": "TARGET_VERIFY",
                "seq_lens_sum": 8200,
                "tensors": {"input_ids": {"shape": [8]}},
            },
        )
        _write_metadata(
            bundle,
            rank,
            "draft_layers",
            {
                f"layer_{layer}.input.hidden": {"shape": [8, 8192]}
                for layer in range(5)
            },
        )
        target_layers = {}
        for layer in range(10):
            target_layers[f"layer_{layer}.input.stream"] = {"shape": [8, 8192]}
            target_layers[f"layer_{layer}.routing"] = {
                "rows": 8,
                "top_k": 8,
            }
        _write_metadata(bundle, rank, "target_layers", target_layers)

    draft = tmp_path / "draft"
    draft.mkdir()
    draft_config = {
        "architectures": ["DFlash2DraftModel"],
        "num_hidden_layers": 5,
        "dflash_config": {
            "block_size": 8,
            "target_layer_ids": [1, 16, 30, 44, 59],
        },
    }
    (draft / "config.json").write_text(json.dumps(draft_config))

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 10}})
    )
    (target / "replay_policy.json").write_text(
        json.dumps(replay_scope_policy(draft_config))
    )
    return bundle, draft, target


def test_run_cli_defaults_to_baseline():
    args = create_parser().parse_args(["run", "bundle", "target", "traces"])

    assert args.mode == "baseline"
    assert args.warmup == 2
    assert args.iterations == 5
    assert args.plan_only is False


def test_session_plan_has_independent_tp4_ep4_scopes(tmp_path):
    bundle, draft, target = _make_inputs(tmp_path)

    plan = build_session_plan(
        bundle_path=bundle,
        target_checkpoint=target,
        draft_checkpoint=draft,
        trace_dir=tmp_path / "traces",
        warmup=3,
        iterations=7,
    )

    assert (plan.tp_size, plan.ep_size, plan.width) == (4, 4, 8)
    assert plan.cu_mask == DEFAULT_CU_MASK
    assert plan.hardware_profile == "cu96_tp4"
    assert plan.environment["HSA_CU_MASK"] == "0-3:0-95"
    assert plan.environment["SGLANG_ENABLE_AMD_MONOKERNEL"] == "0"
    assert plan.full_cycle is False
    assert [scope.name for scope in plan.scopes] == ["draft", "target"]
    assert plan.scopes[0].layers == tuple(range(5))
    assert plan.scopes[0].load_format == "checkpoint"
    assert plan.scopes[1].layers == tuple(range(10))
    assert plan.scopes[1].load_format == "dummy"
    assert plan.scopes[1].warmup == 3
    assert plan.scopes[1].iterations == 7
    assert len(plan.scopes[1].trace_files) == 4
    assert plan.rank_weight_seeds[2] == deterministic_seed(42, 2, "target")


def test_session_plan_rejects_mono_before_loading(tmp_path):
    with pytest.raises(ValueError, match="Mono replay is not implemented"):
        build_session_plan(
            bundle_path=tmp_path / "missing",
            target_checkpoint=tmp_path / "missing-target",
            trace_dir=tmp_path / "traces",
            mode="mono",
        )


def test_execution_does_not_substitute_a_different_workload(tmp_path):
    bundle, draft, target = _make_inputs(tmp_path)
    plan = build_session_plan(
        bundle_path=bundle,
        target_checkpoint=target,
        draft_checkpoint=draft,
        trace_dir=tmp_path / "traces",
    )

    with pytest.raises(ReplayExecutionBlocked, match="No GPU work was launched"):
        execute_session(plan)


def test_session_plan_rejects_target_weight_shards(tmp_path):
    bundle, draft, target = _make_inputs(tmp_path)
    (target / "model.safetensors").write_bytes(b"not allowed")

    with pytest.raises(ValueError, match="config-only"):
        build_session_plan(
            bundle_path=bundle,
            target_checkpoint=target,
            draft_checkpoint=draft,
            trace_dir=tmp_path / "traces",
        )
