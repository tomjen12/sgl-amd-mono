import json
from types import SimpleNamespace

import pytest
import torch

import muse_spark_target_replay as replay

from muse_spark_target_replay import (
    ROUTING_BLOCKER,
    WIDTH,
    ReplayInputError,
    compare_routing,
    create_parser,
    deterministic_seed,
    deterministic_values,
    routing_ids_from_counts,
    validate_inputs,
    write_output,
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
                "cache": {
                    "kv.full": {
                        "layers": {"1": {"k": {"shape": [13763008, 4, 64]}}}
                    },
                    "kv.swa": {
                        "layers": {"0": {"k": {"shape": [2752608, 4, 64]}}}
                    },
                },
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
    assert descriptors[0].full_kv_physical_rows == 13763008
    assert descriptors[0].swa_kv_physical_rows == 2752608
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


def test_routing_ids_reproduce_counts_without_row_duplicates():
    counts = [0] * 256
    counts[3] = 8
    counts[7] = 7
    counts[11] = 6
    for expert in range(12, 55):
        counts[expert] = 1

    rows = routing_ids_from_counts(counts)

    assert len(rows) == WIDTH
    assert all(len(row) == WIDTH and len(set(row)) == WIDTH for row in rows)
    observed = [0] * 256
    for row in rows:
        for expert in row:
            observed[expert] += 1
    assert observed == counts


def test_routing_ids_reject_duplicate_expert_within_row():
    counts = [0] * 256
    counts[0] = 9
    counts[1] = 55

    with pytest.raises(ReplayInputError, match="at most once"):
        routing_ids_from_counts(counts)


def test_cli_requires_explicit_routing_mismatch_opt_in():
    parser = create_parser()
    defaults = parser.parse_args(["bundle", "target"])
    opted_in = parser.parse_args(
        [
            "bundle",
            "target",
            "--allow-routing-mismatch",
            "--warmup",
            "3",
            "--target-kernel-mode",
            "split",
        ]
    )

    assert defaults.allow_routing_mismatch is False
    assert defaults.cuda_graph is False
    assert defaults.iterations == 5
    assert defaults.target_kernel_mode == "layer"
    assert opted_in.allow_routing_mismatch is True
    assert opted_in.warmup == 3
    assert opted_in.target_kernel_mode == "split"

    graph = parser.parse_args(["bundle", "target", "--cuda-graph"])
    assert graph.cuda_graph is True


def test_output_writer_creates_parent_directories(tmp_path):
    destination = tmp_path / "nested" / "summary.json"

    write_output(destination, '{"status":"ok"}\n')

    assert destination.read_text() == '{"status":"ok"}\n'


def test_state_observers_synchronize_reused_buffers(monkeypatch):
    class Stream:
        calls = 0

        def synchronize(self):
            self.calls += 1

    stream = Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    layer = SimpleNamespace(
        forward=lambda *_args, **_kwargs: (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )
    )
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
    )
    baseline_states = {}
    handles = replay._install_baseline_state_observers(model, baseline_states)
    layer.forward()
    handles[0].remove()

    runner = SimpleNamespace(
        ids=torch.tensor([[7]], dtype=torch.int32),
        launch=lambda **_kwargs: (torch.tensor([3.0]), torch.tensor([4.0])),
    )
    monkeypatch.setattr(
        replay,
        "_installed_mono",
        lambda _model: SimpleNamespace(runner=runner),
    )
    mono_states = {}
    handles = replay._install_mono_state_observer(model, mono_states)
    runner.launch(i=0, rows=1)
    handles[0].remove()

    assert stream.calls == 2
    assert baseline_states[0]["hidden"].item() == 1.0
    assert mono_states[0]["routing_ids"].tolist() == [[7]]


def test_graph_eager_comparison_checks_multiple_replays_and_kv():
    output = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    graph_kv = {
        layer: {
            "key": torch.full((1, 2), layer, dtype=torch.int8),
            "value": torch.full((1, 2), -layer, dtype=torch.int8),
        }
        for layer in range(10)
    }
    eager_kv = {
        layer: {name: value.clone() for name, value in state.items()}
        for layer, state in graph_kv.items()
    }

    first = replay._compare_replay_snapshots(
        output, graph_kv, output.clone(), eager_kv
    )
    second = replay._compare_replay_snapshots(
        output, graph_kv, output.clone(), eager_kv
    )

    assert first == second
    assert first["minimum_output_row_cosine"] == pytest.approx(1.0)
    assert first["kv_exact"] is True
    eager_kv[4]["key"][0, 0] += 1
    mismatch = replay._compare_replay_snapshots(
        output, graph_kv, output.clone(), eager_kv
    )
    assert mismatch["kv_exact"] is False
    assert mismatch["mismatched_kv"] == ["layer_4.key"]


def test_mono_graph_state_restores_across_multiple_iterations():
    class Peer:
        def __init__(self, value):
            self.bytes = torch.tensor([value], dtype=torch.uint8)

    runner = SimpleNamespace(
        epoch=torch.tensor([7], dtype=torch.int32),
        scratch_k1=torch.tensor([1], dtype=torch.uint8),
        scratch_kattn=torch.tensor([2], dtype=torch.uint8),
        scratch_k2=torch.tensor([3], dtype=torch.uint8),
        scratch_k3=torch.tensor([4], dtype=torch.uint8),
        scratch_layer=torch.tensor([5], dtype=torch.uint8),
        peers=Peer(6),
        peers_k2=Peer(7),
        peers_k3=Peer(8),
        _k1_done=9,
    )
    pristine = replay._snapshot_mono_replay_state(runner)

    for _ in range(3):
        runner.epoch.add_(1)
        runner.scratch_kattn.add_(10)
        runner.peers_k2.bytes.add_(10)
        replay._restore_mono_replay_state(runner, pristine)
        assert runner.epoch.tolist() == [7]
        assert runner.scratch_kattn.tolist() == [2]
        assert runner.peers_k2.bytes.tolist() == [7]
        assert runner._k1_done == -1
