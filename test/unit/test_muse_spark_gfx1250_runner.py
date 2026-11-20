from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")

from sgl_amd_mono.muse_spark.gfx1250 import runner


def _precompile_runner(use_layer):
    value = runner.MuseMonoRunner.__new__(runner.MuseMonoRunner)
    value.use_kernels = True
    value.use_layer = use_layer
    value.views = [
        SimpleNamespace(kind="mxfp4", shuffled=False),
        SimpleNamespace(kind="other", shuffled=False),
        SimpleNamespace(kind="mxfp4", shuffled=True),
    ]
    return value


def test_split_precompile_builds_each_standalone_stage(monkeypatch):
    value = _precompile_runner(use_layer=False)
    compiled = []
    monkeypatch.setattr(runner, "compile_only", compiled.append)
    monkeypatch.setattr(value, "k1_kernel", lambda rows: ("k1", rows))
    monkeypatch.setattr(
        value,
        "kattn_kernel",
        lambda rows, q_len, splits: ("kattn", rows, q_len, splits),
    )
    monkeypatch.setattr(value, "k2_kernel", lambda rows: ("k2", rows))
    monkeypatch.setattr(
        value,
        "k3_kernel",
        lambda rows, last, shuffled: ("k3", rows, last, shuffled),
    )

    value.precompile(8)

    assert compiled[:3] == [("k1", 8), ("kattn", 8, 8, 6), ("k2", 8)]
    assert set(compiled[3:]) == {
        ("k3", 8, False, False),
        ("k3", 8, True, True),
    }


def test_fused_precompile_remains_default_shape(monkeypatch):
    value = _precompile_runner(use_layer=True)
    compiled = []
    monkeypatch.setattr(runner, "compile_only", compiled.append)
    monkeypatch.setattr(value, "k1_kernel", lambda rows: ("k1", rows))
    monkeypatch.setattr(
        value,
        "layer_kernel",
        lambda rows, q_len, next_k1, last, shuffled: (
            "layer",
            rows,
            q_len,
            next_k1,
            last,
            shuffled,
        ),
    )

    value.precompile(8)

    assert compiled[0] == ("k1", 8)
    assert set(compiled[1:]) == {
        ("layer", 8, 8, False, False, False),
        ("layer", 8, 8, False, True, True),
    }


def test_joint_k2_opt_in_is_limited_to_width_8(monkeypatch):
    monkeypatch.setattr(runner, "USE_K2_JOINT_RP", True)

    assert runner._use_k2_joint_rp(8)
    assert not runner._use_k2_joint_rp(16)
    assert not runner._use_k2_joint_rp(32)


def test_routing_overrides_must_be_paired_and_exact():
    value = runner.MuseMonoRunner.__new__(runner.MuseMonoRunner)
    value.ids = torch.empty(8, 8, dtype=torch.int32)
    value.scores = torch.empty(8, 8, dtype=torch.float32)
    ids = torch.zeros_like(value.ids)
    scores = torch.zeros_like(value.scores)

    value._validate_routing_overrides(3, 8, ids, scores)
    with pytest.raises(ValueError, match="ids and scores together"):
        value._validate_routing_overrides(3, 8, ids, None)
    with pytest.raises(ValueError, match="contiguous \\(8, 8\\) int32"):
        value._validate_routing_overrides(3, 8, ids.to(torch.int64), scores)
    with pytest.raises(ValueError, match="contiguous \\(8, 8\\) float32"):
        value._validate_routing_overrides(3, 8, ids, scores.to(torch.float64))


@pytest.mark.parametrize(
    ("dtype", "lens64"),
    [(torch.int32, 0), (torch.int64, 1)],
)
def test_seq_lens_abi_uses_original_storage(dtype, lens64):
    seq_lens = torch.tensor([2994, 2047], dtype=dtype)

    pointer, flag = runner._seq_lens_abi(seq_lens, 2)

    assert pointer == seq_lens.data_ptr()
    assert flag == lens64


def test_seq_lens_abi_rejects_unsupported_or_noncontiguous_storage():
    with pytest.raises(RuntimeError, match="seq_lens"):
        runner._seq_lens_abi(torch.ones(2, dtype=torch.float32), 2)
    with pytest.raises(RuntimeError, match="contiguous=False"):
        runner._seq_lens_abi(torch.ones(2, 2, dtype=torch.int32)[:, 0], 2)
    with pytest.raises(RuntimeError, match="seq_lens"):
        runner._seq_lens_abi(torch.ones(1, dtype=torch.int32), 2)
