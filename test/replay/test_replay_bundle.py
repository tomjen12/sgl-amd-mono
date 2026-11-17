import json

import pytest

from replay_bundle import REQUIRED_ARTIFACTS, ReplayBundle


def _make_bundle(root, *, complete=True, width=8, draft_tp_size=4):
    for rank in range(4):
        directory = root / "step_000000" / f"rank_{rank:04d}"
        directory.mkdir(parents=True)
        artifacts = []
        for index, (name, kind) in enumerate(sorted(REQUIRED_ARTIFACTS)):
            suffix = "json" if kind == "metadata" else "safetensors"
            path = directory / f"{index}.{suffix}"
            if name == "cycle" and kind == "metadata":
                path.write_text(
                    json.dumps(
                        {
                            "width": width,
                            "batch_size": 1,
                            "target_tp_rank": rank,
                            "draft_tp_size": draft_tp_size,
                            "replay_target_weights": "synthetic_seeded",
                            "synthetic_seed": 42,
                        }
                    )
                )
            elif kind == "metadata":
                path.write_text("{}")
            else:
                path.write_bytes(b"placeholder")
            artifacts.append({"name": name, "kind": kind, "path": path.name})
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "rank": rank,
                    "step": 0,
                    "complete": complete,
                    "artifacts": artifacts,
                }
            )
        )


def test_bundle_validates_four_consistent_ranks(tmp_path):
    _make_bundle(tmp_path)

    bundle = ReplayBundle.load(tmp_path)

    assert bundle.summary()["width"] == 8
    assert bundle.summary()["ranks"] == 4


def test_bundle_rejects_incomplete_capture(tmp_path):
    _make_bundle(tmp_path, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        ReplayBundle.load(tmp_path)


def test_bundle_rejects_non_width8_capture(tmp_path):
    _make_bundle(tmp_path, width=16)

    with pytest.raises(ValueError, match="expected width 8"):
        ReplayBundle.load(tmp_path)


def test_bundle_rejects_non_tp4_capture(tmp_path):
    _make_bundle(tmp_path, draft_tp_size=2)

    with pytest.raises(ValueError, match="expected draft TP4"):
        ReplayBundle.load(tmp_path)


def test_bundle_rejects_activation_values(tmp_path):
    _make_bundle(tmp_path)
    manifest_path = tmp_path / "step_000000" / "rank_0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    value_path = manifest_path.parent / "hidden.safetensors"
    value_path.write_bytes(b"not-needed")
    manifest["artifacts"].append(
        {"name": "hidden", "kind": "tensors", "path": value_path.name}
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="contains tensor values"):
        ReplayBundle.load(tmp_path)
