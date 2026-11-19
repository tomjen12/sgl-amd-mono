import json
from pathlib import Path

import pytest

from validation_contract import (
    COSINE_THRESHOLD,
    finalize_summary,
    load_manifest,
    workload_identity,
)


ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "replay_validation" / "manifest.json"


def test_manifest_and_workload_hash_are_stable():
    manifest = load_manifest(MANIFEST)
    assert manifest["cosine_threshold"] == COSINE_THRESHOLD
    first = workload_identity(manifest, "target", "quick", ["k3", "k2"])
    second = workload_identity(manifest, "target", "quick", ["k2", "k3"])
    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize("status", ["FAIL", "MISSING", "SKIP"])
def test_required_non_pass_gate_fails_closed(status):
    summary = finalize_summary(
        {
            "gates": [
                {
                    "id": "required",
                    "required": True,
                    "status": status,
                }
            ]
        }
    )
    assert summary["overall"] == "FAIL"
    assert summary["first_failing_gate"] == "required"


def test_missing_or_weakened_manifest_is_rejected(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    manifest["cosine_threshold"] = 0.99
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly"):
        load_manifest(path)
