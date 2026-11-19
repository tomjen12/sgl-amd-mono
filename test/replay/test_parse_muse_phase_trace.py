import json

import pytest

from parse_muse_phase_trace import summarize
from sgl_amd_mono.muse_spark.gfx1250.phase_trace import ABI_VERSION, MAGIC


def _record(ticks):
    return {
        "magic": MAGIC,
        "abi_version": ABI_VERSION,
        "family": "K2",
        "rank": 0,
        "layer": 1,
        "task": 0,
        "ticks": ticks,
        "clock_hz": 1_000_000.0,
    }


def test_phase_parser_summarizes_valid_trace(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [_record([0, 1, 2, 3, 4, 5, 6])]}))
    result = summarize([path])
    assert result["status"] == "PASS"
    assert result["records"] == 1
    assert result["families"]["K2"]["intervals"]["start->norm_oproj"][
        "mean_us"
    ] == 1.0


def test_phase_parser_fails_on_unordered_trace(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [_record([0, 1, 2, 1, 4, 5, 6])]}))
    with pytest.raises(ValueError, match="unordered"):
        summarize([path])


def test_phase_parser_fails_on_duplicate_records(tmp_path):
    record = _record([0, 1, 2, 3, 4, 5, 6])
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [record, record]}))
    with pytest.raises(ValueError, match="duplicate"):
        summarize([path])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("abi_version", ABI_VERSION + 1, "ABI version"),
        ("ticks", [0, 1], "expected 7 ticks"),
        (
            "phase_ids",
            ["start", "wrong", "a", "b", "c", "d", "e"],
            "phase IDs do not match ABI",
        ),
    ],
)
def test_phase_parser_rejects_contract_mismatch(tmp_path, field, value, message):
    record = _record([0, 1, 2, 3, 4, 5, 6])
    record[field] = value
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [record]}))
    with pytest.raises(ValueError, match=message):
        summarize([path])


def test_phase_parser_fails_closed_on_empty_artifact(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="no phase records"):
        summarize([path])


def test_phase_parser_reconciles_enclosing_kernel_time(tmp_path):
    record = _record([0, 1, 2, 3, 4, 5, 6])
    record["enclosing_us"] = 20.0
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [record]}))
    with pytest.raises(ValueError, match="does not reconcile"):
        summarize([path])


def test_phase_parser_fails_closed_on_missing_coverage(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"records": [_record([0, 1, 2, 3, 4, 5, 6])]}))
    with pytest.raises(ValueError, match="missing phase families"):
        summarize([path], required_families=("K1", "K2"))
    with pytest.raises(ValueError, match="rank coverage"):
        summarize([path], required_ranks=4)
    with pytest.raises(ValueError, match="layer coverage"):
        summarize(
            [path],
            required_families=("K2",),
            required_layers=10,
        )
