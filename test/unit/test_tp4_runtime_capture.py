# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp4_runtime_capture import (  # noqa: E402
    RuntimeCaptureError,
    StructuralCallSpy,
    build_runtime_dispatch,
    inspect_call_spy,
    inspect_replay_bundle,
    validate_runtime_dispatch,
    write_runtime_dispatch,
)


def _events():
    events = []
    for layer, variant in ((0, "full"), (1, "swa")):
        for stage in ("k1", "kattn", "k2", "k3"):
            critical = {"layer": layer}
            if stage == "kattn":
                critical.update(
                    {
                        "attention_variant": variant,
                        "kv_backend": "aiter",
                        "kv_cache_dtype": "fp8_e4m3",
                        "page_size": 16,
                    }
                )
            events.append(
                {
                    "ordinal": len(events),
                    "stage": stage,
                    "operator": f"production.{stage}",
                    "critical_args": critical,
                }
            )
    return events


def _replay_bundle(root: Path):
    for rank in range(4):
        rank_dir = root / f"rank_{rank:04d}"
        rank_dir.mkdir()
        (rank_dir / "manifest.json").write_text(
            json.dumps({"complete": True, "metadata": {"execution": "eager"}})
        )
        descriptor = {"dtype": "torch.float8_e4m3fn"}
        (rank_dir / "target_verify.json").write_text(
            json.dumps(
                {
                    "cache": {
                        "kv.full": {
                            "page_size": 16,
                            "layout": "nhd",
                            "layers": {"1": {"k": descriptor, "v": descriptor}},
                        },
                        "kv.swa": {
                            "page_size": 16,
                            "layout": "nhd",
                            "layers": {"0": {"k": descriptor, "v": descriptor}},
                        },
                    }
                }
            )
        )
        (rank_dir / "target_layers.json").write_text(
            json.dumps(
                {
                    "layer_0.routing": {
                        "local_expert_start": rank * 64,
                        "local_experts": 64,
                        "num_experts": 256,
                        "top_k": 8,
                    }
                }
            )
        )


def _call_spy_bundle(root: Path):
    for rank in range(4):
        rows = [
            {
                "pid": 100 + rank,
                "ordinal": 0,
                "stage": "kattn",
                "operator": "draft.attention",
                "critical_args": {"layer": 0},
            }
        ]
        for layer in range(62):
            variant = "full" if layer % 4 == 1 else "swa"
            for stage in ("k1", "kattn", "k2", "k3"):
                critical = {"layer": layer}
                operator = f"production.{stage}"
                if stage == "k1":
                    critical["fused_kv_write"] = True
                elif stage == "kattn":
                    critical.update(
                        {
                            "attention_variant": variant,
                            "kv_backend": "AiterAttnBackend",
                            "kv_cache_dtype": "fp8_e4m3",
                            "page_size": 16,
                        }
                    )
                elif stage in ("k2", "k3"):
                    critical["local_expert_start"] = rank * 64
                    critical["local_experts"] = 64
                if stage == "k3" and layer not in (0, 61):
                    operator = "run_gfx1250_a8w4"
                    critical["packed_metadata"] = {
                        name: {
                            "shape": [64, 1, 1],
                            "dtype": (
                                "torch.float4_e2m1fn_x2"
                                if name in ("w13", "w2")
                                else "torch.uint8"
                            ),
                            "stride": [1, 1, 1],
                            "address": 256,
                            "address_mod_256": 0,
                            "is_shuffled": name in ("w13", "w2"),
                        }
                        for name in ("w13", "w2", "s13", "s2")
                    }
                if stage == "k3" and layer in (0, 61):
                    operator = "production_expert_fallback"
                rows.append(
                    {
                        "pid": 100 + rank,
                        "ordinal": len(rows),
                        "stage": stage,
                        "operator": operator,
                        "critical_args": critical,
                    }
                )
        (root / f"pid_{100 + rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_structural_call_spy_records_sequence_and_critical_args():
    spy = StructuralCallSpy()

    def operator(value, *, layer):
        return value + 1

    wrapped = spy.wrap(
        operator,
        stage="k2",
        operator="production.k2",
        critical_args=lambda args, kwargs: {"layer": kwargs["layer"]},
    )
    assert wrapped(4, layer=7) == 5
    assert spy.events == [
        {
            "ordinal": 0,
            "stage": "k2",
            "operator": "production.k2",
            "critical_args": {"layer": 7},
        }
    ]


def test_complete_runtime_dispatch_passes_and_writes(tmp_path):
    manifest = build_runtime_dispatch(
        events=_events(),
        runtime_observed=True,
        call_spy_observed=True,
        attention_variants=("full", "swa"),
        fallback_observation_complete=True,
        kernel_inventory_observed=True,
        first_k1_observed=True,
        next_k1_observed=True,
    )
    assert manifest["status"] == "passed"
    assert manifest["performance_evidence"] is False
    assert manifest["semantic_stage_inventory"] == {
        "k1": 2,
        "k2": 2,
        "k3": 2,
        "kattn": 2,
    }
    output = tmp_path / "tp4_runtime_dispatch.json"
    write_runtime_dispatch(output, manifest)
    assert output.is_file()


def test_replay_bundle_attests_four_rank_full_swa_kv_and_tp4_mapping(tmp_path):
    _replay_bundle(tmp_path)
    evidence = inspect_replay_bundle(tmp_path)
    assert [rank["rank"] for rank in evidence["ranks"]] == [0, 1, 2, 3]
    assert evidence["attention_variants"] == {
        "full_layers": [1],
        "swa_layers": [0],
    }
    assert len(evidence["tp4_runtime_mapping"]) == 256
    assert len(evidence["tp4_runtime_mapping_hash"]) == 64
    assert evidence["warning"] == "TP4 ownership is not a TP8 ownership manifest"


def test_call_spy_attests_complete_stage_order_boundaries_and_expected_bf16(tmp_path):
    _call_spy_bundle(tmp_path)
    evidence = inspect_call_spy(tmp_path)
    assert len(evidence["events"]) == 4 * 62 * 4
    assert evidence["unexpected_fallbacks"] == []
    assert evidence["attention_variants"] == ["full", "swa"]
    assert evidence["first_k1_observed"] is True
    assert evidence["next_k1_observed"] is True
    assert evidence["packed_metadata_summary"]["observations"] == 240
    assert evidence["packed_metadata_summary"]["all_addresses_256_aligned"] is True
    assert {
        event["operator"]
        for event in evidence["events"]
        if event["stage"] == "k3" and event["critical_args"]["layer"] in (0, 61)
    } == {"tpx_bf16_or_native_expert"}


def test_incomplete_runtime_dispatch_is_blocked_fail_closed():
    manifest = build_runtime_dispatch(
        events=[],
        runtime_observed=True,
        call_spy_observed=False,
        blockers=("production call-spy was not installed",),
    )
    assert manifest["status"] == "blocked"
    assert manifest["runtime_observed"] is True
    validate_runtime_dispatch(manifest)


def test_runtime_dispatch_blocks_baseline_sequence_mismatch():
    events = _events()
    events[1], events[2] = events[2], events[1]
    manifest = build_runtime_dispatch(
        events=events,
        runtime_observed=True,
        call_spy_observed=True,
        attention_variants=("full", "swa"),
        fallback_observation_complete=True,
        kernel_inventory_observed=True,
        first_k1_observed=True,
        next_k1_observed=True,
    )
    assert manifest["status"] == "blocked"
    assert manifest["baseline_sequence_comparison"]["mismatches"]["0"] == [
        "k1",
        "k2",
        "kattn",
        "k3",
    ]


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("performance_evidence", True, "timing"),
        ("monokernel_enabled", True, "Mono-off"),
        ("recipe_topology", {"tp_size": 8, "ep_size": 8}, "topology"),
    ],
)
def test_runtime_dispatch_rejects_false_or_out_of_scope_claims(field, bad, message):
    manifest = build_runtime_dispatch(
        events=[],
        runtime_observed=False,
        call_spy_observed=False,
        blockers=("not run",),
    )
    manifest[field] = bad
    with pytest.raises(RuntimeCaptureError, match=message):
        validate_runtime_dispatch(manifest)


def test_runtime_dispatch_rejects_bad_kv_contract():
    events = _events()
    events[1]["critical_args"]["kv_cache_dtype"] = "bf16"
    with pytest.raises(RuntimeCaptureError, match="KV dtype"):
        build_runtime_dispatch(
            events=events,
            runtime_observed=True,
            call_spy_observed=True,
            attention_variants=("full", "swa"),
            fallback_observation_complete=True,
            kernel_inventory_observed=True,
            first_k1_observed=True,
            next_k1_observed=True,
        )
