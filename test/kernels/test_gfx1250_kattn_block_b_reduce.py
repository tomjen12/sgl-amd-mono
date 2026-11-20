"""Focused correctness and device-only A/B for K-attn Block B."""

from __future__ import annotations

import csv
import importlib
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

TOKENS = 8
HEADS = 32
SPLITS = 8
HEAD_DIM = 64


def _reference(partial, maximum, denominator):
    valid_maximum = torch.where(
        denominator > 0,
        maximum,
        torch.full_like(maximum, float("-inf")),
    )
    global_max = valid_maximum.max(dim=2).values
    global_max = torch.where(
        torch.isfinite(global_max), global_max, torch.zeros_like(global_max)
    )
    weights = torch.where(
        denominator > 0,
        torch.exp2(maximum - global_max[:, :, None]),
        torch.zeros_like(denominator),
    )
    total = (denominator * weights).sum(dim=2)
    inverse = torch.where(total > 0, total.reciprocal(), torch.zeros_like(total))
    return (partial * weights[..., None]).sum(dim=2) * inverse[..., None]


def _metrics(got, reference):
    error = (got.float() - reference.float()).abs()
    got_rows = got.float().reshape(-1, HEAD_DIM)
    reference_rows = reference.float().reshape(-1, HEAD_DIM)
    cosine = torch.nn.functional.cosine_similarity(
        got_rows, reference_rows, dim=-1
    )
    both_zero = (got_rows.norm(dim=-1) == 0) & (
        reference_rows.norm(dim=-1) == 0
    )
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return {
        "max_error": error.max().item(),
        "mean_error": error.mean().item(),
        "min_cosine": cosine.min().item(),
    }


def _device_us(invoke, iterations):
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            invoke()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iterations


def test_block_b_reference_handles_empty_splits():
    partial = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    maximum = torch.tensor(
        [[[1.0, -9.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    denominator = torch.tensor(
        [[[2.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    got = _reference(partial, maximum, denominator)
    weights = torch.tensor([0.25, 1.0])
    expected = (
        partial[0, 0, [0, 2]] * weights[:, None]
    ).sum(dim=0) / (2.0 * 0.25 + 4.0)
    torch.testing.assert_close(got[0, 0], expected)
    torch.testing.assert_close(got[0, 1], torch.zeros(3))


def test_gfx1250_kattn_block_b_reduce():
    from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_block_b_reduce import (
        build_block_b_reduce,
        pipeline_metadata,
    )

    torch.manual_seed(125019)
    partial = torch.randn(
        TOKENS,
        HEADS,
        SPLITS,
        HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    maximum = torch.randn(
        TOKENS, HEADS, SPLITS, dtype=torch.float32, device="cuda"
    ) * 4.0
    denominator = torch.rand_like(maximum) * 20.0
    denominator[:, :, -2:] = 0.0
    denominator[0, 0] = 0.0
    reference = _reference(partial, maximum, denominator).to(torch.bfloat16)
    baseline_out = torch.empty(
        TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    prototype_out = torch.empty_like(baseline_out)

    baseline = importlib.import_module(
        "sglang.kernels.ops.attention.swa_verify_splitkv"
    )
    launch = build_block_b_reduce(SPLITS, TOKENS, HEADS)
    compiled = [None]

    def invoke_baseline():
        baseline._swa_verify_reduce[(TOKENS, HEADS)](
            partial,
            maximum,
            denominator,
            baseline_out,
            None,
            HEADS,
            SPLITS,
            baseline_out.stride(0),
            baseline_out.stride(1),
            num_warps=4,
        )

    def invoke_prototype():
        arguments = (
            partial.data_ptr(),
            maximum.data_ptr(),
            denominator.data_ptr(),
            prototype_out.data_ptr(),
            torch.cuda.current_stream(),
        )
        if compiled[0] is None:
            import flydsl.compiler as flyc

            compiled[0] = flyc.compile(launch, *arguments)
        else:
            compiled[0](*arguments)

    invoke_baseline()
    invoke_prototype()
    torch.cuda.synchronize()
    correctness = {
        "baseline_vs_reference": _metrics(baseline_out, reference),
        "prototype_vs_reference": _metrics(prototype_out, reference),
        "prototype_vs_baseline": _metrics(prototype_out, baseline_out),
    }
    print("KATTN_BLOCK_B_CORRECTNESS=" + json.dumps(correctness), flush=True)
    assert correctness["prototype_vs_reference"]["min_cosine"] > 0.999999
    torch.testing.assert_close(prototype_out, baseline_out, rtol=0, atol=0)

    iterations = int(os.environ.get("MUSE_KATTN_BLOCK_B_ITERS", "0"))
    if not iterations:
        return
    warmup = int(os.environ.get("MUSE_KATTN_BLOCK_B_WARMUP", "100"))
    rounds = int(os.environ.get("MUSE_KATTN_BLOCK_B_ROUNDS", "5"))
    for invoke in (invoke_baseline, invoke_prototype):
        for _ in range(warmup):
            invoke()
    torch.cuda.synchronize()

    records = []
    schedule = (
        ("baseline_swa_verify_reduce", invoke_baseline),
        ("flydsl_block_b_reduce", invoke_prototype),
        ("flydsl_block_b_reduce", invoke_prototype),
        ("baseline_swa_verify_reduce", invoke_baseline),
    )
    for round_index in range(rounds):
        for order, (name, invoke) in enumerate(schedule, 1):
            record = {
                "round": round_index + 1,
                "order": order,
                "kernel": name,
                "iterations": iterations,
                "time_us": _device_us(invoke, iterations),
            }
            records.append(record)
            print("KATTN_BLOCK_B_TIMING=" + json.dumps(record), flush=True)

    output = os.environ.get("MUSE_KATTN_BLOCK_B_OUTPUT")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for name, _ in schedule[:2]:
            samples = [r["time_us"] for r in records if r["kernel"] == name]
            summaries[name] = {
                "samples_us": samples,
                "median_us": statistics.median(samples),
                "mean_us": statistics.mean(samples),
                "min_us": min(samples),
                "max_us": max(samples),
            }
        payload = {
            "schema_version": 1,
            "workload": {
                "tokens": TOKENS,
                "heads": HEADS,
                "splits": SPLITS,
                "head_dim": HEAD_DIM,
                "grid": [TOKENS, HEADS],
            },
            "measurement": {
                "clock": (
                    "torch.cuda.Event device timestamps around one captured "
                    "GPU graph containing iterations_per_sample dispatches"
                ),
                "warmup": warmup,
                "iterations_per_sample": iterations,
                "rounds": rounds,
            },
            "correctness": correctness,
            "prototype": pipeline_metadata(SPLITS),
            "summary": summaries,
            "records": records,
        }
        (directory / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        with (directory / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
