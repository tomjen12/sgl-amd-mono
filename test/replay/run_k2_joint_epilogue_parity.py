#!/usr/bin/env python3
"""Device-only benchmark for the K2 Baseline packed Joint epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.moe.router_pooler_topk import _joint_epilogue
from sgl_amd_mono.muse_spark.gfx1250.joint_epilogue import (
    joint_epilogue as flydsl_joint_epilogue,
)


def _launch_triton(
    partial: torch.Tensor,
    bias: torch.Tensor,
    norm: torch.Tensor,
    scores: torch.Tensor,
    ids: torch.Tensor,
    expert_input: torch.Tensor,
    quant: torch.Tensor,
    quant_scale: torch.Tensor,
    multiplier: float,
    route_scale: float,
    eps: float,
) -> None:
    m = partial.shape[1]
    _joint_epilogue[(2 * m,)](
        partial,
        bias,
        norm,
        scores,
        ids,
        expert_input,
        quant,
        quant_scale,
        expert_input,
        expert_input,
        m,
        partial.shape[0],
        multiplier,
        route_scale,
        eps,
        True,
        False,
        num_warps=4,
    )


def _launch_flydsl(
    partial: torch.Tensor,
    bias: torch.Tensor,
    norm: torch.Tensor,
    scores: torch.Tensor,
    ids: torch.Tensor,
    expert_input: torch.Tensor,
    quant: torch.Tensor,
    quant_scale: torch.Tensor,
    multiplier: float,
    route_scale: float,
    eps: float,
    sched_strategy: str | None,
    expert_scheduling: bool,
    diagnostic_task: str,
) -> None:
    flydsl_joint_epilogue(
        partial,
        bias,
        norm,
        scores,
        ids,
        expert_input,
        quant,
        quant_scale,
        multiplier,
        route_scale,
        eps,
        sched_strategy=sched_strategy,
        expert_scheduling=expert_scheduling,
        diagnostic_task=diagnostic_task,
    )


def _device_benchmark(
    launch,
    *,
    dispatches: int,
    warmup: int,
    rounds: int,
) -> list[float]:
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(dispatches):
            launch()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / dispatches)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--sk", type=int, default=1)
    parser.add_argument("--dispatches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=514218945)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--route-scale", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--triton-only", action="store_true")
    parser.add_argument(
        "--flydsl-sched",
        choices=("default", "max-ilp", "max-memory-clause"),
        default="default",
    )
    parser.add_argument("--no-expert-scheduling", action="store_true")
    parser.add_argument(
        "--flydsl-task",
        choices=("both", "router", "pool"),
        default="both",
        help="Diagnostic FlyDSL launch subset; parity requires both",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.m not in (8, 24) or args.sk != 1:
        raise ValueError("Baseline contract supports M=8/24 and SK=1")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    partial = torch.randn(
        (args.sk, args.m, 4608),
        device=device,
        dtype=torch.float32,
    )
    bias = torch.randn((256,), device=device, dtype=torch.float32)
    norm = torch.randn((4096,), device=device, dtype=torch.bfloat16)
    def outputs():
        return (
            torch.empty((args.m, 8), device=device, dtype=torch.float32),
            torch.empty((args.m, 8), device=device, dtype=torch.int32),
            torch.empty(
                (args.m, 4096), device=device, dtype=torch.bfloat16
            ),
            torch.empty((args.m, 2048), device=device, dtype=torch.uint8),
            torch.empty((args.m, 128), device=device, dtype=torch.uint8),
        )

    triton_outputs = outputs()
    flydsl_outputs = None if args.triton_only else outputs()
    triton_launch = lambda: _launch_triton(
        partial,
        bias,
        norm,
        *triton_outputs,
        args.multiplier,
        args.route_scale,
        args.eps,
    )
    flydsl_launch = (
        None
        if args.triton_only
        else lambda: _launch_flydsl(
            partial,
            bias,
            norm,
            *flydsl_outputs,
            args.multiplier,
            args.route_scale,
            args.eps,
            None if args.flydsl_sched == "default" else args.flydsl_sched,
            not args.no_expert_scheduling,
            args.flydsl_task,
        )
    )
    triton_launch()
    if flydsl_launch is not None:
        flydsl_launch()
    torch.cuda.synchronize()

    logits = partial[:, :, 4096:4352].sum(0)
    logits += partial[:, :, 4352:4608].sum(0)
    all_scores = torch.sigmoid(logits * args.multiplier)
    expected_ids = torch.topk(all_scores + bias, 8, dim=-1).indices
    expected_scores = torch.gather(all_scores, 1, expected_ids)
    expected_scores *= args.route_scale / (
        expected_scores.sum(dim=-1, keepdim=True) + 1e-15
    )
    value = partial[:, :, :4096].sum(0).to(torch.bfloat16).float()
    rstd = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + args.eps)
    expected_expert = (value * rstd * norm.float()).to(torch.bfloat16)
    scores, ids, expert_input, _, _ = triton_outputs
    correctness = {
        "triton_vs_reference": {
            "ids_exact": bool(
                torch.equal(ids, expected_ids.to(torch.int32))
            ),
            "scores_max_error": float(
                (scores - expected_scores).abs().max().item()
            ),
            "expert_max_error": float(
                (expert_input.float() - expected_expert.float())
                .abs()
                .max()
                .item()
            ),
        },
    }
    if flydsl_outputs is not None:
        correctness["flydsl_vs_triton"] = {
            "scores_max_error": float(
                (flydsl_outputs[0] - triton_outputs[0]).abs().max().item()
            ),
            "ids_exact": bool(
                torch.equal(flydsl_outputs[1], triton_outputs[1])
            ),
            "expert_exact": bool(
                torch.equal(flydsl_outputs[2], triton_outputs[2])
            ),
            "expert_max_error": float(
                (
                    flydsl_outputs[2].float()
                    - triton_outputs[2].float()
                )
                .abs()
                .max()
                .item()
            ),
            "expert_mismatch_count": int(
                (flydsl_outputs[2] != triton_outputs[2]).sum().item()
            ),
            "quant_exact": bool(
                torch.equal(flydsl_outputs[3], triton_outputs[3])
            ),
            "quant_scale_exact": bool(
                torch.equal(flydsl_outputs[4], triton_outputs[4])
            ),
            "quant_mismatch_count": int(
                (flydsl_outputs[3] != triton_outputs[3]).sum().item()
            ),
            "quant_scale_mismatch_count": int(
                (flydsl_outputs[4] != triton_outputs[4]).sum().item()
            ),
        }

    triton_samples = _device_benchmark(
        triton_launch,
        dispatches=args.dispatches,
        warmup=args.warmup,
        rounds=args.rounds,
    )
    flydsl_samples = (
        None
        if flydsl_launch is None
        else _device_benchmark(
            flydsl_launch,
            dispatches=args.dispatches,
            warmup=args.warmup,
            rounds=args.rounds,
        )
    )
    triton_median = statistics.median(triton_samples)
    flydsl_median = (
        None if flydsl_samples is None else statistics.median(flydsl_samples)
    )
    result = {
        "shape": {"m": args.m, "sk": args.sk, "n": 4608},
        "config": {
            "grid": 2 * args.m,
            "num_warps": 4,
            "quant": True,
            "fp8": False,
            "flydsl_sched": args.flydsl_sched,
            "flydsl_expert_scheduling": not args.no_expert_scheduling,
            "flydsl_task": args.flydsl_task,
        },
        "measurement_contract": {
            "contract": "captured-device-event",
            "python_wrapper_included": False,
            "dispatches_per_graph": args.dispatches,
            "graph_warmup": args.warmup,
            "rounds": args.rounds,
        },
        "correctness": correctness,
        "triton": {
            "samples_us": triton_samples,
            "median_us": triton_median,
            "mean_us": statistics.mean(triton_samples),
            "min_us": min(triton_samples),
            "max_us": max(triton_samples),
        },
    }
    if flydsl_samples is not None:
        result["flydsl"] = {
            "samples_us": flydsl_samples,
            "median_us": flydsl_median,
            "mean_us": statistics.mean(flydsl_samples),
            "min_us": min(flydsl_samples),
            "max_us": max(flydsl_samples),
        }
        result["flydsl_over_triton"] = flydsl_median / triton_median
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
