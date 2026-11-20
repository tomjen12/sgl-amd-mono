#!/usr/bin/env python3
"""Device-only benchmark for the K2 Baseline packed Joint GEMM."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def _launch_gluon(
    x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor
) -> None:
    from aiter.ops.triton.gemm_a16w16 import (
        gemm_a16w16 as gluon_gemm_a16w16,
    )

    result = gluon_gemm_a16w16(
        x,
        weight,
        dtype=torch.float32,
        y=output,
        skip_reduce=True,
        backend="gluon",
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("Gluon Joint GEMM did not use the preallocated output")


def _launch_flydsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: dict,
    *,
    kernarg_preload: bool = False,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
) -> None:
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import (
        gemm_a16w16 as flydsl_gemm_a16w16,
    )

    if config["num_warps"] != 4:
        raise RuntimeError(
            "The exact Gluon-to-FlyDSL wave mapping currently requires four waves"
        )
    result = flydsl_gemm_a16w16(
        x,
        weight,
        dtype=torch.float32,
        y=output,
        tile_m=config["BLOCK_M"],
        tile_n=config["BLOCK_N"],
        tile_k=config["BLOCK_K"],
        m_warp=2,
        n_warp=2,
        num_buffers=config["NUM_BUFFERS"],
        waves_per_eu=config["waves_per_eu"],
        kernarg_preload=kernarg_preload,
        split_k=1,
        sched_strategy=sched_strategy,
        main_loop_unroll=main_loop_unroll,
        variant=config["kernel_type"],
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("FlyDSL Joint GEMM did not use the preallocated output")


def _correctness(output: torch.Tensor, reference: torch.Tensor) -> dict:
    return {
        "max_error": float((output - reference).abs().max().item()),
        "mean_error": float((output - reference).abs().mean().item()),
        "min_row_cosine": float(
            F.cosine_similarity(output, reference, dim=-1).min().item()
        ),
    }


def _device_benchmark(
    launch,
    *,
    dispatches: int,
    warmup: int,
    rounds: int,
) -> list[float]:
    # Compile and allocate before capture. The captured graph contains only GPU
    # dispatches; Python/JIT/wrapper work is excluded from the measured interval.
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
    parser.add_argument("--n", type=int, default=4608)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--dispatches", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=514218945)
    parser.add_argument(
        "--backend",
        choices=("both", "gluon", "flydsl"),
        default="both",
        help="Use one backend for profiler runs to avoid loading both compilers.",
    )
    parser.add_argument(
        "--flydsl-case",
        choices=(
            "all",
            "exact",
            "kernarg-preload",
            "max-ilp",
            "max-memory-clause",
            "unroll2",
            "preload-unroll2",
            "max-ilp-unroll2",
            "max-memory-clause-unroll2",
        ),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    config = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_K": 512,
        "NUM_BUFFERS": 3,
        "num_warps": 4,
        "waves_per_eu": 1,
        "kernel_type": "bandwidth_bound",
    }
    tuned = False
    reference = x.float() @ weight.float().T
    gluon_output = None
    gluon_result = None
    gluon_median = None
    if args.backend in ("both", "gluon"):
        from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

        runtime_config, tuned = get_gemm_config(
            "GEMM-A16W16",
            args.m,
            args.n,
            args.k,
            backend="gluon",
        )
        if runtime_config != config:
            raise RuntimeError(
                f"Baseline Gluon config drifted: {runtime_config} != {config}"
            )
        gluon_output = torch.empty(
            (args.m, args.n), device=device, dtype=torch.float32
        )
        _launch_gluon(x, weight, gluon_output)
        torch.cuda.synchronize()
        gluon_samples = _device_benchmark(
            lambda: _launch_gluon(x, weight, gluon_output),
            dispatches=args.dispatches,
            warmup=args.warmup,
            rounds=args.rounds,
        )
        gluon_median = statistics.median(gluon_samples)
        gluon_result = {
            "samples_us": gluon_samples,
            "median_us": gluon_median,
            "mean_us": statistics.mean(gluon_samples),
            "min_us": min(gluon_samples),
            "max_us": max(gluon_samples),
        }

    flydsl_cases = {
        "exact": {},
        "kernarg-preload": {"kernarg_preload": True},
        "max-ilp": {"sched_strategy": "max-ilp"},
        "max-memory-clause": {"sched_strategy": "max-memory-clause"},
        "unroll2": {"main_loop_unroll": True},
        "preload-unroll2": {
            "kernarg_preload": True,
            "main_loop_unroll": True,
        },
        "max-ilp-unroll2": {
            "sched_strategy": "max-ilp",
            "main_loop_unroll": True,
        },
        "max-memory-clause-unroll2": {
            "sched_strategy": "max-memory-clause",
            "main_loop_unroll": True,
        },
    }
    flydsl_results = {}
    if args.flydsl_case != "all":
        flydsl_cases = {
            args.flydsl_case: flydsl_cases[args.flydsl_case],
        }
    for name, options in (
        flydsl_cases.items()
        if args.backend in ("both", "flydsl")
        else ()
    ):
        flydsl_output = torch.empty(
            (args.m, args.n), device=device, dtype=torch.float32
        )
        _launch_flydsl(x, weight, flydsl_output, config, **options)
        torch.cuda.synchronize()
        samples = _device_benchmark(
            lambda: _launch_flydsl(
                x, weight, flydsl_output, config, **options
            ),
            dispatches=args.dispatches,
            warmup=args.warmup,
            rounds=args.rounds,
        )
        median_us = statistics.median(samples)
        correctness = {
            "vs_reference": _correctness(flydsl_output, reference),
        }
        if gluon_output is not None:
            correctness["vs_gluon"] = _correctness(
                flydsl_output, gluon_output
            )
        flydsl_results[name] = {
            "options": options,
            "correctness": correctness,
            "samples_us": samples,
            "median_us": median_us,
            "mean_us": statistics.mean(samples),
            "min_us": min(samples),
            "max_us": max(samples),
            "over_gluon": (
                median_us / gluon_median
                if gluon_median is not None
                else None
            ),
            "gap_us": (
                median_us - gluon_median
                if gluon_median is not None
                else None
            ),
        }

    result = {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "gluon_config": config,
        "config_tuned": tuned,
        "measurement_contract": {
            "contract": "captured-device-event",
            "python_wrapper_included": False,
            "dispatches_per_graph": args.dispatches,
            "graph_warmup": args.warmup,
            "rounds": args.rounds,
        },
        "correctness": {
            "gluon_vs_reference": (
                _correctness(gluon_output, reference)
                if gluon_output is not None
                else None
            ),
        },
        "gluon": gluon_result,
        "flydsl_cases": flydsl_results,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
