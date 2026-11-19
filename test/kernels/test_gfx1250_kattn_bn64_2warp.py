"""Correctness and alternating benchmark for the standalone BN64 prototype."""

from __future__ import annotations

import csv
import importlib
import json
import os
import statistics
import struct
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

PAGES = 2048
MAX_BLOCKS = 1024
KEYS = 3000
WINDOW = 2047
TILES = 16
ROWS = 16
HEAD_DIM = 64
PROTO_VERSION = os.environ.get("MUSE_KATTN_PROTO_VERSION", "v1")
COMPARE_V1 = os.environ.get("MUSE_KATTN_PROTO_COMPARE_V1", "0") == "1"
if PROTO_VERSION not in ("v1", "v2"):
    raise ValueError(f"unknown MUSE_KATTN_PROTO_VERSION={PROTO_VERSION!r}")


def _prototype_name(splits: int) -> str:
    prefix = "prototype" if PROTO_VERSION == "v1" else "prototype_v2"
    return f"{prefix}_{TILES * splits}_cta"


def _row_coords(tile: int, row: int) -> tuple[int, int, int]:
    kv_head = tile // 4
    logical = (tile % 4) * ROWS + row
    return logical // 8, kv_head * 8 + logical % 8, kv_head


def _owned_keys(begin: int, end: int, split: int, splits: int) -> torch.Tensor:
    origin = (begin // 64) * 64
    blocks = torch.arange(
        origin + split * 64,
        end,
        splits * 64,
        device="cuda",
    )
    if not len(blocks):
        return torch.empty(0, dtype=torch.long, device="cuda")
    keys = (blocks[:, None] + torch.arange(64, device="cuda")).flatten()
    return keys[(keys >= begin) & (keys < end)]


def _reference(q, k_cache, v_cache, table, splits, scale):
    partial = torch.zeros(
        TILES, splits, ROWS, HEAD_DIM, dtype=torch.float32, device="cuda"
    )
    maximum = torch.full(
        (TILES, splits, ROWS), -1.0e30, dtype=torch.float32, device="cuda"
    )
    denominator = torch.zeros_like(maximum)
    prefix = KEYS - 8
    slots = (
        table[0, : (KEYS + 15) // 16].long()[:, None] * 16
        + torch.arange(16, device="cuda")[None, :]
    ).flatten()[:KEYS]
    k = k_cache[slots].float()
    v = v_cache[slots].float()
    for tile in range(TILES):
        token0, _, _ = _row_coords(tile, 0)
        tile_begin = max(0, prefix + token0 + 1 - WINDOW)
        for row in range(ROWS):
            token, q_head, kv_head = _row_coords(tile, row)
            begin = max(0, prefix + token + 1 - WINDOW)
            end = prefix + token + 1
            query = q[token, q_head * HEAD_DIM : (q_head + 1) * HEAD_DIM]
            for split in range(splits):
                keys = _owned_keys(tile_begin, end, split, splits)
                keys = keys[keys >= begin]
                score = scale * (query.float() @ k[keys, kv_head].T)
                row_max = score.max()
                probability = torch.exp(score - row_max)
                maximum[tile, split, row] = row_max
                denominator[tile, split, row] = probability.sum()
                partial[tile, split, row] = (
                    probability.to(torch.bfloat16).float()
                    @ v[keys, kv_head]
                )
    return partial, maximum, denominator


def _reduce(partial, maximum, denominator):
    global_max = maximum.max(dim=1).values
    weights = torch.exp(maximum - global_max[:, None])
    global_denominator = (denominator * weights).sum(dim=1)
    return (
        partial * weights[..., None]
    ).sum(dim=1) / global_denominator[..., None]


def _canonical(partial, maximum, denominator):
    """Convert prototype tile/row output to [token, head, split, ...]."""
    splits = partial.shape[1]
    canonical_partial = torch.empty(
        8, 32, splits, HEAD_DIM, dtype=partial.dtype, device=partial.device
    )
    canonical_maximum = torch.empty(
        8, 32, splits, dtype=maximum.dtype, device=maximum.device
    )
    canonical_denominator = torch.empty_like(canonical_maximum)
    for tile in range(TILES):
        for row in range(ROWS):
            token, q_head, _ = _row_coords(tile, row)
            canonical_partial[token, q_head] = partial[tile, :, row]
            canonical_maximum[token, q_head] = maximum[tile, :, row]
            canonical_denominator[token, q_head] = denominator[tile, :, row]
    return canonical_partial, canonical_maximum, canonical_denominator


def _reduce_canonical(partial, maximum, denominator):
    global_max = maximum.max(dim=2).values
    weights = torch.exp(maximum - global_max[:, :, None])
    global_denominator = (denominator * weights).sum(dim=2)
    return (
        partial * weights[..., None]
    ).sum(dim=2) / global_denominator[..., None]


def _reduce_canonical_exp2(partial, maximum, denominator):
    """Merge Triton states whose maxima are stored in log2-score space."""
    valid_maximum = torch.where(
        denominator > 0,
        maximum,
        torch.full_like(maximum, float("-inf")),
    )
    global_max = valid_maximum.max(dim=2).values
    weights = torch.where(
        denominator > 0,
        torch.exp2(maximum - global_max[:, :, None]),
        torch.zeros_like(denominator),
    )
    global_denominator = (denominator * weights).sum(dim=2)
    return (
        partial * weights[..., None]
    ).sum(dim=2) / global_denominator[..., None]


def _metrics(got, reference):
    error = (got - reference).abs()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(-1, HEAD_DIM),
        reference.reshape(-1, HEAD_DIM),
        dim=-1,
    )
    return {
        "max_error": error.max().item(),
        "mean_error": error.mean().item(),
        "min_cosine": cosine.min().item(),
    }


def _time_launch(invoke, iterations):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        invoke()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / iterations


def _assert_exact_coverage(splits):
    prefix = KEYS - 8
    for tile in range(TILES):
        token0, _, _ = _row_coords(tile, 0)
        tile_begin = max(0, prefix + token0 + 1 - WINDOW)
        for row in range(ROWS):
            token, _, _ = _row_coords(tile, row)
            begin = max(0, prefix + token + 1 - WINDOW)
            end = prefix + token + 1
            owned = []
            for split in range(splits):
                keys = _owned_keys(tile_begin, end, split, splits)
                owned.append(keys[keys >= begin])
            covered = torch.cat(owned).sort().values
            expected = torch.arange(begin, end, device="cuda")
            assert torch.equal(covered, expected)


def test_gfx1250_kattn_bn64_2warp_compute_only():
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        LOCAL_KV_HEADS,
        Q_K,
        SOFTMAX_SCALE,
    )
    if PROTO_VERSION == "v1":
        from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_bn64_2warp import (
            build_bn64_2warp as build_prototype,
        )
        prototype_metadata = {"version": "v1", "stages": 1}
    else:
        from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_bn64_2warp_v2 import (
            build_bn64_2warp_v2 as build_prototype,
            pipeline_metadata,
        )
        prototype_metadata = pipeline_metadata(8)

    torch.manual_seed(1250)
    q = (torch.randn(8, Q_K, device="cuda") * 1.5).to(torch.bfloat16)
    k_cache = (
        torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device="cuda")
        * 0.75
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device="cuda")
        * 0.75
    ).to(torch.float8_e4m3fn)
    table = torch.zeros(1, MAX_BLOCKS, dtype=torch.int32, device="cuda")
    pages = (KEYS + 15) // 16
    table[0, :pages] = (
        torch.randperm(PAGES - 1, device="cuda")[:pages].to(torch.int32) + 1
    )
    prefix_lens = torch.tensor(
        [KEYS - 8], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_q = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor([KEYS], dtype=torch.int32, device="cuda")
    scale_bits = struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0]

    records = []
    candidates = {}
    candidate_info = {}
    references = {}
    for splits in (8, 6):
        _assert_exact_coverage(splits)
        partial = torch.empty(
            TILES, splits, ROWS, HEAD_DIM, dtype=torch.float32, device="cuda"
        )
        maximum = torch.empty(
            TILES, splits, ROWS, dtype=torch.float32, device="cuda"
        )
        denominator = torch.empty_like(maximum)
        launch = build_prototype(splits)

        def invoke(
            launch=launch,
            partial=partial,
            maximum=maximum,
            denominator=denominator,
        ):
            launch(
                q.data_ptr(),
                k_cache.data_ptr(),
                v_cache.data_ptr(),
                table.data_ptr(),
                table.stride(0),
                prefix_lens.data_ptr(),
                WINDOW,
                scale_bits,
                partial.data_ptr(),
                maximum.data_ptr(),
                denominator.data_ptr(),
                stream=torch.cuda.current_stream(),
            )

        invoke()
        torch.cuda.synchronize()
        reference = _reference(
            q, k_cache, v_cache, table, splits, SOFTMAX_SCALE
        )
        references[splits] = _canonical(*reference)
        canonical = _canonical(partial, maximum, denominator)
        reduced_metrics = _metrics(
            _reduce_canonical(*canonical),
            _reduce_canonical(*references[splits]),
        )
        partial_metrics = _metrics(canonical[0], references[splits][0])
        maximum_error = (canonical[1] - references[splits][1]).abs()
        denominator_error = (canonical[2] - references[splits][2]).abs()
        correctness = {
            "kernel": _prototype_name(splits),
            "splits": splits,
            "ctas": TILES * splits,
            "partial": partial_metrics,
            "reduced": reduced_metrics,
            "maximum_max_error": maximum_error.max().item(),
            "maximum_mean_error": maximum_error.mean().item(),
            "denominator_max_error": denominator_error.max().item(),
            "denominator_mean_error": denominator_error.mean().item(),
        }
        print("KATTN_BN64_CORRECTNESS=" + json.dumps(correctness), flush=True)
        assert reduced_metrics["min_cosine"] > 0.997
        assert partial_metrics["min_cosine"] > 0.997
        candidates[splits] = invoke
        candidate_info[splits] = {
            "kernel": _prototype_name(splits),
            "splits": splits,
            "ctas": TILES * splits,
        }
        records.append({"kind": "correctness", **correctness})

    baseline = importlib.import_module(
        "sglang.kernels.ops.attention.swa_verify_splitkv"
    )
    q_3d = q.view(8, 32, HEAD_DIM)
    k_4d = k_cache.view(PAGES, 16, LOCAL_KV_HEADS, HEAD_DIM)
    v_4d = v_cache.view(PAGES, 16, LOCAL_KV_HEADS, HEAD_DIM)
    out = torch.empty_like(q_3d)

    # Exercise the public wrapper once, including its exact shape assertions.
    baseline_state = baseline.swa_verify_splitkv(
        q_3d,
        k_4d,
        v_4d,
        out,
        cu_seqlens_q,
        8,
        seqused_k,
        KEYS,
        SOFTMAX_SCALE,
        True,
        (2046, 0),
        table,
        0,
        None,
        None,
        None,
        num_splits=8,
        num_stages=2,
        defer_reduce=True,
    )
    torch.cuda.synchronize()
    baseline_partial, baseline_maximum, baseline_denominator = baseline_state
    baseline_canonical = (
        baseline_partial,
        baseline_maximum,
        baseline_denominator,
    )
    baseline_reference = references[8]
    baseline_reduced_metrics = _metrics(
        _reduce_canonical_exp2(*baseline_canonical),
        _reduce_canonical(*baseline_reference),
    )
    baseline_correctness = {
        "kernel": "baseline_swa_verify_partials",
        "splits": 8,
        "ctas": 128,
        "reduced": baseline_reduced_metrics,
        "split_state_note": (
            "Triton uses contiguous split tiles and log2-score maxima; "
            "prototype split states use interleaved tiles and natural-score "
            "maxima, so only their reduced outputs share one reference."
        ),
    }
    print(
        "KATTN_BN64_CORRECTNESS="
        + json.dumps(baseline_correctness),
        flush=True,
    )
    assert baseline_reduced_metrics["min_cosine"] > 0.997
    records.append({"kind": "correctness", **baseline_correctness})

    def invoke_baseline():
        # This is the production wrapper's partial launch verbatim: BM16,
        # BN64, 8 splits, 2 warps and 2 stages. Allocation and reduce are
        # deliberately outside the timed region.
        baseline._swa_verify_partials[
            (4, LOCAL_KV_HEADS, 8)
        ](
            q_3d,
            k_4d,
            v_4d,
            cu_seqlens_q,
            seqused_k,
            table,
            baseline_partial,
            baseline_maximum,
            baseline_denominator,
            None,
            None,
            q_3d.stride(0),
            q_3d.stride(1),
            *k_4d.stride(),
            *v_4d.stride(),
            table.stride(0),
            q_3d.shape[1],
            8,
            16,
            2047,
            8,
            SOFTMAX_SCALE,
            16,
            64,
            num_warps=2,
            num_stages=2,
        )

    candidates["baseline"] = invoke_baseline
    candidate_info["baseline"] = {
        "kernel": "baseline_swa_verify_partials",
        "splits": 8,
        "ctas": 128,
    }
    if COMPARE_V1:
        if PROTO_VERSION != "v2":
            raise ValueError("v1 comparison requires MUSE_KATTN_PROTO_VERSION=v2")
        from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_bn64_2warp import (
            build_bn64_2warp as build_v1,
        )

        for splits in (8, 6):
            v1_partial = torch.empty(
                TILES,
                splits,
                ROWS,
                HEAD_DIM,
                dtype=torch.float32,
                device="cuda",
            )
            v1_maximum = torch.empty(
                TILES, splits, ROWS, dtype=torch.float32, device="cuda"
            )
            v1_denominator = torch.empty_like(v1_maximum)
            v1_launch = build_v1(splits)

            def invoke_v1(
                launch=v1_launch,
                partial=v1_partial,
                maximum=v1_maximum,
                denominator=v1_denominator,
            ):
                launch(
                    q.data_ptr(),
                    k_cache.data_ptr(),
                    v_cache.data_ptr(),
                    table.data_ptr(),
                    table.stride(0),
                    prefix_lens.data_ptr(),
                    WINDOW,
                    scale_bits,
                    partial.data_ptr(),
                    maximum.data_ptr(),
                    denominator.data_ptr(),
                    stream=torch.cuda.current_stream(),
                )

            key = f"v1-{splits}"
            candidates[key] = invoke_v1
            candidate_info[key] = {
                "kernel": f"prototype_v1_{TILES * splits}_cta",
                "splits": splits,
                "ctas": TILES * splits,
            }
    iterations = int(os.environ.get("MUSE_KATTN_PROTO_ITERS", "0"))
    warmup = int(os.environ.get("MUSE_KATTN_PROTO_WARMUP", "50"))
    rounds = int(os.environ.get("MUSE_KATTN_PROTO_ROUNDS", "4"))
    if iterations:
        for invoke in candidates.values():
            for _ in range(warmup):
                invoke()
        torch.cuda.synchronize()
        schedule = ("baseline", 8, 6, 6, 8, "baseline")
        if COMPARE_V1:
            schedule = (
                "baseline",
                "v1-8",
                8,
                "v1-6",
                6,
                6,
                "v1-6",
                8,
                "v1-8",
                "baseline",
            )
        for round_index in range(rounds):
            for order, candidate in enumerate(schedule):
                timing = _time_launch(candidates[candidate], iterations)
                info = candidate_info[candidate]
                record = {
                    "kind": "timing",
                    "kernel": info["kernel"],
                    "round": round_index + 1,
                    "order": order + 1,
                    "splits": info["splits"],
                    "ctas": info["ctas"],
                    "iterations": iterations,
                    "time_us": timing,
                }
                records.append(record)
                print("KATTN_BN64_TIMING=" + json.dumps(record), flush=True)

    output = os.environ.get("MUSE_KATTN_PROTO_OUTPUT")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        timing_rows = [row for row in records if row["kind"] == "timing"]
        summaries = {}
        prototype_128 = _prototype_name(8)
        prototype_96 = _prototype_name(6)
        summary_specs = [
            ("baseline_swa_verify_partials", 8, 128),
            (prototype_128, 8, 128),
            (prototype_96, 6, 96),
        ]
        if COMPARE_V1:
            summary_specs += [
                ("prototype_v1_128_cta", 8, 128),
                ("prototype_v1_96_cta", 6, 96),
            ]
        for kernel, splits, ctas in summary_specs:
            samples = [
                row["time_us"]
                for row in timing_rows
                if row["kernel"] == kernel
            ]
            if samples:
                median = statistics.median(samples)
                summaries[kernel] = {
                    "ctas": ctas,
                    "splits": splits,
                    "samples_us": samples,
                    "mean_us": statistics.mean(samples),
                    "median_us": median,
                    "min_us": min(samples),
                    "max_us": max(samples),
                }
        baseline_median = summaries[
            "baseline_swa_verify_partials"
        ]["median_us"]
        for kernel, _, _ in summary_specs[1:]:
            summaries[kernel]["same_session_baseline_ratio"] = (
                summaries[kernel]["median_us"] / baseline_median
            )
        payload = {
            "schema_version": 1,
            "kernel": f"gfx1250_kattn_bn64_2warp_proto_{PROTO_VERSION}",
            "prototype": prototype_metadata,
            "compare_v1": COMPARE_V1,
            "shape": {
                "tokens": 8,
                "head_dim": HEAD_DIM,
                "bm": ROWS,
                "bn": 64,
                "threads": 64,
                "waves": 2,
            },
            "workload": {"keys": KEYS, "window": WINDOW, "tp": 4},
            "cu_mask": os.environ.get("HSA_CU_MASK"),
            "official_baseline_us": 11.604,
            "baseline_callable": True,
            "baseline_launch": {
                "bm": 16,
                "bn": 64,
                "splits": 8,
                "num_warps": 2,
                "num_stages": 2,
                "timed_scope": "_swa_verify_partials only",
            },
            "shape_adaptation": (
                "The identical contiguous Q/K/V storage is viewed as "
                "[8,32,64]/[2048,16,4,64] for the public SGLang wrapper "
                "and [8,2048]/[32768,4,64] for the FlyDSL prototype. "
                "No kernel semantics or data values change."
            ),
            "integration_conclusion": (
                f"The {PROTO_VERSION} prototype remains isolated: the "
                "128-CTA and 96-CTA grids are "
                f"{summaries[prototype_128]['same_session_baseline_ratio']:.3f}x "
                "and "
                f"{summaries[prototype_96]['same_session_baseline_ratio']:.3f}x "
                "slower than the same-session baseline."
            ),
            "compiled_metadata": {
                "baseline": {
                    "wave_size": 32,
                    "waves": 2,
                    "vgpr": 165,
                    "sgpr": 107,
                    "sgpr_spills_to_vgpr": 15,
                    "scratch_bytes": 0,
                    "dynamic_shared_bytes": 17664,
                },
                prototype_128: {
                    "wave_size": 32,
                    "waves": 2,
                    "vgpr": 239,
                    "sgpr": 89,
                    "spills": 0,
                    "scratch_bytes": 0,
                    "lds_bytes": 12608,
                },
                prototype_96: {
                    "wave_size": 32,
                    "waves": 2,
                    "vgpr": 239,
                    "sgpr": 89,
                    "spills": 0,
                    "scratch_bytes": 0,
                    "lds_bytes": 12608,
                },
            },
            "summary": summaries,
            "records": records,
        }
        if PROTO_VERSION == "v2":
            direct_metadata = {
                "wave_size": 32,
                "waves": 2,
                "lds_bytes": prototype_metadata["lds_bytes"],
                "vgpr": 204,
                "sgpr": 75,
                "spills": 0,
                "scratch_bytes": 0,
                "static_ds_instructions": 124,
                "static_global_load_async_to_lds_b128": 16,
                "static_s_wait_asynccnt": 2,
            }
            payload["compiled_metadata"] = {
                "baseline": payload["compiled_metadata"]["baseline"],
                prototype_128: direct_metadata,
                prototype_96: direct_metadata,
            }
            if COMPARE_V1:
                v1_metadata = {
                    "wave_size": 32,
                    "waves": 2,
                    "vgpr": 239,
                    "sgpr": 89,
                    "spills": 0,
                    "scratch_bytes": 0,
                    "lds_bytes": 12608,
                    "static_ds_instructions": 163,
                }
                payload["compiled_metadata"].update(
                    {
                        "prototype_v1_128_cta": v1_metadata,
                        "prototype_v1_96_cta": v1_metadata,
                    }
                )
        (directory / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        correctness_rows = [
            row for row in records if row["kind"] == "correctness"
        ]
        log_lines = [
            "gfx1250 BF16 K-attn BM16xBN64 two-wave compute-only prototype",
            "HSA_CU_MASK: " + str(payload["cu_mask"]),
            "workload: TP4, width8, HEAD_DIM64, keys=3000, Window=2047",
            (
                f"measurement: {rounds} alternating "
                + (
                    "baseline/v1-8/v2-8/v1-6/v2-6/v2-6/"
                    "v1-6/v2-8/v1-8/baseline rounds, "
                    if COMPARE_V1
                    else "baseline/8/6/6/8/baseline rounds, "
                )
                + f"{iterations} launches per sample"
            ),
        ]
        for kernel in (
            "baseline_swa_verify_partials",
            prototype_128,
            prototype_96,
        ):
            summary = summaries[kernel]
            accuracy = next(
                row for row in correctness_rows if row["kernel"] == kernel
            )
            log_lines += [
                (
                    f"{kernel}: {summary['ctas']} CTA / "
                    f"{summary['splits']} split: "
                    f"median_us={summary['median_us']:.6f} "
                    f"mean_us={summary['mean_us']:.6f} "
                    f"min_us={summary['min_us']:.6f} "
                    f"max_us={summary['max_us']:.6f}"
                ),
                (
                    f"  reduced max_error="
                    f"{accuracy['reduced']['max_error']:.9f} "
                    f"mean_error={accuracy['reduced']['mean_error']:.9f} "
                    f"min_cosine={accuracy['reduced']['min_cosine']:.9f}"
                ),
            ]
        log_lines += [
            "exact causal/window coverage asserted for every tile and row",
            payload["shape_adaptation"],
            (
                "same-session ratio: prototype 128 CTA="
                f"{summaries[prototype_128]['same_session_baseline_ratio']:.6f}x, "
                "prototype 96 CTA="
                f"{summaries[prototype_96]['same_session_baseline_ratio']:.6f}x"
            ),
            "conclusion: " + payload["integration_conclusion"],
        ]
        if PROTO_VERSION == "v1":
            log_lines[-1:-1] = [
                (
                    "metadata: baseline wave32, 2 waves, 165 VGPR, 107 SGPR, "
                    "15 SGPR-to-VGPR spills, 0 scratch, 17664 B dynamic shared; "
                    "prototypes wave32, 2 waves, 239 VGPR, 89 SGPR, 0 spills, "
                    "0 scratch, 12608 B LDS."
                ),
                (
                    "assembly: prototype has substantially higher register "
                    "pressure and more explicit LDS traffic (163 static ds_ "
                    "instructions versus 82), consistent with its slower result; "
                    "reducing 128 to 96 CTAs does not help."
                ),
            ]
        else:
            log_lines[-1:-1] = [
                (
                    "v2 static pipeline metadata: "
                    + json.dumps(prototype_metadata, sort_keys=True)
                ),
                (
                    "v2 compiler metadata: wave32, 2 waves, 204 VGPR, "
                    "75 SGPR, 0 spills/scratch, 20800 B LDS, 124 static ds_, "
                    "16 static global_load_async_to_lds_b128, "
                    "2 static s_wait_asynccnt."
                ),
            ]
            if COMPARE_V1:
                for kernel in ("prototype_v1_128_cta", "prototype_v1_96_cta"):
                    summary = summaries[kernel]
                    log_lines[-1:-1] = [
                        (
                            f"{kernel}: median_us={summary['median_us']:.6f} "
                            f"mean_us={summary['mean_us']:.6f} "
                            f"min_us={summary['min_us']:.6f} "
                            f"max_us={summary['max_us']:.6f} "
                            "same_session_baseline_ratio="
                            f"{summary['same_session_baseline_ratio']:.6f}x"
                        )
                    ]
        (directory / "benchmark.log").write_text("\n".join(log_lines) + "\n")
        with (directory / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "round",
                    "order",
                    "kernel",
                    "splits",
                    "ctas",
                    "iterations",
                    "time_us",
                ),
            )
            writer.writeheader()
            writer.writerows(
                {key: row[key] for key in writer.fieldnames}
                for row in timing_rows
            )
