#!/usr/bin/env python3
"""Reproducible compile-time Block A sweep for gfx1250 Target K-attn."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path


MONO = Path(__file__).resolve().parents[2]
ROOT = MONO.parents[2]
TEST = "test/kernels/test_muse_spark_kattn_gfx1250.py"
SCENARIOS = {
    "full": "test_gfx1250_kattn_matches_paged_attention[8200-0-True-8]",
    "window": "test_gfx1250_kattn_matches_paged_attention[3000-2047-False-8]",
}
TIME_RE = re.compile(r"KATTN_BENCH_US=([0-9.]+)")
ACCURACY_RE = re.compile(
    r"KATTN_ACCURACY max_error=([0-9.]+) "
    r"mean_error=([0-9.]+) min_cosine=([0-9.]+)"
)
CONTROL = (32, 6, 64)


def config_id(config: tuple[int, int, int]) -> str:
    bn, splits, load_bits = config
    return f"bn{bn}-s{splits}-l{load_bits}"


def docker_command(
    args: argparse.Namespace,
    scenario: str,
    config: tuple[int, int, int],
) -> list[str]:
    bn, splits, load_bits = config
    python = ROOT / ".venv/mi45x/bin/python3"
    env = {
        "HSA_CU_MASK": args.cu_mask,
        "MUSE_KATTN_BENCH_ITERS": str(args.iterations),
        "MUSE_KATTN_BENCH_WARMUP": str(args.warmup),
        "MUSE_KATTN_FP8_WMMA": "0",
        "MUSE_KATTN_KEY_BLOCK": str(bn),
        "MUSE_KATTN_LOAD_BITS": str(load_bits),
        "MUSE_KATTN_REGISTER_PROBABILITY": "1",
        "MUSE_KATTN_STAGE_Q": "1",
        "MUSE_KATTN_TEST_SPLITS": str(splits),
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add=video",
        "--ipc=host",
        "--network=host",
        "--shm-size=32g",
        f"--volume={ROOT}:{ROOT}",
        f"--workdir={MONO}",
    ]
    command += [f"--env={key}={value}" for key, value in sorted(env.items())]
    return [
        *command,
        args.image,
        str(python),
        "-m",
        "pytest",
        "-q",
        "-s",
        f"{TEST}::{SCENARIOS[scenario]}",
    ]


def persist(output: Path, records: list[dict], metadata: dict) -> None:
    payload = {**metadata, "records": records}
    (output / "sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    columns = [
        "sequence",
        "scenario",
        "config",
        "bn",
        "splits",
        "load_bits",
        "anchor",
        "status",
        "time_us",
        "max_error",
        "mean_error",
        "min_cosine",
        "duration_seconds",
        "log",
    ]
    with (output / "sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in records)


def run_one(
    args: argparse.Namespace,
    output: Path,
    sequence: int,
    scenario: str,
    config: tuple[int, int, int],
    anchor: bool,
) -> dict:
    label = f"{sequence:03d}-{scenario}-{config_id(config)}"
    log = output / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.run(
        docker_command(args, scenario, config),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = process.stdout
    log.write_text(text)
    timing = TIME_RE.search(text)
    accuracy = ACCURACY_RE.search(text)
    bn, splits, load_bits = config
    passed = process.returncode == 0 and timing is not None and accuracy is not None
    record = {
        "sequence": sequence,
        "scenario": scenario,
        "config": config_id(config),
        "bn": bn,
        "splits": splits,
        "load_bits": load_bits,
        "anchor": anchor,
        "status": "PASS" if passed else "FAIL",
        "time_us": float(timing.group(1)) if timing else None,
        "max_error": float(accuracy.group(1)) if accuracy else None,
        "mean_error": float(accuracy.group(2)) if accuracy else None,
        "min_cosine": float(accuracy.group(3)) if accuracy else None,
        "duration_seconds": time.monotonic() - started,
        "log": str(log),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def summarize(records: list[dict]) -> dict:
    summary = {}
    for scenario in SCENARIOS:
        passing = [
            row
            for row in records
            if row["scenario"] == scenario
            and row["status"] == "PASS"
            and not row["anchor"]
        ]
        grouped: dict[str, list[float]] = {}
        for row in passing:
            grouped.setdefault(row["config"], []).append(row["time_us"])
        ranked = sorted(
            (
                {
                    "config": config,
                    "samples_us": samples,
                    "median_us": statistics.median(samples),
                    "mean_us": statistics.mean(samples),
                }
                for config, samples in grouped.items()
            ),
            key=lambda item: item["median_us"],
        )
        summary[scenario] = {"passing": len(grouped), "ranking": ranked}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/replay-profiles/kattn-block-a",
    )
    parser.add_argument(
        "--image", default="rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261112"
    )
    parser.add_argument("--cu-mask", default="0:0-95")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--anchor-every", type=int, default=6)
    parser.add_argument("--repeat-top", type=int, default=3)
    parser.add_argument("--repeat-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1250)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product((16, 32, 64), (2, 3, 4, 6), (64, 128)))
    rng = random.Random(args.seed)
    schedule: list[tuple[str, tuple[int, int, int], bool]] = []
    for scenario in SCENARIOS:
        ordered = configs.copy()
        rng.shuffle(ordered)
        for index, config in enumerate(ordered):
            if index and index % args.anchor_every == 0:
                schedule.append((scenario, CONTROL, True))
            schedule.append((scenario, config, False))
        schedule.append((scenario, CONTROL, True))
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cu_mask": args.cu_mask,
        "image": args.image,
        "control": config_id(CONTROL),
        "production_defaults_unchanged": True,
        "fp8_wmma": False,
    }
    records = []
    for sequence, (scenario, config, anchor) in enumerate(schedule, 1):
        records.append(
            run_one(args, output, sequence, scenario, config, anchor)
        )
        persist(output, records, metadata)
    sequence = len(records)
    first_summary = summarize(records)
    repeats = []
    by_id = {config_id(config): config for config in configs}
    for scenario in SCENARIOS:
        top = first_summary[scenario]["ranking"][: args.repeat_top]
        for _ in range(args.repeat_rounds):
            round_configs = [by_id[item["config"]] for item in top]
            rng.shuffle(round_configs)
            repeats.extend((scenario, config, False) for config in round_configs)
            repeats.append((scenario, CONTROL, True))
    for scenario, config, anchor in repeats:
        sequence += 1
        records.append(
            run_one(args, output, sequence, scenario, config, anchor)
        )
        persist(output, records, metadata)
    metadata["summary"] = summarize(records)
    persist(output, records, metadata)
    print(json.dumps(metadata["summary"], indent=2, sort_keys=True))
    return 0 if all(row["status"] == "PASS" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
