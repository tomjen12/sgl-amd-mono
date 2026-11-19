#!/usr/bin/env python3
"""Run isolated gfx1250 K-attention v2 correctness and optional timing."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MONO = Path(__file__).resolve().parents[2]
TEST = (
    "test/kernels/test_gfx1250_kattn_bn64_2warp.py"
    "::test_gfx1250_kattn_bn64_2warp_compute_only"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--compare-v1",
        action="store_true",
        help="Interleave v1, direct-v2, and baseline timing in one session",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write v2 results.json, benchmark.log, and timings.csv here",
    )
    args = parser.parse_args()
    if args.output_dir and not args.iterations:
        parser.error("--output-dir requires --iterations")

    env = os.environ.copy()
    env.update(
        {
            "MUSE_KATTN_PROTO_VERSION": "v2",
            "MUSE_KATTN_PROTO_ITERS": str(args.iterations),
            "MUSE_KATTN_PROTO_WARMUP": str(args.warmup),
            "MUSE_KATTN_PROTO_ROUNDS": str(args.rounds),
            "MUSE_KATTN_PROTO_COMPARE_V1": "1" if args.compare_v1 else "0",
        }
    )
    if args.output_dir:
        env["MUSE_KATTN_PROTO_OUTPUT"] = str(args.output_dir.resolve())
    command = [sys.executable, "-m", "pytest", "-q", "-s", TEST]
    return subprocess.run(command, cwd=MONO, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
