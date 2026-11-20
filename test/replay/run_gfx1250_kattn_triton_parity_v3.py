#!/usr/bin/env python3
"""Run isolated gfx1250 K-attention Triton-parity v3 correctness and timing."""

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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--compare-v1", action="store_true")
    parser.add_argument("--compare-legacy-v3", action="store_true")
    parser.add_argument("--compare-compact-v3", action="store_true")
    parser.add_argument("--compare-register-v3", action="store_true")
    parser.add_argument("--hoist-q", action="store_true")
    parser.add_argument("--register-probability", action="store_true")
    parser.add_argument("--dedup-page", action="store_true")
    parser.add_argument("--legacy-kv", action="store_true")
    parser.add_argument("--vector-epilogue", action="store_true")
    parser.add_argument("--stage-pages", action="store_true")
    parser.add_argument("--interleave-pv-pack", action="store_true")
    parser.add_argument("--paired-page-load", action="store_true")
    parser.add_argument("--baseline-llvm-attrs", action="store_true")
    parser.add_argument("--inline-wmma-pair", action="store_true")
    parser.add_argument("--direct-compiled", action="store_true")
    parser.add_argument("--device-matrix", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir and not args.iterations:
        parser.error("--output-dir requires --iterations")

    env = os.environ.copy()
    env.update(
        {
            "MUSE_KATTN_PROTO_VERSION": "v3",
            "MUSE_KATTN_PROTO_ITERS": str(args.iterations),
            "MUSE_KATTN_PROTO_WARMUP": str(args.warmup),
            "MUSE_KATTN_PROTO_ROUNDS": str(args.rounds),
            "MUSE_KATTN_PROTO_COMPARE_V1": "1" if args.compare_v1 else "0",
            "MUSE_KATTN_PROTO_COMPARE_LEGACY_V3": (
                "1" if args.compare_legacy_v3 else "0"
            ),
            "MUSE_KATTN_PROTO_COMPARE_COMPACT_V3": (
                "1" if args.compare_compact_v3 else "0"
            ),
            "MUSE_KATTN_PROTO_COMPARE_REGISTER_V3": (
                "1" if args.compare_register_v3 else "0"
            ),
            "MUSE_KATTN_V3_HOIST_Q": "1" if args.hoist_q else "0",
            "MUSE_KATTN_V3_REGISTER_P": (
                "1" if args.register_probability else "0"
            ),
            "MUSE_KATTN_V3_DEDUP_PAGE": "1" if args.dedup_page else "0",
            "MUSE_KATTN_V3_LEGACY_KV": "1" if args.legacy_kv else "0",
            "MUSE_KATTN_V3_VECTOR_EPILOGUE": (
                "1" if args.vector_epilogue else "0"
            ),
            "MUSE_KATTN_V3_STAGE_PAGES": "1" if args.stage_pages else "0",
            "MUSE_KATTN_V3_INTERLEAVE_PV_PACK": (
                "1" if args.interleave_pv_pack else "0"
            ),
            "MUSE_KATTN_V3_PAIRED_PAGE_LOAD": (
                "1" if args.paired_page_load else "0"
            ),
            "MUSE_KATTN_V3_BASELINE_LLVM_ATTRS": (
                "1" if args.baseline_llvm_attrs else "0"
            ),
            "MUSE_KATTN_V3_INLINE_WMMA_PAIR": (
                "1" if args.inline_wmma_pair else "0"
            ),
            "MUSE_KATTN_V3_DIRECT_COMPILED": (
                "1" if args.direct_compiled else "0"
            ),
            "MUSE_KATTN_DEVICE_MATRIX": "1" if args.device_matrix else "0",
        }
    )
    if args.output_dir:
        env["MUSE_KATTN_PROTO_OUTPUT"] = str(args.output_dir.resolve())
    command = [sys.executable, "-m", "pytest", "-q", "-s", TEST]
    return subprocess.run(command, cwd=MONO, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
