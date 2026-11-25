# SPDX-License-Identifier: Apache-2.0
"""Bounded Width-8 independent whole-layer reference matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from run_tp8_phase_c_routing import classify_result, run_bounded_subprocess

CASES = ("layer_v1_full", "layer_v1_swa", "layer_v2_full", "layer_v2_swa")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        case_dir = output_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(workspace / "tp8_synthetic.sh"),
            "--mode", "parity",
            "--stage", "layer",
            "--widths", "8",
            "--impl", "mono",
            "--profile", "none",
            "--warmup", "0",
            "--iterations", "1",
            "--active-gpus", "4",
            "--devices", "0,1,2,3",
            "--output-dir", str(case_dir),
        ]
        env = dict(os.environ)
        env["TP8_WHOLE_LAYER_REFERENCE_CASE"] = case
        returncode, output = run_bounded_subprocess(
            command,
            cwd=workspace,
            env=env,
            timeout_seconds=args.timeout_seconds,
        )
        (case_dir / "subprocess.log").write_text(output)
        status = classify_result(returncode, output)
        results.append(
            {
                "case": case,
                "status": status,
                "returncode": returncode,
                "log": str(case_dir / "subprocess.log"),
            }
        )
        if status == "hsa_fault":
            break
    passed = len(results) == len(CASES) and all(
        item["status"] == "passed" for item in results
    )
    summary = {
        "schema": "tp8-phase-c-whole-layer-reference",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "width": 8,
        "active_gpus": 4,
        "logical_tp": 8,
        "reference": "independent-composed-pytorch",
        "mono_split_dependency": False,
        "cases": results,
        "stopped_after_hsa_fault": bool(
            results and results[-1]["status"] == "hsa_fault"
        ),
    }
    (output_root / "whole_layer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
