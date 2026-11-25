# SPDX-License-Identifier: Apache-2.0
"""Isolated, bounded Width-8 Phase C routing-stress driver.

Each case gets a fresh torchrun/container.  The known Baseline hot-expert fault
is not exercised here: this driver gates Mono against the decoded-weight
PyTorch reference and places hot-expert last.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable

CASES = (
    "zero_local_pick",
    "expert_boundary",
    "score_ties",
    "imbalance",
    "hot_expert",
)
HSA_FAULT_MARKERS = (
    "HSA_STATUS_ERROR_MEMORY_FAULT",
    "Memory access fault by GPU",
    "Queue error",
)


def classify_result(returncode: int, output: str) -> str:
    if any(marker in output for marker in HSA_FAULT_MARKERS):
        return "hsa_fault"
    if returncode == 124:
        return "timeout"
    if returncode != 0:
        return "failed"
    return "passed"


def run_bounded_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    terminate_grace_seconds: float = 30.0,
) -> tuple[int, str]:
    """Run a case in its own process group and terminate the whole group on timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, output
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + terminate_grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 124, output


def run_case_sequence(
    run_case: Callable[[str], tuple[int, str]],
) -> list[dict[str, object]]:
    """Run isolated cases in safe order and stop after any new HSA fault."""
    results: list[dict[str, object]] = []
    for case in CASES:
        returncode, output = run_case(case)
        status = classify_result(returncode, output)
        results.append(
            {
                "case": case,
                "status": status,
                "returncode": returncode,
            }
        )
        if status == "hsa_fault":
            break
    return results


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded isolated Width-8 Mono routing stress"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.timeout_seconds < 60:
        raise ValueError("per-case timeout must be at least 60 seconds")
    workspace = args.workspace.resolve()
    wrapper = workspace / "tp8_synthetic.sh"
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    detailed: dict[str, dict[str, object]] = {}

    def launch(case: str) -> tuple[int, str]:
        case_dir = output_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(wrapper),
            "--mode",
            "parity",
            "--stage",
            "k3",
            "--widths",
            "8",
            "--impl",
            "mono",
            "--profile",
            "none",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--active-gpus",
            "4",
            "--devices",
            args.devices,
            "--output-dir",
            str(case_dir),
        ]
        env = dict(os.environ)
        env["TP8_ROUTING_STRESS_CASE"] = case
        returncode, log = run_bounded_subprocess(
            command,
            cwd=workspace,
            env=env,
            timeout_seconds=args.timeout_seconds,
        )
        (case_dir / "subprocess.log").write_text(log)
        detailed[case] = {
            "command": command,
            "status": classify_result(returncode, log),
            "returncode": returncode,
            "log": str(case_dir / "subprocess.log"),
        }
        return returncode, log

    sequence = run_case_sequence(launch)
    for item in sequence:
        item.update(detailed[item["case"]])
    summary = {
        "schema": "tp8-phase-c-routing-stress",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "width": 8,
        "active_gpus": 4,
        "logical_tp": 8,
        "logical_ep": 8,
        "resident_cu": 96,
        "oracle": "independent-decoded-weight-pytorch",
        "baseline_oracle_used": False,
        "strict_tolerance_relaxed": False,
        "case_order": CASES,
        "results": sequence,
        "stopped_after_hsa_fault": bool(
            sequence and sequence[-1]["status"] == "hsa_fault"
        ),
    }
    statuses = [item["status"] for item in sequence]
    summary["status"] = (
        "passed"
        if len(sequence) == len(CASES) and set(statuses) == {"passed"}
        else "failed"
    )
    (output_root / "routing_stress_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
