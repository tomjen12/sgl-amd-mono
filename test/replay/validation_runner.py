#!/usr/bin/env python3
"""Fail-closed Muse Spark replay validation orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from validation_contract import (
    SUMMARY_SCHEMA_VERSION,
    correctness_environment,
    finalize_summary,
    load_manifest,
    repository_provenance,
    workload_identity,
)

REPLAY_DIR = Path(__file__).resolve().parent
MONO_ROOT = REPLAY_DIR.parents[1]
ROOT = MONO_ROOT.parents[2]
DEFAULT_MANIFEST = ROOT / "replay_validation" / "manifest.json"
DEFAULT_IMAGE = "rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261112"


def _docker(
    image: str,
    cu_mask: str,
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> list[str]:
    docker = [
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
        f"--workdir={MONO_ROOT}",
        f"--env=HSA_CU_MASK={cu_mask}",
    ]
    docker.extend(
        f"--env={key}={value}" for key, value in sorted((environment or {}).items())
    )
    return [*docker, image, *command]


def _run_gate(
    gate_id: str,
    command: list[str],
    output_dir: Path,
    *,
    required: bool = True,
    fail_on_skip: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / f"{gate_id}.log"
    started = time.monotonic()
    print(f"[validation] START {gate_id}", flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        lines: list[str] = []

        def pump() -> None:
            assert process.stdout is not None
            with log.open("w", encoding="utf-8") as stream:
                for line in process.stdout:
                    lines.append(line)
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            exit_code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join(timeout=10)
            text = "".join(lines) + "\nVALIDATION TIMEOUT\n"
            with log.open("a", encoding="utf-8") as stream:
                stream.write("\nVALIDATION TIMEOUT\n")
            status, exit_code = "FAIL", None
            details = {"error": "timeout"}
        else:
            reader.join(timeout=10)
            text = "".join(lines)
            skipped = sum(
                int(value)
                for value in re.findall(r"(\d+)\s+skipped", text)
            )
            passed = sum(
                int(value)
                for value in re.findall(r"(\d+)\s+passed", text)
            )
            status = "PASS" if exit_code == 0 else "FAIL"
            if fail_on_skip and skipped and passed == 0:
                status = "SKIP"
            details = {"passed": passed, "skipped": skipped}
    except OSError as error:
        log.write_text(str(error) + "\n", encoding="utf-8")
        status, exit_code = "MISSING", None
        details = {"error": str(error)}
    elapsed = time.monotonic() - started
    print(
        f"[validation] {status} {gate_id} ({elapsed:.1f}s) log={log}",
        flush=True,
    )
    return {
        "id": gate_id,
        "required": required,
        "status": status,
        "duration_seconds": elapsed,
        "command": command,
        "exit_code": exit_code,
        "artifacts": [str(log)],
        "details": details,
    }


def _pytest_gate(
    gate_id: str,
    tests: list[str],
    output_dir: Path,
    image: str,
    cu_mask: str,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    python = str(ROOT / ".venv/mi45x/bin/python3")
    command = _docker(
        image,
        cu_mask,
        [python, "-m", "pytest", "-q", "-rA", *tests],
        environment=environment,
    )
    return _run_gate(
        gate_id, command, output_dir, fail_on_skip=True
    )


def _full_replay_gates(
    scope: str,
    output_dir: Path,
    image: str,
    cu_mask: str,
    target_kernel_mode: str,
) -> list[dict[str, Any]]:
    python = str(ROOT / ".venv/mi45x/bin/python3")
    bundle = ROOT / "replay_bundle"
    baseline = output_dir / "baseline"
    mono = output_dir / "mono"
    baseline.mkdir(parents=True, exist_ok=True)
    mono.mkdir(parents=True, exist_ok=True)
    gates = []
    if scope == "draft":
        checkpoint = Path("/data/Muse-Spark-Open-DFlash2")
        draft_common = [
            python,
            str(REPLAY_DIR / "muse_spark_draft_replay.py"),
            str(bundle),
            "--checkpoint",
            str(checkpoint),
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
        base_cmd = [
            *draft_common,
            "--output-dir",
            str(baseline),
            "--mode",
            "baseline",
            "--trace",
        ]
        mono_cmd = [
            *draft_common,
            "--output-dir",
            str(mono),
            "--mode",
            "mono",
            "--trace",
        ]
    else:
        target = ROOT / "synthetic-target-0-9"
        phase_dir = mono / "phases"
        base_cmd = [
            python,
            str(REPLAY_DIR / "muse_spark_target_replay.py"),
            str(bundle),
            str(target),
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--snapshot-dir",
            str(baseline),
            "--trace-dir",
            str(baseline / "traces"),
            "--output",
            str(baseline / "summary.json"),
            "--mode",
            "baseline",
        ]
        mono_cmd = [
            python,
            str(REPLAY_DIR / "muse_spark_target_replay.py"),
            str(bundle),
            str(target),
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--routing-snapshot-dir",
            str(baseline),
            "--snapshot-dir",
            str(mono),
            "--trace-dir",
            str(mono / "traces"),
            "--output",
            str(mono / "summary.json"),
            "--mode",
            "mono",
            "--target-kernel-mode",
            target_kernel_mode,
            "--phase-dir",
            str(phase_dir),
            "--check",
        ]
    gates.append(
        _run_gate(
            "baseline_replay",
            _docker(image, cu_mask, base_cmd),
            output_dir,
        )
    )
    gates.append(
        _run_gate(
            "mono_replay",
            _docker(image, cu_mask, mono_cmd),
            output_dir,
        )
    )
    comparator = (
        "compare_muse_spark_gate2.py"
        if scope == "draft"
        else "compare_muse_spark_target_gate2.py"
    )
    compare_cmd = [
        python,
        str(REPLAY_DIR / comparator),
        str(baseline),
        str(mono),
        "--output",
        str(output_dir / "strict-comparison.json"),
    ]
    gates.append(
        _run_gate("strict_comparator", compare_cmd, output_dir)
        if all(item["status"] == "PASS" for item in gates)
        else {
            "id": "strict_comparator",
            "required": True,
            "status": "MISSING",
            "duration_seconds": 0.0,
            "command": compare_cmd,
            "exit_code": None,
            "artifacts": [],
            "details": {"error": "replay prerequisite failed"},
        }
    )
    if scope == "target":
        phase_cmd = [
            python,
            str(REPLAY_DIR / "parse_muse_phase_trace.py"),
            *[
                str(phase_dir / f"target.rank_{rank:04d}.phases.json")
                for rank in range(4)
            ],
            "--output",
            str(output_dir / "phase-summary.json"),
            "--require-family",
            "K1",
            "--require-family",
            "K-attn",
            "--require-family",
            "K2",
            "--require-family",
            "K3",
            "--require-ranks",
            "4",
            "--require-layers",
            "9",
            "--require-layer-start",
            "1",
        ]
        gates.append(
            _run_gate("phase_coverage", phase_cmd, output_dir)
            if gates[1]["status"] == "PASS"
            else {
                "id": "phase_coverage",
                "required": True,
                "status": "MISSING",
                "duration_seconds": 0.0,
                "command": phase_cmd,
                "exit_code": None,
                "artifacts": [],
                "details": {"error": "mono replay prerequisite failed"},
            }
        )
    return gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    selected = manifest["scopes"][args.scope][args.level]
    stages = sorted(filter(None, args.stages.split(","))) if args.stages else []
    image = args.image or os.environ.get("MUSE_REPLAY_IMAGE", DEFAULT_IMAGE)
    cu_mask = args.cu_mask or manifest["workload"]["cu_mask"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gates = [
        _pytest_gate(
            "focused_kernels",
            selected["focused"],
            output_dir,
            image,
            cu_mask,
        ),
        _pytest_gate(
            "state_carry",
            selected["state_carry"],
            output_dir,
            image,
            cu_mask,
            environment=(
                {"MUSE_TARGET_SPLIT_ONLY": "1"}
                if args.scope == "target"
                else None
            ),
        ),
    ]
    if args.scope == "target":
        gates.append(
            _pytest_gate(
                "phase_focused_kernels",
                selected["phase_focused"],
                output_dir,
                image,
                cu_mask,
                environment={"MUSE_PHASE_TRACE": "1"},
            )
        )
    if args.level == "full":
        gates.extend(
            _full_replay_gates(
                args.scope,
                output_dir,
                image,
                cu_mask,
                args.target_kernel_mode,
            )
        )
    provenance_paths = [
        ROOT,
        ROOT / "sources/mi45x/sgl-amd-mono",
        ROOT / "sources/mi45x/sglang",
        ROOT / "sources/mi45x/aiter",
    ]
    summary = finalize_summary(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "scope": args.scope,
            "level": args.level,
            "workload_hash": workload_identity(
                manifest, args.scope, args.level, stages
            ),
            "provenance": [
                repository_provenance(path) for path in provenance_paths
            ],
            "environment": correctness_environment(image, cu_mask),
            "gates": gates,
        }
    )
    summary_path = output_dir / "validation-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["overall"] != "PASS":
        print(
            f"first failing gate: {summary['first_failing_gate']}; "
            f"artifacts: {output_dir}"
        )
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("draft", "target"), required=True)
    parser.add_argument("--level", choices=("quick", "full"), required=True)
    parser.add_argument("--stages", default="")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--cu-mask")
    parser.add_argument(
        "--target-kernel-mode", choices=("split", "layer"), default="split"
    )
    return parser


def main() -> int:
    summary = run(create_parser().parse_args())
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
