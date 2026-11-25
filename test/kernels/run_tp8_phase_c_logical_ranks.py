# SPDX-License-Identifier: Apache-2.0
"""Bounded active1/2 logical-EP-rank coverage for Width-8 Mono K3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from run_tp8_phase_c_routing import classify_result, run_bounded_subprocess
from tp8_p1_host_contract import (
    logical_rank_batches,
    validate_logical_rank_observations,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = _args()
    workspace = args.workspace.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wrapper = workspace / "tp8_synthetic.sh"
    all_topologies: dict[str, object] = {}
    failed = False
    hsa_fault = False
    for active_gpus in (1, 2):
        observations = []
        for batch_index, logical_ranks in enumerate(
            logical_rank_batches(active_gpus)
        ):
            batch_dir = (
                output_root
                / f"active{active_gpus}"
                / (
                    f"batch{batch_index}-"
                    + "-".join(str(value) for value in logical_ranks)
                )
            )
            batch_dir.mkdir(parents=True, exist_ok=True)
            devices = "0" if active_gpus == 1 else "0,1"
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
                str(active_gpus),
                "--devices",
                devices,
                "--output-dir",
                str(batch_dir),
            ]
            env = dict(os.environ)
            env["TP8_LOGICAL_RANKS"] = ",".join(
                str(value) for value in logical_ranks
            )
            returncode, output = run_bounded_subprocess(
                command,
                cwd=workspace,
                env=env,
                timeout_seconds=args.timeout_seconds,
            )
            (batch_dir / "subprocess.log").write_text(output)
            status = classify_result(returncode, output)
            if status == "passed":
                artifact = json.loads(
                    (batch_dir / "logical_rank_batch.json").read_text()
                )
                observations.append(artifact)
            else:
                failed = True
                hsa_fault = status == "hsa_fault"
                observations.append(
                    {
                        "status": status,
                        "logical_ranks": logical_ranks,
                        "active_gpus": active_gpus,
                        "runtime_mapping_hash": "",
                        "returncode": returncode,
                    }
                )
            if hsa_fault:
                break
        if not failed:
            validate_logical_rank_observations(active_gpus, observations)
        all_topologies[f"active{active_gpus}"] = observations
        if hsa_fault:
            break
    hashes = [
        item["runtime_mapping_hash"]
        for observations in all_topologies.values()
        for item in observations
        if item.get("status") == "passed"
    ]
    aggregate_hash = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "schema": "tp8-phase-c-logical-rank-coverage",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failed else "passed",
        "width": 8,
        "specialization": "W8",
        "logical_tp": 8,
        "logical_ep": 8,
        "local_experts_per_rank": 32,
        "coverage_kind": "logical-ep-ownership-batches",
        "physical_communication_scope": "active1/2 proxy slots",
        "topologies": all_topologies,
        "mapping_hashes": hashes,
        "aggregate_mapping_hash": aggregate_hash,
        "stopped_after_hsa_fault": hsa_fault,
    }
    (output_root / "logical_rank_coverage.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
