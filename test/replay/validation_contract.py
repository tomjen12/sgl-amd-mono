"""Shared fail-closed contracts for Muse Spark replay validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

COSINE_THRESHOLD = 0.997
SUMMARY_SCHEMA_VERSION = 1
VALID_GATE_STATUSES = frozenset(("PASS", "FAIL", "MISSING", "SKIP"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid validation manifest {path}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise ValueError("validation manifest schema_version must be 1")
    if manifest.get("cosine_threshold") != COSINE_THRESHOLD:
        raise ValueError(
            f"validation cosine threshold must be exactly {COSINE_THRESHOLD}"
        )
    scopes = manifest.get("scopes")
    if not isinstance(scopes, dict):
        raise ValueError("validation manifest scopes must be an object")
    for scope in ("draft", "target"):
        for level in ("quick", "full"):
            if level not in scopes.get(scope, {}):
                raise ValueError(f"manifest missing {scope}.{level}")
            selected = scopes[scope][level]
            for gate in ("focused", "state_carry"):
                if not selected.get(gate):
                    raise ValueError(f"manifest missing {scope}.{level}.{gate}")
            if scope == "target" and not selected.get("phase_focused"):
                raise ValueError(
                    f"manifest missing {scope}.{level}.phase_focused"
                )
    return manifest


def workload_identity(
    manifest: dict[str, Any], scope: str, level: str, stages: list[str]
) -> str:
    return canonical_hash(
        {
            "schema_version": manifest["schema_version"],
            "scope": scope,
            "level": level,
            "stages": sorted(stages),
            "workload": manifest["workload"],
        }
    )


def repository_provenance(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(path), *args),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        diff = git("diff", "--binary", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        return {"path": str(path), "status": "MISSING", "error": str(error)}
    dirty = bool(status)
    return {
        "path": str(path),
        "commit": commit,
        "dirty": dirty,
        "dirty_hash": canonical_hash({"status": status, "diff": diff})
        if dirty
        else None,
    }


def correctness_environment(image: str, cu_mask: str) -> dict[str, Any]:
    prefixes = ("SGLANG_", "MUSE_", "HSA_", "ROCR_", "HIP_")
    variables = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }
    return {"image": image, "cu_mask": cu_mask, "variables": variables}


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    gates = summary.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("validation summary must contain gates")
    first_failure = None
    for gate in gates:
        status = gate.get("status")
        if status not in VALID_GATE_STATUSES:
            raise ValueError(f"invalid gate status {status!r}")
        if gate.get("required", True) and status != "PASS" and first_failure is None:
            first_failure = gate.get("id", "<unnamed>")
    summary["first_failing_gate"] = first_failure
    summary["overall"] = "FAIL" if first_failure is not None else "PASS"
    return summary
