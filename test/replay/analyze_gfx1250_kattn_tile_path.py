#!/usr/bin/env python3
"""Count one five-tile CTA path in Baseline and FlyDSL gfx1250 ISA."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _instructions(path: Path) -> list[tuple[str | None, str]]:
    label = None
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(".LBB") and ":" in line:
            label = line.split(":", 1)[0]
            continue
        if not line or line.startswith((".", ";")):
            continue
        mnemonic = line.split()[0]
        if re.match(r"^[sv]_", mnemonic) or mnemonic.startswith(
            ("buffer_", "global_", "flat_", "ds_")
        ):
            rows.append((label, mnemonic))
    return rows


def _between(rows, start, end):
    active = False
    selected = []
    for label, mnemonic in rows:
        if label == start:
            active = True
        if active:
            selected.append(mnemonic)
        if label == end:
            break
    return selected


def _classify(mnemonic: str) -> str:
    if "wmma" in mnemonic:
        return "wmma"
    if mnemonic.startswith(("v_exp", "v_log", "v_rcp", "v_sqrt")):
        return "transcendental"
    if mnemonic.startswith("ds_"):
        return "lds"
    if mnemonic.startswith(("buffer_", "global_", "flat_")):
        return "vmem"
    if mnemonic.startswith("s_wait"):
        return "wait"
    if mnemonic.startswith("s_delay"):
        return "delay"
    if mnemonic.startswith(("s_branch", "s_cbranch")):
        return "branch"
    if mnemonic.startswith("v_"):
        return "valu"
    return "salu"


def _address_op(mnemonic: str) -> bool:
    return any(
        token in mnemonic
        for token in ("add", "sub", "mul", "lshl", "lshr", "ashr", "or_b32")
    )


def _count(parts):
    instructions = []
    for values, multiplier in parts:
        instructions.extend(values * multiplier)
    categories = Counter(_classify(value) for value in instructions)
    categories["address_arithmetic"] = sum(
        _address_op(value) for value in instructions
    )
    categories["total"] = len(instructions)
    return dict(sorted(categories.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    baseline = _instructions(args.baseline)
    candidate = _instructions(args.candidate)

    # Representative split has five KV tiles. Triton runs four pipelined loop
    # iterations plus one tail tile; FlyDSL runs one common body five times and
    # its next-tile issue path four times.
    result = {
        "tiles": 5,
        "baseline": _count(
            [
                (_between(baseline, ".LBB0_2", ".LBB0_4"), 4),
                (_between(baseline, ".LBB0_5", ".LBB0_9"), 1),
            ]
        ),
        "flydsl": _count(
            [
                (_between(candidate, ".LBB0_7", ".LBB0_9"), 4),
                (_between(candidate, ".LBB0_9", ".LBB0_11"), 5),
                (_between(candidate, ".LBB0_6", ".LBB0_7"), 5),
            ]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
