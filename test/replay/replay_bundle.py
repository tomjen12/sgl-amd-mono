#!/usr/bin/env python3
"""Load and validate per-rank Muse Spark replay capture bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = frozenset(
    {
        ("cycle", "metadata"),
        ("draft_forward_input", "metadata"),
        ("draft_layers", "metadata"),
        ("draft_forward_output", "metadata"),
        ("target_verify", "metadata"),
        ("target_layers", "metadata"),
        ("cycle_output", "metadata"),
    }
)


@dataclass(frozen=True)
class RankBundle:
    directory: Path
    manifest: dict[str, Any]

    @property
    def rank(self) -> int:
        return int(self.manifest["rank"])

    def artifact_path(self, name: str, kind: str) -> Path:
        matches = [
            artifact
            for artifact in self.manifest["artifacts"]
            if artifact["name"] == name and artifact["kind"] == kind
        ]
        if len(matches) != 1:
            raise ValueError(
                f"rank {self.rank}: expected one {name!r}/{kind!r} artifact, "
                f"found {len(matches)}"
            )
        return self.directory / matches[0]["path"]

    def metadata(self, name: str) -> dict[str, Any]:
        with self.artifact_path(name, "metadata").open(encoding="utf-8") as handle:
            return json.load(handle)

    def tensors(self, name: str, *, device: str = "cpu"):
        from safetensors.torch import load_file

        return load_file(str(self.artifact_path(name, "tensors")), device=device)


@dataclass(frozen=True)
class ReplayBundle:
    root: Path
    step: int
    ranks: tuple[RankBundle, ...]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        step: int = 0,
        expected_ranks: int = 4,
    ) -> "ReplayBundle":
        root = Path(root)
        step_directory = root / f"step_{step:06d}"
        ranks = []
        for rank in range(expected_ranks):
            directory = step_directory / f"rank_{rank:04d}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing replay manifest: {manifest_path}")
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            ranks.append(RankBundle(directory=directory, manifest=manifest))
        bundle = cls(root=root, step=step, ranks=tuple(ranks))
        bundle.validate()
        return bundle

    def validate(self) -> None:
        for expected_rank, rank_bundle in enumerate(self.ranks):
            manifest = rank_bundle.manifest
            if manifest.get("schema_version") != 1:
                raise ValueError(
                    f"rank {expected_rank}: unsupported schema "
                    f"{manifest.get('schema_version')!r}"
                )
            if manifest.get("rank") != expected_rank:
                raise ValueError(
                    f"rank directory {expected_rank} contains rank "
                    f"{manifest.get('rank')!r}"
                )
            if manifest.get("step") != self.step:
                raise ValueError(
                    f"rank {expected_rank}: expected step {self.step}, "
                    f"got {manifest.get('step')!r}"
                )
            if manifest.get("complete") is not True:
                raise ValueError(f"rank {expected_rank}: capture is incomplete")
            artifacts = {
                (artifact.get("name"), artifact.get("kind"))
                for artifact in manifest.get("artifacts", ())
            }
            tensor_artifacts = sorted(
                name for name, kind in artifacts if kind == "tensors"
            )
            if tensor_artifacts:
                raise ValueError(
                    f"rank {expected_rank}: lightweight capture contains tensor "
                    f"values: {tensor_artifacts}"
                )
            missing = REQUIRED_ARTIFACTS - artifacts
            if missing:
                raise ValueError(
                    f"rank {expected_rank}: missing artifacts {sorted(missing)}"
                )
            for artifact in manifest["artifacts"]:
                path = rank_bundle.directory / artifact["path"]
                if not path.is_file():
                    raise FileNotFoundError(
                        f"rank {expected_rank}: missing artifact file {path}"
                    )

        cycles = [rank.metadata("cycle") for rank in self.ranks]
        for rank, cycle in enumerate(cycles):
            if cycle.get("width") != 8:
                raise ValueError(f"rank {rank}: expected width 8, got {cycle.get('width')}")
            if cycle.get("draft_tp_size") != 4:
                raise ValueError(
                    f"rank {rank}: expected draft TP4, got "
                    f"{cycle.get('draft_tp_size')!r}"
                )
            if cycle.get("target_tp_rank") != rank:
                raise ValueError(
                    f"rank {rank}: target_tp_rank={cycle.get('target_tp_rank')!r}"
                )
        common = ("width", "batch_size", "draft_tp_size", "replay_target_weights")
        baseline = {key: cycles[0].get(key) for key in common}
        for rank, cycle in enumerate(cycles[1:], start=1):
            current = {key: cycle.get(key) for key in common}
            if current != baseline:
                raise ValueError(
                    f"rank {rank}: cycle metadata differs: {current} != {baseline}"
                )

    def summary(self) -> dict[str, Any]:
        cycle = self.ranks[0].metadata("cycle")
        return {
            "root": str(self.root),
            "step": self.step,
            "ranks": len(self.ranks),
            "width": cycle["width"],
            "batch_size": cycle["batch_size"],
            "draft_tp_size": cycle["draft_tp_size"],
            "replay_target_weights": cycle["replay_target_weights"],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--expected-ranks", type=int, default=4)
    args = parser.parse_args()
    bundle = ReplayBundle.load(
        args.bundle, step=args.step, expected_ranks=args.expected_ranks
    )
    print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
