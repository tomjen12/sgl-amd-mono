#!/usr/bin/env python3
"""Build the config-only Target 0-9 model used by width-8 replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

TARGET_LAYERS = 10
PER_LAYER_FIELDS = ("layer_types", "mlp_layer_types", "layer_rope_theta")


def deterministic_seed(base_seed: int, rank: int, name: str) -> int:
    payload = f"{base_seed}:{rank}:{name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def truncated_target_config(
    source: dict[str, Any], *, layers: int = TARGET_LAYERS
) -> dict[str, Any]:
    """Preserve the checkpoint schedule while removing layers after ``layers``."""
    result = copy.deepcopy(source)
    text = result.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Muse Spark config is missing text_config.")
    original_layers = int(text.get("num_hidden_layers", 0))
    if not 0 < layers <= original_layers:
        raise ValueError(
            f"Requested {layers} layers from a {original_layers}-layer config."
        )
    for field in PER_LAYER_FIELDS:
        values = text.get(field)
        if not isinstance(values, list) or len(values) != original_layers:
            raise ValueError(
                f"text_config.{field} must contain {original_layers} entries."
            )
        text[field] = values[:layers]
    text["num_hidden_layers"] = layers
    result["_muse_replay"] = {
        "source_num_hidden_layers": original_layers,
        "target_layers": list(range(layers)),
        "target_weights": "synthetic_seeded",
    }
    return result


def dflash_target_layer_ids(draft_config: dict[str, Any]) -> tuple[int, ...]:
    values = draft_config.get("dflash_config", {}).get("target_layer_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("DFlash2 config has no target_layer_ids.")
    return tuple(int(value) for value in values)


def replay_scope_policy(
    draft_config: dict[str, Any], *, target_layers: int = TARGET_LAYERS
) -> dict[str, Any]:
    """Explain whether a truncated Target can run a complete DFlash cycle."""
    taps = dflash_target_layer_ids(draft_config)
    out_of_range = tuple(tap for tap in taps if tap >= target_layers)
    return {
        "draft_target_layer_ids": list(taps),
        "target_layers": list(range(target_layers)),
        "full_cycle_supported": not out_of_range,
        "out_of_range_taps": list(out_of_range),
        "validation_scopes": (
            ["full_speculative_cycle"]
            if not out_of_range
            else ["draft/layer_0..4", f"target/layer_0..{target_layers - 1}"]
        ),
    }


def build_config_directory(
    target_checkpoint: str | Path,
    draft_checkpoint: str | Path,
    output: str | Path,
    *,
    layers: int = TARGET_LAYERS,
) -> dict[str, Any]:
    target_checkpoint = Path(target_checkpoint)
    draft_checkpoint = Path(draft_checkpoint)
    output = Path(output)
    target_config = json.loads(
        (target_checkpoint / "config.json").read_text(encoding="utf-8")
    )
    draft_config = json.loads(
        (draft_checkpoint / "config.json").read_text(encoding="utf-8")
    )
    truncated = truncated_target_config(target_config, layers=layers)
    policy = replay_scope_policy(draft_config, target_layers=layers)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(truncated, indent=2) + "\n", encoding="utf-8"
    )
    (output / "replay_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard_files = list(output.glob("*.safetensors"))
    if shard_files:
        raise RuntimeError(
            f"Synthetic config directory must not contain Target shards: {shard_files}"
        )
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_checkpoint")
    parser.add_argument("draft_checkpoint")
    parser.add_argument("output")
    parser.add_argument("--layers", type=int, default=TARGET_LAYERS)
    args = parser.parse_args()
    policy = build_config_directory(
        args.target_checkpoint,
        args.draft_checkpoint,
        args.output,
        layers=args.layers,
    )
    print(json.dumps(policy, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
