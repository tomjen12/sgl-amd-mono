# SPDX-License-Identifier: Apache-2.0
"""Host-side MI455 checkpoint packing and TP8 ownership evidence contracts."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp8_controlled_routing import GLOBAL_EXPERTS, validate_runtime_mapping

LOCAL_EXPERTS = 32
MOE_HIDDEN = 4096
INTERMEDIATE = 4096
PACKED_ALIGNMENT = 256

SOURCE_TENSORS = {
    "w13": {
        "shape": (LOCAL_EXPERTS, 2 * INTERMEDIATE, MOE_HIDDEN // 2),
        "dtype": "uint8",
    },
    "s13": {
        "shape": (LOCAL_EXPERTS, 2 * INTERMEDIATE, MOE_HIDDEN // 32),
        "dtype": "uint8",
    },
    "w2": {
        "shape": (LOCAL_EXPERTS, MOE_HIDDEN, INTERMEDIATE // 2),
        "dtype": "uint8",
    },
    "s2": {
        "shape": (LOCAL_EXPERTS, MOE_HIDDEN, INTERMEDIATE // 32),
        "dtype": "uint8",
    },
}

PACKING_CONTRACT = {
    "quantization": {
        "format": "mxfp4",
        "weight_elements": "e2m1",
        "scale_elements": "e8m0",
        "block_size": 32,
        "packed_weights": "two-e2m1-per-byte",
    },
    "gfx1250": {
        "w13_layout": "gate-up-row-interleaved+16x16-byte-tile-shuffled",
        "w2_layout": "16x16-byte-tile-shuffled",
        "scale_layout": "n32k4",
        "weight_dtype": "float4_e2m1fn_x2",
        "scale_dtype": "uint8",
        "alignment_bytes": PACKED_ALIGNMENT,
        "source_parameters_dropped": True,
    },
}


class CheckpointContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointContractError(message)


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Read metadata only; tensor payloads are never mapped or loaded."""
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        _require(len(raw_size) == 8, f"{path.name}: truncated safetensors header")
        size = struct.unpack("<Q", raw_size)[0]
        raw_header = stream.read(size)
    _require(len(raw_header) == size, f"{path.name}: truncated safetensors metadata")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise CheckpointContractError(
            f"{path.name}: invalid safetensors metadata"
        ) from exc
    _require(isinstance(header, dict), f"{path.name}: metadata is not an object")
    return 8 + size, header


def build_checkpoint_source_manifest(checkpoint: str | Path) -> dict[str, Any]:
    """Attest representative boundary/middle-layer tensors from a real checkpoint.

    This records on-disk source metadata.  It intentionally does not claim GPU
    packed addresses or a TP8 runtime ownership mapping.
    """
    root = Path(checkpoint)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    _require(config_path.is_file(), "checkpoint config.json is missing")
    _require(index_path.is_file(), "checkpoint safetensors index is missing")
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    try:
        text = config["text_config"]
        quant = config["quantization_config"]["global_quant_config"]
        weight_map = index["weight_map"]
    except (KeyError, TypeError) as exc:
        raise CheckpointContractError("checkpoint config/index schema mismatch") from exc
    _require(text["num_hidden_layers"] == 62, "checkpoint layer count mismatch")
    _require(text["num_local_experts"] == GLOBAL_EXPERTS, "checkpoint expert count mismatch")
    weight_quant = quant["weight"]
    input_quant = quant["input_tensors"]
    expected_quant = {
        "dtype": "fp4",
        "group_size": 32,
        "scale_format": "e8m0",
        "qscheme": "per_group",
        "round_method": "half_even",
    }
    for name, spec in (("weight", weight_quant), ("input", input_quant)):
        for field, expected in expected_quant.items():
            _require(spec.get(field) == expected, f"{name} quant {field} mismatch")
    _require(weight_quant.get("is_dynamic") is False, "weight quant must be static")
    _require(input_quant.get("is_dynamic") is True, "input quant must be dynamic")

    selected: list[str] = []
    for layer in (0, 1, 60, 61):
        for expert in (0, GLOBAL_EXPERTS - 1):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts."
                    f"{expert}.{projection}"
                )
                selected.append(f"{prefix}.weight")
                if layer not in (0, 61):
                    selected.append(f"{prefix}.weight_scale")
    missing = [name for name in selected if name not in weight_map]
    _require(not missing, f"checkpoint tensors missing: {missing[:3]}")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    tensors: dict[str, Any] = {}
    for name in selected:
        shard = str(weight_map[name])
        if shard not in headers:
            shard_path = root / shard
            _require(shard_path.is_file(), f"checkpoint shard missing: {shard}")
            headers[shard] = _safetensors_header(shard_path)
        payload_start, header = headers[shard]
        _require(name in header, f"{shard}: tensor metadata missing for {name}")
        item = header[name]
        begin, end = item["data_offsets"]
        tensors[name] = {
            "shard": shard,
            "dtype": item["dtype"],
            "shape": item["shape"],
            "file_offset": payload_start + begin,
            "storage_bytes": end - begin,
            "file_offset_alignment": (
                (payload_start + begin) & -(payload_start + begin)
            ),
        }
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        if layer in (0, 61):
            _require(item["dtype"] == "BF16", f"{name}: boundary expert dtype mismatch")
            _require(item["shape"] == [4096, 4096], f"{name}: boundary shape mismatch")
        elif name.endswith(".weight_scale"):
            _require(item["dtype"] == "U8", f"{name}: scale dtype mismatch")
            _require(item["shape"] == [4096, 128], f"{name}: scale shape mismatch")
        else:
            _require(item["dtype"] == "U8", f"{name}: packed FP4 dtype mismatch")
            _require(item["shape"] == [4096, 2048], f"{name}: packed FP4 shape mismatch")

    shards = sorted(set(weight_map.values()))
    digest_input = {
        "quantization": expected_quant,
        "tensors": tensors,
        "shards": shards,
    }
    return {
        "schema": "mi455-checkpoint-source-metadata",
        "version": 1,
        "status": "passed",
        "checkpoint": str(root),
        "total_size": index.get("metadata", {}).get("total_size"),
        "shard_count": len(shards),
        "quantization": {
            **expected_quant,
            "weight_dynamic": False,
            "input_dynamic": True,
            "packed_weight_elements": "two-e2m1-per-byte",
        },
        "representative_tensors": tensors,
        "runtime_packed_alignment_observed": False,
        "runtime_ownership_observed": False,
        "manifest_hash": hashlib.sha256(
            json.dumps(digest_input, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    }


def build_checkpoint_runtime_manifest(
    source: Mapping[str, Any], packed: Mapping[str, Any]
) -> dict[str, Any]:
    """Join real checkpoint headers with TP4 production packing observations."""
    _require(source.get("status") == "passed", "checkpoint source manifest did not pass")
    _require(
        packed.get("observations") == 4 * 60,
        "packed metadata does not cover four ranks and 60 MXFP4 layers",
    )
    _require(
        packed.get("all_addresses_256_aligned") is True,
        "runtime packed addresses are not 256-byte aligned",
    )
    _require(
        packed.get("weight_dtype") == "torch.float4_e2m1fn_x2",
        "runtime packed weight dtype mismatch",
    )
    _require(
        packed.get("scale_dtype") == "torch.uint8",
        "runtime packed scale dtype mismatch",
    )
    _require(packed.get("weights_shuffled") is True, "runtime weights are not shuffled")
    digest_input = {"source_hash": source["manifest_hash"], "packed": packed}
    return {
        "schema": "mi455-checkpoint-runtime-packing",
        "version": 1,
        "status": "passed",
        "source_manifest_hash": source["manifest_hash"],
        "source_quantization": source["quantization"],
        "runtime_packing": dict(packed),
        "packing_contract": PACKING_CONTRACT,
        "runtime_scope": "Mono-off TP4; packing semantics only",
        "tp8_shape_or_performance_evidence": False,
        "tp8_ownership_observed": False,
        "manifest_hash": hashlib.sha256(
            json.dumps(digest_input, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    }


def validate_source_metadata(metadata: Mapping[str, Mapping[str, Any]]) -> None:
    _require(set(metadata) == set(SOURCE_TENSORS), "checkpoint tensor set mismatch")
    for name, expected in SOURCE_TENSORS.items():
        actual = metadata[name]
        _require(tuple(actual.get("shape", ())) == expected["shape"], f"{name} shape mismatch")
        _require(actual.get("dtype") == expected["dtype"], f"{name} dtype mismatch")
        _require(actual.get("contiguous") is True, f"{name} must be contiguous before packing")


def validate_packed_metadata(metadata: Mapping[str, Mapping[str, Any]]) -> None:
    _require(set(metadata) == set(SOURCE_TENSORS), "packed tensor set mismatch")
    for name in ("w13", "w2"):
        item = metadata[name]
        _require(item.get("dtype") == "float4_e2m1fn_x2", f"{name} packed dtype mismatch")
        _require(item.get("is_shuffled") is True, f"{name} shuffled marker missing")
        _require(int(item.get("address", -1)) % PACKED_ALIGNMENT == 0, f"{name} alignment mismatch")
    for name in ("s13", "s2"):
        item = metadata[name]
        _require(item.get("dtype") == "uint8", f"{name} packed scale dtype mismatch")
        _require(int(item.get("address", -1)) % PACKED_ALIGNMENT == 0, f"{name} alignment mismatch")
    _require(metadata["w13"].get("layout") == PACKING_CONTRACT["gfx1250"]["w13_layout"], "w13 layout mismatch")
    _require(metadata["w2"].get("layout") == PACKING_CONTRACT["gfx1250"]["w2_layout"], "w2 layout mismatch")
    _require(metadata["s13"].get("layout") == "n32k4", "s13 scale layout mismatch")
    _require(metadata["s2"].get("layout") == "n32k4", "s2 scale layout mismatch")


def ownership_manifest(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a hash only from a complete mapping captured from production runtime."""
    locations = validate_runtime_mapping(entries, world_size=8)
    canonical = [
        {
            "global_expert": item.global_expert,
            "owner": item.owner,
            "local_expert": item.local_expert,
        }
        for item in locations
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": "mi455-logical-tp8-expert-ownership",
        "version": 1,
        "source": "production-runtime-capture",
        "logical_tp": 8,
        "logical_ep": 8,
        "global_experts": GLOBAL_EXPERTS,
        "entries": canonical,
        "mapping_hash": digest,
    }


def validate_ownership_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema") == "mi455-logical-tp8-expert-ownership", "ownership schema mismatch")
    _require(manifest.get("version") == 1, "ownership version mismatch")
    _require(manifest.get("source") == "production-runtime-capture", "ownership is not runtime-derived")
    rebuilt = ownership_manifest(manifest.get("entries", ()))
    _require(manifest.get("mapping_hash") == rebuilt["mapping_hash"], "ownership mapping hash mismatch")
