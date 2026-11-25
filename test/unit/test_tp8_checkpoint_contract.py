# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_checkpoint_contract import (  # noqa: E402
    PACKED_ALIGNMENT,
    SOURCE_TENSORS,
    CheckpointContractError,
    build_checkpoint_runtime_manifest,
    build_checkpoint_source_manifest,
    ownership_manifest,
    validate_ownership_manifest,
    validate_packed_metadata,
    validate_source_metadata,
)


def _write_checkpoint_fixture(root: Path, *, input_dynamic: bool = True) -> None:
    quant = {
        "dtype": "fp4",
        "group_size": 32,
        "scale_format": "e8m0",
        "qscheme": "per_group",
        "round_method": "half_even",
    }
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 62,
                    "num_local_experts": 256,
                },
                "quantization_config": {
                    "global_quant_config": {
                        "weight": {**quant, "is_dynamic": False},
                        "input_tensors": {**quant, "is_dynamic": input_dynamic},
                    }
                },
            }
        )
    )
    header = {}
    offset = 0
    for layer in (0, 1, 60, 61):
        for expert in (0, 255):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts."
                    f"{expert}.{projection}"
                )
                shape = [4096, 4096] if layer in (0, 61) else [4096, 2048]
                size = shape[0] * shape[1] * (2 if layer in (0, 61) else 1)
                header[f"{prefix}.weight"] = {
                    "dtype": "BF16" if layer in (0, 61) else "U8",
                    "shape": shape,
                    "data_offsets": [offset, offset + size],
                }
                offset += size
                if layer not in (0, 61):
                    scale_size = 4096 * 128
                    header[f"{prefix}.weight_scale"] = {
                        "dtype": "U8",
                        "shape": [4096, 128],
                        "data_offsets": [offset, offset + scale_size],
                    }
                    offset += scale_size
    raw = json.dumps(header, separators=(",", ":")).encode()
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": offset},
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in header
                },
            }
        )
    )


def _source():
    return {
        name: {**expected, "contiguous": True}
        for name, expected in SOURCE_TENSORS.items()
    }


def _packed():
    return {
        "w13": {
            "dtype": "float4_e2m1fn_x2",
            "is_shuffled": True,
            "address": 4 * PACKED_ALIGNMENT,
            "layout": "gate-up-row-interleaved+16x16-byte-tile-shuffled",
        },
        "w2": {
            "dtype": "float4_e2m1fn_x2",
            "is_shuffled": True,
            "address": 8 * PACKED_ALIGNMENT,
            "layout": "16x16-byte-tile-shuffled",
        },
        "s13": {
            "dtype": "uint8",
            "address": 12 * PACKED_ALIGNMENT,
            "layout": "n32k4",
        },
        "s2": {
            "dtype": "uint8",
            "address": 16 * PACKED_ALIGNMENT,
            "layout": "n32k4",
        },
    }


def _mapping():
    return [
        {
            "global_expert": expert,
            "owner": expert // 32,
            "local_expert": expert % 32,
        }
        for expert in range(256)
    ]


def test_checkpoint_source_and_packed_metadata_contracts():
    validate_source_metadata(_source())
    validate_packed_metadata(_packed())


def test_real_checkpoint_source_manifest_reads_headers_without_payload(tmp_path):
    _write_checkpoint_fixture(tmp_path)
    manifest = build_checkpoint_source_manifest(tmp_path)
    assert manifest["status"] == "passed"
    assert manifest["shard_count"] == 1
    assert len(manifest["representative_tensors"]) == 36
    assert manifest["runtime_packed_alignment_observed"] is False
    assert len(manifest["manifest_hash"]) == 64


def test_checkpoint_source_manifest_rejects_quantization_drift(tmp_path):
    _write_checkpoint_fixture(tmp_path, input_dynamic=False)
    with pytest.raises(CheckpointContractError, match="input quant must be dynamic"):
        build_checkpoint_source_manifest(tmp_path)


def test_checkpoint_runtime_manifest_joins_real_source_and_packing(tmp_path):
    _write_checkpoint_fixture(tmp_path)
    source = build_checkpoint_source_manifest(tmp_path)
    packed = {
        "observations": 240,
        "all_addresses_256_aligned": True,
        "shapes": {"w13": [[64, 1]], "w2": [[64, 1]], "s13": [[64, 1]], "s2": [[64, 1]]},
        "weight_dtype": "torch.float4_e2m1fn_x2",
        "scale_dtype": "torch.uint8",
        "weights_shuffled": True,
    }
    manifest = build_checkpoint_runtime_manifest(source, packed)
    assert manifest["status"] == "passed"
    assert manifest["runtime_scope"] == "Mono-off TP4; packing semantics only"
    assert manifest["tp8_shape_or_performance_evidence"] is False
    assert manifest["tp8_ownership_observed"] is False


def test_checkpoint_runtime_manifest_rejects_unaligned_runtime_packing(tmp_path):
    _write_checkpoint_fixture(tmp_path)
    source = build_checkpoint_source_manifest(tmp_path)
    with pytest.raises(CheckpointContractError, match="256-byte"):
        build_checkpoint_runtime_manifest(
            source,
            {
                "observations": 240,
                "all_addresses_256_aligned": False,
                "weight_dtype": "torch.float4_e2m1fn_x2",
                "scale_dtype": "torch.uint8",
                "weights_shuffled": True,
            },
        )


@pytest.mark.parametrize(
    ("tensor", "field", "bad", "message"),
    [
        ("w13", "shape", (32, 1, 1), "shape"),
        ("s13", "dtype", "float32", "dtype"),
        ("w2", "contiguous", False, "contiguous"),
    ],
)
def test_checkpoint_source_metadata_rejects_drift(tensor, field, bad, message):
    metadata = _source()
    metadata[tensor][field] = bad
    with pytest.raises(CheckpointContractError, match=message):
        validate_source_metadata(metadata)


@pytest.mark.parametrize(
    ("tensor", "field", "bad", "message"),
    [
        ("w13", "dtype", "uint8", "dtype"),
        ("w2", "is_shuffled", False, "shuffled"),
        ("s13", "address", 257, "alignment"),
        ("s2", "layout", "row-major", "layout"),
    ],
)
def test_packed_metadata_rejects_layout_quant_and_alignment_drift(
    tensor, field, bad, message
):
    metadata = _packed()
    metadata[tensor][field] = bad
    with pytest.raises(CheckpointContractError, match=message):
        validate_packed_metadata(metadata)


def test_runtime_ownership_manifest_is_complete_and_hashed():
    manifest = ownership_manifest(_mapping())
    validate_ownership_manifest(manifest)
    assert manifest["source"] == "production-runtime-capture"
    assert len(manifest["entries"]) == 256
    assert len(manifest["mapping_hash"]) == 64


def test_ownership_manifest_rejects_non_runtime_source_and_tampering():
    manifest = ownership_manifest(_mapping())
    manifest["source"] = "synthetic-contiguous-assumption"
    with pytest.raises(CheckpointContractError, match="runtime-derived"):
        validate_ownership_manifest(manifest)
    manifest = ownership_manifest(_mapping())
    manifest["mapping_hash"] = "0" * 64
    with pytest.raises(CheckpointContractError, match="hash"):
        validate_ownership_manifest(manifest)
