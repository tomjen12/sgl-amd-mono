import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("SGLANG_AMD_MONO_ARCH", "gfx1250")
os.environ.setdefault("SGLANG_AMD_MONO_BLOCKS", "96")

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

from tp8_shadow_protocol import seed_missing_rank_mailboxes, step_tag  # noqa: E402
from tp8_synthetic_contract import ContractError  # noqa: E402


def _buffer():
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import k3_sym_layout

    return torch.full(
        (scratch_bytes(k3_sym_layout(8)) + guard.FLAG_BYTES,),
        0xA5,
        dtype=torch.uint8,
    )


def _region_pairs(buffer, name):
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import k3_sym_layout

    offset, nbytes = k3_sym_layout(8)[name]
    return buffer[offset : offset + nbytes].view(torch.int32).reshape(-1, 2)


def test_shadow_protocol_publishes_current_tag_and_zero_payloads():
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
    )
    from sgl_amd_mono.muse_spark.gfx1250.device import ATTN_COLS
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import POOL_PARTS

    buffer = _buffer()
    counts = seed_missing_rank_mailboxes(buffer, rows=8, step=3, layer=9)
    tag = step_tag(3, 9)
    assert counts["tag"] == tag

    attn = _region_pairs(buffer, "attn")
    source_pairs = MAX_TOKENS * ATTN_COLS // 2
    assert torch.all(attn[4 * source_pairs :, 0] == 0)
    assert torch.all(attn[4 * source_pairs :, 1] == tag)
    assert torch.all(attn[: 4 * source_pairs, 1] != tag)

    lgx = _region_pairs(buffer, "lgx").reshape(
        POOL_PARTS, MAX_TOKENS, EXPERTS, 2
    )
    assert torch.all(lgx[:, :8, 128:, 0] == 0)
    assert torch.all(lgx[:, :8, 128:, 1] == tag)
    assert torch.all(lgx[:, 8:, 128:, 1] != tag)

    poolr = _region_pairs(buffer, "poolr").reshape(MAX_TOKENS, MOE_HIDDEN, 2)
    assert torch.all(poolr[:8, MOE_HIDDEN // 2 :, 1] == tag)
    assert torch.all(poolr[:8, : MOE_HIDDEN // 2, 1] != tag)

    ffn = _region_pairs(buffer, "ffn")
    source_pairs = MAX_TOKENS * MOE_HIDDEN // 2
    assert torch.all(ffn[4 * source_pairs :, 1] == tag)
    assert torch.all(ffn[: 4 * source_pairs, 1] != tag)

    unp = _region_pairs(buffer, "unp").reshape(MAX_TOKENS, HIDDEN // 2, 2)
    assert torch.all(unp[:8, HIDDEN // 4 :, 1] == tag)
    assert torch.all(unp[:8, : HIDDEN // 4, 1] != tag)


def test_shadow_protocol_overwrites_stale_tags_for_the_next_step():
    buffer = _buffer()
    seed_missing_rank_mailboxes(buffer, rows=8, step=0, layer=9)
    stale = step_tag(0, 9)
    seed_missing_rank_mailboxes(buffer, rows=8, step=1, layer=9)
    current = step_tag(1, 9)
    assert current != stale
    for name in ("attn", "lgx", "poolr", "ffn", "unp"):
        tags = _region_pairs(buffer, name)[:, 1]
        assert torch.any(tags == current), name


@pytest.mark.parametrize("active", [1, 2, 4])
def test_shadow_protocol_supports_active_one_two_or_four(active):
    from sgl_amd_mono.muse_spark.gfx1250.config import MAX_TOKENS
    from sgl_amd_mono.muse_spark.gfx1250.device import ATTN_COLS

    buffer = _buffer()
    tag = step_tag(1, 9)
    seed_missing_rank_mailboxes(
        buffer,
        rows=8,
        step=1,
        layer=9,
        active_npes=active,
    )
    attn = _region_pairs(buffer, "attn")
    source_pairs = MAX_TOKENS * ATTN_COLS // 2
    assert torch.all(attn[active * source_pairs :, 0] == 0)
    assert torch.all(attn[active * source_pairs :, 1] == tag)
    assert torch.all(attn[: active * source_pairs, 1] != tag)


def test_shadow_protocol_publishes_router_fixture_bits():
    from sgl_amd_mono.muse_spark.gfx1250.config import EXPERTS
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import POOL_PARTS

    rows = 8
    missing = EXPERTS // 2
    fixture = torch.arange(
        POOL_PARTS * rows * missing, dtype=torch.float32
    ).reshape(POOL_PARTS, rows, missing)
    buffer = _buffer()
    tag = step_tag(2, 5)
    seed_missing_rank_mailboxes(
        buffer,
        rows=rows,
        step=2,
        layer=5,
        missing_router_partials=fixture,
    )
    lgx = _region_pairs(buffer, "lgx").reshape(
        POOL_PARTS, -1, EXPERTS, 2
    )
    payload = lgx[:, :rows, missing:, 0].contiguous().view(torch.float32)
    assert torch.equal(payload, fixture)
    assert torch.all(lgx[:, :rows, missing:, 1] == tag)


def test_shadow_protocol_rejects_wrong_router_fixture_shape():
    with pytest.raises(ContractError, match="lgx fixture shape"):
        seed_missing_rank_mailboxes(
            _buffer(),
            rows=8,
            step=0,
            layer=9,
            missing_router_partials=torch.zeros(8, 8, 127),
        )


def test_shadow_protocol_rejects_unimplemented_rsag(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_RSAG", "1")
    with pytest.raises(ContractError, match="does not support RSAG"):
        seed_missing_rank_mailboxes(_buffer(), rows=8, step=0, layer=9)


@pytest.mark.parametrize("rows", [1, 32, 64])
def test_shadow_protocol_rejects_deferred_or_non_verify_widths(rows):
    with pytest.raises(ContractError):
        seed_missing_rank_mailboxes(_buffer(), rows=rows, step=0, layer=9)
