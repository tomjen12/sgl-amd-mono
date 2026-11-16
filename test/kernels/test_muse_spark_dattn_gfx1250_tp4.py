"""Fixed-seed gfx1250 DFlash2 TP4 D-attn reference test."""

import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

SLOTS = 4096
TABLE_STRIDE = 4096


def _epoch(device):
    from sgl_amd_mono.common import guard

    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(20.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return epoch


def _reference(q, k_cache, v_cache, slots, prefix, window, scale):
    from sgl_amd_mono.muse_spark.gfx1250.draft_attn import (
        GQA,
        HEAD_DIM,
        LOCAL_Q_HEADS,
        Q_K,
    )

    keys = prefix + q.shape[0]
    first = max(0, prefix - window) if window else 0
    selected = slots[first:keys].long()
    k = k_cache[selected].float()
    v = v_cache[selected].float()
    out = torch.empty_like(q)
    for token in range(q.shape[0]):
        queries = q[token].float().reshape(LOCAL_Q_HEADS, HEAD_DIM)
        heads = []
        for query_head in range(LOCAL_Q_HEADS):
            kv_head = query_head // GQA
            scores = scale * (queries[query_head] @ k[:, kv_head].T)
            probability = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
            heads.append(probability @ v[:, kv_head])
        out[token] = torch.stack(heads).reshape(Q_K).to(torch.bfloat16)
    return out


@pytest.mark.parametrize("resident_blocks", [96, 192])
@pytest.mark.parametrize("prefix,window", [(300, 0), (3000, 2048)])
def test_gfx1250_dflash2_tp4_dattn(
    resident_blocks, prefix, window
):
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn as D

    torch.manual_seed(0xDFA1250)
    device = torch.device("cuda")
    q = (torch.randn(D.BLOCK, D.Q_K, device=device) * 0.4).to(torch.bfloat16)
    k_cache = (
        torch.randn(SLOTS, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=device) * 1.5
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        SLOTS, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)

    keys = prefix + D.BLOCK
    token_slots = torch.randperm(SLOTS - 1, device=device)[:keys] + 1
    req_to_token = torch.zeros(
        3, TABLE_STRIDE, dtype=torch.int32, device=device
    )
    request_row = 2
    req_to_token[request_row, :keys] = token_slots.to(torch.int32)
    req_pool = torch.tensor([request_row], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([prefix], dtype=torch.int32, device=device)
    out = torch.full_like(q, float("nan"))
    epoch = _epoch(device)
    scale = D.HEAD_DIM**-0.5
    scale_bits = struct.unpack("i", struct.pack("f", scale))[0]

    launch = D.build_dattn(
        D.DAttnBuild(
            tokens=D.BLOCK,
            q_len=D.BLOCK,
            resident_blocks=resident_blocks,
        )
    )
    launch(
        q.data_ptr(),
        out.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        req_to_token.data_ptr(),
        TABLE_STRIDE,
        req_to_token.numel() * req_to_token.element_size(),
        req_pool.data_ptr(),
        1,
        seq_lens.data_ptr(),
        0,
        window,
        scale_bits,
        epoch.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    reference = _reference(
        q, k_cache, v_cache, token_slots, prefix, window, scale
    )
    got = out.float()
    want = reference.float()
    assert not got.isnan().any(), "NaN in D-attn output"
    error = (got - want).abs()
    tolerance = want.abs() * (1 / 32) + 2e-2
    bad = (error > tolerance).float().mean().item()
    assert bad == 0.0, (
        f"{bad:.2%} outputs exceed tolerance; max error {error.max().item():.4f}"
    )
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(D.BLOCK, D.LOCAL_Q_HEADS, D.HEAD_DIM),
        want.reshape(D.BLOCK, D.LOCAL_Q_HEADS, D.HEAD_DIM),
        dim=-1,
    )
    assert cosine.min().item() > 0.999
