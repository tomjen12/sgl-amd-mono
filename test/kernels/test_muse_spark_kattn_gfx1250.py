"""gfx1250 TP4 variable-width K-attn against exact paged-cache attention."""

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

PAGES = 2048
MAX_BLOCKS = 1024


def _reference(q, k_cache, v_cache, table, keys, q_len, window, scale):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        LOCAL_Q_HEADS,
        Q_K,
    )

    out = torch.zeros_like(q, dtype=torch.float32)
    prefix = keys - q_len
    gqa = LOCAL_Q_HEADS // LOCAL_KV_HEADS
    key_index = torch.arange(keys, device=q.device)
    for request in range(q.shape[0] // q_len):
        slots = (
            table[request, : (keys + 15) // 16].long()[:, None] * 16
            + torch.arange(16, device=q.device)[None, :]
        ).reshape(-1)[:keys]
        k = k_cache[slots].float()
        v = v_cache[slots].float()
        for query_token in range(q_len):
            token = request * q_len + query_token
            end = prefix + query_token + 1
            allowed = key_index < end
            if window:
                allowed &= key_index >= max(0, end - window)
            for head in range(LOCAL_Q_HEADS):
                kv_head = head // gqa
                query = q[token, head * HEAD_DIM : (head + 1) * HEAD_DIM].float()
                score = scale * (query @ k[:, kv_head].T)
                score = torch.where(
                    allowed, score, torch.full_like(score, float("-inf"))
                )
                probability = torch.softmax(score, dim=-1).to(torch.bfloat16).float()
                out[token, head * HEAD_DIM : (head + 1) * HEAD_DIM] = (
                    probability @ v[:, kv_head]
                )
    return out


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "keys,window,zero_q",
    [(300, 0, False), (3000, 2047, False), (8200, 0, True)],
)
def test_gfx1250_kattn_matches_paged_attention(width, keys, window, zero_q):
    if zero_q and width != 8:
        pytest.skip("the 8K uniform-attention regression targets width 8")
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        DRAFT_TOKENS,
        Q_K,
        SOFTMAX_SCALE,
    )
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        KAttnBuild,
        build_tiled_kattn,
        tiled_kattn_scratch_bytes,
    )

    torch.manual_seed(keys + window)
    device = torch.device("cuda")
    q = (torch.randn(width, Q_K, device=device) * 5.47).to(torch.bfloat16)
    if zero_q:
        q.zero_()
    k_cache = (
        torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=device) * 2
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    pages = (keys + 15) // 16
    requests = width // DRAFT_TOKENS
    table = torch.zeros(requests, MAX_BLOCKS, dtype=torch.int32, device=device)
    table[:, :pages] = (
        torch.randperm(PAGES - 1, device=device)[: requests * pages]
        .reshape(requests, pages)
        .to(torch.int32)
        + 1
    )
    prefix_lens = torch.full(
        (requests,), keys - DRAFT_TOKENS, dtype=torch.int32, device=device
    )
    out = torch.full_like(q, float("nan"))
    scale_bits = struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0]

    scratch = torch.zeros(
        tiled_kattn_scratch_bytes(), dtype=torch.uint8, device=device
    )
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    launch = build_tiled_kattn(
        KAttnBuild(tokens=width, q_len=DRAFT_TOKENS)
    )
    launch(
        q.data_ptr(),
        out.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        table.data_ptr(),
        table.stride(0),
        prefix_lens.data_ptr(),
        window,
        scale_bits,
        scratch.data_ptr(),
        epoch.data_ptr(),
        0,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    reference = _reference(
        q, k_cache, v_cache, table, keys, DRAFT_TOKENS, window, SOFTMAX_SCALE
    )
    got = out.float()
    assert not got.isnan().any(), "NaN in K-attn output"
    error = (got - reference).abs()
    tolerance = reference.abs() * (1 / 32) + 2e-2
    bad = (error > tolerance).float().mean().item()
    assert bad == 0.0, (
        f"{bad:.2%} outputs exceed tolerance; max error {error.max().item():.4f}"
    )
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(width, -1, HEAD_DIM),
        reference.reshape(width, -1, HEAD_DIM),
        dim=-1,
    )
    assert cosine.min().item() > 0.999, f"least head cosine {cosine.min().item():.6f}"
