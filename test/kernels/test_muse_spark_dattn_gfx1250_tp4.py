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


def _reference(
    q, k_cache, v_cache, slots, prefix, window, scale, k_new=None, v_new=None
):
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
    if k_new is not None:
        k[-q.shape[0] :] = k_new.float()
    if v_new is not None:
        v[-q.shape[0] :] = v_new.float()
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


@pytest.mark.parametrize("resident_blocks", [96])
@pytest.mark.parametrize("total_rows", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "base_prefix,window",
    [
        (0, 0),
        (8, 0),
        (128, 0),
        (300, 0),
        (1024, 0),
        (2048, 2048),
        (3000, 2048),
    ],
)
@pytest.mark.parametrize("q_scale", [0.4, 2.0])
def test_gfx1250_dflash2_tp4_dattn(
    resident_blocks, total_rows, base_prefix, window, q_scale
):
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn as D

    torch.manual_seed(0xDFA1250)
    device = torch.device("cuda")
    requests = total_rows // D.BLOCK
    q = (torch.randn(total_rows, D.Q_K, device=device) * q_scale).to(
        torch.bfloat16
    )
    k_cache = (
        torch.randn(SLOTS, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=device) * 1.5
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        SLOTS, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)

    prefixes = [max(base_prefix - 7 * request, 0) for request in range(requests)]
    req_to_token = torch.zeros(
        requests + 2, TABLE_STRIDE, dtype=torch.int32, device=device
    )
    req_pool_values = list(reversed(range(1, requests + 1)))
    request_slots = []
    for request, prefix in enumerate(prefixes):
        keys = prefix + D.BLOCK
        slots = torch.randperm(SLOTS - 1, device=device)[:keys] + 1
        request_slots.append(slots)
        req_to_token[req_pool_values[request], :keys] = slots.to(torch.int32)
    req_pool = torch.tensor(req_pool_values, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(prefixes, dtype=torch.int32, device=device)
    out = torch.full_like(q, float("nan"))
    current_slots = torch.cat(
        [slots[-D.BLOCK :] for slots in request_slots]
    )
    k_new = k_cache[current_slots].to(torch.bfloat16)
    v_new = v_cache[current_slots].to(torch.bfloat16)
    scratch = torch.zeros(
        D.dattn_scratch_bytes(), dtype=torch.uint8, device=device
    )
    epoch = _epoch(device)
    scale = D.HEAD_DIM**-0.5
    scale_bits = struct.unpack("i", struct.pack("f", scale))[0]

    launch = D.build_dattn(
        D.DAttnBuild(
            tokens=total_rows,
            q_len=D.BLOCK,
            resident_blocks=resident_blocks,
        )
    )
    launch(
        q.data_ptr(),
        out.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        k_new.data_ptr(),
        v_new.data_ptr(),
        req_to_token.data_ptr(),
        TABLE_STRIDE,
        req_to_token.numel() * req_to_token.element_size(),
        req_pool.data_ptr(),
        1,
        seq_lens.data_ptr(),
        0,
        window,
        scale_bits,
        scratch.data_ptr(),
        epoch.data_ptr(),
        0,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    reference = torch.cat(
        [
            _reference(
                q[request * D.BLOCK : (request + 1) * D.BLOCK],
                k_cache,
                v_cache,
                request_slots[request],
                prefixes[request],
                window,
                scale,
                k_new[
                    request * D.BLOCK : (request + 1) * D.BLOCK
                ],
                v_new[
                    request * D.BLOCK : (request + 1) * D.BLOCK
                ],
            )
            for request in range(requests)
        ]
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
        got.reshape(total_rows, D.LOCAL_Q_HEADS, D.HEAD_DIM),
        want.reshape(total_rows, D.LOCAL_Q_HEADS, D.HEAD_DIM),
        dim=-1,
    )
    assert cosine.min().item() > 0.999
