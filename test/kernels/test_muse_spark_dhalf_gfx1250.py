"""Fixed-seed gfx1250 width-8 DFlash2 D-pre/D-post correctness tests."""

import pytest
import torch

LAYER = 6
SLOTS = 64
MAX_POS = 256

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)


def _epoch(device, seconds=600.0):
    from sgl_amd_mono.common import guard

    value = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    value[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    value[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return value


def _conv(x, delta, base):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        BLOCK,
        CONV_GROUP,
        CONV_GROUPS,
    )

    rows = x.shape[0]
    grouped = x.float().reshape(rows, CONV_GROUPS, CONV_GROUP)
    coefficients = base.float().reshape(1, 2, CONV_GROUPS, CONV_GROUP) + delta.float().unsqueeze(-1)
    previous = torch.cat([torch.zeros_like(grouped[:1]), grouped[:-1]], dim=0)
    live = (torch.arange(rows, device=x.device) % BLOCK != 0).view(-1, 1, 1)
    return (
        coefficients[:, 0] * grouped
        + coefficients[:, 1] * torch.where(live, previous, 0)
    ).flatten(-2)


def _rope_norm(x, weight, cos_sin, positions):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import HEAD_DIM, RMS_EPS

    heads = x.float().reshape(x.shape[0], -1, HEAD_DIM)
    normalized = (
        heads
        * torch.rsqrt(heads.square().mean(-1, keepdim=True) + RMS_EPS)
        * weight.float()
    ).to(torch.bfloat16).float()
    half = HEAD_DIM // 2
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    lo, hi = normalized[..., :half], normalized[..., half:]
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1).to(torch.bfloat16)


def _weights(device, seed=1250):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        HEAD_DIM,
        HIDDEN,
        QKV_ROWS,
        Q_K,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    bf = torch.bfloat16
    base = (0.03 * torch.randn(2, 2, HIDDEN, device=device, generator=generator)).to(bf)
    base[:, 0] += 1
    return {
        "w_in": (1 + 0.05 * torch.randn(HIDDEN, device=device, generator=generator)).to(bf),
        "w_kp": (0.01 * torch.randn(COEF_ROWS, HIDDEN, device=device, generator=generator)).to(bf),
        "base": base,
        "w_qkv": (0.015 * torch.randn(QKV_ROWS, HIDDEN, device=device, generator=generator)).to(bf),
        "q_norm": (1 + 0.05 * torch.randn(HEAD_DIM, device=device, generator=generator)).to(bf),
        "k_norm": (1 + 0.05 * torch.randn(HEAD_DIM, device=device, generator=generator)).to(bf),
        "w_o": (0.015 * torch.randn(HIDDEN, Q_K, device=device, generator=generator)).to(bf),
    }


def _pre_reference(x, res, first, weights, cos_sin, positions):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        CONV_GROUPS,
        HIDDEN,
        KV_K,
        Q_K,
        RMS_EPS,
    )

    res2 = x if first else (res.float() + x.float()).to(torch.bfloat16)
    h = (
        res2.float()
        * torch.rsqrt(res2.float().square().mean(-1, keepdim=True) + RMS_EPS)
        * weights["w_in"].float()
    ).to(torch.bfloat16)
    coef = h.float() @ weights["w_kp"].float().T
    delta = coef.reshape(x.shape[0], 2, 2, CONV_GROUPS)
    hc = _conv(h, delta[:, 0], weights["base"][0]).to(torch.bfloat16)
    qkv = (hc.float() @ weights["w_qkv"].float().T).to(torch.bfloat16)
    q, k, v = qkv[:, :Q_K], qkv[:, Q_K : Q_K + KV_K], qkv[:, Q_K + KV_K :]
    q = _rope_norm(q, weights["q_norm"], cos_sin, positions).reshape(x.shape[0], Q_K)
    k = _rope_norm(k, weights["k_norm"], cos_sin, positions).reshape(x.shape[0], KV_K)
    return res2, q, k, v, coef


def _close(got, expected, what, rtol=1 / 16, atol=0.04, bad_max=0.02, cosine=0.998):
    error = (got.float() - expected.float()).abs()
    bad = (error > expected.float().abs() * rtol + atol).float().mean().item()
    similarity = torch.nn.functional.cosine_similarity(
        got.float().reshape(got.shape[0], -1), expected.float().reshape(expected.shape[0], -1), dim=-1
    )
    assert bad < bad_max, f"{what}: {bad:.2%} outside tolerance"
    assert similarity.min().item() > cosine, f"{what}: cosine {similarity.min().item():.6f}"


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_gfx1250_dpre_matches_torch(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as D
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        HEAD_DIM,
        HIDDEN,
        LOCAL_KV_HEADS,
        Q_K,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1251)
    x = (0.5 * torch.randn(rows, HIDDEN, device=device, generator=generator)).to(torch.bfloat16)
    res = torch.randn(rows, HIDDEN, device=device, generator=generator).to(torch.bfloat16)
    weights = _weights(device)
    positions = torch.randint(0, MAX_POS, (rows,), device=device, generator=generator, dtype=torch.int32)
    angle = torch.arange(MAX_POS, device=device)[:, None] * (
        500000.0 ** (-torch.arange(HEAD_DIM // 2, device=device) / (HEAD_DIM // 2))
    )[None, :]
    cos_sin = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
    slots = torch.randperm(SLOTS, device=device, generator=generator)[:rows].to(torch.int32)
    k_cache = torch.zeros(SLOTS, LOCAL_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    res_out = torch.empty_like(x)
    q = torch.empty(rows, Q_K, device=device, dtype=torch.bfloat16)
    coef = torch.empty(rows, COEF_ROWS, device=device)
    scratch = torch.zeros(D.dp_scratch_bytes(), dtype=torch.uint8, device=device)
    epoch = _epoch(device)
    launch = D.build_dpre(D.DPreBuild(tokens=rows))
    launch(
        x.data_ptr(), res.data_ptr(), 0, weights["w_in"].data_ptr(),
        weights["w_kp"].data_ptr(), weights["base"].data_ptr(),
        weights["w_qkv"].data_ptr(), weights["q_norm"].data_ptr(),
        weights["k_norm"].data_ptr(), cos_sin.data_ptr(), positions.data_ptr(), 0,
        slots.data_ptr(), 0, k_cache.data_ptr(), v_cache.data_ptr(),
        res_out.data_ptr(), q.data_ptr(), coef.data_ptr(), scratch.data_ptr(),
        epoch.data_ptr(), LAYER, stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist() == [0, 0, 0]
    expected = _pre_reference(x, res, False, weights, cos_sin, positions)
    assert torch.equal(res_out, expected[0])
    _close(q, expected[1], "q", bad_max=0.08, cosine=0.995)
    _close(k_cache[slots.long()], expected[2].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM), "fp8 k", rtol=1 / 8, atol=0.06, cosine=0.995)
    _close(v_cache[slots.long()], expected[3].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM), "fp8 v", rtol=1 / 8, atol=0.06, cosine=0.995)
    # The scalar fixed-order dot intentionally differs from rocBLAS's reduction
    # tree; the downstream q/k/v comparisons constrain the projected values too.
    _close(
        coef,
        expected[4],
        "coef",
        rtol=1 / 8,
        atol=0.05,
        bad_max=0.15,
        cosine=0.995,
    )


def _post_reference(outputs, weights, coef):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import CONV_GROUPS

    reduced = (outputs[0].float() @ weights[0]["w_o"].float().T).to(torch.bfloat16).float()
    for rank in range(1, len(outputs)):
        partial = (outputs[rank].float() @ weights[rank]["w_o"].float().T).to(torch.bfloat16)
        reduced = reduced + partial.float()
    delta = coef.reshape(coef.shape[0], 2, 2, CONV_GROUPS)
    return _conv(reduced, delta[:, 1], weights[0]["base"][1]).to(torch.bfloat16)


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_gfx1250_dpost_tp1_matches_torch(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as D
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import COEF_ROWS, Q_K

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1252)
    output = (0.5 * torch.randn(rows, Q_K, device=device, generator=generator)).to(torch.bfloat16)
    weights = _weights(device)
    coef = (0.1 * torch.randn(rows, COEF_ROWS, device=device, generator=generator))
    expected = _post_reference([output], [weights], coef)
    got = torch.empty_like(expected)
    scratch = torch.zeros(D.do_scratch_bytes(), dtype=torch.uint8, device=device)
    peers = PeerBuffer(D.do_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device)
    peers.bytes.zero_()
    epoch = _epoch(device)
    try:
        D.build_dpost(D.DPostBuild(tokens=rows, npes=1))(
            output.data_ptr(), weights["w_o"].data_ptr(), coef.data_ptr(),
            weights["base"].data_ptr(), got.data_ptr(), scratch.data_ptr(),
            peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist() == [0, 0, 0]
        _close(got, expected, "TP1 D-post")
    finally:
        peers.close()
