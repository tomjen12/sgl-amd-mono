"""Single-rank numerical test for the correctness-first gfx1250 K2."""

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

LAYER = 7


def _epoch(device, seconds=30.0):
    from sgl_amd_mono.common import guard

    out = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    out[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    out[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return out


def _rms(value, eps, weight=None):
    value = value.float()
    out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(torch.bfloat16)


def _reference(o, g, res, weights, partials=None):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        POST_EPS,
        RMS_EPS,
        ROUTE_EPS,
        ROUTE_SCALE,
        ROUTER_MULT,
        TOP_K,
    )

    rows = o.shape[0]
    of = o.float().reshape(rows, -1, HEAD_DIM)
    og = (
        of
        * torch.rsqrt(of.square().mean(-1, keepdim=True) + RMS_EPS)
        * torch.sigmoid(g.float().reshape(rows, -1, HEAD_DIM))
    ).reshape(rows, -1).to(torch.bfloat16)
    if partials is None:
        partials = [(og.float() @ weights["w_o"].float().T).to(torch.bfloat16)]
    ao = partials[0].float()
    for partial in partials[1:]:
        ao = ao + partial.float()
    ao = ao.to(torch.bfloat16)
    branch = _rms(ao, POST_EPS).float()
    new = weights["alpha"] * res.float() + weights["beta"] * branch
    x = _rms(new, RMS_EPS, weights["pre_w"])
    logits = (
        x.float() @ weights["w_hi"].float().T
        + x.float() @ weights["w_lo"].float().T
    )
    sigmoid = torch.sigmoid(logits * ROUTER_MULT)
    order = torch.argsort(
        sigmoid + weights["bias"], dim=-1, descending=True, stable=True
    )
    ids = order[:, :TOP_K]
    picked = sigmoid.gather(1, ids)
    scores = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = (x.float() @ weights["w_pool"].float().T).to(torch.bfloat16)
    xin = _rms(pooled, RMS_EPS, weights["xn_w"])
    return new, ids.to(torch.int32), scores, xin


def _inputs(device, seed=1252):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HIDDEN,
        DRAFT_TOKENS as MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
    )

    torch.manual_seed(seed)
    bf = torch.bfloat16
    router = torch.randn(EXPERTS, HIDDEN, device=device) * 0.02
    w_hi = router.to(bf)
    return (
        torch.randn(MAX_TOKENS, Q_K, device=device).to(bf),
        torch.randn(MAX_TOKENS, Q_K, device=device).to(bf),
        torch.randn(MAX_TOKENS, HIDDEN, device=device),
        {
            "w_o": (torch.randn(HIDDEN, Q_K, device=device) * 0.02).to(bf),
            "alpha": torch.rand(HIDDEN, device=device) * 0.5 + 0.5,
            "beta": torch.rand(HIDDEN, device=device) * 0.5,
            "pre_w": (1 + 0.1 * torch.randn(HIDDEN, device=device)).to(bf),
            "w_hi": w_hi,
            "w_lo": (router - w_hi.float()).to(bf),
            "bias": torch.randn(EXPERTS, device=device) * 0.01,
            "w_pool": (
                torch.randn(MOE_HIDDEN, HIDDEN, device=device) * 0.02
            ).to(bf),
            "xn_w": (1 + 0.1 * torch.randn(MOE_HIDDEN, device=device)).to(bf),
        },
    )


def _assert_close(got, expected, rtol, atol, label):
    error = (got.float() - expected.float()).abs()
    assert (error <= expected.float().abs() * rtol + atol).all(), (
        f"{label}: max error {error.max().item()}"
    )


def test_gfx1250_k2_matches_explicit_reference_for_two_epochs():
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import (
        K2Build,
        build_k2,
        scratch_bytes,
        sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        DRAFT_TOKENS as MAX_TOKENS,
        MOE_HIDDEN,
        TOP_K,
    )

    device = torch.device("cuda")
    o, g, stream, w = _inputs(device)
    reference = _reference(o, g, stream, w)
    ids = torch.full((MAX_TOKENS, TOP_K), -1, dtype=torch.int32, device=device)
    scores = torch.full((MAX_TOKENS, TOP_K), float("nan"), device=device)
    xin = torch.full(
        (MAX_TOKENS, MOE_HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    scratch = torch.zeros(scratch_bytes(), dtype=torch.uint8, device=device)
    peers = PeerBuffer(sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device)
    peers.bytes.zero_()
    epoch = _epoch(device)
    launch = build_k2(K2Build(tokens=MAX_TOKENS, npes=1))
    got = stream.clone()
    try:
        for _ in range(2):
            got.copy_(stream)
            launch(
                o.data_ptr(), g.data_ptr(), got.data_ptr(), w["w_o"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
                w["w_hi"].data_ptr(), w["w_lo"].data_ptr(), w["bias"].data_ptr(),
                w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), ids.data_ptr(),
                scores.data_ptr(), 0, 0, xin.data_ptr(), scratch.data_ptr(), peers.local,
                peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all()
        _assert_close(got, reference[0], 1 / 64, 1e-2, "residual")
        assert torch.equal(ids, reference[1])
        _assert_close(scores, reference[2], 1 / 64, 1e-4, "scores")
        _assert_close(xin, reference[3], 1 / 32, 2e-2, "expert input")

        override_ids = torch.flip(reference[1], dims=(1,)).contiguous()
        override_scores = torch.flip(reference[2], dims=(1,)).contiguous()
        got.copy_(stream)
        launch(
            o.data_ptr(), g.data_ptr(), got.data_ptr(), w["w_o"].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
            w["w_hi"].data_ptr(), w["w_lo"].data_ptr(), w["bias"].data_ptr(),
            w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), ids.data_ptr(),
            scores.data_ptr(), override_ids.data_ptr(), override_scores.data_ptr(),
            xin.data_ptr(), scratch.data_ptr(), peers.local,
            peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        torch.cuda.synchronize()
        assert torch.equal(ids, override_ids)
        assert torch.equal(scores, override_scores)
        _assert_close(xin, reference[3], 1 / 32, 2e-2, "overridden expert input")
    finally:
        peers.close()
