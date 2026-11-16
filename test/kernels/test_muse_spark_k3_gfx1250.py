"""Standalone gfx1250 K3 against an explicit native-Quark PyTorch reference."""

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

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TABLE_EXPERTS = 2
LAYER = 11


def _bf(x):
    return x.to(torch.bfloat16).float()


def _epoch(device):
    from sgl_amd_mono.common import guard

    out = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(30.0)
    lo = ticks & 0xFFFFFFFF
    out[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    out[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return out


def _native_weight(experts, rows, k, device, gen):
    packed = torch.randint(
        0,
        256,
        (experts, rows, k // 2),
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )
    scales = torch.randint(
        120,
        128,
        (experts, rows, k // 32),
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )
    codes = torch.stack((packed & 0xF, packed >> 4), -1).reshape(
        experts, rows, k
    )
    mag = E2M1.to(device)[(codes & 7).long()]
    value = torch.where((codes & 8) != 0, -mag, mag)
    scale = (scales.to(torch.int32) << 23).view(torch.float32)
    dense = (
        value.reshape(experts, rows, k // 32, 32) * scale[..., None]
    ).reshape(experts, rows, k)
    return packed, scales, dense.to(torch.bfloat16)


def _mx_dequant(x):
    """Exact explicit form of Quark's dynamic 32-value MXFP4 quantizer."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    exponent = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    scaled = blocks * torch.ldexp(torch.ones_like(amax), -exponent)
    absolute = scaled.abs()
    abits = absolute.view(torch.int32)
    den = ((absolute + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    normal = (
        (abits + ((-126 << 23) + 0x1FFFFF) + ((abits >> 22) & 1)) >> 22
    ) & 0xFF
    code = torch.where(
        absolute >= 6.0,
        torch.full_like(den, 7),
        torch.where(absolute < 1.0, den, normal),
    )
    mag = E2M1.to(x.device)[code.long()]
    quantized = torch.where(scaled < 0, -mag, mag)
    return (
        quantized * torch.ldexp(torch.ones_like(amax), exponent)
    ).reshape(shape)


def _local_partial(xin, ids, scores, local_start, d13, d2, pen_w):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        INTER,
        LOCAL_EXPERTS,
        MOE_HIDDEN,
        POST_EPS,
        TOP_K,
    )

    out = torch.zeros(xin.shape[0], MOE_HIDDEN, device=xin.device)
    xq = _mx_dequant(xin.float())
    for token in range(xin.shape[0]):
        for slot in range(TOP_K):
            expert = int(ids[token, slot]) - local_start
            if not 0 <= expert < LOCAL_EXPERTS:
                continue
            gate_up = xq[token] @ d13[expert].float().T
            gate, up = gate_up[:INTER], gate_up[INTER:]
            mid = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
            y = _bf(mid @ d2[expert].float().T)
            value = _bf(y * pen_w.float())
            scale = scores[token, slot] * torch.rsqrt(
                value.square().mean() + POST_EPS
            )
            out[token] += scale * value
    return _bf(out)


def _boundary(partials, weights, stream, last):
    from sgl_amd_mono.muse_spark.gfx1250.config import POST_EPS, RMS_EPS

    # Partial tensors are intentionally added rank-by-rank before one BF16 round.
    ffn = torch.zeros_like(partials[0].float())
    for partial in partials:
        ffn += partial.float()
    ffn = _bf(ffn)
    branch = _bf(ffn @ weights["w_unpool"].float().T)
    post = _bf(branch * weights["pfw"].float())
    normed = _bf(
        post * torch.rsqrt(post.square().mean(-1, keepdim=True) + POST_EPS)
    )
    new = weights["alpha"] * stream + weights["beta"] * normed
    pre = _bf(new) if last else new
    attn = (
        pre
        * torch.rsqrt(pre.square().mean(-1, keepdim=True) + RMS_EPS)
        * weights["npw"].float()
    ).to(torch.bfloat16)
    return new, attn


def _picks(device):
    # Only local IDs 0/1 are routed into the compact two-expert test table.
    ids = torch.empty(8, 8, dtype=torch.int32, device=device)
    for token in range(8):
        ids[token] = torch.tensor(
            [token % 2, (token + 1) % 2, 64, 65, 130, 131, 200, 201],
            dtype=torch.int32,
            device=device,
        )
    return ids


@pytest.mark.parametrize("last", [False, True])
def test_k3_matches_explicit_native_quark_reference(last):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import (
        K3Build,
        build_k3,
        k3_scratch_bytes,
        k3_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        INTER,
        MOE_HIDDEN,
        TOP_K,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(12503 + int(last))
    bf16 = torch.bfloat16
    w13, s13, d13 = _native_weight(
        TABLE_EXPERTS, 2 * INTER, MOE_HIDDEN, device, gen
    )
    w2, s2, d2 = _native_weight(
        TABLE_EXPERTS, MOE_HIDDEN, INTER, device, gen
    )
    xin = (torch.randn(8, MOE_HIDDEN, device=device) * 0.25).to(bf16)
    ids = _picks(device)
    scores = torch.rand(8, TOP_K, device=device, generator=gen) * 0.25
    base_stream = torch.randn(8, HIDDEN, device=device, generator=gen)
    weights = {
        "pen_w": (1 + 0.05 * torch.randn(MOE_HIDDEN, device=device)).to(bf16),
        "pfw": (1 + 0.05 * torch.randn(HIDDEN, device=device)).to(bf16),
        "alpha": torch.rand(HIDDEN, device=device, generator=gen) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=device, generator=gen) * 0.5,
        "npw": (1 + 0.05 * torch.randn(HIDDEN, device=device)).to(bf16),
        "w_unpool": (
            torch.randn(HIDDEN, MOE_HIDDEN, device=device, generator=gen)
            * 0.01
        ).to(bf16),
    }
    partial = _local_partial(
        xin, ids, scores, 0, d13, d2, weights["pen_w"]
    )
    ref_stream, ref_attn = _boundary([partial], weights, base_stream, last)

    scratch = torch.zeros(k3_scratch_bytes(), dtype=torch.uint8, device=device)
    peers = PeerBuffer(
        k3_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    peers.bytes.zero_()
    epoch = _epoch(device)
    attn = torch.full(
        (8, HIDDEN), float("nan"), dtype=bf16, device=device
    )
    got = base_stream.clone()
    launch = build_k3(K3Build(npes=1))
    try:
        for _ in range(2):
            got.copy_(base_stream)
            launch(
                xin.data_ptr(),
                ids.data_ptr(),
                scores.data_ptr(),
                0,
                w13.data_ptr(),
                s13.data_ptr(),
                w2.data_ptr(),
                s2.data_ptr(),
                weights["pen_w"].data_ptr(),
                got.data_ptr(),
                weights["pfw"].data_ptr(),
                weights["alpha"].data_ptr(),
                weights["beta"].data_ptr(),
                weights["npw"].data_ptr(),
                weights["w_unpool"].data_ptr(),
                attn.data_ptr(),
                scratch.data_ptr(),
                peers.local,
                peers.addresses.data_ptr(),
                0,
                epoch.data_ptr(),
                LAYER,
                int(last),
                stream=torch.cuda.current_stream(),
            )
            epoch[0] += 1
        torch.cuda.synchronize()
        assert (
            epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0
        ).all()
        torch.testing.assert_close(got, ref_stream, rtol=1 / 32, atol=2e-2)
        torch.testing.assert_close(attn, ref_attn, rtol=1 / 32, atol=2e-2)
    finally:
        peers.close()
