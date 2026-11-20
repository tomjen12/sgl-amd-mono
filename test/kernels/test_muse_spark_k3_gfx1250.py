"""Standalone gfx1250 K3 against an explicit native-Quark PyTorch reference."""

import json
import os
from types import SimpleNamespace

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
TABLE_EXPERTS = 64
ROUTED_EXPERTS = 2
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
    dense_packed = packed[:ROUTED_EXPERTS]
    dense_scales = scales[:ROUTED_EXPERTS]
    codes = torch.stack(
        (dense_packed & 0xF, dense_packed >> 4), -1
    ).reshape(
        ROUTED_EXPERTS, rows, k
    )
    mag = E2M1.to(device)[(codes & 7).long()]
    value = torch.where((codes & 8) != 0, -mag, mag)
    scale = (dense_scales.to(torch.int32) << 23).view(torch.float32)
    dense = (
        value.reshape(ROUTED_EXPERTS, rows, k // 32, 32)
        * scale[..., None]
    ).reshape(ROUTED_EXPERTS, rows, k)
    return packed, scales, dense.to(torch.bfloat16)


def _mx_dequant(x):
    """Exact explicit form of AITER's per-1x32 MXFP8 activation quantizer."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    inv_max = torch.tensor(
        0x3B124925, dtype=torch.int32, device=x.device
    ).view(torch.float32)
    working = (amax * inv_max).view(torch.int32)
    exponent = (working >> 23) & 0xFF
    e8m0 = (exponent + ((working & 0x7FFFFF) != 0)).clamp_max(255)
    dequant_scale = torch.ldexp(
        torch.ones_like(amax), e8m0 - 127
    )
    return (
        (blocks / dequant_scale)
        .to(torch.float8_e4m3fn)
        .float()
        * dequant_scale
    ).reshape(shape)


def test_mxfp8_reference_matches_aiter_quantizer():
    from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

    device = torch.device("cuda")
    x = (
        torch.randn(8, 4096, device=device)
        * torch.linspace(0.01, 8.0, 8, device=device)[:, None]
    ).to(torch.bfloat16)
    payload, scales = per_1x32_mx_quant(x, quant_mode="fp8")
    dequant_scale = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32),
        scales.to(torch.int32) - 127,
    )
    got = (
        payload.float().reshape(8, -1, 32) * dequant_scale[..., None]
    ).reshape_as(x)
    torch.testing.assert_close(got, _mx_dequant(x), rtol=0, atol=0)


def _local_partial(
    xin, ids, scores, local_start, d13, d2, pen_w, diagnostics=False
):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        INTER,
        LOCAL_EXPERTS,
        MOE_HIDDEN,
        POST_EPS,
        TOP_K,
    )

    out = torch.zeros(xin.shape[0], MOE_HIDDEN, device=xin.device)
    xq = _mx_dequant(xin.float())
    diag = None
    if diagnostics:
        shape = (xin.shape[0] * TOP_K, MOE_HIDDEN)
        diag = {
            "qx": torch.zeros(shape, dtype=torch.bfloat16, device=xin.device),
            "qmid": torch.zeros(shape, dtype=torch.bfloat16, device=xin.device),
            "y": torch.zeros(shape, dtype=torch.bfloat16, device=xin.device),
        }
    for token in range(xin.shape[0]):
        for slot in range(TOP_K):
            expert = int(ids[token, slot]) - local_start
            if not 0 <= expert < LOCAL_EXPERTS:
                continue
            pick = token * TOP_K + slot
            gate_up = xq[token] @ d13[expert].float().T
            gate, up = gate_up[:INTER], gate_up[INTER:]
            mid = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
            y = _bf(mid @ d2[expert].float().T)
            if diag is not None:
                diag["qx"][pick] = xq[token]
                diag["qmid"][pick] = mid
                diag["y"][pick] = y
            value = _bf(y * pen_w.float())
            scale = scores[token, slot] * torch.rsqrt(
                value.square().mean() + POST_EPS
            )
            out[token] += scale * value
    partial = _bf(out)
    if diag is not None:
        diag["fbuf"] = partial
        return partial, diag
    return partial


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


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("last", [False, True])
def test_k3_matches_explicit_native_quark_reference(last, shuffled):
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
        POST_EPS,
        TOP_K,
    )

    tdm_expert = os.environ.get("SGLANG_AMD_MONO_MUSE_K3_TDM") == "1"
    if tdm_expert and not shuffled:
        pytest.skip("K3 TDM consumes AITER-preshuffled MXFP4 weights")

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(12503 + int(last))
    bf16 = torch.bfloat16
    w13, s13, d13 = _native_weight(
        TABLE_EXPERTS, 2 * INTER, MOE_HIDDEN, device, gen
    )
    w2, s2, d2 = _native_weight(
        TABLE_EXPERTS, MOE_HIDDEN, INTER, device, gen
    )
    if shuffled:
        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

        w13 = moe_shuffle_weight(
            w13, is_guinterleave=True, gate_up=True
        ).view(torch.uint8)
        s13 = moe_shuffle_scale(
            s13, experts_cnt=TABLE_EXPERTS,
            is_guinterleave=True, gate_up=True,
        ).view(torch.uint8)
        w2 = moe_shuffle_weight(w2).view(torch.uint8)
        s2 = moe_shuffle_scale(
            s2, experts_cnt=TABLE_EXPERTS
        ).view(torch.uint8)
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
    partial, diag = _local_partial(
        xin,
        ids,
        scores,
        0,
        d13,
        d2,
        weights["pen_w"],
        diagnostics=True,
    )
    aiter_fbuf = None
    aiter_rows = None
    aiter_y = None
    if shuffled:
        import sglang.srt.rocm_muse_spark.mxfp4_moe as mxfp4_moe

        mask = torch.zeros(256, dtype=torch.int32, device=device)
        mask[:TABLE_EXPERTS] = 1
        experts = SimpleNamespace(
            gfx1250_a8w4=(w13, w2, s13, s2, mask),
            num_local_experts=TABLE_EXPERTS,
            expert_hidden_size=MOE_HIDDEN,
        )
        capture = {}
        original_finalize = mxfp4_moe._finalize_rows

        def capture_finalize(y, rows, *args, **kwargs):
            capture["y"], capture["rows"] = y, rows
            return original_finalize(y, rows, *args, **kwargs)

        mxfp4_moe._finalize_rows = capture_finalize
        try:
            aiter_fbuf = mxfp4_moe.run_gfx1250_a8w4(
                xin,
                scores,
                ids,
                experts,
                0,
                weights["pen_w"],
                POST_EPS,
                0.0,
            )
            torch.cuda.synchronize()
            aiter_y = capture["y"].detach().clone().reshape(-1, MOE_HIDDEN)
            aiter_rows = capture["rows"].detach().clone().reshape(-1).long()
        finally:
            mxfp4_moe._finalize_rows = original_finalize
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
    launch = build_k3(
        K3Build(
            npes=1,
            shuffled=shuffled,
            staged_profile=os.environ.get("MUSE_K3_STAGED_PROFILE") == "1",
            phase_trace=os.environ.get("MUSE_K3_PHASE_TRACE") == "1",
            unpool_profile_mode=int(
                os.environ.get("MUSE_K3_UNPOOL_PROFILE_MODE", "2")
            ),
            expert_profile_mode=(
                os.environ.get("MUSE_K3_EXPERT_PROFILE_MODE", "1") == "1"
            ),
            tdm_expert=tdm_expert,
            tdm_fast_drain=(
                os.environ.get("SGLANG_AMD_MONO_MUSE_K3_TDM_FAST_DRAIN")
                == "1"
            ),
        )
    )
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
        bench_iters = int(os.environ.get("MUSE_K3_BENCH_ITERS", "0"))
        if bench_iters:
            samples = []
            for _ in range(bench_iters):
                got.copy_(base_stream)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
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
                end.record()
                end.synchronize()
                epoch[0] += 1
                samples.append(start.elapsed_time(end) * 1000.0)
            print(
                "K3_BENCH_US="
                + json.dumps(
                    {
                        "mean": sum(samples) / len(samples),
                        "min": min(samples),
                        "max": max(samples),
                        "samples": samples,
                    }
                )
            )
        from sgl_amd_mono.muse_spark.gfx1250.build_k3 import _scratch_layout

        layout = _scratch_layout()
        if os.environ.get("MUSE_K3_PHASE_TRACE") == "1":
            offset, _ = layout["k3_phase_ticks"]
            ticks = (
                scratch[offset : offset + 8 * 7 * 8]
                .view(torch.int64)
                .view(8, 7)
                .cpu()
            )
            deltas = ticks[:, 1:] - ticks[:, :-1]
            assert (deltas > 0).all()
            print(
                "K3_PHASE_TICKS="
                + json.dumps(
                    {
                        "ticks": ticks.tolist(),
                        "deltas": deltas.tolist(),
                    }
                )
            )

        def scratch_bf16(name, shape):
            offset, _ = layout[name]
            size = torch.tensor(shape).prod().item()
            return scratch[offset : offset + size * 2].view(bf16).view(shape)

        def scratch_mxfp8(name, rows, cols):
            offset, _ = layout[name]
            if tdm_expert and name == "k3_qmid":
                raw = scratch[offset : offset + rows * cols * 2].view(
                    rows, cols * 2
                )
                payload = raw[:, :cols].view(torch.float8_e4m3fn).float()
                scale_slab = raw[:, cols:]
                scale_i = torch.arange(
                    cols // 32, device=device, dtype=torch.long
                )
                scales = scale_slab[
                    :,
                    (scale_i // 4) * 128 + scale_i % 4,
                ].to(torch.int32)
                if os.environ.get("MUSE_K3_TDM_DEBUG") == "1":
                    print(
                        "K3_TDM_QMID_RAW="
                        + json.dumps(
                            {
                                "payload_nonzero": int(
                                    torch.count_nonzero(payload).item()
                                ),
                                "payload_absmax": float(
                                    payload.abs().max().item()
                                ),
                                "payload_nonzero_rows": torch.count_nonzero(
                                    payload, dim=1
                                ).tolist(),
                                "payload_row0_indices": torch.nonzero(
                                    payload[0], as_tuple=False
                                )
                                .flatten()[:64]
                                .tolist(),
                                "scale_min": int(scales.min().item()),
                                "scale_max": int(scales.max().item()),
                                "scale_nonzero_rows": torch.count_nonzero(
                                    scales, dim=1
                                ).tolist(),
                                "slab_nonzero": int(
                                    torch.count_nonzero(scale_slab).item()
                                ),
                            }
                        )
                    )
                dequant_scale = torch.ldexp(
                    torch.ones_like(scales, dtype=torch.float32),
                    scales - 127,
                )
                return (
                    payload.view(rows, cols // 32, 32)
                    * dequant_scale[..., None]
                ).view(rows, cols)
            raw = scratch[offset : offset + rows * cols * 2].view(
                rows, cols * 2
            )
            payload = raw[:, :cols].view(torch.float8_e4m3fn).float()
            scales = raw[:, cols : cols + cols // 32].to(torch.int32)
            dequant_scale = torch.ldexp(
                torch.ones_like(scales, dtype=torch.float32),
                scales - 127,
            )
            return (
                payload.view(rows, cols // 32, 32)
                * dequant_scale[..., None]
            ).view(rows, cols)

        held = ((ids >= 0) & (ids < ROUTED_EXPERTS)).reshape(-1)
        for name in ("qx", "qmid"):
            if tdm_expert:
                # Compact TDM stages use expert-grouped rows and AITER's
                # shuffled scale plane; end-to-end y/AITER checks below cover
                # the production contract until the compact decoder is used.
                continue
            got_stage = scratch_mxfp8(
                f"k3_{name}", 8 * TOP_K, MOE_HIDDEN
            )[held]
            ref_stage = diag[name][held].float()
            cosine = torch.nn.functional.cosine_similarity(
                got_stage, ref_stage, dim=-1
            )
            assert cosine.min() > 0.997, (
                f"{name} cosine {cosine.min().item():.6f}"
            )
        got_y = scratch_bf16(
            "k3_y", (8 * TOP_K, MOE_HIDDEN)
        )[held].float()
        y_cosine = torch.nn.functional.cosine_similarity(
            got_y, diag["y"][held].float(), dim=-1
        )
        assert y_cosine.min() > 0.997, (
            f"y cosine {y_cosine.min().item():.6f}"
        )
        got_fbuf = scratch_bf16("k3_fbuf", (8, MOE_HIDDEN)).float()
        fbuf_cosine = torch.nn.functional.cosine_similarity(
            got_fbuf, diag["fbuf"].float(), dim=-1
        )
        assert fbuf_cosine.min() > 0.997, (
            f"fbuf cosine {fbuf_cosine.min().item():.6f}"
        )
        if os.environ.get("MUSE_K3_UNPOOL_DEBUG") == "1":
            offset, _ = layout["k3_unpool_part"]
            partials = (
                scratch[offset : offset + 4 * 64 * HIDDEN * 4]
                .view(torch.float32)
                .view(4, 64, HIDDEN)
            )
            got_branch = partials[:, :8].sum(dim=0)
            ref_branch = got_fbuf @ weights["w_unpool"].float().T
            branch_cosine = torch.nn.functional.cosine_similarity(
                got_branch, ref_branch, dim=-1
            )
            print(
                "K3_UNPOOL_SPLITK="
                + json.dumps(
                    {
                        "cosine": branch_cosine.tolist(),
                        "got_absmax": float(got_branch.abs().max().item()),
                        "ref_absmax": float(ref_branch.abs().max().item()),
                    }
                )
            )
        if aiter_fbuf is not None:
            local_picks = held.nonzero(as_tuple=False).flatten()
            rows = aiter_rows[local_picks]
            valid = rows >= 0
            aiter_y_cosine = torch.nn.functional.cosine_similarity(
                scratch_bf16("k3_y", (8 * TOP_K, MOE_HIDDEN))[
                    local_picks[valid]
                ].float(),
                aiter_y[rows[valid]].float(),
                dim=-1,
            )
            assert aiter_y_cosine.min() > 0.997, (
                f"AITER y cosine {aiter_y_cosine.min().item():.6f}"
            )
            aiter_cosine = torch.nn.functional.cosine_similarity(
                got_fbuf, aiter_fbuf.float(), dim=-1
            )
            assert aiter_cosine.min() > 0.997, (
                f"AITER fbuf cosine {aiter_cosine.min().item():.6f}"
            )
        assert (
            epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0
        ).all()
        expected_stream, expected_attn = ref_stream, ref_attn
        if tdm_expert:
            # TDM is the AITER-aligned path; preserve the existing strict
            # boundary tolerance while using AITER's A8W4 local partial.
            expected_stream, expected_attn = _boundary(
                [aiter_fbuf], weights, base_stream, last
            )
        torch.testing.assert_close(
            got, expected_stream, rtol=1 / 32, atol=2e-2
        )
        torch.testing.assert_close(
            attn, expected_attn, rtol=1 / 32, atol=2e-2
        )
    finally:
        peers.close()
