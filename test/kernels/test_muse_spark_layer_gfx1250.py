"""End-to-end gfx1250 K-attn/K2/K3 split and fused correctness tests."""

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

PAGES = 512
MAX_BLOCKS = 512
TABLE_EXPERTS = 2
LAYER = 17
ROUTED_IDS = (0, 1, 64, 65, 128, 129, 192, 193)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _bf(value):
    return value.to(torch.bfloat16).float()


def _epoch(device, deadline_s=60.0):
    from sgl_amd_mono.common import guard

    out = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    out[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    out[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return out


def _attention_reference(q, k_cache, v_cache, table, keys, window, scale):
    """Independent paged-cache GQA attention, including BF16 probability round."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        LOCAL_Q_HEADS,
        Q_K,
    )

    if table.dim() == 2:
        return torch.cat(
            [
                _attention_reference(
                    q[start : start + 8],
                    k_cache,
                    v_cache,
                    table[request],
                    keys,
                    window,
                    scale,
                )
                for request, start in enumerate(range(0, q.shape[0], 8))
            ]
        )

    tokens = q.shape[0]
    slots = (
        table[: (keys + 15) // 16].long()[:, None] * 16
        + torch.arange(16, device=q.device)[None, :]
    ).reshape(-1)[:keys]
    key = k_cache[slots].float()
    value = v_cache[slots].float()
    out = torch.zeros(tokens, Q_K, device=q.device)
    prefix = keys - tokens
    key_index = torch.arange(keys, device=q.device)
    group = LOCAL_Q_HEADS // LOCAL_KV_HEADS
    for token in range(tokens):
        end = prefix + token + 1
        allowed = key_index < end
        if window:
            allowed &= key_index >= max(0, end - window)
        for head in range(LOCAL_Q_HEADS):
            query = q[token, head * HEAD_DIM : (head + 1) * HEAD_DIM].float()
            products = (
                key[:, head // group] * query
            ).reshape(keys, 8, 8)
            lane_sums = torch.zeros(
                keys, 8, dtype=torch.float32, device=q.device
            )
            for element in range(8):
                lane_sums += products[:, :, element]
            lanes = torch.arange(8, device=q.device)
            for offset in (4, 2, 1):
                lane_sums = lane_sums + lane_sums[:, lanes ^ offset]
            logits = scale * lane_sums[:, 0]
            logits = torch.where(
                allowed, logits, torch.full_like(logits, float("-inf"))
            )
            probability = torch.softmax(logits, -1).to(torch.bfloat16).float()
            out[token, head * HEAD_DIM : (head + 1) * HEAD_DIM] = (
                probability @ value[:, head // group]
            )
    return out


def _rms(value, epsilon, weight=None):
    value = value.float()
    out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    if weight is not None:
        out *= weight.float()
    return out.to(torch.bfloat16)


def _k2_reference(o, g, base_res, weights, partials=None):
    """Independent K2 math; partials model rank-ordered BF16 TP projection."""
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
    projected = partials[0].float()
    for partial in partials[1:]:
        projected += partial.float()
    projected = projected.to(torch.bfloat16)
    branch = _rms(projected, POST_EPS).float()
    res = weights["attn_alpha"] * base_res.float() + weights["attn_beta"] * branch
    x = _rms(res, RMS_EPS, weights["pre_w"])
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
    return res, ids.to(torch.int32), scores, xin, og


def _native_weight(experts, rows, columns, device, generator):
    """Create native Quark tensors and their exact BF16 dense interpretation."""
    packed = torch.randint(
        0,
        256,
        (experts, rows, columns // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scales = torch.randint(
        120,
        128,
        (experts, rows, columns // 32),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    codes = torch.stack((packed & 0xF, packed >> 4), -1).reshape(
        experts, rows, columns
    )
    magnitude = E2M1.to(device)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -magnitude, magnitude)
    scale = (scales.to(torch.int32) << 23).view(torch.float32)
    dense = (
        values.reshape(experts, rows, columns // 32, 32) * scale[..., None]
    ).reshape(experts, rows, columns)
    return packed, scales, dense.to(torch.bfloat16)


def _mx_dequant(value):
    """Exact explicit form of Quark's dynamic 32-value MXFP4 quantizer."""
    shape = value.shape
    blocks = value.float().reshape(*shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    exponent = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    scaled = blocks * torch.ldexp(torch.ones_like(amax), -exponent)
    absolute = scaled.abs()
    abits = absolute.view(torch.int32)
    denormal = ((absolute + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    normal = (
        (abits + ((-126 << 23) + 0x1FFFFF) + ((abits >> 22) & 1)) >> 22
    ) & 0xFF
    code = torch.where(
        absolute >= 6.0,
        torch.full_like(denormal, 7),
        torch.where(absolute < 1.0, denormal, normal),
    )
    magnitude = E2M1.to(value.device)[code.long()]
    quantized = torch.where(scaled < 0, -magnitude, magnitude)
    return (
        quantized * torch.ldexp(torch.ones_like(amax), exponent)
    ).reshape(shape)


def _local_k3_partial(xin, ids, scores, local_start, d13, d2, pen_w):
    """Exact native-Quark local expert reference with kernel rounding points."""
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
            middle = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
            y = _bf(middle @ d2[expert].float().T)
            value = _bf(y * pen_w.float())
            scale = scores[token, slot] * torch.rsqrt(
                value.square().mean() + POST_EPS
            )
            out[token] += scale * value
    return _bf(out)


def _k3_boundary(partials, weights, stream, last):
    """Rank-ordered BF16 EP reduction and exact K3 boundary reference."""
    from sgl_amd_mono.muse_spark.gfx1250.config import POST_EPS, RMS_EPS

    ffn = torch.zeros_like(partials[0].float())
    for partial in partials:
        ffn += partial.float()
    ffn = _bf(ffn)
    branch = _bf(ffn @ weights["w_unpool"].float().T)
    post = _bf(branch * weights["pfw"].float())
    normed = _bf(
        post * torch.rsqrt(post.square().mean(-1, keepdim=True) + POST_EPS)
    )
    res = weights["ffn_alpha"] * stream + weights["ffn_beta"] * normed
    pre = _bf(res) if last else res
    attn_out = (
        pre
        * torch.rsqrt(pre.square().mean(-1, keepdim=True) + RMS_EPS)
        * weights["npw"].float()
    ).to(torch.bfloat16)
    return res, attn_out


def _make_case(device, seed, keys):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HEAD_DIM,
        HIDDEN,
        INTER,
        LOCAL_KV_HEADS,
        DRAFT_TOKENS as MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        SOFTMAX_SCALE,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    q = (torch.randn(MAX_TOKENS, Q_K, device=device, generator=generator) * 5.47).to(bf16)
    g = torch.randn(MAX_TOKENS, Q_K, device=device, generator=generator).to(bf16)
    k_cache = (
        torch.randn(
            PAGES * 16,
            LOCAL_KV_HEADS,
            HEAD_DIM,
            device=device,
            generator=generator,
        )
        * 2
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        PAGES * 16,
        LOCAL_KV_HEADS,
        HEAD_DIM,
        device=device,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    pages = (keys + 15) // 16
    table = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)
    table[:pages] = (
        torch.randperm(PAGES - 1, device=device, generator=generator)[:pages] + 1
    ).to(torch.int32)
    bias = torch.full((EXPERTS,), -100.0, device=device)
    for priority, expert in enumerate(ROUTED_IDS):
        bias[expert] = float(len(ROUTED_IDS) - priority)
    weights = {
        "w_o": (
            torch.randn(HIDDEN, Q_K, device=device, generator=generator) * 0.02
        ).to(bf16),
        "attn_alpha": torch.rand(HIDDEN, device=device, generator=generator) * 0.5
        + 0.5,
        "attn_beta": torch.rand(HIDDEN, device=device, generator=generator) * 0.5,
        "pre_w": (
            1 + 0.1 * torch.randn(HIDDEN, device=device, generator=generator)
        ).to(bf16),
        "w_hi": torch.zeros(EXPERTS, HIDDEN, dtype=bf16, device=device),
        "w_lo": torch.zeros(EXPERTS, HIDDEN, dtype=bf16, device=device),
        "bias": bias,
        "w_pool": (
            torch.randn(
                MOE_HIDDEN, HIDDEN, device=device, generator=generator
            )
            * 0.02
        ).to(bf16),
        "xn_w": (
            1 + 0.1 * torch.randn(MOE_HIDDEN, device=device, generator=generator)
        ).to(bf16),
        "pen_w": (
            1 + 0.05 * torch.randn(MOE_HIDDEN, device=device, generator=generator)
        ).to(bf16),
        "pfw": (
            1 + 0.05 * torch.randn(HIDDEN, device=device, generator=generator)
        ).to(bf16),
        "ffn_alpha": torch.rand(HIDDEN, device=device, generator=generator) * 0.5
        + 0.5,
        "ffn_beta": torch.rand(HIDDEN, device=device, generator=generator) * 0.5,
        "npw": (
            1 + 0.05 * torch.randn(HIDDEN, device=device, generator=generator)
        ).to(bf16),
        "w_unpool": (
            torch.randn(
                HIDDEN, MOE_HIDDEN, device=device, generator=generator
            )
            * 0.01
        ).to(bf16),
    }
    w13, s13, d13 = _native_weight(
        TABLE_EXPERTS, 2 * INTER, MOE_HIDDEN, device, generator
    )
    w2, s2, d2 = _native_weight(
        TABLE_EXPERTS, MOE_HIDDEN, INTER, device, generator
    )
    return {
        "q": q,
        "g": g,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "table": table,
        "prefix": torch.tensor(
            [keys - MAX_TOKENS], dtype=torch.int32, device=device
        ),
        "scale_bits": struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0],
        "base_res": torch.randn(
            MAX_TOKENS, HIDDEN, device=device, generator=generator
        ),
        "weights": weights,
        "w13": w13,
        "s13": s13,
        "d13": d13,
        "w2": w2,
        "s2": s2,
        "d2": d2,
    }


def _outputs(device, rows=8):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        MOE_HIDDEN,
        Q_K,
        TOP_K,
    )

    return {
        "o": torch.full(
            (rows, Q_K), float("nan"), dtype=torch.bfloat16, device=device
        ),
        "ids": torch.full(
            (rows, TOP_K), -1, dtype=torch.int32, device=device
        ),
        "scores": torch.full(
            (rows, TOP_K), float("nan"), device=device
        ),
        "xin": torch.full(
            (rows, MOE_HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
        "res": torch.empty(rows, HIDDEN, device=device),
        "attn_out": torch.full(
            (rows, HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _reset(outputs, base_res):
    outputs["o"].fill_(float("nan"))
    outputs["ids"].fill_(-1)
    outputs["scores"].fill_(float("nan"))
    outputs["xin"].fill_(float("nan"))
    outputs["res"].copy_(base_res)
    outputs["attn_out"].fill_(float("nan"))


def _assert_close(got, expected, rtol, atol, label):
    error = (got.float() - expected.float()).abs()
    limit = expected.float().abs() * rtol + atol
    assert torch.isfinite(got).all(), f"{label}: non-finite output"
    assert (error <= limit).all(), f"{label}: max error {error.max().item()}"


def _check_outputs(got, reference, prefix):
    _assert_close(got["o"], reference["o"], 1 / 32, 2e-2, f"{prefix} o")
    assert torch.equal(got["ids"], reference["ids"]), f"{prefix} ids"
    _assert_close(
        got["scores"], reference["scores"], 1 / 64, 1e-4, f"{prefix} scores"
    )
    _assert_close(got["xin"], reference["xin"], 1 / 32, 2e-2, f"{prefix} xin")
    _assert_close(got["res"], reference["res"], 1 / 32, 2e-2, f"{prefix} res")
    _assert_close(
        got["attn_out"],
        reference["attn_out"],
        1 / 32,
        2e-2,
        f"{prefix} attn_out",
    )


def _check_path_against_oracle(got, case, oracle_o, last, prefix):
    """Validate each stage without amplifying K-attn approximation through MoE."""
    weights = case["weights"]
    _assert_close(got["o"], oracle_o, 1 / 32, 2e-2, f"{prefix} o")
    k2 = _k2_reference(
        got["o"], case["g"], case["base_res"], weights
    )
    assert tuple(k2[1][0].tolist()) == ROUTED_IDS
    assert torch.equal(got["ids"], k2[1]), f"{prefix} ids"
    _assert_close(got["scores"], k2[2], 1 / 64, 1e-4, f"{prefix} scores")
    _assert_close(got["xin"], k2[3], 1 / 32, 2e-2, f"{prefix} xin")
    local = _local_k3_partial(
        got["xin"],
        got["ids"],
        got["scores"],
        0,
        case["d13"],
        case["d2"],
        weights["pen_w"],
    )
    expected_res, expected_attn = _k3_boundary([local], weights, k2[0], last)
    _assert_close(got["res"], expected_res, 1 / 32, 2e-2, f"{prefix} res")
    _assert_close(
        got["attn_out"],
        expected_attn,
        1 / 32,
        2e-2,
        f"{prefix} attn_out",
    )


def test_gfx1250_fused_target_normal_and_fault_variants_compile():
    from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
        LayerBuild,
        build_layer,
    )
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    for key in (
        LayerBuild(npes=4, next_k1=True),
        LayerBuild(
            npes=4,
            next_k1=True,
            fault_site=1,
            fault_after=1,
        ),
    ):
        compile_only(build_layer(key))


@pytest.mark.parametrize(
    "keys,window,last", [(300, 0, False), (3000, 2047, True)]
)
def test_gfx1250_fused_layer_matches_oracle_and_split(keys, window, last):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import (
        K2Build,
        build_k2,
        scratch_bytes as k2_scratch_bytes,
        sym_bytes as k2_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import (
        K3Build,
        build_k3,
        k3_scratch_bytes,
        k3_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
        LayerBuild,
        build_layer,
        layer_scratch_bytes,
        layer_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        DRAFT_TOKENS as MAX_TOKENS,
        SOFTMAX_SCALE,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import KAttnBuild, build_kattn

    device = torch.device("cuda")
    case = _make_case(device, 125000 + keys + window, keys)
    weights = case["weights"]
    oracle_o = _attention_reference(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["table"],
        keys,
        window,
        SOFTMAX_SCALE,
    )

    split = _outputs(device)
    fused = _outputs(device)
    split_scratch2 = torch.zeros(
        k2_scratch_bytes(), dtype=torch.uint8, device=device
    )
    split_scratch3 = torch.zeros(
        k3_scratch_bytes(), dtype=torch.uint8, device=device
    )
    fused_scratch = torch.zeros(
        layer_scratch_bytes(), dtype=torch.uint8, device=device
    )
    split_peers2 = PeerBuffer(
        k2_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    split_peers3 = PeerBuffer(
        k3_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    fused_peers = PeerBuffer(
        layer_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    split_peers2.bytes.zero_()
    split_peers3.bytes.zero_()
    fused_peers.bytes.zero_()
    split_epoch = _epoch(device)
    fused_epoch = _epoch(device)
    kattn = build_kattn(KAttnBuild(tokens=MAX_TOKENS, q_len=MAX_TOKENS))
    k2_launch = build_k2(K2Build(tokens=MAX_TOKENS, npes=1))
    k3_launch = build_k3(K3Build(npes=1, last=last))
    fused_launch = build_layer(
        LayerBuild(tokens=MAX_TOKENS, q_len=MAX_TOKENS, npes=1, last=last)
    )
    stream = torch.cuda.current_stream()

    try:
        for _ in range(2):
            _reset(split, case["base_res"])
            _reset(fused, case["base_res"])
            kattn(
                case["q"].data_ptr(), split["o"].data_ptr(),
                case["k_cache"].data_ptr(), case["v_cache"].data_ptr(),
                case["table"].data_ptr(), case["table"].stride(0),
                case["prefix"].data_ptr(), window,
                case["scale_bits"], stream=stream,
            )  # fmt: skip
            k2_launch(
                split["o"].data_ptr(), case["g"].data_ptr(), split["res"].data_ptr(),
                weights["w_o"].data_ptr(), weights["attn_alpha"].data_ptr(),
                weights["attn_beta"].data_ptr(), weights["pre_w"].data_ptr(),
                weights["w_hi"].data_ptr(), weights["w_lo"].data_ptr(),
                weights["bias"].data_ptr(), weights["w_pool"].data_ptr(),
                weights["xn_w"].data_ptr(), split["ids"].data_ptr(),
                split["scores"].data_ptr(), split["xin"].data_ptr(),
                split_scratch2.data_ptr(), split_peers2.local,
                split_peers2.addresses.data_ptr(), 0, split_epoch.data_ptr(), LAYER,
                stream=stream,
            )  # fmt: skip
            k3_launch(
                split["xin"].data_ptr(), split["ids"].data_ptr(),
                split["scores"].data_ptr(), 0, case["w13"].data_ptr(),
                case["s13"].data_ptr(), case["w2"].data_ptr(), case["s2"].data_ptr(),
                weights["pen_w"].data_ptr(), split["res"].data_ptr(),
                weights["pfw"].data_ptr(), weights["ffn_alpha"].data_ptr(),
                weights["ffn_beta"].data_ptr(), weights["npw"].data_ptr(),
                weights["w_unpool"].data_ptr(), split["attn_out"].data_ptr(),
                split_scratch3.data_ptr(), split_peers3.local,
                split_peers3.addresses.data_ptr(), 0, split_epoch.data_ptr(), LAYER,
                int(last), stream=stream,
            )  # fmt: skip
            fused_launch(
                case["q"].data_ptr(), fused["o"].data_ptr(),
                case["k_cache"].data_ptr(), case["v_cache"].data_ptr(),
                case["table"].data_ptr(), case["table"].stride(0),
                case["prefix"].data_ptr(), window,
                case["scale_bits"], case["g"].data_ptr(), fused["res"].data_ptr(),
                weights["w_o"].data_ptr(), weights["attn_alpha"].data_ptr(),
                weights["attn_beta"].data_ptr(), weights["pre_w"].data_ptr(),
                weights["w_hi"].data_ptr(), weights["w_lo"].data_ptr(),
                weights["bias"].data_ptr(), weights["w_pool"].data_ptr(),
                weights["xn_w"].data_ptr(), fused["ids"].data_ptr(),
                fused["scores"].data_ptr(), fused["xin"].data_ptr(), 0,
                case["w13"].data_ptr(), case["s13"].data_ptr(),
                case["w2"].data_ptr(), case["s2"].data_ptr(),
                weights["pen_w"].data_ptr(), weights["pfw"].data_ptr(),
                weights["ffn_alpha"].data_ptr(), weights["ffn_beta"].data_ptr(),
                weights["npw"].data_ptr(), weights["w_unpool"].data_ptr(),
                fused["attn_out"].data_ptr(),
                fused["attn_out"].data_ptr(), fused["attn_out"].data_ptr(),
                fused["attn_out"].data_ptr(), fused["attn_out"].data_ptr(),
                fused["attn_out"].data_ptr(), fused["attn_out"].data_ptr(),
                fused["attn_out"].data_ptr(), fused["attn_out"].data_ptr(),
                fused["attn_out"].data_ptr(), 0, 0, 0, 0,
                fused_scratch.data_ptr(),
                fused_peers.local, fused_peers.addresses.data_ptr(), 0,
                fused_epoch.data_ptr(), LAYER, int(last), stream=stream,
            )  # fmt: skip
            split_epoch[0] += 1
            fused_epoch[0] += 1
        torch.cuda.synchronize()
        _check_path_against_oracle(split, case, oracle_o, last, "split")
        _check_path_against_oracle(fused, case, oracle_o, last, "fused")
        _assert_close(
            fused["o"], split["o"], 1 / 32, 2e-2, "fused vs split o"
        )
    finally:
        split_peers2.close()
        split_peers3.close()
        fused_peers.close()


def test_gfx1250_layer_v2_matches_layer_v1_plus_k1():
    from test_muse_spark_k1_gfx1250 import (
        POSITIONS,
        SLOTS,
        _close as k1_close,
        _cos_sin,
        _reference as k1_reference,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
        LayerBuild,
        build_layer,
        layer_scratch_bytes,
        layer_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        HIDDEN,
        KV_K,
        LOCAL_KV_HEADS,
        DRAFT_TOKENS as MAX_TOKENS,
        Q_K,
        QKVG_ROWS,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx1250.layout import (
        k1_scratch_layout,
        scratch_bytes,
    )

    device, bf16 = torch.device("cuda"), torch.bfloat16
    case = _make_case(device, 125020, 300)
    weights = case["weights"]
    generator = torch.Generator(device=device).manual_seed(125021)
    next_w = (
        torch.randn(
            QKVG_ROWS, HIDDEN, device=device, generator=generator
        )
        * 0.02
    ).to(bf16)
    cos_sin = _cos_sin(device)
    positions = torch.randint(
        0,
        POSITIONS,
        (MAX_TOKENS,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    slots = torch.randperm(SLOTS, device=device, generator=generator)[
        :MAX_TOKENS
    ]
    swa_map = torch.arange(SLOTS, dtype=torch.int64, device=device)

    split = _outputs(device)
    fused = _outputs(device)
    split_q = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=device)
    split_k = torch.empty(MAX_TOKENS, KV_K, dtype=bf16, device=device)
    split_v = torch.empty_like(split_k)
    split_g = torch.empty_like(split_q)
    fused_q = case["q"].clone()
    fused_k = torch.empty_like(split_k)
    fused_v = torch.empty_like(split_v)
    fused_g = case["g"].clone()
    split_k_cache = torch.zeros(
        SLOTS, LOCAL_KV_HEADS, HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    split_v_cache = torch.zeros_like(split_k_cache)
    fused_k_cache = torch.zeros_like(split_k_cache)
    fused_v_cache = torch.zeros_like(split_k_cache)

    split_layer_scratch = torch.zeros(
        layer_scratch_bytes(), dtype=torch.uint8, device=device
    )
    fused_layer_scratch = torch.zeros_like(split_layer_scratch)
    split_k1_scratch = torch.zeros(
        scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=device
    )
    split_peers = PeerBuffer(
        layer_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    fused_peers = PeerBuffer(
        layer_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    split_peers.bytes.zero_()
    fused_peers.bytes.zero_()
    split_epoch, fused_epoch = _epoch(device), _epoch(device)
    layer_v1 = build_layer(
        LayerBuild(npes=1, next_k1=False, last=False)
    )
    layer_v2 = build_layer(
        LayerBuild(npes=1, next_k1=True, last=False)
    )
    k1 = build_k1(K1Build(tokens=MAX_TOKENS))
    stream = torch.cuda.current_stream()

    def launch_layer(
        launch,
        outputs,
        q,
        gate,
        scratch,
        peers,
        epoch,
        next_args,
    ):
        launch(
            q.data_ptr(), outputs["o"].data_ptr(),
            case["k_cache"].data_ptr(), case["v_cache"].data_ptr(),
            case["table"].data_ptr(), case["table"].stride(0),
            case["prefix"].data_ptr(), 0,
            case["scale_bits"], gate.data_ptr(), outputs["res"].data_ptr(),
            weights["w_o"].data_ptr(), weights["attn_alpha"].data_ptr(),
            weights["attn_beta"].data_ptr(), weights["pre_w"].data_ptr(),
            weights["w_hi"].data_ptr(), weights["w_lo"].data_ptr(),
            weights["bias"].data_ptr(), weights["w_pool"].data_ptr(),
            weights["xn_w"].data_ptr(), outputs["ids"].data_ptr(),
            outputs["scores"].data_ptr(), outputs["xin"].data_ptr(), 0,
            case["w13"].data_ptr(), case["s13"].data_ptr(),
            case["w2"].data_ptr(), case["s2"].data_ptr(),
            weights["pen_w"].data_ptr(), weights["pfw"].data_ptr(),
            weights["ffn_alpha"].data_ptr(), weights["ffn_beta"].data_ptr(),
            weights["npw"].data_ptr(), weights["w_unpool"].data_ptr(),
            outputs["attn_out"].data_ptr(), *next_args, scratch.data_ptr(),
            peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(),
            LAYER, 0, stream=stream,
        )  # fmt: skip

    try:
        for _ in range(2):
            _reset(split, case["base_res"])
            _reset(fused, case["base_res"])
            fused_q.copy_(case["q"])
            fused_g.copy_(case["g"])
            launch_layer(
                layer_v1,
                split,
                case["q"],
                case["g"],
                split_layer_scratch,
                split_peers,
                split_epoch,
                (
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    split["attn_out"].data_ptr(),
                    0,
                    0,
                    0,
                    0,
                ),
            )
            k1(
                split["attn_out"].data_ptr(), next_w.data_ptr(),
                cos_sin.data_ptr(), positions.data_ptr(), slots.data_ptr(),
                swa_map.data_ptr(), split_k_cache.data_ptr(),
                split_v_cache.data_ptr(), split_q.data_ptr(),
                split_k.data_ptr(), split_v.data_ptr(), split_g.data_ptr(),
                split_k1_scratch.data_ptr(), split_epoch.data_ptr(),
                LAYER + 1, 1, 0, 1, 1, stream=stream,
            )  # fmt: skip
            launch_layer(
                layer_v2,
                fused,
                fused_q,
                fused_g,
                fused_layer_scratch,
                fused_peers,
                fused_epoch,
                (
                    next_w.data_ptr(),
                    cos_sin.data_ptr(),
                    positions.data_ptr(),
                    slots.data_ptr(),
                    swa_map.data_ptr(),
                    fused_k_cache.data_ptr(),
                    fused_v_cache.data_ptr(),
                    fused_k.data_ptr(),
                    fused_v.data_ptr(),
                    1,
                    0,
                    1,
                    1,
                ),
            )
            split_epoch[0] += 1
            fused_epoch[0] += 1
        torch.cuda.synchronize()

        _check_outputs(fused, split, "Layer V2 vs Layer V1")
        torch.testing.assert_close(fused_q, split_q, rtol=0, atol=0)
        torch.testing.assert_close(fused_k, split_k, rtol=0, atol=0)
        torch.testing.assert_close(fused_v, split_v, rtol=0, atol=0)
        torch.testing.assert_close(fused_g, split_g, rtol=0, atol=0)
        torch.testing.assert_close(
            fused_k_cache.float(), split_k_cache.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused_v_cache.float(), split_v_cache.float(), rtol=0, atol=0
        )
        q_ref, k_ref, v_ref, g_ref = k1_reference(
            fused["attn_out"], next_w, cos_sin, positions, True
        )
        k1_close(fused_q, q_ref, 1 / 64, "Layer V2 q")
        k1_close(fused_k, k_ref, 1 / 64, "Layer V2 k")
        k1_close(fused_v, v_ref, 1 / 128, "Layer V2 v")
        k1_close(fused_g, g_ref, 1 / 128, "Layer V2 gate")
        assert (
            fused_epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0
        ).all()
    finally:
        split_peers.close()
        fused_peers.close()
