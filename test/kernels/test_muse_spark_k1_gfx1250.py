"""The gfx1250 TP4 Muse Spark K1 launch against the fused ROCm model math.

This test is intentionally separate from ``test_muse_spark_k1.py``: that file
pins the gfx950 TP8/wave64 implementation, while this one pins the gfx1250
TP4/wave32 BS1 implementation.
"""

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
POSITIONS = 4096
THETA = 500000.0


def _cos_sin(device):
    from sgl_amd_mono.muse_spark.gfx1250.config import HEAD_DIM

    inv = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.arange(POSITIONS, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


def _bf(t):
    return t.to(torch.bfloat16).float()


def _reference(x, w, cos_sin, positions, rope):
    """Reference the SGLang fused qkv projection, norm, scale and RoPE order."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        G_OFF,
        HEAD_DIM,
        K_OFF,
        LOCAL_KV_HEADS,
        LOCAL_Q_HEADS,
        QK_SCALE,
        RMS_EPS,
        V_OFF,
    )

    y = (x.float() @ w.float().T).to(torch.bfloat16)
    rows = x.shape[0]
    q = y[:, :K_OFF].reshape(rows, LOCAL_Q_HEADS, HEAD_DIM)
    k = y[:, K_OFF:V_OFF].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM)
    v = y[:, V_OFF:G_OFF].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM)
    g = y[:, G_OFF:]

    def norm(t):
        tf = t.float()
        rstd = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return _bf(tf * rstd)

    def rotate(value):
        if not rope:
            return value.to(torch.bfloat16)
        cs = cos_sin[positions]
        dim = torch.arange(HEAD_DIM, device=value.device)
        cos = _bf(cs[:, None, dim // 2])
        sin = _bf(cs[:, None, HEAD_DIM // 2 + dim // 2])
        other = (
            value.reshape(rows, value.shape[1], HEAD_DIM // 2, 2)
            .flip(-1)
            .reshape(rows, value.shape[1], HEAD_DIM)
        )
        rotated = torch.where(dim % 2 == 0, -other, other)
        return (_bf(value * cos) + _bf(rotated * sin)).to(torch.bfloat16)

    q = rotate(_bf(norm(q) * QK_SCALE))
    k = rotate(norm(k))
    return q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1), g


def _fp8(t):
    return t.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _close(got, ref, rtol, what, flips=1e-3):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    bad = (err > rtol * ref.abs() + 1e-3).float().mean().item()
    worst = (err > 4 * rtol * ref.abs() + 1e-3).float().mean().item()
    assert bad <= flips and worst == 0.0, (
        f"{what}: {bad:.3%} exceed {rtol} relative; "
        f"{worst:.3%} exceed {4 * rtol}"
    )


def _one_ulp_fp8(got, ref, what):
    gotf, reff = got.float(), ref.float()
    assert ((gotf - reff).abs() <= reff.abs() / 8 + 1e-3).all(), (
        f"{what}: a cache value differs by more than one fp8 code"
    )
    exact = (gotf == reff).float().mean().item()
    assert exact > 0.95, f"{what}: only {exact:.1%} match exactly"


def _epoch(device):
    from sgl_amd_mono.common import guard

    out = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(30.0)
    lo = ticks & 0xFFFFFFFF
    out[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    out[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return out


@pytest.mark.parametrize(
    "rope,swa,slot64",
    [(True, True, True), (False, False, False)],
)
def test_gfx1250_k1_matches_the_fused_kernel_math(rope, swa, slot64):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        HIDDEN,
        KV_K,
        LOCAL_KV_HEADS,
        MAX_TOKENS,
        Q_K,
        QKVG_ROWS,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx1250.layout import (
        k1_scratch_layout,
        scratch_bytes,
    )

    rows = MAX_TOKENS
    torch.manual_seed(1250 + int(rope) + 2 * int(swa))
    device, bf16 = torch.device("cuda"), torch.bfloat16
    x = torch.randn(rows, HIDDEN, device=device).to(bf16)
    w = (torch.randn(QKVG_ROWS, HIDDEN, device=device) * 0.02).to(bf16)
    cos_sin = _cos_sin(device)
    positions = torch.randint(0, POSITIONS, (rows,), device=device, dtype=torch.int64)
    full_slots = torch.randperm(SLOTS, device=device)[:rows]
    slots = full_slots if slot64 else full_slots.to(torch.int32)
    swa_map = torch.randperm(SLOTS, device=device, dtype=torch.int64)

    k_cache = torch.zeros(
        SLOTS, LOCAL_KV_HEADS, HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    q = torch.full((MAX_TOKENS, Q_K), float("nan"), device=device, dtype=bf16)
    k = torch.full((MAX_TOKENS, KV_K), float("nan"), device=device, dtype=bf16)
    v = torch.full_like(k, float("nan"))
    g = torch.full_like(q, float("nan"))
    scratch = torch.zeros(
        scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=device
    )
    epoch = _epoch(device)
    launch = build_k1(K1Build(tokens=rows))

    for _ in range(2):
        launch(
            x.data_ptr(),
            w.data_ptr(),
            cos_sin.data_ptr(),
            positions.data_ptr(),
            slots.data_ptr(),
            swa_map.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            g.data_ptr(),
            scratch.data_ptr(),
            epoch.data_ptr(),
            3,
            int(rope),
            int(swa),
            int(slot64),
            1,
            stream=torch.cuda.current_stream(),
        )
        epoch[0] += 1
    torch.cuda.synchronize()
    from sgl_amd_mono.common import guard

    assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all()

    q_ref, k_ref, v_ref, g_ref = _reference(x, w, cos_sin, positions, rope)
    _close(q, q_ref, 1 / 64, "q")
    _close(k, k_ref, 1 / 64, "k")
    _close(v, v_ref, 1 / 128, "v")
    _close(g, g_ref, 1 / 128, "gate")

    cache_slots = swa_map[full_slots] if swa else full_slots
    _one_ulp_fp8(k_cache[cache_slots].reshape(rows, -1), _fp8(k_ref), "K cache")
    _one_ulp_fp8(v_cache[cache_slots].reshape(rows, -1), _fp8(v_ref), "V cache")
    untouched = torch.ones(SLOTS, dtype=torch.bool, device=device)
    untouched[cache_slots] = False
    assert (k_cache[untouched].float() == 0).all()
    assert (v_cache[untouched].float() == 0).all()
