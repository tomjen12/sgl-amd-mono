# SPDX-License-Identifier: Apache-2.0
"""Wave32-WMMA stages for the fixed-width gfx1250 Muse Spark K2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    rsrc,
)
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)

THREADS = 256
WAVES = THREADS // 32
ROWS = 16
WMMA_K = 32
WMMA_ACC = 8
RED_FLOATS = WAVES * 32 * WMMA_ACC


def _bf(ptr, element):
    word = fx.Int32(
        bo.buffer_load(rsrc(ptr), element // 2, vec_width=1, dtype=T.i32)
    )
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _f32(ptr, element):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), element, vec_width=1, dtype=T.f32)
    )


def _st_bf2(ptr, pair, lo, hi, cache_modifier=0):
    bo.buffer_store(
        bf16_pair(lo, hi).bitcast(fx.Int32),
        rsrc(ptr),
        pair,
        cache_modifier=cache_modifier,
    )


def _sigmoid(x):
    return fx.Float32(1.0) / (
        fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E))
    )


def _fragment(resource, base, k0, lane, cache_modifier=0):
    kh = (lane // 16) * 8
    lo = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + 16 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    return fx.Vector.from_elements(
        [lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32
    ).bitcast(fx.BFloat16)


def _wmma(a, b, acc):
    result = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(WMMA_ACC, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(acc),
        reuseA=False,
        reuseB=False,
    ).result
    return fx.Vector(result)


def _wave_value(red, wave, row, token):
    lane = ((row % ROWS) // 8) * 16 + token
    elem = row % 8
    return fx.ptr_load(red + (wave * 32 + lane) * WMMA_ACC + elem)


@flyc.jit
def _block_sum(value, lane, wave, red):
    value = butterfly(value, (16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(value, red + wave)
    gpu.barrier()
    total = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        total = total + fx.ptr_load(red + w)
    return total


@flyc.jit
def stage_norm_gate(c):
    """Per-head RMS norm and sigmoid gate, rounded once to bf16."""
    lane = c["lane"]
    head_task = c["bid"]
    token, head = head_task // (Q_K // HEAD_DIM), head_task % (Q_K // HEAD_DIM)
    base = token * Q_K + head * HEAD_DIM + 2 * lane
    o0, o1 = _bf(c["o"], base), _bf(c["o"], base + 1)
    g0, g1 = _bf(c["g"], base), _bf(c["g"], base + 1)
    ss = o0 * o0 + o1 * o1
    ss = butterfly(ss, (16, 8, 4, 2, 1))
    rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
    _st_bf2(
        c["og"],
        base // 2,
        o0 * rstd * _sigmoid(g0),
        o1 * rstd * _sigmoid(g1),
    )


@flyc.jit
def stage_o_projection(c):
    """Eight full-K O tiles per CTA: one independent tile per wave."""
    lane, wave = c["lane"], c["wave"]
    token = fx.min(lane % ROWS, MAX_TOKENS - 1)
    r_w, r_x = rsrc(c["w_o"]), rsrc(c["og"])
    tile = c["bid"] * WAVES + wave
    row0 = tile * ROWS
    row = row0 + lane % ROWS
    acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range_constexpr(Q_K // WMMA_K):
        k0 = step * WMMA_K
        acc = _wmma(
            _fragment(r_w, row * Q_K, k0, lane, cache_modifier=2),
            _fragment(r_x, token * Q_K, k0, lane),
            acc,
        )
    fx.ptr_store(acc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
    gpu.barrier()
    for p in range_constexpr(2):
        out = c["tid"] + p * THREADS
        tile_wave = out // (MAX_TOKENS * (ROWS // 2))
        within = out % (MAX_TOKENS * (ROWS // 2))
        out_token = within // (ROWS // 2)
        dim = 2 * (within % (ROWS // 2))
        element = (c["bid"] * WAVES + tile_wave) * ROWS + dim
        a0 = bf16_round(_wave_value(c["red"], tile_wave, dim, out_token))
        a1 = bf16_round(
            _wave_value(c["red"], tile_wave, dim + 1, out_token)
        )
        peer_element = (
            c["rank"] * MAX_TOKENS + out_token
        ) * HIDDEN + element
        for peer in range_constexpr(c["W"]):
            c["put_bf"](
                c["peer_attn"][peer], peer_element, [a0, a1]
            )
    gpu.barrier()


@flyc.jit
def stage_boundary(c):
    """Rank-ordered partial sum, post norm/residual update, and pre-FFN norm."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    token = c["bid"]
    vals = []
    ss = fx.Float32(0.0)
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        pair = token * (HIDDEN // 2) + tid + j * THREADS
        got = c["poll"](
            [
                (
                    c["own_attn"],
                    src * (MAX_TOKENS * HIDDEN // 2) + pair,
                    1,
                )
                for src in range(c["W"])
            ]
        )
        v0, v1 = bf2_f32(got[0][0])
        for src in range_constexpr(1, c["W"]):
            p0, p1 = bf2_f32(got[src][0])
            v0, v1 = v0 + p0, v1 + p1
        v0, v1 = bf16_round(v0), bf16_round(v1)
        vals += [v0, v1]
        ss = ss + v0 * v0 + v1 * v1
    rstd = hw_rsq(_block_sum(ss, lane, wave, c["red"]) / float(HIDDEN) + POST_EPS)
    new = []
    ss2 = fx.Float32(0.0)
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        for q in range_constexpr(2):
            old = _f32(c["res"], token * HIDDEN + e + q)
            value = (
                _f32(c["alpha"], e + q) * old
                + _f32(c["beta"], e + q) * bf16_round(vals[2 * j + q] * rstd)
            )
            bo.buffer_store(value, rsrc(c["res"]), token * HIDDEN + e + q)
            new.append(value)
            ss2 = ss2 + value * value
    gpu.barrier()
    rstd2 = hw_rsq(
        _block_sum(ss2, lane, wave, c["red"]) / float(HIDDEN) + RMS_EPS
    )
    for j in range_constexpr(HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        _st_bf2(
            c["x"],
            (token * HIDDEN + e) // 2,
            new[2 * j] * rstd2 * _bf(c["pre_w"], e),
            new[2 * j + 1] * rstd2 * _bf(c["pre_w"], e + 1),
        )


@flyc.jit
def stage_router_pool(c):
    """Eight full-K router/pooler tiles per CTA, one per wave."""
    lane, wave = c["lane"], c["wave"]
    token = fx.min(lane % ROWS, MAX_TOKENS - 1)
    r_x, r_w0 = rsrc(c["x"]), rsrc(c["w0"])
    r_w1 = rsrc(c["w1"]) if c["router"] else r_w0
    tile = c["bid"] * WAVES + wave
    row0 = tile * ROWS
    row = row0 + lane % ROWS
    vacc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
    for step in range_constexpr(HIDDEN // WMMA_K):
        k0 = step * WMMA_K
        x = _fragment(r_x, token * HIDDEN, k0, lane)
        vacc = _wmma(
            _fragment(r_w0, row * HIDDEN, k0, lane, cache_modifier=2),
            x,
            vacc,
        )
        if c["router"]:
            vacc = _wmma(
                _fragment(r_w1, row * HIDDEN, k0, lane, cache_modifier=2),
                x,
                vacc,
            )
    fx.ptr_store(vacc, c["red"] + (wave * 32 + lane) * WMMA_ACC)
    gpu.barrier()
    if c["router"]:
        for p in range_constexpr(4):
            out = c["tid"] + p * THREADS
            tile_wave = out // (MAX_TOKENS * ROWS)
            within = out % (MAX_TOKENS * ROWS)
            out_token, dim = within // ROWS, within % ROWS
            value = _wave_value(c["red"], tile_wave, dim, out_token)
            bo.buffer_store(
                value,
                rsrc(c["out"]),
                out_token * EXPERTS
                + (c["bid"] * WAVES + tile_wave) * ROWS
                + dim,
            )
    else:
        for p in range_constexpr(2):
            out = c["tid"] + p * THREADS
            tile_wave = out // (MAX_TOKENS * (ROWS // 2))
            within = out % (MAX_TOKENS * (ROWS // 2))
            out_token = within // (ROWS // 2)
            dim = 2 * (within % (ROWS // 2))
            row = (c["bid"] * WAVES + tile_wave) * ROWS + dim
            _st_bf2(
                c["out"],
                (out_token * MOE_HIDDEN + row) // 2,
                _wave_value(c["red"], tile_wave, dim, out_token),
                _wave_value(c["red"], tile_wave, dim + 1, out_token),
            )
    gpu.barrier()


@flyc.jit
def stage_top8(c):
    """Stable repeated argmax, preserving the lower-id tie rule."""
    if c["tid"] == 0:
        token = c["bid"]
        selected = [fx.Int32(-1) for _ in range(TOP_K)]
        denom = fx.Float32(0.0)
        for pick in range_constexpr(TOP_K):
            best, best_id = fx.Float32(-1.0e30), fx.Int32(0)
            expert = fx.Int32(0)
            while expert < EXPERTS:
                used = fx.Int32(0)
                for q in range_constexpr(pick):
                    used = used | fx.Int32(expert == selected[q])
                score = _sigmoid(
                    _f32(c["logits"], token * EXPERTS + expert)
                    * fx.Float32(ROUTER_MULT)
                )
                biased = score + _f32(c["bias"], expert)
                take = (used == 0) & (
                    (biased > best) | ((biased == best) & (expert < best_id))
                )
                best, best_id = take.select(biased, best), take.select(expert, best_id)
                expert = expert + 1
            selected[pick] = best_id
            score = _sigmoid(
                _f32(c["logits"], token * EXPERTS + best_id)
                * fx.Float32(ROUTER_MULT)
            )
            fx.ptr_store(score, c["picked"] + pick)
            denom = denom + score
        for pick in range_constexpr(TOP_K):
            bo.buffer_store(
                selected[pick], rsrc(c["ids"]), token * TOP_K + pick
            )
            bo.buffer_store(
                fx.ptr_load(c["picked"] + pick)
                / (denom + fx.Float32(ROUTE_EPS))
                * fx.Float32(ROUTE_SCALE),
                rsrc(c["scores"]),
                token * TOP_K + pick,
            )


@flyc.jit
def stage_pooled_norm(c):
    """RMS norm the rounded pooler row and apply its bf16 weight."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    token = c["bid"]
    vals = []
    ss = fx.Float32(0.0)
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        v0 = _bf(c["pool"], token * MOE_HIDDEN + e)
        v1 = _bf(c["pool"], token * MOE_HIDDEN + e + 1)
        vals += [v0, v1]
        ss = ss + v0 * v0 + v1 * v1
    rstd = hw_rsq(
        _block_sum(ss, lane, wave, c["red"]) / float(MOE_HIDDEN) + RMS_EPS
    )
    for j in range_constexpr(MOE_HIDDEN // (2 * THREADS)):
        e = 2 * (tid + j * THREADS)
        _st_bf2(
            c["xin"],
            (token * MOE_HIDDEN + e) // 2,
            vals[2 * j] * rstd * _bf(c["xn_w"], e),
            vals[2 * j + 1] * rstd * _bf(c["xn_w"], e + 1),
        )
