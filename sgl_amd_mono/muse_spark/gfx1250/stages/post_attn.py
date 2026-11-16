# SPDX-License-Identifier: Apache-2.0
"""Correctness-first wave32 stages for the fixed-width Muse Spark K2.

The implementation intentionally uses scalar dot products and separate kernel
launches.  It is the executable numerical specification for a later WMMA port:
every launch boundary is also a model-operation rounding boundary.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
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
    """Two output rows per thread, then push this rank's bf16 partial to all ranks."""
    pair = c["bid"] * THREADS + c["tid"]
    if pair < MAX_TOKENS * HIDDEN // 2:
        token, row0 = pair // (HIDDEN // 2), 2 * (pair % (HIDDEN // 2))
        a0, a1 = fx.Float32(0.0), fx.Float32(0.0)
        k = fx.Int32(0)
        while k < Q_K:
            x = _bf(c["og"], token * Q_K + k)
            a0 = a0 + _bf(c["w_o"], row0 * Q_K + k) * x
            a1 = a1 + _bf(c["w_o"], (row0 + 1) * Q_K + k) * x
            k = k + 1
        a0, a1 = bf16_round(a0), bf16_round(a1)
        element = (c["rank"] * MAX_TOKENS + token) * HIDDEN + row0
        for peer in range_constexpr(c["W"]):
            c["put_bf"](c["peer_attn"][peer], element, [a0, a1])


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
    """One scalar, fixed-order dot product per router or pooler output."""
    out = c["bid"] * THREADS + c["tid"]
    total = c["rows"] * MAX_TOKENS
    if out < total:
        token, row = out // c["rows"], out % c["rows"]
        acc = fx.Float32(0.0)
        k = fx.Int32(0)
        while k < HIDDEN:
            xv = _bf(c["x"], token * HIDDEN + k)
            if c["router"]:
                acc = acc + xv * _bf(c["w0"], row * HIDDEN + k)
                acc = acc + xv * _bf(c["w1"], row * HIDDEN + k)
            else:
                acc = acc + xv * _bf(c["w0"], row * HIDDEN + k)
            k = k + 1
        if c["router"]:
            bo.buffer_store(acc, rsrc(c["out"]), token * EXPERTS + row)
        else:
            # Adjacent threads own adjacent rows.  Only even rows publish the pair.
            fx.ptr_store(acc, c["red"] + c["tid"])
            gpu.barrier()
            if (row & 1) == 0:
                _st_bf2(
                    c["out"],
                    (token * MOE_HIDDEN + row) // 2,
                    acc,
                    fx.ptr_load(c["red"] + c["tid"] + 1),
                )


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
