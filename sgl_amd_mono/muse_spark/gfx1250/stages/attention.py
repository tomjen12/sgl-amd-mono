# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 wave32 K-attn for one eight-row TP4 request."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf16_round,
    butterfly,
    hw_exp2,
    rsrc,
)
from sgl_amd_mono.muse_spark.gfx1250.config import (
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    LOG2E,
    MAX_TOKENS,
    Q_K,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    GQA,
    KATTN_OUT_FLOATS,
    KATTN_WAVE_STATS,
    WAVES,
)

NEG = -1.0e30


def _load_bf16x8(resource, element):
    words = fx.Vector(
        bo.buffer_load(resource, element // 2, vec_width=4, dtype=T.i32)
    )
    return words.bitcast(fx.BFloat16).to(fx.Float32)


def _load_fp8x8(resource, byte):
    words = fx.Vector(
        bo.buffer_load(resource, byte // 4, vec_width=2, dtype=T.i32)
    )
    values = []
    for i in range(8):
        raw = fx.Int32((words[i // 4] >> (8 * (i % 4))) & 0xFF)
        sign, exponent, mantissa = raw >> 7, (raw >> 3) & 0xF, raw & 0x7
        normal_bits = ((exponent + 120) << 23) | (mantissa << 20)
        normal = fx.Int32(normal_bits).bitcast(fx.Float32)
        subnormal = fx.Float32(mantissa) * (1.0 / 512.0)
        value = (exponent == 0).select(subnormal, normal)
        values.append((sign != 0).select(-value, value))
    return fx.Vector.from_elements(values, fx.Float32)


def _score(args, qv, key, kv_head, lane8, scale):
    page = fx.Int32(
        bo.buffer_load(
            rsrc(args["table"]), key // 16, vec_width=1, dtype=T.i32
        )
    )
    slot = page * 16 + key % 16
    byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane8 * 8
    kv = _load_fp8x8(rsrc(args["k_cache"]), byte)
    dot = fx.Float32(0.0)
    for i in range_constexpr(8):
        dot = dot + qv[i] * kv[i]
    return butterfly(dot, (4, 2, 1)) * scale, slot


@flyc.jit
def run_kattn(c):
    """One CTA computes one (verify token, local query head) attention row."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    args, stats, partial = c["args"], c["stats"], c["partial"]
    query_token = c["bid"] // LOCAL_Q_HEADS
    query_head = c["bid"] % LOCAL_Q_HEADS
    kv_head = query_head // GQA
    lane8 = lane % 8

    prefix = fx.Int32(
        bo.buffer_load(rsrc(args["prefix_lens"]), 0, vec_width=1, dtype=T.i32)
    )
    end = prefix + query_token + 1
    begin = fx.Int32(0)
    if args["window"] > 0:
        lower = end - args["window"]
        begin = (lower > 0).select(lower, begin)
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    qv = _load_bf16x8(
        rsrc(args["q"]),
        query_token * Q_K + query_head * HEAD_DIM + lane8 * 8,
    )

    # Pass 1: a maximum per wave, then one block maximum.
    local_max = fx.Float32(NEG)
    key = begin + wave
    while key < end:
        score, _ = _score(args, qv, key, kv_head, lane8, scale)
        if lane == 0:
            local_max = (score > local_max).select(score, local_max)
        key = key + WAVES
    if lane == 0:
        fx.ptr_store(local_max, stats + wave)
    gpu.barrier()
    if tid == 0:
        block_max = fx.ptr_load(stats)
        for w in range_constexpr(1, WAVES):
            value = fx.ptr_load(stats + w)
            block_max = (value > block_max).select(value, block_max)
        fx.ptr_store(block_max, stats + 2 * WAVES)
    gpu.barrier()
    block_max = fx.ptr_load(stats + 2 * WAVES)

    # Pass 2: denominator per wave, then the block denominator.
    local_sum = fx.Float32(0.0)
    key = begin + wave
    while key < end:
        score, _ = _score(args, qv, key, kv_head, lane8, scale)
        if lane == 0:
            local_sum = local_sum + hw_exp2((score - block_max) * LOG2E)
        key = key + WAVES
    if lane == 0:
        fx.ptr_store(local_sum, stats + WAVES + wave)
    gpu.barrier()
    if tid == 0:
        block_sum = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            block_sum = block_sum + fx.ptr_load(stats + WAVES + w)
        fx.ptr_store(block_sum, stats + 2 * WAVES + 1)
    gpu.barrier()
    block_sum = fx.ptr_load(stats + 2 * WAVES + 1)

    # Pass 3: BF16 probabilities times V, one partial output per wave.
    output = fx.Vector.filled(8, 0.0, fx.Float32)
    key = begin + wave
    while key < end:
        score, slot = _score(args, qv, key, kv_head, lane8, scale)
        probability = bf16_round(hw_exp2((score - block_max) * LOG2E) / block_sum)
        byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane8 * 8
        value = _load_fp8x8(rsrc(args["v_cache"]), byte)
        output = output + value * probability
        key = key + WAVES
    if lane < 8:
        fx.ptr_store(output, partial + wave * HEAD_DIM + lane * 8)
    gpu.barrier()

    if tid < 8:
        total = fx.Vector.filled(8, 0.0, fx.Float32)
        for w in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                partial + w * HEAD_DIM + tid * 8,
                result_type=fx.Vector.make_type(8, fx.Float32),
            )
        bo.buffer_store(
            total.to(fx.BFloat16),
            rsrc(args["out"]),
            query_token * Q_K + query_head * HEAD_DIM + tid * 8,
        )


def kattn_lds():
    @fx.struct
    class Lds:
        stats: fx.Array[fx.Float32, KATTN_WAVE_STATS, 16]
        partial: fx.Array[fx.Float32, KATTN_OUT_FLOATS, 16]

    return Lds
