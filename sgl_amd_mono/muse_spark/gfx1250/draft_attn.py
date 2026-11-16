# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 DFlash2 TP4 draft attention.

One CTA computes one of the 8 x 16 (token, local query head) rows.  Its eight
wave32 waves split the keys and use three deliberately simple passes: maximum,
normalizer, and BF16-probability times V.  Every row attends the same noncausal
eight-row draft block plus the compact suffix of the prefix selected by
``window``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import bf16_round, butterfly, hw_exp2, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard
from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
    BLOCK,
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    RESIDENT_BLOCKS,
)
from sgl_amd_mono.muse_spark.gfx1250.draft_layout import THREADS, WAVES

GQA = LOCAL_Q_HEADS // LOCAL_KV_HEADS
TASKS = BLOCK * LOCAL_Q_HEADS
LOG2E = 1.4426950408889634
NEG = -1.0e30
STATS_FLOATS = 2 * WAVES + 2
PARTIAL_FLOATS = WAVES * HEAD_DIM

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


def _load_bf16x8(resource, element):
    words = fx.Vector(
        bo.buffer_load(resource, element // 2, vec_width=4, dtype=T.i32)
    )
    return words.bitcast(fx.BFloat16).to(fx.Float32)


def _load_fp8x8(resource, byte):
    """Load and widen eight e4m3fn values without wave64-only conversion ops."""
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


def _i32_at(address, index, is_i64):
    word = (is_i64 != 0).select(2 * index, index)
    return fx.Int32(
        bo.buffer_load(rsrc(address), word, vec_width=1, dtype=T.i32)
    )


def _score(args, qv, request_row, key, kv_head, lane16, scale):
    slot = fx.Int32(
        bo.buffer_load(
            rsrc(args["req_to_token"], args["table_bytes"]),
            request_row * args["table_stride"] + key,
            vec_width=1,
            dtype=T.i32,
        )
    )
    byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane16 * 8
    kval = _load_fp8x8(rsrc(args["k_cache"]), byte)
    dot = fx.Float32(0.0)
    for i in range_constexpr(8):
        dot = dot + qv[i] * kval[i]
    return butterfly(dot, (8, 4, 2, 1)) * scale, slot


@flyc.jit
def run_dattn(c):
    """Compute one noncausal draft-block attention row."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    args, stats, partial = c["args"], c["stats"], c["partial"]
    token = c["task"] // LOCAL_Q_HEADS
    query_head = c["task"] % LOCAL_Q_HEADS
    kv_head = query_head // GQA
    lane16 = lane % 16

    prefix = _i32_at(args["seq_lens"], 0, args["lens64"])
    keys = prefix + BLOCK
    request_row = _i32_at(args["req_pool"], 0, args["rpi64"])
    begin = fx.Int32(0)
    if args["window"] > 0:
        lower = prefix - args["window"]
        begin = (lower > 0).select(lower, begin)
    scale = fx.Int32(args["scale_bits"]).bitcast(fx.Float32)
    qv = _load_bf16x8(
        rsrc(args["q"]),
        token * Q_K + query_head * HEAD_DIM + lane16 * 8,
    )

    local_max = fx.Float32(NEG)
    key = begin + wave
    while key < keys:
        score, _ = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
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

    local_sum = fx.Float32(0.0)
    key = begin + wave
    while key < keys:
        score, _ = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
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

    output = fx.Vector.filled(8, 0.0, fx.Float32)
    key = begin + wave
    while key < keys:
        score, slot = _score(
            args, qv, request_row, key, kv_head, lane16, scale
        )
        probability = bf16_round(
            hw_exp2((score - block_max) * LOG2E) / block_sum
        )
        byte = (slot * LOCAL_KV_HEADS + kv_head) * HEAD_DIM + lane16 * 8
        value = _load_fp8x8(rsrc(args["v_cache"]), byte)
        output = output + value * probability
        key = key + WAVES
    if lane < HEAD_DIM // 8:
        fx.ptr_store(output, partial + wave * HEAD_DIM + lane * 8)
    gpu.barrier()

    if tid < HEAD_DIM // 8:
        total = fx.Vector.filled(8, 0.0, fx.Float32)
        for w in range_constexpr(WAVES):
            total = total + fx.ptr_load(
                partial + w * HEAD_DIM + tid * 8,
                result_type=fx.Vector.make_type(8, fx.Float32),
            )
        bo.buffer_store(
            total.to(fx.BFloat16),
            rsrc(args["out"]),
            token * Q_K + query_head * HEAD_DIM + tid * 8,
        )


def dattn_lds():
    @fx.struct
    class Lds:
        stats: fx.Array[fx.Float32, STATS_FLOATS, 16]
        partial: fx.Array[fx.Float32, PARTIAL_FLOATS, 16]

    return Lds


class DAttnBuild(msgspec.Struct, frozen=True):
    tokens: int = BLOCK
    q_len: int = BLOCK
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


def build_dattn(key: DAttnBuild):
    """Build standalone gfx1250 D-attn for the initial width/q_len-8 profile."""
    if (key.tokens, key.q_len) != (MAX_TOKENS, BLOCK):
        raise ValueError(
            f"gfx1250 D-attn supports tokens=q_len={BLOCK}, "
            f"got {(key.tokens, key.q_len)}"
        )
    if key.resident_blocks not in (96, 192):
        raise ValueError(
            "gfx1250 D-attn resident_blocks must be 96 or 192, "
            f"got {key.resident_blocks}"
        )

    name = kernel_symbol(
        "muse_spark_gfx1250_dattn",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
        f=key.fault_site,
    )
    keyed = key_tuple(key, SOURCES)
    Lds = dattn_lds()
    rounds = (TASKS + key.resident_blocks - 1) // key.resident_blocks

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        epoch: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        args = {
            "q": q,
            "out": out,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "req_to_token": req_to_token,
            "table_stride": table_stride,
            "table_bytes": table_bytes,
            "req_pool": req_pool,
            "rpi64": rpi64,
            "seq_lens": seq_lens,
            "lens64": lens64,
            "window": window,
            "scale_bits": scale_bits,
        }
        active_guard = mailbox_guard(
            epoch,
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
        with guard.scope(active_guard):
            for task_round in range_constexpr(rounds):
                task = bid + task_round * key.resident_blocks
                if task < TASKS:
                    run_dattn(
                        {
                            "tid": tid,
                            "lane": tid % 32,
                            "wave": tid // 32,
                            "task": task,
                            "stats": lds.stats.ptr,
                            "partial": lds.partial.ptr,
                            "args": args,
                        }
                    )

    @flyc.jit
    def launch(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        req_to_token: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        req_pool: Int64,
        rpi64: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dattn(
            q,
            out,
            k_cache,
            v_cache,
            req_to_token,
            table_stride,
            table_bytes,
            req_pool,
            rpi64,
            seq_lens,
            lens64,
            window,
            scale_bits,
            epoch,
        ).launch(
            grid=(key.resident_blocks,),
            block=(THREADS,),
            stream=stream,
        )

    return launch
