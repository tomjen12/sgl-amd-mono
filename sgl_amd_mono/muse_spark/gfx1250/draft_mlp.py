# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 DFlash2 D-MLP monokernel.

This is deliberately a scalar-dot-product implementation.  It keeps the whole
eight-token MLP half in one persistent launch and uses tagged global payloads
for every inter-CTA and inter-rank dependency.  It is therefore useful as the
executable numerical baseline for a later wave32-WMMA optimization without
borrowing any of gfx950's wave64 lane mapping.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import bf2_f32, bf16_pair, bf16_round, butterfly, hw_exp2, hw_rsq, peer_bases, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag, wait_stores
from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
    BLOCK,
    COEF_ROWS,
    CONV_GROUP,
    CONV_GROUPS,
    CONV_TAPS,
    HIDDEN,
    LOCAL_INTERMEDIATE,
    MAX_TOKENS,
    RMS_EPS,
    RESIDENT_BLOCKS,
)

THREADS = 256
WAVE_SIZE = 32
WAVES = THREADS // WAVE_SIZE
H = HIDDEN
I = LOCAL_INTERMEDIATE
GU_ROWS = 2 * I
GROUP = CONV_GROUP
GROUPS = CONV_GROUPS
TAPS = CONV_TAPS
EPS = RMS_EPS
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


def dm_scratch_layout() -> dict:
    """Rank-private tagged payloads. Sizes are in 32-bit mailbox words."""
    return pair_layout(
        [
            ("dm_h", MAX_TOKENS * H // 2),
            ("dm_coef", MAX_TOKENS * COEF_ROWS),
            ("dm_act", MAX_TOKENS * I // 2),
        ],
        align=256,
    )


def dm_sym_layout(npes: int) -> dict:
    if npes not in (1, 4):
        raise ValueError(f"gfx1250 D-MLP supports TP1 or TP4, got TP{npes}")
    return pair_layout([("dm", npes * MAX_TOKENS * H // 2)], align=256)


def _bytes(layout):
    return max(offset + size for offset, size in layout.values())


def dm_scratch_bytes() -> int:
    return _bytes(dm_scratch_layout())


def dm_sym_bytes(npes: int) -> int:
    return _bytes(dm_sym_layout(npes))


class DMLPBuild(msgspec.Struct, frozen=True):
    tokens: int = MAX_TOKENS
    npes: int = 4
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


def _bf(ptr, element):
    word = fx.Int32(bo.buffer_load(rsrc(ptr), element // 2, vec_width=1, dtype=T.i32))
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _store_bf2(ptr, pair, lo, hi):
    bo.buffer_store(bf16_pair(lo, hi).bitcast(fx.Int32), rsrc(ptr), pair)


@flyc.jit
def _block_sum(value, lane, wave, red):
    value = butterfly(value, (16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(value, red + wave)
    gpu.barrier()
    total = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        total = total + fx.ptr_load(red + w)
    gpu.barrier()
    return total


@flyc.jit
def _poll_bf(c, region, element):
    got = c["poll"]([(region, element // 2, 1)])
    lo, hi = bf2_f32(got[0][0])
    return ((element & 1) == 0).select(lo, hi)


@flyc.jit
def _dot_h(c, weight, token, row):
    total = fx.Float32(0.0)
    k = fx.Int32(0)
    while k < H:
        total = total + _bf(weight, row * H + k) * _poll_bf(c, c["h"], token * H + k)
        k = k + 1
    return total


@flyc.jit
def stage_norm(c):
    """bf16 residual add and RMSNorm; one CTA owns each token."""
    bid, tid, lane, wave = c["bid"], c["tid"], c["lane"], c["wave"]
    if bid < c["S"]:
        token = bid
        ss = fx.Float32(0.0)
        vals = []
        for j in range_constexpr(H // (2 * THREADS)):
            element = token * H + 2 * (tid + j * THREADS)
            v0 = bf16_round(_bf(c["args"]["res"], element) + _bf(c["args"]["attn_out"], element))
            v1 = bf16_round(_bf(c["args"]["res"], element + 1) + _bf(c["args"]["attn_out"], element + 1))
            vals += [v0, v1]
            ss = ss + v0 * v0 + v1 * v1
            _store_bf2(c["args"]["res_out"], element // 2, v0, v1)
        rstd = hw_rsq(_block_sum(ss, lane, wave, c["red"]) / float(H) + EPS)
        for j in range_constexpr(H // (2 * THREADS)):
            element = token * H + 2 * (tid + j * THREADS)
            h0 = vals[2 * j] * rstd * _bf(c["args"]["w_post"], element % H)
            h1 = vals[2 * j + 1] * rstd * _bf(c["args"]["w_post"], element % H + 1)
            c["put_bf"](c["h"], element, [h0, h1])


@flyc.jit
def stage_coef(c):
    """Kernel-projection fp32 rows. Each mailbox word is one coefficient."""
    total = c["S"] * COEF_ROWS
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    while task < total:
        token, row = task // COEF_ROWS, task % COEF_ROWS
        value = _dot_h(c, c["args"]["w_kp"], token, row)
        c["put"](c["coef"], token * COEF_ROWS + row, value)
        task = task + stride


@flyc.jit
def _hc(c, token, k):
    group = k // GROUP
    current = _poll_bf(c, c["h"], token * H + k)
    d0 = c["poll"]([(c["coef"], token * COEF_ROWS + group, 1)])[0][0].bitcast(fx.Float32)
    b0 = _bf(c["args"]["base"], k)
    value = (b0 + d0) * current
    if (token % BLOCK) >= 1:
        previous = _poll_bf(c, c["h"], (token - 1) * H + k)
        d1 = c["poll"]([(c["coef"], token * COEF_ROWS + GROUPS + group, 1)])[0][0].bitcast(fx.Float32)
        b1 = _bf(c["args"]["base"], H + k)
        value = value + (b1 + d1) * previous
    return bf16_round(value)


@flyc.jit
def _gu(c, token, row):
    total = fx.Float32(0.0)
    k = fx.Int32(0)
    while k < H:
        total = total + _bf(c["args"]["w_gu"], row * H + k) * _hc(c, token, k)
        k = k + 1
    return bf16_round(total)


@flyc.jit
def stage_act(c):
    """Grouped conv0, gate/up projection and bf16 SiLU activation."""
    total = c["S"] * I // 2
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    while task < total:
        token, pair = task // (I // 2), task % (I // 2)
        dim = 2 * pair
        g0, g1 = _gu(c, token, dim), _gu(c, token, dim + 1)
        u0, u1 = _gu(c, token, I + dim), _gu(c, token, I + dim + 1)
        s0 = bf16_round(g0 / (fx.Float32(1.0) + hw_exp2(g0 * fx.Float32(-1.4426950408889634))))
        s1 = bf16_round(g1 / (fx.Float32(1.0) + hw_exp2(g1 * fx.Float32(-1.4426950408889634))))
        c["put_bf"](c["act"], token * I + dim, [s0 * u0, s1 * u1])
        task = task + stride


@flyc.jit
def _down(c, token, row):
    total = fx.Float32(0.0)
    k = fx.Int32(0)
    while k < I:
        total = total + _bf(c["args"]["w_down"], row * I + k) * _poll_bf(c, c["act"], token * I + k)
        k = k + 1
    return bf16_round(total)


@flyc.jit
def _rank_sum(c, token, element):
    got = c["poll"](
        [
            (c["own_dm"], (src * MAX_TOKENS * H + token * H + element) // 2, 1)
            for src in range(c["W"])
        ]
    )
    lo, hi = bf2_f32(got[0][0])
    for src in range_constexpr(1, c["W"]):
        p0, p1 = bf2_f32(got[src][0])
        lo, hi = lo + p0, hi + p1
    return lo, hi


@flyc.jit
def stage_down_out(c):
    """Down projection, TP rank-ordered reduction, grouped conv1 and output."""
    total = c["S"] * H // 2
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    while task < total:
        token, pair = task // (H // 2), task % (H // 2)
        element = 2 * pair
        d0, d1 = _down(c, token, element), _down(c, token, element + 1)
        mailbox_element = (c["rank"] * MAX_TOKENS + token) * H + element
        for peer in range_constexpr(c["W"]):
            c["put_bf"](c["peer_dm"][peer], mailbox_element, [d0, d1])
        # In TP1 the producer immediately consumes its own global payload.  The
        # explicit gfx1250 store wait prevents the tag dword becoming observable
        # before the adjacent bf16 payload dword has retired.
        wait_stores()
        v0, v1 = _rank_sum(c, token, element)
        group = element // GROUP
        q0 = c["poll"]([(c["coef"], token * COEF_ROWS + 2 * GROUPS + group, 1)])[0][0].bitcast(fx.Float32)
        b0, b1 = _bf(c["args"]["base"], 2 * H + element), _bf(c["args"]["base"], 2 * H + element + 1)
        o0, o1 = (b0 + q0) * v0, (b1 + q0) * v1
        if (token % BLOCK) >= 1:
            p0, p1 = _rank_sum(c, token - 1, element)
            q1 = c["poll"]([(c["coef"], token * COEF_ROWS + 3 * GROUPS + group, 1)])[0][0].bitcast(fx.Float32)
            b0 = _bf(c["args"]["base"], 3 * H + element)
            b1 = _bf(c["args"]["base"], 3 * H + element + 1)
            o0, o1 = o0 + (b0 + q1) * p0, o1 + (b1 + q1) * p1
        _store_bf2(c["args"]["out"], (token * H + element) // 2, o0, o1)
        task = task + stride


def build_dmlp(key: DMLPBuild):
    if key.tokens != MAX_TOKENS:
        raise ValueError(f"first gfx1250 D-MLP supports width {MAX_TOKENS}, got {key.tokens}")
    if key.npes not in (1, 4):
        raise ValueError(f"gfx1250 D-MLP supports TP1 or TP4, got TP{key.npes}")
    if key.resident_blocks not in (96, 192):
        raise ValueError(f"gfx1250 D-MLP CU profile must be 96 or 192, got {key.resident_blocks}")
    if key.resident_blocks != RESIDENT_BLOCKS:
        raise ValueError(f"active profile has {RESIDENT_BLOCKS} CUs, got {key.resident_blocks}")

    layout, sy = dm_scratch_layout(), dm_sym_layout(key.npes)
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol(
        "muse_spark_gfx1250_dmlp",
        s=key.tokens,
        w=key.npes,
        c=key.resident_blocks,
        f=key.fault_site,
    )

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dmlp(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64,
        w_gu: Int64, w_down: Int64, res_out: Int64, out: Int64, scratch: Int64,
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        active_guard = mailbox_guard(
            epoch, own=sym, peers=peers, rank=rank, npes=key.npes,
            flags_offset=dm_sym_bytes(key.npes), fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
        mb = Mailbox(step_tag(epoch, layer), active_guard)
        bases = peer_bases(peers, key.npes)
        c = {
            "S": key.tokens, "W": key.npes, "B": key.resident_blocks,
            "tid": tid, "bid": bid, "lane": tid % WAVE_SIZE, "wave": tid // WAVE_SIZE,
            "rank": rank, "red": lds.red.ptr, "put": mb.put, "put_bf": mb.put_bf,
            "poll": mb.poll, "h": sreg(scratch, layout["dm_h"][0], "dm_h"),
            "coef": sreg(scratch, layout["dm_coef"][0], "dm_coef"),
            "act": sreg(scratch, layout["dm_act"][0], "dm_act"),
            "own_dm": preg(sym, sy["dm"][0], "dm"),
            "peer_dm": [preg(basep, sy["dm"][0], "dm") for basep in bases],
            "args": {
                "res": res, "attn_out": attn_out, "w_post": w_post, "w_kp": w_kp,
                "base": base, "w_gu": w_gu, "w_down": w_down,
                "res_out": res_out, "out": out,
            },
        }
        stage_norm(c)
        stage_coef(c)
        stage_act(c)
        stage_down_out(c)

    @flyc.jit
    def launch(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64,
        w_gu: Int64, w_down: Int64, res_out: Int64, out: Int64, scratch: Int64,
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dmlp(
            res, attn_out, w_post, w_kp, base, w_gu, w_down, res_out, out,
            scratch, sym, peers, rank, epoch, layer,
        ).launch(grid=(key.resident_blocks,), block=(THREADS,), stream=stream)

    return launch
