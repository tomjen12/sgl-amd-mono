# SPDX-License-Identifier: Apache-2.0
"""Correctness-first gfx1250 DFlash2 attention boundary monokernels.

The implementation is intentionally independent of gfx950's wave64 MFMA lane
layout.  Work is distributed over the selected 96/192 resident-CTA profile and
all cross-CTA/rank dependencies use bounded, generation-tagged mailboxes.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    peer_bases,
    rsrc,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag
from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
    BLOCK,
    COEF_ROWS,
    CONV_GROUP,
    CONV_GROUPS,
    CONV_TAPS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    QKV_ROWS,
    Q_K,
    RESIDENT_BLOCKS,
    RMS_EPS,
)

THREADS = 256
WAVE_SIZE = 32
WAVES = THREADS // WAVE_SIZE
H = HIDDEN
GROUP = CONV_GROUP
GROUPS = CONV_GROUPS
TAPS = CONV_TAPS
EPS = RMS_EPS
KV_K = LOCAL_KV_HEADS * HEAD_DIM
QKV_HEADS = LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


def dp_scratch_layout() -> dict:
    """D-pre tagged bf16 h/hc/qkv rows and fp32 coefficients."""
    return pair_layout(
        [
            ("dp_h", MAX_TOKENS * H // 2),
            ("dp_hc", MAX_TOKENS * H // 2),
            ("dp_qkv", MAX_TOKENS * QKV_ROWS // 2),
            ("dp_coef", MAX_TOKENS * COEF_ROWS),
        ],
        align=256,
    )


def do_scratch_layout() -> dict:
    """D-post needs no private payload, but retains an aligned guardable region."""
    return pair_layout([("do_pad", 1)], align=256)


def do_sym_layout(npes: int) -> dict:
    if npes not in (1, 4):
        raise ValueError(f"gfx1250 D-post supports TP1 or TP4, got TP{npes}")
    return pair_layout([("do", npes * MAX_TOKENS * H // 2)], align=256)


def _bytes(layout: dict) -> int:
    return max(offset + size for offset, size in layout.values())


def dp_scratch_bytes() -> int:
    return _bytes(dp_scratch_layout())


def do_scratch_bytes() -> int:
    return _bytes(do_scratch_layout())


def do_sym_bytes(npes: int) -> int:
    return _bytes(do_sym_layout(npes))


class DPreBuild(msgspec.Struct, frozen=True):
    tokens: int = MAX_TOKENS
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


class DPostBuild(msgspec.Struct, frozen=True):
    tokens: int = MAX_TOKENS
    npes: int = 4
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


def _validate(tokens: int, resident_blocks: int):
    if tokens != MAX_TOKENS:
        raise ValueError(f"first gfx1250 D-attn half supports width {MAX_TOKENS}, got {tokens}")
    if resident_blocks not in (96, 192):
        raise ValueError(f"gfx1250 CU profile must be 96 or 192, got {resident_blocks}")
    if resident_blocks != RESIDENT_BLOCKS:
        raise ValueError(f"active profile has {RESIDENT_BLOCKS} CUs, got {resident_blocks}")


def _bf(ptr, element):
    word = fx.Int32(bo.buffer_load(rsrc(ptr), element // 2, vec_width=1, dtype=T.i32))
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _store_bf2(ptr, pair, lo, hi):
    bo.buffer_store(bf16_pair(lo, hi).bitcast(fx.Int32), rsrc(ptr), pair)


@flyc.jit
def _poll_bf(c, region, element):
    got = c["poll"]([(region, element // 2, 1)])
    lo, hi = bf2_f32(got[0][0])
    return ((element & 1) == 0).select(lo, hi)


@flyc.jit
def _block_sum(value, lane, wave, red):
    # Native wave32 butterfly, followed by an eight-wave LDS reduction.
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
def stage_norm_in(c):
    """Residual boundary and input RMSNorm; one CTA owns one token."""
    bid, tid, lane, wave = c["bid"], c["tid"], c["lane"], c["wave"]
    if bid < c["S"]:
        token = bid
        ss = fx.Float32(0.0)
        values = []
        for j in range_constexpr(H // (2 * THREADS)):
            element = token * H + 2 * (tid + j * THREADS)
            x0, x1 = _bf(c["args"]["x"], element), _bf(c["args"]["x"], element + 1)
            r0, r1 = _bf(c["args"]["res"], element), _bf(c["args"]["res"], element + 1)
            v0 = (c["args"]["first"] != 0).select(x0, bf16_round(r0 + x0))
            v1 = (c["args"]["first"] != 0).select(x1, bf16_round(r1 + x1))
            values += [v0, v1]
            ss = ss + v0 * v0 + v1 * v1
            _store_bf2(c["args"]["res_out"], element // 2, v0, v1)
        rstd = hw_rsq(_block_sum(ss, lane, wave, c["red"]) / float(H) + EPS)
        for j in range_constexpr(H // (2 * THREADS)):
            element = token * H + 2 * (tid + j * THREADS)
            h0 = values[2 * j] * rstd * _bf(c["args"]["w_in"], element % H)
            h1 = values[2 * j + 1] * rstd * _bf(c["args"]["w_in"], element % H + 1)
            c["put_bf"](c["h"], element, [h0, h1])


@flyc.jit
def stage_coef(c):
    """Kernel projection, with adjacent rows sharing each tagged h load."""
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    total = c["S"] * COEF_ROWS // 2
    while task < total:
        token, pair = task // (COEF_ROWS // 2), task % (COEF_ROWS // 2)
        row0 = 2 * pair
        a0, a1 = fx.Float32(0.0), fx.Float32(0.0)
        k = fx.Int32(0)
        while k < H:
            x = _poll_bf(c, c["h"], token * H + k)
            a0 = a0 + _bf(c["args"]["w_kp"], row0 * H + k) * x
            a1 = a1 + _bf(c["args"]["w_kp"], (row0 + 1) * H + k) * x
            k = k + 1
        c["put"](c["coef"], token * COEF_ROWS + row0, a0)
        c["put"](c["coef"], token * COEF_ROWS + row0 + 1, a1)
        bo.buffer_store(a0, rsrc(c["args"]["coef_out"]), token * COEF_ROWS + row0)
        bo.buffer_store(a1, rsrc(c["args"]["coef_out"]), token * COEF_ROWS + row0 + 1)
        task = task + stride


@flyc.jit
def stage_conv0(c):
    """Materialize bf16 grouped two-tap conv0 for the QKV projection."""
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    total = c["S"] * H // 2
    while task < total:
        token, pair = task // (H // 2), task % (H // 2)
        element = 2 * pair
        group = element // GROUP
        got = c["poll"](
            [
                (c["coef"], token * COEF_ROWS + group, 1),
                (c["coef"], token * COEF_ROWS + GROUPS + group, 1),
            ]
        )
        d0, d1 = got[0][0].bitcast(fx.Float32), got[1][0].bitcast(fx.Float32)
        vals = []
        for q in range_constexpr(2):
            e = element + q
            cur = _poll_bf(c, c["h"], token * H + e)
            value = (_bf(c["args"]["base"], e) + d0) * cur
            if (token % BLOCK) >= 1:
                prev = _poll_bf(c, c["h"], (token - 1) * H + e)
                value = value + (_bf(c["args"]["base"], H + e) + d1) * prev
            vals.append(bf16_round(value))
        c["put_bf"](c["hc"], token * H + element, vals)
        task = task + stride


@flyc.jit
def stage_qkv(c):
    """QKV projection, rounded to bf16 before per-head processing."""
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    total = c["S"] * QKV_ROWS // 2
    while task < total:
        token, pair = task // (QKV_ROWS // 2), task % (QKV_ROWS // 2)
        row0 = 2 * pair
        a0, a1 = fx.Float32(0.0), fx.Float32(0.0)
        k = fx.Int32(0)
        while k < H:
            x = _poll_bf(c, c["hc"], token * H + k)
            a0 = a0 + _bf(c["args"]["w_qkv"], row0 * H + k) * x
            a1 = a1 + _bf(c["args"]["w_qkv"], (row0 + 1) * H + k) * x
            k = k + 1
        c["put_bf"](c["qkv"], token * QKV_ROWS + row0, [bf16_round(a0), bf16_round(a1)])
        task = task + stride


def _cs(ptr, pos, index):
    return fx.Float32(bo.buffer_load(rsrc(ptr), pos * HEAD_DIM + index, vec_width=1, dtype=T.f32))


@flyc.jit
def stage_heads(c):
    """Per-head q/k RMSNorm, neox RoPE, and q plus fp8 K/V publication."""
    job = c["bid"]
    while job < c["S"] * QKV_HEADS:
        token, head = job // QKV_HEADS, job % QKV_HEADS
        is_q = head < LOCAL_Q_HEADS
        is_k = (head >= LOCAL_Q_HEADS) & (head < LOCAL_Q_HEADS + LOCAL_KV_HEADS)
        kv_head = is_k.select(head - LOCAL_Q_HEADS, head - LOCAL_Q_HEADS - LOCAL_KV_HEADS)
        row0 = is_q.select(head * HEAD_DIM, is_k.select(Q_K, Q_K + KV_K) + kv_head * HEAD_DIM)
        if c["wave"] == 0:
            dim = 4 * c["lane"]
            raw = [_poll_bf(c, c["qkv"], token * QKV_ROWS + row0 + dim + j) for j in range(4)]
            ss = raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]
            ss = butterfly(ss, (16, 8, 4, 2, 1))
            rstd = hw_rsq(ss / float(HEAD_DIM) + EPS)
            normed = []
            for j in range_constexpr(4):
                weight = _bf(is_k.select(c["args"]["k_norm"], c["args"]["q_norm"]), dim + j)
                normed.append(bf16_round(raw[j] * rstd * weight))
            pos = fx.Int32(bo.buffer_load(rsrc(c["args"]["positions"]), (c["args"]["pos64"] != 0).select(2 * token, token), vec_width=1, dtype=T.i32))
            out = []
            for j in range_constexpr(4):
                d = dim + j
                half = d % (HEAD_DIM // 2)
                mate_d = (d < HEAD_DIM // 2).select(d + HEAD_DIM // 2, d - HEAD_DIM // 2)
                mate_raw = _poll_bf(c, c["qkv"], token * QKV_ROWS + row0 + mate_d)
                mate_w = _bf(is_k.select(c["args"]["k_norm"], c["args"]["q_norm"]), mate_d)
                mate = bf16_round(mate_raw * rstd * mate_w)
                rotated = (d < HEAD_DIM // 2).select(
                    normed[j] * _cs(c["args"]["cos_sin"], pos, half) - mate * _cs(c["args"]["cos_sin"], pos, HEAD_DIM // 2 + half),
                    normed[j] * _cs(c["args"]["cos_sin"], pos, half) + mate * _cs(c["args"]["cos_sin"], pos, HEAD_DIM // 2 + half),
                )
                out.append((is_q | is_k).select(bf16_round(rotated), raw[j]))
            if is_q:
                bo.buffer_store(
                    fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                    rsrc(c["args"]["q"]),
                    token * Q_K + head * HEAD_DIM + dim,
                )
            else:
                slot = fx.Int32(bo.buffer_load(rsrc(c["args"]["slots"]), (c["args"]["slot64"] != 0).select(2 * token, token), vec_width=1, dtype=T.i32))
                packed = fp8_pack4(clamp_fp8(out[0]), clamp_fp8(out[1]), clamp_fp8(out[2]), clamp_fp8(out[3]))
                index = (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + dim // 4
                bo.buffer_store(packed, rsrc(is_k.select(c["args"]["k_cache"], c["args"]["v_cache"])), index)
        job = job + c["B"]


@flyc.jit
def run_dpre(c):
    stage_norm_in(c)
    stage_coef(c)
    stage_conv0(c)
    stage_qkv(c)
    stage_heads(c)


@flyc.jit
def stage_o(c):
    """Local [8192, 2048] projection and bf16 rank-partial publication."""
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    total = c["S"] * H // 2
    while task < total:
        token, pair = task // (H // 2), task % (H // 2)
        row0 = 2 * pair
        a0, a1 = fx.Float32(0.0), fx.Float32(0.0)
        k = fx.Int32(0)
        while k < Q_K:
            x = _bf(c["args"]["o"], token * Q_K + k)
            a0 = a0 + _bf(c["args"]["w_o"], row0 * Q_K + k) * x
            a1 = a1 + _bf(c["args"]["w_o"], (row0 + 1) * Q_K + k) * x
            k = k + 1
        element = (c["rank"] * MAX_TOKENS + token) * H + row0
        for peer in range_constexpr(c["W"]):
            c["put_bf"](c["peer_do"][peer], element, [bf16_round(a0), bf16_round(a1)])
        task = task + stride


@flyc.jit
def _rank_sum(c, token, element):
    got = c["poll"](
        [(c["own_do"], (src * MAX_TOKENS * H + token * H + element) // 2, 1) for src in range(c["W"])]
    )
    lo, hi = bf2_f32(got[0][0])
    for src in range_constexpr(1, c["W"]):
        p0, p1 = bf2_f32(got[src][0])
        lo, hi = lo + p0, hi + p1
    return lo, hi


@flyc.jit
def stage_post(c):
    """Rank-ordered TP reduction followed by side-1 grouped convolution."""
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    total = c["S"] * H // 2
    while task < total:
        token, pair = task // (H // 2), task % (H // 2)
        element = 2 * pair
        v0, v1 = _rank_sum(c, token, element)
        group = element // GROUP
        d0 = fx.Float32(bo.buffer_load(rsrc(c["args"]["coef"]), token * COEF_ROWS + 2 * GROUPS + group, vec_width=1, dtype=T.f32))
        d1 = fx.Float32(bo.buffer_load(rsrc(c["args"]["coef"]), token * COEF_ROWS + 3 * GROUPS + group, vec_width=1, dtype=T.f32))
        o0 = (_bf(c["args"]["base"], 2 * H + element) + d0) * v0
        o1 = (_bf(c["args"]["base"], 2 * H + element + 1) + d0) * v1
        if (token % BLOCK) >= 1:
            p0, p1 = _rank_sum(c, token - 1, element)
            o0 = o0 + (_bf(c["args"]["base"], 3 * H + element) + d1) * p0
            o1 = o1 + (_bf(c["args"]["base"], 3 * H + element + 1) + d1) * p1
        _store_bf2(c["args"]["attn_out"], (token * H + element) // 2, o0, o1)
        task = task + stride


def build_dpre(key: DPreBuild):
    _validate(key.tokens, key.resident_blocks)
    layout = dp_scratch_layout()
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol(
        "muse_spark_gfx1250_dpre",
        s=key.tokens,
        c=key.resident_blocks,
        f=key.fault_site,
    )

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpre(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64,
        w_qkv: Int64, q_norm: Int64, k_norm: Int64, cos_sin: Int64,
        positions: Int64, pos64: Int32, slots: Int64, slot64: Int32,
        k_cache: Int64, v_cache: Int64, res_out: Int64, q: Int64,
        coef_out: Int64, scratch: Int64, epoch: Int64, layer: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = Mailbox(
            step_tag(epoch, layer),
            mailbox_guard(
                epoch,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
                stop=False,
            ),
        )
        c = {
            "S": key.tokens, "B": key.resident_blocks, "tid": tid, "bid": bid,
            "lane": tid % WAVE_SIZE, "wave": tid // WAVE_SIZE, "red": lds.red.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll,
            "h": sreg(scratch, layout["dp_h"][0], "dp_h"),
            "hc": sreg(scratch, layout["dp_hc"][0], "dp_hc"),
            "qkv": sreg(scratch, layout["dp_qkv"][0], "dp_qkv"),
            "coef": sreg(scratch, layout["dp_coef"][0], "dp_coef"),
            "args": {
                "x": x, "res": res, "first": first, "w_in": w_in, "w_kp": w_kp,
                "base": base, "w_qkv": w_qkv, "q_norm": q_norm, "k_norm": k_norm,
                "cos_sin": cos_sin, "positions": positions, "pos64": pos64,
                "slots": slots, "slot64": slot64, "k_cache": k_cache,
                "v_cache": v_cache, "res_out": res_out, "q": q, "coef_out": coef_out,
            },
        }
        run_dpre(c)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64,
        w_qkv: Int64, q_norm: Int64, k_norm: Int64, cos_sin: Int64,
        positions: Int64, pos64: Int32, slots: Int64, slot64: Int32,
        k_cache: Int64, v_cache: Int64, res_out: Int64, q: Int64,
        coef_out: Int64, scratch: Int64, epoch: Int64, layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dpre(
            x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin,
            positions, pos64, slots, slot64, k_cache, v_cache, res_out, q,
            coef_out, scratch, epoch, layer,
        ).launch(grid=(key.resident_blocks,), block=(THREADS,), stream=stream)

    return launch


def build_dpost(key: DPostBuild):
    _validate(key.tokens, key.resident_blocks)
    if key.npes not in (1, 4):
        raise ValueError(f"gfx1250 D-post supports TP1 or TP4, got TP{key.npes}")
    sy = do_sym_layout(key.npes)
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol(
        "muse_spark_gfx1250_dpost",
        s=key.tokens,
        w=key.npes,
        c=key.resident_blocks,
        f=key.fault_site,
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpost(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        epoch: Int64, layer: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        active_guard = mailbox_guard(
            epoch, own=sym, peers=peers, rank=rank, npes=key.npes,
            flags_offset=do_sym_bytes(key.npes), fault_site=key.fault_site or None,
            fault_after=key.fault_after, stop=False,
        )
        mb = Mailbox(step_tag(epoch, layer), active_guard)
        bases = peer_bases(peers, key.npes)
        c = {
            "S": key.tokens, "W": key.npes, "B": key.resident_blocks,
            "tid": tid, "bid": bid, "rank": rank, "put_bf": mb.put_bf,
            "poll": mb.poll, "own_do": preg(sym, sy["do"][0], "do"),
            "peer_do": [preg(basep, sy["do"][0], "do") for basep in bases],
            "args": {"o": o, "w_o": w_o, "coef": coef, "base": base, "attn_out": attn_out},
        }
        stage_o(c)
        stage_post(c)

    @flyc.jit
    def launch(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dpost(
            o, w_o, coef, base, attn_out, scratch, sym, peers, rank, epoch, layer
        ).launch(grid=(key.resident_blocks,), block=(THREADS,), stream=stream)

    return launch


assert (LOCAL_Q_HEADS, LOCAL_KV_HEADS) == (16, 2)
assert (Q_K, QKV_ROWS, COEF_ROWS) == (2048, 2560, 2048)
assert TAPS == 2
