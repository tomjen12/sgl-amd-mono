# SPDX-License-Identifier: Apache-2.0
"""gfx1250 DFlash2 attention boundaries with wave32 BF16 WMMA projections."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
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
from sgl_amd_mono.muse_spark.gfx1250.device import (
    mailbox_guard,
    step_tag,
    wait_stores,
)
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
    SUPPORTED_TOKENS,
)

THREADS = 256
WAVE_SIZE = 32
WAVES = THREADS // WAVE_SIZE
ROWS = 16
WMMA_K = 32
WMMA_ACC = 8
RED_FLOATS = WAVES * WAVE_SIZE * WMMA_ACC
H = HIDDEN
GROUP = CONV_GROUP
GROUPS = CONV_GROUPS
TAPS = CONV_TAPS
EPS = RMS_EPS
KV_K = LOCAL_KV_HEADS * HEAD_DIM
QKV_HEADS = LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS
COEF_TASKS = COEF_ROWS // ROWS
QKV_TASKS = QKV_ROWS // ROWS
O_TASKS = H // ROWS
K_PER_WAVE = H // WAVES
K_STEPS = K_PER_WAVE // WMMA_K
O_K_PER_WAVE = Q_K // WAVES
O_STEPS = O_K_PER_WAVE // WMMA_K
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

assert (COEF_TASKS, QKV_TASKS, O_TASKS) == (128, 160, 512)
assert K_PER_WAVE == 1024 and K_STEPS == 32
assert O_K_PER_WAVE == 256 and O_STEPS == 8


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
    if tokens not in SUPPORTED_TOKENS:
        raise ValueError(
            f"gfx1250 D-attn half supports widths {SUPPORTED_TOKENS}, got {tokens}"
        )
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


def _weight_fragment(resource, base, k0, lane, cache_modifier=2):
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


def _lds_fragment(xbuf, base, k0, lane):
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    kh = (lane // 16) * 8
    lo = fx.Vector(fx.ptr_load(xbuf + base + k0 + kh, result_type=V8B))
    hi = fx.Vector(fx.ptr_load(xbuf + base + k0 + 16 + kh, result_type=V8B))
    return fx.Vector.from_elements(
        [lo[i] for i in range(8)] + [hi[i] for i in range(8)],
        fx.BFloat16,
    )


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


def _tile_value(red, row, token):
    lane = ((row % ROWS) // 8) * 16 + token
    elem = row % 8
    value = fx.Float32(0.0)
    for wave in range_constexpr(WAVES):
        value = value + fx.ptr_load(
            red + (wave * WAVE_SIZE + lane) * WMMA_ACC + elem
        )
    return value


@flyc.jit
def _stage_lds_input(c, region, tile_start, tile_rows):
    """Copy one at-most-16-row tagged BF16 tile into every CTA's LDS."""
    pair_groups = tile_rows * H // 4
    for j in range_constexpr(pair_groups // THREADS):
        local_pair = 2 * (c["tid"] + j * THREADS)
        global_pair = tile_start * H // 2 + local_pair
        got = c["poll"]([(region, global_pair, 2)])
        words = fx.Vector.from_elements([got[0][0], got[0][1]], fx.Int32)
        fx.ptr_store(
            words.bitcast(fx.BFloat16), c["xbuf"] + local_pair * 2
        )
    gpu.barrier()


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
    """Full-K WMMA kernel projection with fp32 coefficients."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_kp"])
    tile_rows = min(ROWS, c["S"])
    for token_tile in range_constexpr(c["TOKEN_TILES"]):
        tile_start = token_tile * ROWS
        _stage_lds_input(c, c["h"], tile_start, tile_rows)
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        task = c["bid"]
        while task < COEF_TASKS:
            row0 = task * ROWS
            row = row0 + lane % ROWS
            acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            for step in range_constexpr(K_STEPS):
                k0 = wave * K_PER_WAVE + step * WMMA_K
                acc = _wmma(
                    _weight_fragment(r_w, row * H, k0, lane),
                    _lds_fragment(c["xbuf"], token_local * H, k0, lane),
                    acc,
                )
            fx.ptr_store(
                acc, c["wmma_red"] + (wave * WAVE_SIZE + lane) * WMMA_ACC
            )
            gpu.barrier()
            if c["tid"] < tile_rows * ROWS:
                local_token = c["tid"] // ROWS
                dim = c["tid"] % ROWS
                value = _tile_value(c["wmma_red"], dim, local_token)
                out_token = tile_start + local_token
                out_row = row0 + dim
                c["put"](
                    c["coef"], out_token * COEF_ROWS + out_row, value
                )
                bo.buffer_store(
                    value,
                    rsrc(c["args"]["coef_out"]),
                    out_token * COEF_ROWS + out_row,
                )
            gpu.barrier()
            task = task + c["B"]


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
    """Full-K QKV WMMA, rounded to BF16 before per-head processing."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_qkv"])
    tile_rows = min(ROWS, c["S"])
    for token_tile in range_constexpr(c["TOKEN_TILES"]):
        tile_start = token_tile * ROWS
        _stage_lds_input(c, c["hc"], tile_start, tile_rows)
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        task = c["bid"]
        while task < QKV_TASKS:
            row0 = task * ROWS
            row = row0 + lane % ROWS
            acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            for step in range_constexpr(K_STEPS):
                k0 = wave * K_PER_WAVE + step * WMMA_K
                acc = _wmma(
                    _weight_fragment(r_w, row * H, k0, lane),
                    _lds_fragment(c["xbuf"], token_local * H, k0, lane),
                    acc,
                )
            fx.ptr_store(
                acc, c["wmma_red"] + (wave * WAVE_SIZE + lane) * WMMA_ACC
            )
            gpu.barrier()
            if c["tid"] < tile_rows * (ROWS // 2):
                local_token = c["tid"] // (ROWS // 2)
                out_token = tile_start + local_token
                dim = 2 * (c["tid"] % (ROWS // 2))
                c["put_bf"](
                    c["qkv"],
                    out_token * QKV_ROWS + row0 + dim,
                    [
                        bf16_round(
                            _tile_value(c["wmma_red"], dim, local_token)
                        ),
                        bf16_round(
                            _tile_value(c["wmma_red"], dim + 1, local_token)
                        ),
                    ],
                )
            gpu.barrier()
            task = task + c["B"]


def _cs(ptr, pos, index):
    return fx.Float32(bo.buffer_load(rsrc(ptr), pos * HEAD_DIM + index, vec_width=1, dtype=T.f32))


@flyc.jit
def stage_heads(c):
    """Per-head q/k RMSNorm, RoPE, BF16 current K/V and fp8 cache publication."""
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
                bo.buffer_store(
                    fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                    rsrc(
                        is_k.select(
                            c["args"]["k_out"], c["args"]["v_out"]
                        )
                    ),
                    token * KV_K + kv_head * HEAD_DIM + dim,
                )
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
    """Full-K O projection WMMA and BF16 rank-partial publication."""
    lane, wave = c["lane"], c["wave"]
    r_w, r_o = rsrc(c["args"]["w_o"]), rsrc(c["args"]["o"])
    tile_rows = min(ROWS, c["S"])
    for token_tile in range_constexpr(c["TOKEN_TILES"]):
        tile_start = token_tile * ROWS
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        token = tile_start + token_local
        task = c["bid"]
        while task < O_TASKS:
            row0 = task * ROWS
            row = row0 + lane % ROWS
            acc = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            for step in range_constexpr(O_STEPS):
                k0 = wave * O_K_PER_WAVE + step * WMMA_K
                acc = _wmma(
                    _weight_fragment(r_w, row * Q_K, k0, lane),
                    _weight_fragment(
                        r_o, token * Q_K, k0, lane, cache_modifier=0
                    ),
                    acc,
                )
            fx.ptr_store(
                acc, c["wmma_red"] + (wave * WAVE_SIZE + lane) * WMMA_ACC
            )
            gpu.barrier()
            if c["tid"] < tile_rows * (ROWS // 2):
                local_token = c["tid"] // (ROWS // 2)
                out_token = tile_start + local_token
                dim = 2 * (c["tid"] % (ROWS // 2))
                element = row0 + dim
                d0 = bf16_round(
                    _tile_value(c["wmma_red"], dim, local_token)
                )
                d1 = bf16_round(
                    _tile_value(c["wmma_red"], dim + 1, local_token)
                )
                peer_element = (
                    (c["rank"] * MAX_TOKENS + out_token) * H + element
                )
                for peer in range_constexpr(c["W"]):
                    c["put_bf"](
                        c["peer_do"][peer], peer_element, [d0, d1]
                    )
            gpu.barrier()
            wait_stores()
            gpu.barrier()
            task = task + c["B"]


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
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        xbuf: fx.Array[fx.BFloat16, ROWS * H, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpre(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64,
        w_qkv: Int64, q_norm: Int64, k_norm: Int64, cos_sin: Int64,
        positions: Int64, pos64: Int32, slots: Int64, slot64: Int32,
        k_cache: Int64, v_cache: Int64, res_out: Int64, q: Int64,
        k_out: Int64, v_out: Int64,
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
            "S": key.tokens, "TOKEN_TILES": (key.tokens + ROWS - 1) // ROWS,
            "B": key.resident_blocks, "tid": tid, "bid": bid,
            "lane": tid % WAVE_SIZE, "wave": tid // WAVE_SIZE,
            "red": lds.red.ptr, "wmma_red": lds.red.ptr, "xbuf": lds.xbuf.ptr,
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
                "v_cache": v_cache, "res_out": res_out, "q": q,
                "k_out": k_out, "v_out": v_out, "coef_out": coef_out,
            },
        }
        run_dpre(c)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64,
        w_qkv: Int64, q_norm: Int64, k_norm: Int64, cos_sin: Int64,
        positions: Int64, pos64: Int32, slots: Int64, slot64: Int32,
        k_cache: Int64, v_cache: Int64, res_out: Int64, q: Int64,
        k_out: Int64, v_out: Int64,
        coef_out: Int64, scratch: Int64, epoch: Int64, layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dpre(
            x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin,
            positions, pos64, slots, slot64, k_cache, v_cache, res_out, q,
            k_out, v_out,
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

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, RED_FLOATS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpost(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        epoch: Int64, layer: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        active_guard = mailbox_guard(
            epoch, own=sym, peers=peers, rank=rank, npes=key.npes,
            flags_offset=do_sym_bytes(key.npes), fault_site=key.fault_site or None,
            fault_after=key.fault_after, stop=False,
        )
        mb = Mailbox(step_tag(epoch, layer), active_guard)
        bases = peer_bases(peers, key.npes)
        c = {
            "S": key.tokens, "TOKEN_TILES": (key.tokens + ROWS - 1) // ROWS,
            "W": key.npes, "B": key.resident_blocks,
            "tid": tid, "bid": bid, "lane": tid % WAVE_SIZE,
            "wave": tid // WAVE_SIZE, "rank": rank,
            "wmma_red": lds.red.ptr, "put_bf": mb.put_bf,
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
