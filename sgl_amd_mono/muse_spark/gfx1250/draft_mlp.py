# SPDX-License-Identifier: Apache-2.0
"""gfx1250 DFlash2 D-MLP with full-K wave32 BF16 WMMA projections."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
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
I = LOCAL_INTERMEDIATE
GU_ROWS = 2 * I
GU_TASKS = I // ROWS
DOWN_TASKS = H // ROWS
GU_K_PER_WAVE = H // WAVES
DOWN_K_PER_WAVE = I // WAVES
GU_STEPS = GU_K_PER_WAVE // WMMA_K
DOWN_STEPS = DOWN_K_PER_WAVE // WMMA_K
GROUP = CONV_GROUP
GROUPS = CONV_GROUPS
TAPS = CONV_TAPS
EPS = RMS_EPS
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

assert (GU_TASKS, DOWN_TASKS) == (448, 512)
assert (GU_K_PER_WAVE, DOWN_K_PER_WAVE) == (1024, 896)
assert (GU_STEPS, DOWN_STEPS) == (32, 28)


def dm_scratch_layout() -> dict:
    """Rank-private tagged payloads. Sizes are in 32-bit mailbox words."""
    return pair_layout(
        [
            ("dm_h", MAX_TOKENS * H // 2),
            ("dm_coef", MAX_TOKENS * COEF_ROWS),
            ("dm_hc", MAX_TOKENS * H // 2),
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


def _weight_fragment(resource, base, k0, lane):
    """One row-major v16bf16 A fragment using legal dwordx4 loads."""
    kh = (lane // 16) * 8
    lo = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=2,
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            resource,
            (base + k0 + 16 + kh) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=2,
        )
    )
    return fx.Vector.from_elements(
        [lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32
    ).bitcast(fx.BFloat16)


def _lds_fragment(xbuf, base, k0, lane):
    """One LDS v16bf16 B fragment, with one token repeated as a WMMA column."""
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


def _tile_value(red, plane, row, token):
    lane = ((row % ROWS) // 8) * 16 + token
    elem = row % 8
    value = fx.Float32(0.0)
    for wave in range_constexpr(WAVES):
        value = value + fx.ptr_load(
            red
            + plane * RED_FLOATS
            + (wave * WAVE_SIZE + lane) * WMMA_ACC
            + elem
        )
    return value


@flyc.jit
def _stage_lds_input(c, region, elements, tile_start, tile_rows):
    """Copy one at-most-16-row tagged BF16 tile into every CTA's LDS."""
    pair_groups = tile_rows * elements // 4
    for j in range_constexpr(pair_groups // THREADS):
        local_pair = 2 * (c["tid"] + j * THREADS)
        global_pair = tile_start * elements // 2 + local_pair
        got = c["poll"]([(region, global_pair, 2)])
        words = fx.Vector.from_elements([got[0][0], got[0][1]], fx.Int32)
        fx.ptr_store(
            words.bitcast(fx.BFloat16), c["xbuf"] + local_pair * 2
        )
    gpu.barrier()


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
def stage_hc(c):
    """Materialize conv0 once at the original BF16 boundary."""
    total = c["S"] * H // 2
    task = c["bid"] * THREADS + c["tid"]
    stride = c["B"] * THREADS
    while task < total:
        token, pair = task // (H // 2), task % (H // 2)
        element = 2 * pair
        c["put_bf"](
            c["hc"],
            token * H + element,
            [_hc(c, token, element), _hc(c, token, element + 1)],
        )
        task = task + stride


@flyc.jit
def stage_act(c):
    """Fused Gate/Up full-K WMMA and BF16 SiLU activation."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_gu"])
    tile_rows = min(ROWS, c["S"])
    for token_tile in range_constexpr(c["TOKEN_TILES"]):
        tile_start = token_tile * ROWS
        _stage_lds_input(c, c["hc"], H, tile_start, tile_rows)
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        task = c["bid"]
        while task < GU_TASKS:
            row0 = task * ROWS
            row = row0 + lane % ROWS
            gate = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            up = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            for step in range_constexpr(GU_STEPS):
                k0 = wave * GU_K_PER_WAVE + step * WMMA_K
                x = _lds_fragment(c["xbuf"], token_local * H, k0, lane)
                gate = _wmma(
                    _weight_fragment(r_w, row * H, k0, lane), x, gate
                )
                up = _wmma(
                    _weight_fragment(r_w, (I + row) * H, k0, lane), x, up
                )
            fx.ptr_store(
                gate,
                c["wmma_red"] + (wave * WAVE_SIZE + lane) * WMMA_ACC,
            )
            fx.ptr_store(
                up,
                c["wmma_red"]
                + RED_FLOATS
                + (wave * WAVE_SIZE + lane) * WMMA_ACC,
            )
            gpu.barrier()
            if c["tid"] < tile_rows * (ROWS // 2):
                local_token = c["tid"] // (ROWS // 2)
                out_token = tile_start + local_token
                dim = 2 * (c["tid"] % (ROWS // 2))
                g0 = bf16_round(
                    _tile_value(c["wmma_red"], 0, dim, local_token)
                )
                g1 = bf16_round(
                    _tile_value(c["wmma_red"], 0, dim + 1, local_token)
                )
                u0 = bf16_round(
                    _tile_value(c["wmma_red"], 1, dim, local_token)
                )
                u1 = bf16_round(
                    _tile_value(c["wmma_red"], 1, dim + 1, local_token)
                )
                s0 = bf16_round(
                    g0
                    / (
                        fx.Float32(1.0)
                        + hw_exp2(g0 * fx.Float32(-1.4426950408889634))
                    )
                )
                s1 = bf16_round(
                    g1
                    / (
                        fx.Float32(1.0)
                        + hw_exp2(g1 * fx.Float32(-1.4426950408889634))
                    )
                )
                c["put_bf"](
                    c["act"],
                    out_token * I + row0 + dim,
                    [s0 * u0, s1 * u1],
                )
            gpu.barrier()
            task = task + c["B"]


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
    """Full-K Down WMMA, ordered TP sum, grouped conv1 and output."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_down"])
    tile_rows = min(ROWS, c["S"])
    for token_tile in range_constexpr(c["TOKEN_TILES"]):
        tile_start = token_tile * ROWS
        _stage_lds_input(c, c["act"], I, tile_start, tile_rows)
        token_local = fx.min(lane % ROWS, tile_rows - 1)
        task = c["bid"]
        while task < DOWN_TASKS:
            row0 = task * ROWS
            row = row0 + lane % ROWS
            down = fx.Vector.filled(WMMA_ACC, 0.0, fx.Float32)
            for step in range_constexpr(DOWN_STEPS):
                k0 = wave * DOWN_K_PER_WAVE + step * WMMA_K
                down = _wmma(
                    _weight_fragment(r_w, row * I, k0, lane),
                    _lds_fragment(c["xbuf"], token_local * I, k0, lane),
                    down,
                )
            fx.ptr_store(
                down,
                c["wmma_red"] + (wave * WAVE_SIZE + lane) * WMMA_ACC,
            )
            gpu.barrier()
            local_token = c["tid"] // (ROWS // 2)
            out_token = tile_start + local_token
            dim = 2 * (c["tid"] % (ROWS // 2))
            element = row0 + dim
            active = c["tid"] < tile_rows * (ROWS // 2)
            if active:
                d0 = bf16_round(
                    _tile_value(c["wmma_red"], 0, dim, local_token)
                )
                d1 = bf16_round(
                    _tile_value(c["wmma_red"], 0, dim + 1, local_token)
                )
                mailbox_element = (
                    (c["rank"] * MAX_TOKENS + out_token) * H + element
                )
                for peer in range_constexpr(c["W"]):
                    c["put_bf"](
                        c["peer_dm"][peer], mailbox_element, [d0, d1]
                    )
            gpu.barrier()
            wait_stores()
            gpu.barrier()
            if active:
                v0, v1 = _rank_sum(c, out_token, element)
                group = element // GROUP
                q0 = c["poll"](
                    [
                        (
                            c["coef"],
                            out_token * COEF_ROWS
                            + 2 * GROUPS
                            + group,
                            1,
                        )
                    ]
                )[0][0].bitcast(fx.Float32)
                b0 = _bf(c["args"]["base"], 2 * H + element)
                b1 = _bf(c["args"]["base"], 2 * H + element + 1)
                o0, o1 = (b0 + q0) * v0, (b1 + q0) * v1
                if (out_token % BLOCK) >= 1:
                    p0, p1 = _rank_sum(c, out_token - 1, element)
                    q1 = c["poll"](
                        [
                            (
                                c["coef"],
                                out_token * COEF_ROWS
                                + 3 * GROUPS
                                + group,
                                1,
                            )
                        ]
                    )[0][0].bitcast(fx.Float32)
                    b0 = _bf(c["args"]["base"], 3 * H + element)
                    b1 = _bf(c["args"]["base"], 3 * H + element + 1)
                    o0, o1 = (
                        o0 + (b0 + q1) * p0,
                        o1 + (b1 + q1) * p1,
                    )
                _store_bf2(
                    c["args"]["out"],
                    (out_token * H + element) // 2,
                    o0,
                    o1,
                )
            gpu.barrier()
            task = task + c["B"]


def build_dmlp(key: DMLPBuild):
    if key.tokens not in SUPPORTED_TOKENS:
        raise ValueError(
            f"gfx1250 D-MLP supports widths {SUPPORTED_TOKENS}, got {key.tokens}"
        )
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
        red: fx.Array[fx.Float32, 2 * RED_FLOATS, 16]
        xbuf: fx.Array[fx.BFloat16, ROWS * H, 16]

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
            "S": key.tokens, "TOKEN_TILES": (key.tokens + ROWS - 1) // ROWS,
            "W": key.npes, "B": key.resident_blocks,
            "tid": tid, "bid": bid, "lane": tid % WAVE_SIZE, "wave": tid // WAVE_SIZE,
            "rank": rank, "red": lds.red.ptr, "wmma_red": lds.red.ptr,
            "xbuf": lds.xbuf.ptr, "put": mb.put, "put_bf": mb.put_bf,
            "poll": mb.poll, "h": sreg(scratch, layout["dm_h"][0], "dm_h"),
            "coef": sreg(scratch, layout["dm_coef"][0], "dm_coef"),
            "hc": sreg(scratch, layout["dm_hc"][0], "dm_hc"),
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
        stage_hc(c)
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
