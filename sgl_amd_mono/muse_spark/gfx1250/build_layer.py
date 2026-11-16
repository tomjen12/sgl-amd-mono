# SPDX-License-Identifier: Apache-2.0
"""True persistent gfx1250 Muse Spark K-attn -> K2 -> K3 [-> next K1] layer.

The returned launcher accepts one private ``scratch`` allocation and one
peer-symmetric ``sym`` allocation per rank.  ``peers`` is the usual array of
peer symmetric bases.  :func:`layer_scratch_bytes` and
:func:`layer_sym_bytes` report the required allocation sizes.

All profile CTAs must be resident together (one CTA per enabled CU).
Device hand-offs use
tagged polling with a bounded gfx1250 cycle-counter deadline and peer failure
propagation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import peer_bases
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx1250.build_k2 import k2_scratch_layout
from sgl_amd_mono.muse_spark.gfx1250.build_k3 import _scratch_layout as k3_scratch_layout
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    MAX_TOKENS,
    MOE_HIDDEN,
    Q_K,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    KATTN_TASKS,
    RESIDENT_BLOCKS,
    THREADS,
    k1_scratch_layout,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import (
    attention,
    ffn,
    post_attn,
    pre_attn,
)

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)
_ALIGN = 256
O_GROUPS = HIDDEN // (post_attn.ROWS * post_attn.WAVES)
ROUTER_GROUPS = EXPERTS // (post_attn.ROWS * post_attn.WAVES)
POOL_GROUPS = MOE_HIDDEN // (post_attn.ROWS * post_attn.WAVES)


def _align(value):
    return (value + _ALIGN - 1) // _ALIGN * _ALIGN


def _private_layout():
    """Combined raw K2/K3 data and tagged device-scope phase flags."""
    out = {}
    offset = 0
    for name, (old_offset, size) in k2_scratch_layout().items():
        out[name] = (old_offset, size)
        offset = max(offset, old_offset + size)
    k3_base = _align(offset)
    for name, (old_offset, size) in k3_scratch_layout().items():
        out[name] = (k3_base + old_offset, size)
        offset = max(offset, k3_base + old_offset + size)
    k1_base = _align(offset)
    for name, (old_offset, size) in k1_scratch_layout().items():
        out[name] = (k1_base + old_offset, size)
        offset = max(offset, k1_base + old_offset + size)
    for name, size in (
        ("attn_stats", attention.FUSED_STAT_FLOATS * 4),
        ("attn_global", attention.FUSED_GLOBAL_FLOATS * 4),
        ("attn_part", attention.FUSED_PART_FLOATS * 4),
    ):
        offset = _align(offset)
        out[name] = (offset, size)
        offset += size
    flags = pair_layout(
        [
            ("attn_sflag", attention.FUSED_TASKS),
            ("attn_gflag", attention.FUSED_TILES),
            ("attn_pflag", attention.FUSED_TASKS),
            ("attn_done", KATTN_TASKS),
            ("boundary_done", MAX_TOKENS),
            ("router_done", ROUTER_GROUPS),
            ("pool_done", POOL_GROUPS),
            ("top_done", MAX_TOKENS),
            ("norm_done", MAX_TOKENS),
            ("expert_done", MAX_TOKENS * TOP_K),
            ("finalize_done", MAX_TOKENS),
            ("k3_boundary_done", MAX_TOKENS),
        ],
        align=_ALIGN,
        start=_align(offset),
    )
    out.update(flags)
    return out


def _symmetric_layout(npes):
    """Disjoint peer payload mailboxes for K2 and K3."""
    if npes not in (1, 4):
        raise ValueError(f"gfx1250 fused layer supports TP/EP1 or TP/EP4, got {npes}")
    return pair_layout(
        [
            ("k2_attn", npes * MAX_TOKENS * HIDDEN // 2),
            ("k3_partial", npes * MAX_TOKENS * (MOE_HIDDEN // 2)),
        ],
        align=_ALIGN,
    )


def layer_scratch_bytes():
    """Bytes of rank-private scratch required by the fused launch."""
    layout = _private_layout()
    return max(offset + size for offset, size in layout.values())


def layer_sym_bytes(npes):
    """Bytes of peer-symmetric storage required per rank."""
    layout = _symmetric_layout(npes)
    return max(offset + size for offset, size in layout.values())


class LayerBuild(msgspec.Struct, frozen=True):
    tokens: int = MAX_TOKENS
    q_len: int = MAX_TOKENS
    npes: int = 4
    next_k1: bool = False
    last: bool = False
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


@flyc.jit
def _publish(put, flag, index, tid):
    """Publish prior global stores without gfx1250's unsupported s_waitcnt helper."""
    gpu.barrier()
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
    if tid == 0:
        put(flag, index, fx.Int32(1))


@flyc.jit
def _wait(poll, specs):
    """Poll tagged device flags, then acquire the data they announce."""
    values = poll(specs, batch=4)
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
    return values


@flyc.jit
def _finish_attention_tile(c):
    """Normalize/gate split zero's rows and publish the 256-row contract."""
    if c["split"] == 0:
        gpu.barrier()
        if c["wave"] == 0:
            for row in range_constexpr(attention.FUSED_ROWS):
                token, query_head, _ = attention.fused_row_coords(
                    c["tile"], row
                )
                task = token * (Q_K // HEAD_DIM) + query_head
                post_attn.stage_norm_gate(
                    {
                        "bid": task,
                        "lane": c["lane"],
                        "o": c["o"],
                        "g": c["g"],
                        "og": c["og"],
                    }
                )
        gpu.barrier()
        fx.memory_fence(
            syncscope="agent", ordering=fx.AtomicOrdering.Release
        )
        if c["tid"] < attention.FUSED_ROWS:
            token, query_head, _ = attention.fused_row_coords(
                c["tile"], c["tid"]
            )
            task = token * (Q_K // HEAD_DIM) + query_head
            c["put"](c["attn_done"], task, fx.Int32(1))


def build_layer(key: LayerBuild):
    """Build one 96-CTA, 256-thread, wave32 persistent layer launch.

    Launch argument order is K-attention inputs/outputs, K2 tensors, K3 tensors,
    combined private scratch, combined peer-symmetric storage, peer bases and
    rank/epoch controls.  ``o`` is both K-attention output and K2 norm/gate input;
    ``res`` is updated by both boundaries.
    """
    if (key.tokens, key.q_len) != (MAX_TOKENS, MAX_TOKENS):
        raise ValueError(
            f"gfx1250 fused layer supports tokens=q_len={MAX_TOKENS}, "
            f"got {(key.tokens, key.q_len)}"
        )
    if key.npes not in (1, 4):
        raise ValueError(
            f"gfx1250 fused layer supports TP/EP1 or TP/EP4, got {key.npes}"
        )
    if key.resident_blocks != RESIDENT_BLOCKS:
        raise ValueError(
            f"Layer profile has {RESIDENT_BLOCKS} resident blocks, "
            f"got {key.resident_blocks}"
        )
    if RESIDENT_BLOCKS != attention.FUSED_TASKS:
        raise ValueError(
            "gfx1250 fused Target attention supports only cu96_tp4; "
            f"active profile has {RESIDENT_BLOCKS} resident blocks"
        )

    W = key.npes
    layout = _private_layout()
    sy = _symmetric_layout(W)
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol(
        "muse_spark_gfx1250_layer",
        s=key.tokens,
        q=key.q_len,
        w=W,
        n=int(key.next_k1),
        l=int(key.last),
        c=key.resident_blocks,
        f=key.fault_site,
    )
    AttnLds = attention.fused_attn_lds()
    K3Lds = ffn.k3_lds()
    K1Lds = pre_attn.k1_lds()

    @fx.struct
    class LayerLds:
        attn: AttnLds
        k3: K3Lds
        k1: K1Lds
        post_red: fx.Array[fx.Float32, post_attn.RED_FLOATS, 16]
        picked: fx.Array[fx.Float32, TOP_K, 16]

    def addr(base, region):
        return base + fx.Int64(layout[region][0])

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        prefix_lens: Int64,
        window: Int32,
        scale_bits: Int32,
        g: Int64,
        res: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        pre_w: Int64,
        w_hi: Int64,
        w_lo: Int64,
        bias: Int64,
        w_pool: Int64,
        xn_w: Int64,
        ids: Int64,
        scores: Int64,
        xin: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        pfw: Int64,
        ffn_alpha: Int64,
        ffn_beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        next_w: Int64,
        next_cos_sin: Int64,
        next_positions: Int64,
        next_slots: Int64,
        next_swa_map: Int64,
        next_k_cache: Int64,
        next_v_cache: Int64,
        next_k_out: Int64,
        next_v_out: Int64,
        next_use_rope: Int32,
        next_use_swa: Int32,
        next_slot64: Int32,
        next_swa64: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer_id: Int32,
        last: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 32
        wave = tid // 32
        lds = fx.SharedAllocator().allocate(LayerLds).peek()
        active_guard = mailbox_guard(
            epoch,
            own=sym,
            peers=peers,
            rank=rank,
            npes=W,
            flags_offset=layer_sym_bytes(W),
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
        mb = Mailbox(step_tag(epoch, layer_id), active_guard)
        mb1 = Mailbox(step_tag(epoch, layer_id + 1), active_guard)
        bases = peer_bases(peers, W)

        flags = {
            name: sreg(scratch, layout[name][0], name)
            for name in (
                "attn_done",
                "boundary_done",
                "router_done",
                "pool_done",
                "top_done",
                "norm_done",
                "expert_done",
                "finalize_done",
                "k3_boundary_done",
            )
        }
        attn_regions = {
            name[5:]: sreg(scratch, layout[name][0], name)
            for name in (
                "attn_stats",
                "attn_global",
                "attn_part",
                "attn_sflag",
                "attn_gflag",
                "attn_pflag",
            )
        }

        # All 96 cu96_tp4 CTAs first run one (16-row tile, key split) task.
        # Split zero merges in fixed order, normalizes/gates its 16 rows, then
        # publishes the existing 256-row attn_done contract.
        attn_args = {
            "q": q,
            "out": o,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "table": table,
            "prefix_lens": prefix_lens,
            "window": window,
            "scale_bits": scale_bits,
        }
        tile, split = bid // attention.FUSED_SPLITS, bid % attention.FUSED_SPLITS
        attention.run_fused_attn(
            {
                "tid": tid,
                "lane": lane,
                "wave": wave,
                "tile": tile,
                "split": split,
                "mb": mb,
                "wave_stats": lds.attn.wave_stats.ptr,
                "row_stats": lds.attn.row_stats.ptr,
                "probabilities": lds.attn.probabilities.ptr,
                "pv": lds.attn.pv.ptr,
                **attn_regions,
                "args": attn_args,
            }
        )
        _finish_attention_tile(
            {
                "tid": tid,
                "lane": lane,
                "wave": wave,
                "tile": tile,
                "split": split,
                "o": o,
                "g": g,
                "og": addr(scratch, "k2_og"),
                "put": mb.put,
                "attn_done": flags["attn_done"],
            }
        )

        if bid < O_GROUPS:
            _wait(
                mb.poll,
                [(flags["attn_done"], task, 1) for task in range(KATTN_TASKS)],
            )
            post_attn.stage_o_projection(
                {
                    "W": W,
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "red": lds.post_red.ptr,
                    "rank": rank,
                    "og": addr(scratch, "k2_og"),
                    "w_o": w_o,
                    "put_bf": mb.put_bf,
                    "peer_attn": [
                        preg(base, sy["k2_attn"][0], "k2_attn")
                        for base in bases
                    ],
                }
            )

        if bid < MAX_TOKENS:
            post_attn.stage_boundary(
                {
                    "W": W,
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "red": lds.post_red.ptr,
                    "res": res,
                    "alpha": attn_alpha,
                    "beta": attn_beta,
                    "pre_w": pre_w,
                    "x": addr(scratch, "k2_x"),
                    "poll": mb.poll,
                    "own_attn": preg(sym, sy["k2_attn"][0], "k2_attn"),
                }
            )
            _publish(mb.put, flags["boundary_done"], bid, tid)

        if bid < ROUTER_GROUPS:
            _wait(
                mb.poll,
                [
                    (flags["boundary_done"], token, 1)
                    for token in range(MAX_TOKENS)
                ],
            )
            post_attn.stage_router_pool(
                {
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "rows": EXPERTS,
                    "router": True,
                    "x": addr(scratch, "k2_x"),
                    "w0": w_hi,
                    "w1": w_lo,
                    "out": addr(scratch, "k2_logits"),
                    "red": lds.post_red.ptr,
                }
            )
            _publish(mb.put, flags["router_done"], bid, tid)

        if bid < POOL_GROUPS:
            _wait(
                mb.poll,
                [
                    (flags["boundary_done"], token, 1)
                    for token in range(MAX_TOKENS)
                ],
            )
            post_attn.stage_router_pool(
                {
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "rows": MOE_HIDDEN,
                    "router": False,
                    "x": addr(scratch, "k2_x"),
                    "w0": w_pool,
                    "w1": fx.Int64(0),
                    "out": addr(scratch, "k2_pool"),
                    "red": lds.post_red.ptr,
                }
            )
            _publish(mb.put, flags["pool_done"], bid, tid)

        if bid < MAX_TOKENS:
            _wait(
                mb.poll,
                [
                    (flags["router_done"], task, 1)
                    for task in range(ROUTER_GROUPS)
                ],
            )
            post_attn.stage_top8(
                {
                    "bid": bid,
                    "tid": tid,
                    "bias": bias,
                    "ids": ids,
                    "scores": scores,
                    "logits": addr(scratch, "k2_logits"),
                    "picked": lds.picked.ptr,
                }
            )
            _publish(mb.put, flags["top_done"], bid, tid)

            _wait(
                mb.poll,
                [
                    (flags["pool_done"], block, 1)
                    for block in range(POOL_GROUPS)
                ],
            )
            post_attn.stage_pooled_norm(
                {
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "red": lds.post_red.ptr,
                    "pool": addr(scratch, "k2_pool"),
                    "xn_w": xn_w,
                    "xin": xin,
                }
            )
            _publish(mb.put, flags["norm_done"], bid, tid)

        k3_args = {
            "xin": xin,
            "ids": ids,
            "scores": scores,
            "local_start": local_start,
            "w13": w13,
            "s13": s13,
            "w2": w2,
            "s2": s2,
            "pen_w": pen_w,
            "res": res,
            "pfw": pfw,
            "alpha": ffn_alpha,
            "beta": ffn_beta,
            "npw": npw,
            "w_unpool": w_unpool,
            "attn_out": attn_out,
            "last": fx.Int32(1) if key.last else last,
        }
        k3 = {
            "S": key.tokens,
            "W": W,
            "tid": tid,
            "bid": bid,
            "lane": lane,
            "wave": wave,
            "rank": rank,
            "args": k3_args,
            "red": lds.k3.red.ptr,
            "aux": lds.k3.aux.ptr,
            "sums": lds.k3.sums.ptr,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "own_partial": preg(sym, sy["k3_partial"][0], "k3_partial"),
            "peer_partial": [
                preg(base, sy["k3_partial"][0], "k3_partial")
                for base in bases
            ],
        }
        for region in ("k3_qx", "k3_qmid", "k3_y", "k3_fbuf", "k3_branch"):
            k3[region[3:]] = sreg(scratch, layout[region][0], region)

        if bid < MAX_TOKENS * TOP_K:
            expert_token = bid // TOP_K
            _wait(
                mb.poll,
                [
                    (flags["top_done"], expert_token, 1),
                    (flags["norm_done"], expert_token, 1),
                ],
            )
            ffn.stage_expert_pick(k3)
            _publish(mb.put, flags["expert_done"], bid, tid)

        if bid < MAX_TOKENS:
            _wait(
                mb.poll,
                [
                    (flags["expert_done"], bid * TOP_K + pick, 1)
                    for pick in range(TOP_K)
                ],
            )
            ffn.stage_finalize_ep(k3)
            _publish(mb.put, flags["finalize_done"], bid, tid)
            _wait(mb.poll, [(flags["finalize_done"], bid, 1)])
            ffn.stage_unpool_boundary(k3)
            _publish(mb.put, flags["k3_boundary_done"], bid, tid)

        if key.next_k1:
            _wait(
                mb.poll,
                [
                    (flags["k3_boundary_done"], token, 1)
                    for token in range(MAX_TOKENS)
                ],
            )
            pre_attn.run_k1(
                {
                    "S": key.tokens,
                    "tid": tid,
                    "bid": bid,
                    "lane": lane,
                    "wave": wave,
                    "red": lds.k1.red.ptr,
                    "put": mb1.put,
                    "poll": mb1.poll,
                    "part": sreg(
                        scratch, layout["k1_part"][0], "k1_part"
                    ),
                    "args": {
                        "x": attn_out,
                        "w": next_w,
                        "cos_sin": next_cos_sin,
                        "positions": next_positions,
                        "slots": next_slots,
                        "swa_map": next_swa_map,
                        "k_cache": next_k_cache,
                        "v_cache": next_v_cache,
                        "q_out": q,
                        "k_out": next_k_out,
                        "v_out": next_v_out,
                        "g_out": g,
                        "use_rope": next_use_rope,
                        "use_swa": next_use_swa,
                        "slot64": next_slot64,
                        "swa64": next_swa64,
                    },
                },
                bid,
            )

    @flyc.jit
    def launch(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        prefix_lens: Int64,
        window: Int32,
        scale_bits: Int32,
        g: Int64,
        res: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        pre_w: Int64,
        w_hi: Int64,
        w_lo: Int64,
        bias: Int64,
        w_pool: Int64,
        xn_w: Int64,
        ids: Int64,
        scores: Int64,
        xin: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        pfw: Int64,
        ffn_alpha: Int64,
        ffn_beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        next_w: Int64,
        next_cos_sin: Int64,
        next_positions: Int64,
        next_slots: Int64,
        next_swa_map: Int64,
        next_k_cache: Int64,
        next_v_cache: Int64,
        next_k_out: Int64,
        next_v_out: Int64,
        next_use_rope: Int32,
        next_use_swa: Int32,
        next_slot64: Int32,
        next_swa64: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer_id: Int32,
        last: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        layer(
            q,
            o,
            k_cache,
            v_cache,
            table,
            prefix_lens,
            window,
            scale_bits,
            g,
            res,
            w_o,
            attn_alpha,
            attn_beta,
            pre_w,
            w_hi,
            w_lo,
            bias,
            w_pool,
            xn_w,
            ids,
            scores,
            xin,
            local_start,
            w13,
            s13,
            w2,
            s2,
            pen_w,
            pfw,
            ffn_alpha,
            ffn_beta,
            npw,
            w_unpool,
            attn_out,
            next_w,
            next_cos_sin,
            next_positions,
            next_slots,
            next_swa_map,
            next_k_cache,
            next_v_cache,
            next_k_out,
            next_v_out,
            next_use_rope,
            next_use_swa,
            next_slot64,
            next_swa64,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            layer_id,
            last,
        ).launch(
            grid=(RESIDENT_BLOCKS,),
            block=(THREADS,),
            stream=stream,
        )

    return launch
