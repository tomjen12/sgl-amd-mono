# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness-first gfx1250 Muse Spark K2 builder."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import peer_bases, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    MAX_TOKENS,
    MOE_HIDDEN,
    Q_K,
    RESIDENT_BLOCKS,
    TOP_K,
    WIDTHS,
)
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag
from sgl_amd_mono.muse_spark.gfx1250.stages import post_attn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)
THREADS = 256


def _raw_layout(items):
    out, offset = {}, 0
    for name, size in items:
        offset = (offset + 255) // 256 * 256
        out[name] = (offset, size)
        offset += size
    return out


def k2_scratch_layout():
    """Private raw intermediates.  Kept here to avoid changing shared layout.py."""
    return _raw_layout(
        [
            ("k2_og", MAX_TOKENS * Q_K * 2),
            ("k2_x", MAX_TOKENS * HIDDEN * 2),
            ("k2_logits", MAX_TOKENS * EXPERTS * 4),
            ("k2_pool", MAX_TOKENS * MOE_HIDDEN * 2),
        ]
    )


def scratch_bytes(layout=None):
    layout = k2_scratch_layout() if layout is None else layout
    return max(offset + size for offset, size in layout.values())


def k2_sym_layout(npes):
    return pair_layout(
        [("k2_attn", npes * MAX_TOKENS * HIDDEN // 2)], align=256
    )


def sym_bytes(npes):
    layout = k2_sym_layout(npes)
    return max(offset + size for offset, size in layout.values())


class K2Build(msgspec.Struct, frozen=True):
    tokens: int
    npes: int = 4
    fault_site: int = 0
    fault_after: int = 0


def build_k2(key: K2Build):
    """Build K2 for a supported Target width."""
    if key.tokens not in WIDTHS:
        raise ValueError(f"gfx1250 K2 supports widths {WIDTHS}, got {key.tokens}")
    if key.npes not in (1, 4):
        raise ValueError(f"gfx1250 K2 V1 supports TP1 or TP4, got {key.npes}")

    W = key.npes
    name = kernel_symbol(
        "muse_spark_gfx1250_k2",
        s=key.tokens,
        w=W,
        f=key.fault_site,
    )
    keyed = key_tuple(key, SOURCES)
    layout = k2_scratch_layout()
    sy = k2_sym_layout(W)

    def ptr(base, region):
        return base + fx.Int64(layout[region][0])

    @fx.struct
    class ProjectionLds:
        red: fx.Array[fx.Float32, post_attn.RED_FLOATS, 16]

    @flyc.kernel(name=f"{name}_norm_gate", known_block_size=[32, 1, 1])
    def norm_gate(o: Int64, g: Int64, scratch: Int64):
        post_attn.stage_norm_gate(
            {
                "bid": fx.block_idx.x,
                "lane": fx.thread_idx.x,
                "o": o,
                "g": g,
                "og": ptr(scratch, "k2_og"),
            }
        )

    @flyc.kernel(name=f"{name}_oproj", known_block_size=[THREADS, 1, 1])
    def oproj(
        w_o: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
    ):
        lds = fx.SharedAllocator().allocate(ProjectionLds).peek()
        tid = fx.thread_idx.x
        mb = Mailbox(
            step_tag(epoch, layer),
            mailbox_guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=sym_bytes(W),
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            ),
        )
        bases = peer_bases(peers, W)
        post_attn.stage_o_projection(
            {
                "W": W,
                "S": key.tokens,
                "B": RESIDENT_BLOCKS,
                "bid": fx.block_idx.x,
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "red": lds.red.ptr,
                "rank": rank,
                "og": ptr(scratch, "k2_og"),
                "w_o": w_o,
                "put_bf": mb.put_bf,
                "peer_attn": [
                    preg(base, sy["k2_attn"][0], "k2_attn") for base in bases
                ],
            }
        )

    @fx.struct
    class ReductionLds:
        red: fx.Array[fx.Float32, THREADS, 16]

    @flyc.kernel(name=f"{name}_boundary", known_block_size=[THREADS, 1, 1])
    def boundary(
        res: Int64,
        alpha: Int64,
        beta: Int64,
        pre_w: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
    ):
        lds = fx.SharedAllocator().allocate(ReductionLds).peek()
        mb = Mailbox(
            step_tag(epoch, layer),
            mailbox_guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=sym_bytes(W),
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            ),
        )
        tid = fx.thread_idx.x
        post_attn.stage_boundary(
            {
                "W": W,
                "bid": fx.block_idx.x,
                "S": key.tokens,
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "red": lds.red.ptr,
                "res": res,
                "alpha": alpha,
                "beta": beta,
                "pre_w": pre_w,
                "x": ptr(scratch, "k2_x"),
                "poll": mb.poll,
                "own_attn": preg(sym, sy["k2_attn"][0], "k2_attn"),
            }
        )

    @flyc.kernel(name=f"{name}_router", known_block_size=[THREADS, 1, 1])
    def router(
        w_hi: Int64, w_lo: Int64, scratch: Int64
    ):
        lds = fx.SharedAllocator().allocate(ProjectionLds).peek()
        tid = fx.thread_idx.x
        post_attn.stage_router_pool(
            {
                "bid": fx.block_idx.x,
                "S": key.tokens,
                "B": EXPERTS // post_attn.ROWS,
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "rows": EXPERTS,
                "router": True,
                "x": ptr(scratch, "k2_x"),
                "w0": w_hi,
                "w1": w_lo,
                "out": ptr(scratch, "k2_logits"),
                "red": lds.red.ptr,
            }
        )

    @flyc.kernel(name=f"{name}_pool", known_block_size=[THREADS, 1, 1])
    def pool(w_pool: Int64, scratch: Int64):
        lds = fx.SharedAllocator().allocate(ProjectionLds).peek()
        tid = fx.thread_idx.x
        post_attn.stage_router_pool(
            {
                "bid": fx.block_idx.x,
                "S": key.tokens,
                "B": RESIDENT_BLOCKS,
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "rows": MOE_HIDDEN,
                "router": False,
                "x": ptr(scratch, "k2_x"),
                "w0": w_pool,
                "w1": fx.Int64(0),
                "out": ptr(scratch, "k2_pool"),
                "red": lds.red.ptr,
            }
        )

    @fx.struct
    class TopLds:
        picked: fx.Array[fx.Float32, 8, 16]

    @flyc.kernel(name=f"{name}_top8", known_block_size=[THREADS, 1, 1])
    def top8(
        bias: Int64,
        ids: Int64,
        scores: Int64,
        override_ids: Int64,
        override_scores: Int64,
        scratch: Int64,
    ):
        lds = fx.SharedAllocator().allocate(TopLds).peek()
        tid = fx.thread_idx.x
        post_attn.stage_top8(
            {
                "bid": fx.block_idx.x,
                "tid": tid,
                "bias": bias,
                "ids": ids,
                "scores": scores,
                "logits": ptr(scratch, "k2_logits"),
                "picked": lds.picked.ptr,
            }
        )
        if override_ids != 0 and tid == 0:
            for pick in range_constexpr(TOP_K):
                offset = fx.block_idx.x * TOP_K + pick
                replacement = fx.Int32(
                    bo.buffer_load(
                        rsrc(override_ids), offset, vec_width=1, dtype=T.i32
                    )
                )
                bo.buffer_store(replacement, rsrc(ids), offset)
                if override_scores != 0:
                    score = fx.Float32(
                        bo.buffer_load(
                            rsrc(override_scores),
                            offset,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    bo.buffer_store(score, rsrc(scores), offset)

    @flyc.kernel(name=f"{name}_pooled_norm", known_block_size=[THREADS, 1, 1])
    def pooled_norm(xn_w: Int64, xin: Int64, scratch: Int64):
        lds = fx.SharedAllocator().allocate(ReductionLds).peek()
        tid = fx.thread_idx.x
        post_attn.stage_pooled_norm(
            {
                "bid": fx.block_idx.x,
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "red": lds.red.ptr,
                "pool": ptr(scratch, "k2_pool"),
                "xn_w": xn_w,
                "xin": xin,
            }
        )

    @flyc.jit
    def launch(
        o: Int64,
        g: Int64,
        res: Int64,
        w_o: Int64,
        alpha: Int64,
        beta: Int64,
        pre_w: Int64,
        w_hi: Int64,
        w_lo: Int64,
        bias: Int64,
        w_pool: Int64,
        xn_w: Int64,
        ids: Int64,
        scores: Int64,
        override_ids: Int64,
        override_scores: Int64,
        xin: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        norm_gate(o, g, scratch).launch(
            grid=(key.tokens * (Q_K // HEAD_DIM),),
            block=(32,),
            stream=stream,
        )
        oproj(w_o, scratch, sym, peers, rank, epoch, layer).launch(
            grid=(HIDDEN // (post_attn.ROWS * post_attn.WAVES),),
            block=(THREADS,),
            stream=stream,
        )
        boundary(
            res, alpha, beta, pre_w, scratch, sym, peers, rank, epoch, layer
        ).launch(grid=(key.tokens,), block=(THREADS,), stream=stream)
        router(w_hi, w_lo, scratch).launch(
            grid=(EXPERTS // (post_attn.ROWS * post_attn.WAVES),),
            block=(THREADS,),
            stream=stream,
        )
        pool(w_pool, scratch).launch(
            grid=(MOE_HIDDEN // (post_attn.ROWS * post_attn.WAVES),),
            block=(THREADS,),
            stream=stream,
        )
        top8(
            bias, ids, scores, override_ids, override_scores, scratch
        ).launch(
            grid=(key.tokens,), block=(THREADS,), stream=stream
        )
        pooled_norm(xn_w, xin, scratch).launch(
            grid=(key.tokens,), block=(THREADS,), stream=stream
        )

    return launch
