# SPDX-License-Identifier: Apache-2.0
"""Build the fixed-width gfx1250 TP4/EP4 Muse Spark K3 pipeline."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import peer_bases
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx1250.config import (
    DRAFT_TOKENS,
    HIDDEN,
    MAX_TOKENS,
    MOE_HIDDEN,
    TOP_K,
    WIDTHS,
)
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    RESIDENT_BLOCKS,
    resident_rounds,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import ffn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)
THREADS = 256
_ALIGN = 256


class K3Build(msgspec.Struct, frozen=True):
    """One fixed-width EP K3 build."""

    tokens: int = DRAFT_TOKENS
    npes: int = 4
    last: bool = False
    shuffled: bool = False
    fault_site: int = 0
    fault_after: int = 0


def _byte_layout(items, align=256):
    out, offset = {}, 0
    for name, size in items:
        offset = (offset + align - 1) // align * align
        out[name] = (offset, size)
        offset += size
    return out


def _scratch_layout():
    picks = MAX_TOKENS * TOP_K
    return _byte_layout(
        [
            ("k3_qx", picks * MOE_HIDDEN * 2),
            ("k3_qmid", picks * MOE_HIDDEN * 2),
            ("k3_y", picks * MOE_HIDDEN * 2),
            ("k3_fbuf", MAX_TOKENS * MOE_HIDDEN * 2),
            ("k3_branch", MAX_TOKENS * HIDDEN * 2),
        ]
    )


def _persistent_layout():
    """K3 data plus tagged device-scope hand-off flags."""
    out = _scratch_layout()
    offset = max(off + size for off, size in out.values())
    out.update(
        pair_layout(
            [
                ("k3_expert_done", MAX_TOKENS * TOP_K),
                ("k3_finalize_done", MAX_TOKENS),
            ],
            align=_ALIGN,
            start=(offset + _ALIGN - 1) // _ALIGN * _ALIGN,
        )
    )
    return out


def _sym_layout(npes):
    # One tagged mailbox pair carries two BF16 elements.
    return _byte_layout(
        [("k3_partial", npes * MAX_TOKENS * (MOE_HIDDEN // 2) * 8)]
    )


def k3_scratch_bytes():
    return max(off + size for off, size in _persistent_layout().values())


def k3_sym_bytes(npes):
    return max(off + size for off, size in _sym_layout(npes).values())


def _context(key, lds, args, scratch, sym, peers, rank, epoch, tag):
    mb = Mailbox(
        tag,
        mailbox_guard(
            epoch,
            own=sym,
            peers=peers,
            rank=rank,
            npes=key.npes,
            flags_offset=k3_sym_bytes(key.npes),
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        ),
    )
    layout = _scratch_layout()
    partial_off = _sym_layout(key.npes)["k3_partial"][0]
    bases = peer_bases(peers, key.npes)
    c = {
        "S": key.tokens,
        "W": key.npes,
        "tid": fx.thread_idx.x,
        "bid": fx.block_idx.x,
        "lane": fx.thread_idx.x % 32,
        "wave": fx.thread_idx.x // 32,
        "rank": rank,
        "shuffled": key.shuffled,
        "args": args,
        "red": lds.red.ptr,
        "aux": lds.aux.ptr,
        "sums": lds.sums.ptr,
        "put_bf": mb.put_bf,
        "put": mb.put,
        "poll": mb.poll,
        "own_partial": preg(sym, partial_off, "k3_partial"),
        "peer_partial": [
            preg(base, partial_off, "k3_partial") for base in bases
        ],
    }
    for name, (offset, _) in layout.items():
        c[name[3:]] = sreg(scratch, offset, name)
    return c


@flyc.jit
def _publish(put, flag, index, tid):
    """Release prior global stores and publish one tagged phase flag."""
    gpu.barrier()
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
    if tid == 0:
        put(flag, index, fx.Int32(1))


@flyc.jit
def _wait(poll, specs):
    """Poll tagged local phase flags, then acquire their payload stores."""
    values = poll(specs, batch=4)
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
    return values


def build_k3(key: K3Build):
    """Return a JIT launcher for the three stream-ordered K3 kernels.

    Arguments follow the gfx950 standalone K3 contract: ``xin``, routing, native
    Quark expert tensors, boundary weights/buffers, scratch, peer memory, rank,
    epoch, layer and runtime ``last``.
    """

    assert key.tokens in WIDTHS, (key.tokens, WIDTHS)
    assert key.npes in (1, 4), key.npes
    Lds = ffn.k3_lds()
    keyed = key_tuple(key, SOURCES)
    suffix = dict(
        s=key.tokens,
        w=key.npes,
        l=int(key.last),
        a=int(key.shuffled),
        f=key.fault_site,
    )
    persistent_name = kernel_symbol("muse_spark_gfx1250_k3", **suffix)
    persistent_layout = _persistent_layout()

    def arguments(
        xin,
        ids,
        scores,
        local_start,
        w13,
        s13,
        w2,
        s2,
        pen_w,
        res,
        pfw,
        alpha,
        beta,
        npw,
        w_unpool,
        attn_out,
        last,
    ):
        return {
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
            "alpha": alpha,
            "beta": beta,
            "npw": npw,
            "w_unpool": w_unpool,
            "attn_out": attn_out,
            "last": fx.Int32(1) if key.last else last,
        }

    @flyc.kernel(name=persistent_name, known_block_size=[THREADS, 1, 1])
    def persistent(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        c = _context(
            key,
            lds,
            arguments(
                xin,
                ids,
                scores,
                local_start,
                w13,
                s13,
                w2,
                s2,
                pen_w,
                res,
                pfw,
                alpha,
                beta,
                npw,
                w_unpool,
                attn_out,
                last,
            ),
            scratch,
            sym,
            peers,
            rank,
            epoch,
            step_tag(epoch, layer),
        )
        expert_done = sreg(
            scratch,
            persistent_layout["k3_expert_done"][0],
            "k3_expert_done",
        )
        finalize_done = sreg(
            scratch,
            persistent_layout["k3_finalize_done"][0],
            "k3_finalize_done",
        )
        bid, tid = c["bid"], c["tid"]
        for expert_round in range_constexpr(
            resident_rounds(key.tokens * TOP_K)
        ):
            expert_task = bid + expert_round * RESIDENT_BLOCKS
            if expert_task < key.tokens * TOP_K:
                ffn.stage_expert_pick(c, expert_task)
                _publish(c["put"], expert_done, expert_task, tid)

        if bid < key.tokens:
            _wait(
                c["poll"],
                [
                    (expert_done, bid * TOP_K + pick, 1)
                    for pick in range(TOP_K)
                ],
            )
            ffn.stage_finalize_ep(c)
            _publish(c["put"], finalize_done, bid, tid)
            _wait(c["poll"], [(finalize_done, bid, 1)])
            ffn.stage_unpool_boundary(c)

    @flyc.kernel(
        name=kernel_symbol("muse_spark_gfx1250_k3_expert", **suffix),
        known_block_size=[THREADS, 1, 1],
    )
    def expert(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        c = _context(
            key,
            lds,
            arguments(
                xin,
                ids,
                scores,
                local_start,
                w13,
                s13,
                w2,
                s2,
                pen_w,
                res,
                pfw,
                alpha,
                beta,
                npw,
                w_unpool,
                attn_out,
                last,
            ),
            scratch,
            sym,
            peers,
            rank,
            epoch,
            step_tag(epoch, layer),
        )
        ffn.stage_expert_pick(c)

    @flyc.kernel(
        name=kernel_symbol("muse_spark_gfx1250_k3_finalize", **suffix),
        known_block_size=[THREADS, 1, 1],
    )
    def finalize(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        c = _context(
            key,
            lds,
            arguments(
                xin,
                ids,
                scores,
                local_start,
                w13,
                s13,
                w2,
                s2,
                pen_w,
                res,
                pfw,
                alpha,
                beta,
                npw,
                w_unpool,
                attn_out,
                last,
            ),
            scratch,
            sym,
            peers,
            rank,
            epoch,
            step_tag(epoch, layer),
        )
        ffn.stage_finalize_ep(c)

    @flyc.kernel(
        name=kernel_symbol("muse_spark_gfx1250_k3_boundary", **suffix),
        known_block_size=[THREADS, 1, 1],
    )
    def boundary(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        c = _context(
            key,
            lds,
            arguments(
                xin,
                ids,
                scores,
                local_start,
                w13,
                s13,
                w2,
                s2,
                pen_w,
                res,
                pfw,
                alpha,
                beta,
                npw,
                w_unpool,
                attn_out,
                last,
            ),
            scratch,
            sym,
            peers,
            rank,
            epoch,
            step_tag(epoch, layer),
        )
        ffn.stage_unpool_boundary(c)

    @flyc.jit
    def launch(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        args = (
            xin,
            ids,
            scores,
            local_start,
            w13,
            s13,
            w2,
            s2,
            pen_w,
            res,
            pfw,
            alpha,
            beta,
            npw,
            w_unpool,
            attn_out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            layer,
            last,
        )
        persistent(*args).launch(
            grid=(RESIDENT_BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
