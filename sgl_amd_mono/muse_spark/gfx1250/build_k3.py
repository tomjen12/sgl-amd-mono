# SPDX-License-Identifier: Apache-2.0
"""Build the fixed-width gfx1250 TP4/EP4 Muse Spark K3 pipeline."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import peer_bases, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx1250.config import (
    DRAFT_TOKENS,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    MAX_TOKENS,
    MOE_HIDDEN,
    TOP_K,
    WIDTHS,
)
from sgl_amd_mono.muse_spark.gfx1250.device import (
    clock,
    mailbox_guard,
    step_tag,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    RESIDENT_BLOCKS,
    resident_rounds,
)
from sgl_amd_mono.muse_spark.gfx1250.k3_tdm import (
    TILE_M as K3_TDM_TILE_M,
)
from sgl_amd_mono.muse_spark.gfx1250.k3_tdm import build_k3_tdm_tasks
from sgl_amd_mono.muse_spark.gfx1250.stages import ffn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)
THREADS = 256
_ALIGN = 256
_K3_COMPACT_TILES = (
    LOCAL_EXPERTS
    + (MAX_TOKENS * TOP_K + K3_TDM_TILE_M - 1) // K3_TDM_TILE_M
    - 1
)
_K3_COMPACT_ROWS = _K3_COMPACT_TILES * K3_TDM_TILE_M


class K3Build(msgspec.Struct, frozen=True):
    """One fixed-width EP K3 build."""

    tokens: int = DRAFT_TOKENS
    npes: int = 4
    last: bool = False
    shuffled: bool = False
    fault_site: int = 0
    fault_after: int = 0
    staged_profile: bool = False
    phase_trace: bool = False
    unpool_profile_mode: int = 2
    expert_profile_mode: bool = True
    finalize_profile_mode: bool = True
    tdm_expert: bool = False
    tdm_fast_drain: bool = False


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
            ("k3_compact_qx", _K3_COMPACT_ROWS * MOE_HIDDEN * 2),
            ("k3_compact_qmid", _K3_COMPACT_ROWS * INTER),
            ("k3_compact_qmid_scale", _K3_COMPACT_ROWS * (INTER // 32)),
            ("k3_compact_y", _K3_COMPACT_ROWS * MOE_HIDDEN * 2),
            ("k3_unpool_part", 4 * MAX_TOKENS * HIDDEN * 4),
            ("k3_route_row", picks * 4),
            ("k3_route_count", LOCAL_EXPERTS * 4),
            ("k3_route_tile_base", LOCAL_EXPERTS * 4),
            ("k3_tile_expert", _K3_COMPACT_TILES * 4),
            ("k3_tile_rows", _K3_COMPACT_TILES * 4),
            ("k3_tile_count", 4),
            ("k3_fbuf", MAX_TOKENS * MOE_HIDDEN * 2),
            ("k3_branch", MAX_TOKENS * HIDDEN * 2),
            ("k3_finalize_scale", picks * 4),
            ("k3_phase_ticks", MAX_TOKENS * 7 * 8),
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
                ("k3_qx_done", MAX_TOKENS * TOP_K),
                ("k3_route_done", 1),
                ("k3_gate_worker_done", RESIDENT_BLOCKS),
                ("k3_down_worker_done", RESIDENT_BLOCKS),
                ("k3_finalize_scale_done", MAX_TOKENS * TOP_K),
                ("k3_finalize_worker_done", MAX_TOKENS * 12),
                ("k3_unpool_worker_done", RESIDENT_BLOCKS),
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


@flyc.jit
def _stamp_phase(c, enabled, phase):
    if enabled:
        if c["bid"] < c["S"] and c["tid"] == 0:
            bo.buffer_store(
                clock(),
                rsrc(c["phase_ticks"].value),
                c["bid"] * 7 + phase,
            )


def build_k3(key: K3Build):
    """Return a JIT launcher for persistent K3.

    Arguments follow the gfx950 standalone K3 contract: ``xin``, routing, native
    Quark expert tensors, boundary weights/buffers, scratch, peer memory, rank,
    epoch, layer and runtime ``last``. ``staged_profile`` retains the old
    stream-ordered expert/finalize/boundary launches solely to attribute time;
    production always uses the single persistent launch.
    """

    assert key.tokens in WIDTHS, (key.tokens, WIDTHS)
    assert key.npes in (1, 4), key.npes
    assert key.unpool_profile_mode in (0, 1, 2), key.unpool_profile_mode
    if key.tdm_expert and not key.expert_profile_mode:
        raise ValueError("K3 TDM requires the staged expert scheduler")
    if key.tdm_fast_drain and not key.tdm_expert:
        raise ValueError("K3 TDM fast drain requires the TDM expert path")
    Lds = ffn.k3_lds()
    tdm = (
        build_k3_tdm_tasks(
            threads=THREADS,
            persistent_workers=RESIDENT_BLOCKS,
            shuffled=key.shuffled,
            fast_drain=key.tdm_fast_drain,
        )
        if key.tdm_expert
        else None
    )
    keyed = key_tuple(key, SOURCES)
    suffix = dict(
        s=key.tokens,
        w=key.npes,
        l=int(key.last),
        a=int(key.shuffled),
        f=key.fault_site,
        d=int(key.staged_profile),
        t=int(key.phase_trace),
        u=key.unpool_profile_mode,
        e=int(key.expert_profile_mode),
        z=int(key.finalize_profile_mode),
        x=int(key.tdm_expert),
        r=int(key.tdm_fast_drain),
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
        shared = fx.SharedAllocator(static=False)
        lds = shared.allocate(Lds).peek()
        if tdm is not None:
            tdm_arena = shared.allocate(tdm.contract.arena_bytes, 128)._ptr
            tdm.bind_lds(tdm_arena)
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
        unpool_worker_done = sreg(
            scratch,
            persistent_layout["k3_unpool_worker_done"][0],
            "k3_unpool_worker_done",
        )
        qx_done = sreg(
            scratch,
            persistent_layout["k3_qx_done"][0],
            "k3_qx_done",
        )
        route_done = sreg(
            scratch,
            persistent_layout["k3_route_done"][0],
            "k3_route_done",
        )
        gate_worker_done = sreg(
            scratch,
            persistent_layout["k3_gate_worker_done"][0],
            "k3_gate_worker_done",
        )
        down_worker_done = sreg(
            scratch,
            persistent_layout["k3_down_worker_done"][0],
            "k3_down_worker_done",
        )
        finalize_scale_done = sreg(
            scratch,
            persistent_layout["k3_finalize_scale_done"][0],
            "k3_finalize_scale_done",
        )
        finalize_worker_done = sreg(
            scratch,
            persistent_layout["k3_finalize_worker_done"][0],
            "k3_finalize_worker_done",
        )
        bid, tid = c["bid"], c["tid"]
        _stamp_phase(c, key.phase_trace, 0)
        expert_picks = key.tokens * TOP_K
        if key.expert_profile_mode:
            if tdm is not None:
                if bid == 0:
                    ffn.stage_k3_route_compact(
                        c, expert_picks, K3_TDM_TILE_M
                    )
                    _publish(c["put"], route_done, 0, tid)
                _wait(c["poll"], [(route_done, 0, 1)])
            if bid < expert_picks:
                if tdm is not None:
                    ffn.stage_expert_quantize_compact(c, bid)
                else:
                    ffn.stage_expert_quantize(c, bid)
                _publish(c["put"], qx_done, bid, tid)
            _wait(
                c["poll"],
                [(qx_done, pick, 1) for pick in range(expert_picks)],
            )
            _stamp_phase(c, key.phase_trace, 1)

            gate_tasks = (
                _K3_COMPACT_TILES * tdm.contract.gate_tiles_per_pick
                if tdm is not None
                else expert_picks * (MOE_HIDDEN // 32)
            )
            for gate_round in range(resident_rounds(gate_tasks)):
                gate_task = bid + gate_round * RESIDENT_BLOCKS
                if gate_task < gate_tasks:
                    if tdm is not None:
                        tdm.gate(c, gate_task)
                    else:
                        ffn.stage_expert_gate_up_block(c, gate_task)
            _publish(c["put"], gate_worker_done, bid, tid)
            _wait(
                c["poll"],
                [
                    (gate_worker_done, worker, 1)
                    for worker in range(RESIDENT_BLOCKS)
                ],
            )
            _stamp_phase(c, key.phase_trace, 2)

            down_tasks = (
                _K3_COMPACT_TILES * tdm.contract.down_tiles_per_pick
                if tdm is not None
                else expert_picks * (MOE_HIDDEN // 16)
            )
            for down_round in range(resident_rounds(down_tasks)):
                down_task = bid + down_round * RESIDENT_BLOCKS
                if down_task < down_tasks:
                    if tdm is not None:
                        tdm.down(c, down_task)
                    else:
                        ffn.stage_expert_down_tile(c, down_task)
            _publish(c["put"], down_worker_done, bid, tid)
            _wait(
                c["poll"],
                [
                    (down_worker_done, worker, 1)
                    for worker in range(RESIDENT_BLOCKS)
                ],
            )
            _stamp_phase(c, key.phase_trace, 3)
            if bid < expert_picks:
                if tdm is not None:
                    ffn.stage_expert_gather_compact_y(c, bid)
                _publish(c["put"], expert_done, bid, tid)
        else:
            for expert_round in range_constexpr(
                resident_rounds(expert_picks)
            ):
                expert_task = bid + expert_round * RESIDENT_BLOCKS
                if expert_task < expert_picks:
                    ffn.stage_expert_pick(c, expert_task)
                    _publish(c["put"], expert_done, expert_task, tid)

        if key.finalize_profile_mode:
            if bid < expert_picks:
                _wait(c["poll"], [(expert_done, bid, 1)])
                ffn.stage_finalize_scale(c, bid)
                _publish(c["put"], finalize_scale_done, bid, tid)

            finalize_chunks = 12
            finalize_tasks = key.tokens * finalize_chunks
            for finalize_round in range(resident_rounds(finalize_tasks)):
                finalize_task = bid + finalize_round * RESIDENT_BLOCKS
                if finalize_task < finalize_tasks:
                    token = finalize_task // finalize_chunks
                    _wait(
                        c["poll"],
                        [
                            (finalize_scale_done, token * TOP_K + pick, 1)
                            for pick in range(TOP_K)
                        ],
                    )
                    if bid < key.tokens:
                        _stamp_phase(c, key.phase_trace, 4)
                    ffn.stage_finalize_publish(
                        c, finalize_task, finalize_chunks
                    )
                    _publish(
                        c["put"],
                        finalize_worker_done,
                        finalize_task,
                        tid,
                    )
            if bid < key.tokens:
                _wait(
                    c["poll"],
                    [
                        (
                            finalize_worker_done,
                            bid * finalize_chunks + chunk,
                            1,
                        )
                        for chunk in range(finalize_chunks)
                    ],
                )
                _stamp_phase(c, key.phase_trace, 5)
                _publish(c["put"], finalize_done, bid, tid)
        elif bid < key.tokens:
            _wait(
                c["poll"],
                [
                    (expert_done, bid * TOP_K + pick, 1)
                    for pick in range(TOP_K)
                ],
            )
            _stamp_phase(c, key.phase_trace, 4)
            ffn.stage_finalize_ep(c)
            _stamp_phase(c, key.phase_trace, 5)
            _publish(c["put"], finalize_done, bid, tid)

        if bid < key.tokens and key.unpool_profile_mode == 0:
            _wait(
                c["poll"],
                [(finalize_done, bid, 1)],
            )
            ffn.stage_unpool_boundary(c)
            _stamp_phase(c, key.phase_trace, 6)

        if key.unpool_profile_mode:
            _wait(
                c["poll"],
                [(finalize_done, token, 1) for token in range(key.tokens)],
            )
            unpool_tasks = (
                key.tokens * (HIDDEN // 16)
                if key.unpool_profile_mode == 1
                else
                ((key.tokens + 15) // 16) * (HIDDEN // 64) * 4
            )
            for unpool_round in range(resident_rounds(unpool_tasks)):
                unpool_task = bid + unpool_round * RESIDENT_BLOCKS
                if unpool_task < unpool_tasks:
                    if key.unpool_profile_mode == 1:
                        ffn.stage_unpool_gemv(c, unpool_task)
                    else:
                        ffn.stage_unpool_splitk_tile(c, unpool_task)
            _publish(c["put"], unpool_worker_done, bid, tid)
            if bid < key.tokens:
                _wait(
                    c["poll"],
                    [
                        (unpool_worker_done, worker, 1)
                        for worker in range(RESIDENT_BLOCKS)
                    ],
                )
                if key.unpool_profile_mode == 2:
                    ffn.stage_unpool_reduce_post(c)
                else:
                    ffn.stage_unpool_post(c)
                _stamp_phase(c, key.phase_trace, 6)

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
        if key.staged_profile:
            expert(*args).launch(
                grid=(key.tokens * TOP_K,), block=(THREADS,), stream=stream
            )
            finalize(*args).launch(
                grid=(key.tokens,), block=(THREADS,), stream=stream
            )
            boundary(*args).launch(
                grid=(key.tokens,), block=(THREADS,), stream=stream
            )
        else:
            persistent(*args).launch(
                grid=(RESIDENT_BLOCKS,), block=(THREADS,), stream=stream
            )

    return launch
