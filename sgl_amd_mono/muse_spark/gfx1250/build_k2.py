# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness-first gfx1250 Muse Spark K2 builder."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import peer_bases, rsrc
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
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
from sgl_amd_mono.muse_spark.gfx1250.device import (
    clock,
    mailbox_guard,
    step_tag,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import post_attn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)
THREADS = 256
WAVES = THREADS // 32
_ALIGN = 256
JOINT_TDM_ARENA_ELEMS = (
    post_attn.JOINT_TDM_BUFFERS * post_attn.JOINT_TDM_SLOT_ELEMS
)
O_GROUPS = HIDDEN // (post_attn.ROWS * post_attn.WAVES)
ROUTER_GROUPS = EXPERTS // (post_attn.ROWS * post_attn.WAVES)
POOL_GROUPS = MOE_HIDDEN // (post_attn.ROWS * post_attn.WAVES)
ROUTER_TILES = EXPERTS // post_attn.ROWS
POOL_TILES = MOE_HIDDEN // post_attn.ROWS
K2_PHASE_POINTS = 7


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
            (
                "k2_joint",
                MAX_TOKENS * post_attn.JOINT_ROWS * 4,
            ),
            ("k2_joint_arrivals", 4),
            (
                "k2_phase_ticks",
                RESIDENT_BLOCKS * K2_PHASE_POINTS * 8,
            ),
        ]
    )


def _persistent_layout():
    """K2 data plus tagged device-scope hand-off flags."""
    out = k2_scratch_layout()
    offset = max(off + size for off, size in out.values())
    out.update(
        pair_layout(
            [
                ("k2_norm_done", MAX_TOKENS * (Q_K // HEAD_DIM)),
                ("k2_boundary_done", MAX_TOKENS),
                ("k2_boundary_all_done", 1),
                ("k2_router_done", ROUTER_TILES),
                ("k2_pool_done", RESIDENT_BLOCKS),
                ("k2_joint_counter_ready", 1),
                ("k2_joint_all_done", 1),
            ],
            align=_ALIGN,
            start=(offset + _ALIGN - 1) // _ALIGN * _ALIGN,
        )
    )
    return out


def scratch_bytes(layout=None):
    layout = _persistent_layout() if layout is None else layout
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
    staged_profile: bool = False
    lds_og: bool = False
    cooperative_rp: bool = False
    aggregate_boundary: bool = False
    single_boundary_wait: bool = False
    lds_x_rp: bool = False
    joint_rp: bool = False
    joint_rp_tdm: bool = False
    phase_trace: bool = False


@flyc.jit
def _publish(put, flag, index, tid):
    gpu.barrier()
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
    if tid == 0:
        put(flag, index, fx.Int32(1))


@flyc.jit
def _publish_wave(put, flag, index, lane):
    """Publish one wave-owned result without a CTA-wide barrier."""
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
    if lane == 0:
        put(flag, index, fx.Int32(1))


@flyc.jit
def _wait(poll, specs):
    values = poll(specs, batch=4)
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
    return values


@flyc.jit
def _stamp_phase(enabled, ticks, bid, tid, phase):
    if enabled:
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(
                clock(),
                rsrc(ticks),
                bid * K2_PHASE_POINTS + phase,
            )


def build_k2(key: K2Build):
    """Build K2 for a supported Target width."""
    if key.tokens not in WIDTHS:
        raise ValueError(f"gfx1250 K2 supports widths {WIDTHS}, got {key.tokens}")
    if key.npes not in (1, 4):
        raise ValueError(f"gfx1250 K2 V1 supports TP1 or TP4, got {key.npes}")
    if key.lds_x_rp and key.tokens != 8:
        raise ValueError("K2 LDS-X Router/Pooler currently supports width 8")
    if key.joint_rp and key.tokens != 8:
        raise ValueError("K2 joint Router/Pooler currently supports width 8")
    if key.joint_rp and (key.cooperative_rp or key.lds_x_rp):
        raise ValueError("K2 joint Router/Pooler is a distinct execution path")
    if key.joint_rp and key.lds_og:
        raise ValueError("K2 joint Router/Pooler overlays the O-projection LDS")
    if key.joint_rp and key.staged_profile:
        raise ValueError("K2 joint Router/Pooler requires the persistent path")
    if key.joint_rp_tdm and not key.joint_rp:
        raise ValueError("K2 joint TDM requires the packed joint Router/Pooler")

    W = key.npes
    name = kernel_symbol(
        "muse_spark_gfx1250_k2",
        s=key.tokens,
        w=W,
        f=key.fault_site,
        d=int(key.staged_profile),
        l=int(key.lds_og),
        r=int(key.cooperative_rp),
        a=int(key.aggregate_boundary),
        q=int(key.single_boundary_wait),
        u=int(key.lds_x_rp),
        j=int(key.joint_rp),
        m=int(key.joint_rp_tdm),
        t=int(key.phase_trace),
    )
    keyed = key_tuple(key, SOURCES)
    layout = k2_scratch_layout()
    persistent_layout = _persistent_layout()
    sy = k2_sym_layout(W)

    def ptr(base, region):
        return base + fx.Int64(layout[region][0])

    @fx.struct
    class ProjectionLds:
        red: fx.Array[fx.Float32, post_attn.RED_FLOATS, 16]
        xrp: fx.Array[fx.Float32, post_attn.RP_X_PAIRS, 16]

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
        sync = {"put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll}
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
                "local_og": False,
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
        context = {
            "bid": fx.block_idx.x,
            "S": key.tokens,
            "B": EXPERTS // post_attn.ROWS,
            "task": fx.block_idx.x,
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
            "lds_x": key.lds_x_rp,
            "x_lds": lds.xrp.ptr,
        }
        if key.cooperative_rp:
            post_attn.stage_router_pool_cooperative(context)
        else:
            post_attn.stage_router_pool(context)

    @flyc.kernel(name=f"{name}_pool", known_block_size=[THREADS, 1, 1])
    def pool(w_pool: Int64, scratch: Int64):
        lds = fx.SharedAllocator().allocate(ProjectionLds).peek()
        tid = fx.thread_idx.x
        context = {
            "bid": fx.block_idx.x,
            "S": key.tokens,
            "B": RESIDENT_BLOCKS,
            "task": fx.block_idx.x,
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
            "lds_x": key.lds_x_rp,
            "x_lds": lds.xrp.ptr,
        }
        if key.cooperative_rp:
            post_attn.stage_router_pool_cooperative(context)
        else:
            post_attn.stage_router_pool(context)

    @fx.struct
    class TopLds:
        red: fx.Array[fx.Float32, post_attn.RED_FLOATS, 16]
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
                "red": lds.red.ptr,
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

    if key.joint_rp:

        @fx.struct
        class PersistentLds:
            # The TDM arena overlays O-projection/boundary reduction storage.
            # At 253,632 B it fits the gfx1250 256-KiB LDS budget while a
            # separate reduction allocation would not.
            arena: fx.Array[fx.BFloat16, JOINT_TDM_ARENA_ELEMS, 16]

        def persistent_lds_views(lds):
            arena_addr = fx.Int32(fx.ptrtoint(lds.arena.ptr))
            lds_f32 = fx.inttoptr(
                fx.PointerType.get(
                    elem_ty=fx.Float32.ir_type,
                    address_space=fx.AddressSpace.Shared,
                    alignment=16,
                ),
                arena_addr,
            )
            return lds_f32, lds_f32 + post_attn.RED_FLOATS

    else:

        @fx.struct
        class PersistentLds:
            og: fx.Array[fx.Float32, key.tokens * Q_K // 2, 16]
            red: fx.Array[fx.Float32, post_attn.RED_FLOATS, 16]
            xrp: fx.Array[fx.Float32, post_attn.RP_X_PAIRS, 16]
            picked: fx.Array[fx.Float32, TOP_K, 16]

        def persistent_lds_views(lds):
            return lds.red.ptr, lds.picked.ptr

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def persistent(
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
    ):
        lds = fx.SharedAllocator().allocate(PersistentLds).peek()
        red_lds, picked_lds = persistent_lds_views(lds)
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        lane, wave = tid % 32, tid // 32
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
        sync = {"put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll}
        bases = peer_bases(peers, W)
        norm_done = sreg(
            scratch,
            persistent_layout["k2_norm_done"][0],
            "k2_norm_done",
        )
        boundary_done = sreg(
            scratch,
            persistent_layout["k2_boundary_done"][0],
            "k2_boundary_done",
        )
        boundary_all_done = sreg(
            scratch,
            persistent_layout["k2_boundary_all_done"][0],
            "k2_boundary_all_done",
        )
        router_done = sreg(
            scratch,
            persistent_layout["k2_router_done"][0],
            "k2_router_done",
        )
        pool_done = sreg(
            scratch,
            persistent_layout["k2_pool_done"][0],
            "k2_pool_done",
        )
        joint_counter_ready = sreg(
            scratch,
            persistent_layout["k2_joint_counter_ready"][0],
            "k2_joint_counter_ready",
        )
        joint_all_done = sreg(
            scratch,
            persistent_layout["k2_joint_all_done"][0],
            "k2_joint_all_done",
        )
        joint_arrivals = ptr(scratch, "k2_joint_arrivals")
        phase_ticks = ptr(scratch, "k2_phase_ticks")
        _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 0)
        if key.joint_rp and bid == 0:
            if tid == 0:
                bo.buffer_store(
                    fx.Int32(0), rsrc(joint_arrivals), 0, cache_modifier=0
                )
            _publish(sync["put"], joint_counter_ready, 0, tid)

        if key.lds_og:
            if bid < O_GROUPS:
                post_attn.stage_norm_gate_lds(
                    {
                        "S": key.tokens,
                        "lane": lane,
                        "wave": wave,
                        "o": o,
                        "g": g,
                        "og": lds.og.ptr,
                    }
                )
        else:
            norm_task = bid * WAVES + wave
            if norm_task < key.tokens * (Q_K // HEAD_DIM):
                post_attn.stage_norm_gate(
                    {
                        "bid": norm_task,
                        "lane": lane,
                        "o": o,
                        "g": g,
                        "og": ptr(scratch, "k2_og"),
                    }
                )
                _publish_wave(sync["put"], norm_done, norm_task, lane)

        if bid < O_GROUPS:
            if not key.lds_og:
                _wait(
                    sync["poll"],
                    [
                        (norm_done, task, 1)
                        for task in range(
                            key.tokens * (Q_K // HEAD_DIM)
                        )
                    ],
                )
            post_attn.stage_o_projection(
                {
                    "W": W,
                    "S": key.tokens,
                    "bid": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "red": red_lds,
                    "rank": rank,
                    "og": (
                        lds.og.ptr
                        if key.lds_og
                        else ptr(scratch, "k2_og")
                    ),
                    "local_og": key.lds_og,
                    "w_o": w_o,
                    "put_bf": sync["put_bf"],
                    "peer_attn": [
                        preg(base, sy["k2_attn"][0], "k2_attn")
                        for base in bases
                    ],
                }
            )
        _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 1)

        if bid < key.tokens:
            post_attn.stage_boundary(
                {
                    "W": W,
                    "bid": bid,
                    "S": key.tokens,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "red": red_lds,
                    "res": res,
                    "alpha": alpha,
                    "beta": beta,
                    "pre_w": pre_w,
                    "x": ptr(scratch, "k2_x"),
                    "poll": sync["poll"],
                    "own_attn": preg(sym, sy["k2_attn"][0], "k2_attn"),
                }
            )
            _publish(sync["put"], boundary_done, bid, tid)
        _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 2)

        if key.aggregate_boundary and bid == RESIDENT_BLOCKS - 1:
            _wait(
                sync["poll"],
                [(boundary_done, token, 1) for token in range(key.tokens)],
            )
            _publish(sync["put"], boundary_all_done, 0, tid)

        if key.joint_rp:
            _wait(sync["poll"], [(joint_counter_ready, 0, 1)])
            _wait(
                sync["poll"],
                [
                    (boundary_done, token, 1)
                    for token in range(key.tokens)
                ],
            )
            if bid < post_attn.JOINT_TASKS:
                joint_context = {
                    "S": key.tokens,
                    "task": bid,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "x": ptr(scratch, "k2_x"),
                    # Joint mode reuses the existing w_pool ABI slot.  The
                    # runner binds it to LayerViews.w_joint.
                    "w_joint": w_pool,
                    "out": ptr(scratch, "k2_joint"),
                    "red": red_lds,
                    "tdm": lds.arena.ptr,
                }
                if key.joint_rp_tdm:
                    post_attn.stage_joint_router_pool_tdm(joint_context)
                else:
                    post_attn.stage_joint_router_pool(joint_context)
                fx.memory_fence(
                    syncscope="agent", ordering=fx.AtomicOrdering.Release
                )
                if tid == 0:
                    arrival = fx.Int32(
                        comm_ops.atomic_add_agent(
                            joint_arrivals, fx.Int32(1)
                        )
                    )
                    if arrival == post_attn.JOINT_TASKS - 1:
                        _publish_wave(
                            sync["put"], joint_all_done, 0, fx.Int32(0)
                        )
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 3)
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 4)
        elif key.cooperative_rp:
            _wait(
                sync["poll"],
                (
                    [(boundary_all_done, 0, 1)]
                    if key.aggregate_boundary
                    else [
                        (boundary_done, token, 1)
                        for token in range(key.tokens)
                    ]
                ),
            )
            if (bid >= 64) & (bid < 64 + ROUTER_TILES):
                router_task = bid - 64
                post_attn.stage_router_pool_cooperative(
                    {
                        "S": key.tokens,
                        "task": router_task,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "router": True,
                        "x": ptr(scratch, "k2_x"),
                        "w0": w_hi,
                        "w1": w_lo,
                        "out": ptr(scratch, "k2_logits"),
                        "red": red_lds,
                        "lds_x": key.lds_x_rp,
                        "x_lds": lds.xrp.ptr,
                    }
                )
                _publish(sync["put"], router_done, router_task, tid)
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 3)

            pool_task = bid
            while pool_task < POOL_TILES:
                post_attn.stage_router_pool_cooperative(
                    {
                        "S": key.tokens,
                        "task": pool_task,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "router": False,
                        "x": ptr(scratch, "k2_x"),
                        "w0": w_pool,
                        "w1": fx.Int64(0),
                        "out": ptr(scratch, "k2_pool"),
                        "red": red_lds,
                        "lds_x": key.lds_x_rp,
                        "x_lds": lds.xrp.ptr,
                    }
                )
                pool_task = pool_task + RESIDENT_BLOCKS
            _publish(sync["put"], pool_done, bid, tid)
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 4)
        else:
            if key.single_boundary_wait and bid < POOL_GROUPS:
                _wait(
                    sync["poll"],
                    (
                        [(boundary_all_done, 0, 1)]
                        if key.aggregate_boundary
                        else [
                            (boundary_done, token, 1)
                            for token in range(key.tokens)
                        ]
                    ),
                )
            if bid < ROUTER_GROUPS:
                if not key.single_boundary_wait:
                    _wait(
                        sync["poll"],
                        [
                            (boundary_done, token, 1)
                            for token in range(key.tokens)
                        ],
                    )
                post_attn.stage_router_pool(
                    {
                        "bid": bid,
                        "S": key.tokens,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "rows": EXPERTS,
                        "router": True,
                        "x": ptr(scratch, "k2_x"),
                        "w0": w_hi,
                        "w1": w_lo,
                        "out": ptr(scratch, "k2_logits"),
                        "red": red_lds,
                        "lds_x": key.lds_x_rp,
                        "x_lds": lds.xrp.ptr,
                    }
                )
                _publish(sync["put"], router_done, bid, tid)
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 3)

            if bid < POOL_GROUPS:
                if not key.single_boundary_wait:
                    _wait(
                        sync["poll"],
                        [
                            (boundary_done, token, 1)
                            for token in range(key.tokens)
                        ],
                    )
                post_attn.stage_router_pool(
                    {
                        "bid": bid,
                        "S": key.tokens,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "rows": MOE_HIDDEN,
                        "router": False,
                        "x": ptr(scratch, "k2_x"),
                        "w0": w_pool,
                        "w1": fx.Int64(0),
                        "out": ptr(scratch, "k2_pool"),
                        "red": red_lds,
                        "lds_x": key.lds_x_rp,
                        "x_lds": lds.xrp.ptr,
                    }
                )
                _publish(sync["put"], pool_done, bid, tid)
            _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 4)

        if bid < key.tokens:
            if key.joint_rp:
                _wait(sync["poll"], [(joint_all_done, 0, 1)])
            else:
                _wait(
                    sync["poll"],
                    [
                        (router_done, task, 1)
                        for task in range(
                            ROUTER_TILES
                            if key.cooperative_rp
                            else ROUTER_GROUPS
                        )
                    ],
                )
            if key.joint_rp:
                post_attn.stage_joint_top8(
                    {
                        "bid": bid,
                        "tid": tid,
                        "bias": bias,
                        "ids": ids,
                        "scores": scores,
                        "joint": ptr(scratch, "k2_joint"),
                        "red": red_lds,
                        "picked": picked_lds,
                    }
                )
            else:
                post_attn.stage_top8(
                    {
                        "bid": bid,
                        "tid": tid,
                        "bias": bias,
                        "ids": ids,
                        "scores": scores,
                        "logits": ptr(scratch, "k2_logits"),
                        "red": red_lds,
                        "picked": picked_lds,
                    }
                )
            if override_ids != 0 and tid == 0:
                for pick in range_constexpr(TOP_K):
                    offset = bid * TOP_K + pick
                    replacement = fx.Int32(
                        bo.buffer_load(
                            rsrc(override_ids),
                            offset,
                            vec_width=1,
                            dtype=T.i32,
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
        _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 5)

        if key.joint_rp:
            if (bid >= key.tokens) & (bid < 2 * key.tokens):
                _wait(sync["poll"], [(joint_all_done, 0, 1)])
                post_attn.stage_joint_pooled_norm(
                    {
                        "bid": bid - key.tokens,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "red": red_lds,
                        "joint": ptr(scratch, "k2_joint"),
                        "xn_w": xn_w,
                        "xin": xin,
                    }
                )
        else:
            if bid < key.tokens:
                _wait(
                    sync["poll"],
                    [
                        (pool_done, task, 1)
                        for task in range(
                            RESIDENT_BLOCKS
                            if key.cooperative_rp
                            else POOL_GROUPS
                        )
                    ],
                )
                post_attn.stage_pooled_norm(
                    {
                        "bid": bid,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "red": red_lds,
                        "pool": ptr(scratch, "k2_pool"),
                        "xn_w": xn_w,
                        "xin": xin,
                    }
                )
        _stamp_phase(key.phase_trace, phase_ticks, bid, tid, 6)

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
        if key.staged_profile:
            norm_gate(o, g, scratch).launch(
                grid=(key.tokens * (Q_K // HEAD_DIM),),
                block=(32,),
                stream=stream,
            )
            oproj(
                w_o, scratch, sym, peers, rank, epoch, layer
            ).launch(grid=(O_GROUPS,), block=(THREADS,), stream=stream)
            boundary(
                res,
                alpha,
                beta,
                pre_w,
                scratch,
                sym,
                peers,
                rank,
                epoch,
                layer,
            ).launch(grid=(key.tokens,), block=(THREADS,), stream=stream)
            if key.cooperative_rp:
                router(w_hi, w_lo, scratch).launch(
                    grid=(ROUTER_TILES,),
                    block=(THREADS,),
                    stream=stream,
                )
                pool(w_pool, scratch).launch(
                    grid=(POOL_TILES,),
                    block=(THREADS,),
                    stream=stream,
                )
            else:
                router(w_hi, w_lo, scratch).launch(
                    grid=(ROUTER_GROUPS,),
                    block=(THREADS,),
                    stream=stream,
                )
                pool(w_pool, scratch).launch(
                    grid=(POOL_GROUPS,),
                    block=(THREADS,),
                    stream=stream,
                )
            top8(
                bias,
                ids,
                scores,
                override_ids,
                override_scores,
                scratch,
            ).launch(grid=(key.tokens,), block=(THREADS,), stream=stream)
            pooled_norm(xn_w, xin, scratch).launch(
                grid=(key.tokens,), block=(THREADS,), stream=stream
            )
        else:
            persistent(
                o,
                g,
                res,
                w_o,
                alpha,
                beta,
                pre_w,
                w_hi,
                w_lo,
                bias,
                w_pool,
                xn_w,
                ids,
                scores,
                override_ids,
                override_scores,
                xin,
                scratch,
                sym,
                peers,
                rank,
                epoch,
                layer,
            ).launch(
                grid=(RESIDENT_BLOCKS,),
                block=(THREADS,),
                stream=stream,
            )

    return launch
