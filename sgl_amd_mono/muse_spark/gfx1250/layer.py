# SPDX-License-Identifier: Apache-2.0
"""Standalone gfx1250 Muse Spark K1 builder."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.plan import kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.muse_spark.gfx1250.config import DRAFT_TOKENS, MAX_TOKENS, WIDTHS
from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard, step_tag
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    FUSED_KEY_BLOCK,
    FUSED_KEY_BLOCK_CHOICES,
    FUSED_LOAD_BITS_CHOICES,
    FUSED_SPLITS,
    RESIDENT_BLOCKS,
    THREADS,
    fused_tasks,
    kattn_tasks,
    k1_scratch_layout,
    resident_rounds,
    validate_fused_splits,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import attention, pre_attn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


class K1Build(msgspec.Struct, frozen=True):
    tokens: int
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0
    phase_trace: bool = False


def build_k1(key: K1Build):
    """Build a gfx1250 TP4 K1 launcher."""
    if key.tokens not in WIDTHS:
        raise ValueError(f"gfx1250 K1 supports widths {WIDTHS}, got {key.tokens}")
    if key.resident_blocks != RESIDENT_BLOCKS:
        raise ValueError(
            f"K1 profile has {RESIDENT_BLOCKS} resident blocks, "
            f"got {key.resident_blocks}"
        )
    name = kernel_symbol(
        "muse_spark_gfx1250_k1",
        s=key.tokens,
        c=key.resident_blocks,
        f=key.fault_site,
        t=int(key.phase_trace),
    )
    keyed = key_tuple(key, SOURCES)
    layout = k1_scratch_layout()
    Lds = pre_attn.k1_lds()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mailbox = Mailbox(
            step_tag(epoch, layer),
            mailbox_guard(
                epoch,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            ),
        )
        c = {
            "S": key.tokens,
            "tid": tid,
            "bid": fx.block_idx.x,
            "lane": tid % 32,
            "wave": tid // 32,
            "red": lds.red.ptr,
            "put": mailbox.put,
            "poll": mailbox.poll,
            "part": sreg(scratch, layout["k1_part"][0], "k1_part"),
            "phase_trace": key.phase_trace,
            "phase_ticks": sreg(
                scratch,
                layout["k1_phase_ticks"][0],
                "k1_phase_ticks",
            ),
            "args": {
                "x": x,
                "w": w,
                "cos_sin": cos_sin,
                "positions": positions,
                "slots": slots,
                "swa_map": swa_map,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "q_out": q_out,
                "k_out": k_out,
                "v_out": v_out,
                "g_out": g_out,
                "use_rope": use_rope,
                "use_swa": use_swa,
                "slot64": slot64,
                "swa64": swa64,
            },
        }
        pre_attn.run_k1(c, c["bid"])

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            x,
            w,
            cos_sin,
            positions,
            slots,
            swa_map,
            k_cache,
            v_cache,
            q_out,
            k_out,
            v_out,
            g_out,
            scratch,
            epoch,
            layer,
            use_rope,
            use_swa,
            slot64,
            swa64,
        ).launch(
            grid=(RESIDENT_BLOCKS,),
            block=(THREADS,),
            stream=stream,
        )

    return launch


class KAttnBuild(msgspec.Struct, frozen=True):
    tokens: int
    q_len: int = DRAFT_TOKENS
    resident_blocks: int = RESIDENT_BLOCKS
    phase_trace: bool = False
    splits: int = FUSED_SPLITS
    fp8_wmma: bool = False
    register_probability: bool = True
    stage_q: bool = True
    key_block: int = FUSED_KEY_BLOCK
    load_bits: int = 64
    triton_parity_v3: bool = False
    block_b_parity: bool = False


def build_kattn(key: KAttnBuild):
    """Build the correctness-first gfx1250 TP4 K-attn launcher."""
    if key.tokens not in WIDTHS or key.q_len != DRAFT_TOKENS or key.tokens % key.q_len:
        raise ValueError(
            f"gfx1250 K-attn supports tokens in {WIDTHS}, q_len={DRAFT_TOKENS}; "
            f"got {(key.tokens, key.q_len)}"
        )
    if key.resident_blocks != RESIDENT_BLOCKS:
        raise ValueError(
            f"K-attn profile has {RESIDENT_BLOCKS} resident blocks, "
            f"got {key.resident_blocks}"
        )
    if key.splits != FUSED_SPLITS:
        raise ValueError("the legacy K-attn kernel only supports six splits")
    name = kernel_symbol(
        "muse_spark_gfx1250_kattn",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
    )
    keyed = key_tuple(key, SOURCES)
    Lds = attention.kattn_lds()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        args = {
            "q": q,
            "out": out,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "table": table,
            "table_stride": table_stride,
            "seq_lens": seq_lens,
            "lens64": lens64,
            "window": window,
            "scale_bits": scale_bits,
        }
        for task_round in range_constexpr(
            resident_rounds(kattn_tasks(key.tokens))
        ):
            task = bid + task_round * RESIDENT_BLOCKS
            if task < kattn_tasks(key.tokens):
                attention.run_kattn(
                    {
                        "QL": key.q_len,
                        "tid": tid,
                        "bid": task,
                        "lane": tid % 32,
                        "wave": tid // 32,
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
        table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kattn(
            q,
            out,
            k_cache,
            v_cache,
            table,
            table_stride,
            seq_lens,
            lens64,
            window,
            scale_bits,
        ).launch(grid=(RESIDENT_BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def _tiled_kattn_layout():
    """Global split-key state and tagged hand-offs for standalone K-attn."""
    out = {}
    offset = 0
    for name, size in (
        ("attn_stats", attention.FUSED_STAT_FLOATS * 4),
        ("attn_global", attention.FUSED_GLOBAL_FLOATS * 4),
        ("attn_part", attention.FUSED_PART_FLOATS * 4),
        (
            "attn_phase_ticks",
            attention.FUSED_TASKS * attention.FUSED_TRACE_WORDS * 8,
        ),
    ):
        offset = (offset + 255) // 256 * 256
        out[name] = (offset, size)
        offset += size
    out.update(
        pair_layout(
            [
                ("attn_sflag", attention.FUSED_TASKS),
                ("attn_gflag", attention.FUSED_TILES),
                ("attn_pflag", attention.FUSED_TASKS),
            ],
            align=256,
            start=(offset + 255) // 256 * 256,
        )
    )
    return out


def tiled_kattn_scratch_bytes():
    layout = _tiled_kattn_layout()
    return max(offset + size for offset, size in layout.values())


def build_tiled_kattn(key: KAttnBuild):
    """Build the split-key WMMA standalone K-attn used by Target Split replay."""
    if key.tokens not in WIDTHS or key.q_len != DRAFT_TOKENS or key.tokens % key.q_len:
        raise ValueError(
            f"gfx1250 tiled K-attn supports tokens in {WIDTHS}, "
            f"q_len={DRAFT_TOKENS}; got {(key.tokens, key.q_len)}"
        )
    if key.resident_blocks != RESIDENT_BLOCKS or RESIDENT_BLOCKS != 96:
        raise ValueError(
            "gfx1250 tiled K-attn requires the cu96 profile; "
            f"got {key.resident_blocks}/{RESIDENT_BLOCKS}"
        )
    validate_fused_splits(key.splits)
    if key.key_block not in FUSED_KEY_BLOCK_CHOICES:
        raise ValueError(
            f"K-attn key block {key.key_block}, "
            f"want one of {FUSED_KEY_BLOCK_CHOICES}"
        )
    if key.load_bits not in FUSED_LOAD_BITS_CHOICES:
        raise ValueError(
            f"K-attn load width {key.load_bits}, "
            f"want one of {FUSED_LOAD_BITS_CHOICES}"
        )
    if key.fp8_wmma and (
        key.key_block != FUSED_KEY_BLOCK or key.load_bits != 64
    ):
        raise ValueError("FP8 K-attn only supports the retained BN32/64-bit load")
    if key.triton_parity_v3 and (
        key.tokens != DRAFT_TOKENS
        or key.splits != FUSED_SPLITS
        or key.fp8_wmma
    ):
        raise ValueError(
            "Triton-parity v3 opt-in requires width8, six splits, "
            "and BF16 WMMA"
        )
    if key.block_b_parity and not key.triton_parity_v3:
        raise ValueError("Block B parity requires the Block A v3 opt-in")
    if key.stage_q and (key.fp8_wmma or not key.register_probability):
        raise ValueError(
            "Q LDS staging requires BF16 WMMA with register probabilities"
        )
    name = kernel_symbol(
        "muse_spark_gfx1250_kattn_tiled",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
        t=int(key.phase_trace),
        p=key.splits,
        x=int(key.fp8_wmma),
        r=int(key.register_probability),
        y=int(key.stage_q),
        n=key.key_block,
        l=key.load_bits,
        v=int(key.triton_parity_v3),
        b=int(key.block_b_parity),
    )
    keyed = key_tuple(key, SOURCES)
    layout = _tiled_kattn_layout()
    if key.triton_parity_v3:
        from sgl_amd_mono.muse_spark.gfx1250.prototypes.kattn_triton_parity_v3 import (
            run_triton_parity_v3_compute,
            triton_parity_v3_lds,
        )
        Lds = triton_parity_v3_lds(compact_kv=True)
    else:
        Lds = (
            attention.fused_attn_fp8_lds()
            if key.fp8_wmma
            else attention.fused_attn_lds(
                key.register_probability, key.key_block
            )
        )
    run_attention = (
        attention.run_fused_attn_fp8
        if key.fp8_wmma
        else attention.run_fused_attn
    )
    block_threads = 64 if key.triton_parity_v3 else THREADS

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def tiled_kattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer_id: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 32
        wave = tid // 32
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = Mailbox(step_tag(epoch, layer_id))
        regions = {
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
        args = {
            "q": q,
            "out": out,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "table": table,
            "table_stride": table_stride,
            "seq_lens": seq_lens,
            "lens64": lens64,
            "window": window,
            "scale_bits": scale_bits,
        }
        for task_round in range_constexpr(
            resident_rounds(fused_tasks(key.tokens, key.splits))
        ):
            task = bid + task_round * RESIDENT_BLOCKS
            if task < fused_tasks(key.tokens, key.splits):
                context = (
                    {
                        "S": key.tokens,
                        "QL": key.q_len,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "tile": task // key.splits,
                        "split": task % key.splits,
                        "splits": key.splits,
                        "key_block": key.key_block,
                        "load_bits": key.load_bits,
                        "mb": mb,
                        "phase_trace": key.phase_trace,
                        "register_probability": key.register_probability,
                        "stage_q": key.stage_q,
                        "phase_ticks": sreg(
                            scratch,
                            layout["attn_phase_ticks"][0],
                            "attn_phase_ticks",
                        ),
                        **regions,
                        "args": args,
                    }
                )
                if const_expr(not key.triton_parity_v3):
                    context.update(
                        {
                            "wave_stats": lds.wave_stats.ptr,
                            "row_stats": lds.row_stats.ptr,
                            "q": (
                                lds.q.ptr
                                if not key.fp8_wmma
                                and key.register_probability
                                else lds.probabilities.ptr
                            ),
                            "probabilities": (
                                lds.q.ptr
                                if not key.fp8_wmma
                                and key.register_probability
                                else lds.probabilities.ptr
                            ),
                            "pv": lds.pv.ptr,
                            "kv": lds.kv.ptr,
                        }
                    )
                if const_expr(key.triton_parity_v3):
                    attention._fused_stamp(context, 0)
                    run_triton_parity_v3_compute(
                        {
                            "tid": tid,
                            "lane": lane,
                            "wave": wave,
                            "tile": task // key.splits,
                            "split": task % key.splits,
                            "splits": key.splits,
                            "q": lds.q.ptr,
                            "probabilities": lds.probabilities.ptr,
                            "maximum": lds.maximum.ptr,
                            "denominator": lds.denominator.ptr,
                            "alpha": lds.alpha.ptr,
                            "page_indices": lds.page_indices.ptr,
                            "k": lds.k.ptr,
                            "v": lds.v.ptr,
                            "hoist_q": True,
                            "register_probability": True,
                            "dedup_page": True,
                            "compact_kv": True,
                            "vector_epilogue": True,
                            "stage_pages": True,
                            "interleave_pv_pack": True,
                            "paired_page_load": True,
                            "inline_wmma_pair": False,
                            "interleaved_stats": True,
                            "partial": regions["part"].value,
                            "maximum_out": regions["stats"].value,
                            "denominator_out": regions["stats"].value,
                            "args": args,
                        }
                    )
                    attention._fused_stamp(context, 1)
                    merge_context = {
                        **context,
                        "merge_lds": fx.recast_iter(
                            fx.Float32, lds.q.ptr
                        ),
                        "threads": 64,
                    }
                    if const_expr(key.block_b_parity):
                        attention.merge_fused_precomputed_one_stage(
                            merge_context
                        )
                    else:
                        attention.merge_fused_precomputed(merge_context)
                else:
                    run_attention(context)

    @flyc.jit
    def launch(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        lens64: Int32,
        window: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer_id: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        tiled_kattn(
            q,
            out,
            k_cache,
            v_cache,
            table,
            table_stride,
            seq_lens,
            lens64,
            window,
            scale_bits,
            scratch,
            epoch,
            layer_id,
        ).launch(
            grid=(RESIDENT_BLOCKS,),
            block=(block_threads,),
            stream=stream,
        )

    return launch
