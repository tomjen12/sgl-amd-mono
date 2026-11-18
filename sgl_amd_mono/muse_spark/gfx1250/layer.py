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
    RESIDENT_BLOCKS,
    THREADS,
    fused_tasks,
    kattn_tasks,
    k1_scratch_layout,
    resident_rounds,
)
from sgl_amd_mono.muse_spark.gfx1250.stages import attention, pre_attn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


class K1Build(msgspec.Struct, frozen=True):
    tokens: int
    resident_blocks: int = RESIDENT_BLOCKS
    fault_site: int = 0
    fault_after: int = 0


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
        prefix_lens: Int64,
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
            "prefix_lens": prefix_lens,
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
        prefix_lens: Int64,
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
            prefix_lens,
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
    name = kernel_symbol(
        "muse_spark_gfx1250_kattn_tiled",
        s=key.tokens,
        q=key.q_len,
        c=key.resident_blocks,
    )
    keyed = key_tuple(key, SOURCES)
    layout = _tiled_kattn_layout()
    Lds = attention.fused_attn_lds()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def tiled_kattn(
        q: Int64,
        out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        table: Int64,
        table_stride: Int32,
        prefix_lens: Int64,
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
            "prefix_lens": prefix_lens,
            "window": window,
            "scale_bits": scale_bits,
        }
        for task_round in range_constexpr(
            resident_rounds(fused_tasks(key.tokens))
        ):
            task = bid + task_round * RESIDENT_BLOCKS
            if task < fused_tasks(key.tokens):
                attention.run_fused_attn(
                    {
                        "S": key.tokens,
                        "QL": key.q_len,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "tile": task // attention.FUSED_SPLITS,
                        "split": task % attention.FUSED_SPLITS,
                        "mb": mb,
                        "wave_stats": lds.wave_stats.ptr,
                        "row_stats": lds.row_stats.ptr,
                        "probabilities": lds.probabilities.ptr,
                        "pv": lds.pv.ptr,
                        **regions,
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
        prefix_lens: Int64,
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
            prefix_lens,
            window,
            scale_bits,
            scratch,
            epoch,
            layer_id,
        ).launch(grid=(RESIDENT_BLOCKS,), block=(THREADS,), stream=stream)

    return launch
