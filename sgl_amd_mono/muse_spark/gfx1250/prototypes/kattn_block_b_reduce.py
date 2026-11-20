# SPDX-License-Identifier: Apache-2.0
"""Isolated Baseline-shaped K-attention split-state reduction.

The kernel consumes Triton's canonical ``[token, head, split]`` maximum and
denominator tensors plus ``[token, head, split, 64]`` FP32 partials.  It is a
compute-only prototype: no persistent mailbox or production scratch is used.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import butterfly, hw_exp2, rsrc
from sgl_amd_mono.muse_spark.gfx1250.config import HEAD_DIM

_STREAM = fx.Stream(None)
THREADS = 128
WAVES = 4
DIMS_PER_WAVE = HEAD_DIM // WAVES
NEG_INF = float("-inf")


def block_b_lds(splits: int):
    @fx.struct
    class Lds:
        weights: fx.Array[fx.Float32, splits, 16]
        inverse: fx.Array[fx.Float32, 1, 16]

    return Lds


def pipeline_metadata(splits: int) -> dict[str, object]:
    return {
        "symbol": kernel_symbol("gfx1250_kattn_block_b_reduce", p=splits),
        "splits": splits,
        "threads": THREADS,
        "waves": WAVES,
        "grid": "[tokens, heads]",
        "reduction_owner": "wave0 lanes 0..splits-1",
        "output_owner": "four waves x 16 contiguous dimensions",
        "stat_load_width_bytes": 4,
        "partial_load_width_bytes": 4,
        "output_store_width_bytes": 2,
        "lds_bytes": (splits + 1) * 4,
        "mailbox": False,
    }


@flyc.jit
def run_block_b_reduce(c):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    task, splits = c["task"], c["splits"]
    weights_lds, inverse_lds = c["weights"], c["inverse"]
    state_base = task * splits

    # Baseline's split axis is a power of two.  One lane owns each split
    # statistic; wave shuffles form the stable global max and denominator.
    if wave == 0:
        live = lane < splits
        source = fx.min(lane, fx.Int32(splits - 1))
        maximum = live.select(
            fx.Float32(
                bo.buffer_load(
                    rsrc(c["maximum"]),
                    state_base + source,
                    vec_width=1,
                    dtype=T.f32,
                )
            ),
            fx.Float32(NEG_INF),
        )
        denominator = live.select(
            fx.Float32(
                bo.buffer_load(
                    rsrc(c["denominator"]),
                    state_base + source,
                    vec_width=1,
                    dtype=T.f32,
                )
            ),
            fx.Float32(0.0),
        )
        maximum = (denominator > 0.0).select(maximum, fx.Float32(NEG_INF))
        global_max = butterfly(maximum, (16, 8, 4, 2, 1), op=fx.max)
        global_max = (global_max > NEG_INF).select(global_max, fx.Float32(0.0))
        weight = ((denominator > 0.0) & live).select(
            hw_exp2(maximum - global_max),
            fx.Float32(0.0),
        )
        total = butterfly(denominator * weight, (16, 8, 4, 2, 1))
        if lane < splits:
            fx.ptr_store(weight, weights_lds + lane)
        if lane == 0:
            inverse = (total > 0.0).select(
                fx.Float32(1.0) / total,
                fx.Float32(0.0),
            )
            fx.ptr_store(inverse, inverse_lds)
    gpu.barrier()

    if lane < DIMS_PER_WAVE:
        dim = wave * DIMS_PER_WAVE + lane
        accum = fx.Float32(0.0)
        for source in range_constexpr(splits):
            partial = fx.Float32(
                bo.buffer_load(
                    rsrc(c["partial"]),
                    (state_base + source) * HEAD_DIM + dim,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            accum = accum + partial * fx.ptr_load(weights_lds + source)
        bo.buffer_store(
            (accum * fx.ptr_load(inverse_lds)).to(fx.BFloat16),
            rsrc(c["out"]),
            task * HEAD_DIM + dim,
        )


def build_block_b_reduce(splits: int, tokens: int, heads: int):
    """Build an exact ``tokens * heads`` CTA compute-only launcher."""
    if splits not in (2, 4, 8, 16, 32):
        raise ValueError(f"Block B splits must be a power of two, got {splits}")
    tasks = tokens * heads
    Lds = block_b_lds(splits)
    name = kernel_symbol("gfx1250_kattn_block_b_reduce", p=splits)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
        out: Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        run_block_b_reduce(
            {
                "tid": tid,
                "lane": tid % 32,
                "wave": tid // 32,
                "task": fx.block_idx.x,
                "splits": splits,
                "weights": lds.weights.ptr,
                "inverse": lds.inverse.ptr,
                "partial": partial,
                "maximum": maximum,
                "denominator": denominator,
                "out": out,
            }
        )

    @flyc.jit
    def launch(
        partial: Int64,
        maximum: Int64,
        denominator: Int64,
        out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(partial, maximum, denominator, out).launch(
            grid=(tasks,), block=(THREADS,), stream=stream
        )

    return launch
