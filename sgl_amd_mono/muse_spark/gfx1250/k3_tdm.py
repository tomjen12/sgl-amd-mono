# SPDX-License-Identifier: Apache-2.0
"""AITER TDM GEMM tasks embedded in the standalone gfx1250 K3 pipeline."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import (
    build_gemm_a8w4_tdm_task,
)
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import rsrc
from sgl_amd_mono.muse_spark.gfx1250.config import (
    INTER,
    LOCAL_EXPERTS,
    MOE_HIDDEN,
)

TILE_M = 64
TILE_N = 256
TILE_K = 128
M_WARP = 2
N_WARP = 4
NUM_BUFFERS = 3
QUANT_WMMA_REP = TILE_M // M_WARP // 16
ROW_STRIDE_BYTES = MOE_HIDDEN * 2
GATE_N = 2 * INTER
DOWN_N = MOE_HIDDEN


@dataclass(frozen=True)
class K3TdmContract:
    """CPU-visible geometry and LDS contract for the embedded TDM tasks."""

    block: int
    arena_bytes: int
    gate_tiles_per_pick: int
    down_tiles_per_pick: int
    gate_name: str
    down_name: str


@dataclass(frozen=True)
class K3TdmTasks:
    """Traced task bodies plus their CPU-visible contract."""

    contract: K3TdmContract
    bind_lds: object
    gate: object
    down: object


def _build_task(
    *,
    stage1_act: int,
    stage1_quant_out: int,
    row_major_ascale: int,
    a_row_stride_bytes: int,
    persistent_workers: int,
    lds_holder: list,
    tile_meta_holder: list,
):
    return build_gemm_a8w4_tdm_task(
        K=MOE_HIDDEN,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        m_warp=M_WARP,
        n_warp=N_WARP,
        out_is_f16=0,
        num_buffers=NUM_BUFFERS,
        a_is_fp4=0,
        n_experts=LOCAL_EXPERTS,
        stage1_act=stage1_act,
        has_bias=0,
        stage1_quant_out=stage1_quant_out,
        quant_wmma_rep=QUANT_WMMA_REP,
        cluster_n=1,
        cluster_m=1,
        next_stage_prefetch=0,
        num_waves_per_tensor_tdm=2,
        tdm_as_in_prologue=0,
        tdm_b_th=0,
        enable_ep_scatter=0,
        ep_arena_handle=0,
        ep_combine_input_offset=0,
        ep_slot_stride_bytes=0,
        ep_destination_stride=0,
        ep_world_size=0,
        ep_quant_bits=0,
        row_major_ascale=row_major_ascale,
        a_row_stride_bytes=a_row_stride_bytes,
        a_scale_row_stride_bytes=a_row_stride_bytes,
        lds_soa_load_interleave=0,
        persistent_workers=persistent_workers,
        rolled_drain=0,
        a_preshuffled=0,
        m_align=0,
        explicit_tile=1,
        lds_base=lds_holder,
        a_ready=None,
        tile_meta=tile_meta_holder,
        item_hooks=None,
        raw_drain=0,
        defer_c_wait=0,
        tdm_snop=1,
        k_waves=1,
        single_row_store=0,
    )


def build_k3_tdm_tasks(*, threads: int, persistent_workers: int, shuffled: bool):
    """Build Gate/Up and Down helpers for one 256-thread persistent K3.

    AITER's TDM path consumes the preshuffled Muse weights.  Native Quark
    layout remains on the existing scalar-fragment path.
    """

    if not shuffled:
        raise ValueError("K3 TDM requires AITER-preshuffled MXFP4 weights")

    gate_lds, down_lds = [None], [None]
    gate_meta, down_meta = [None], [None]
    gate_task, gate_block, gate_name, _, gate_layout = _build_task(
        stage1_act=1,
        stage1_quant_out=1,
        row_major_ascale=1,
        a_row_stride_bytes=ROW_STRIDE_BYTES,
        persistent_workers=persistent_workers,
        lds_holder=gate_lds,
        tile_meta_holder=gate_meta,
    )
    down_task, down_block, down_name, _, down_layout = _build_task(
        stage1_act=0,
        stage1_quant_out=0,
        row_major_ascale=0,
        a_row_stride_bytes=MOE_HIDDEN,
        persistent_workers=persistent_workers,
        lds_holder=down_lds,
        tile_meta_holder=down_meta,
    )
    if gate_block != threads or down_block != threads:
        raise ValueError(
            f"K3 TDM block mismatch: gate={gate_block}, down={down_block}, "
            f"persistent={threads}"
        )
    if gate_layout["arena"] != down_layout["arena"]:
        raise ValueError(
            "K3 TDM Gate/Up and Down must reuse one equal-sized LDS arena"
        )

    def ptr(address):
        # Every K3 row/weight/scale allocation and offset is at least
        # 16-byte aligned.  AITER recasts activation and weight scales to
        # int32 pointers, so advertising byte alignment would make that
        # otherwise valid reinterpretation fail during FlyDSL compilation.
        return fx.inttoptr(
            fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=16,
            ),
            address,
        )

    def bind_lds(arena):
        gate_lds[0] = arena
        down_lds[0] = arena

    @flyc.jit
    def gate(c, task):
        m_tile = task // (GATE_N // TILE_N)
        n_tile = task % (GATE_N // TILE_N)
        tile_count = fx.Int32(
            bo.buffer_load(
                rsrc(c["tile_count"].value),
                0,
                vec_width=1,
                dtype=T.i32,
            )
        )
        if m_tile < tile_count:
            expert = fx.Int32(
                bo.buffer_load(
                    rsrc(c["tile_expert"].value),
                    m_tile,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            rows = fx.Int32(
                bo.buffer_load(
                    rsrc(c["tile_rows"].value),
                    m_tile,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            gate_meta[0] = (expert, rows)
            qx = c["compact_qx"].value
            qmid = c["compact_qmid"].value
            a = c["args"]
            dummy = ptr(a["ids"])
            gate_task(
                ptr(qmid),
                ptr(qx),
                ptr(a["w13"]),
                ptr(qx + MOE_HIDDEN),
                ptr(a["s13"]),
                dummy,
                dummy,
                ptr(c["compact_qmid_scale"].value),
                dummy,
                tile_count * TILE_M,
                fx.Int32(GATE_N),
                fx.Float32(float("inf")),
                fx.Float32(1.0),
                fx.Float32(1.0),
                m_tile,
                n_tile,
            )

    @flyc.jit
    def down(c, task):
        m_tile = task // (DOWN_N // TILE_N)
        n_tile = task % (DOWN_N // TILE_N)
        tile_count = fx.Int32(
            bo.buffer_load(
                rsrc(c["tile_count"].value),
                0,
                vec_width=1,
                dtype=T.i32,
            )
        )
        if m_tile < tile_count:
            expert = fx.Int32(
                bo.buffer_load(
                    rsrc(c["tile_expert"].value),
                    m_tile,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            rows = fx.Int32(
                bo.buffer_load(
                    rsrc(c["tile_rows"].value),
                    m_tile,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            down_meta[0] = (expert, rows)
            a = c["args"]
            dummy = ptr(a["ids"])
            down_task(
                ptr(c["compact_y"].value),
                ptr(c["compact_qmid"].value),
                ptr(a["w2"]),
                ptr(c["compact_qmid_scale"].value),
                ptr(a["s2"]),
                dummy,
                dummy,
                dummy,
                dummy,
                tile_count * TILE_M,
                fx.Int32(DOWN_N),
                fx.Float32(float("inf")),
                fx.Float32(1.0),
                fx.Float32(1.0),
                m_tile,
                n_tile,
            )

    contract = K3TdmContract(
        block=gate_block,
        arena_bytes=int(gate_layout["arena"]),
        gate_tiles_per_pick=GATE_N // TILE_N,
        down_tiles_per_pick=DOWN_N // TILE_N,
        gate_name=gate_name,
        down_name=down_name,
    )
    return K3TdmTasks(contract, bind_lds, gate, down)
