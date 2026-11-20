"""Standalone FlyDSL parity kernel for the K2 packed Joint epilogue."""

from functools import cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.quant_utils import (
    emit_cvt_scalef32_pk8_fp4_bf16,
    emit_mx_e8m0_scale,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    ptr_arg,
    ptr_rsrc,
)
from aiter.utility.mx_types import MxScaleRoundModeInt
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_rsq,
    xshfl,
)
from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import (
    _sigmoid,
)

M = 24
N = 4608
POOL = 4096
EXPERTS = 256
TOP_K = 8
THREADS = 128
WAVES = THREADS // 32
VALUES_PER_THREAD = POOL // THREADS


def _load_f32(resource, element):
    return fx.Float32(
        bo.buffer_load(resource, element, vec_width=1, dtype=T.f32)
    )


def _load_bf16(resource, element):
    word = fx.Int32(
        bo.buffer_load(resource, element // 2, vec_width=1, dtype=T.i32)
    )
    lo, hi = bf2_f32(word)
    return ((element & 1) == 0).select(lo, hi)


def _store_bf16_pair(resource, pair, lo, hi):
    bo.buffer_store(
        bf16_pair(lo, hi).bitcast(fx.Int32),
        resource,
        pair,
    )


@cache
def _compile_joint_epilogue(
    sched_strategy=None,
    expert_scheduling=True,
    diagnostic_task="both",
):
    @fx.struct
    class Shared:
        wave_sum: fx.Array[fx.Float32, WAVES, 16]
        candidate_score: fx.Array[fx.Float32, EXPERTS, 16]
        candidate_biased: fx.Array[fx.Float32, EXPERTS, 16]

    @flyc.kernel(
        name="muse_spark_gfx1250_joint_epilogue",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        partial: fx.Pointer,
        bias: fx.Pointer,
        norm: fx.Pointer,
        scores: fx.Pointer,
        ids: fx.Pointer,
        expert_input: fx.Pointer,
        quant: fx.Pointer,
        quant_scale: fx.Pointer,
        multiplier: fx.Float32,
        route_scale: fx.Float32,
        eps: fx.Float32,
    ):
        pid = fx.block_idx.x
        if diagnostic_task == "pool":
            pid += M
        tid = fx.thread_idx.x
        lane = tid & 31
        wave = tid >> 5
        storage = fx.SharedAllocator().allocate(Shared)
        wave_sum = storage.wave_sum.peek().view(fx.make_layout(WAVES, 1))
        candidate_score = storage.candidate_score.peek().view(
            fx.make_layout(EXPERTS, 1)
        )
        candidate_biased = storage.candidate_biased.peek().view(
            fx.make_layout(EXPERTS, 1)
        )
        partial_rsrc = ptr_rsrc(partial)
        bias_rsrc = ptr_rsrc(bias)
        norm_rsrc = ptr_rsrc(norm)
        scores_rsrc = ptr_rsrc(scores)
        ids_rsrc = ptr_rsrc(ids)
        expert_rsrc = ptr_rsrc(expert_input)
        quant_rsrc = ptr_rsrc(quant)
        quant_scale_rsrc = ptr_rsrc(quant_scale)

        if pid < M:
            candidates = (tid, tid + THREADS)
            raw = []
            biased = []
            for expert in candidates:
                logit = _load_f32(
                    partial_rsrc, pid * N + POOL + expert
                )
                logit += _load_f32(
                    partial_rsrc,
                    pid * N + POOL + EXPERTS + expert,
                )
                score = _sigmoid(logit * multiplier)
                raw.append(score)
                biased.append(score + _load_f32(bias_rsrc, expert))
            for slot, expert in enumerate(candidates):
                candidate_score[expert] = raw[slot]
                candidate_biased[expert] = biased[slot]
            gpu.barrier()

            if wave == 0:
                local_id = []
                local_score = []
                local_biased = []
                for slot in range_constexpr(EXPERTS // 32):
                    expert = lane + slot * 32
                    local_id.append(expert)
                    local_score.append(candidate_score[expert])
                    local_biased.append(candidate_biased[expert])
                picked_ids = []
                picked_scores = []
                for pick in range_constexpr(TOP_K):
                    best = fx.Float32(-1.0e30)
                    best_score = fx.Float32(0.0)
                    best_id = fx.Int32(EXPERTS)
                    for slot in range_constexpr(EXPERTS // 32):
                        candidate = local_biased[slot]
                        expert = fx.Int32(local_id[slot])
                        take = (candidate > best) | (
                            (candidate == best) & (expert < best_id)
                        )
                        best = take.select(candidate, best)
                        best_score = take.select(
                            local_score[slot], best_score
                        )
                        best_id = take.select(expert, best_id)
                    for offset in (16, 8, 4, 2, 1):
                        other = xshfl(best, offset)
                        other_score = xshfl(best_score, offset)
                        other_id = xshfl(best_id, offset)
                        take = (other > best) | (
                            (other == best) & (other_id < best_id)
                        )
                        best = take.select(other, best)
                        best_score = take.select(other_score, best_score)
                        best_id = take.select(other_id, best_id)
                    picked_ids.append(best_id)
                    picked_scores.append(best_score)
                    for slot in range_constexpr(EXPERTS // 32):
                        local_biased[slot] = (
                            fx.Int32(local_id[slot]) == best_id
                        ).select(
                            fx.Float32(-1.0e30),
                            local_biased[slot],
                        )
                if lane == 0:
                    denominator = fx.Float32(0.0)
                    for pick in range_constexpr(TOP_K):
                        denominator += picked_scores[pick]
                    for pick in range_constexpr(TOP_K):
                        bo.buffer_store(
                            picked_ids[pick],
                            ids_rsrc,
                            pid * TOP_K + pick,
                        )
                        bo.buffer_store(
                            picked_scores[pick]
                            * route_scale
                            / (denominator + fx.Float32(1e-15)),
                            scores_rsrc,
                            pid * TOP_K + pick,
                        )
        else:
            row = pid - M
            base = row * N + tid * VALUES_PER_THREAD
            values = []
            sum_sq = fx.Float32(0.0)
            for index in range_constexpr(VALUES_PER_THREAD):
                value = bf16_round(_load_f32(partial_rsrc, base + index))
                values.append(value)
                sum_sq += value * value
            sum_sq = butterfly(sum_sq, (16, 8, 4, 2, 1))
            if lane == 0:
                wave_sum[wave] = sum_sq
            gpu.barrier()
            total = fx.Float32(0.0)
            for source_wave in range_constexpr(WAVES):
                total += wave_sum[source_wave]
            rstd = hw_rsq(total / fx.Float32(POOL) + eps)

            outputs = []
            amax = fx.Float32(0.0)
            output_base = row * POOL + tid * VALUES_PER_THREAD
            for index in range_constexpr(VALUES_PER_THREAD):
                output = bf16_round(
                    values[index]
                    * rstd
                    * _load_bf16(
                        norm_rsrc, tid * VALUES_PER_THREAD + index
                    )
                )
                outputs.append(output)
                amax = fx.max(amax, abs(output))
            for pair in range_constexpr(VALUES_PER_THREAD // 2):
                _store_bf16_pair(
                    expert_rsrc,
                    output_base // 2 + pair,
                    outputs[2 * pair],
                    outputs[2 * pair + 1],
                )

            e8m0 = fx.Int32(
                emit_mx_e8m0_scale(
                    amax,
                    mode=MxScaleRoundModeInt.Even,
                )
            )
            scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            for group in range_constexpr(VALUES_PER_THREAD // 8):
                vector = fx.Vector.from_elements(
                    outputs[group * 8 : (group + 1) * 8],
                    fx.Float32,
                ).truncf(fx.Vector.make_type(8, fx.BFloat16))
                packed = emit_cvt_scalef32_pk8_fp4_bf16(
                    vector,
                    scale,
                    i32_ty=T.i32,
                )
                bo.buffer_store(
                    fx.Int32(packed),
                    quant_rsrc,
                    (row * 2048) // 4 + tid * 4 + group,
                )
            bo.buffer_store(
                fx.Uint8(e8m0),
                quant_scale_rsrc,
                row * 128 + tid,
            )

    @flyc.jit
    def launch(
        partial: fx.Pointer,
        bias: fx.Pointer,
        norm: fx.Pointer,
        scores: fx.Pointer,
        ids: fx.Pointer,
        expert_input: fx.Pointer,
        quant: fx.Pointer,
        quant_scale: fx.Pointer,
        multiplier: fx.Float32,
        route_scale: fx.Float32,
        eps: fx.Float32,
        stream: fx.Stream,
    ):
        launcher = kernel(
            partial,
            bias,
            norm,
            scores,
            ids,
            expert_input,
            quant,
            quant_scale,
            multiplier,
            route_scale,
            eps,
            value_attrs={
                "rocdl.flat_work_group_size": f"{THREADS},{THREADS}",
                "rocdl.waves_per_eu": 1,
            },
        )
        grid_x = 2 * M if diagnostic_task == "both" else M
        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": expert_scheduling,
        "unroll-threshold": 0,
    }
    if sched_strategy is not None:
        launch.compile_hints["llvm_options"][
            "amdgpu-sched-strategy"
        ] = sched_strategy
    return launch


def joint_epilogue(
    partial: torch.Tensor,
    bias: torch.Tensor,
    norm: torch.Tensor,
    scores: torch.Tensor,
    ids: torch.Tensor,
    expert_input: torch.Tensor,
    quant: torch.Tensor,
    quant_scale: torch.Tensor,
    multiplier: float,
    route_scale: float,
    eps: float,
    *,
    sched_strategy=None,
    expert_scheduling=True,
    diagnostic_task="both",
) -> None:
    """Launch the fixed M24/SK1 Joint epilogue parity kernel."""
    assert partial.shape == (1, M, N) and partial.dtype == torch.float32
    launch = _compile_joint_epilogue(
        sched_strategy,
        expert_scheduling,
        diagnostic_task,
    )
    stream = torch.cuda.current_stream(device=partial.device)
    _run_compiled(
        launch,
        ptr_arg(partial, fx.Float32),
        ptr_arg(bias, fx.Float32),
        ptr_arg(norm, fx.BFloat16),
        ptr_arg(scores, fx.Float32),
        ptr_arg(ids, fx.Int32),
        ptr_arg(expert_input, fx.BFloat16),
        ptr_arg(quant, fx.Uint8),
        ptr_arg(quant_scale, fx.Uint8),
        fx.Float32(multiplier),
        fx.Float32(route_scale),
        fx.Float32(eps),
        fx.Stream(stream.cuda_stream),
    )


__all__ = ["joint_epilogue"]
