# SPDX-License-Identifier: Apache-2.0
"""Lane-map probe for the gfx1250 FP8 QK tile used by Target K-attn."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)


def _build_qk_probe():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
    from flydsl._mlir.dialects import rocdl as _rocdl
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Int64, T

    from sgl_amd_mono.common.ops import rsrc

    def pack_fp8(values):
        words = []
        for word in range_constexpr(8):
            lo = _rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(values[word * 4])),
                _to_raw(fx.Float32(values[word * 4 + 1])),
                fx.Int32(0).ir_value(),
                False,
            )
            packed = _rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(values[word * 4 + 2])),
                _to_raw(fx.Float32(values[word * 4 + 3])),
                lo,
                True,
            )
            words.append(fx.Int32(packed))
        return fx.Vector.from_elements(words, fx.Int32)

    @flyc.kernel(
        name="gfx1250_kattn_fp8_qk_lane_probe",
        known_block_size=[32, 1, 1],
    )
    def kernel(q: Int64, k: Int64, out: Int64):
        lane = fx.thread_idx.x
        query = lane % 16
        half = lane // 16
        q_words = []
        for chunk in range_constexpr(4):
            loaded = fx.Vector(
                bo.buffer_load(
                    rsrc(q),
                    query * 32 + half * 16 + chunk * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            q_words += [loaded[index] for index in range_constexpr(4)]
        q_values = fx.Vector.from_elements(
            q_words, fx.Int32
        ).bitcast(fx.BFloat16).to(fx.Float32)
        q_absmax = fx.Float32(1.0e-20)
        for index in range_constexpr(32):
            value = fx.Float32(q_values[index])
            q_absmax = fx.max(q_absmax, fx.max(value, -value))
        q_absmax = fx.max(
            q_absmax,
            q_absmax.shuffle_xor(fx.Int32(16), fx.Int32(32)),
        )
        q_inv = fx.Float32(448.0) / q_absmax
        q_fp8 = pack_fp8(
            [
                fx.Float32(q_values[index]) * q_inv
                for index in range_constexpr(32)
            ]
        )

        k_lo = fx.Vector(
            bo.buffer_load(
                rsrc(k),
                query * 16 + half * 8,
                vec_width=4,
                dtype=T.i32,
            )
        )
        k_hi = fx.Vector(
            bo.buffer_load(
                rsrc(k),
                query * 16 + half * 8 + 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        k_fp8 = k_lo.shuffle(k_hi, list(range(8)))
        zero = fx.Vector.filled(8, 0.0, fx.Float32)
        score = fx.Vector(
            _rocdl.wmma_f32_16x16x64_fp8_fp8_(
                T.vec(8, T.f32),
                _to_raw(k_fp8),
                _to_raw(q_fp8),
                _to_raw(zero),
            )
        )
        q_scale = q_absmax * fx.Float32(1.0 / 448.0)
        for element in range_constexpr(8):
            bo.buffer_store(
                fx.Float32(score[element]) * q_scale,
                rsrc(out),
                query * 16 + half * 8 + element,
            )

    @flyc.jit
    def launch(
        q: Int64,
        k: Int64,
        out: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(q, k, out).launch(grid=(1,), block=(32,), stream=stream)

    return launch


def test_gfx1250_kattn_fp8_qk_lane_map():
    torch.manual_seed(17)
    q = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda") * 0.5
    k = (
        torch.randn(16, 64, dtype=torch.float32, device="cuda") * 0.5
    ).to(torch.float8_e4m3fn)
    out = torch.empty(16, 16, dtype=torch.float32, device="cuda")

    _build_qk_probe()(
        q.data_ptr(),
        k.data_ptr(),
        out.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    qf = q.float()
    scale = qf.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-20) / 448.0
    quantized_q = (qf / scale).to(torch.float8_e4m3fn).float() * scale
    reference = quantized_q @ k.float().T
    torch.testing.assert_close(out, reference, atol=2.0e-3, rtol=2.0e-3)


def _build_pv_probe():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
    from flydsl._mlir.dialects import rocdl as _rocdl
    from flydsl.expr import gpu, range_constexpr
    from flydsl.expr.typing import Int64, T

    from sgl_amd_mono.common.ops import rsrc

    def pack_fp8(values):
        words = []
        for word in range_constexpr(8):
            lo = _rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(values[word * 4])),
                _to_raw(fx.Float32(values[word * 4 + 1])),
                fx.Int32(0).ir_value(),
                False,
            )
            packed = _rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(values[word * 4 + 2])),
                _to_raw(fx.Float32(values[word * 4 + 3])),
                lo,
                True,
            )
            words.append(fx.Int32(packed))
        return fx.Vector.from_elements(words, fx.Int32)

    @fx.struct
    class Lds:
        v: fx.Array[fx.Int32, 64 * 80 // 4, 16]

    @flyc.kernel(
        name="gfx1250_kattn_fp8_pv_lane_probe",
        known_block_size=[32, 1, 1],
    )
    def kernel(p: Int64, v: Int64, out: Int64):
        lane = fx.thread_idx.x
        query = lane % 16
        half = lane // 16
        lds = fx.SharedAllocator().allocate(Lds).peek()
        words = lds.v.ptr
        base = fx.recast_iter(fx.Int8, words)

        for iteration in range_constexpr(8):
            chunk = lane + iteration * 32
            row = chunk // 4
            dim = (chunk % 4) * 16
            packed = fx.Vector(
                bo.buffer_load(
                    rsrc(v),
                    row * 16 + dim // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            fx.ptr_store(
                packed,
                words + (row * 80 + dim) // 4,
            )
        gpu.barrier()

        probabilities = []
        for tile in range_constexpr(4):
            for element in range_constexpr(8):
                probabilities.append(
                    fx.Float32(
                        bo.buffer_load(
                            rsrc(p),
                            query * 64 + tile * 16 + half * 8 + element,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                )
        p_fp8 = pack_fp8(probabilities)
        tr_row = (
            (lane // 16) * 8
            + lane % 4
            + ((lane % 16) // 8) * 4
        )
        tr_col = (((lane % 16) // 4) % 2) * 8
        zero = fx.Vector.filled(8, 0.0, fx.Float32)
        for dim_tile in range_constexpr(4):
            raws = []
            for tile in range_constexpr(4):
                pointer = fx.to_llvm_ptr(
                    base
                    + tile * 16 * 80
                    + dim_tile * 16
                    + tr_row * 80
                    + tr_col
                )
                raws.append(
                    fx.Vector(
                        _rocdl.ds_load_tr8_b64(T.vec(2, T.i32), pointer)
                    )
                )
            v_fp8 = (
                raws[0]
                .shuffle(raws[1], [0, 1, 2, 3])
                .shuffle(
                    raws[2].shuffle(raws[3], [0, 1, 2, 3]),
                    list(range(8)),
                )
            )
            result = fx.Vector(
                _rocdl.wmma_f32_16x16x64_fp8_fp8_(
                    T.vec(8, T.f32),
                    _to_raw(v_fp8),
                    _to_raw(p_fp8),
                    _to_raw(zero),
                )
            )
            for element in range_constexpr(8):
                bo.buffer_store(
                    fx.Float32(result[element]),
                    rsrc(out),
                    query * 64 + dim_tile * 16 + half * 8 + element,
                )

    @flyc.jit
    def launch(
        p: Int64,
        v: Int64,
        out: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(p, v, out).launch(grid=(1,), block=(32,), stream=stream)

    return launch


def test_gfx1250_kattn_fp8_pv_lane_map():
    torch.manual_seed(23)
    p = torch.rand(16, 64, dtype=torch.float32, device="cuda") * 0.5
    v = (
        torch.randn(64, 64, dtype=torch.float32, device="cuda") * 0.5
    ).to(torch.float8_e4m3fn)
    out = torch.empty(16, 64, dtype=torch.float32, device="cuda")

    _build_pv_probe()(
        p.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    quantized_p = p.to(torch.float8_e4m3fn).float()
    reference = quantized_p @ v.float()
    torch.testing.assert_close(out, reference, atol=2.0e-3, rtol=2.0e-3)
