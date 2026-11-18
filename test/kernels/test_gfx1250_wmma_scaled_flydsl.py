"""Minimal FlyDSL lowering probe for gfx1250 mixed scaled WMMA."""

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


def _build_probe():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import Int64

    from sgl_amd_mono.common.ops import rsrc

    @flyc.kernel(
        name="gfx1250_probe_wmma_scale_f4f8", known_block_size=[32, 1, 1]
    )
    def kernel(out: Int64):
        a = fx.Vector.filled(8, 0, fx.Int32)
        b = fx.Vector.filled(16, 0, fx.Int32)
        c = fx.Vector.filled(8, 0.0, fx.Float32)
        result = fx.rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            fx.Vector.make_type(8, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(c),
            fx.Int32(0x7F7F7F7F).ir_value(),
            fx.Int32(0x7F7F7F7F).ir_value(),
            fmtA=4,
            fmtB=0,
        )
        values = fx.Vector(result)
        if fx.thread_idx.x == 0:
            for i in range(8):
                bo.buffer_store(values[i], rsrc(out), i)

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):
        kernel(out).launch(grid=(1,), block=(32,), stream=stream)

    return launch


def test_gfx1250_mixed_scaled_wmma_lowers_and_runs():
    out = torch.full((8,), -1.0, dtype=torch.float32, device="cuda")
    _build_probe()(out.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros_like(out))
