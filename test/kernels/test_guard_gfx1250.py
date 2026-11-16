"""gfx1250 clock and bounded mailbox-guard probes."""

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


def _build_cyclecounter_probe():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import rocdl
    from flydsl.expr.typing import Int64, T

    from sgl_amd_mono.common.ops import rsrc

    @flyc.kernel(
        name="gfx1250_probe_readcyclecounter", known_block_size=[64, 1, 1]
    )
    def kernel(out: Int64):
        start = fx.Int64(
            llvm.call_intrinsic(T.i64, "llvm.readcyclecounter", [], [], [])
        )
        iteration = fx.Int32(0)
        while iteration < 100_000:
            rocdl.s_sleep(127)
            iteration = iteration + 1
        end = fx.Int64(
            llvm.call_intrinsic(T.i64, "llvm.readcyclecounter", [], [], [])
        )
        elapsed = end - start
        if fx.thread_idx.x == 0:
            bo.buffer_store(fx.Int32(elapsed), rsrc(out), 0)
            bo.buffer_store(fx.Int32(elapsed >> 32), rsrc(out), 1)

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):
        kernel(out).launch(grid=(1,), block=(64,), stream=stream)

    return launch


def test_readcyclecounter_is_available_and_advances():
    import time

    launch = _build_cyclecounter_probe()
    elapsed = torch.zeros(2, dtype=torch.int32, device="cuda")
    started = time.monotonic()
    launch(elapsed.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    host_seconds = time.monotonic() - started

    def value(words):
        return (int(words[1]) & 0xFFFFFFFF) << 32 | (
            int(words[0]) & 0xFFFFFFFF
        )

    cycles = value(elapsed)
    print(f"readcyclecounter: {cycles} cycles in {host_seconds:.6f} s")
    assert cycles > 0


def _epoch(deadline_s):
    from sgl_amd_mono.common import guard

    out = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device="cuda")
    ticks = guard.deadline_ticks(deadline_s)
    out[guard.GUARD_DEADLINE] = ticks & 0xFFFFFFFF
    out[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return out


def _build_wait(publish, fault_site=None, stop=False):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import const_expr
    from flydsl.expr.typing import Int64

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.ops import rsrc
    from sgl_amd_mono.common.sync import Mailbox, sreg
    from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard

    name = (
        f"gfx1250_guard_p{int(publish)}_f{fault_site or 0}_s{int(stop)}"
    )

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(epoch: Int64, box: Int64):
        with guard.scope(
            mailbox_guard(
                epoch,
                fault_site=fault_site,
                stop=stop,
            )
        ):
            state = {"mailbox": Mailbox(fx.Int32(7))}
            if const_expr(publish):
                if (fx.block_idx.x == 1) & (fx.thread_idx.x == 0):
                    state["mailbox"].put(
                        sreg(box, 0, "probe"), 0, fx.Int32(42)
                    )
            got = state["mailbox"].poll(
                [(sreg(box, 0, "probe"), 0, 1)]
            )
            if (fx.block_idx.x == 0) & (fx.thread_idx.x == 0):
                bo.buffer_store(got[0][0], rsrc(box), 2)

    @flyc.jit
    def launch(
        epoch: Int64,
        box: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(epoch, box).launch(grid=(2,), block=(64,), stream=stream)

    return launch


def _error_words(epoch):
    from sgl_amd_mono.common import guard

    return [int(epoch[index]) for index in guard.ERROR_WORDS]


def test_satisfied_guarded_wait_is_clean():
    epoch = _epoch(0.2)
    box = torch.zeros(4, dtype=torch.int32, device="cuda")
    _build_wait(True)(
        epoch.data_ptr(), box.data_ptr(), stream=torch.cuda.current_stream()
    )
    torch.cuda.synchronize()
    assert _error_words(epoch) == [0, 0, 0]
    assert int(box[2]) == 42


def test_unsatisfied_guarded_wait_returns_with_deadline_error():
    import time

    epoch = _epoch(0.05)
    box = torch.zeros(4, dtype=torch.int32, device="cuda")
    started = time.monotonic()
    _build_wait(False)(
        epoch.data_ptr(), box.data_ptr(), stream=torch.cuda.current_stream()
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert _error_words(epoch)[0] == 2
    assert 0.02 < elapsed < 5.0


def test_fault_injection_times_out_named_wait():
    epoch = _epoch(0.05)
    box = torch.zeros(4, dtype=torch.int32, device="cuda")
    _build_wait(True, fault_site=1)(
        epoch.data_ptr(), box.data_ptr(), stream=torch.cuda.current_stream()
    )
    torch.cuda.synchronize()
    assert _error_words(epoch)[0] == 2
