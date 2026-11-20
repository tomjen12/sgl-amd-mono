"""Focused gfx1250 ``global_load_async_to_lds_b128`` probe."""

from __future__ import annotations

import json

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    ptr_arg,
    ptr_buf_tensor,
)
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T


THREADS = 32
BYTES_PER_LANE = 16
COPY_BYTES = THREADS * BYTES_PER_LANE
_STREAM = fx.Stream(None)

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)


def _dword_iter(pointer):
    """Match the pointer recast used by the AITER MLA pagesize-64 kernel."""
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, pointer.memspace, 4),
        pointer,
    )


def build_direct_to_lds_probe():
    """Build one 16-byte global-to-LDS copy per lane, then copy LDS out."""

    @flyc.kernel(
        name="gfx1250_global_load_async_to_lds_b128_probe",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(src: fx.Pointer, dst: fx.Pointer):
        lane = fx.Int32(fx.thread_idx.x)
        lds_base = (
            fx.SharedAllocator(static=False).allocate(COPY_BYTES)._ptr
        )
        src_dwords = _dword_iter(src)

        # This is the exact pointer construction and API sequence used by
        # mla_pagesize64_fp8_fp8._rope_row_to_lds.
        rocdl.sched_barrier(0)
        gptr = fx.add_offset(src_dwords, lane * 4).llvm_ptr
        lds_ptr = fx.recast_iter(
            fx.Int8,
            fx.add_offset(lds_base, lane * BYTES_PER_LANE),
        ).llvm_ptr
        rocdl.sched_barrier(0)
        from flydsl._mlir.dialects import llvm as _llvm

        _llvm.inline_asm(
            None,
            [],
            "s_wait_alu depctr_va_vdst(0)",
            "",
            has_side_effects=True,
        )
        fx.rocdl.global_load_async_to_lds_b128(gptr, lds_ptr, 0)
        rocdl.sched_barrier(0)
        rocdl.s_wait_asynccnt(0)
        gpu.barrier()

        lds_dwords = fx.recast_iter(
            fx.PointerType.get(
                fx.Int32.ir_type,
                lds_base.memspace,
                4,
            ),
            lds_base,
        )
        words = fx.Vector(
            fx.ptr_load(
                lds_dwords + lane * 4,
                result_type=T.vec(4, T.i32),
            )
        )
        output = ptr_buf_tensor(dst, fx.Int32)
        for index in range(4):
            output[lane * 4 + index] = words[index]

    @flyc.jit
    def launch(
        src: fx.Pointer,
        dst: fx.Pointer,
        stream: fx.Stream = _STREAM,
    ):
        kernel(src, dst).launch(
            grid=(1,),
            block=(THREADS,),
            stream=stream,
        )

    return launch


def test_gfx1250_global_load_async_to_lds_b128_single_copy():
    source = torch.arange(COPY_BYTES, dtype=torch.uint8, device="cuda")
    destination = torch.full_like(source, 0xA5)
    launch = build_direct_to_lds_probe()

    _run_compiled(
        launch,
        ptr_arg(source, fx.Int8),
        ptr_arg(destination, fx.Int32),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    exact = torch.equal(destination, source)
    result = {
        "kernel": "gfx1250_global_load_async_to_lds_b128_probe",
        "threads": THREADS,
        "bytes": COPY_BYTES,
        "pointer_form": "fx.Pointer recast to i32 global / i8 LDS",
        "api": "fx.rocdl.global_load_async_to_lds_b128",
        "exact": exact,
    }
    print("GFX1250_DIRECT_LDS_PROBE=" + json.dumps(result), flush=True)
    assert exact


def build_ds_load_2addr_b32_probe():
    """Load two dwords 16 elements apart with one gfx1250 DS instruction."""

    @flyc.kernel(
        name="gfx1250_ds_load_2addr_b32_probe",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(dst: fx.Pointer):
        lane = fx.Int32(fx.thread_idx.x)
        storage = fx.SharedAllocator(static=False).allocate(32 * 4)
        words = fx.recast_iter(fx.Int32, storage._ptr)
        fx.ptr_store(lane, words + lane)
        gpu.barrier()

        address = fx.Int32(fx.ptrtoint(words + lane % 16))
        pair = fx.Vector(
            llvm.inline_asm(
                T.vec(2, T.i32),
                [address.ir_value()],
                "ds_load_2addr_b32 $0, $1 offset1:16",
                "=v,v,~{memory}",
                has_side_effects=True,
            )
        )
        rocdl.s_wait_dscnt(0)
        output = ptr_buf_tensor(dst, fx.Int32)
        output[lane * 2] = pair[0]
        output[lane * 2 + 1] = pair[1]

    @flyc.jit
    def launch(dst: fx.Pointer, stream: fx.Stream = _STREAM):
        kernel(dst).launch(grid=(1,), block=(THREADS,), stream=stream)

    return launch


def test_gfx1250_ds_load_2addr_b32_pair():
    destination = torch.full(
        (THREADS, 2), -1, dtype=torch.int32, device="cuda"
    )
    launch = build_ds_load_2addr_b32_probe()
    _run_compiled(
        launch,
        ptr_arg(destination, fx.Int32),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    lane = torch.arange(THREADS, dtype=torch.int32, device="cuda") % 16
    expected = torch.stack((lane, lane + 16), dim=1)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
