# SPDX-License-Identifier: Apache-2.0
"""Small gfx1250 device helpers used by the standalone K1 V1 kernel."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import rsrc, uniform
from sgl_amd_mono.muse_spark.gfx1250.config import LAYER_SLOTS

# ``llvm.readcyclecounter`` lowers on gfx1250 to a per-CU shader cycle counter.
# The host deadline uses gfx950's 100 MHz units.  Scaling by 25 bounds waits at
# a conservative 2.5 GHz; the target normally runs around 2.2 GHz, so the
# actual deadline is slightly longer than requested rather than prematurely
# timing out.
DEADLINE_SCALE = 25


def step_tag(epoch_ptr, layer):
    """Return this step/layer's tagged-mailbox generation."""
    step = uniform(bo.buffer_load(rsrc(epoch_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + fx.Int32(layer) + 1


def clock():
    """A monotonic clock within one gfx1250 CTA (not synchronized across CUs)."""
    return fx.Int64(
        llvm.call_intrinsic(T.i64, "llvm.readcyclecounter", [], [], [])
    )


def wait_stores():
    """Wait for global stores; FlyDSL's arch table does not list gfx1250 yet."""
    llvm.inline_asm(
        None,
        [],
        "s_wait_storecnt 0",
        "",
        has_side_effects=True,
    )


def mailbox_guard(
    epoch,
    *,
    own=None,
    peers=None,
    rank=None,
    npes=1,
    flags_offset=0,
    fault_site=None,
    fault_after=0,
    stop=True,
):
    """Build the common mailbox guard with gfx1250's shader-cycle clock."""
    return guard.Guard(
        epoch,
        own=own,
        peers=peers,
        rank=rank,
        npes=npes,
        flags_offset=flags_offset,
        fault_site=fault_site,
        fault_after=fault_after,
        stop=stop,
        clock_fn=clock,
        deadline_scale=DEADLINE_SCALE,
        wait_stores_fn=wait_stores,
    )
