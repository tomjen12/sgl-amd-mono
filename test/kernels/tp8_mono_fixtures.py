# SPDX-License-Identifier: Apache-2.0
"""Prepared PR #172 Mono launches for the logical-TP8/active-four benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tp8_layer_contract import deterministic_router_bias
from tp8_synthetic_contract import deterministic_ep8_routing


@dataclass
class PreparedMono:
    stage: str
    launch: Callable[[], None]
    before_each: Callable[[], None]
    close: Callable[[], None]
    launch_count: int = 1


def prepare_mono(
    stage: str,
    rows: int,
    rank: int,
    *,
    window: bool = False,
    peer_group=None,
) -> PreparedMono:
    import test_muse_spark_gfx1250_layer as blueprint
    import test_muse_spark_tp8_layer_gfx1250 as layer_test

    blueprint.PAGES = 512
    device = torch.device("cuda", rank)
    layer = blueprint.Layer(
        rows,
        8,
        device,
        seed=700 + rows,
        lens=[1000 + 137 * request for request in range(rows // 8)],
    )
    layer.router.zero_()
    layer.w2k["bias"].copy_(
        torch.tensor(deterministic_router_bias(), device=device)
    )
    kernels = layer_test._kernels(rows, 8)
    buffers = layer_test._buffers(
        blueprint, rows, 8, rank, device, group=peer_group
    )
    epoch = blueprint._epoch(device, deadline_s=60.0)
    stream = torch.cuda.current_stream()

    def before_each() -> None:
        epoch[0] += 1

    def run_k1() -> None:
        blueprint._k1(
            kernels["k1"],
            layer,
            buffers,
            layer.x0,
            layer.w_qkvg,
            layer.k_cache,
            layer.v_cache,
            epoch,
            layer_test.LAYER,
            0,
            stream,
        )

    buffers.res.copy_(layer.res0)
    run_k1()
    epoch[0] += 1
    torch.cuda.synchronize()

    if stage == "k1":
        launch = run_k1
    elif stage in ("kattn_full", "kattn_window"):
        # The layer fixture is full attention by default.  SWA changes only the
        # window argument consumed by the same production K-attn builder.
        if window:
            layer.window = 2047

        def kattn() -> None:
            kernels["ka"](
                *blueprint._attn_args(layer, buffers),
                epoch.data_ptr(),
                layer_test.LAYER,
                stream=stream,
            )

        launch = kattn
    elif stage == "k2":
        buffers.o[:rows].normal_()
        buffers.g[:rows].normal_()

        def k2() -> None:
            kernels["k2"](
                buffers.o.data_ptr(),
                *blueprint._k2_args(layer, buffers),
                buffers.peers.local,
                buffers.peers.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                layer_test.LAYER,
                stream=stream,
            )

        launch = k2
    elif stage == "k3":
        ids, scores = deterministic_ep8_routing(rows)
        buffers.xin[:rows].normal_()
        buffers.ids[:rows].copy_(
            torch.tensor(ids, device=device, dtype=torch.int32)
        )
        buffers.scores[:rows].copy_(
            torch.tensor(scores, device=device, dtype=torch.float32)
        )

        def k3() -> None:
            args = layer_test._k3_args(blueprint, layer, buffers, rank)
            kernels["k3"](
                buffers.xin.data_ptr(),
                buffers.ids.data_ptr(),
                buffers.scores.data_ptr(),
                args[0],
                *args[1:5],
                args[5],
                buffers.res.data_ptr(),
                *args[6:],
                buffers.peers.local,
                buffers.peers.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                layer_test.LAYER,
                0,
                stream=stream,
            )

        launch = k3
    elif stage in ("layer_v1", "layer_v2"):
        next_k1 = stage == "layer_v2"

        def layer_launch() -> None:
            layer_test._fused(
                blueprint,
                kernels,
                layer,
                buffers,
                epoch,
                rank,
                next_k1,
                stream,
            )

        launch = layer_launch
    else:
        buffers.close()
        raise ValueError(f"unknown Mono stage {stage!r}")

    return PreparedMono(
        stage=stage,
        launch=launch,
        before_each=before_each,
        close=buffers.close,
    )
