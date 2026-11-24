# SPDX-License-Identifier: Apache-2.0
"""Prepared PR #172 Mono launches for the logical-TP8/active-four benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from tp8_layer_contract import deterministic_router_bias
from tp8_synthetic_contract import (
    CONTRACT,
    deterministic_ep8_routing,
    validate_tensor,
)


@dataclass
class PreparedMono:
    stage: str
    launch: Callable[[], None]
    before_each: Callable[[], None]
    close: Callable[[], None]
    tensors: dict[str, dict]
    launch_count: int = 1
    outputs: Callable[[], dict[str, torch.Tensor]] | None = None


def prepare_mono(
    stage: str,
    rows: int,
    rank: int,
    *,
    window: bool = False,
    peer_group=None,
    active_npes: int = CONTRACT.active_npes,
    logical_case: Any | None = None,
) -> PreparedMono:
    import test_muse_spark_gfx1250_layer as blueprint
    import test_muse_spark_tp8_layer_gfx1250 as layer_test

    layer_test.ACTIVE_NPES = active_npes
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
    if stage == "k1" and logical_case is not None:
        layer.x0 = logical_case.x
        layer.w_qkvg = logical_case.weight
        layer.positions = logical_case.positions
        layer.cos_sin = logical_case.cos_sin
        layer.slots = logical_case.locations
        layer.k_cache = logical_case.cache_k
        layer.v_cache = logical_case.cache_v
        layer.k_cache.zero_()
        layer.v_cache.zero_()
    epoch = blueprint._epoch(device, deadline_s=60.0)
    stream = torch.cuda.current_stream()
    tensors = {
        "layer.x": validate_tensor(
            "layer.x",
            layer.x0,
            shape=(rows, CONTRACT.global_hidden),
            dtype=torch.bfloat16,
        ),
        "peer.addresses": validate_tensor(
            "peer.addresses",
            buffers.peers.addresses,
            shape=(CONTRACT.logical_tp,),
            dtype=torch.int64,
        ),
    }

    def before_each() -> None:
        epoch[0] += 1
        if stage == "k2" and logical_case is not None:
            buffers.res[:rows].copy_(logical_case.stream)
        if stage == "k3" and logical_case is not None:
            buffers.res[:rows].copy_(logical_case.stream)
        if stage in ("k2", "k3", "layer_v1", "layer_v2"):
            from tp8_shadow_protocol import seed_missing_rank_mailboxes

            seed_missing_rank_mailboxes(
                buffers.peers.bytes,
                rows=rows,
                step=int(epoch[0].item()),
                layer=layer_test.LAYER,
                active_npes=buffers.peers.active_npes,
                missing_router_partials=(
                    logical_case.missing_router_partials
                    if stage == "k2" and logical_case is not None
                    else None
                ),
            )

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
    outputs: Callable[[], dict[str, torch.Tensor]] | None = None
    if stage in ("kattn_full", "kattn_window") and logical_case is not None:
        layer.window = logical_case.window
        layer.k_cache = logical_case.cache_k.reshape(
            -1, CONTRACT.local_kv_heads, CONTRACT.head_dim
        )
        layer.v_cache = logical_case.cache_v.reshape(
            -1, CONTRACT.local_kv_heads, CONTRACT.head_dim
        )
        layer.table = logical_case.block_table
        layer.seq_lens = logical_case.sequence_lengths
        layer.prefix = (
            logical_case.sequence_lengths - CONTRACT.q_len
        ).contiguous()
        buffers.q[:rows].copy_(logical_case.q)
        buffers.k[:rows].copy_(logical_case.k_new)
        buffers.v[:rows].copy_(logical_case.v_new)
        buffers.g[:rows].copy_(logical_case.gate)

    if stage == "k1":
        tensors["k1.weight"] = validate_tensor(
            "k1.weight",
            layer.w_qkvg,
            shape=(CONTRACT.local_qkvg_rows, CONTRACT.global_hidden),
            dtype=torch.bfloat16,
        )
        launch = run_k1
        outputs = lambda: {
            "cache_k": layer.k_cache[logical_case.locations].clone(),
            "cache_v": layer.v_cache[logical_case.locations].clone(),
            "gate": buffers.g[:rows].clone(),
            "k": buffers.k[:rows].clone(),
            "q": buffers.q[:rows].clone(),
            "v": buffers.v[:rows].clone(),
        } if logical_case is not None else None
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
        outputs = lambda: {"output": buffers.o[:rows].clone()}
    elif stage == "k2":
        if logical_case is None:
            buffers.o[:rows].normal_()
            buffers.g[:rows].normal_()
        else:
            if logical_case.active_npes != active_npes:
                buffers.close()
                raise ValueError(
                    f"K2 logical case active_npes={logical_case.active_npes}, "
                    f"runner active_npes={active_npes}"
                )
            layer.w2k = {
                "w_o": logical_case.w_o,
                "alpha": logical_case.weights["alpha"],
                "beta": logical_case.weights["beta"],
                "pre_w": logical_case.weights["pre_w"],
                "w_router": logical_case.weights["router"],
                "bias": logical_case.weights["bias"],
                "w_pool": logical_case.weights["w_pool"],
                "xn_w": logical_case.weights["xn_w"],
            }
            layer.router = logical_case.weights["router"]
            buffers.o[:rows].copy_(logical_case.o)
            buffers.g[:rows].copy_(logical_case.gate)
            buffers.res[:rows].copy_(logical_case.stream)

        def k2() -> None:
            kernels["k2"](
                buffers.o.data_ptr(),
                *layer_test._k2_args(blueprint, layer, buffers, rank),
                buffers.peers.local,
                buffers.peers.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                layer_test.LAYER,
                stream=stream,
            )

        launch = k2
        outputs = lambda: {
            "expert_input": buffers.xin[:rows].clone(),
            "ids": buffers.ids[:rows].clone(),
            "scores": buffers.scores[:rows].clone(),
            "stream": buffers.res[:rows].clone(),
        }
    elif stage == "k3":
        if logical_case is None:
            ids, scores = deterministic_ep8_routing(rows)
            buffers.xin[:rows].normal_()
            buffers.ids[:rows].copy_(
                torch.tensor(ids, device=device, dtype=torch.int32)
            )
            buffers.scores[:rows].copy_(
                torch.tensor(scores, device=device, dtype=torch.float32)
            )
        else:
            if logical_case.active_npes != active_npes:
                buffers.close()
                raise ValueError(
                    f"K3 logical case active_npes={logical_case.active_npes}, "
                    f"runner active_npes={active_npes}"
                )
            layer.w13 = logical_case.w13
            layer.s13 = logical_case.s13
            layer.w2 = logical_case.w2
            layer.s2 = logical_case.s2
            layer.w3k = logical_case.weights
            buffers.xin[:rows].copy_(logical_case.xin)
            buffers.ids[:rows].copy_(logical_case.ids)
            buffers.scores[:rows].copy_(logical_case.scores)
            buffers.res[:rows].copy_(logical_case.stream)

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
        outputs = lambda: {
            "attention_input": buffers.attn_in[:rows].clone(),
            "stream": buffers.res[:rows].clone(),
        }
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
        tensors=tensors,
        outputs=outputs,
    )
