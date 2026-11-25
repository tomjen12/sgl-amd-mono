# SPDX-License-Identifier: Apache-2.0
"""Prepared PR #172 Mono launches for the logical-TP8/active-four benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from tp8_layer_contract import deterministic_router_bias
from tp8_synthetic_contract import (
    CONTRACT,
    ContractError,
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
    protocol: dict[str, Any] | None = None


def validate_peer_contract(peers: Any, *, active_npes: int, actual_world: int) -> None:
    if actual_world != active_npes:
        raise ContractError(
            f"active world size {actual_world}, expected {active_npes}"
        )
    if peers.logical_npes != CONTRACT.logical_tp:
        raise ContractError(
            f"logical peer slots {peers.logical_npes}, expected {CONTRACT.logical_tp}"
        )
    expected_shadows = CONTRACT.logical_tp - active_npes
    if len(peers.shadow_bytes) != expected_shadows:
        raise ContractError(
            f"shadow slots {len(peers.shadow_bytes)}, expected {expected_shadows}"
        )


def validate_sharded_pooler(value: torch.Tensor) -> dict[str, Any]:
    return validate_tensor(
        "k2.sharded_pooler",
        value,
        shape=(
            CONTRACT.moe_hidden // CONTRACT.logical_tp,
            CONTRACT.global_hidden,
        ),
        dtype=torch.bfloat16,
        alignment_bytes=16,
    )


def prepare_mono(
    stage: str,
    rows: int,
    rank: int,
    *,
    window: bool = False,
    peer_group=None,
    active_npes: int = CONTRACT.active_npes,
    logical_case: Any | None = None,
    seed_missing_ranks: bool = True,
    guard_stop: bool = True,
    deadline_s: float = 60.0,
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
    if not guard_stop:
        if stage != "k2":
            raise ValueError("test-only guard_stop=False is supported only for K2")
        from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2

        kernels["k2"] = build_k2(
            K2Build(tokens=rows, npes=CONTRACT.logical_tp, pshard=True, stop=False)
        )
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
    epoch = blueprint._epoch(device, deadline_s=deadline_s)
    def launch_stream() -> torch.cuda.Stream:
        return torch.cuda.current_stream(device)
    tensors = {
        "layer.x": validate_tensor(
            "layer.x",
            layer.x0,
            shape=(rows, CONTRACT.global_hidden),
            dtype=torch.bfloat16,
            stride=(CONTRACT.global_hidden, 1),
            alignment_bytes=16,
        ),
        "peer.addresses": validate_tensor(
            "peer.addresses",
            buffers.peers.addresses,
            shape=(CONTRACT.logical_tp,),
            dtype=torch.int64,
            stride=(1,),
            alignment_bytes=16,
        ),
    }

    def checked(
        name: str,
        value: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        contiguous: bool = True,
    ) -> None:
        tensors[name] = validate_tensor(
            name,
            value,
            shape=shape,
            dtype=dtype,
            contiguous=contiguous,
            alignment_bytes=16,
        )

    def validate_launch_tensors() -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise ContractError("Mono launch requires an initialized process group")
        actual_world = dist.get_world_size(peer_group)
        validate_peer_contract(
            buffers.peers,
            active_npes=active_npes,
            actual_world=actual_world,
        )
        checked(
            "layer.x",
            layer.x0,
            (rows, CONTRACT.global_hidden),
            torch.bfloat16,
        )
        checked(
            "peer.addresses",
            buffers.peers.addresses,
            (CONTRACT.logical_tp,),
            torch.int64,
        )
        if stage == "k1":
            checked(
                "k1.weight",
                layer.w_qkvg,
                (CONTRACT.local_qkvg_rows, CONTRACT.global_hidden),
                torch.bfloat16,
            )
            checked("k1.positions", layer.positions, (rows,), torch.int64)
            checked("k1.locations", layer.slots, (rows,), torch.int64)
        if stage in ("kattn_full", "kattn_window", "layer_v1", "layer_v2"):
            checked(
                "kattn.q",
                buffers.q[:rows],
                (rows, CONTRACT.local_q_width),
                torch.bfloat16,
            )
            checked(
                "kattn.k",
                buffers.k[:rows],
                (rows, CONTRACT.local_kv_width),
                torch.bfloat16,
            )
            checked(
                "kattn.v",
                buffers.v[:rows],
                (rows, CONTRACT.local_kv_width),
                torch.bfloat16,
            )
            checked(
                "kattn.gate",
                buffers.g[:rows],
                (rows, CONTRACT.local_q_width),
                torch.bfloat16,
            )
        if stage in ("k2", "layer_v1", "layer_v2"):
            pooler = getattr(buffers, "pool_w", None)
            if pooler is None:
                layer_test._k2_args(blueprint, layer, buffers, rank)
                pooler = buffers.pool_w
            tensors["k2.sharded_pooler"] = validate_sharded_pooler(pooler)
            checked(
                "k2.residual",
                buffers.res[:rows],
                (rows, CONTRACT.global_hidden),
                torch.float32,
            )
            checked(
                "k2.w_o",
                layer.w2k["w_o"],
                (CONTRACT.global_hidden, CONTRACT.local_q_width),
                torch.bfloat16,
            )
            checked(
                "k2.router",
                layer.w2k["w_router"],
                (CONTRACT.global_hidden, CONTRACT.global_experts),
                torch.float32,
            )
            checked(
                "k2.pooler",
                layer.w2k["w_pool"],
                (CONTRACT.moe_hidden, CONTRACT.global_hidden),
                torch.bfloat16,
            )
        if stage in ("k3", "layer_v1", "layer_v2"):
            checked(
                "k3.x",
                buffers.xin[:rows],
                (rows, CONTRACT.moe_hidden),
                torch.bfloat16,
            )
            checked(
                "k3.ids",
                buffers.ids[:rows],
                (rows, CONTRACT.top_k),
                torch.int32,
            )
            checked(
                "k3.scores",
                buffers.scores[:rows],
                (rows, CONTRACT.top_k),
                torch.float32,
            )
            for name in ("w13", "w2", "s13", "s2"):
                value = getattr(layer, name)
                if int(value.shape[0]) != CONTRACT.local_experts:
                    raise ContractError(
                        f"k3.{name} local experts {value.shape[0]}, "
                        f"expected {CONTRACT.local_experts}"
                    )
                checked(
                    f"k3.{name}",
                    value,
                    tuple(int(item) for item in value.shape),
                    value.dtype,
                )

    def before_each() -> None:
        epoch[0] += 1
        if stage == "k2" and logical_case is not None:
            buffers.res[:rows].copy_(logical_case.stream)
        if stage == "k3" and logical_case is not None:
            buffers.res[:rows].copy_(logical_case.stream)
        if seed_missing_ranks and stage in ("k2", "k3", "layer_v1", "layer_v2"):
            from tp8_shadow_protocol import seed_missing_rank_mailboxes

            # Publish only the absent logical ranks' current tagged payloads.
            # Active-rank mailboxes, shadow destinations, and scratch retain
            # prior bytes exactly as they do in the production Runner.
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
            # Other ranks read this rank's peer storage after the host control
            # barrier, so make these narrow publications visible first.
            torch.cuda.synchronize()

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
            launch_stream(),
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
                stream=launch_stream(),
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
                stream=launch_stream(),
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
            args = list(layer_test._k3_args(blueprint, layer, buffers, rank))
            if logical_case is not None:
                args[0] = logical_case.local_start
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
                stream=launch_stream(),
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
                launch_stream(),
            )

        launch = layer_launch
        outputs = lambda: {
            "residual": buffers.res[:rows].clone(),
            "ids": buffers.ids[:rows].clone(),
            "scores": buffers.scores[:rows].clone(),
            "expert_input": buffers.xin[:rows].clone(),
            "attention_input": buffers.attn_in[:rows].clone(),
            "cache_k": layer.k_cache[layer.slots].clone(),
            "cache_v": layer.v_cache[layer.slots].clone(),
            **(
                {
                    "next_q": buffers.q[:rows].clone(),
                    "next_k": buffers.k[:rows].clone(),
                    "next_v": buffers.v[:rows].clone(),
                    "next_gate": buffers.g[:rows].clone(),
                    "next_cache_k": layer.k_cache1[
                        layer.swa_map[layer.slots]
                    ].clone(),
                    "next_cache_v": layer.v_cache1[
                        layer.swa_map[layer.slots]
                    ].clone(),
                }
                if next_k1
                else {}
            ),
        }
    else:
        buffers.close()
        raise ValueError(f"unknown Mono stage {stage!r}")

    raw_launch = launch

    def validated_launch() -> None:
        validate_launch_tensors()
        raw_launch()

    return PreparedMono(
        stage=stage,
        launch=validated_launch,
        before_each=before_each,
        close=buffers.close,
        tensors=tensors,
        outputs=outputs,
        protocol={
            "epoch": epoch,
            "buffers": buffers,
            "layer": layer,
            "validate": validate_launch_tensors,
        },
    )
