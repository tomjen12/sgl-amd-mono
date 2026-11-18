"""Three-layer Muse Spark gfx1250 TP4/EP4 integration correctness.

This exercises checkpoint language-model layers 1, 2 and 3 at an 8192-token
prefix.  Layer 1 is full attention without RoPE; layers 2 and 3 are sliding
attention with RoPE.  Three paths are checked:

* explicit PyTorch model math,
* split K1 -> K-attn -> K2 -> K3 for every layer, and
* standalone first K1, Layer V2 for layers 1/2, then Layer V1 for layer 3.

All 64 rank-local experts start in native packed MXFP4 storage for the
reference, then are converted to AITER's shuffled gfx1250 layout for both
kernel paths.  The reference dequantizes only routed expert slices.
"""

import os
import sys
import time
import traceback

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYERS = (1, 2, 3)
PREFIX = 8192
KEYS = PREFIX + 8
FULL_PAGES = 1024
SWA_PAGES = 256
MAX_BLOCKS = 1024
POSITIONS = 16384
THETA = 500000.0
HOST_CEILING_S = 1500.0
WORKER_TIMEOUT_S = 1800.0


def _epoch(device, deadline_s=60.0):
    from sgl_amd_mono.common import guard

    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return epoch


def _cos_sin(device):
    from sgl_amd_mono.muse_spark.gfx1250.config import HEAD_DIM

    inv = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    frequencies = (
        torch.arange(POSITIONS, dtype=torch.float32)[:, None] * inv[None, :]
    )
    return torch.cat((frequencies.cos(), frequencies.sin()), -1).to(device)


def _packed_weight(experts, rows, columns, device, generator):
    """Allocate a complete native Quark MXFP4 table without a dense copy."""
    packed = torch.randint(
        0,
        256,
        (experts, rows, columns // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scales = torch.randint(
        120,
        128,
        (experts, rows, columns // 32),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return packed, scales


def _dequant_expert(packed, scales):
    """Dequantize one routed expert slice exactly as the native Quark layout."""
    from test_muse_spark_layer_gfx1250 import E2M1

    rows, half_columns = packed.shape
    columns = 2 * half_columns
    codes = torch.stack((packed & 0xF, packed >> 4), -1).reshape(rows, columns)
    magnitude = E2M1.to(packed.device)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -magnitude, magnitude)
    scale = (scales.to(torch.int32) << 23).view(torch.float32)
    return (
        values.reshape(rows, columns // 32, 32) * scale[..., None]
    ).reshape(rows, columns).to(torch.bfloat16)


def _local_expert_reference(xin, ids, scores, local_start, layer):
    """Compute only routed local experts, holding at most one dense pair."""
    from test_muse_spark_layer_gfx1250 import _bf, _mx_dequant

    from sgl_amd_mono.muse_spark.gfx1250.config import (
        INTER,
        LOCAL_EXPERTS,
        MOE_HIDDEN,
        POST_EPS,
    )

    out = torch.zeros(xin.shape[0], MOE_HIDDEN, device=xin.device)
    quantized_input = _mx_dequant(xin.float())
    local_ids = ids.long() - local_start
    routed = torch.unique(local_ids[(local_ids >= 0) & (local_ids < LOCAL_EXPERTS)])
    for expert_tensor in routed:
        expert = int(expert_tensor)
        dense13 = _dequant_expert(
            layer.get("ref_w13", layer["w13"])[expert],
            layer.get("ref_s13", layer["s13"])[expert],
        )
        dense2 = _dequant_expert(
            layer.get("ref_w2", layer["w2"])[expert],
            layer.get("ref_s2", layer["s2"])[expert],
        )
        locations = (local_ids == expert).nonzero()
        tokens, slots = locations[:, 0], locations[:, 1]
        gate_up = quantized_input[tokens] @ dense13.float().T
        gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
        middle = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
        value = _bf(
            _bf(middle @ dense2.float().T) * layer["weights"]["pen_w"].float()
        )
        scale = scores[tokens, slots] * torch.rsqrt(
            value.square().mean(-1) + POST_EPS
        )
        out.index_add_(0, tokens, scale[:, None] * value)
        del dense13, dense2
    return _bf(out)


def _layer_outputs(device, rows=8):
    from test_muse_spark_layer_gfx1250 import _outputs

    return _outputs(device, rows)


def _snapshot(outputs, q, k, v, gate):
    result = {name: value.clone() for name, value in outputs.items()}
    result.update(q=q.clone(), k=k.clone(), v=v.clone(), g=gate.clone())
    return result


def _make_page_layout(device, generator, rows):
    """Build one logical page sequence per eight-row verify request."""
    requests = rows // 8
    pages = (KEYS + 15) // 16
    prefix_pages = PREFIX // 16
    base_pages = torch.randperm(
        FULL_PAGES, device=device, generator=generator
    )
    current_pages = base_pages[prefix_pages : prefix_pages + requests]
    full_table = torch.zeros(
        requests, MAX_BLOCKS, dtype=torch.int32, device=device
    )
    full_table[:, :prefix_pages] = base_pages[:prefix_pages].to(torch.int32)
    full_table[:, prefix_pages] = current_pages.to(torch.int32)

    # Real deployments map a full-cache physical page into a smaller SWA pool.
    # A random page permutation followed by modulo gives a ring-like mapping;
    # the active 2048-token suffix is explicitly made collision-free.
    page_map = torch.randint(
        0,
        SWA_PAGES,
        (FULL_PAGES,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    active_count = min(129, pages)
    active_pages = torch.unique(full_table[:, pages - active_count : pages]).long()
    page_map[active_pages] = torch.randperm(
        SWA_PAGES, device=device, generator=generator
    )[: active_pages.numel()]
    swa_map = (
        page_map[:, None] * 16
        + torch.arange(16, dtype=torch.int64, device=device)[None, :]
    ).reshape(-1)
    swa_table = torch.zeros_like(full_table)
    swa_table[:, :pages] = page_map[full_table[:, :pages].long()].to(torch.int32)
    offsets = torch.arange(8, dtype=torch.int64, device=device)
    slots = (
        full_table[:, PREFIX // 16].long()[:, None] * 16 + offsets[None, :]
    ).reshape(-1)
    return full_table, swa_table, swa_map, slots


def _make_layer(rank, index, device, shared_generator, local_generator, layout):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HEAD_DIM,
        HIDDEN,
        INTER,
        LOCAL_EXPERTS,
        LOCAL_KV_HEADS,
        MOE_HIDDEN,
        Q_K,
        QKVG_ROWS,
    )

    bf16 = torch.bfloat16
    full_table, swa_table, swa_map, slots = layout
    sliding = index in (2, 3)
    weights = {
        "w_o": (
            torch.randn(
                HIDDEN, Q_K, device=device, generator=local_generator
            )
            * 0.02
        ).to(bf16),
        "attn_alpha": (
            torch.rand(HIDDEN, device=device, generator=shared_generator) * 0.5
            + 0.5
        ),
        "attn_beta": (
            torch.rand(HIDDEN, device=device, generator=shared_generator) * 0.5
        ),
        "pre_w": (
            1
            + 0.1
            * torch.randn(HIDDEN, device=device, generator=shared_generator)
        ).to(bf16),
        "w_hi": torch.zeros(EXPERTS, HIDDEN, dtype=bf16, device=device),
        "w_lo": torch.zeros(EXPERTS, HIDDEN, dtype=bf16, device=device),
        "bias": torch.full((EXPERTS,), -100.0, device=device),
        "w_pool": (
            torch.randn(
                MOE_HIDDEN, HIDDEN, device=device, generator=shared_generator
            )
            * 0.02
        ).to(bf16),
        "xn_w": (
            1
            + 0.1
            * torch.randn(MOE_HIDDEN, device=device, generator=shared_generator)
        ).to(bf16),
        "pen_w": (
            1
            + 0.05
            * torch.randn(MOE_HIDDEN, device=device, generator=shared_generator)
        ).to(bf16),
        "pfw": (
            1
            + 0.05
            * torch.randn(HIDDEN, device=device, generator=shared_generator)
        ).to(bf16),
        "ffn_alpha": (
            torch.rand(HIDDEN, device=device, generator=shared_generator) * 0.5
            + 0.5
        ),
        "ffn_beta": (
            torch.rand(HIDDEN, device=device, generator=shared_generator) * 0.5
        ),
        "npw": (
            1
            + 0.05
            * torch.randn(HIDDEN, device=device, generator=shared_generator)
        ).to(bf16),
        "w_unpool": (
            torch.randn(
                HIDDEN, MOE_HIDDEN, device=device, generator=shared_generator
            )
            * 0.01
        ).to(bf16),
    }
    routed_ids = (0, 1, 64, 65, 128, 129, 192, 193)
    for priority, expert in enumerate(routed_ids):
        weights["bias"][expert] = float(len(routed_ids) - priority)

    qkvg = (
        torch.randn(
            QKVG_ROWS, HIDDEN, device=device, generator=local_generator
        )
        * 0.02
    ).to(bf16)
    w13, s13 = _packed_weight(
        LOCAL_EXPERTS, 2 * INTER, MOE_HIDDEN, device, local_generator
    )
    w2, s2 = _packed_weight(
        LOCAL_EXPERTS, MOE_HIDDEN, INTER, device, local_generator
    )
    cache_slots = (
        SWA_PAGES * 16
        if sliding
        else FULL_PAGES * 16
    )
    k_initial = (
        torch.randn(
            cache_slots,
            LOCAL_KV_HEADS,
            HEAD_DIM,
            device=device,
            generator=local_generator,
        )
        * 2
    ).to(torch.float8_e4m3fn)
    v_initial = torch.randn(
        cache_slots,
        LOCAL_KV_HEADS,
        HEAD_DIM,
        device=device,
        generator=local_generator,
    ).to(torch.float8_e4m3fn)
    return {
        "index": index,
        "weights": weights,
        "qkvg": qkvg,
        "w13": w13,
        "s13": s13,
        "w2": w2,
        "s2": s2,
        "table": swa_table if sliding else full_table,
        "swa_map": swa_map,
        "slots": slots,
        "sliding": sliding,
        "rope": sliding,
        "window": 2047 if sliding else 0,
        "k_initial": k_initial,
        "v_initial": v_initial,
        "local_start": 64 * rank,
    }


def _distributed_layer_reference(
    layer, attn_input, stream, k_cache, v_cache, cos_sin, positions
):
    import torch.distributed as dist

    from test_muse_spark_k1_gfx1250 import _fp8, _reference as k1_reference
    from test_muse_spark_layer_gfx1250 import (
        _attention_reference,
        _k2_reference,
        _k3_boundary,
    )

    from sgl_amd_mono.muse_spark.gfx1250.config import SOFTMAX_SCALE

    q, k, v, gate = k1_reference(
        attn_input, layer["qkvg"], cos_sin, positions, layer["rope"]
    )
    write_slots = (
        layer["swa_map"][layer["slots"]]
        if layer["sliding"]
        else layer["slots"]
    )
    k_cache[write_slots] = _fp8(k.reshape(attn_input.shape[0], -1, 64))
    v_cache[write_slots] = _fp8(v.reshape(attn_input.shape[0], -1, 64))
    o = _attention_reference(
        q,
        k_cache,
        v_cache,
        layer["table"],
        KEYS,
        layer["window"],
        SOFTMAX_SCALE,
    )
    preliminary = _k2_reference(
        o, gate, stream, layer["weights"]
    )
    local_projection = (
        preliminary[4].float() @ layer["weights"]["w_o"].float().T
    ).to(torch.bfloat16)
    gathered_projection = [None] * NP
    dist.all_gather_object(gathered_projection, local_projection.cpu())
    k2 = _k2_reference(
        o,
        gate,
        stream,
        layer["weights"],
        partials=[partial.to(attn_input.device) for partial in gathered_projection],
    )
    local_experts = _local_expert_reference(
        k2[3], k2[1], k2[2], layer["local_start"], layer
    )
    gathered_experts = [None] * NP
    dist.all_gather_object(gathered_experts, local_experts.cpu())
    res, attn_out = _k3_boundary(
        [partial.to(attn_input.device) for partial in gathered_experts],
        layer["weights"],
        k2[0],
        layer["index"] == LAYERS[-1],
    )
    outputs = {
        "o": o,
        "ids": k2[1],
        "scores": k2[2],
        "xin": k2[3],
        "res": res,
        "attn_out": attn_out,
        "q": q,
        "k": k,
        "v": v,
        "g": gate,
    }
    return outputs, res, attn_out


def _distributed_post_reference(layer, o, gate, stream):
    """Reference K2/K3 from the path's actual attention output."""
    import torch.distributed as dist

    from test_muse_spark_layer_gfx1250 import _k2_reference, _k3_boundary

    preliminary = _k2_reference(o, gate, stream, layer["weights"])
    local_projection = (
        preliminary[4].float() @ layer["weights"]["w_o"].float().T
    ).to(torch.bfloat16)
    gathered_projection = [None] * NP
    dist.all_gather_object(gathered_projection, local_projection.cpu())
    k2 = _k2_reference(
        o,
        gate,
        stream,
        layer["weights"],
        partials=[partial.to(o.device) for partial in gathered_projection],
    )
    local_experts = _local_expert_reference(
        k2[3], k2[1], k2[2], layer["local_start"], layer
    )
    gathered_experts = [None] * NP
    dist.all_gather_object(gathered_experts, local_experts.cpu())
    res, attn_out = _k3_boundary(
        [partial.to(o.device) for partial in gathered_experts],
        layer["weights"],
        k2[0],
        layer["index"] == LAYERS[-1],
    )
    return {
        "o": o,
        "ids": k2[1],
        "scores": k2[2],
        "xin": k2[3],
        "res": res,
        "attn_out": attn_out,
    }


def _worker(rank, port, queue, rows):
    import torch.distributed as dist

    from test_muse_spark_k1_gfx1250 import (
        _close as k1_close,
        _fp8,
    )
    from test_muse_spark_layer_gfx1250 import (
        _assert_close,
        _reset,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import (
        K2Build,
        build_k2,
        scratch_bytes as k2_scratch_bytes,
        sym_bytes as k2_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import (
        K3Build,
        build_k3,
        k3_scratch_bytes,
        k3_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
        LayerBuild,
        build_layer,
        layer_scratch_bytes,
        layer_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        KV_K,
        Q_K,
        SOFTMAX_SCALE,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        K1Build,
        KAttnBuild,
        build_k1,
        build_kattn,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import (
        k1_scratch_layout,
        scratch_bytes as k1_scratch_bytes,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device = torch.device("cuda", rank)
    peers = []
    started = time.monotonic()
    try:
        # Establish the small uncached IPC mappings before allocating the
        # multi-GiB exact-shape expert tables.  This mirrors server startup and
        # avoids making peer VA placement depend on allocator fragmentation.
        split_k2_peers = PeerBuffer(
            k2_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        split_k3_peers = PeerBuffer(
            k3_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        fused_peers = PeerBuffer(
            layer_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        peers.extend((split_k2_peers, split_k3_peers, fused_peers))
        for peer in peers:
            peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()

        shared_generator = torch.Generator(device=device).manual_seed(125030)
        local_generator = torch.Generator(device=device).manual_seed(125100 + rank)
        layout = _make_page_layout(device, shared_generator, rows)
        layers = [
            _make_layer(
                rank,
                index,
                device,
                shared_generator,
                local_generator,
                layout,
            )
            for index in LAYERS
        ]
        cos_sin = _cos_sin(device)
        positions = torch.arange(
            PREFIX, KEYS, dtype=torch.int64, device=device
        ).repeat(rows // 8)
        prefix = torch.full(
            (rows // 8,), PREFIX, dtype=torch.int32, device=device
        )
        scale_bits = torch.tensor(
            [SOFTMAX_SCALE], dtype=torch.float32
        ).view(torch.int32).item()
        initial_stream = torch.randn(
            rows,
            8192,
            device=device,
            generator=shared_generator,
        )
        initial_attn_input = initial_stream.to(torch.bfloat16)

        oracle_k = [layer["k_initial"].clone() for layer in layers]
        oracle_v = [layer["v_initial"].clone() for layer in layers]
        oracle_layers = []
        oracle_stream = initial_stream
        oracle_attn_input = initial_attn_input
        for layer_index, layer in enumerate(layers):
            reference, oracle_stream, oracle_attn_input = (
                _distributed_layer_reference(
                layer,
                oracle_attn_input,
                oracle_stream,
                oracle_k[layer_index],
                oracle_v[layer_index],
                cos_sin,
                positions,
            )
            )
            oracle_layers.append(reference)

        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

        for layer in layers:
            for name in ("w13", "s13", "w2", "s2"):
                layer[f"ref_{name}"] = layer[name]
            layer["w13"] = moe_shuffle_weight(
                layer["w13"], is_guinterleave=True, gate_up=True
            ).view(torch.uint8)
            layer["s13"] = moe_shuffle_scale(
                layer["s13"], experts_cnt=64,
                is_guinterleave=True, gate_up=True,
            ).view(torch.uint8)
            layer["w2"] = moe_shuffle_weight(layer["w2"]).view(torch.uint8)
            layer["s2"] = moe_shuffle_scale(
                layer["s2"], experts_cnt=64
            ).view(torch.uint8)

        k1 = build_k1(K1Build(tokens=rows))
        kattn = build_kattn(
            KAttnBuild(tokens=rows, q_len=8)
        )
        k2 = build_k2(K2Build(tokens=rows, npes=NP))
        k3_variants = {
            False: build_k3(
                K3Build(tokens=rows, npes=NP, last=False, shuffled=True)
            ),
            True: build_k3(
                K3Build(tokens=rows, npes=NP, last=True, shuffled=True)
            ),
        }
        layer_v2 = build_layer(
            LayerBuild(
                tokens=rows,
                q_len=8,
                npes=NP,
                next_k1=True,
                last=False,
                shuffled=True,
            )
        )
        layer_v1_last = build_layer(
            LayerBuild(
                tokens=rows,
                q_len=8,
                npes=NP,
                next_k1=False,
                last=True,
                shuffled=True,
            )
        )
        split_k1_scratch = torch.zeros(
            k1_scratch_bytes(k1_scratch_layout()),
            dtype=torch.uint8,
            device=device,
        )
        fused_k1_scratch = torch.zeros_like(split_k1_scratch)
        k2_scratch = torch.zeros(
            k2_scratch_bytes(), dtype=torch.uint8, device=device
        )
        k3_scratch = torch.zeros(
            k3_scratch_bytes(), dtype=torch.uint8, device=device
        )
        fused_scratch = torch.zeros(
            layer_scratch_bytes(), dtype=torch.uint8, device=device
        )
        split_epoch, fused_epoch = _epoch(device), _epoch(device)
        oracle_epoch = _epoch(device)
        oracle_k_cache = torch.empty_like(layers[0]["k_initial"])
        oracle_v_cache = torch.empty_like(layers[0]["v_initial"])
        oracle_k1_scratch = torch.zeros_like(split_k1_scratch)
        stream = torch.cuda.current_stream()

        def empty_qkvg():
            q = torch.empty(
                rows, Q_K, dtype=torch.bfloat16, device=device
            )
            k = torch.empty(
                rows, KV_K, dtype=torch.bfloat16, device=device
            )
            return q, k, torch.empty_like(k), torch.empty_like(q)

        # Path 2: every layer is explicitly split into K1/K-attn/K2/K3.
        split_k_cache = [layer["k_initial"].clone() for layer in layers]
        split_v_cache = [layer["v_initial"].clone() for layer in layers]
        split_layers = []
        split_stream = initial_stream.clone()
        split_attn_input = initial_attn_input.clone()
        torch.cuda.synchronize()
        dist.barrier()
        for layer_index, layer in enumerate(layers):
            outputs = _layer_outputs(device, rows)
            _reset(outputs, split_stream)
            base_stream = split_stream.clone()
            base_attn_input = split_attn_input.clone()
            q, k, v, gate = empty_qkvg()
            print(f"rank {rank} split layer {layer['index']} K1", flush=True)
            k1(
                split_attn_input.data_ptr(), layer["qkvg"].data_ptr(),
                cos_sin.data_ptr(), positions.data_ptr(), layer["slots"].data_ptr(),
                layer["swa_map"].data_ptr(), split_k_cache[layer_index].data_ptr(),
                split_v_cache[layer_index].data_ptr(), q.data_ptr(), k.data_ptr(),
                v.data_ptr(), gate.data_ptr(), split_k1_scratch.data_ptr(),
                split_epoch.data_ptr(), layer["index"], int(layer["rope"]),
                int(layer["sliding"]), 1, 1, stream=stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            print(f"rank {rank} split layer {layer['index']} K-attn", flush=True)
            kattn(
                q.data_ptr(), outputs["o"].data_ptr(),
                split_k_cache[layer_index].data_ptr(),
                split_v_cache[layer_index].data_ptr(), layer["table"].data_ptr(),
                layer["table"].stride(0), prefix.data_ptr(), layer["window"],
                scale_bits, stream=stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            weights = layer["weights"]
            print(f"rank {rank} split layer {layer['index']} K2", flush=True)
            k2(
                outputs["o"].data_ptr(), gate.data_ptr(), outputs["res"].data_ptr(),
                weights["w_o"].data_ptr(), weights["attn_alpha"].data_ptr(),
                weights["attn_beta"].data_ptr(), weights["pre_w"].data_ptr(),
                weights["w_hi"].data_ptr(), weights["w_lo"].data_ptr(),
                weights["bias"].data_ptr(), weights["w_pool"].data_ptr(),
                weights["xn_w"].data_ptr(), outputs["ids"].data_ptr(),
                outputs["scores"].data_ptr(), 0, 0, outputs["xin"].data_ptr(),
                k2_scratch.data_ptr(), split_k2_peers.local,
                split_k2_peers.addresses.data_ptr(), rank, split_epoch.data_ptr(),
                layer["index"], stream=stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            last = layer["index"] == LAYERS[-1]
            print(f"rank {rank} split layer {layer['index']} K3", flush=True)
            k3_variants[last](
                outputs["xin"].data_ptr(), outputs["ids"].data_ptr(),
                outputs["scores"].data_ptr(), layer["local_start"],
                layer["w13"].data_ptr(), layer["s13"].data_ptr(),
                layer["w2"].data_ptr(), layer["s2"].data_ptr(),
                weights["pen_w"].data_ptr(), outputs["res"].data_ptr(),
                weights["pfw"].data_ptr(), weights["ffn_alpha"].data_ptr(),
                weights["ffn_beta"].data_ptr(), weights["npw"].data_ptr(),
                weights["w_unpool"].data_ptr(), outputs["attn_out"].data_ptr(),
                k3_scratch.data_ptr(), split_k3_peers.local,
                split_k3_peers.addresses.data_ptr(), rank, split_epoch.data_ptr(),
                layer["index"], int(last), stream=stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            snapshot = _snapshot(outputs, q, k, v, gate)
            snapshot["base_stream"] = base_stream
            snapshot["base_attn_input"] = base_attn_input
            split_layers.append(snapshot)
            split_stream = outputs["res"].clone()
            split_attn_input = outputs["attn_out"].clone()
            split_epoch[0] += 1

        # Path 3: first K1 stands alone, then V2, V2 and final V1.
        fused_k_cache = [layer["k_initial"].clone() for layer in layers]
        fused_v_cache = [layer["v_initial"].clone() for layer in layers]
        fused_layers = []
        fused_stream = initial_stream.clone()
        fused_attn_input = initial_attn_input.clone()
        q, k, v, gate = empty_qkvg()
        first = layers[0]
        print(f"rank {rank} fused first K1", flush=True)
        k1(
            fused_attn_input.data_ptr(), first["qkvg"].data_ptr(),
            cos_sin.data_ptr(), positions.data_ptr(), first["slots"].data_ptr(),
            first["swa_map"].data_ptr(), fused_k_cache[0].data_ptr(),
            fused_v_cache[0].data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
            gate.data_ptr(), fused_k1_scratch.data_ptr(), fused_epoch.data_ptr(), 1,
            int(first["rope"]), int(first["sliding"]), 1, 1, stream=stream,
        )  # fmt: skip
        torch.cuda.synchronize()
        dist.barrier()

        for layer_index, layer in enumerate(layers):
            outputs = _layer_outputs(device, rows)
            _reset(outputs, fused_stream)
            base_stream = fused_stream.clone()
            base_attn_input = fused_attn_input.clone()
            weights = layer["weights"]
            last = layer_index == len(layers) - 1
            current_q, current_k = q.clone(), k.clone()
            current_v, current_gate = v.clone(), gate.clone()
            if last:
                launch = layer_v1_last
                dummy = outputs["attn_out"].data_ptr()
                next_args = (
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy,
                    dummy, 0, 0, 0, 0,
                )
            else:
                launch = layer_v2
                next_layer = layers[layer_index + 1]
                next_k = torch.empty_like(k)
                next_v = torch.empty_like(v)
                next_args = (
                    next_layer["qkvg"].data_ptr(), cos_sin.data_ptr(),
                    positions.data_ptr(), next_layer["slots"].data_ptr(),
                    next_layer["swa_map"].data_ptr(),
                    fused_k_cache[layer_index + 1].data_ptr(),
                    fused_v_cache[layer_index + 1].data_ptr(),
                    next_k.data_ptr(), next_v.data_ptr(), int(next_layer["rope"]),
                    int(next_layer["sliding"]), 1, 1,
                )
                next_kv = (next_k, next_v)
            print(
                f"rank {rank} fused layer {layer['index']} "
                f"{'V1' if last else 'V2'}",
                flush=True,
            )
            launch(
                q.data_ptr(), outputs["o"].data_ptr(),
                fused_k_cache[layer_index].data_ptr(),
                fused_v_cache[layer_index].data_ptr(), layer["table"].data_ptr(),
                layer["table"].stride(0), prefix.data_ptr(), layer["window"],
                scale_bits, gate.data_ptr(),
                outputs["res"].data_ptr(), weights["w_o"].data_ptr(),
                weights["attn_alpha"].data_ptr(), weights["attn_beta"].data_ptr(),
                weights["pre_w"].data_ptr(), weights["w_hi"].data_ptr(),
                weights["w_lo"].data_ptr(), weights["bias"].data_ptr(),
                weights["w_pool"].data_ptr(), weights["xn_w"].data_ptr(),
                outputs["ids"].data_ptr(), outputs["scores"].data_ptr(),
                outputs["xin"].data_ptr(), layer["local_start"],
                layer["w13"].data_ptr(), layer["s13"].data_ptr(),
                layer["w2"].data_ptr(), layer["s2"].data_ptr(),
                weights["pen_w"].data_ptr(), weights["pfw"].data_ptr(),
                weights["ffn_alpha"].data_ptr(), weights["ffn_beta"].data_ptr(),
                weights["npw"].data_ptr(), weights["w_unpool"].data_ptr(),
                outputs["attn_out"].data_ptr(), *next_args,
                fused_scratch.data_ptr(), fused_peers.local,
                fused_peers.addresses.data_ptr(), rank, fused_epoch.data_ptr(),
                layer["index"], int(last),
                stream=stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            snapshot = _snapshot(
                outputs, current_q, current_k, current_v, current_gate
            )
            snapshot["base_stream"] = base_stream
            snapshot["base_attn_input"] = base_attn_input
            fused_layers.append(snapshot)
            fused_stream = outputs["res"].clone()
            fused_attn_input = outputs["attn_out"].clone()
            fused_epoch[0] += 1
            if not last:
                k, v = next_kv

        for layer_offset, (layer, split, fused) in enumerate(
            zip(layers, split_layers, fused_layers)
        ):
            layer_index = layer_offset + 1
            for path_name, path, k_cache, v_cache in (
                ("split", split, split_k_cache, split_v_cache),
                ("fused", fused, fused_k_cache, fused_v_cache),
            ):
                # Standalone K1 is the arithmetic-order oracle for the fused
                # chained K1 stage.  Its all-width tests independently compare
                # against PyTorch, while this catches integration corruption
                # without depending on a different GEMV reduction order.
                q_ref, k_ref, v_ref, g_ref = empty_qkvg()
                k1(
                    path["base_attn_input"].data_ptr(),
                    layer["qkvg"].data_ptr(),
                    cos_sin.data_ptr(),
                    positions.data_ptr(),
                    layer["slots"].data_ptr(),
                    layer["swa_map"].data_ptr(),
                    oracle_k_cache.data_ptr(),
                    oracle_v_cache.data_ptr(),
                    q_ref.data_ptr(),
                    k_ref.data_ptr(),
                    v_ref.data_ptr(),
                    g_ref.data_ptr(),
                    oracle_k1_scratch.data_ptr(),
                    oracle_epoch.data_ptr(),
                    layer["index"],
                    int(layer["rope"]),
                    int(layer["sliding"]),
                    1,
                    1,
                    stream=stream,
                )
                torch.cuda.synchronize()
                oracle_epoch[0] += 1
                for name, expected, tolerance in (
                    ("q", q_ref, 1 / 32),
                    ("k", k_ref, 1 / 32),
                    ("v", v_ref, 1 / 128),
                    ("g", g_ref, 1 / 128),
                ):
                    k1_close(
                        path[name],
                        expected,
                        tolerance,
                        f"{path_name} layer {layer_index} {name}",
                    )
                # Use the standalone K-attn kernel as the arithmetic-order
                # oracle for this integration check.  Its all-width tests
                # independently compare it with paged PyTorch attention.
                # Unlike comparing the split/fused paths directly, this
                # remains valid after their hidden states have diverged.
                attention_oracle = torch.empty_like(path["o"])
                kattn(
                    path["q"].data_ptr(),
                    attention_oracle.data_ptr(),
                    k_cache[layer_offset].data_ptr(),
                    v_cache[layer_offset].data_ptr(),
                    layer["table"].data_ptr(),
                    layer["table"].stride(0),
                    prefix.data_ptr(),
                    layer["window"],
                    scale_bits,
                    stream=stream,
                )
                torch.cuda.synchronize()
                _assert_close(
                    path["o"],
                    attention_oracle,
                    1 / 32,
                    2e-2,
                    f"rank {rank} {path_name} layer {layer_index} o",
                )
                post_ref = _distributed_post_reference(
                    layer, path["o"], path["g"], path["base_stream"]
                )
                assert torch.equal(path["ids"], post_ref["ids"])
                _assert_close(
                    path["scores"],
                    post_ref["scores"],
                    1 / 64,
                    1e-4,
                    f"rank {rank} {path_name} layer {layer_index} scores",
                )
                _assert_close(
                    path["xin"],
                    post_ref["xin"],
                    1 / 32,
                    2e-2,
                    f"rank {rank} {path_name} layer {layer_index} xin",
                )
                _assert_close(
                    path["res"],
                    post_ref["res"],
                    1 / 32,
                    0.15,
                    f"rank {rank} {path_name} layer {layer_index} res",
                )
                _assert_close(
                    path["attn_out"],
                    post_ref["attn_out"],
                    1 / 32,
                    0.3,
                    f"rank {rank} {path_name} layer {layer_index} attn_out",
                )

                expected_k = layer["k_initial"].clone()
                expected_v = layer["v_initial"].clone()
                write_slots = (
                    layer["swa_map"][layer["slots"]]
                    if layer["sliding"]
                    else layer["slots"]
                )
                expected_k[write_slots] = _fp8(
                    path["k"].reshape(rows, -1, 64)
                )
                expected_v[write_slots] = _fp8(
                    path["v"].reshape(rows, -1, 64)
                )
                torch.testing.assert_close(
                    k_cache[layer_offset].float(),
                    expected_k.float(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    v_cache[layer_offset].float(),
                    expected_v.float(),
                    rtol=0,
                    atol=0,
                )

        final_oracle = oracle_layers[-1]["attn_out"].float()
        for path_name, path_layers in (
            ("split", split_layers),
            ("fused", fused_layers),
        ):
            final = path_layers[-1]["attn_out"].float()
            nrmse = (
                (final - final_oracle).square().mean().sqrt()
                / final_oracle.square().mean().sqrt()
            )
            cosine = torch.nn.functional.cosine_similarity(
                final.flatten(), final_oracle.flatten(), dim=0
            )
            assert nrmse < 0.5, (
                f"{path_name} three-layer NRMSE {nrmse.item():.4f}"
            )
            assert cosine > 0.85, (
                f"{path_name} three-layer cosine {cosine.item():.4f}"
            )

        for epoch_name, epoch in (
            ("split", split_epoch),
            ("fused", fused_epoch),
        ):
            errors = epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
            assert errors == [0, 0, 0], f"{epoch_name} guard errors: {errors}"
        elapsed = time.monotonic() - started
        assert elapsed < HOST_CEILING_S, (
            f"three-layer rank {rank} elapsed {elapsed:.1f}s exceeds "
            f"{HOST_CEILING_S:.0f}s ceiling"
        )
        queue.put((rank, None, elapsed))
    except Exception as error:  # return useful child failures instead of a timeout
        queue.put(
            (
                rank,
                "".join(traceback.format_exception(error)),
                time.monotonic() - started,
            )
        )
    finally:
        for peer in peers:
            peer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_tp4_ep4(rows):
    import queue as queue_module
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 33000 + os.getpid() % 1000
    processes = [
        context.Process(target=_worker, args=(rank, port, queue, rows))
        for rank in range(NP)
    ]
    started = time.monotonic()
    for process in processes:
        process.start()
    results = []
    try:
        for _ in range(NP):
            remaining = max(1.0, WORKER_TIMEOUT_S - (time.monotonic() - started))
            results.append(queue.get(timeout=remaining))
    except queue_module.Empty:
        results.append(
            (-1, f"host timeout after {WORKER_TIMEOUT_S:.0f}s", time.monotonic() - started)
        )
    finally:
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
            if process.exitcode not in (0, None):
                results.append(
                    (-1, f"worker exit code {process.exitcode}", time.monotonic() - started)
                )
    return sorted(results)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and torch.cuda.device_count() >= NP
        and all(
            "gfx1250" in torch.cuda.get_device_properties(device).gcnArchName
            for device in range(NP)
        )
    ),
    reason="needs four idle gfx1250 GPUs",
)
@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_gfx1250_tp4_ep4_three_checkpoint_layers_match_all_paths(rows):
    results = _run_tp4_ep4(rows)
    failures = [
        (rank, error, elapsed)
        for rank, error, elapsed in results
        if error is not None
    ]
    for rank, error, elapsed in failures:
        print(f"rank {rank} failed after {elapsed:.2f}s:\n{error}")
    assert not failures, failures
    assert max(elapsed for _, _, elapsed in results) < HOST_CEILING_S


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)
def test_gfx1250_tp4_three_layer_synthetic_widths_compile():
    """Compile split and fused Target chains at every supported width."""
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.build_layer import LayerBuild, build_layer
    from sgl_amd_mono.muse_spark.gfx1250.config import DRAFT_TOKENS, WIDTHS
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        K1Build,
        KAttnBuild,
        build_k1,
        build_kattn,
    )
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    assert WIDTHS == (8, 16, 32, 64)
    for width in WIDTHS:
        compile_only(build_k1(K1Build(tokens=width)))
        compile_only(
            build_kattn(KAttnBuild(tokens=width, q_len=DRAFT_TOKENS))
        )
        compile_only(build_k2(K2Build(tokens=width, npes=NP)))
        compile_only(
            build_k3(
                K3Build(
                    tokens=width,
                    npes=NP,
                    last=False,
                    shuffled=True,
                )
            )
        )
        compile_only(
            build_k3(
                K3Build(
                    tokens=width,
                    npes=NP,
                    last=True,
                    shuffled=True,
                )
            )
        )
        compile_only(
            build_layer(
                LayerBuild(
                    tokens=width,
                    q_len=DRAFT_TOKENS,
                    npes=NP,
                    next_k1=True,
                    last=False,
                )
            )
        )
        compile_only(
            build_layer(
                LayerBuild(
                    tokens=width,
                    q_len=DRAFT_TOKENS,
                    npes=NP,
                    next_k1=False,
                    last=True,
                )
            )
        )
