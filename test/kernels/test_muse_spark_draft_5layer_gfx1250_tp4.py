"""Five-layer gfx1250 DFlash2 TP4 integration correctness.

The test uses the checkpoint's exact rank-local tensor shapes, width 8, a
2048-token sliding window inside a longer prefix, and independent cache/state
storage per layer.  Every layer executes D-pre -> D-attn -> D-post -> D-MLP and
is checked against explicit PyTorch math before its hidden/residual pair feeds
the following layer.
"""

import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYERS = 5
PREFIX = 3000
KEYS = PREFIX + 8
SLOTS = 4096
TABLE_STRIDE = 4096
MAX_POS = 8192
HOST_TIMEOUT_S = 3600


def _epoch(device, seconds=1800.0):
    from sgl_amd_mono.common import guard

    value = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    value[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    value[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return value


def _cos_sin(device):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import HEAD_DIM

    angle = torch.arange(MAX_POS, device=device)[:, None] * (
        500000.0
        ** (-torch.arange(HEAD_DIM // 2, device=device) / (HEAD_DIM // 2))
    )[None, :]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def _layer_weights(device, layer, rank):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        GATE_UP_ROWS,
        HEAD_DIM,
        HIDDEN,
        LOCAL_INTERMEDIATE,
        QKV_ROWS,
        Q_K,
    )

    bf = torch.bfloat16
    shared_gen = torch.Generator(device=device).manual_seed(0xDFA000 + layer)
    local_gen = torch.Generator(device=device).manual_seed(
        0xDFA400 + 16 * layer + rank
    )

    def normal(shape, scale, generator):
        return (
            scale * torch.randn(*shape, device=device, generator=generator)
        ).to(bf)

    attn_base = normal((2, 2, HIDDEN), 0.02, shared_gen)
    attn_base[:, 0] += 1
    mlp_base = normal((2, 2, HIDDEN), 0.02, shared_gen)
    mlp_base[:, 0] += 1
    attn = {
        "w_in": (
            1 + normal((HIDDEN,), 0.03, shared_gen).float()
        ).to(bf),
        "w_kp": normal((COEF_ROWS, HIDDEN), 0.002, shared_gen),
        "base": attn_base,
        "w_qkv": normal((QKV_ROWS, HIDDEN), 0.005, local_gen),
        "q_norm": (
            1 + normal((HEAD_DIM,), 0.03, shared_gen).float()
        ).to(bf),
        "k_norm": (
            1 + normal((HEAD_DIM,), 0.03, shared_gen).float()
        ).to(bf),
        "w_o": normal((HIDDEN, Q_K), 0.005, local_gen),
    }
    mlp = {
        "w_post": (
            1 + normal((HIDDEN,), 0.03, shared_gen).float()
        ).to(bf),
        "w_kp": normal((COEF_ROWS, HIDDEN), 0.002, shared_gen),
        "base": mlp_base,
        "w_gu": normal((GATE_UP_ROWS, HIDDEN), 0.005, local_gen),
        "w_down": normal((HIDDEN, LOCAL_INTERMEDIATE), 0.005, local_gen),
    }
    return attn, mlp


def _checkpoint_layer_weights(path, device, layer, rank):
    """Load one layer and apply the same TP4 shards as SGLang's weight loaders."""
    from safetensors import safe_open

    from sgl_amd_mono.muse_spark.gfx1250.draft_config import TP

    def shard(tensor, dim):
        return tensor.chunk(TP, dim=dim)[rank].contiguous()

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())

        def get(suffix):
            candidates = (
                f"layers.{layer}.{suffix}",
                f"model.layers.{layer}.{suffix}",
            )
            for name in candidates:
                if name in keys:
                    return checkpoint.get_tensor(name)
            raise KeyError(
                f"layer {layer}: none of {candidates} exists in {path}"
            )

        try:
            q = shard(get("self_attn.q_proj.weight"), 0)
            k = shard(get("self_attn.k_proj.weight"), 0)
            v = shard(get("self_attn.v_proj.weight"), 0)
            qkv = torch.cat((q, k, v), dim=0)
        except KeyError:
            fused = get("self_attn.qkv_proj.weight")
            q_all, k_all, v_all = fused.split((8192, 1024, 1024), dim=0)
            qkv = torch.cat(
                (shard(q_all, 0), shard(k_all, 0), shard(v_all, 0)), dim=0
            )

        try:
            gate = shard(get("mlp.gate_proj.weight"), 0)
            up = shard(get("mlp.up_proj.weight"), 0)
            gate_up = torch.cat((gate, up), dim=0)
        except KeyError:
            fused = get("mlp.gate_up_proj.weight")
            gate_all, up_all = fused.chunk(2, dim=0)
            gate_up = torch.cat(
                (shard(gate_all, 0), shard(up_all, 0)), dim=0
            )

        attn = {
            "w_in": get("input_layernorm.weight"),
            "w_kp": get("attention_conv.kernel_projection.weight"),
            "base": get("attention_conv.base_kernel"),
            "w_qkv": qkv,
            "q_norm": get("self_attn.q_norm.weight"),
            "k_norm": get("self_attn.k_norm.weight"),
            "w_o": shard(get("self_attn.o_proj.weight"), 1),
        }
        mlp = {
            "w_post": get("post_attention_layernorm.weight"),
            "w_kp": get("mlp_conv.kernel_projection.weight"),
            "base": get("mlp_conv.base_kernel"),
            "w_gu": gate_up,
            "w_down": shard(get("mlp.down_proj.weight"), 1),
        }
    return (
        {name: value.to(device).contiguous() for name, value in attn.items()},
        {name: value.to(device).contiguous() for name, value in mlp.items()},
    )


def _rank_ordered_sum(partial, group):
    """Gather bf16 rank partials through Gloo, then sum in rank order."""
    import torch.distributed as dist

    host = partial.detach().cpu()
    gathered = [torch.empty_like(host) for _ in range(NP)]
    dist.all_gather(gathered, host, group=group)
    total = gathered[0].float()
    for rank in range(1, NP):
        total += gathered[rank].float()
    return total.to(partial.device)


def _close(got, expected, what, rtol=1 / 16, atol=0.04, cosine=0.997):
    gotf, want = got.float(), expected.float()
    bad = ((gotf - want).abs() > want.abs() * rtol + atol).float().mean().item()
    similarity = torch.nn.functional.cosine_similarity(
        gotf.reshape(got.shape[0], -1),
        want.reshape(expected.shape[0], -1),
        dim=-1,
    )
    assert bad < 0.02, f"{what}: {bad:.2%} outside tolerance"
    assert similarity.min().item() > cosine, (
        f"{what}: cosine {similarity.min().item():.6f}"
    )


def _worker(rank, port, queue, checkpoint=None):
    import torch.distributed as dist

    from test_muse_spark_dattn_gfx1250_tp4 import _reference as attn_reference
    from test_muse_spark_dhalf_gfx1250 import (
        _conv as attn_conv,
        _pre_reference,
    )
    from test_muse_spark_dmlp_gfx1250 import (
        _conv as mlp_conv,
        _prefix_reference,
        _rank_partial,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn as DA
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as DM
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        HEAD_DIM,
        HIDDEN,
        LOCAL_KV_HEADS,
        MAX_TOKENS,
        Q_K,
        RESIDENT_BLOCKS,
        WINDOW,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device = torch.device("cuda", rank)
    post_peers = mlp_peers = None
    try:
        post_peers = PeerBuffer(
            DH.do_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        mlp_peers = PeerBuffer(
            DM.dm_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        post_peers.bytes.zero_()
        mlp_peers.bytes.zero_()

        generator = torch.Generator(device=device).manual_seed(0xDFA800)
        hidden = (
            0.3
            * torch.randn(
                MAX_TOKENS, HIDDEN, device=device, generator=generator
            )
        ).to(torch.bfloat16)
        residual = hidden
        first = True
        positions = torch.arange(
            PREFIX, KEYS, dtype=torch.int32, device=device
        )
        cos_sin = _cos_sin(device)
        token_slots = (
            torch.randperm(SLOTS - 1, device=device, generator=generator)[:KEYS]
            + 1
        )
        current_slots = token_slots[PREFIX:].to(torch.int32)
        table = torch.zeros(1, TABLE_STRIDE, dtype=torch.int32, device=device)
        table[0, :KEYS] = token_slots.to(torch.int32)
        request_pool = torch.zeros(1, dtype=torch.int64, device=device)
        prefix_lens = torch.tensor([PREFIX], dtype=torch.int32, device=device)
        epoch = _epoch(device)

        dpre = DH.build_dpre(
            DH.DPreBuild(resident_blocks=RESIDENT_BLOCKS)
        )
        dattn = DA.build_dattn(
            DA.DAttnBuild(resident_blocks=RESIDENT_BLOCKS)
        )
        dpost = DH.build_dpost(
            DH.DPostBuild(npes=NP, resident_blocks=RESIDENT_BLOCKS)
        )
        dmlp = DM.build_dmlp(
            DM.DMLPBuild(npes=NP, resident_blocks=RESIDENT_BLOCKS)
        )
        pre_scratch = torch.zeros(
            DH.dp_scratch_bytes(), dtype=torch.uint8, device=device
        )
        post_scratch = torch.zeros(
            DH.do_scratch_bytes(), dtype=torch.uint8, device=device
        )
        mlp_scratch = torch.zeros(
            DM.dm_scratch_bytes(), dtype=torch.uint8, device=device
        )
        scale = HEAD_DIM**-0.5
        scale_bits = struct.unpack("i", struct.pack("f", scale))[0]

        layer_data = []
        for layer in range(LAYERS):
            attn_weights, mlp_weights = (
                _checkpoint_layer_weights(checkpoint, device, layer, rank)
                if checkpoint
                else _layer_weights(device, layer, rank)
            )
            k_cache = (
                0.7
                * torch.randn(
                    SLOTS,
                    LOCAL_KV_HEADS,
                    HEAD_DIM,
                    device=device,
                    generator=generator,
                )
            ).to(torch.float8_e4m3fn)
            v_cache = (
                0.7
                * torch.randn(
                    SLOTS,
                    LOCAL_KV_HEADS,
                    HEAD_DIM,
                    device=device,
                    generator=generator,
                )
            ).to(torch.float8_e4m3fn)
            layer_data.append((attn_weights, mlp_weights, k_cache, v_cache))

        for layer, (aw, mw, k_cache, v_cache) in enumerate(layer_data):
            expected_pre = _pre_reference(
                hidden, residual, first, aw, cos_sin, positions
            )
            res_pre_ref, q_ref, k_ref, v_ref, coef_ref = expected_pre
            k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
            k_cache_ref[current_slots.long()] = k_ref.reshape(
                MAX_TOKENS, LOCAL_KV_HEADS, HEAD_DIM
            ).to(torch.float8_e4m3fn)
            v_cache_ref[current_slots.long()] = v_ref.reshape(
                MAX_TOKENS, LOCAL_KV_HEADS, HEAD_DIM
            ).to(torch.float8_e4m3fn)

            res_pre = torch.empty_like(hidden)
            q = torch.empty(
                MAX_TOKENS, Q_K, dtype=torch.bfloat16, device=device
            )
            coef = torch.empty(
                MAX_TOKENS, COEF_ROWS, dtype=torch.float32, device=device
            )
            dpre(
                hidden.data_ptr(), residual.data_ptr(), int(first),
                aw["w_in"].data_ptr(), aw["w_kp"].data_ptr(),
                aw["base"].data_ptr(), aw["w_qkv"].data_ptr(),
                aw["q_norm"].data_ptr(), aw["k_norm"].data_ptr(),
                cos_sin.data_ptr(), positions.data_ptr(), 0,
                current_slots.data_ptr(), 0, k_cache.data_ptr(),
                v_cache.data_ptr(), res_pre.data_ptr(), q.data_ptr(),
                coef.data_ptr(), pre_scratch.data_ptr(), epoch.data_ptr(),
                layer, stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            assert torch.equal(res_pre, res_pre_ref)
            _close(q, q_ref, f"layer {layer} D-pre q")
            _close(coef, coef_ref, f"layer {layer} D-pre coef", rtol=1 / 32)
            _close(
                k_cache[current_slots.long()],
                k_ref.reshape(MAX_TOKENS, LOCAL_KV_HEADS, HEAD_DIM),
                f"layer {layer} D-pre K cache",
                rtol=1 / 8,
                atol=0.06,
                cosine=0.994,
            )
            _close(
                v_cache[current_slots.long()],
                v_ref.reshape(MAX_TOKENS, LOCAL_KV_HEADS, HEAD_DIM),
                f"layer {layer} D-pre V cache",
                rtol=1 / 8,
                atol=0.06,
                cosine=0.994,
            )

            o_ref = attn_reference(
                q_ref,
                k_cache_ref,
                v_cache_ref,
                token_slots,
                PREFIX,
                WINDOW,
                scale,
            )
            o = torch.empty_like(q)
            dattn(
                q.data_ptr(), o.data_ptr(), k_cache.data_ptr(),
                v_cache.data_ptr(), table.data_ptr(), TABLE_STRIDE,
                table.numel() * table.element_size(), request_pool.data_ptr(),
                1, prefix_lens.data_ptr(), 0, WINDOW, scale_bits,
                epoch.data_ptr(), stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            _close(o, o_ref, f"layer {layer} D-attn", rtol=1 / 32, atol=0.02)

            local_post = (
                o_ref.float() @ aw["w_o"].float().T
            ).to(torch.bfloat16)
            reduced_post = _rank_ordered_sum(local_post, dist.group.WORLD)
            post_delta = coef_ref.reshape(
                MAX_TOKENS, 2, 2, COEF_ROWS // 4
            )
            attn_out_ref = attn_conv(
                reduced_post, post_delta[:, 1], aw["base"][1]
            ).to(torch.bfloat16)
            attn_out = torch.empty_like(hidden)
            dist.barrier()
            dpost(
                o.data_ptr(), aw["w_o"].data_ptr(), coef.data_ptr(),
                aw["base"].data_ptr(), attn_out.data_ptr(),
                post_scratch.data_ptr(), post_peers.local,
                post_peers.addresses.data_ptr(), rank, epoch.data_ptr(), layer,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            _close(attn_out, attn_out_ref, f"layer {layer} D-post")

            res_out_ref, mlp_coef, hc = _prefix_reference(
                res_pre_ref, attn_out_ref, mw
            )
            local_down = _rank_partial(
                hc, mw["w_gu"], mw["w_down"]
            )
            reduced_down = _rank_ordered_sum(local_down, dist.group.WORLD)
            out_ref = mlp_conv(
                reduced_down, mlp_coef[:, 1], mw["base"][1]
            ).to(torch.bfloat16)
            res_out = torch.empty_like(hidden)
            out = torch.empty_like(hidden)
            dist.barrier()
            dmlp(
                res_pre.data_ptr(), attn_out.data_ptr(),
                mw["w_post"].data_ptr(), mw["w_kp"].data_ptr(),
                mw["base"].data_ptr(), mw["w_gu"].data_ptr(),
                mw["w_down"].data_ptr(), res_out.data_ptr(), out.data_ptr(),
                mlp_scratch.data_ptr(), mlp_peers.local,
                mlp_peers.addresses.data_ptr(), rank, epoch.data_ptr(), layer,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            assert torch.equal(res_out, res_out_ref)
            _close(out, out_ref, f"layer {layer} D-MLP")
            hidden, residual, first = out, res_out, False

        assert epoch[
            guard.ERR_DEADLINE : guard.ERR_FAULT + 1
        ].tolist() == [0, 0, 0]
        queue.put((rank, None))
    except BaseException as error:
        queue.put((rank, repr(error)))
    finally:
        if post_peers is not None:
            post_peers.close()
        if mlp_peers is not None:
            mlp_peers.close()
        if dist.is_initialized():
            dist.destroy_process_group()


GPU_READY = (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= NP
    and all(
        "gfx1250" in torch.cuda.get_device_properties(device).gcnArchName
        for device in range(NP)
    )
)


def _run_processes(checkpoint=None):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 34000 + os.getpid() % 1000
    processes = [
        context.Process(
            target=_worker, args=(rank, port, queue, checkpoint)
        )
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=HOST_TIMEOUT_S) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert not [(rank, error) for rank, error in results if error], results


@pytest.mark.skipif(not GPU_READY, reason="needs four idle gfx1250 GPUs")
def test_gfx1250_dflash2_tp4_five_checkpoint_layers_match_torch():
    _run_processes()


CHECKPOINT = os.environ.get(
    "SGLANG_MUSE_DRAFT_CHECKPOINT",
    "/data/models/Muse-Spark-Open-DFlash2/model.safetensors",
)


@pytest.mark.skipif(
    not os.path.isfile(CHECKPOINT),
    reason="DFlash2 model.safetensors is not installed",
)
@pytest.mark.skipif(not GPU_READY, reason="needs four idle gfx1250 GPUs")
def test_gfx1250_dflash2_real_checkpoint_five_layer_smoke():
    _run_processes(CHECKPOINT)
