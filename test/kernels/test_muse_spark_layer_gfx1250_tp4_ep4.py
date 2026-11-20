"""Four-process TP4/EP4 correctness for split and fused gfx1250 layers."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYER = 19
CASES = ((300, 0, False, False), (3000, 2047, True, True))


def _worker(rank, port, queue):
    import torch.distributed as dist

    from test_muse_spark_layer_gfx1250 import (
        ROUTED_IDS,
        TABLE_EXPERTS,
        _assert_close,
        _attention_reference,
        _check_outputs,
        _epoch,
        _k2_reference,
        _k3_boundary,
        _local_k3_partial,
        _make_case,
        _native_weight,
        _outputs,
        _reset,
    )
    from test_muse_spark_k1_gfx1250 import POSITIONS, SLOTS, _cos_sin

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
        HEAD_DIM,
        HIDDEN,
        INTER,
        KV_K,
        LOCAL_KV_HEADS,
        DRAFT_TOKENS as MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        QKVG_ROWS,
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
    active_peers = []
    try:
        for case_index, (keys, window, last, shuffled) in enumerate(CASES):
            # All boundary data is replicated.  TP projection and EP expert
            # weights are rank-local, and their BF16 oracle partials are
            # gathered and added strictly in rank order.
            case = _make_case(device, 125400 + case_index, keys)
            weights = case["weights"]
            local_generator = torch.Generator(device=device).manual_seed(
                126000 + 101 * case_index + rank
            )
            weights["w_o"] = (
                torch.randn(
                    HIDDEN, Q_K, device=device, generator=local_generator
                )
                * 0.02
            ).to(torch.bfloat16)
            case["w13"], case["s13"], case["d13"] = _native_weight(
                TABLE_EXPERTS,
                2 * INTER,
                MOE_HIDDEN,
                device,
                local_generator,
            )
            case["w2"], case["s2"], case["d2"] = _native_weight(
                TABLE_EXPERTS,
                MOE_HIDDEN,
                INTER,
                device,
                local_generator,
            )
            if shuffled:
                from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

                case["w13"] = moe_shuffle_weight(
                    case["w13"], is_guinterleave=True, gate_up=True
                ).view(torch.uint8)
                case["s13"] = moe_shuffle_scale(
                    case["s13"], experts_cnt=TABLE_EXPERTS,
                    is_guinterleave=True, gate_up=True,
                ).view(torch.uint8)
                case["w2"] = moe_shuffle_weight(case["w2"]).view(torch.uint8)
                case["s2"] = moe_shuffle_scale(
                    case["s2"], experts_cnt=TABLE_EXPERTS
                ).view(torch.uint8)
            next_w = (
                torch.randn(
                    QKVG_ROWS,
                    HIDDEN,
                    device=device,
                    generator=local_generator,
                )
                * 0.02
            ).to(torch.bfloat16)
            cos_sin = _cos_sin(device)
            positions = torch.randint(
                0,
                POSITIONS,
                (MAX_TOKENS,),
                dtype=torch.int64,
                device=device,
                generator=local_generator,
            )
            slots = torch.randperm(
                SLOTS, device=device, generator=local_generator
            )[:MAX_TOKENS]
            swa_map = torch.arange(SLOTS, dtype=torch.int64, device=device)

            oracle_o = _attention_reference(
                case["q"],
                case["k_cache"],
                case["v_cache"],
                case["table"],
                keys,
                window,
                SOFTMAX_SCALE,
            )

            split = _outputs(device)
            fused = _outputs(device)
            split_scratch2 = torch.zeros(
                k2_scratch_bytes(), dtype=torch.uint8, device=device
            )
            split_scratch3 = torch.zeros(
                k3_scratch_bytes(), dtype=torch.uint8, device=device
            )
            split_scratch1 = torch.zeros(
                k1_scratch_bytes(k1_scratch_layout()),
                dtype=torch.uint8,
                device=device,
            )
            fused_scratch = torch.zeros(
                layer_scratch_bytes(), dtype=torch.uint8, device=device
            )
            split_peers2 = PeerBuffer(
                k2_sym_bytes(NP) + guard.FLAG_BYTES,
                dist.group.WORLD,
                rank,
                NP,
                device,
            )
            split_peers3 = PeerBuffer(
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
            active_peers.extend((split_peers2, split_peers3, fused_peers))
            split_peers2.bytes.zero_()
            split_peers3.bytes.zero_()
            fused_peers.bytes.zero_()
            split_epoch = _epoch(device, 120.0)
            fused_epoch = _epoch(device, 120.0)
            split_q = torch.empty(
                MAX_TOKENS, Q_K, dtype=torch.bfloat16, device=device
            )
            split_k = torch.empty(
                MAX_TOKENS, KV_K, dtype=torch.bfloat16, device=device
            )
            split_v, split_g = torch.empty_like(split_k), torch.empty_like(split_q)
            fused_q, fused_g = case["q"].clone(), case["g"].clone()
            fused_k, fused_v = torch.empty_like(split_k), torch.empty_like(split_v)
            split_k_cache = torch.zeros(
                SLOTS, LOCAL_KV_HEADS, HEAD_DIM, device=device
            ).to(torch.float8_e4m3fn)
            split_v_cache = torch.zeros_like(split_k_cache)
            fused_k_cache = torch.zeros_like(split_k_cache)
            fused_v_cache = torch.zeros_like(split_k_cache)
            kattn = build_kattn(
                KAttnBuild(tokens=MAX_TOKENS, q_len=MAX_TOKENS)
            )
            k2_launch = build_k2(K2Build(tokens=MAX_TOKENS, npes=NP))
            k3_launch = build_k3(
                K3Build(npes=NP, last=last, shuffled=shuffled)
            )
            k1_launch = build_k1(K1Build(tokens=MAX_TOKENS))
            fused_launch = build_layer(
                LayerBuild(
                    tokens=MAX_TOKENS,
                    q_len=MAX_TOKENS,
                    npes=NP,
                    next_k1=True,
                    last=last,
                    shuffled=shuffled,
                )
            )
            stream = torch.cuda.current_stream()
            torch.cuda.synchronize()
            dist.barrier()

            for _ in range(2):
                _reset(split, case["base_res"])
                _reset(fused, case["base_res"])
                fused_q.copy_(case["q"])
                fused_g.copy_(case["g"])
                kattn(
                    case["q"].data_ptr(), split["o"].data_ptr(),
                    case["k_cache"].data_ptr(), case["v_cache"].data_ptr(),
                    case["table"].data_ptr(), case["table"].stride(0),
                    case["prefix"].data_ptr(),
                    int(case["prefix"].dtype == torch.int64), window,
                    case["scale_bits"], stream=stream,
                )  # fmt: skip
                torch.cuda.synchronize()
                dist.barrier()
                k2_launch(
                    split["o"].data_ptr(), case["g"].data_ptr(),
                    split["res"].data_ptr(), weights["w_o"].data_ptr(),
                    weights["attn_alpha"].data_ptr(), weights["attn_beta"].data_ptr(),
                    weights["pre_w"].data_ptr(), weights["w_hi"].data_ptr(),
                    weights["w_lo"].data_ptr(), weights["bias"].data_ptr(),
                    weights["w_pool"].data_ptr(), weights["xn_w"].data_ptr(),
                    split["ids"].data_ptr(), split["scores"].data_ptr(),
                    0, 0, split["xin"].data_ptr(), split_scratch2.data_ptr(),
                    split_peers2.local, split_peers2.addresses.data_ptr(), rank,
                    split_epoch.data_ptr(), LAYER, stream=stream,
                )  # fmt: skip
                k3_launch(
                    split["xin"].data_ptr(), split["ids"].data_ptr(),
                    split["scores"].data_ptr(), 64 * rank,
                    case["w13"].data_ptr(), case["s13"].data_ptr(),
                    case["w2"].data_ptr(), case["s2"].data_ptr(),
                    weights["pen_w"].data_ptr(), split["res"].data_ptr(),
                    weights["pfw"].data_ptr(), weights["ffn_alpha"].data_ptr(),
                    weights["ffn_beta"].data_ptr(), weights["npw"].data_ptr(),
                    weights["w_unpool"].data_ptr(), split["attn_out"].data_ptr(),
                    split_scratch3.data_ptr(), split_peers3.local,
                    split_peers3.addresses.data_ptr(), rank, split_epoch.data_ptr(),
                    LAYER, int(last), stream=stream,
                )  # fmt: skip
                k1_launch(
                    split["attn_out"].data_ptr(), next_w.data_ptr(),
                    cos_sin.data_ptr(), positions.data_ptr(), slots.data_ptr(),
                    swa_map.data_ptr(), split_k_cache.data_ptr(),
                    split_v_cache.data_ptr(), split_q.data_ptr(),
                    split_k.data_ptr(), split_v.data_ptr(), split_g.data_ptr(),
                    split_scratch1.data_ptr(), split_epoch.data_ptr(),
                    LAYER + 1, 1, 0, 1, 1, stream=stream,
                )  # fmt: skip
                torch.cuda.synchronize()
                dist.barrier()
                fused_launch(
                    fused_q.data_ptr(), fused["o"].data_ptr(),
                    case["k_cache"].data_ptr(), case["v_cache"].data_ptr(),
                    case["table"].data_ptr(), case["table"].stride(0),
                    case["prefix"].data_ptr(),
                    int(case["prefix"].dtype == torch.int64), window,
                    case["scale_bits"], fused_g.data_ptr(), fused["res"].data_ptr(),
                    weights["w_o"].data_ptr(), weights["attn_alpha"].data_ptr(),
                    weights["attn_beta"].data_ptr(), weights["pre_w"].data_ptr(),
                    weights["w_hi"].data_ptr(), weights["w_lo"].data_ptr(),
                    weights["bias"].data_ptr(), weights["w_pool"].data_ptr(),
                    weights["xn_w"].data_ptr(), fused["ids"].data_ptr(),
                    fused["scores"].data_ptr(), fused["xin"].data_ptr(), 64 * rank,
                    case["w13"].data_ptr(), case["s13"].data_ptr(),
                    case["w2"].data_ptr(), case["s2"].data_ptr(),
                    weights["pen_w"].data_ptr(), weights["pfw"].data_ptr(),
                    weights["ffn_alpha"].data_ptr(), weights["ffn_beta"].data_ptr(),
                    weights["npw"].data_ptr(), weights["w_unpool"].data_ptr(),
                    fused["attn_out"].data_ptr(),
                    next_w.data_ptr(), cos_sin.data_ptr(),
                    positions.data_ptr(), slots.data_ptr(), swa_map.data_ptr(),
                    fused_k_cache.data_ptr(), fused_v_cache.data_ptr(),
                    fused_k.data_ptr(), fused_v.data_ptr(), 1, 0, 1, 1,
                    fused_scratch.data_ptr(),
                    fused_peers.local, fused_peers.addresses.data_ptr(), rank,
                    fused_epoch.data_ptr(), LAYER, int(last), stream=stream,
                )  # fmt: skip
                split_epoch[0] += 1
                fused_epoch[0] += 1
                torch.cuda.synchronize()
                dist.barrier()

            def distributed_reference(got):
                preliminary = _k2_reference(
                    got["o"], case["g"], case["base_res"], weights
                )
                local_projection = (
                    preliminary[4].float() @ weights["w_o"].float().T
                ).to(torch.bfloat16)
                gathered_projection = [None] * NP
                dist.all_gather_object(
                    gathered_projection, local_projection.cpu()
                )
                k2_reference = _k2_reference(
                    got["o"],
                    case["g"],
                    case["base_res"],
                    weights,
                    partials=[
                        value.to(device) for value in gathered_projection
                    ],
                )
                assert tuple(k2_reference[1][0].tolist()) == ROUTED_IDS
                local_partial = _local_k3_partial(
                    got["xin"],
                    got["ids"],
                    got["scores"],
                    64 * rank,
                    case["d13"],
                    case["d2"],
                    weights["pen_w"],
                )
                gathered_experts = [None] * NP
                dist.all_gather_object(gathered_experts, local_partial.cpu())
                oracle_res, oracle_attn = _k3_boundary(
                    [value.to(device) for value in gathered_experts],
                    weights,
                    k2_reference[0],
                    last,
                )
                return {
                    "o": oracle_o,
                    "ids": k2_reference[1],
                    "scores": k2_reference[2],
                    "xin": k2_reference[3],
                    "res": oracle_res,
                    "attn_out": oracle_attn,
                }

            split_reference = distributed_reference(split)
            fused_reference = distributed_reference(fused)
            _check_outputs(
                split, split_reference, f"rank {rank} split case {case_index}"
            )
            _check_outputs(
                fused, fused_reference, f"rank {rank} fused case {case_index}"
            )
            _assert_close(
                fused["o"],
                split["o"],
                1 / 32,
                2e-2,
                f"rank {rank} fused/split case {case_index} o",
            )
            torch.testing.assert_close(fused_q, split_q, rtol=0, atol=0)
            torch.testing.assert_close(fused_k, split_k, rtol=0, atol=0)
            torch.testing.assert_close(fused_v, split_v, rtol=0, atol=0)
            torch.testing.assert_close(fused_g, split_g, rtol=0, atol=0)
            torch.testing.assert_close(
                fused_k_cache.float(), split_k_cache.float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                fused_v_cache.float(), split_v_cache.float(), rtol=0, atol=0
            )
            assert (
                split_epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0
            ).all()
            assert (
                fused_epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0
            ).all()

            # The replicated inputs and rank-ordered reductions must produce
            # bitwise-identical public layer outputs on every rank.
            local_result = {
                name: value.cpu()
                for name, value in fused.items()
            }
            rank_results = [None] * NP
            dist.all_gather_object(rank_results, local_result)
            for other_rank, other in enumerate(rank_results):
                for name in local_result:
                    assert torch.equal(local_result[name], other[name]), (
                        f"case {case_index} rank {rank}/{other_rank} disagree on {name}"
                    )

            split_peers2.close()
            split_peers3.close()
            fused_peers.close()
            active_peers.clear()
            dist.barrier()
        queue.put((rank, None))
    except Exception as error:  # report child failures instead of only timing out
        queue.put((rank, repr(error)))
    finally:
        for peers in active_peers:
            peers.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_tp4_ep4():
    import queue as queue_module
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 32000 + os.getpid() % 1000
    processes = [
        context.Process(target=_worker, args=(rank, port, queue))
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = []
    try:
        for _ in range(NP):
            results.append(queue.get(timeout=1800))
    except queue_module.Empty:
        results.append((-1, "timed out waiting for worker results"))
    finally:
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
            if process.exitcode not in (0, None):
                results.append((-1, f"worker exit code {process.exitcode}"))
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
    reason="needs four gfx1250 GPUs",
)
def test_gfx1250_fused_layer_tp4_ep4_matches_oracle_and_split():
    failures = [
        (rank, error)
        for rank, error in _run_tp4_ep4()
        if error is not None
    ]
    assert not failures, failures
