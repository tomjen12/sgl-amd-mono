"""Four-rank gfx1250 EP4 K3 test with exact per-rank BF16 partial rounding."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYER = 13


def _worker(rank, port, queue):
    import torch.distributed as dist

    from test_muse_spark_k3_gfx1250 import (
        TABLE_EXPERTS,
        _boundary,
        _epoch,
        _local_partial,
        _native_weight,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import (
        K3Build,
        build_k3,
        k3_scratch_bytes,
        k3_sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        INTER,
        MOE_HIDDEN,
        TOP_K,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device, bf16 = torch.device("cuda", rank), torch.bfloat16
    peers = None
    try:
        # Replicated inputs and boundary weights use the same seed on every rank.
        common = torch.Generator(device=device).manual_seed(125040)
        xin = (torch.randn(8, MOE_HIDDEN, device=device, generator=common) * 0.25).to(bf16)
        ids = torch.empty(8, TOP_K, dtype=torch.int32, device=device)
        for token in range(8):
            ids[token] = torch.tensor(
                [
                    token % 2,
                    64 + (token + 1) % 2,
                    128 + token % 2,
                    192 + (token + 1) % 2,
                    (token + 1) % 2,
                    64 + token % 2,
                    128 + (token + 1) % 2,
                    192 + token % 2,
                ],
                dtype=torch.int32,
                device=device,
            )
        scores = torch.rand(8, TOP_K, device=device, generator=common) * 0.25
        base_stream = torch.randn(8, HIDDEN, device=device, generator=common)
        weights = {
            "pen_w": (
                1 + 0.05 * torch.randn(MOE_HIDDEN, device=device, generator=common)
            ).to(bf16),
            "pfw": (
                1 + 0.05 * torch.randn(HIDDEN, device=device, generator=common)
            ).to(bf16),
            "alpha": torch.rand(HIDDEN, device=device, generator=common) * 0.5
            + 0.5,
            "beta": torch.rand(HIDDEN, device=device, generator=common) * 0.5,
            "npw": (
                1 + 0.05 * torch.randn(HIDDEN, device=device, generator=common)
            ).to(bf16),
            "w_unpool": (
                torch.randn(
                    HIDDEN, MOE_HIDDEN, device=device, generator=common
                )
                * 0.01
            ).to(bf16),
        }

        # Only local IDs 0/1 are represented, preserving local_start indexing
        # without allocating the other 62 enormous expert tables.
        local_gen = torch.Generator(device=device).manual_seed(125100 + rank)
        w13, s13, d13 = _native_weight(
            TABLE_EXPERTS, 2 * INTER, MOE_HIDDEN, device, local_gen
        )
        w2, s2, d2 = _native_weight(
            TABLE_EXPERTS, MOE_HIDDEN, INTER, device, local_gen
        )
        tdm_expert = (
            os.environ.get("SGLANG_AMD_MONO_MUSE_K3_TDM") == "1"
        )
        if tdm_expert:
            from aiter.ops.shuffle import (
                moe_shuffle_scale,
                moe_shuffle_weight,
            )

            w13 = moe_shuffle_weight(
                w13, is_guinterleave=True, gate_up=True
            ).view(torch.uint8)
            s13 = moe_shuffle_scale(
                s13,
                experts_cnt=TABLE_EXPERTS,
                is_guinterleave=True,
                gate_up=True,
            ).view(torch.uint8)
            w2 = moe_shuffle_weight(w2).view(torch.uint8)
            s2 = moe_shuffle_scale(
                s2, experts_cnt=TABLE_EXPERTS
            ).view(torch.uint8)
        partial = _local_partial(
            xin,
            ids,
            scores,
            64 * rank,
            d13,
            d2,
            weights["pen_w"],
        )
        gathered = [None for _ in range(NP)]
        dist.all_gather_object(gathered, partial.cpu())
        # This is the important EP reference rule: each rank rounded its local
        # slot sum to BF16 before these tensors are added in rank order.
        rank_partials = [value.to(device) for value in gathered]
        ref_stream, ref_attn = _boundary(
            rank_partials, weights, base_stream, False
        )

        scratch = torch.zeros(
            k3_scratch_bytes(), dtype=torch.uint8, device=device
        )
        peers = PeerBuffer(
            k3_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        peers.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        epoch = _epoch(device)
        attn = torch.full(
            (8, HIDDEN), float("nan"), dtype=bf16, device=device
        )
        got = base_stream.clone()
        phase_trace = (
            os.environ.get("MUSE_PHASE_TRACE") == "1"
            or os.environ.get("MUSE_K3_PHASE_TRACE") == "1"
        )
        launch = build_k3(
            K3Build(
                npes=NP,
                shuffled=tdm_expert,
                phase_trace=phase_trace,
                tdm_expert=tdm_expert,
                tdm_fast_drain=(
                    os.environ.get(
                        "SGLANG_AMD_MONO_MUSE_K3_TDM_FAST_DRAIN"
                    )
                    == "1"
                ),
            )
        )
        repeated = []
        for _ in range(2):
            got.copy_(base_stream)
            torch.cuda.synchronize()
            dist.barrier()
            launch(
                xin.data_ptr(),
                ids.data_ptr(),
                scores.data_ptr(),
                64 * rank,
                w13.data_ptr(),
                s13.data_ptr(),
                w2.data_ptr(),
                s2.data_ptr(),
                weights["pen_w"].data_ptr(),
                got.data_ptr(),
                weights["pfw"].data_ptr(),
                weights["alpha"].data_ptr(),
                weights["beta"].data_ptr(),
                weights["npw"].data_ptr(),
                weights["w_unpool"].data_ptr(),
                attn.data_ptr(),
                scratch.data_ptr(),
                peers.local,
                peers.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                LAYER,
                0,
                stream=torch.cuda.current_stream(),
            )
            epoch[0] += 1
            torch.cuda.synchronize()
            repeated.append((got.clone(), attn.clone()))
        repeat_stream = torch.nn.functional.cosine_similarity(
            repeated[0][0].float(), repeated[1][0].float(), dim=-1
        ).min()
        repeat_attn = torch.nn.functional.cosine_similarity(
            repeated[0][1].float(), repeated[1][1].float(), dim=-1
        ).min()
        bench_iters = int(os.environ.get("MUSE_K3_BENCH_ITERS", "0"))
        samples = []
        for _ in range(bench_iters):
            got.copy_(base_stream)
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(
                xin.data_ptr(),
                ids.data_ptr(),
                scores.data_ptr(),
                64 * rank,
                w13.data_ptr(),
                s13.data_ptr(),
                w2.data_ptr(),
                s2.data_ptr(),
                weights["pen_w"].data_ptr(),
                got.data_ptr(),
                weights["pfw"].data_ptr(),
                weights["alpha"].data_ptr(),
                weights["beta"].data_ptr(),
                weights["npw"].data_ptr(),
                weights["w_unpool"].data_ptr(),
                attn.data_ptr(),
                scratch.data_ptr(),
                peers.local,
                peers.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                LAYER,
                0,
                stream=torch.cuda.current_stream(),
            )
            end.record()
            end.synchronize()
            epoch[0] += 1
            samples.append(start.elapsed_time(end) * 1000.0)
        if samples:
            print(
                "K3_EP4_BENCH_US="
                + json.dumps(
                    {
                        "rank": rank,
                        "mean": sum(samples) / len(samples),
                        "min": min(samples),
                        "max": max(samples),
                        "samples": samples,
                    }
                ),
                flush=True,
            )
        if phase_trace:
            from sgl_amd_mono.muse_spark.gfx1250.build_k3 import _scratch_layout

            offset, _ = _scratch_layout()["k3_phase_ticks"]
            ticks = (
                scratch[offset : offset + 8 * 7 * 8]
                .view(torch.int64)
                .view(8, 7)
                .cpu()
            )
            print(
                "K3_EP4_PHASE_TICKS="
                + json.dumps(
                    {
                        "rank": rank,
                        "deltas": (ticks[:, 1:] - ticks[:, :-1]).tolist(),
                    }
                ),
                flush=True,
            )
        stream_cosine = torch.nn.functional.cosine_similarity(
            got.float(), ref_stream.float(), dim=-1
        ).min()
        attn_cosine = torch.nn.functional.cosine_similarity(
            attn.float(), ref_attn.float(), dim=-1
        ).min()
        stream_ok = stream_cosine > 0.997 and repeat_stream > 0.99999
        attn_ok = attn_cosine > 0.997 and repeat_attn > 0.99999
        from sgl_amd_mono.muse_spark.gfx1250.build_k3 import _scratch_layout

        offset, _ = _scratch_layout()["k3_fbuf"]
        got_fbuf = (
            scratch[offset : offset + 8 * MOE_HIDDEN * 2]
            .view(bf16)
            .view(8, MOE_HIDDEN)
        )
        ref_fbuf = torch.zeros_like(rank_partials[0].float())
        for value in rank_partials:
            ref_fbuf += value.float()
        ref_fbuf = ref_fbuf.to(bf16)
        fbuf_cosine = torch.nn.functional.cosine_similarity(
            got_fbuf.float(), ref_fbuf.float(), dim=-1
        ).min()
        errors = epoch[
            guard.ERR_DEADLINE : guard.ERR_FAULT + 1
        ].tolist()
        queue.put(
            (
                rank,
                stream_ok,
                attn_ok,
                errors,
                float(fbuf_cosine),
                float(stream_cosine),
                float(attn_cosine),
                float(repeat_stream),
                float(repeat_attn),
            )
        )
    except Exception as exc:
        queue.put(
            (rank, False, False, [repr(exc)], 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


def _run():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = 29500 + os.getpid() % 2000
    processes = [
        ctx.Process(target=_worker, args=(rank, port, queue))
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1200) for _ in range(NP)]
    for process in processes:
        process.join(timeout=120)
    return sorted(results)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= NP
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs four gfx1250 GPUs",
)
def test_k3_ep4_models_rank_partial_rounding():
    for (
        rank,
        stream_ok,
        attn_ok,
        errors,
        fbuf,
        stream_cosine,
        attn_cosine,
        repeat_stream,
        repeat_attn,
    ) in _run():
        assert stream_ok and attn_ok, (
            f"rank {rank}: output mismatch; fbuf cosine={fbuf:.6f}, "
            f"stream cosine={stream_cosine:.6f}, "
            f"attn cosine={attn_cosine:.6f}, "
            f"repeat stream={repeat_stream:.6f}, "
            f"repeat attn={repeat_attn:.6f}"
        )
        assert errors == [0, 0, 0], f"rank {rank}: mailbox errors {errors}"


if __name__ == "__main__":
    print(_run())
