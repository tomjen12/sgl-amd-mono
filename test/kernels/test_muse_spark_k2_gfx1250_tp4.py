"""Executable four-process TP4 test for the gfx1250 K2 peer boundary."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYER = 11


def _worker(rank, port, queue):
    import torch.distributed as dist

    from test_muse_spark_k2_gfx1250 import (
        _assert_close,
        _epoch,
        _inputs,
        _reference,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import (
        K2Build,
        build_k2,
        scratch_bytes,
        sym_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        HIDDEN,
        DRAFT_TOKENS as MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        RMS_EPS,
        TOP_K,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device = torch.device("cuda", rank)
    peers = None
    try:
        o, g, stream, w = _inputs(device, seed=1254)

        # Generate each rank's local projection in rank order on every process.
        # The explicit bf16 conversion is the all-reduce contract: each per-rank
        # GEMM partial rounds before the rank-ordered fp32 sum.
        of = o.float().reshape(MAX_TOKENS, -1, HEAD_DIM)
        og = (
            of
            * torch.rsqrt(of.square().mean(-1, keepdim=True) + RMS_EPS)
            * torch.sigmoid(g.float().reshape(MAX_TOKENS, -1, HEAD_DIM))
        ).to(torch.bfloat16).reshape(MAX_TOKENS, Q_K)
        partials, local_w_o = [], None
        for source_rank in range(NP):
            generator = torch.Generator(device=device).manual_seed(8000 + source_rank)
            candidate = (
                torch.randn(
                    HIDDEN, Q_K, device=device, generator=generator
                )
                * 0.02
            ).to(torch.bfloat16)
            partials.append(
                (og.float() @ candidate.float().T).to(torch.bfloat16)
            )
            if source_rank == rank:
                local_w_o = candidate
        w["w_o"] = local_w_o
        expected = _reference(o, g, stream, w, partials=partials)

        ids = torch.full(
            (MAX_TOKENS, TOP_K), -1, dtype=torch.int32, device=device
        )
        scores = torch.full(
            (MAX_TOKENS, TOP_K), float("nan"), device=device
        )
        xin = torch.full(
            (MAX_TOKENS, MOE_HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        scratch = torch.zeros(scratch_bytes(), dtype=torch.uint8, device=device)
        peers = PeerBuffer(
            sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        peers.bytes.zero_()
        epoch = _epoch(device, 120.0)
        launch = build_k2(K2Build(tokens=MAX_TOKENS, npes=NP))
        got = stream.clone()
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(2):
            got.copy_(stream)
            dist.barrier()
            launch(
                o.data_ptr(), g.data_ptr(), got.data_ptr(), w["w_o"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
                w["w_hi"].data_ptr(), w["w_lo"].data_ptr(), w["bias"].data_ptr(),
                w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), ids.data_ptr(),
                scores.data_ptr(), 0, 0, xin.data_ptr(), scratch.data_ptr(), peers.local,
                peers.addresses.data_ptr(), rank, epoch.data_ptr(), LAYER,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
            torch.cuda.synchronize()

        errors = epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        assert errors == [0, 0, 0], f"mailbox errors {errors}"
        _assert_close(got, expected[0], 1 / 64, 1e-2, "TP4 residual")
        assert torch.equal(ids, expected[1])
        _assert_close(scores, expected[2], 1 / 64, 1e-4, "TP4 scores")
        _assert_close(xin, expected[3], 1 / 32, 2e-2, "TP4 expert input")
        queue.put((rank, None))
    except Exception as error:  # report child failures instead of timing out
        queue.put((rank, repr(error)))
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


def _run_tp4():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 31000 + os.getpid() % 1000
    processes = [
        context.Process(target=_worker, args=(rank, port, queue))
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1200) for _ in range(NP)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    return sorted(results)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= NP
        and all(
            "gfx1250" in torch.cuda.get_device_properties(device).gcnArchName
            for device in range(NP)
        )
    ),
    reason="needs four gfx1250 GPUs",
)
def test_gfx1250_k2_tp4_rank_ordered_bf16_partials():
    failures = [(rank, error) for rank, error in _run_tp4() if error is not None]
    assert not failures, failures
