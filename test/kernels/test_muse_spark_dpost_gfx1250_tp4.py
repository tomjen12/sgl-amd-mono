"""Four-process rank-ordered TP4 test for the gfx1250 D-post monokernel."""

import os
import socket
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 4
LAYER = 6


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _worker(rank, port, queue):
    import torch.distributed as dist

    from test_muse_spark_dhalf_gfx1250 import (
        _close,
        _epoch,
        _post_reference,
        _weights,
    )

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as D
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        MAX_TOKENS,
        Q_K,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device = torch.device("cuda", rank)
    peers = None
    try:
        shared = _weights(device, 1250)
        outputs, weights = [], []
        for source in range(NP):
            generator = torch.Generator(device=device).manual_seed(1260 + source)
            outputs.append(
                (0.5 * torch.randn(MAX_TOKENS, Q_K, device=device, generator=generator)).to(torch.bfloat16)
            )
            local = _weights(device, 1300 + source)
            local["base"] = shared["base"]
            weights.append(local)
        generator = torch.Generator(device=device).manual_seed(1270)
        coef = 0.1 * torch.randn(MAX_TOKENS, COEF_ROWS, device=device, generator=generator)
        expected = _post_reference(outputs, weights, coef)
        got = torch.empty_like(expected)
        scratch = torch.zeros(D.do_scratch_bytes(), dtype=torch.uint8, device=device)
        peers = PeerBuffer(
            D.do_sym_bytes(NP) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        peers.bytes.zero_()
        epoch = _epoch(device)
        launch = D.build_dpost(D.DPostBuild(npes=NP))
        torch.cuda.synchronize()
        dist.barrier()
        launch(
            outputs[rank].data_ptr(), weights[rank]["w_o"].data_ptr(),
            coef.data_ptr(), shared["base"].data_ptr(), got.data_ptr(),
            scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), rank,
            epoch.data_ptr(), LAYER, stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist() == [0, 0, 0]
        _close(got, expected, "TP4 D-post")
        queue.put((rank, None))
    except Exception as error:
        queue.put((rank, repr(error)))
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


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
def test_gfx1250_dpost_width8_tp4_rank_ordered_reduction():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, port, queue))
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1800) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert not [(rank, error) for rank, error in results if error], results
