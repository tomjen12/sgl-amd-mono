"""Four-rank gfx1250 guard failure propagation."""

import os

import pytest
import torch

NP = 4
FLAGS_OFFSET = 256


def _build_peer_wait():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int32, Int64

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.sync import Mailbox, sreg
    from sgl_amd_mono.muse_spark.gfx1250.device import mailbox_guard

    @flyc.kernel(
        name="gfx1250_guard_peer_failure", known_block_size=[64, 1, 1]
    )
    def kernel(
        epoch: Int64,
        box: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
    ):
        state = {
            "mailbox": Mailbox(
                fx.Int32(7),
                mailbox_guard(
                    epoch,
                    own=sym,
                    peers=peers,
                    rank=rank,
                    npes=NP,
                    flags_offset=FLAGS_OFFSET,
                    stop=False,
                ),
            )
        }
        state["mailbox"].poll([(sreg(box, 0, "missing"), 0, 1)])

    @flyc.jit
    def launch(
        epoch: Int64,
        box: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(epoch, box, sym, peers, rank).launch(
            grid=(1,), block=(64,), stream=stream
        )

    return launch


def _worker(rank, port, queue):
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    device = torch.device("cuda", rank)
    peers = None
    try:
        peers = PeerBuffer(
            FLAGS_OFFSET + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            device,
        )
        peers.bytes.zero_()
        epoch = torch.zeros(
            guard.GUARD_WORDS, dtype=torch.int32, device=device
        )
        seconds = 0.05 if rank == 0 else 2.0
        ticks = guard.deadline_ticks(seconds)
        epoch[guard.GUARD_DEADLINE] = ticks & 0xFFFFFFFF
        epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        box = torch.zeros(2, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        dist.barrier()
        _build_peer_wait()(
            epoch.data_ptr(),
            box.data_ptr(),
            peers.local,
            peers.addresses.data_ptr(),
            rank,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        errors = [int(epoch[index]) for index in guard.ERROR_WORDS]
        flags = (
            peers.bytes[
                FLAGS_OFFSET : FLAGS_OFFSET + NP * 4
            ]
            .view(torch.int32)
            .cpu()
            .tolist()
        )
        queue.put((rank, errors, flags, None))
    except Exception as error:
        queue.put((rank, None, None, repr(error)))
    finally:
        if peers is not None:
            peers.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 33000 + os.getpid() % 1000
    processes = [
        context.Process(target=_worker, args=(rank, port, queue))
        for rank in range(NP)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=300) for _ in range(NP)]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
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
def test_one_rank_timeout_stops_peer_waits():
    results = _run()
    assert all(error is None for _, _, _, error in results), results
    for rank, errors, flags, _ in results:
        if rank == 0:
            assert errors[0] == 2 and errors[1:] == [0, 0], errors
        else:
            assert errors[0] == 0 and errors[1] == 1, errors
        assert flags[0] == 1, flags
