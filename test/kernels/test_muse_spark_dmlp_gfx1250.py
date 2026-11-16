"""gfx1250 width-8 DFlash2 D-MLP correctness, including TP4 reduction."""

import os

import pytest
import torch

NP = 4
LAYER = 7


def _epoch(device, seconds=600.0):
    from sgl_amd_mono.common import guard

    value = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    value[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    value[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return value


def _conv(x, delta, base):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        BLOCK,
        CONV_GROUP,
        CONV_GROUPS,
        CONV_TAPS,
    )

    rows = x.shape[0]
    grouped = x.float().reshape(rows, CONV_GROUPS, CONV_GROUP)
    coefficients = (
        base.float().reshape(1, CONV_TAPS, CONV_GROUPS, CONV_GROUP)
        + delta.float().reshape(rows, CONV_TAPS, CONV_GROUPS, 1)
    )
    previous = torch.cat([torch.zeros_like(grouped[:1]), grouped[:-1]], dim=0)
    live = (torch.arange(rows, device=x.device) % BLOCK != 0).view(-1, 1, 1)
    return (
        coefficients[:, 0] * grouped
        + coefficients[:, 1] * torch.where(live, previous, 0)
    ).flatten(-2)


def _prefix_reference(res, attn, shared):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        CONV_GROUPS,
        CONV_TAPS,
        HIDDEN,
        RMS_EPS,
    )

    res2 = (res.float() + attn.float()).to(torch.bfloat16)
    h = (
        res2.float()
        * torch.rsqrt(res2.float().square().mean(-1, keepdim=True) + RMS_EPS)
        * shared["w_post"].float()
    ).to(torch.bfloat16)
    coef = (h.float() @ shared["w_kp"].float().T).reshape(
        res.shape[0], 2, CONV_TAPS, CONV_GROUPS
    )
    hc = _conv(h, coef[:, 0], shared["base"][0]).to(torch.bfloat16)
    return res2, coef, hc


def _rank_partial(hc, w_gu, w_down):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import LOCAL_INTERMEDIATE

    gu = hc.float() @ w_gu.float().T
    gate = gu[:, :LOCAL_INTERMEDIATE].to(torch.bfloat16).float()
    up = gu[:, LOCAL_INTERMEDIATE:].to(torch.bfloat16).float()
    act = (torch.nn.functional.silu(gate).to(torch.bfloat16).float() * up).to(
        torch.bfloat16
    )
    return (act.float() @ w_down.float().T).to(torch.bfloat16)


def _weights(device, rank):
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import (
        COEF_ROWS,
        GATE_UP_ROWS,
        HIDDEN,
        LOCAL_INTERMEDIATE,
    )

    shared_gen = torch.Generator(device=device).manual_seed(1250)
    local_gen = torch.Generator(device=device).manual_seed(12500 + rank)
    bf = torch.bfloat16
    shared = {
        "w_post": (1 + 0.05 * torch.randn(HIDDEN, device=device, generator=shared_gen)).to(bf),
        "w_kp": (0.01 * torch.randn(COEF_ROWS, HIDDEN, device=device, generator=shared_gen)).to(bf),
        "base": (0.03 * torch.randn(2, 2, HIDDEN, device=device, generator=shared_gen)).to(bf),
    }
    shared["base"][:, 0] += 1
    local = {
        "w_gu": (0.015 * torch.randn(GATE_UP_ROWS, HIDDEN, device=device, generator=local_gen)).to(bf),
        "w_down": (0.015 * torch.randn(HIDDEN, LOCAL_INTERMEDIATE, device=device, generator=local_gen)).to(bf),
    }
    return shared, local


def _close(got, expected, what):
    error = (got.float() - expected.float()).abs()
    bad = (error > expected.float().abs() / 16 + 0.04).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(got.float(), expected.float(), dim=-1)
    assert bad < 0.01, f"{what}: {bad:.2%} outside tolerance"
    assert cosine.min().item() > 0.998, f"{what}: cosine {cosine.min().item():.6f}"


def _run_rank(rank, npes, group=None):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import HIDDEN, MAX_TOKENS

    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    generator = torch.Generator(device=device).manual_seed(1251)
    res = torch.randn(MAX_TOKENS, HIDDEN, device=device, generator=generator).to(torch.bfloat16)
    attn = (0.5 * torch.randn(MAX_TOKENS, HIDDEN, device=device, generator=generator)).to(torch.bfloat16)
    shared, local = _weights(device, rank)
    if group is not None:
        # Make the replicated TP inputs explicit; device RNG streams are not an
        # interchange contract across ROCm agents.
        torch.distributed.broadcast(res, 0, group)
        torch.distributed.broadcast(attn, 0, group)
        for value in shared.values():
            torch.distributed.broadcast(value, 0, group)
    res_ref, coef, hc = _prefix_reference(res, attn, shared)
    partials = []
    for source in range(npes):
        _, source_local = _weights(device, source)
        partials.append(_rank_partial(hc, source_local["w_gu"], source_local["w_down"]))
    reduced = partials[0].float()
    for source in range(1, npes):
        reduced = reduced + partials[source].float()
    expected = _conv(reduced, coef[:, 1], shared["base"][1]).to(torch.bfloat16)

    scratch = torch.zeros(D.dm_scratch_bytes(), dtype=torch.uint8, device=device)
    peers = PeerBuffer(D.dm_sym_bytes(npes) + guard.FLAG_BYTES, group, rank, npes, device)
    peers.bytes.zero_()
    torch.cuda.synchronize()
    res_out = torch.empty_like(res)
    out = torch.empty_like(res)
    epoch = _epoch(device)
    launch = D.build_dmlp(D.DMLPBuild(npes=npes))
    try:
        if group is not None:
            torch.distributed.barrier()
        launch(
            res.data_ptr(), attn.data_ptr(), shared["w_post"].data_ptr(),
            shared["w_kp"].data_ptr(), shared["base"].data_ptr(),
            local["w_gu"].data_ptr(), local["w_down"].data_ptr(),
            res_out.data_ptr(), out.data_ptr(), scratch.data_ptr(), peers.local,
            peers.addresses.data_ptr(), rank, epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist() == [0, 0, 0]
        assert torch.equal(res_out, res_ref)
        _close(out, expected, f"TP{npes} output")
    finally:
        peers.close()


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)
def test_gfx1250_dmlp_width8_tp1():
    _run_rank(0, 1)


def _worker(rank, port, queue):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        dist.init_process_group("gloo", rank=rank, world_size=NP)
        _run_rank(rank, NP, dist.group.WORLD)
        queue.put((rank, None))
    except Exception as error:
        queue.put((rank, repr(error)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= NP
        and all("gfx1250" in torch.cuda.get_device_properties(i).gcnArchName for i in range(NP))
    ),
    reason="needs four gfx1250 GPUs",
)
def test_gfx1250_dmlp_width8_tp4_rank_ordered_reduction():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 32000 + os.getpid() % 1000
    processes = [context.Process(target=_worker, args=(rank, port, queue)) for rank in range(NP)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1800) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert not [(rank, error) for rank, error in results if error], results
