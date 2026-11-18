"""Standalone TP1 timing for the gfx1250 Draft D-MLP persistent kernel."""

import argparse
import statistics

import torch

from test_muse_spark_dmlp_gfx1250 import LAYER, _epoch, _weights


def _launch_args(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx1250.draft_config import HIDDEN

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(1251)
    bf16 = torch.bfloat16
    res = torch.randn(rows, HIDDEN, device=device, generator=generator).to(bf16)
    attn = (
        0.5 * torch.randn(rows, HIDDEN, device=device, generator=generator)
    ).to(bf16)
    shared, local = _weights(device, 0)
    outputs = (torch.empty_like(res), torch.empty_like(res))
    scratch = torch.zeros(D.dm_scratch_bytes(), dtype=torch.uint8, device=device)
    peers = PeerBuffer(
        D.dm_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, device
    )
    peers.bytes.zero_()
    epoch = _epoch(device)
    args = (
        res.data_ptr(),
        attn.data_ptr(),
        shared["w_post"].data_ptr(),
        shared["w_kp"].data_ptr(),
        shared["base"].data_ptr(),
        local["w_gu"].data_ptr(),
        local["w_down"].data_ptr(),
        outputs[0].data_ptr(),
        outputs[1].data_ptr(),
        scratch.data_ptr(),
        peers.local,
        peers.addresses.data_ptr(),
        0,
        epoch.data_ptr(),
        LAYER,
    )
    return D, args, epoch, peers


def _time(launch, args, epoch, warmup, iterations):
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        epoch[0] += 1
        launch(*args, stream=stream)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        epoch[0] += 1
        start.record(stream)
        launch(*args, stream=stream)
        end.record(stream)
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--variant",
        choices=("scalar", "wmma", "both"),
        default="both",
    )
    options = parser.parse_args()

    D, args, epoch, peers = _launch_args(options.rows)
    variants = (
        ("scalar", False),
        ("wmma", True),
    )
    try:
        for name, wmma_coef in variants:
            if options.variant not in (name, "both"):
                continue
            launch = D.build_dmlp(
                D.DMLPBuild(
                    tokens=options.rows,
                    npes=1,
                    wmma_coef=wmma_coef,
                )
            )
            values = _time(
                launch,
                args,
                epoch,
                options.warmup,
                options.iterations,
            )
            print(
                f"{name}: median={statistics.median(values):.3f} us "
                f"p10={_percentile(values, 0.1):.3f} us "
                f"p90={_percentile(values, 0.9):.3f} us "
                f"min={min(values):.3f} us"
            )
    finally:
        peers.close()


if __name__ == "__main__":
    main()
