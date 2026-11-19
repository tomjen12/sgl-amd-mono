"""gfx1250 TP4 variable-width K-attn against exact paged-cache attention."""

import json
import os
import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

PAGES = 2048
MAX_BLOCKS = 1024


def _reference(q, k_cache, v_cache, table, keys, q_len, window, scale):
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        LOCAL_Q_HEADS,
        Q_K,
    )

    out = torch.zeros_like(q, dtype=torch.float32)
    prefix = keys - q_len
    gqa = LOCAL_Q_HEADS // LOCAL_KV_HEADS
    key_index = torch.arange(keys, device=q.device)
    for request in range(q.shape[0] // q_len):
        slots = (
            table[request, : (keys + 15) // 16].long()[:, None] * 16
            + torch.arange(16, device=q.device)[None, :]
        ).reshape(-1)[:keys]
        k = k_cache[slots].float()
        v = v_cache[slots].float()
        for query_token in range(q_len):
            token = request * q_len + query_token
            end = prefix + query_token + 1
            allowed = key_index < end
            if window:
                allowed &= key_index >= max(0, end - window)
            for head in range(LOCAL_Q_HEADS):
                kv_head = head // gqa
                query = q[token, head * HEAD_DIM : (head + 1) * HEAD_DIM].float()
                score = scale * (query @ k[:, kv_head].T)
                score = torch.where(
                    allowed, score, torch.full_like(score, float("-inf"))
                )
                probability = torch.softmax(score, dim=-1).to(torch.bfloat16).float()
                out[token, head * HEAD_DIM : (head + 1) * HEAD_DIM] = (
                    probability @ v[:, kv_head]
                )
    return out


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "keys,window,zero_q",
    [(300, 0, False), (3000, 2047, False), (8200, 0, True)],
)
def test_gfx1250_kattn_matches_paged_attention(width, keys, window, zero_q):
    if zero_q and width != 8:
        pytest.skip("the 8K uniform-attention regression targets width 8")
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        DRAFT_TOKENS,
        Q_K,
        SOFTMAX_SCALE,
    )
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        KAttnBuild,
        build_tiled_kattn,
        tiled_kattn_scratch_bytes,
    )

    torch.manual_seed(keys + window)
    device = torch.device("cuda")
    q = (torch.randn(width, Q_K, device=device) * 5.47).to(torch.bfloat16)
    if zero_q:
        q.zero_()
    k_cache = (
        torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=device) * 2
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    pages = (keys + 15) // 16
    requests = width // DRAFT_TOKENS
    table = torch.zeros(requests, MAX_BLOCKS, dtype=torch.int32, device=device)
    table[:, :pages] = (
        torch.randperm(PAGES - 1, device=device)[: requests * pages]
        .reshape(requests, pages)
        .to(torch.int32)
        + 1
    )
    prefix_lens = torch.full(
        (requests,), keys - DRAFT_TOKENS, dtype=torch.int32, device=device
    )
    out = torch.full_like(q, float("nan"))
    scale_bits = struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0]

    scratch = torch.zeros(
        tiled_kattn_scratch_bytes(), dtype=torch.uint8, device=device
    )
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    phase_trace = os.environ.get("MUSE_PHASE_TRACE") == "1"
    splits = int(os.environ.get("MUSE_KATTN_TEST_SPLITS", "6"))
    fp8_wmma = os.environ.get("MUSE_KATTN_FP8_WMMA") == "1"
    register_probability = (
        os.environ.get("MUSE_KATTN_REGISTER_PROBABILITY", "1") == "1"
    )
    stage_q = (
        os.environ.get("MUSE_KATTN_STAGE_Q", "1") == "1"
        and not fp8_wmma
        and register_probability
    )
    key_block = int(os.environ.get("MUSE_KATTN_KEY_BLOCK", "32"))
    load_bits = int(os.environ.get("MUSE_KATTN_LOAD_BITS", "64"))
    launch = build_tiled_kattn(
        KAttnBuild(
            tokens=width,
            q_len=DRAFT_TOKENS,
            phase_trace=phase_trace,
            splits=splits,
            fp8_wmma=fp8_wmma,
            register_probability=register_probability,
            stage_q=stage_q,
            key_block=key_block,
            load_bits=load_bits,
        )
    )

    def invoke(candidate, iteration):
        candidate(
            q.data_ptr(),
            out.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            table.data_ptr(),
            table.stride(0),
            prefix_lens.data_ptr(),
            window,
            scale_bits,
            scratch.data_ptr(),
            epoch.data_ptr(),
            iteration,
            stream=torch.cuda.current_stream(),
        )

    invoke(launch, 0)
    torch.cuda.synchronize()
    bench_iters = int(os.environ.get("MUSE_KATTN_BENCH_ITERS", "0"))
    bench_warmup = int(os.environ.get("MUSE_KATTN_BENCH_WARMUP", "0"))
    ab_rounds = int(os.environ.get("MUSE_KATTN_STAGE_Q_AB_ROUNDS", "0"))
    if ab_rounds:
        candidates = [
            (
                enabled,
                build_tiled_kattn(
                    KAttnBuild(
                        tokens=width,
                        q_len=DRAFT_TOKENS,
                        phase_trace=False,
                        splits=splits,
                        fp8_wmma=False,
                        register_probability=True,
                        stage_q=enabled,
                    )
                ),
            )
            for enabled in (False, True)
        ]
        for _, candidate in candidates:
            invoke(candidate, 1)
        torch.cuda.synchronize()
        for trial in range(ab_rounds):
            for enabled, candidate in candidates:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for iteration in range(bench_iters):
                    invoke(candidate, trial * bench_iters + iteration + 2)
                stop.record()
                stop.synchronize()
                print(
                    f"KATTN_STAGE_Q_AB_US="
                    f"{start.elapsed_time(stop) * 1000 / bench_iters:.3f} "
                    f"trial={trial + 1} stage_q={int(enabled)} "
                    f"keys={keys} window={window}",
                    flush=True,
                )
    if bench_iters and not ab_rounds:
        for iteration in range(bench_warmup):
            invoke(launch, iteration + 1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for iteration in range(bench_iters):
            invoke(launch, bench_warmup + iteration + 1)
        stop.record()
        stop.synchronize()
        print(
            f"KATTN_BENCH_US={start.elapsed_time(stop) * 1000 / bench_iters:.3f} "
            f"splits={splits} keys={keys} window={window} fp8={int(fp8_wmma)} "
            f"register_probability={int(register_probability)} "
            f"stage_q={int(stage_q)} bn={key_block} load_bits={load_bits}",
            flush=True,
        )
    if phase_trace:
        from sgl_amd_mono.muse_spark.gfx1250.layer import _tiled_kattn_layout
        from sgl_amd_mono.muse_spark.gfx1250.layout import (
            FUSED_SPLITS,
            fused_tasks,
        )

        offset, _ = _tiled_kattn_layout()["attn_phase_ticks"]
        tasks = fused_tasks(width, splits)
        tiles = tasks // splits
        task_ids = [
            tile * FUSED_SPLITS + split
            for tile in range(tiles)
            for split in range(splits)
        ]
        max_task = task_ids[-1] + 1
        words = (
            scratch[offset : offset + max_task * 8 * 8]
            .view(torch.int64)
            .view(max_task, 8)[task_ids]
        )
        ticks = words[:, :6]
        deltas = ticks[:, 1:] - ticks[:, :-1]
        split_zero = deltas[::splits]
        assert torch.all(ticks[:, 1:] >= ticks[:, :-1])
        assert torch.all(ticks[:, -1] > ticks[:, 0])
        print(
            "KATTN_PHASE_TICKS="
            + json.dumps(
                {
                    "width": width,
                    "keys": keys,
                    "window": window,
                    "points": [
                        "start",
                        "online_qk_softmax_pv",
                        "global_stats",
                        "reserved",
                        "partial_publish",
                        "output",
                    ],
                    "delta_means": (
                        deltas.float().mean(dim=0).cpu().tolist()
                    ),
                    "split0_delta_means": (
                        split_zero.float().mean(dim=0).cpu().tolist()
                    ),
                    "split0_span_mean": split_zero.sum(dim=1).float().mean().item(),
                    "split0_span_max": split_zero.sum(dim=1).max().item(),
                    "qk_softmax_mean": words[:, 6].float().mean().item(),
                    "pv_mean": words[:, 7].float().mean().item(),
                }
            ),
            flush=True,
        )

    reference = _reference(
        q, k_cache, v_cache, table, keys, DRAFT_TOKENS, window, SOFTMAX_SCALE
    )
    got = out.float()
    assert not got.isnan().any(), "NaN in K-attn output"
    error = (got - reference).abs()
    tolerance = reference.abs() * (1 / 32) + 2e-2
    bad = (error > tolerance).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(width, -1, HEAD_DIM),
        reference.reshape(width, -1, HEAD_DIM),
        dim=-1,
    )
    print(
        f"KATTN_ACCURACY max_error={error.max().item():.6f} "
        f"mean_error={error.mean().item():.6f} "
        f"min_cosine={cosine.min().item():.9f} fp8={int(fp8_wmma)}",
        flush=True,
    )
    assert bad == 0.0, (
        f"{bad:.2%} outputs exceed tolerance; max error {error.max().item():.4f}"
    )
    assert cosine.min().item() > 0.999, f"least head cosine {cosine.min().item():.6f}"
