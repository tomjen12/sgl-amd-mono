"""Active4/logical8 whole-layer correctness on four gfx1250 GPUs.

For widths 8 and 16, compare the standalone K-attn -> K2 -> K3 -> next K1
chain with the fused layer launch.  Both Layer V2 (with next K1) and Layer V1
(without it) are covered.  TP8-local tensor shapes and Top-8 routing remain
unchanged; only four physical peers participate, and logical EP ranks 4..7
contribute zero expert output.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tp8_layer_contract import (  # noqa: E402
    ACTIVE_NPES,
    LOGICAL_NPES,
    LOCAL_EXPERTS,
    WIDTHS,
    active_expert_ranges,
    deterministic_router_bias,
    deterministic_routing,
    validate_case_result,
)

LAYER = 9


def test_active4_logical8_routing_contract_cpu():
    ids, scores = deterministic_routing()
    assert tuple(expert // LOCAL_EXPERTS for expert in ids) == tuple(range(8))
    assert sum(scores) == 1.0
    assert active_expert_ranges(0) == ((0, 32), (128, 160))
    assert active_expert_ranges(3) == ((96, 128), (224, 256))
    bias = deterministic_router_bias()
    assert tuple(sorted(range(len(bias)), key=bias.__getitem__, reverse=True)[:8]) == ids


def _buffers(T, rows, q_len, rank, dev, group=None):
    """The single-GPU blueprint's buffers with a four-rank PeerBuffer."""
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        KV_K,
        MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import (
        k1_scratch_layout,
        kattn_scratch_layout,
        scratch_bytes,
    )
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import k3_scratch_layout
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    B = object.__new__(T.Buffers)
    bf, nan = torch.bfloat16, float("nan")
    B.q = torch.full((MAX_TOKENS, Q_K), nan, device=dev, dtype=bf)
    B.k = torch.full((MAX_TOKENS, KV_K), nan, device=dev, dtype=bf)
    B.v = torch.full_like(B.k, nan)
    B.g = torch.full_like(B.q, nan)
    B.o = torch.full_like(B.q, nan)
    B.ids = torch.full(
        (MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev
    )
    B.scores = torch.full((MAX_TOKENS, TOP_K), nan, device=dev)
    B.xin = torch.full(
        (MAX_TOKENS, MOE_HIDDEN), nan, device=dev, dtype=bf
    )
    B.attn_in = torch.full(
        (MAX_TOKENS, HIDDEN), nan, device=dev, dtype=bf
    )
    B.res = torch.zeros(rows, HIDDEN, device=dev)
    B.scratch_k1 = torch.zeros(
        scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=dev
    )
    B.scratch_a = torch.zeros(
        scratch_bytes(kattn_scratch_layout(rows, q_len)),
        dtype=torch.uint8,
        device=dev,
    )
    B.scratch_k2 = torch.zeros(
        scratch_bytes(k2_scratch_layout()), dtype=torch.uint8, device=dev
    )
    B.scratch_k3 = torch.zeros(
        scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
    )
    B.peers = PeerBuffer(
        sym_bytes(LOGICAL_NPES) + guard.FLAG_BYTES,
        dist.group.WORLD if group is None else group,
        rank,
        ACTIVE_NPES,
        dev,
        logical_npes=LOGICAL_NPES,
    )
    B.peers.bytes.zero_()
    for shadow in B.peers.shadow_bytes:
        shadow.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    return B


def _kernels(rows, q_len):
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.layer import (
        K1Build,
        KAttnBuild,
        LayerBuild,
        build_k1,
        build_kattn,
        build_layer,
    )

    # Match Runner: split K2/K3 do not hand MXFP4 activation through scratch;
    # the merged Layer uses its production XQ_PRE path.
    split = dict(tokens=rows, npes=LOGICAL_NPES)
    return {
        "k1": build_k1(K1Build(tokens=rows)),
        "ka": build_kattn(KAttnBuild(tokens=rows, q_len=q_len)),
        "k2": build_k2(K2Build(**split, pshard=True)),
        "k3": build_k3(K3Build(**split)),
        "v1": build_layer(
            LayerBuild(
                tokens=rows,
                q_len=q_len,
                npes=LOGICAL_NPES,
                next_k1=False,
                xq_pre=True,
                pshard=True,
            )
        ),
        "v2": build_layer(
            LayerBuild(
                tokens=rows,
                q_len=q_len,
                npes=LOGICAL_NPES,
                next_k1=True,
                xq_pre=True,
                pshard=True,
            )
        ),
    }


def _k3_args(T, layer, buffers, rank):
    args = list(T._k3_args(layer, buffers))
    args[0] = rank * LOCAL_EXPERTS
    return tuple(args)


def _k2_args(T, layer, buffers, rank):
    from sgl_amd_mono.muse_spark.gfx1250.config import MOE_HIDDEN

    rows_per_rank = MOE_HIDDEN // LOGICAL_NPES
    buffers.pool_w = layer.w2k["w_pool"][
        rank * rows_per_rank : (rank + 1) * rows_per_rank
    ].contiguous()
    args = list(T._k2_args(layer, buffers))
    args[9] = buffers.pool_w.data_ptr()
    return tuple(args)


def _split(T, kernels, layer, buffers, epoch, rank, next_k1, stream):
    kernels["ka"](
        *T._attn_args(layer, buffers),
        epoch.data_ptr(),
        LAYER,
        stream=stream,
    )
    kernels["k2"](
        buffers.o.data_ptr(),
        *_k2_args(T, layer, buffers, rank),
        buffers.peers.local,
        buffers.peers.addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        LAYER,
        stream=stream,
    )
    a3 = _k3_args(T, layer, buffers, rank)
    kernels["k3"](
        buffers.xin.data_ptr(),
        buffers.ids.data_ptr(),
        buffers.scores.data_ptr(),
        a3[0],
        *a3[1:5],
        a3[5],
        buffers.res.data_ptr(),
        *a3[6:],
        buffers.peers.local,
        buffers.peers.addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        LAYER,
        0,
        stream=stream,
    )
    if next_k1:
        T._k1(
            kernels["k1"],
            layer,
            buffers,
            buffers.attn_in,
            layer.w_qkvg1,
            layer.k_cache1,
            layer.v_cache1,
            epoch,
            LAYER + 1,
            1,
            stream,
        )


def _fused(T, kernels, layer, buffers, epoch, rank, next_k1, stream):
    kernels["v2" if next_k1 else "v1"](
        *T._attn_args(layer, buffers),
        *_k2_args(T, layer, buffers, rank),
        *_k3_args(T, layer, buffers, rank),
        *T._k1_next_args(layer, buffers),
        buffers.peers.local,
        buffers.peers.addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        LAYER,
        0,
        stream=stream,
    )


def _snapshot(T, layer, buffers, epoch, rows, next_k1):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import sym_bytes

    current_rows = layer.slots
    out = {
        "residual": buffers.res.clone().cpu(),
        "ids": buffers.ids[:rows].clone().cpu(),
        "scores": buffers.scores[:rows].clone().cpu(),
        "xin": buffers.xin[:rows].clone().cpu(),
        "attention_input": buffers.attn_in[:rows].clone().cpu(),
        "current K cache": layer.k_cache[current_rows].clone().cpu(),
        "current V cache": layer.v_cache[current_rows].clone().cpu(),
        "guard": epoch[list(guard.ERROR_WORDS)].clone().cpu(),
        "peer guard": buffers.peers.bytes[
            sym_bytes(LOGICAL_NPES) : sym_bytes(LOGICAL_NPES) + guard.FLAG_BYTES
        ]
        .clone()
        .cpu(),
    }
    if next_k1:
        next_rows = layer.swa_map[layer.slots]
        out.update(
            {
                "next Q": buffers.q[:rows].clone().cpu(),
                "next K": buffers.k[:rows].clone().cpu(),
                "next V": buffers.v[:rows].clone().cpu(),
                "next gate": buffers.g[:rows].clone().cpu(),
                "next K cache": layer.k_cache1[next_rows].clone().cpu(),
                "next V cache": layer.v_cache1[next_rows].clone().cpu(),
            }
        )
    return out


def _run_path(T, kernels, layer, rows, rank, next_k1, fused):
    import torch.distributed as dist

    dev = layer.dev
    buffers = _buffers(T, rows, layer.ql, rank, dev)
    epoch = T._epoch(dev, deadline_s=60.0)
    stream = torch.cuda.current_stream()
    layer.k_cache1.zero_()
    layer.v_cache1.zero_()
    try:
        for _ in range(2):
            buffers.res.copy_(layer.res0)
            T._k1(
                kernels["k1"],
                layer,
                buffers,
                layer.x0,
                layer.w_qkvg,
                layer.k_cache,
                layer.v_cache,
                epoch,
                LAYER,
                0,
                stream,
            )
            from tp8_shadow_protocol import seed_missing_rank_mailboxes

            seed_missing_rank_mailboxes(
                buffers.peers.bytes,
                rows=rows,
                step=int(epoch[0].item()),
                layer=LAYER,
                active_npes=ACTIVE_NPES,
            )
            if fused:
                _fused(
                    T, kernels, layer, buffers, epoch, rank, next_k1, stream
                )
            else:
                _split(
                    T, kernels, layer, buffers, epoch, rank, next_k1, stream
                )
            epoch[0] += 1
        torch.cuda.synchronize()
        dist.barrier()
        return _snapshot(T, layer, buffers, epoch, rows, next_k1)
    finally:
        buffers.close()


def _assert_close(name, got, expected, tol=1.0 / 128.0):
    g, e = got.float(), expected.float()
    assert not g.isnan().any(), f"{name}: fused output contains NaN"
    assert not e.isnan().any(), f"{name}: split output contains NaN"
    error = (g - e).abs()
    assert (error <= tol * e.abs() + tol).all(), (
        f"{name}: max difference {error.max().item():.6g}"
    )


def _check_case(T, rows, rank, next_k1, window):
    import torch.distributed as dist

    # Keep this direct-builder test small while preserving NHD cache semantics.
    T.PAGES = 512
    layer = T.Layer(
        rows,
        8,
        torch.device("cuda", rank),
        seed=700 + rows,
        lens=[1000 + 137 * request for request in range(rows // 8)],
        window=window,
    )
    layer.router.zero_()
    layer.w2k["bias"].copy_(
        torch.tensor(deterministic_router_bias(), device=layer.dev)
    )
    kernels = _kernels(rows, 8)

    split = _run_path(T, kernels, layer, rows, rank, next_k1, fused=False)
    dist.barrier()
    fused = _run_path(T, kernels, layer, rows, rank, next_k1, fused=True)

    route_ids, route_scores = deterministic_routing()
    expected_ids = torch.tensor(route_ids, dtype=torch.int32).repeat(rows, 1)
    expected_scores = torch.tensor(route_scores).repeat(rows, 1)
    assert torch.equal(split["ids"], expected_ids)
    assert torch.equal(fused["ids"], expected_ids)
    assert torch.allclose(split["scores"], expected_scores, rtol=0, atol=1e-6)
    assert torch.allclose(fused["scores"], expected_scores, rtol=0, atol=1e-6)
    assert torch.equal(fused["guard"], split["guard"])
    assert not fused["guard"].any() and not split["guard"].any()
    assert not fused["peer guard"].any() and not split["peer guard"].any()

    names = [
        "residual",
        "scores",
        "xin",
        "attention_input",
        "current K cache",
        "current V cache",
    ]
    if next_k1:
        names += [
            "next Q",
            "next K",
            "next V",
            "next gate",
            "next K cache",
            "next V cache",
        ]
    nan_counts = {
        name: {
            "fused": int(torch.isnan(fused[name].float()).sum()),
            "split": int(torch.isnan(split[name].float()).sum()),
        }
        for name in names
    }
    assert not any(
        count for paths in nan_counts.values() for count in paths.values()
    ), f"NaN outputs: {nan_counts}"
    for name in names:
        _assert_close(name, fused[name], split[name])

    return {
        "width": rows,
        "next_k1": next_k1,
        "attention": "swa" if window else "full",
        "window": window,
        "status": "ok",
        "guard": [int(value) for value in fused["guard"].tolist()],
    }


def run_rank_case(rank, rows):
    import test_muse_spark_gfx1250_layer as T
    import torch.distributed as dist

    cases = []
    for window in (0, 2047):
        for next_k1 in (False, True):
            cases.append(_check_case(T, rows, rank, next_k1, window))
            dist.barrier()
    return {"rank": rank, "cases": cases}


def run_rank(rank):
    results = [run_rank_case(rank, rows) for rows in WIDTHS]
    return {
        "rank": rank,
        "cases": [case for result in results for case in result["cases"]],
    }


def _worker(rank, port, queue):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=ACTIVE_NPES)
    try:
        queue.put(run_rank(rank))
    except Exception as exc:
        import traceback

        queue.put(
            {"rank": rank, "error": f"{exc}\n{traceback.format_exc()}"}
        )
    finally:
        dist.destroy_process_group()


def run():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 31500 + os.getpid() % 1000
    processes = [
        context.Process(target=_worker, args=(rank, port, queue))
        for rank in range(ACTIVE_NPES)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=1800) for _ in range(ACTIVE_NPES)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0, (
            f"layer worker pid={process.pid} exit={process.exitcode}"
        )
    return sorted(results, key=lambda result: result["rank"])


four_gfx1250 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= ACTIVE_NPES
        and all(
            "gfx1250" in torch.cuda.get_device_properties(index).gcnArchName
            for index in range(min(torch.cuda.device_count(), ACTIVE_NPES))
        )
    ),
    reason="needs four gfx1250 GPUs",
)


@four_gfx1250
def test_active4_logical8_whole_layer_matches_split():
    results = run()
    for result in results:
        assert "error" not in result, result.get("error")
        validate_case_result(result)

