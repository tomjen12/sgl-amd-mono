# SPDX-License-Identifier: Apache-2.0
"""Bounded four-GPU mailbox protocol gate for logical TP8, Width 8/16."""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from tp8_logical_case import build_k2_case, build_k3_case
from tp8_mono_fixtures import prepare_mono
from tp8_shadow_protocol import step_tag
from tp8_synthetic_contract import CONTRACT


def _barrier(group) -> None:
    dist.monitored_barrier(
        group=group,
        timeout=timedelta(seconds=30),
        wait_all_ranks=True,
    )


def _poison(protocol: dict, stage: str) -> None:
    from sgl_amd_mono.common import guard

    buffers = protocol["buffers"]
    buffers.peers.bytes[: -guard.FLAG_BYTES].fill_(0xA5)
    for shadow in buffers.peers.shadow_bytes:
        shadow.fill_(0xA5)
    getattr(buffers, f"scratch_{stage}").fill_(0xA5)
    torch.cuda.synchronize()


def _snapshot(prepared) -> dict[str, torch.Tensor]:
    assert prepared.outputs is not None
    return {
        name: value.detach().clone()
        for name, value in prepared.outputs().items()
    }


def _assert_same(
    stage: str,
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            error = (expected[name].float() - actual[name].float()).abs().max()
            raise AssertionError(
                f"{stage}.{name} changed across mailbox reuse; max error={error.item()}"
            )


def _assert_clean_guard(protocol: dict) -> None:
    from sgl_amd_mono.common import guard

    errors = protocol["epoch"][list(guard.ERROR_WORDS)].cpu().tolist()
    if errors != [0, 0, 0]:
        raise AssertionError(f"guard errors after protocol launch: {errors}")


def _exercise_stage(
    stage: str, width: int, rank: int, world: int, group
) -> dict:
    device = torch.device("cuda", rank)
    case = (
        build_k2_case(width, rank, world, device)
        if stage == "k2"
        else build_k3_case(width, rank, world, device)
    )
    prepared = prepare_mono(
        stage,
        width,
        rank,
        peer_group=group,
        active_npes=world,
        logical_case=case,
    )
    assert prepared.protocol is not None
    protocol = prepared.protocol
    pointers = {
        "peer": protocol["buffers"].peers.bytes.data_ptr(),
        "scratch": getattr(protocol["buffers"], f"scratch_{stage}").data_ptr(),
    }
    try:
        # Deliberately begin with stale nonzero mailbox, shadow, and scratch
        # bytes. before_each() may publish only current missing-rank slots.
        _poison(protocol, stage)
        prepared.before_each()
        _barrier(group)
        prepared.launch()
        torch.cuda.synchronize()
        expected = _snapshot(prepared)
        _assert_clean_guard(protocol)

        # Reuse the same allocations and prior launch bytes without clearing.
        epochs = [int(protocol["epoch"][0].item())]
        for _ in range(3):
            prepared.before_each()
            _barrier(group)
            prepared.launch()
            torch.cuda.synchronize()
            _assert_same(stage, expected, _snapshot(prepared))
            _assert_clean_guard(protocol)
            epochs.append(int(protocol["epoch"][0].item()))

        # Cross the signed int32 tag boundary with stale bytes still resident.
        from sgl_amd_mono.muse_spark.gfx1250.config import LAYER_SLOTS
        from test_muse_spark_tp8_layer_gfx1250 import LAYER

        rollover_step = (2**31 - 1 - LAYER) // LAYER_SLOTS + 1
        protocol["epoch"][0] = rollover_step - 1
        prepared.before_each()
        _barrier(group)
        prepared.launch()
        torch.cuda.synchronize()
        _assert_same(stage, expected, _snapshot(prepared))
        _assert_clean_guard(protocol)
        rollover_tag = step_tag(rollover_step, LAYER)
        if rollover_tag >= 0:
            raise AssertionError(f"expected signed tag rollover, got {rollover_tag}")

        if pointers != {
            "peer": protocol["buffers"].peers.bytes.data_ptr(),
            "scratch": getattr(
                protocol["buffers"], f"scratch_{stage}"
            ).data_ptr(),
        }:
            raise AssertionError(f"{stage} protocol allocations were rebuilt")
        return {
            "stage": stage,
            "epochs": epochs,
            "rollover_step": rollover_step,
            "rollover_tag": rollover_tag,
            "reused_allocations": True,
            "stale_sentinel": "0xa5",
        }
    finally:
        prepared.close()


def _exercise_missing_rank_timeout(
    width: int, rank: int, world: int, group
) -> dict:
    from sgl_amd_mono.common import guard

    device = torch.device("cuda", rank)
    case = build_k2_case(width, rank, world, device)
    prepared = prepare_mono(
        "k2",
        width,
        rank,
        peer_group=group,
        active_npes=world,
        logical_case=case,
        seed_missing_ranks=False,
        guard_stop=False,
        deadline_s=0.2,
    )
    assert prepared.protocol is not None
    try:
        prepared.before_each()
        _barrier(group)
        started = time.monotonic()
        prepared.launch()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        errors = prepared.protocol["epoch"][list(guard.ERROR_WORDS)].cpu().tolist()
        if not any(errors):
            raise AssertionError(f"missing rank did not fail closed: {errors}")
        if elapsed >= 10.0:
            raise AssertionError(f"missing-rank timeout took {elapsed:.3f}s")
        all_errors: list[list[int] | None] = [None] * world
        dist.all_gather_object(all_errors, errors, group=group)
        if not any(item and item[0] for item in all_errors):
            raise AssertionError(
                f"missing-rank failure had no originating deadline: {all_errors}"
            )
        return {
            "missing_rank_fail_closed": True,
            "guard_errors": errors,
            "elapsed_s": elapsed,
        }
    finally:
        prepared.close()


def run_protocol(width: int, rank: int, world: int) -> dict | None:
    if world != 4:
        raise RuntimeError(f"Phase-A protocol gate requires 4 ranks, got {world}")
    group = dist.new_group(backend="gloo")
    try:
        results = [
            _exercise_stage("k2", width, rank, world, group),
            _exercise_stage("k3", width, rank, world, group),
            _exercise_missing_rank_timeout(width, rank, world, group),
        ]
        gathered: list[list[dict] | None] = [None] * world
        dist.all_gather_object(gathered, results, group=group)
        return (
            {
                "status": "passed",
                "width": width,
                "logical_tp_ep": 8,
                "active_ranks": 4,
                "resident_cu": 96,
                "ranks": gathered,
            }
            if rank == 0
            else None
        )
    finally:
        dist.destroy_process_group(group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    try:
        result = run_protocol(args.width, rank, world)
        if result is not None:
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
