#!/usr/bin/env python3
"""Run the isolated K-attention test under rocprofiler counter injection."""

from __future__ import annotations

import sys

import rocm_sdk


_initialize_process = rocm_sdk.initialize_process


def _initialize_without_profiler_reloads(*, preload_shortnames=None, **kwargs):
    """Avoid compiler/profiler reloads after rocprofv3 initializes the process."""
    excluded = {
        "amd_comgr",
        "rocprofiler-sdk",
        "rocprofiler-sdk-roctx",
        "roctracer64",
        "roctx64",
    }
    return _initialize_process(
        preload_shortnames=[
            name
            for name in (preload_shortnames or ())
            if name not in excluded
        ],
        **kwargs,
    )


rocm_sdk.initialize_process = _initialize_without_profiler_reloads

import pytest  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "-s",
                "test/kernels/test_gfx1250_kattn_bn64_2warp.py"
                "::test_gfx1250_kattn_bn64_2warp_compute_only",
                *sys.argv[1:],
            ]
        )
    )
