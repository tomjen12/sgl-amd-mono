# SPDX-License-Identifier: Apache-2.0
"""One TP4/EP4 rank's fixed-width gfx1250 Muse Spark launches."""

from __future__ import annotations

import inspect
import os
import struct
from pathlib import Path

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.muse_spark.gfx1250.build_k2 import sym_bytes as k2_sym_bytes
from sgl_amd_mono.muse_spark.gfx1250.build_k3 import k3_sym_bytes
from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
    layer_scratch_bytes,
    layer_sym_bytes,
)
from sgl_amd_mono.muse_spark.gfx1250.config import (
    DRAFT_TOKENS,
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    KV_K,
    LAYER_SLOTS,
    LOCAL_EXPERTS,
    LOCAL_KV_HEADS,
    MAX_TOKENS,
    MOE_HIDDEN,
    PAGE,
    Q_K,
    TOP_K,
    WINDOW,
    WIDTHS,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    k1_scratch_layout,
    scratch_bytes,
)

FP8_CACHE = torch.float8_e4m3fn
USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_MUSE_KERNELS", "1") == "1"

# SGLang classes imported lazily by cache validation and page-table binding.
SGLANG_READS = {
    "srt/layers/attention/aiter_backend.py": ("AiterAttnBackend",),
    "srt/layers/attention/triton_backend.py": ("TritonAttnBackend",),
}


def _ptr(value) -> int:
    return value.data_ptr()


def _f32_bits(value: float) -> int:
    return struct.unpack("i", struct.pack("f", value))[0]


def _ensure_writable_flydsl_cache() -> None:
    current = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if current and os.access(current, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def no_capture(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"Muse Spark gfx1250 monokernels: {what} inside graph capture")


def compile_only(launch) -> None:
    from flydsl.utils.env import CompileEnvManager

    names = [name for name in inspect.signature(launch.func).parameters if name != "stream"]
    variable = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(variable)
    os.environ[variable] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = old


class MuseMonoRunner:
    """Buffers, cache binding and persistent layer launches for gfx1250 width 8."""

    def __init__(
        self, *, views, modules, boundaries, rank: int, npes: int, cpu_group, device
    ):
        _ensure_writable_flydsl_cache()
        if npes != 4:
            raise ValueError(f"gfx1250 Muse Spark requires TP4/EP4, got {npes}")
        if not len(views) == len(modules) == len(boundaries) <= LAYER_SLOTS:
            raise ValueError("invalid Muse Spark layer views")
        self.views = list(views)
        self.modules = list(modules)
        self.boundaries = list(boundaries)
        self.rank, self.npes, self.device = rank, npes, torch.device(device)
        self.use_kernels = USE_KERNELS
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        self.deadline_s = None
        self.backend = self.pool = None
        self.caches = self.windows = self.swa_map = None
        self.slot_row = self.full_page_table = None
        self.swa_page_starts = self.swa_page_table = None
        self._k1_done = -1
        self._k1, self._layer = {}, {}

        dev, bf16 = self.device, torch.bfloat16
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        self.scratch_k1 = torch.zeros(
            scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=dev
        )
        self.scratch_layer = torch.zeros(
            layer_scratch_bytes(), dtype=torch.uint8, device=dev
        )
        symmetric = max(
            k2_sym_bytes(npes), k3_sym_bytes(npes), layer_sym_bytes(npes)
        )
        self.peers = PeerBuffer(
            symmetric + guard.FLAG_BYTES, cpu_group, rank, npes, dev
        )
        self.peers.bytes.zero_()
        self.q = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.k = torch.empty(MAX_TOKENS, KV_K, dtype=bf16, device=dev)
        self.v = torch.empty(MAX_TOKENS, KV_K, dtype=bf16, device=dev)
        self.g = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.o = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.ids = torch.empty(MAX_TOKENS, TOP_K, dtype=torch.int32, device=dev)
        self.scores = torch.empty(
            MAX_TOKENS, TOP_K, dtype=torch.float32, device=dev
        )
        self.xin = torch.empty(
            MAX_TOKENS, MOE_HIDDEN, dtype=bf16, device=dev
        )
        self.attn_in = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.prefix = torch.empty(
            MAX_TOKENS // DRAFT_TOKENS, dtype=torch.int32, device=dev
        )
        tables = [view.cos_sin for view in self.views if view.cos_sin is not None]
        if not tables:
            raise RuntimeError("Muse Spark gfx1250: no rotary table")
        self.any_cos_sin = tables[0]

    def set_deadline(self, seconds: float) -> None:
        self.deadline_s = float(seconds)
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = (
            lo - (1 << 32) if lo >= 1 << 31 else lo
        )
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def attention_debug(self, rows: int):
        """Views of the most recent fused-attention intermediates.

        The regions are dedicated to attention and remain valid after the
        remainder of the fused layer launch.  This is diagnostics-only: callers
        must consume or clone the views before launching another layer.
        """
        from sgl_amd_mono.muse_spark.gfx1250.build_layer import _private_layout
        from sgl_amd_mono.muse_spark.gfx1250.layout import (
            FUSED_ROWS,
            FUSED_SPLITS,
            fused_tiles,
        )

        layout = _private_layout()
        tiles = fused_tiles(rows)

        def floats(name):
            offset, size = layout[name]
            return self.scratch_layer[offset : offset + size].view(torch.float32)

        return {
            "stats": floats("attn_stats")[
                : tiles * FUSED_SPLITS * FUSED_ROWS * 2
            ].view(tiles, FUSED_SPLITS, FUSED_ROWS, 2),
            "global": floats("attn_global")[
                : tiles * FUSED_ROWS * 2
            ].view(tiles, FUSED_ROWS, 2),
            "part": floats("attn_part")[
                : tiles * FUSED_SPLITS * FUSED_ROWS * HEAD_DIM
            ].view(tiles, FUSED_SPLITS, FUSED_ROWS, HEAD_DIM),
        }

    def layer_debug(self, rows: int):
        """Views of K2 outputs that remain live after the fused layer."""
        from sgl_amd_mono.muse_spark.gfx1250.build_layer import _private_layout

        layout = _private_layout()

        def bf16(name, columns):
            offset, _ = layout[name]
            count = rows * columns
            return self.scratch_layer[offset : offset + count * 2].view(
                torch.bfloat16
            ).view(rows, columns)

        return {
            "k2_x": bf16("k2_x", HIDDEN),
            "k2_pool": bf16("k2_pool", MOE_HIDDEN),
            "k3_qmid": bf16("k3_qmid", TOP_K * MOE_HIDDEN).view(
                rows * TOP_K, MOE_HIDDEN
            ),
            "k3_y": bf16("k3_y", TOP_K * MOE_HIDDEN).view(
                rows * TOP_K, MOE_HIDDEN
            ),
            "k3_fbuf": bf16("k3_fbuf", MOE_HIDDEN),
            "k3_branch": bf16("k3_branch", HIDDEN),
            "xin": self.xin[:rows],
            "ids": self.ids[:rows],
            "scores": self.scores[:rows],
        }

    def k1_kernel(self, rows: int):
        if rows not in self._k1:
            from sgl_amd_mono.muse_spark.gfx1250.layer import K1Build, build_k1

            no_capture("K1 first reached")
            self._k1[rows] = build_k1(
                K1Build(
                    tokens=rows,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                )
            )
        return self._k1[rows]

    def layer_kernel(
        self, rows: int, q_len: int, next_k1: bool, last: bool, shuffled: bool
    ):
        key = rows, q_len, next_k1, last, shuffled
        if key not in self._layer:
            from sgl_amd_mono.muse_spark.gfx1250.build_layer import (
                LayerBuild,
                build_layer,
            )

            no_capture(f"Layer V{2 if next_k1 else 1} first reached")
            self._layer[key] = build_layer(
                LayerBuild(
                    tokens=rows,
                    q_len=q_len,
                    npes=self.npes,
                    next_k1=next_k1,
                    last=last,
                    shuffled=shuffled,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                )
            )
        return self._layer[key]

    def _next_k1(self, index: int) -> bool:
        return (
            index + 1 < len(self.views)
            and self.views[index + 1].kind == "mxfp4"
        )

    def precompile(self, rows: int, q_len: int = DRAFT_TOKENS) -> None:
        if not self.use_kernels:
            return
        if rows not in WIDTHS or q_len != DRAFT_TOKENS or rows % q_len:
            raise ValueError(
                f"gfx1250 Muse Spark supports rows in {WIDTHS}, q_len={DRAFT_TOKENS}; "
                f"got {(rows, q_len)}"
            )
        compile_only(self.k1_kernel(rows))
        variants = {
            (self._next_k1(i), i == len(self.views) - 1, view.shuffled)
            for i, view in enumerate(self.views)
            if view.kind == "mxfp4"
        }
        for next_k1, last, shuffled in variants:
            compile_only(
                self.layer_kernel(rows, q_len, next_k1, last, shuffled)
            )

    def cache_reasons(self, backend, pool) -> list[str]:
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        if not isinstance(backend, (AiterAttnBackend, TritonAttnBackend)):
            return [
                f"attention backend {type(backend).__name__}; want Aiter or Triton"
            ]
        reasons = []
        if backend.page_size != PAGE:
            reasons.append(f"page size {backend.page_size}, want {PAGE}")
        if isinstance(backend, AiterAttnBackend) and backend.kv_cache_dtype != FP8_CACHE:
            reasons.append(f"KV cache {backend.kv_cache_dtype}, want {FP8_CACHE}")
        if (
            isinstance(backend, AiterAttnBackend)
            and not backend.use_triton_unified_attention
        ):
            reasons.append("Aiter unified attention page tables are disabled")
        if isinstance(backend, TritonAttnBackend):
            table = backend.req_to_token
            if (
                table.dtype != torch.int32
                or table.dim() != 2
                or not table.is_contiguous()
            ):
                reasons.append(
                    f"Triton request-to-token table: {table.dtype} {tuple(table.shape)}"
                )
        if (
            not backend.use_sliding_window_kv_pool
            or type(pool).__name__ != "SWAKVPool"
        ):
            return reasons + [f"KV pool {type(pool).__name__}, want SWAKVPool"]
        mapping = pool.full_to_swa_index_mapping
        if (
            mapping is None
            or mapping.dtype not in (torch.int32, torch.int64)
            or not mapping.is_contiguous()
        ):
            reasons.append("no contiguous full-to-SWA slot mapping")
        for index, layer in enumerate(self.modules):
            attention = layer.self_attn.attn
            _, is_swa = pool.layers_mapping[attention.layer_id]
            if is_swa != (attention.sliding_window_size == WINDOW):
                reasons.append(f"layer {index}: sliding-window pool mismatch")
            k_cache, v_cache = pool.get_kv_buffer(attention.layer_id)
            for cache, name in ((k_cache, "K"), (v_cache, "V")):
                if not (
                    cache.dtype == FP8_CACHE
                    and cache.is_contiguous()
                    and cache.dim() == 3
                    and tuple(cache.shape[-2:]) == (LOCAL_KV_HEADS, HEAD_DIM)
                ):
                    reasons.append(
                        f"layer {index} {name}: {cache.dtype} {tuple(cache.shape)}"
                    )
        return reasons

    def bind(self, backend, pool) -> None:
        no_capture("caches bound")
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.caches, self.windows = [], []
        for layer in self.modules:
            attention = layer.self_attn.attn
            _, is_swa = pool.layers_mapping[attention.layer_id]
            k_cache, v_cache = pool.get_kv_buffer(attention.layer_id)
            self.caches.append(
                (_ptr(k_cache), _ptr(v_cache), bool(is_swa))
            )
            window = int(attention.sliding_window_size or -1)
            self.windows.append(window if window > 0 else 0)
        self.swa_map = pool.full_to_swa_index_mapping
        if isinstance(backend, TritonAttnBackend):
            # Triton exposes token slot ids, while K-attn consumes one physical
            # page id per logical page. These address-stable buffers are filled
            # by captured torch ops before each layer launch.
            request_table = backend.req_to_token
            pages = (request_table.shape[1] + PAGE - 1) // PAGE
            max_requests = MAX_TOKENS // DRAFT_TOKENS
            self.slot_row = torch.empty(
                (max_requests, request_table.shape[1]),
                dtype=request_table.dtype,
                device=request_table.device,
            )
            self.full_page_table = torch.empty(
                (max_requests, pages), dtype=torch.int32, device=request_table.device
            )
            self.swa_page_starts = torch.empty(
                (max_requests * pages,),
                dtype=self.swa_map.dtype,
                device=request_table.device,
            )
            self.swa_page_table = torch.empty_like(self.full_page_table)
        self.backend, self.pool = backend, pool

    def _triton_page_table(
        self, is_swa: bool, forward_batch, requests: int
    ) -> torch.Tensor:
        """Build the absolute page table from Triton's graph-stable request table."""
        request = forward_batch.req_pool_indices[:requests]
        torch.index_select(
            self.backend.req_to_token, 0, request, out=self.slot_row[:requests]
        )
        page_starts = self.slot_row[:requests, ::PAGE]
        table = self.full_page_table[:requests]
        if is_swa:
            torch.index_select(
                self.swa_map,
                0,
                page_starts.reshape(-1),
                out=self.swa_page_starts[: page_starts.numel()],
            )
            page_starts = self.swa_page_starts[: page_starts.numel()].reshape(
                requests, -1
            )
            table = self.swa_page_table[:requests]
        torch.div(page_starts, PAGE, rounding_mode="floor", out=table)
        return table

    def _attention_args(
        self, index: int, rows: int, q_len: int, forward_batch
    ) -> tuple:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        metadata = self.backend.forward_metadata
        _, _, is_swa = self.caches[index]
        requests = rows // q_len
        table = (
            self._triton_page_table(is_swa, forward_batch, requests)
            if isinstance(self.backend, TritonAttnBackend)
            else metadata.swa_page_table
            if is_swa
            else metadata.kv_indices
        )
        if (
            table is None
            or table.dtype != torch.int32
            or table.dim() != 2
            or not table.is_contiguous()
        ):
            raise RuntimeError(
                f"gfx1250 K-attn page table: "
                f"{None if table is None else (table.dtype, tuple(table.shape))}"
            )
        if forward_batch.seq_lens.numel() < requests or table.shape[0] < requests:
            raise RuntimeError(
                f"gfx1250 attention needs {requests} requests, got "
                f"{forward_batch.seq_lens.numel()} lengths and table {tuple(table.shape)}"
            )
        self.prefix[:requests].copy_(forward_batch.seq_lens[:requests])
        scale = float(self.modules[index].self_attn.attn.scaling)
        return (
            _ptr(table),
            table.stride(0),
            _ptr(self.prefix),
            self.windows[index],
            _f32_bits(scale),
        )

    def launch(
        self,
        *,
        i: int,
        rows: int,
        q_len: int,
        positions,
        forward_batch,
        stream,
        attn_input,
        pre_norm_dtype,
    ):
        view = self.views[i]
        if os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1":
            self.check_q_input = None
        if (
            not self.use_kernels
            or view.kind != "mxfp4"
            or rows not in WIDTHS
            or q_len != DRAFT_TOKENS
            or rows % q_len
        ):
            result = self.modules[i](
                positions,
                stream,
                forward_batch,
                attn_input=attn_input,
                ffn_boundary=self.boundaries[i],
                pre_norm_dtype=pre_norm_dtype,
            )
            return result[0], result[1]
        if self.pool is None:
            raise RuntimeError("Muse Spark gfx1250 launch before cache bind")
        if stream.dtype != torch.float32 or tuple(stream.shape) != (
            rows,
            HIDDEN,
        ):
            raise RuntimeError(
                f"gfx1250 stream {tuple(stream.shape)} {stream.dtype}"
            )
        if attn_input.dtype != torch.bfloat16 or tuple(attn_input.shape) != (
            rows,
            HIDDEN,
        ):
            raise RuntimeError(
                f"gfx1250 attention input {tuple(attn_input.shape)} {attn_input.dtype}"
            )
        slots = forward_batch.out_cache_loc
        if slots.dtype not in (torch.int32, torch.int64) or slots.numel() < rows:
            raise RuntimeError("gfx1250 invalid cache slots")

        current = torch.cuda.current_stream()
        k_cache, v_cache, is_swa = self.caches[i]
        mapping = self.swa_map
        cos_sin = view.cos_sin if view.cos_sin is not None else self.any_cos_sin
        if self._k1_done != i:
            self.k1_kernel(rows)(
                _ptr(attn_input),
                _ptr(view.w_qkvg),
                _ptr(cos_sin),
                _ptr(positions),
                _ptr(slots),
                _ptr(mapping),
                k_cache,
                v_cache,
                _ptr(self.q),
                _ptr(self.k),
                _ptr(self.v),
                _ptr(self.g),
                _ptr(self.scratch_k1),
                _ptr(self.epoch),
                i,
                int(view.use_rope),
                int(is_swa),
                int(slots.dtype == torch.int64),
                int(mapping.dtype == torch.int64),
                stream=current,
            )
        if os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1":
            # Preserve the exact Q consumed by the fused layer.  The prior
            # layer's pipelined K1 may be replaced by standalone K1 here, and
            # this same buffer is reused by next-K1 before check mode runs.
            self.check_q_input = self.q[:rows].clone()
        self._k1_done = -1

        next_k1 = self._next_k1(i)
        next_view = self.views[i + 1] if next_k1 else view
        next_k_cache, next_v_cache, next_is_swa = (
            self.caches[i + 1] if next_k1 else self.caches[i]
        )
        next_cos_sin = (
            next_view.cos_sin
            if next_view.cos_sin is not None
            else self.any_cos_sin
        )
        table, table_stride, prefix, window, scale_bits = self._attention_args(
            i, rows, q_len, forward_batch
        )
        from sglang.srt.rocm_muse_spark.replay_capture import (
            get_routing_override,
            get_routing_score_override,
            record_route_workload,
        )

        override_ids = get_routing_override(i)
        override_scores = get_routing_score_override(i)
        if (override_ids is None) != (override_scores is None):
            raise ValueError(
                f"gfx1250 replay layer {i} must override routing ids and scores together."
            )
        if override_ids is not None:
            expected = (rows, TOP_K)
            if (
                tuple(override_ids.shape) != expected
                or override_ids.dtype != torch.int32
                or override_ids.device != self.ids.device
                or not override_ids.is_contiguous()
            ):
                raise ValueError(
                    f"gfx1250 replay routing override layer {i} must be a "
                    f"contiguous {expected} int32 tensor on {self.ids.device}, got "
                    f"{tuple(override_ids.shape)} {override_ids.dtype} "
                    f"{override_ids.device} contiguous={override_ids.is_contiguous()}."
                )
        if override_scores is not None:
            expected = (rows, TOP_K)
            if (
                tuple(override_scores.shape) != expected
                or override_scores.dtype != torch.float32
                or override_scores.device != self.scores.device
                or not override_scores.is_contiguous()
            ):
                raise ValueError(
                    f"gfx1250 replay routing score override layer {i} must be a "
                    f"contiguous {expected} float32 tensor on {self.scores.device}, got "
                    f"{tuple(override_scores.shape)} {override_scores.dtype} "
                    f"{override_scores.device} "
                    f"contiguous={override_scores.is_contiguous()}."
                )
        last = i == len(self.modules) - 1
        self.layer_kernel(rows, q_len, next_k1, last, view.shuffled)(
            _ptr(self.q),
            _ptr(self.o),
            k_cache,
            v_cache,
            table,
            table_stride,
            prefix,
            window,
            scale_bits,
            _ptr(self.g),
            _ptr(stream),
            _ptr(view.w_o),
            _ptr(view.alpha),
            _ptr(view.beta),
            _ptr(view.pre_w),
            _ptr(view.gate_hi),
            _ptr(view.gate_lo),
            _ptr(view.bias),
            _ptr(view.w_pool),
            _ptr(view.xn_w),
            _ptr(self.ids),
            _ptr(self.scores),
            _ptr(override_ids) if override_ids is not None else 0,
            _ptr(override_scores) if override_scores is not None else 0,
            _ptr(self.xin),
            int(self.modules[i].mlp.local_expert_start),
            _ptr(view.w13),
            _ptr(view.s13),
            _ptr(view.w2),
            _ptr(view.s2),
            _ptr(view.pen_w),
            _ptr(view.pfw),
            _ptr(view.alpha2),
            _ptr(view.beta2),
            _ptr(view.npw),
            _ptr(view.w_unpool),
            _ptr(self.attn_in),
            _ptr(next_view.w_qkvg),
            _ptr(next_cos_sin),
            _ptr(positions),
            _ptr(slots),
            _ptr(mapping),
            next_k_cache,
            next_v_cache,
            _ptr(self.k),
            _ptr(self.v),
            int(next_view.use_rope),
            int(next_is_swa),
            int(slots.dtype == torch.int64),
            int(mapping.dtype == torch.int64),
            _ptr(self.scratch_layer),
            self.peers.local,
            _ptr(self.peers.addresses),
            self.rank,
            _ptr(self.epoch),
            i,
            int(last),
            stream=current,
        )
        record_route_workload(
            layer=i,
            topk_ids=self.ids[:rows],
            topk_scores=self.scores[:rows],
            num_experts=EXPERTS,
            local_start=int(self.modules[i].mlp.local_expert_start),
            local_experts=LOCAL_EXPERTS,
        )
        if next_k1:
            self._k1_done = i + 1
        return stream, self.attn_in[:rows]

    def end_step(self) -> None:
        self.epoch[0].add_(1)
        self._k1_done = -1
