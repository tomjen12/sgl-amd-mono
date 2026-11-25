# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

KERNEL_TESTS = Path(__file__).resolve().parents[1] / "kernels"
sys.path.insert(0, str(KERNEL_TESTS))

import tp8_baseline_production as production  # noqa: E402


class Tensor:
    def __init__(
        self,
        shape,
        dtype=torch.bfloat16,
        contiguous=True,
        data_ptr=0x1000,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._contiguous = contiguous
        self._data_ptr = data_ptr

    def stride(self):
        stride, values = 1, []
        for size in reversed(self.shape):
            values.append(stride)
            stride *= size
        return tuple(reversed(values))

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        value = 1
        for size in self.shape:
            value *= size
        return value

    def element_size(self):
        return 2

    def data_ptr(self):
        return self._data_ptr


def _environment():
    return dict(production.PRODUCTION_ENV)


@pytest.mark.parametrize("active", [1, 2, 4])
def test_active_logical_tp8_accepts_supported_world_sizes(active):
    collective = production.ActiveLogicalTP8(active)
    assert collective.active_world_size == active
    assert collective.logical_world_size == 8


@pytest.mark.parametrize("active", [0, 3, 8])
def test_active_logical_tp8_rejects_unsupported_world_sizes(active):
    with pytest.raises(production.BaselineContractError):
        production.ActiveLogicalTP8(active)


def test_production_environment_accepts_recipe_and_synthetic_blocks():
    recipe = dict(production.PRODUCTION_ENV)
    recipe.pop("SGLANG_AMD_MONO_BLOCKS")
    assert production.validate_production_environment(
        _environment(), recipe_env=recipe
    )["HSA_CU_MASK"] == "0-3:0-95"


def test_pinned_recipe_contains_the_validated_production_environment():
    pytest.importorskip("yaml")
    recipe = (
        Path(__file__).resolve().parents[5]
        / "recipe.mi45x.pr50707.tp4.yaml"
    )
    recipe_env = production.load_recipe_environment(recipe)
    process = dict(recipe_env)
    process["SGLANG_AMD_MONO_BLOCKS"] = "96"
    checked = production.validate_recipe_environment(recipe, process)
    assert checked["SGLANG_USE_AITER_UNIFIED_ATTN"] == "1"


@pytest.mark.parametrize(
    "where,name,value",
    [
        ("process", "SGLANG_USE_AITER", "0"),
        ("process", "SGLANG_AMD_MONO_BLOCKS", "128"),
        ("recipe", "SGLANG_USE_AITER_UNIFIED_ATTN", "0"),
    ],
)
def test_production_environment_fails_closed(where, name, value):
    process, recipe = _environment(), dict(production.PRODUCTION_ENV)
    recipe.pop("SGLANG_AMD_MONO_BLOCKS")
    (process if where == "process" else recipe)[name] = value
    with pytest.raises(production.BaselineContractError):
        production.validate_production_environment(process, recipe_env=recipe)


def test_k1_calls_sglang_entrypoint_and_records_tp8_tensors(monkeypatch):
    called = {}

    def entrypoint(**kw):
        called.update(kw)
        return "qkvg"

    monkeypatch.setattr(production, "_import_attr", lambda *_: entrypoint)
    kw = {
        "x": Tensor((8, 8192)),
        "weight": Tensor((2304, 8192)),
        "positions": Tensor((8,), torch.int64),
        "cos_sin": None,
        "cache_k": Tensor((64, 2, 64), torch.float8_e4m3fnuz),
        "cache_v": Tensor((64, 2, 64), torch.float8_e4m3fnuz),
        "locations": Tensor((8,), torch.int64),
        "mapping": None,
        "eps": 1e-5,
        "q_scale": 1.0,
        "k_scale": None,
        "v_scale": None,
        "num_heads": 16,
        "num_kv_heads": 2,
    }
    result = production.K1BaselineAdapter()(**kw)
    assert result.value == "qkvg"
    assert called["x"] is kw["x"]
    assert result.tensors["k1.weight"]["shape"] == [2304, 8192]


def test_k1_rejects_tp4_weight_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        production,
        "_import_attr",
        lambda *_: pytest.fail("must fail before production dispatch"),
    )
    with pytest.raises(production.BaselineContractError, match="weight shape"):
        production.K1BaselineAdapter()(
            x=Tensor((8, 8192)),
            weight=Tensor((4608, 8192)),
            positions=Tensor((8,)),
            locations=Tensor((8,)),
            cache_k=Tensor((64, 2, 64)),
            cache_v=Tensor((64, 2, 64)),
        )


def test_kattn_requires_aiter_deferred_partials(monkeypatch):
    class Deferral:
        requested = False

        @staticmethod
        def take():
            return None

    monkeypatch.setattr(production, "_import_attr", lambda *_: Deferral())
    attention = SimpleNamespace(
        attn=lambda *args, **kwargs: Tensor((8, 1024)),
        _fused_o_norm_gate=SimpleNamespace(),
    )
    with pytest.raises(production.BaselineContractError, match="deferred partials"):
        production.KAttnBaselineAdapter()(
            attention=attention,
            q=Tensor((8, 1024)),
            k=Tensor((8, 128)),
            v=Tensor((8, 128)),
            forward_batch=object(),
            gate=Tensor((8, 1024)),
        )


def test_k2_uses_boundary_and_shared_router_pooler(monkeypatch):
    ffn = Tensor((16, 8192))
    projected = Tensor((16, 8192))
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            o_proj=lambda value: (projected, None)
        ),
        _fused_attn_boundary=SimpleNamespace(
            forward=lambda stream, output, branch_dtype: (ffn, stream, None)
        ),
        mlp=SimpleNamespace(
            router_pooler_packed_weight=Tensor((4608, 8192))
        ),
    )
    monkeypatch.setattr(
        production,
        "_import_attr",
        lambda *_: lambda moe, value: ("scores", "ids", "expert", None),
    )
    result = production.K2BaselineAdapter()(
        layer=layer,
        stream=Tensor((16, 8192), torch.float32, contiguous=False),
        attention_output=Tensor((16, 1024)),
    )
    assert result.value[1:3] == ("scores", "ids")
    assert result.tensors["k2.projected"]["shape"] == [16, 8192]
    assert result.tensors["k2.ffn_input"]["shape"] == [16, 8192]


def test_k3_calls_aiter_path_active_collective_and_sglang_unpool(monkeypatch):
    calls = []
    local = Tensor((8, 4096))

    def run(*args, **kwargs):
        calls.append("aiter")
        return local

    monkeypatch.setattr(production, "_import_attr", lambda *_: run)

    class Distributed:
        def all_reduce(self, value):
            assert value is local
            calls.append("all_reduce")
            return value

    def unpool(value):
        assert value is local
        calls.append("unpool")
        return "output"

    unpool.__module__ = "sglang.srt.rocm_muse_spark.model"
    result = production.K3BaselineAdapter(Distributed())(
        x=Tensor((8, 4096)),
        scores=Tensor((8, 8), torch.float32),
        ids=Tensor((8, 8), torch.int32),
        experts=SimpleNamespace(
            num_local_experts=32,
            gfx1250_a8w4=tuple(Tensor((32, 1)) for _ in range(4)),
        ),
        local_start=0,
        norm_weight=Tensor((4096,)),
        eps=1e-5,
        gain=0.0,
        unpool=unpool,
    )
    assert result.value == "output"
    assert calls == ["aiter", "all_reduce", "unpool"]
