# SPDX-License-Identifier: Apache-2.0
"""CPU-only P1 contracts for the logical-TP8/active-four synthetic harness.

This module describes inputs, launch ordering, and result schemas.  It does not
execute a kernel and must never be used as evidence of GPU numerical
correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

LOGICAL_TP = 8
LOGICAL_EP = 8
ACTIVE_WORLD = 4
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
VERIFY_WIDTHS = (8, 16)
SWA_WINDOW = 2047
BF16_EXPERT_LAYERS = frozenset((0, 61))


class P1ContractError(ValueError):
    """A host-side P1 harness declaration violates the agreed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P1ContractError(message)


@dataclass(frozen=True)
class RoutingCase:
    name: str
    ids: tuple[tuple[int, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    expected_zero_local_rank: int | None = None
    tied_scores: bool = False


def _normalized(raw: Sequence[float]) -> tuple[float, ...]:
    total = float(sum(raw))
    return tuple(float(value) / total for value in raw)


def _rows(value: Sequence[int], rows: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value) for _ in range(rows))


def deterministic_routing_cases(rows: int) -> tuple[RoutingCase, ...]:
    """Return the five required deterministic routing stress fixtures."""
    _require(rows in VERIFY_WIDTHS, f"rows {rows}, expected one of {VERIFY_WIDTHS}")

    hot_ids = tuple(
        tuple([0] + [shard * LOCAL_EXPERTS + (token % 8) for shard in range(1, 8)])
        for token in range(rows)
    )
    zero_rank = 2
    zero_ids = _rows((0, 32, 96, 128, 160, 192, 224, 255), rows)
    boundary_ids = _rows((31, 32, 63, 127, 128, 159, 191, 255), rows)

    # Equal scores have an explicit expert-id tie break.  Rotate the candidate
    # construction to ensure the selected order does not depend on insertion.
    tie_ids = tuple(
        tuple(sorted(((token * 17 + offset * 29) % GLOBAL_EXPERTS for offset in range(TOP_K))))
        for token in range(rows)
    )
    imbalance_ids = tuple(
        tuple(
            [((token * 5 + offset) % LOCAL_EXPERTS) for offset in range(5)]
            + [32 + token % LOCAL_EXPERTS, 64 + token % LOCAL_EXPERTS, 128 + token % LOCAL_EXPERTS]
        )
        for token in range(rows)
    )

    descending = _normalized(range(TOP_K, 0, -1))
    equal = (1.0 / TOP_K,) * TOP_K
    cases = (
        RoutingCase("hot_expert", hot_ids, _rows(descending, rows)),
        RoutingCase(
            "zero_local_pick",
            zero_ids,
            _rows(descending, rows),
            expected_zero_local_rank=zero_rank,
        ),
        RoutingCase("expert_boundary", boundary_ids, _rows(descending, rows)),
        RoutingCase("score_ties", tie_ids, _rows(equal, rows), tied_scores=True),
        RoutingCase("imbalance", imbalance_ids, _rows(descending, rows)),
    )
    for case in cases:
        validate_routing_case(case, rows)
    return cases


def validate_routing_case(case: RoutingCase, rows: int) -> None:
    _require(len(case.ids) == rows, f"{case.name}: wrong row count")
    _require(len(case.scores) == rows, f"{case.name}: wrong score row count")
    for row_index, (ids, scores) in enumerate(zip(case.ids, case.scores)):
        _require(len(ids) == TOP_K, f"{case.name} row {row_index}: not Top-{TOP_K}")
        _require(len(set(ids)) == TOP_K, f"{case.name} row {row_index}: duplicate expert")
        _require(
            all(0 <= expert < GLOBAL_EXPERTS for expert in ids),
            f"{case.name} row {row_index}: expert outside global table",
        )
        _require(
            len(scores) == TOP_K and abs(sum(scores) - 1.0) < 1e-12,
            f"{case.name} row {row_index}: scores are not normalized",
        )
        _require(all(score >= 0 for score in scores), f"{case.name}: negative score")
        if case.tied_scores:
            _require(ids == tuple(sorted(ids)), f"{case.name}: tie break is not expert-id order")
            _require(len(set(scores)) == 1, f"{case.name}: scores are not tied")
    if case.expected_zero_local_rank is not None:
        start = case.expected_zero_local_rank * LOCAL_EXPERTS
        end = start + LOCAL_EXPERTS
        _require(
            not any(start <= expert < end for row in case.ids for expert in row),
            f"{case.name}: rank {case.expected_zero_local_rank} has a local pick",
        )


def routing_histogram(case: RoutingCase) -> tuple[int, ...]:
    counts = [0] * LOGICAL_EP
    for row in case.ids:
        for expert in row:
            counts[expert // LOCAL_EXPERTS] += 1
    return tuple(counts)


@dataclass(frozen=True)
class LayerVariant:
    width: int
    attention: str
    layer_version: str
    next_k1: bool
    window: int | None
    kv_pool: str

    @property
    def case_id(self) -> str:
        return f"w{self.width}-{self.attention}-{self.layer_version}"


def layer_variant_matrix() -> tuple[LayerVariant, ...]:
    variants = []
    for width in VERIFY_WIDTHS:
        for attention, window, pool in (
            ("full", None, "full"),
            ("swa", SWA_WINDOW, "swa"),
        ):
            for version, next_k1 in (("v1", False), ("v2", True)):
                variants.append(
                    LayerVariant(width, attention, version, next_k1, window, pool)
                )
    return tuple(variants)


@dataclass(frozen=True)
class ExecutionEvent:
    stage: str
    layer: int
    attention: str | None = None
    window: int | None = None
    kv_pool: str | None = None
    expert_family: str | None = None


def expected_execution(
    *,
    layer: int,
    variant: LayerVariant,
    bootstrap: bool = False,
    final_layer: int = 61,
    next_attention: str | None = None,
) -> tuple[ExecutionEvent, ...]:
    """Expected production sequence; current-layer K1 is never in a merged launch."""
    _require(0 <= layer <= final_layer, f"layer {layer} outside model")
    _require(
        not (variant.next_k1 and layer == final_layer),
        "last layer cannot launch a nonexistent next K1",
    )
    events: list[ExecutionEvent] = []
    if bootstrap:
        events.append(ExecutionEvent("k1_bootstrap", layer))
    events.extend(
        (
            ExecutionEvent(
                "kattn",
                layer,
                attention=variant.attention,
                window=variant.window,
                kv_pool=variant.kv_pool,
            ),
            ExecutionEvent("k2", layer),
            ExecutionEvent(
                "k3",
                layer,
                expert_family="bf16" if layer in BF16_EXPERT_LAYERS else "mxfp4",
            ),
        )
    )
    if variant.next_k1:
        mode = next_attention or variant.attention
        _require(mode in ("full", "swa"), f"invalid next attention {mode!r}")
        events.append(
            ExecutionEvent(
                "k1_next",
                layer + 1,
                attention=mode,
                window=SWA_WINDOW if mode == "swa" else None,
                kv_pool=mode,
            )
        )
    return tuple(events)


def validate_execution(
    actual: Sequence[ExecutionEvent], expected: Sequence[ExecutionEvent]
) -> None:
    _require(
        tuple(actual) == tuple(expected),
        f"execution mismatch: actual={tuple(actual)!r}, expected={tuple(expected)!r}",
    )


@dataclass(frozen=True)
class KVContract:
    layout: str = "NHD"
    page_size: int = 16
    local_kv_heads: int = 2
    head_dim: int = 64
    dtype: str = "fp8_e4m3"
    full_pool: str = "full"
    swa_pool: str = "swa"


REQUIRED_ENV = {
    "SGLANG_AMD_MONO_ARCH": "gfx1250",
    "SGLANG_ROCM_MUSE_GFX1250": "1",
    "SGLANG_ROCM_MUSE_EXTEND": "1",
    "SGLANG_ROCM_MUSE_EXTEND_GLUON": "0",
    "SGLANG_ROCM_MUSE_EXTEND_SWA_GLUON": "0",
    "HIP_FORCE_DEV_KERNARG": "1",
}


@dataclass(frozen=True)
class HarnessContract:
    logical_tp: int = LOGICAL_TP
    logical_ep: int = LOGICAL_EP
    active_world: int = ACTIVE_WORLD
    peer_slots: int = LOGICAL_TP
    local_q_heads: int = 16
    local_kv_heads: int = 2
    local_experts: int = LOCAL_EXPERTS
    pooler: str = "sharded"


def validate_launch_contract(
    declaration: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    kv: KVContract,
) -> None:
    expected = asdict(HarnessContract())
    for name, value in expected.items():
        _require(
            declaration.get(name) == value,
            f"{name}={declaration.get(name)!r}, expected {value!r}",
        )
    _require(kv == KVContract(), f"KV contract {kv!r}, expected {KVContract()!r}")
    for name, value in REQUIRED_ENV.items():
        _require(env.get(name) == value, f"environment {name}={env.get(name)!r}, expected {value!r}")


@dataclass(frozen=True)
class RepeatFixturePlan:
    fixture_id: str
    scratch_id: str
    shadow_id: str
    seed: int
    epochs: tuple[int, ...]
    poison_before_each: bool = True
    initialized_regions: tuple[str, ...] = ("scratch", "shadow", "outputs")


def repeat_fixture_plan(repeats: int = 3) -> RepeatFixturePlan:
    _require(repeats >= 2, "repeat determinism requires at least two launches")
    return RepeatFixturePlan(
        fixture_id="tp8-p1-shared-fixture",
        scratch_id="scratch-allocation-0",
        shadow_id="shadow-allocation-0",
        seed=1729,
        epochs=tuple(range(41, 41 + repeats)),
    )


def validate_repeat_observations(
    plan: RepeatFixturePlan, observations: Sequence[Mapping[str, Any]]
) -> None:
    """Validate reuse, stale-state poisoning, initialization, and determinism."""
    _require(len(observations) == len(plan.epochs), "incomplete repeat observations")
    semantic_digest = observations[0].get("semantic_digest")
    poison_digests: set[Any] = set()
    for index, (epoch, item) in enumerate(zip(plan.epochs, observations)):
        prefix = f"repeat {index}"
        _require(item.get("fixture_id") == plan.fixture_id, f"{prefix}: fixture rebuilt")
        _require(item.get("scratch_id") == plan.scratch_id, f"{prefix}: scratch rebuilt")
        _require(item.get("shadow_id") == plan.shadow_id, f"{prefix}: shadow rebuilt")
        _require(item.get("seed") == plan.seed, f"{prefix}: seed changed")
        _require(item.get("epoch") == epoch, f"{prefix}: wrong epoch")
        _require(item.get("poisoned") is True, f"{prefix}: stale-state poison missing")
        _require(
            tuple(item.get("initialized_regions", ())) == plan.initialized_regions,
            f"{prefix}: incomplete initialization",
        )
        _require(
            item.get("semantic_digest") == semantic_digest,
            f"{prefix}: nondeterministic semantic output",
        )
        poison_digests.add(item.get("poison_digest"))
    _require(None not in poison_digests, "poison digest missing")
    _require(
        len(poison_digests) == len(observations),
        "each repeat must use distinguishable stale-state poison",
    )


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_INTERMEDIATES = (
    "residual",
    "routing_ids",
    "routing_scores",
    "xin",
    "current_k_cache",
    "current_v_cache",
    "attention_input",
)
NEXT_K1_INTERMEDIATES = (
    "next_q",
    "next_k",
    "next_v",
    "next_gate",
    "next_k_cache",
    "next_v_cache",
)


def whole_layer_reference_schema(variant: LayerVariant) -> dict[str, Any]:
    """Schema an independent reference must attest before GPU comparison."""
    intermediates = list(REFERENCE_INTERMEDIATES)
    if variant.next_k1:
        intermediates.extend(NEXT_K1_INTERMEDIATES)
    return {
        "schema": "tp8-active4-whole-layer-reference",
        "version": REFERENCE_SCHEMA_VERSION,
        "producer_kind": "independent_reference",
        "forbidden_dependencies": ["mono_fused", "mono_split"],
        "variant": asdict(variant),
        "logical_tp": LOGICAL_TP,
        "active_world": ACTIVE_WORLD,
        "required_inputs": [
            "logical_inputs",
            "model_weights",
            "kv_fixture",
            "routing_fixture",
        ],
        "required_intermediates": intermediates,
        "comparison": {
            "routing_ids": "exact",
            "routing_scores": "tolerance",
            "floating_tensors": "declared_rtol_atol",
        },
    }


def validate_reference_manifest(
    manifest: Mapping[str, Any], variant: LayerVariant
) -> None:
    schema = whole_layer_reference_schema(variant)
    for key in ("schema", "version", "producer_kind", "variant", "logical_tp", "active_world"):
        _require(manifest.get(key) == schema[key], f"reference {key} mismatch")
    dependencies = set(manifest.get("dependencies", ()))
    forbidden = dependencies.intersection(schema["forbidden_dependencies"])
    _require(not forbidden, f"reference shares Mono implementation: {sorted(forbidden)}")
    _require(
        set(manifest.get("inputs", ())) == set(schema["required_inputs"]),
        "reference input contract incomplete",
    )
    _require(
        set(manifest.get("intermediates", ())) == set(schema["required_intermediates"]),
        "reference intermediate contract incomplete",
    )
    tolerances = manifest.get("tolerances")
    _require(isinstance(tolerances, Mapping), "reference tolerances missing")
    _require(
        isinstance(tolerances.get("rtol"), (float, int))
        and isinstance(tolerances.get("atol"), (float, int))
        and tolerances["rtol"] >= 0
        and tolerances["atol"] >= 0,
        "reference tolerances invalid",
    )
