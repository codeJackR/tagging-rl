"""CPU-only reward primitives for the GRPO Run 2 candidate family.

Candidates U and UA are composed here one record at a time and share one TRL
batch adapter. Candidate CB uses that same batch path through a factory that
loads its immutable class-weight lookup once. Rollout replay remains deferred.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

from training.audit_data_boundaries import sha256_file
from training.build_cb_class_weights import (
    DEFAULT_OUTPUT as DEFAULT_CB_CLASS_WEIGHT_ARTIFACT,
    NOT_APPLICABLE_CLASS,
    VERSION as CB_CLASS_WEIGHT_ARTIFACT_VERSION,
)
from training.reward_scale_contract import (
    ABSTAIN,
    CLASS_WEIGHT_MAX,
    CLASS_WEIGHT_MIN,
    CORRECT,
    MALFORMED_FLOOR,
    UNKNOWN_ABSTAIN,
    UNKNOWN_COMMIT,
    WRONG,
    class_weight,
    combine_known_unknown,
    details_payoff,
    normalized_mean,
    rule_adjustment,
    valid_total,
)
from training.rewards import _completion_texts, _gold_record, _resolve_pack
from verifier import Pack, verify_record


VERSION = "grpo-run2-candidate-rewards-v12"
CB_CLASS_WEIGHT_ARTIFACT_SHA256 = (
    "7b53323a7f1c170fa68c6b1a0d1356c67fd827f70f466ba2972b857418f4ab37"
)


class _DuplicateJsonKey(ValueError):
    """Internal signal used to stop JSON decoding at the first duplicate key."""


@dataclass(frozen=True)
class SemanticGateResult:
    """Eligibility decision and diagnostics for one literal model completion."""

    eligible: bool
    parsed: dict[str, Any] | None
    errors: tuple[str, ...]
    rule_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnownFieldOutcome:
    """One gold-known field's auditable Candidate U contribution."""

    field_name: str
    outcome: Literal["correct", "abstain", "wrong", "partial"]
    utility: float
    set_f1: float | None = None


@dataclass(frozen=True)
class UniformKnownScore:
    """Candidate U's normalized known-field semantic result."""

    semantic_score: float
    scorable_fields: int
    excluded_gold_unknown_fields: tuple[str, ...]
    field_outcomes: tuple[KnownFieldOutcome, ...]


@dataclass(frozen=True)
class CBClassWeightLookup:
    """Validated, read-only Candidate CB weights prepared once before scoring."""

    artifact_version: str
    field_names: tuple[str, ...]
    unknown_token: str
    weights: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class ClassBalancedFieldOutcome:
    """One gold-known field's utility and auditable Candidate CB weight."""

    field_name: str
    outcome: Literal["correct", "abstain", "wrong", "partial"]
    utility: float
    gold_class_keys: tuple[str, ...]
    gold_class_weights: tuple[float, ...]
    field_weight: float
    set_f1: float | None = None


@dataclass(frozen=True)
class ClassBalancedKnownScore:
    """Candidate CB's normalized class-weighted known-field result."""

    semantic_score: float
    scorable_fields: int
    total_field_weight: float
    excluded_gold_unknown_fields: tuple[str, ...]
    field_outcomes: tuple[ClassBalancedFieldOutcome, ...]


@dataclass(frozen=True)
class UnknownFieldOutcome:
    """One gold-unknown field's auditable Candidate UA/CB contribution."""

    field_name: str
    outcome: Literal["abstain", "commit"]
    utility: float


@dataclass(frozen=True)
class UniformUnknownScore:
    """The separately normalized gold-unknown component shared by UA and CB."""

    semantic_score: float | None
    scorable_fields: int
    excluded_gold_known_fields: tuple[str, ...]
    field_outcomes: tuple[UnknownFieldOutcome, ...]


@dataclass(frozen=True)
class UniformUnknownAwareSemantics:
    """Candidate UA's combined, auditable semantic score."""

    semantic_score: float
    known_semantics: UniformKnownScore
    unknown_semantics: UniformUnknownScore


@dataclass(frozen=True)
class ClassBalancedUnknownAwareSemantics:
    """Candidate CB's combined, auditable semantic score."""

    semantic_score: float
    known_semantics: ClassBalancedKnownScore
    unknown_semantics: UniformUnknownScore


@dataclass(frozen=True)
class CandidateUResult:
    """Auditable end-to-end reward result for one Candidate U completion."""

    candidate: Literal["U"]
    reward: float
    eligible: bool
    gate_errors: tuple[str, ...]
    rule_violations: tuple[str, ...]
    rule_adjustment: float | None
    known_semantics: UniformKnownScore | None


@dataclass(frozen=True)
class CandidateUAResult:
    """Auditable end-to-end reward result for one Candidate UA completion."""

    candidate: Literal["UA"]
    reward: float
    eligible: bool
    gate_errors: tuple[str, ...]
    rule_violations: tuple[str, ...]
    rule_adjustment: float | None
    unknown_aware_semantics: UniformUnknownAwareSemantics | None


@dataclass(frozen=True)
class CandidateCBResult:
    """Auditable end-to-end reward result for one Candidate CB completion."""

    candidate: Literal["CB"]
    reward: float
    eligible: bool
    gate_errors: tuple[str, ...]
    rule_violations: tuple[str, ...]
    rule_adjustment: float | None
    unknown_aware_semantics: ClassBalancedUnknownAwareSemantics | None


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid(*errors: str, parsed: dict[str, Any] | None = None) -> SemanticGateResult:
    return SemanticGateResult(eligible=False, parsed=parsed, errors=tuple(errors))


def _semantic_record_checks(
    record: Mapping[str, Any], pack: Pack
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check one decoded record and return (errors, rule violations)."""
    expected = set(pack.field_names)
    observed = set(record)
    errors: list[str] = []
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"gate: missing fields: {','.join(missing)}")
    if extra:
        errors.append(f"gate: extra fields: {','.join(extra)}")
    if errors:
        return tuple(errors), ()

    verified = verify_record(dict(record), pack)
    if not verified.schema_valid or not verified.vocab_valid:
        return tuple(verified.errors), ()

    for field_name, spec in pack.specs.items():
        if spec.kind != "multi":
            continue
        value = record[field_name]
        if value is None:
            continue
        if not value:
            errors.append(f"gate: {field_name} must not be empty")
            continue
        if len(value) != len(set(value)):
            errors.append(f"gate: {field_name} contains duplicate values")
        if pack.unknown_token in value and value != [pack.unknown_token]:
            errors.append(
                f"gate: {field_name} cannot mix {pack.unknown_token} with commitments"
            )

    return tuple(errors), tuple(verified.rule_violations)


def strict_semantic_gate(raw_output: str, pack: Pack) -> SemanticGateResult:
    """Return whether a completion may receive Run 2 semantic credit.

    The gate is intentionally stricter than the reusable base verifier.  It
    requires a complete record and rejects ambiguous multi-value abstention.
    Rule violations do not fail this gate; valid records retain semantic credit
    and receive their separately bounded coherence cost later.

    A non-string input is an integration bug and raises.  Malformed model text
    is ordinary model behavior and returns an ineligible result.
    """
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")

    try:
        parsed = json.loads(
            raw_output,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKey as exc:
        return _invalid(f"gate: {exc}")
    except json.JSONDecodeError as exc:
        return _invalid(f"gate: not literal JSON: {exc}")

    if not isinstance(parsed, dict):
        return _invalid(
            f"gate: top level must be an object, got {type(parsed).__name__}"
        )

    errors, rule_violations = _semantic_record_checks(parsed, pack)
    if errors:
        return _invalid(*errors, parsed=parsed)

    return SemanticGateResult(
        eligible=True,
        parsed=parsed,
        errors=(),
        rule_violations=rule_violations,
    )


def _is_abstention(value: Any, *, multi: bool, unknown_token: str) -> bool:
    return value == ([unknown_token] if multi else unknown_token)


def _set_f1(predicted: list[str], gold: list[str]) -> float:
    predicted_set = set(predicted)
    gold_set = set(gold)
    overlap = len(predicted_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2.0 * precision * recall / (precision + recall)


def _known_field_outcome(
    field_name: str,
    prediction: Any,
    gold: Any,
    *,
    multi: bool,
    unknown_token: str,
) -> KnownFieldOutcome:
    if _is_abstention(prediction, multi=multi, unknown_token=unknown_token):
        return KnownFieldOutcome(field_name, "abstain", ABSTAIN)

    if multi and isinstance(gold, list) and isinstance(prediction, list):
        quality = _set_f1(prediction, gold)
        utility = details_payoff(quality)
        if quality == 0.0:
            outcome = "wrong"
        elif quality == 1.0:
            outcome = "correct"
        else:
            outcome = "partial"
        return KnownFieldOutcome(field_name, outcome, utility, quality)

    if prediction == gold:
        return KnownFieldOutcome(field_name, "correct", CORRECT)
    return KnownFieldOutcome(field_name, "wrong", WRONG)


def _require_known_gold(gold: Mapping[str, Any], pack: Pack) -> None:
    if not isinstance(gold, Mapping):
        raise TypeError("gold must be a mapping")
    errors, _ = _semantic_record_checks(gold, pack)
    if errors:
        raise ValueError(
            "gold must satisfy the complete record contract: " + "; ".join(errors)
        )
    known_fields = [
        field_name
        for field_name, spec in pack.specs.items()
        if not _is_abstention(
            gold[field_name],
            multi=spec.kind == "multi",
            unknown_token=pack.unknown_token,
        )
    ]
    if not known_fields:
        raise ValueError("known-field semantics require at least one gold-known field")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def prepare_cb_class_weight_lookup(
    artifact: Mapping[str, Any], pack: Pack
) -> CBClassWeightLookup:
    """Validate one CB artifact and freeze its hot-path weight lookup.

    This is intentionally a one-time boundary operation. It independently
    verifies the artifact's class inventory, medians, support formula, clipping
    states and pack compatibility before any completion can be scored.
    """
    artifact = _require_mapping(artifact, "CB class-weight artifact")
    if artifact.get("version") != CB_CLASS_WEIGHT_ARTIFACT_VERSION:
        raise ValueError(
            "CB class-weight artifact version mismatch: "
            f"expected {CB_CLASS_WEIGHT_ARTIFACT_VERSION!r}, "
            f"got {artifact.get('version')!r}"
        )
    if artifact.get("status") != "passed":
        raise ValueError("CB class-weight artifact status must be 'passed'")

    selection_boundary = _require_mapping(
        artifact.get("selection_boundary"), "CB selection boundary"
    )
    if selection_boundary.get("candidate_completion_rewards_calculated") is not False:
        raise ValueError("CB artifact must precede candidate completion scoring")
    invariants = _require_mapping(artifact.get("invariants"), "CB invariants")
    if not invariants or any(value is not True for value in invariants.values()):
        raise ValueError("CB class-weight artifact contains a failed invariant")

    contract = _require_mapping(artifact.get("weight_contract"), "CB weight contract")
    if _require_finite_number(contract.get("minimum"), "CB minimum weight") != CLASS_WEIGHT_MIN:
        raise ValueError("CB minimum weight differs from the locked scale")
    if _require_finite_number(contract.get("maximum"), "CB maximum weight") != CLASS_WEIGHT_MAX:
        raise ValueError("CB maximum weight differs from the locked scale")
    if contract.get("unknown_handling") != "excluded":
        raise ValueError("CB weight contract must exclude gold unknown")
    if contract.get("not_applicable_key") != NOT_APPLICABLE_CLASS:
        raise ValueError("CB not-applicable key differs from the locked contract")
    if contract.get("multi_label_field_weight_later") != "mean of gold-label weights":
        raise ValueError("CB multi-label weighting differs from the locked contract")

    weight_map = _require_mapping(artifact.get("weight_map"), "CB weight map")
    attributes = _require_mapping(weight_map.get("attributes"), "CB attributes")
    expected_fields = set(pack.field_names)
    observed_fields = set(attributes)
    if observed_fields != expected_fields:
        missing = sorted(expected_fields - observed_fields)
        extra = sorted(observed_fields - expected_fields)
        raise ValueError(
            f"CB attributes differ from pack fields: missing={missing}, extra={extra}"
        )

    frozen_weights: dict[str, Mapping[str, float]] = {}
    all_supports: list[int] = []
    all_weights: list[float] = []
    clipping_counts = {"minimum": 0, "maximum": 0, "unclipped": 0}
    for field_name, spec in pack.specs.items():
        attribute = _require_mapping(
            attributes[field_name], f"CB attribute {field_name}"
        )
        classes = _require_mapping(
            attribute.get("classes"), f"CB classes for {field_name}"
        )
        if not classes:
            raise ValueError(f"CB attribute {field_name} has no observed classes")
        if attribute.get("observed_classes") != len(classes):
            raise ValueError(f"CB observed-class count mismatch for {field_name}")
        invalid_classes = sorted(
            set(classes) - (set(spec.values) | {NOT_APPLICABLE_CLASS})
        )
        if invalid_classes:
            raise ValueError(
                f"CB {field_name} contains classes outside the pack: {invalid_classes}"
            )
        if pack.unknown_token in classes:
            raise ValueError(f"CB {field_name} must not contain the unknown token")

        entries: list[tuple[str, Mapping[str, Any], int]] = []
        for class_name in sorted(classes):
            entry = _require_mapping(
                classes[class_name], f"CB entry {field_name}/{class_name}"
            )
            support = entry.get("support")
            if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
                raise ValueError(
                    f"CB support must be a positive integer for {field_name}/{class_name}"
                )
            entries.append((class_name, entry, support))

        median_support = statistics.median(support for _, _, support in entries)
        reported_median = _require_finite_number(
            attribute.get("median_positive_support"),
            f"CB median support for {field_name}",
        )
        if reported_median != median_support:
            raise ValueError(f"CB median support mismatch for {field_name}")

        field_weights: dict[str, float] = {}
        for class_name, entry, support in entries:
            raw_weight = math.sqrt(median_support / support)
            reported_raw = _require_finite_number(
                entry.get("raw_weight"), f"CB raw weight {field_name}/{class_name}"
            )
            if not math.isclose(reported_raw, raw_weight, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"CB raw weight mismatch for {field_name}/{class_name}")
            expected_weight = class_weight(support, median_support)
            reported_weight = _require_finite_number(
                entry.get("weight"), f"CB weight {field_name}/{class_name}"
            )
            if not math.isclose(
                reported_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"CB clipped weight mismatch for {field_name}/{class_name}")
            expected_clipping = (
                "minimum"
                if raw_weight < CLASS_WEIGHT_MIN
                else "maximum"
                if raw_weight > CLASS_WEIGHT_MAX
                else None
            )
            if entry.get("clipped_at") != expected_clipping:
                raise ValueError(f"CB clipping state mismatch for {field_name}/{class_name}")
            clipping_counts[expected_clipping or "unclipped"] += 1
            all_supports.append(support)
            all_weights.append(reported_weight)
            field_weights[class_name] = reported_weight
        frozen_weights[field_name] = MappingProxyType(field_weights)

    if weight_map.get("observed_attribute_class_pairs") != len(all_supports):
        raise ValueError("CB global attribute/class count mismatch")
    if weight_map.get("clipping_counts") != clipping_counts:
        raise ValueError("CB global clipping counts mismatch")
    if weight_map.get("minimum_support") != min(all_supports):
        raise ValueError("CB global minimum support mismatch")
    if weight_map.get("maximum_support") != max(all_supports):
        raise ValueError("CB global maximum support mismatch")
    if not math.isclose(
        _require_finite_number(
            weight_map.get("minimum_derived_weight"), "CB minimum derived weight"
        ),
        min(all_weights),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("CB global minimum weight mismatch")
    if not math.isclose(
        _require_finite_number(
            weight_map.get("maximum_derived_weight"), "CB maximum derived weight"
        ),
        max(all_weights),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("CB global maximum weight mismatch")

    return CBClassWeightLookup(
        artifact_version=CB_CLASS_WEIGHT_ARTIFACT_VERSION,
        field_names=pack.field_names,
        unknown_token=pack.unknown_token,
        weights=MappingProxyType(frozen_weights),
    )


def load_cb_class_weight_lookup(
    pack: Pack,
    artifact_path: str | Path | None = None,
) -> CBClassWeightLookup:
    """Hash-check and prepare the immutable published Candidate CB artifact."""
    path = (
        Path(artifact_path)
        if artifact_path is not None
        else Path(__file__).resolve().parent.parent / DEFAULT_CB_CLASS_WEIGHT_ARTIFACT
    ).resolve()
    observed_hash = sha256_file(path)
    if observed_hash != CB_CLASS_WEIGHT_ARTIFACT_SHA256:
        raise ValueError(
            "CB class-weight artifact hash mismatch: "
            f"expected {CB_CLASS_WEIGHT_ARTIFACT_SHA256}, got {observed_hash}"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CB class-weight artifact is not valid JSON: {exc}") from exc
    return prepare_cb_class_weight_lookup(artifact, pack)


def _known_gold_class_keys(gold_value: Any, *, multi: bool) -> tuple[str, ...]:
    if gold_value is None:
        return (NOT_APPLICABLE_CLASS,)
    if multi:
        return tuple(gold_value)
    return (gold_value,)


def _require_cb_lookup_matches_pack(
    class_weights: CBClassWeightLookup,
    pack: Pack,
) -> None:
    if not isinstance(class_weights, CBClassWeightLookup):
        raise TypeError("class_weights must be a prepared CBClassWeightLookup")
    if (
        class_weights.field_names != pack.field_names
        or class_weights.unknown_token != pack.unknown_token
    ):
        raise ValueError("CB class-weight lookup does not match the active pack")


def _class_balanced_gold_weights(
    gold: Mapping[str, Any],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> dict[str, tuple[tuple[str, ...], tuple[float, ...], float]]:
    """Resolve every gold-known field's classes and one field-level weight."""
    _require_cb_lookup_matches_pack(class_weights, pack)
    resolved: dict[str, tuple[tuple[str, ...], tuple[float, ...], float]] = {}
    for field_name, spec in pack.specs.items():
        gold_value = gold[field_name]
        multi = spec.kind == "multi"
        if _is_abstention(
            gold_value,
            multi=multi,
            unknown_token=pack.unknown_token,
        ):
            continue
        class_keys = _known_gold_class_keys(gold_value, multi=multi)
        try:
            gold_weights = tuple(
                class_weights.weights[field_name][class_name]
                for class_name in class_keys
            )
        except KeyError as exc:
            missing_class = exc.args[0]
            raise ValueError(
                f"CB lookup has no weight for {field_name}/{missing_class}"
            ) from exc
        resolved[field_name] = (
            class_keys,
            gold_weights,
            sum(gold_weights) / len(gold_weights),
        )
    return resolved


def score_uniform_known_fields(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    pack: Pack,
) -> UniformKnownScore:
    """Score Candidate U semantics over gold-known fields only.

    Both mappings must satisfy the strict complete-record contract.  Gold
    ``unknown`` fields are recorded but excluded.  The active Run 2 pool has at
    least two known fields per product; an all-unknown gold record therefore
    fails closed instead of inventing an arbitrary average.
    """
    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping")

    prediction_errors, _ = _semantic_record_checks(prediction, pack)
    if prediction_errors:
        raise ValueError(
            "prediction must pass the strict semantic gate: "
            + "; ".join(prediction_errors)
        )
    _require_known_gold(gold, pack)

    outcomes: list[KnownFieldOutcome] = []
    excluded: list[str] = []
    for field_name, spec in pack.specs.items():
        gold_value = gold[field_name]
        multi = spec.kind == "multi"
        if _is_abstention(
            gold_value,
            multi=multi,
            unknown_token=pack.unknown_token,
        ):
            excluded.append(field_name)
            continue
        outcomes.append(
            _known_field_outcome(
                field_name,
                prediction[field_name],
                gold_value,
                multi=multi,
                unknown_token=pack.unknown_token,
            )
        )

    score = normalized_mean([outcome.utility for outcome in outcomes])
    return UniformKnownScore(
        semantic_score=score,
        scorable_fields=len(outcomes),
        excluded_gold_unknown_fields=tuple(excluded),
        field_outcomes=tuple(outcomes),
    )


def score_class_balanced_known_fields(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> ClassBalancedKnownScore:
    """Score Candidate CB's weighted semantics over gold-known fields only.

    Field utilities are exactly Candidate U's utilities. Only aggregation
    changes: each scalar/N/A field uses its gold-class weight, multi-label
    ``details`` uses the mean of its gold-label weights, and the final semantic
    score is normalized by the sum of field weights.
    """
    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping")

    prediction_errors, _ = _semantic_record_checks(prediction, pack)
    if prediction_errors:
        raise ValueError(
            "prediction must pass the strict semantic gate: "
            + "; ".join(prediction_errors)
        )
    _require_known_gold(gold, pack)
    gold_weights_by_field = _class_balanced_gold_weights(gold, pack, class_weights)

    outcomes: list[ClassBalancedFieldOutcome] = []
    excluded: list[str] = []
    for field_name, spec in pack.specs.items():
        gold_value = gold[field_name]
        multi = spec.kind == "multi"
        if _is_abstention(
            gold_value,
            multi=multi,
            unknown_token=pack.unknown_token,
        ):
            excluded.append(field_name)
            continue

        base = _known_field_outcome(
            field_name,
            prediction[field_name],
            gold_value,
            multi=multi,
            unknown_token=pack.unknown_token,
        )
        class_keys, gold_weights, field_weight = gold_weights_by_field[field_name]
        outcomes.append(
            ClassBalancedFieldOutcome(
                field_name=field_name,
                outcome=base.outcome,
                utility=base.utility,
                gold_class_keys=class_keys,
                gold_class_weights=gold_weights,
                field_weight=field_weight,
                set_f1=base.set_f1,
            )
        )

    weights = [outcome.field_weight for outcome in outcomes]
    score = normalized_mean(
        [outcome.utility for outcome in outcomes],
        weights=weights,
    )
    return ClassBalancedKnownScore(
        semantic_score=score,
        scorable_fields=len(outcomes),
        total_field_weight=sum(weights),
        excluded_gold_unknown_fields=tuple(excluded),
        field_outcomes=tuple(outcomes),
    )


def score_uniform_unknown_fields(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    pack: Pack,
) -> UniformUnknownScore:
    """Score Candidate UA/CB behavior on gold-unknown fields only.

    Honest shape-correct abstention receives ``+1`` and any scalar, null or
    multi-value commitment receives ``-1``.  Known gold fields are excluded.
    If no gold-unknown field exists, the component is absent and returns
    ``semantic_score=None`` so the later combiner can renormalize to known
    semantics instead of treating absence as a measured zero.
    """
    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping")
    if not isinstance(gold, Mapping):
        raise TypeError("gold must be a mapping")

    prediction_errors, _ = _semantic_record_checks(prediction, pack)
    if prediction_errors:
        raise ValueError(
            "prediction must pass the strict semantic gate: "
            + "; ".join(prediction_errors)
        )
    gold_errors, _ = _semantic_record_checks(gold, pack)
    if gold_errors:
        raise ValueError(
            "gold must satisfy the complete record contract: " + "; ".join(gold_errors)
        )

    outcomes: list[UnknownFieldOutcome] = []
    excluded: list[str] = []
    for field_name, spec in pack.specs.items():
        multi = spec.kind == "multi"
        if not _is_abstention(
            gold[field_name],
            multi=multi,
            unknown_token=pack.unknown_token,
        ):
            excluded.append(field_name)
            continue
        prediction_abstains = _is_abstention(
            prediction[field_name],
            multi=multi,
            unknown_token=pack.unknown_token,
        )
        outcomes.append(
            UnknownFieldOutcome(
                field_name=field_name,
                outcome="abstain" if prediction_abstains else "commit",
                utility=UNKNOWN_ABSTAIN if prediction_abstains else UNKNOWN_COMMIT,
            )
        )

    score = (
        normalized_mean([outcome.utility for outcome in outcomes])
        if outcomes
        else None
    )
    return UniformUnknownScore(
        semantic_score=score,
        scorable_fields=len(outcomes),
        excluded_gold_known_fields=tuple(excluded),
        field_outcomes=tuple(outcomes),
    )


def score_uniform_unknown_aware_semantics(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    pack: Pack,
) -> UniformUnknownAwareSemantics:
    """Combine Candidate UA's separately normalized semantic populations.

    The fixed mix weights come from the locked active training pool.  When no
    gold-unknown fields exist, ``combine_known_unknown`` returns the known score
    unchanged instead of shrinking it by an absent component.
    """
    known = score_uniform_known_fields(prediction, gold, pack)
    unknown = score_uniform_unknown_fields(prediction, gold, pack)
    combined = combine_known_unknown(known.semantic_score, unknown.semantic_score)
    return UniformUnknownAwareSemantics(
        semantic_score=combined,
        known_semantics=known,
        unknown_semantics=unknown,
    )


def score_class_balanced_unknown_aware_semantics(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> ClassBalancedUnknownAwareSemantics:
    """Combine Candidate CB's weighted known and shared unknown components.

    The same fixed active-pool population weights used by Candidate UA are used
    here. If a product has no gold-unknown fields, ``combine_known_unknown``
    returns the CB known score unchanged rather than shrinking it by an absent
    component.
    """
    known = score_class_balanced_known_fields(
        prediction,
        gold,
        pack,
        class_weights,
    )
    unknown = score_uniform_unknown_fields(prediction, gold, pack)
    combined = combine_known_unknown(known.semantic_score, unknown.semantic_score)
    return ClassBalancedUnknownAwareSemantics(
        semantic_score=combined,
        known_semantics=known,
        unknown_semantics=unknown,
    )


def score_candidate_u(
    raw_output: str,
    gold: Mapping[str, Any],
    pack: Pack,
) -> CandidateUResult:
    """Compose Candidate U for one literal completion.

    Trusted gold is validated first so corrupted training data cannot hide
    behind a malformed model completion.  Ineligible model output receives only
    the fixed floor.  Eligible output receives the uniform known-field semantic
    score plus the bounded per-rule adjustment.
    """
    _require_known_gold(gold, pack)
    gate = strict_semantic_gate(raw_output, pack)
    if not gate.eligible:
        return CandidateUResult(
            candidate="U",
            reward=MALFORMED_FLOOR,
            eligible=False,
            gate_errors=gate.errors,
            rule_violations=(),
            rule_adjustment=None,
            known_semantics=None,
        )

    if gate.parsed is None:  # Defensive invariant; eligible always has a record.
        raise RuntimeError("eligible semantic-gate result has no parsed record")
    semantics = score_uniform_known_fields(gate.parsed, gold, pack)
    coherence = rule_adjustment(len(gate.rule_violations))
    reward = valid_total(semantics.semantic_score, len(gate.rule_violations))
    return CandidateUResult(
        candidate="U",
        reward=reward,
        eligible=True,
        gate_errors=(),
        rule_violations=gate.rule_violations,
        rule_adjustment=coherence,
        known_semantics=semantics,
    )


def score_candidate_ua(
    raw_output: str,
    gold: Mapping[str, Any],
    pack: Pack,
) -> CandidateUAResult:
    """Compose Candidate UA for one literal completion.

    Trusted gold is validated before model text.  Ineligible output receives
    only the fixed floor.  Eligible output receives the combined unknown-aware
    semantic score plus the same bounded rule adjustment used by Candidate U.
    """
    _require_known_gold(gold, pack)
    gate = strict_semantic_gate(raw_output, pack)
    if not gate.eligible:
        return CandidateUAResult(
            candidate="UA",
            reward=MALFORMED_FLOOR,
            eligible=False,
            gate_errors=gate.errors,
            rule_violations=(),
            rule_adjustment=None,
            unknown_aware_semantics=None,
        )

    if gate.parsed is None:  # Defensive invariant; eligible always has a record.
        raise RuntimeError("eligible semantic-gate result has no parsed record")
    semantics = score_uniform_unknown_aware_semantics(gate.parsed, gold, pack)
    coherence = rule_adjustment(len(gate.rule_violations))
    reward = valid_total(semantics.semantic_score, len(gate.rule_violations))
    return CandidateUAResult(
        candidate="UA",
        reward=reward,
        eligible=True,
        gate_errors=(),
        rule_violations=gate.rule_violations,
        rule_adjustment=coherence,
        unknown_aware_semantics=semantics,
    )


def score_candidate_cb(
    raw_output: str,
    gold: Mapping[str, Any],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> CandidateCBResult:
    """Compose Candidate CB for one literal completion.

    Trusted gold and its class-weight coverage are validated before model text.
    Ineligible output receives only the fixed malformed floor. Eligible output
    receives the completed class-balanced unknown-aware semantics plus the same
    bounded rule adjustment used by Candidates U and UA.
    """
    _require_known_gold(gold, pack)
    _class_balanced_gold_weights(gold, pack, class_weights)
    gate = strict_semantic_gate(raw_output, pack)
    if not gate.eligible:
        return CandidateCBResult(
            candidate="CB",
            reward=MALFORMED_FLOOR,
            eligible=False,
            gate_errors=gate.errors,
            rule_violations=(),
            rule_adjustment=None,
            unknown_aware_semantics=None,
        )

    if gate.parsed is None:  # Defensive invariant; eligible always has a record.
        raise RuntimeError("eligible semantic-gate result has no parsed record")
    semantics = score_class_balanced_unknown_aware_semantics(
        gate.parsed,
        gold,
        pack,
        class_weights,
    )
    coherence = rule_adjustment(len(gate.rule_violations))
    reward = valid_total(semantics.semantic_score, len(gate.rule_violations))
    return CandidateCBResult(
        candidate="CB",
        reward=reward,
        eligible=True,
        gate_errors=(),
        rule_violations=gate.rule_violations,
        rule_adjustment=coherence,
        unknown_aware_semantics=semantics,
    )


def _candidate_reward_batch(
    completions: Sequence[Any],
    gold: Sequence[Any],
    *,
    pack: Pack | None,
    scorer: Callable[
        [str, Mapping[str, Any], Pack],
        CandidateUResult | CandidateUAResult | CandidateCBResult,
    ],
    gold_validator: Callable[[Mapping[str, Any], Pack], None] | None = None,
) -> list[float]:
    """Validate one aligned batch and delegate every pair to one scorer."""
    texts = _completion_texts(completions)
    if isinstance(gold, (str, bytes)):
        raise TypeError("gold must be a sequence aligned with completions")
    if len(gold) != len(texts):
        raise ValueError(
            "gold and completions must have the same length: "
            f"got {len(gold)} and {len(texts)}"
        )

    resolved_pack = _resolve_pack(pack)
    gold_records = [
        _gold_record(answer, index=index) for index, answer in enumerate(gold)
    ]
    for record in gold_records:
        _require_known_gold(record, resolved_pack)
        if gold_validator is not None:
            gold_validator(record, resolved_pack)

    return [
        scorer(text, answer, resolved_pack).reward
        for text, answer in zip(texts, gold_records, strict=True)
    ]


def candidate_u_reward(
    completions: Sequence[Any],
    gold: Sequence[Any],
    *,
    pack: Pack | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible batched Candidate U reward function.

    The adapter accepts TRL's plain-text or one-assistant-message completion
    shape and dict or JSON-string gold.  It validates alignment and the entire
    trusted gold batch before delegating each pair to ``score_candidate_u``.
    Extra trainer columns are accepted and ignored, matching TRL's reward-call
    convention without importing TRL itself.
    """
    return _candidate_reward_batch(
        completions,
        gold,
        pack=pack,
        scorer=score_candidate_u,
    )


def candidate_ua_reward(
    completions: Sequence[Any],
    gold: Sequence[Any],
    *,
    pack: Pack | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible batched Candidate UA reward function.

    All representation, alignment and trusted-gold validation is shared with
    Candidate U.  This wrapper adds no reward math and delegates every aligned
    pair to ``score_candidate_ua``.
    """
    return _candidate_reward_batch(
        completions,
        gold,
        pack=pack,
        scorer=score_candidate_ua,
    )


def make_candidate_cb_reward(
    *,
    pack: Pack | None = None,
    artifact_path: str | Path | None = None,
) -> Callable[..., list[float]]:
    """Construct one TRL-compatible Candidate CB batch reward callable.

    Pack resolution plus artifact hash and invariant validation happen once at
    construction. The returned adapter shares U/UA's representation, alignment
    and gold-first batch checks, adds CB gold-weight coverage validation, and
    delegates all reward math to ``score_candidate_cb``.
    """
    resolved_pack = _resolve_pack(pack)
    class_weights = load_cb_class_weight_lookup(resolved_pack, artifact_path)

    def validate_cb_gold(record: Mapping[str, Any], active_pack: Pack) -> None:
        _class_balanced_gold_weights(record, active_pack, class_weights)

    def score_cb(
        raw_output: str,
        gold_record: Mapping[str, Any],
        active_pack: Pack,
    ) -> CandidateCBResult:
        return score_candidate_cb(
            raw_output,
            gold_record,
            active_pack,
            class_weights,
        )

    def candidate_cb_reward(
        completions: Sequence[Any],
        gold: Sequence[Any],
        **_: Any,
    ) -> list[float]:
        return _candidate_reward_batch(
            completions,
            gold,
            pack=resolved_pack,
            scorer=score_cb,
            gold_validator=validate_cb_gold,
        )

    candidate_cb_reward.__name__ = "candidate_cb_reward"
    candidate_cb_reward.__qualname__ = "candidate_cb_reward"
    candidate_cb_reward.__doc__ = (
        "TRL-compatible Candidate CB reward bound to one validated class map."
    )
    return candidate_cb_reward
