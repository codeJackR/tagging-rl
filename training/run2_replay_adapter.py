#!/usr/bin/env python3
"""Validate and adapt one GRPO Run 2 replay group without reading files.

The adapter converts the nested raw-ledger schema into one group-level analyzer
observation plus separate field- and class-contribution child records. Keeping
those grains separate prevents multi-label class rows from multiplying rollout
or product counts. This module deliberately contains no file-I/O entry point.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from training.analyze_run2_candidates import GroupObservation, REWARD_LABELS
from training.reward_scale_contract import (
    KNOWN_MIX_WEIGHT,
    MALFORMED_FLOOR,
    UNKNOWN_MIX_WEIGHT,
)
from training.run2_comparison_contract import EXPECTED_GROUP_SIZE


VERSION = "grpo-run2-replay-ledger-adapter-v1"
EXPECTED_CANDIDATES = ("U", "UA", "CB")
EXPECTED_FIELDS = 15
CLASS_WEIGHT_VERSION = "grpo-run2-cb-class-weights-v1"


@dataclass(frozen=True)
class ClassSupportEntry:
    field_name: str
    class_key: str
    support: int
    weight: float
    support_band: str


@dataclass(frozen=True)
class GroupSegments:
    product_category: str
    difficulty_band: str
    gold_known_field_count: int


@dataclass(frozen=True)
class FieldContribution:
    group_id: str
    rollout_index: int
    candidate: str
    field_name: str
    component: str
    utility: float
    signed_contribution: float
    absolute_contribution: float


@dataclass(frozen=True)
class ClassContribution:
    group_id: str
    rollout_index: int
    candidate: str
    field_name: str
    class_key: str
    class_support: int
    class_support_band: str
    class_weight: float
    absolute_contribution: float


@dataclass(frozen=True)
class AdaptedReplayGroup:
    version: str
    group_position: int
    observation: GroupObservation
    segments: GroupSegments
    field_contributions: tuple[FieldContribution, ...]
    class_contributions: tuple[ClassContribution, ...]
    boundary: Mapping[str, bool]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def class_support_band(support: int) -> str:
    """Map locked training support to predeclared, non-outcome-based bands."""
    support = _integer(support, "class support", minimum=1)
    if support < 5:
        return "rare_1_4"
    if support < 10:
        return "low_5_9"
    if support < 50:
        return "medium_10_49"
    return "common_50_plus"


def difficulty_band(pass_rate: float) -> str:
    """Band the fixed k=8 starting-policy pass rate, including full scope."""
    pass_rate = _finite(pass_rate, "difficulty pass rate")
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("difficulty pass rate must lie in [0, 1]")
    scaled = pass_rate * EXPECTED_GROUP_SIZE
    if not _close(scaled, round(scaled)):
        raise ValueError("difficulty pass rate must be a multiple of 1/8")
    if pass_rate == 0.0:
        return "always_failed"
    if pass_rate <= 0.25:
        return "low_mixed_1_2_of_8"
    if pass_rate < 0.75:
        return "middle_mixed_3_5_of_8"
    if pass_rate < 1.0:
        return "high_mixed_6_7_of_8"
    return "always_passed"


def build_class_support_lookup(
    artifact: Mapping[str, Any],
) -> dict[tuple[str, str], ClassSupportEntry]:
    """Validate an in-memory CB artifact and index each attribute/class once."""
    artifact = _mapping(artifact, "class-weight artifact")
    if artifact.get("version") != CLASS_WEIGHT_VERSION:
        raise ValueError("unexpected class-weight artifact version")
    weight_map = _mapping(artifact.get("weight_map"), "weight_map")
    attributes = _mapping(weight_map.get("attributes"), "weight_map.attributes")
    if not attributes:
        raise ValueError("class-weight artifact has no attributes")
    lookup: dict[tuple[str, str], ClassSupportEntry] = {}
    for field_name, raw_attribute in attributes.items():
        _string(field_name, "class-weight field name")
        attribute = _mapping(raw_attribute, f"weight_map.attributes.{field_name}")
        classes = _mapping(attribute.get("classes"), f"{field_name}.classes")
        if not classes:
            raise ValueError(f"{field_name}: class-weight map is empty")
        for class_key, raw_entry in classes.items():
            _string(class_key, f"{field_name} class key")
            entry = _mapping(raw_entry, f"{field_name}.{class_key}")
            support = _integer(
                entry.get("support"), f"{field_name}.{class_key}.support", minimum=1
            )
            weight = _finite(entry.get("weight"), f"{field_name}.{class_key}.weight")
            if weight <= 0.0:
                raise ValueError(f"{field_name}.{class_key}.weight must be positive")
            key = (field_name, class_key)
            if key in lookup:
                raise ValueError(f"duplicate class support key: {key}")
            lookup[key] = ClassSupportEntry(
                field_name=field_name,
                class_key=class_key,
                support=support,
                weight=weight,
                support_band=class_support_band(support),
            )
    expected_pairs = weight_map.get("observed_attribute_class_pairs")
    if expected_pairs is not None and expected_pairs != len(lookup):
        raise ValueError("class-weight pair count does not match the weight map")
    return lookup


def _product_category(gold_record: Mapping[str, Any]) -> str:
    value = gold_record.get("garment_category")
    if value is None:
        return "__not_applicable__"
    value = _string(value, "gold garment_category")
    return "__gold_unknown__" if value == "unknown" else value


def _source_targets(
    source: Mapping[str, Any], *, group_id: str, known_fields: int, unknown_fields: int
) -> dict[str, float | None]:
    source_sku = _string(source.get("sku_id"), f"{group_id}.source_rollout.sku_id")
    if source_sku != group_id:
        raise ValueError(f"{group_id}: source rollout SKU differs from group")
    scorable = _integer(
        source.get("scorable_labels"), f"{group_id}.scorable_labels", minimum=1
    )
    if scorable != known_fields:
        raise ValueError(f"{group_id}: source scorable-label denominator drifted")
    correct = _integer(source.get("correct_labels"), f"{group_id}.correct_labels")
    if correct > scorable:
        raise ValueError(f"{group_id}: correct labels exceed scorable labels")
    false_abstentions = tuple(
        _string(value, f"{group_id}.false_abstention_labels")
        for value in _sequence(
            source.get("false_abstention_labels"),
            f"{group_id}.false_abstention_labels",
        )
    )
    if len(set(false_abstentions)) != len(false_abstentions):
        raise ValueError(f"{group_id}: duplicate false-abstention field")
    committed = scorable - len(false_abstentions)
    if committed < 0 or correct > committed:
        raise ValueError(f"{group_id}: invalid known coverage/correctness accounting")
    excluded_unknown = tuple(
        _string(value, f"{group_id}.excluded_gold_unknown_labels")
        for value in _sequence(
            source.get("excluded_gold_unknown_labels"),
            f"{group_id}.excluded_gold_unknown_labels",
        )
    )
    correct_unknown = tuple(
        _string(value, f"{group_id}.correct_abstention_labels")
        for value in _sequence(
            source.get("correct_abstention_labels"),
            f"{group_id}.correct_abstention_labels",
        )
    )
    if len(set(excluded_unknown)) != len(excluded_unknown):
        raise ValueError(f"{group_id}: duplicate gold-unknown field")
    if len(set(correct_unknown)) != len(correct_unknown):
        raise ValueError(f"{group_id}: duplicate correct-abstention field")
    if len(excluded_unknown) != unknown_fields:
        raise ValueError(f"{group_id}: source unknown-label denominator drifted")
    if not set(correct_unknown) <= set(excluded_unknown):
        raise ValueError(f"{group_id}: correct abstentions are not gold-unknown fields")
    rules = _sequence(source.get("rule_violations"), f"{group_id}.rule_violations")
    return {
        "known_exact_rate": correct / scorable,
        "known_coverage": committed / scorable,
        "selective_correctness": correct / committed if committed else None,
        "unknown_abstention_rate": (
            len(correct_unknown) / unknown_fields if unknown_fields else None
        ),
        "rule_quality": -float(len(rules)),
    }


def _field_outcomes(
    score: Mapping[str, Any], name: str, *, expected_count: int
) -> tuple[Mapping[str, Any], ...]:
    scorable = _integer(score.get("scorable_fields"), f"{name}.scorable_fields")
    if scorable != expected_count:
        raise ValueError(f"{name}: saved scorable-field denominator drifted")
    outcomes = tuple(
        _mapping(value, f"{name}.field_outcomes")
        for value in _sequence(score.get("field_outcomes"), f"{name}.field_outcomes")
    )
    if len(outcomes) != expected_count:
        raise ValueError(f"{name}: field outcome count differs from gold denominator")
    names = [_string(value.get("field_name"), f"{name}.field_name") for value in outcomes]
    if len(set(names)) != len(names):
        raise ValueError(f"{name}: duplicate field outcome")
    return outcomes


def _validate_shared_known_outcomes(
    u_outcomes: Sequence[Mapping[str, Any]],
    other_outcomes: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    u = [
        (
            outcome["field_name"],
            _finite(outcome.get("utility"), f"U.{outcome['field_name']}.utility"),
        )
        for outcome in u_outcomes
    ]
    other = [
        (
            outcome["field_name"],
            _finite(outcome.get("utility"), f"{name}.{outcome['field_name']}.utility"),
        )
        for outcome in other_outcomes
    ]
    if len(u) != len(other) or any(
        left_name != right_name or not _close(left_value, right_value)
        for (left_name, left_value), (right_name, right_value) in zip(u, other, strict=True)
    ):
        raise ValueError(f"{name}: known field utility ledger drifted from U")


def _field_contribution(
    *,
    group_id: str,
    rollout_index: int,
    candidate: str,
    outcome: Mapping[str, Any],
    component: str,
    multiplier: float,
) -> FieldContribution:
    field_name = _string(outcome.get("field_name"), f"{candidate}.field_name")
    utility = _finite(outcome.get("utility"), f"{candidate}.{field_name}.utility")
    contribution = utility * multiplier
    return FieldContribution(
        group_id=group_id,
        rollout_index=rollout_index,
        candidate=candidate,
        field_name=field_name,
        component=component,
        utility=utility,
        signed_contribution=contribution,
        absolute_contribution=abs(contribution),
    )


def _eligible_ledgers(
    *,
    group_id: str,
    rollout_index: int,
    candidates: Mapping[str, Mapping[str, Any]],
    known_fields: int,
    unknown_fields: int,
    class_supports: Mapping[tuple[str, str], ClassSupportEntry],
) -> tuple[float, float, float, list[FieldContribution], list[ClassContribution]]:
    u = candidates["U"]
    ua = candidates["UA"]
    cb = candidates["CB"]
    u_known = _mapping(u.get("known_semantics"), "U.known_semantics")
    ua_semantics = _mapping(
        ua.get("unknown_aware_semantics"), "UA.unknown_aware_semantics"
    )
    ua_known = _mapping(ua_semantics.get("known_semantics"), "UA.known_semantics")
    ua_unknown = _mapping(
        ua_semantics.get("unknown_semantics"), "UA.unknown_semantics"
    )
    cb_semantics = _mapping(
        cb.get("unknown_aware_semantics"), "CB.unknown_aware_semantics"
    )
    cb_known = _mapping(cb_semantics.get("known_semantics"), "CB.known_semantics")
    cb_unknown = _mapping(
        cb_semantics.get("unknown_semantics"), "CB.unknown_semantics"
    )

    u_outcomes = _field_outcomes(u_known, "U.known_semantics", expected_count=known_fields)
    ua_outcomes = _field_outcomes(ua_known, "UA.known_semantics", expected_count=known_fields)
    cb_outcomes = _field_outcomes(cb_known, "CB.known_semantics", expected_count=known_fields)
    _validate_shared_known_outcomes(u_outcomes, ua_outcomes, "UA")
    _validate_shared_known_outcomes(u_outcomes, cb_outcomes, "CB")
    ua_unknown_outcomes = _field_outcomes(
        ua_unknown, "UA.unknown_semantics", expected_count=unknown_fields
    )
    cb_unknown_outcomes = _field_outcomes(
        cb_unknown, "CB.unknown_semantics", expected_count=unknown_fields
    )
    _validate_shared_known_outcomes(ua_unknown_outcomes, cb_unknown_outcomes, "CB unknown")

    u_semantic = _finite(u_known.get("semantic_score"), "U semantic score")
    ua_semantic = _finite(ua_semantics.get("semantic_score"), "UA semantic score")
    cb_known_semantic = _finite(
        cb_known.get("semantic_score"), "CB known semantic score"
    )
    cb_semantic = _finite(cb_semantics.get("semantic_score"), "CB semantic score")
    known_multiplier = KNOWN_MIX_WEIGHT if unknown_fields else 1.0
    unknown_multiplier = UNKNOWN_MIX_WEIGHT if unknown_fields else 0.0
    field_rows: list[FieldContribution] = []
    class_rows: list[ClassContribution] = []

    for outcome in u_outcomes:
        field_rows.append(
            _field_contribution(
                group_id=group_id,
                rollout_index=rollout_index,
                candidate="U",
                outcome=outcome,
                component="known",
                multiplier=1.0 / known_fields,
            )
        )
    for outcome in ua_outcomes:
        field_rows.append(
            _field_contribution(
                group_id=group_id,
                rollout_index=rollout_index,
                candidate="UA",
                outcome=outcome,
                component="known",
                multiplier=known_multiplier / known_fields,
            )
        )
    if unknown_fields:
        for outcome in ua_unknown_outcomes:
            field_rows.append(
                _field_contribution(
                    group_id=group_id,
                    rollout_index=rollout_index,
                    candidate="UA",
                    outcome=outcome,
                    component="unknown",
                    multiplier=unknown_multiplier / unknown_fields,
                )
            )

    total_field_weight = _finite(
        cb_known.get("total_field_weight"), "CB total field weight"
    )
    if total_field_weight <= 0.0:
        raise ValueError("CB total field weight must be positive")
    for outcome in cb_outcomes:
        field_name = _string(outcome.get("field_name"), "CB field name")
        field_weight = _finite(outcome.get("field_weight"), f"CB.{field_name}.field_weight")
        if field_weight <= 0.0:
            raise ValueError(f"CB.{field_name}.field_weight must be positive")
        field_row = _field_contribution(
            group_id=group_id,
            rollout_index=rollout_index,
            candidate="CB",
            outcome=outcome,
            component="known",
            multiplier=known_multiplier * field_weight / total_field_weight,
        )
        field_rows.append(field_row)
        class_keys = tuple(
            _string(value, f"CB.{field_name}.gold_class_keys")
            for value in _sequence(
                outcome.get("gold_class_keys"), f"CB.{field_name}.gold_class_keys"
            )
        )
        class_weights = tuple(
            _finite(value, f"CB.{field_name}.gold_class_weights")
            for value in _sequence(
                outcome.get("gold_class_weights"),
                f"CB.{field_name}.gold_class_weights",
            )
        )
        if not class_keys or len(class_keys) != len(class_weights):
            raise ValueError(f"CB.{field_name}: class keys and weights are misaligned")
        if len(set(class_keys)) != len(class_keys):
            raise ValueError(f"CB.{field_name}: duplicate gold class key")
        if not _close(field_weight, statistics.fmean(class_weights)):
            raise ValueError(f"CB.{field_name}: field weight is not the class-weight mean")
        class_weight_sum = sum(class_weights)
        for class_key, class_weight in zip(class_keys, class_weights, strict=True):
            support = class_supports.get((field_name, class_key))
            if support is None:
                raise ValueError(f"missing class support for {(field_name, class_key)}")
            if not _close(class_weight, support.weight):
                raise ValueError(f"class weight drifted for {(field_name, class_key)}")
            class_rows.append(
                ClassContribution(
                    group_id=group_id,
                    rollout_index=rollout_index,
                    candidate="CB",
                    field_name=field_name,
                    class_key=class_key,
                    class_support=support.support,
                    class_support_band=support.support_band,
                    class_weight=class_weight,
                    absolute_contribution=(
                        field_row.absolute_contribution
                        * class_weight
                        / class_weight_sum
                    ),
                )
            )
    if unknown_fields:
        for outcome in cb_unknown_outcomes:
            field_rows.append(
                _field_contribution(
                    group_id=group_id,
                    rollout_index=rollout_index,
                    candidate="CB",
                    outcome=outcome,
                    component="unknown",
                    multiplier=unknown_multiplier / unknown_fields,
                )
            )

    for candidate, expected in (("U", u_semantic), ("UA", ua_semantic), ("CB", cb_semantic)):
        observed = sum(
            row.signed_contribution
            for row in field_rows
            if row.candidate == candidate
        )
        if not _close(observed, expected):
            raise ValueError(f"{candidate}: field contributions do not reconstruct semantics")
    return u_semantic, cb_known_semantic, cb_semantic, field_rows, class_rows


def adapt_replay_group(
    group: Mapping[str, Any],
    *,
    class_supports: Mapping[tuple[str, str], ClassSupportEntry],
) -> AdaptedReplayGroup:
    """Convert one validated nested replay group into disjoint analysis grains."""
    group = _mapping(group, "replay group")
    group_position = _integer(group.get("group_position"), "group_position")
    group_id = _string(group.get("sku_id"), "sku_id")
    known_fields = _integer(group.get("gold_known_fields"), "gold_known_fields", minimum=1)
    unknown_fields = _integer(group.get("gold_unknown_fields"), "gold_unknown_fields")
    if known_fields + unknown_fields != EXPECTED_FIELDS:
        raise ValueError(f"{group_id}: gold known/unknown counts do not sum to 15")
    gold_record = _mapping(group.get("gold_record"), f"{group_id}.gold_record")
    completions = tuple(
        _mapping(value, f"{group_id}.completions")
        for value in _sequence(group.get("completions"), f"{group_id}.completions")
    )
    if len(completions) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"{group_id}: replay group must contain exactly 8 completions")
    indices = [
        _integer(value.get("rollout_index"), f"{group_id}.rollout_index")
        for value in completions
    ]
    if indices != list(range(EXPECTED_GROUP_SIZE)):
        raise ValueError(f"{group_id}: rollout indices must be ordered 0 through 7")
    sources = tuple(
        _mapping(completion.get("source_rollout"), f"{group_id}.source_rollout")
        for completion in completions
    )
    passed: list[bool] = []
    for source in sources:
        value = source.get("passed")
        if not isinstance(value, bool):
            raise TypeError(f"{group_id}: source rollout passed must be boolean")
        passed.append(value)
    reconstructed_pass_rate = sum(passed) / EXPECTED_GROUP_SIZE
    recorded_pass_rate = group.get("difficulty_sft_pass_rate")
    if recorded_pass_rate is None:
        pass_rate = reconstructed_pass_rate
    else:
        if isinstance(recorded_pass_rate, bool):
            raise TypeError(f"{group_id}.difficulty_sft_pass_rate must be numeric")
        pass_rate = _finite(
            recorded_pass_rate, f"{group_id}.difficulty_sft_pass_rate"
        )
        if not _close(pass_rate, reconstructed_pass_rate):
            raise ValueError(
                f"{group_id}: recorded difficulty differs from source rollouts"
            )

    rewards: dict[str, list[float]] = {label: [] for label in REWARD_LABELS}
    targets: dict[str, list[float | None]] = {
        "canonical_known_utility": [],
        "known_exact_rate": [],
        "known_coverage": [],
        "selective_correctness": [],
        "unknown_abstention_rate": [],
        "rule_quality": [],
        "class_balanced_known_utility": [],
    }
    field_rows: list[FieldContribution] = []
    class_rows: list[ClassContribution] = []

    for completion, source, rollout_index in zip(
        completions, sources, indices, strict=True
    ):
        source_index = _integer(
            source.get("rollout_index"), f"{group_id}.source_rollout.rollout_index"
        )
        if source_index != rollout_index:
            raise ValueError(f"{group_id}: source rollout index differs from completion")
        source_targets = _source_targets(
            source,
            group_id=group_id,
            known_fields=known_fields,
            unknown_fields=unknown_fields,
        )
        original = _mapping(completion.get("original_reward"), "original_reward")
        rewards["original"].append(
            _finite(original.get("weighted_total"), "original weighted total")
        )
        raw_candidates = _mapping(completion.get("candidates"), "candidates")
        if set(raw_candidates) != set(EXPECTED_CANDIDATES):
            raise ValueError(f"{group_id}: candidate ledger set drifted")
        candidates = {
            name: _mapping(raw_candidates[name], f"candidate {name}")
            for name in EXPECTED_CANDIDATES
        }
        eligibilities: list[bool] = []
        for name in EXPECTED_CANDIDATES:
            if candidates[name].get("candidate") != name:
                raise ValueError(f"{group_id}: {name} candidate identity drifted")
            eligible = candidates[name].get("eligible")
            if not isinstance(eligible, bool):
                raise TypeError(f"{group_id}: {name}.eligible must be boolean")
            eligibilities.append(eligible)
            rewards[name].append(
                _finite(candidates[name].get("reward"), f"{name}.reward")
            )
        if len(set(eligibilities)) != 1:
            raise ValueError(f"{group_id}: candidate semantic-gate eligibility drifted")

        if not eligibilities[0]:
            for name in EXPECTED_CANDIDATES:
                if not _close(rewards[name][-1], MALFORMED_FLOOR):
                    raise ValueError(f"{group_id}: ineligible {name} did not receive floor")
                semantic_key = (
                    "known_semantics" if name == "U" else "unknown_aware_semantics"
                )
                if candidates[name].get(semantic_key) is not None:
                    raise ValueError(f"{group_id}: ineligible {name} retained semantics")
            known_utility = None
            cb_known_utility = None
            cb_reward_semantic = None
        else:
            candidate_rules = [
                tuple(
                    _string(value, f"{name}.rule violation")
                    for value in _sequence(
                        candidates[name].get("rule_violations"), f"{name}.rules"
                    )
                )
                for name in EXPECTED_CANDIDATES
            ]
            if len(set(candidate_rules)) != 1:
                raise ValueError(f"{group_id}: candidate rule ledgers drifted")
            source_rules = tuple(
                _string(value, f"{group_id}.source rule violation")
                for value in _sequence(
                    source.get("rule_violations"), f"{group_id}.source rules"
                )
            )
            if candidate_rules[0] != source_rules:
                raise ValueError(f"{group_id}: source and candidate rule ledgers drifted")
            (
                known_utility,
                cb_known_utility,
                cb_reward_semantic,
                completion_fields,
                completion_classes,
            ) = _eligible_ledgers(
                group_id=group_id,
                rollout_index=rollout_index,
                candidates=candidates,
                known_fields=known_fields,
                unknown_fields=unknown_fields,
                class_supports=class_supports,
            )
            field_rows.extend(completion_fields)
            class_rows.extend(completion_classes)
            for name, semantic in (
                ("U", known_utility),
                (
                    "UA",
                    _finite(
                        _mapping(
                            candidates["UA"]["unknown_aware_semantics"],
                            "UA semantics",
                        )["semantic_score"],
                        "UA semantic score",
                    ),
                ),
                ("CB", cb_reward_semantic),
            ):
                adjustment = _finite(
                    candidates[name].get("rule_adjustment"), f"{name}.rule_adjustment"
                )
                if not _close(rewards[name][-1], semantic + adjustment):
                    raise ValueError(f"{group_id}: {name} reward does not reconstruct")

        targets["canonical_known_utility"].append(known_utility)
        targets["class_balanced_known_utility"].append(cb_known_utility)
        for name, value in source_targets.items():
            targets[name].append(value)

    return AdaptedReplayGroup(
        version=VERSION,
        group_position=group_position,
        observation=GroupObservation(
            group_id=group_id,
            rewards={name: tuple(values) for name, values in rewards.items()},
            targets={name: tuple(values) for name, values in targets.items()},
        ),
        segments=GroupSegments(
            product_category=_product_category(gold_record),
            difficulty_band=difficulty_band(pass_rate),
            gold_known_field_count=known_fields,
        ),
        field_contributions=tuple(field_rows),
        class_contributions=tuple(class_rows),
        boundary={
            "input_was_in_memory": True,
            "file_io_performed": False,
            "real_candidate_replay_opened": False,
            "aggregate_candidate_metrics_calculated": False,
            "acceptance_gates_applied": False,
            "winner_selected": False,
        },
    )
