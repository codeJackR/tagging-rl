#!/usr/bin/env python3
"""Pure in-memory dominance and segment summaries for GRPO Run 2.

Product-level reward metrics are calculated only from one observation per
product. Field and class contribution tables are summarized separately and can
never alter product/completion denominators. This layer performs no file I/O,
gate application, or candidate selection.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from training.analyze_run2_candidates import (
    KNOWN_COVERAGE,
    KNOWN_UTILITY,
    REWARD_LABELS,
    numeric_distribution,
    summarize_directional_alignment_groups,
    summarize_harmful_coverage_groups,
    summarize_reward_groups,
)
from training.run2_comparison_contract import (
    EXPECTED_CANDIDATES,
    EXPECTED_GROUP_SIZE,
    MIN_SEGMENT_GROUP_SUPPORT,
)
from training.run2_replay_adapter import (
    AdaptedReplayGroup,
    ClassContribution,
    FieldContribution,
    class_support_band,
)


VERSION = "grpo-run2-contribution-segment-summaries-v1"
FIELD_DOMINANCE_THRESHOLD_REFERENCE = 0.20
CB_CLASS_DOMINANCE_THRESHOLD_REFERENCE = 0.15


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_groups(
    groups: Sequence[AdaptedReplayGroup],
) -> tuple[AdaptedReplayGroup, ...]:
    if not groups:
        raise ValueError("at least one adapted replay group is required")
    group_ids = [group.observation.group_id for group in groups]
    if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
        raise ValueError("every adapted group requires a nonempty group ID")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("adapted replay groups contain duplicate group IDs")
    positions = [group.group_position for group in groups]
    if positions != list(range(len(groups))):
        raise ValueError("adapted replay group positions must be canonical and contiguous")
    return tuple(groups)


def _validate_field_rows(
    groups: Sequence[AdaptedReplayGroup],
) -> tuple[FieldContribution, ...]:
    group_ids = {group.observation.group_id for group in groups}
    seen: set[tuple[str, int, str, str, str]] = set()
    rows: list[FieldContribution] = []
    for group in groups:
        expected_group = group.observation.group_id
        for row in group.field_contributions:
            if row.group_id != expected_group or row.group_id not in group_ids:
                raise ValueError("field contribution has an unknown or mismatched group ID")
            if row.candidate not in EXPECTED_CANDIDATES:
                raise ValueError(f"unexpected field-contribution candidate: {row.candidate}")
            if not 0 <= row.rollout_index < EXPECTED_GROUP_SIZE:
                raise ValueError("field contribution rollout index lies outside 0..7")
            if not row.field_name:
                raise ValueError("field contribution requires a field name")
            if row.component not in {"known", "unknown"}:
                raise ValueError("field contribution component must be known or unknown")
            utility = _finite(row.utility, "field utility")
            signed = _finite(row.signed_contribution, "signed field contribution")
            absolute = _finite(row.absolute_contribution, "absolute field contribution")
            if absolute < 0.0 or not _close(absolute, abs(signed)):
                raise ValueError("absolute field contribution does not match signed value")
            _ = utility
            key = (
                row.group_id,
                row.rollout_index,
                row.candidate,
                row.field_name,
                row.component,
            )
            if key in seen:
                raise ValueError(f"duplicate field-contribution key: {key}")
            seen.add(key)
            rows.append(row)
    return tuple(rows)


def _validate_class_rows(
    groups: Sequence[AdaptedReplayGroup],
    field_rows: Sequence[FieldContribution],
) -> tuple[ClassContribution, ...]:
    group_ids = {group.observation.group_id for group in groups}
    cb_known = {
        (row.group_id, row.rollout_index, row.field_name): row
        for row in field_rows
        if row.candidate == "CB" and row.component == "known"
    }
    seen: set[tuple[str, int, str, str]] = set()
    rows: list[ClassContribution] = []
    allocations: dict[tuple[str, int, str], float] = defaultdict(float)
    for group in groups:
        expected_group = group.observation.group_id
        for row in group.class_contributions:
            if row.group_id != expected_group or row.group_id not in group_ids:
                raise ValueError("class contribution has an unknown or mismatched group ID")
            if row.candidate != "CB":
                raise ValueError("class contribution candidate must be CB")
            if not 0 <= row.rollout_index < EXPECTED_GROUP_SIZE:
                raise ValueError("class contribution rollout index lies outside 0..7")
            if not row.field_name or not row.class_key:
                raise ValueError("class contribution requires field and class keys")
            if row.class_support <= 0:
                raise ValueError("class contribution support must be positive")
            if row.class_support_band != class_support_band(row.class_support):
                raise ValueError("class support band does not match support")
            if _finite(row.class_weight, "class weight") <= 0.0:
                raise ValueError("class weight must be positive")
            absolute = _finite(row.absolute_contribution, "absolute class contribution")
            if absolute < 0.0:
                raise ValueError("absolute class contribution cannot be negative")
            key = (row.group_id, row.rollout_index, row.field_name, row.class_key)
            if key in seen:
                raise ValueError(f"duplicate class-contribution key: {key}")
            seen.add(key)
            parent = key[:3]
            if parent not in cb_known:
                raise ValueError(f"class contribution has no CB known-field parent: {parent}")
            allocations[parent] += absolute
            rows.append(row)
    if set(allocations) != set(cb_known):
        missing = sorted(set(cb_known) - set(allocations))
        raise ValueError(f"CB known fields are missing class allocations: {missing[:3]}")
    for key, parent in cb_known.items():
        if not _close(allocations[key], parent.absolute_contribution):
            raise ValueError(f"class allocations do not reconstruct CB field: {key}")
    return tuple(rows)


def _share_rows(
    absolute_by_key: Mapping[str, float], row_counts: Mapping[str, int]
) -> tuple[dict[str, Any], str | None, float]:
    total = sum(absolute_by_key.values())
    summaries: dict[str, Any] = {}
    for key in sorted(absolute_by_key):
        absolute = absolute_by_key[key]
        summaries[key] = {
            "absolute_contribution": absolute,
            "share": absolute / total if total else 0.0,
            "child_rows": row_counts[key],
        }
    if not summaries:
        return summaries, None, 0.0
    largest_key = min(
        summaries,
        key=lambda key: (-summaries[key]["share"], key),
    )
    return summaries, largest_key, summaries[largest_key]["share"]


def summarize_contribution_dominance(
    groups: Sequence[AdaptedReplayGroup],
) -> dict[str, Any]:
    """Summarize field and CB-class contribution shares without joining grains."""
    groups = _validate_groups(groups)
    field_rows = _validate_field_rows(groups)
    class_rows = _validate_class_rows(groups, field_rows)

    field_summary: dict[str, Any] = {}
    for candidate in EXPECTED_CANDIDATES:
        candidate_rows = [row for row in field_rows if row.candidate == candidate]
        by_field: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        by_component: dict[str, float] = defaultdict(float)
        by_group_field: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for row in candidate_rows:
            by_field[row.field_name] += row.absolute_contribution
            counts[row.field_name] += 1
            by_component[row.component] += row.absolute_contribution
            by_group_field[row.group_id][row.field_name] += row.absolute_contribution
        fields, largest_field, largest_share = _share_rows(by_field, counts)
        per_group_largest: list[float] = []
        for values in by_group_field.values():
            total = sum(values.values())
            if total:
                per_group_largest.append(max(values.values()) / total)
        field_summary[candidate] = {
            "child_rows": len(candidate_rows),
            "absolute_contribution_total": sum(by_field.values()),
            "absolute_contribution_by_component": dict(sorted(by_component.items())),
            "fields": fields,
            "largest_field": largest_field,
            "largest_field_share": largest_share,
            "per_group_largest_field_share_distribution": (
                numeric_distribution(per_group_largest) if per_group_largest else None
            ),
            "groups_with_nonzero_absolute_contribution": len(per_group_largest),
            "dominance_threshold_reference": FIELD_DOMINANCE_THRESHOLD_REFERENCE,
            "dominance_gate_applied": False,
        }

    by_class: dict[str, float] = defaultdict(float)
    class_counts: dict[str, int] = defaultdict(int)
    by_support_band: dict[str, float] = defaultdict(float)
    support_band_rows: dict[str, int] = defaultdict(int)
    by_group_class: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    class_metadata: dict[str, dict[str, Any]] = {}
    for row in class_rows:
        class_id = f"{row.field_name}::{row.class_key}"
        by_class[class_id] += row.absolute_contribution
        class_counts[class_id] += 1
        by_support_band[row.class_support_band] += row.absolute_contribution
        support_band_rows[row.class_support_band] += 1
        by_group_class[row.group_id][class_id] += row.absolute_contribution
        metadata = {
            "field_name": row.field_name,
            "class_key": row.class_key,
            "class_support": row.class_support,
            "class_support_band": row.class_support_band,
            "class_weight": row.class_weight,
        }
        if class_id in class_metadata and class_metadata[class_id] != metadata:
            raise ValueError(f"class metadata drifted across rows: {class_id}")
        class_metadata[class_id] = metadata
    classes, largest_class, largest_class_share = _share_rows(by_class, class_counts)
    for class_id, metadata in class_metadata.items():
        classes[class_id].update(metadata)
    class_total = sum(by_class.values())
    support_bands = {
        band: {
            "absolute_contribution": by_support_band[band],
            "share": by_support_band[band] / class_total if class_total else 0.0,
            "child_rows": support_band_rows[band],
        }
        for band in sorted(by_support_band)
    }
    per_group_class_largest = []
    for values in by_group_class.values():
        total = sum(values.values())
        if total:
            per_group_class_largest.append(max(values.values()) / total)
    return {
        "version": VERSION,
        "groups": len(groups),
        "field_contributions": field_summary,
        "cb_class_contributions": {
            "child_rows": len(class_rows),
            "absolute_contribution_total": class_total,
            "classes": classes,
            "support_bands": support_bands,
            "largest_class": largest_class,
            "largest_class_share": largest_class_share,
            "per_group_largest_class_share_distribution": (
                numeric_distribution(per_group_class_largest)
                if per_group_class_largest
                else None
            ),
            "groups_with_nonzero_absolute_contribution": len(per_group_class_largest),
            "dominance_threshold_reference": CB_CLASS_DOMINANCE_THRESHOLD_REFERENCE,
            "dominance_gate_applied": False,
        },
        "grain_guardrails": {
            "product_counts_derived_from_group_observations_only": True,
            "field_child_rows_used_as_product_denominator": False,
            "class_child_rows_used_as_product_denominator": False,
            "class_allocations_reconstruct_cb_known_fields": True,
        },
    }


def _segment_summary(groups: Sequence[AdaptedReplayGroup]) -> dict[str, Any]:
    observations = [group.observation for group in groups]
    reward_groups = {
        label: {
            observation.group_id: observation.rewards[label]
            for observation in observations
        }
        for label in REWARD_LABELS
    }
    target_groups = {
        target: {
            observation.group_id: observation.targets[target]
            for observation in observations
        }
        for target in observations[0].targets
    }
    return {
        "groups": len(groups),
        "completions": len(groups) * EXPECTED_GROUP_SIZE,
        "ordered_group_ids": [group.observation.group_id for group in groups],
        "ordered_group_id_sha256": _ordered_hash(
            [group.observation.group_id for group in groups]
        ),
        "interpretation_allowed": len(groups) >= MIN_SEGMENT_GROUP_SUPPORT,
        "minimum_groups_for_interpretation": MIN_SEGMENT_GROUP_SUPPORT,
        "reward_summaries": {
            label: summarize_reward_groups(values)
            for label, values in reward_groups.items()
        },
        "directional_alignments": {
            label: {
                target: summarize_directional_alignment_groups(
                    values, target_groups[target]
                )
                for target in target_groups
            }
            for label, values in reward_groups.items()
        },
        "harmful_coverage": {
            label: summarize_harmful_coverage_groups(
                values,
                target_groups[KNOWN_COVERAGE],
                target_groups[KNOWN_UTILITY],
            )
            for label, values in reward_groups.items()
        },
    }


def summarize_product_segments(
    groups: Sequence[AdaptedReplayGroup],
) -> dict[str, Any]:
    """Report all primary metrics by three pre-outcome product dimensions."""
    groups = _validate_groups(groups)
    dimensions: tuple[tuple[str, Callable[[AdaptedReplayGroup], str]], ...] = (
        ("product_category", lambda group: group.segments.product_category),
        ("difficulty_band", lambda group: group.segments.difficulty_band),
        (
            "gold_known_field_count",
            lambda group: str(group.segments.gold_known_field_count),
        ),
    )
    result: dict[str, Any] = {}
    for dimension, key_fn in dimensions:
        members: dict[str, list[AdaptedReplayGroup]] = defaultdict(list)
        for group in groups:
            key = key_fn(group)
            if not key:
                raise ValueError(f"{dimension} segment key cannot be empty")
            members[key].append(group)
        segment_summaries = {
            key: _segment_summary(segment_groups)
            for key, segment_groups in sorted(members.items())
        }
        membership_count = sum(value["groups"] for value in segment_summaries.values())
        if membership_count != len(groups):
            raise RuntimeError(f"{dimension} segment membership does not cover groups once")
        result[dimension] = {
            "segments": segment_summaries,
            "segment_count": len(segment_summaries),
            "group_memberships": membership_count,
            "every_group_appears_exactly_once": True,
        }
    return {
        "version": VERSION,
        "groups": len(groups),
        "dimensions": result,
        "segment_keys_are_pre_outcome": True,
        "bootstrap_performed": False,
        "acceptance_gates_applied": False,
        "winner_selected": False,
    }


def summarize_in_memory_diagnostics(
    groups: Sequence[AdaptedReplayGroup],
) -> dict[str, Any]:
    """Compose pure dominance and segment summaries without reading real data."""
    groups = _validate_groups(groups)
    return {
        "version": VERSION,
        "status": "in_memory_diagnostics_completed",
        "boundary": {
            "file_io_performed": False,
            "real_candidate_replay_opened_by_this_module": False,
            "acceptance_gates_applied": False,
            "winner_selected": False,
        },
        "dominance": summarize_contribution_dominance(groups),
        "product_segments": summarize_product_segments(groups),
    }
