#!/usr/bin/env python3
"""Pure in-memory calculator for GRPO Run 2 comparison Gate G10.

Gate G10 asks whether no more than 40% of product groups have one canonical
reward level across their eight completions. This module performs no file I/O;
an orchestrator must separately prove the source scope and artifact identities.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from training.replay_run2_full_training_candidates import (
    EXPECTED_COMPLETIONS,
    EXPECTED_TRAINING_GROUPS,
)
from training.run2_comparison_contract import (
    COMPARISON_DECIMALS,
    EXPECTED_CANDIDATES,
    EXPECTED_GROUP_SIZE,
    group_reward_shape,
)


VERSION = "grpo-run2-gate-g10-core-v1"
GATE_ID = "G10_full_training_variation"
THRESHOLD_NUMERATOR = 2
THRESHOLD_DENOMINATOR = 5
THRESHOLD = THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR
ORIGINAL_ZERO_VARIANCE_GROUPS = 1_571
ORIGINAL_ZERO_VARIANCE_SHARE = (
    ORIGINAL_ZERO_VARIANCE_GROUPS / EXPECTED_TRAINING_GROUPS
)


@dataclass(frozen=True)
class GateG10Group:
    """One product's eight aligned rewards for one candidate."""

    group_id: str
    rewards: Sequence[float]


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def calculate_gate_g10(
    groups: Sequence[GateG10Group],
    *,
    candidate: str,
    expected_groups: int = EXPECTED_TRAINING_GROUPS,
) -> dict[str, Any]:
    """Count zero-variance product groups and evaluate the locked 40% gate.

    Equality uses the comparison contract's 12-decimal canonicalization through
    ``group_reward_shape``. The threshold itself is evaluated as exact integer
    arithmetic, avoiding floating-point ambiguity at exactly 40%.
    """
    if candidate not in EXPECTED_CANDIDATES:
        raise ValueError(f"candidate must be one of {EXPECTED_CANDIDATES}")
    if (
        not isinstance(expected_groups, int)
        or isinstance(expected_groups, bool)
        or expected_groups <= 0
    ):
        raise ValueError("expected_groups must be a positive integer")
    if len(groups) != expected_groups:
        raise ValueError(
            "Gate G10 denominator mismatch: "
            f"expected={expected_groups}, observed={len(groups)}"
        )

    seen: set[str] = set()
    ordered_group_ids: list[str] = []
    unique_level_counts: list[int] = []
    for position, group in enumerate(groups):
        if not isinstance(group, GateG10Group):
            raise TypeError(f"group {position} must be a GateG10Group")
        if not isinstance(group.group_id, str) or not group.group_id:
            raise ValueError(f"group {position} has an invalid group_id")
        if group.group_id in seen:
            raise ValueError(f"duplicate Gate G10 group_id: {group.group_id}")
        seen.add(group.group_id)
        shape = group_reward_shape(group.rewards)
        ordered_group_ids.append(group.group_id)
        unique_level_counts.append(int(shape["unique_reward_levels"]))

    zero_variance_groups = sum(levels == 1 for levels in unique_level_counts)
    varying_groups = expected_groups - zero_variance_groups
    maximum_allowed = (
        expected_groups * THRESHOLD_NUMERATOR // THRESHOLD_DENOMINATOR
    )
    passes = (
        zero_variance_groups * THRESHOLD_DENOMINATOR
        <= expected_groups * THRESHOLD_NUMERATOR
    )
    zero_variance_share = zero_variance_groups / expected_groups
    return {
        "version": VERSION,
        "gate_id": GATE_ID,
        "candidate": candidate,
        "metric": "full authoritative-training zero-variance group share",
        "unit": "product group",
        "groups": expected_groups,
        "completions": expected_groups * EXPECTED_GROUP_SIZE,
        "zero_variance_groups": zero_variance_groups,
        "varying_groups": varying_groups,
        "zero_variance_share": zero_variance_share,
        "unique_reward_levels_per_group_histogram": {
            str(levels): count
            for levels, count in sorted(Counter(unique_level_counts).items())
        },
        "ordered_group_id_sha256": _ordered_hash(ordered_group_ids),
        "comparison": {
            "canonical_decimal_places": COMPARISON_DECIMALS,
            "operator": "less_than_or_equal",
            "threshold": THRESHOLD,
            "threshold_exact_fraction": (
                f"{THRESHOLD_NUMERATOR}/{THRESHOLD_DENOMINATOR}"
            ),
            "maximum_allowed_zero_variance_groups": maximum_allowed,
            "passes": passes,
            "margin_groups": maximum_allowed - zero_variance_groups,
            "margin_share": THRESHOLD - zero_variance_share,
        },
        "locked_original_baseline": {
            "groups": EXPECTED_TRAINING_GROUPS,
            "completions": EXPECTED_COMPLETIONS,
            "zero_variance_groups": ORIGINAL_ZERO_VARIANCE_GROUPS,
            "zero_variance_share": ORIGINAL_ZERO_VARIANCE_SHARE,
        },
        "boundary": {
            "input_kind": "already-materialized in-memory candidate reward groups",
            "file_io_performed": False,
            "real_full_training_replay_opened_by_calculator": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
        },
    }
