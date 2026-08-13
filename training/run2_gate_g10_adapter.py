#!/usr/bin/env python3
"""Adapt one in-memory full-replay ledger group for Gate G10.

This module validates product, rollout and candidate alignment, then extracts
only the saved final candidate rewards needed by the pure Gate G10 calculator.
It has no file-I/O entry point and does not calculate aggregate metrics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from training.run2_comparison_contract import (
    EXPECTED_CANDIDATES,
    EXPECTED_GROUP_SIZE,
)
from training.run2_gate_g10 import GateG10Group


VERSION = "grpo-run2-gate-g10-ledger-adapter-v1"


@dataclass(frozen=True)
class AdaptedGateG10Group:
    """Three candidate inputs extracted from one aligned product group."""

    version: str
    group_position: int
    sku_id: str
    candidate_groups: tuple[GateG10Group, ...]
    ordered_rollout_key_sha256: str
    boundary: Mapping[str, bool]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def adapt_gate_g10_group(group: Mapping[str, Any]) -> AdaptedGateG10Group:
    """Validate one nested replay group and extract U/UA/CB reward vectors."""
    group = _mapping(group, "Gate G10 replay group")
    group_position = _integer(group.get("group_position"), "group_position")
    sku_id = _string(group.get("sku_id"), "sku_id")
    completions = tuple(
        _mapping(value, f"{sku_id}.completions")
        for value in _sequence(group.get("completions"), f"{sku_id}.completions")
    )
    if len(completions) != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"{sku_id}: replay group must contain exactly "
            f"{EXPECTED_GROUP_SIZE} completions"
        )

    rewards: dict[str, list[float]] = {
        candidate: [] for candidate in EXPECTED_CANDIDATES
    }
    rollout_keys: list[str] = []
    observed_indices: list[int] = []
    for completion_position, completion in enumerate(completions):
        rollout_index = _integer(
            completion.get("rollout_index"),
            f"{sku_id}.completions[{completion_position}].rollout_index",
        )
        observed_indices.append(rollout_index)
        source = _mapping(
            completion.get("source_rollout"),
            f"{sku_id}.completions[{completion_position}].source_rollout",
        )
        source_sku = _string(
            source.get("sku_id"),
            f"{sku_id}.completions[{completion_position}].source_rollout.sku_id",
        )
        source_index = _integer(
            source.get("rollout_index"),
            f"{sku_id}.completions[{completion_position}].source_rollout.rollout_index",
        )
        if source_sku != sku_id or source_index != rollout_index:
            raise ValueError(
                f"{sku_id}: source rollout key differs from completion key"
            )
        rollout_keys.append(f"{source_sku}\t{source_index}")

        candidates = _mapping(
            completion.get("candidates"),
            f"{sku_id}.completions[{completion_position}].candidates",
        )
        if set(candidates) != set(EXPECTED_CANDIDATES):
            raise ValueError(
                f"{sku_id}: candidate ledger set must be {EXPECTED_CANDIDATES}"
            )
        eligibilities: list[bool] = []
        for candidate in EXPECTED_CANDIDATES:
            ledger = _mapping(
                candidates[candidate],
                f"{sku_id}.completions[{completion_position}].{candidate}",
            )
            if ledger.get("candidate") != candidate:
                raise ValueError(f"{sku_id}: {candidate} candidate identity drifted")
            eligible = ledger.get("eligible")
            if not isinstance(eligible, bool):
                raise TypeError(f"{sku_id}: {candidate}.eligible must be boolean")
            eligibilities.append(eligible)
            rewards[candidate].append(
                _finite(ledger.get("reward"), f"{sku_id}.{candidate}.reward")
            )
        if len(set(eligibilities)) != 1:
            raise ValueError(f"{sku_id}: candidate eligibility alignment drifted")

    if observed_indices != list(range(EXPECTED_GROUP_SIZE)):
        raise ValueError(f"{sku_id}: rollout indices must be ordered 0 through 7")
    if len(set(rollout_keys)) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"{sku_id}: source rollout keys are not unique")

    return AdaptedGateG10Group(
        version=VERSION,
        group_position=group_position,
        sku_id=sku_id,
        candidate_groups=tuple(
            GateG10Group(group_id=sku_id, rewards=tuple(rewards[candidate]))
            for candidate in EXPECTED_CANDIDATES
        ),
        ordered_rollout_key_sha256=_ordered_hash(rollout_keys),
        boundary={
            "input_was_one_in_memory_replay_group": True,
            "file_io_performed": False,
            "real_full_training_replay_opened": False,
            "gate_g10_calculated": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
        },
    )
