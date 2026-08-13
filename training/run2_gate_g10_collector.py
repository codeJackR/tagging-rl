#!/usr/bin/env python3
"""Compose the Gate G10 adapter and calculator over in-memory groups.

The collector validates denominator-wide order and identity, adapts each nested
product exactly once, and calculates Gate G10 once per candidate on the same
ordered products. It contains no file-I/O entry point.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from training.run2_comparison_contract import (
    EXPECTED_CANDIDATES,
    EXPECTED_GROUP_SIZE,
)
from training.run2_gate_g10 import GateG10Group, calculate_gate_g10
from training.run2_gate_g10_adapter import adapt_gate_g10_group


VERSION = "grpo-run2-gate-g10-collector-v1"


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def collect_and_calculate_gate_g10(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_groups: int,
) -> dict[str, Any]:
    """Calculate all candidates on one validated in-memory product sequence."""
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError("Gate G10 groups must be an ordered sequence")
    if (
        not isinstance(expected_groups, int)
        or isinstance(expected_groups, bool)
        or expected_groups <= 0
    ):
        raise ValueError("expected_groups must be a positive integer")
    if len(groups) != expected_groups:
        raise ValueError(
            "Gate G10 collector denominator mismatch: "
            f"expected={expected_groups}, observed={len(groups)}"
        )

    candidate_groups: dict[str, list[GateG10Group]] = {
        candidate: [] for candidate in EXPECTED_CANDIDATES
    }
    ordered_skus: list[str] = []
    ordered_rollout_keys: list[str] = []
    seen_skus: set[str] = set()
    per_group_rollout_hashes: list[str] = []
    for expected_position, group in enumerate(groups):
        adapted = adapt_gate_g10_group(group)
        if adapted.group_position != expected_position:
            raise ValueError(
                "Gate G10 group positions must be contiguous from zero: "
                f"expected={expected_position}, observed={adapted.group_position}"
            )
        if adapted.sku_id in seen_skus:
            raise ValueError(f"duplicate Gate G10 SKU: {adapted.sku_id}")
        seen_skus.add(adapted.sku_id)
        if len(adapted.candidate_groups) != len(EXPECTED_CANDIDATES):
            raise ValueError("Gate G10 adapted candidate count drifted")
        for candidate, candidate_group in zip(
            EXPECTED_CANDIDATES,
            adapted.candidate_groups,
            strict=True,
        ):
            if candidate_group.group_id != adapted.sku_id:
                raise ValueError("Gate G10 candidate group identity drifted")
            candidate_groups[candidate].append(candidate_group)
        ordered_skus.append(adapted.sku_id)
        group_rollout_keys = [
            f"{adapted.sku_id}\t{rollout_index}"
            for rollout_index in range(EXPECTED_GROUP_SIZE)
        ]
        if adapted.ordered_rollout_key_sha256 != _ordered_hash(group_rollout_keys):
            raise RuntimeError("Gate G10 adapter and collector rollout order disagree")
        ordered_rollout_keys.extend(group_rollout_keys)
        per_group_rollout_hashes.append(adapted.ordered_rollout_key_sha256)

    results = {
        candidate: calculate_gate_g10(
            candidate_groups[candidate],
            candidate=candidate,
            expected_groups=expected_groups,
        )
        for candidate in EXPECTED_CANDIDATES
    }
    candidate_group_hashes = {
        candidate: result["ordered_group_id_sha256"]
        for candidate, result in results.items()
    }
    if len(set(candidate_group_hashes.values())) != 1:
        raise RuntimeError("Gate G10 candidate denominators are not identically ordered")
    ordered_sku_sha256 = _ordered_hash(ordered_skus)
    if next(iter(candidate_group_hashes.values())) != ordered_sku_sha256:
        raise RuntimeError("Gate G10 calculator and collector SKU order disagree")

    return {
        "version": VERSION,
        "status": "in_memory_gate_g10_completed_unverified_source_scope",
        "candidate_order": list(EXPECTED_CANDIDATES),
        "lineage": {
            "groups": expected_groups,
            "completions": expected_groups * EXPECTED_GROUP_SIZE,
            "unique_skus": len(seen_skus),
            "ordered_sku_sha256": ordered_sku_sha256,
            "ordered_rollout_key_sha256": _ordered_hash(ordered_rollout_keys),
            "per_group_rollout_key_sha256": per_group_rollout_hashes,
            "candidate_ordered_group_sha256": candidate_group_hashes,
            "contiguous_group_positions": True,
            "all_candidates_share_ordered_denominator": True,
            "groups_adapted_once": True,
            "calculator_calls": len(EXPECTED_CANDIDATES),
        },
        "candidate_results": results,
        "boundary": {
            "input_kind": "ordered in-memory nested replay groups",
            "file_io_performed": False,
            "real_full_training_replay_opened_by_collector": False,
            "gate_g10_calculated_for_supplied_inputs": True,
            "source_scope_verified_by_collector": False,
            "real_gate_g10_result_authorized": False,
            "active_candidate_aggregates_calculated": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
        },
    }
