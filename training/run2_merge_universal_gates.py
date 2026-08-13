"""Merge GRPO Run 2 G1-G9 and G10 evidence without selecting a candidate.

The two sources intentionally use different product scopes.  This module joins
only by the locked candidate IDs and preserves both denominators; it never
combines their rows, averages their metrics, ranks candidates, or authorizes a
GPU run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


VERSION = "grpo-run2-universal-gate-merge-v1"
CANDIDATES = ("U", "UA", "CB")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _candidate_order(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    order = tuple(value)
    if order != CANDIDATES:
        raise ValueError(f"{label} must be exactly {CANDIDATES}")
    return order


def _gate_10_contract(comparison_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = comparison_contract.get("universal_acceptance_gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        raise TypeError("universal acceptance gates must be a sequence")
    matches = [
        _mapping(gate, "universal gate")
        for gate in gates
        if isinstance(gate, Mapping)
        and gate.get("id") == "G10_full_training_variation"
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one locked G10 contract")
    return matches[0]


def merge_universal_gates(
    *,
    comparison_contract: Mapping[str, Any],
    gates_g1_g9: Mapping[str, Any],
    gate_g10: Mapping[str, Any],
    input_identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Return all-ten gate eligibility while preserving the selection boundary."""

    contract = _mapping(comparison_contract, "comparison contract")
    first = _mapping(gates_g1_g9, "G1-G9 artifact")
    tenth = _mapping(gate_g10, "G10 artifact")
    identities = _mapping(input_identities, "input identities")
    if identities.get("all_verified") is not True:
        raise ValueError("input identities were not verified")

    if first.get("status") != "gates_g1_g9_applied_pending_g10_merge_and_selection":
        raise ValueError("G1-G9 artifact status drifted")
    if tenth.get("status") != "production_gate_g10_completed":
        raise ValueError("G10 artifact status drifted")
    _candidate_order(first.get("candidate_order"), "G1-G9 candidate order")
    _candidate_order(tenth.get("candidate_order"), "G10 candidate order")

    first_boundary = _mapping(first.get("selection_boundary"), "G1-G9 boundary")
    expected_first_boundary = {
        "candidate_rankings_calculated": False,
        "complexity_aware_selection_applied": False,
        "gate_g10_merged": False,
        "gates_g1_through_g9_applied": True,
        "gpu_training_authorized": False,
        "winner_selected": False,
    }
    if dict(first_boundary) != expected_first_boundary:
        raise ValueError("G1-G9 selection boundary drifted")

    tenth_boundary = _mapping(tenth.get("selection_boundary"), "G10 boundary")
    for key in (
        "candidate_rankings_calculated",
        "gates_g1_through_g9_applied",
        "gpu_training_authorized",
        "winner_selected",
    ):
        if tenth_boundary.get(key) is not False:
            raise ValueError(f"G10 boundary {key} drifted")
    if tenth_boundary.get("gate_g10_calculated") is not True:
        raise ValueError("G10 source did not calculate Gate 10")
    if tenth_boundary.get("gate_g10_threshold_applied") is not True:
        raise ValueError("G10 source did not apply its threshold")

    locked_g10 = _gate_10_contract(contract)
    observed_g10 = _mapping(tenth.get("gate_contract"), "G10 result contract")
    contract_key_map = {
        "gate_id": "id",
        "metric": "metric",
        "operator": "operator",
        "threshold": "threshold",
    }
    for result_key, contract_key in contract_key_map.items():
        if observed_g10.get(result_key) != locked_g10.get(contract_key):
            raise ValueError(
                f"G10 result {result_key} disagrees with locked contract"
            )

    active_lineage = _mapping(first.get("lineage"), "G1-G9 lineage")
    full_lineage = _mapping(tenth.get("lineage"), "G10 lineage")
    if active_lineage.get("groups") != 1_438 or active_lineage.get("completions") != 11_504:
        raise ValueError("active-scope lineage drifted")
    if full_lineage.get("groups") != 3_240 or full_lineage.get("completions") != 25_920:
        raise ValueError("full-scope lineage drifted")
    if full_lineage.get("all_candidates_share_ordered_denominator") is not True:
        raise ValueError("G10 candidates do not share one ordered denominator")
    if full_lineage.get("manifest_lineage_matches_stream") is not True:
        raise ValueError("G10 manifest and stream lineage disagree")

    first_results = _mapping(first.get("candidate_results"), "G1-G9 results")
    tenth_results = _mapping(tenth.get("candidate_results"), "G10 results")
    if set(first_results) != set(CANDIDATES) or set(tenth_results) != set(CANDIDATES):
        raise ValueError("candidate result sets do not match the locked candidates")

    merged: dict[str, Any] = {}
    for candidate in CANDIDATES:
        g1_g9 = _mapping(first_results[candidate], f"{candidate} G1-G9 result")
        g10 = _mapping(tenth_results[candidate], f"{candidate} G10 result")
        if g10.get("candidate") != candidate:
            raise ValueError(f"{candidate} G10 identity drifted")
        if g10.get("gate_id") != "G10_full_training_variation":
            raise ValueError(f"{candidate} G10 gate ID drifted")
        if g10.get("groups") != 3_240 or g10.get("completions") != 25_920:
            raise ValueError(f"{candidate} G10 denominator drifted")
        comparison = _mapping(g10.get("comparison"), f"{candidate} G10 comparison")
        passes_g10 = comparison.get("passes")
        if not isinstance(passes_g10, bool):
            raise TypeError(f"{candidate} G10 pass result must be boolean")
        if comparison.get("threshold") != locked_g10.get("threshold"):
            raise ValueError(f"{candidate} G10 threshold drifted")

        failed = g1_g9.get("failed_gate_ids")
        if not isinstance(failed, list) or not all(isinstance(item, str) for item in failed):
            raise TypeError(f"{candidate} failed G1-G9 gate IDs must be a list")
        failed_all = list(failed)
        if not passes_g10:
            failed_all.append("G10_full_training_variation")
        passes_g1_g9 = g1_g9.get("all_applicable_gates_passed")
        if not isinstance(passes_g1_g9, bool):
            raise TypeError(f"{candidate} G1-G9 pass result must be boolean")

        merged[candidate] = {
            "all_ten_universal_gates_passed": passes_g1_g9 and passes_g10,
            "failed_gate_ids": failed_all,
            "gates_g1_g9": {
                "active_scope_completions": 11_504,
                "active_scope_groups": 1_438,
                "passed": passes_g1_g9,
            },
            "gate_g10": {
                "full_scope_completions": 25_920,
                "full_scope_groups": 3_240,
                "maximum_allowed_zero_variance_groups": comparison.get(
                    "maximum_allowed_zero_variance_groups"
                ),
                "passed": passes_g10,
                "threshold": comparison.get("threshold"),
                "zero_variance_groups": g10.get("zero_variance_groups"),
                "zero_variance_share": g10.get("zero_variance_share"),
            },
        }

    return {
        "version": VERSION,
        "status": "all_universal_gates_merged_pending_complexity_selection",
        "role": "training_only_offline_reward_universal_eligibility",
        "candidate_order": list(CANDIDATES),
        "inputs": dict(identities),
        "scope_guardrail": {
            "active_and_full_scope_denominators_combined": False,
            "active_scope": {"completions": 11_504, "groups": 1_438},
            "full_training_scope": {"completions": 25_920, "groups": 3_240},
            "join_key": "candidate_id",
            "one_to_one_candidate_join": True,
        },
        "candidate_results": merged,
        "summary": {
            candidate: {
                "all_ten_universal_gates_passed": result[
                    "all_ten_universal_gates_passed"
                ],
                "failed_gate_ids": result["failed_gate_ids"],
            }
            for candidate, result in merged.items()
        },
        "selection_boundary": {
            "all_universal_gates_merged": True,
            "candidate_rankings_calculated": False,
            "complexity_aware_selection_applied": False,
            "gpu_training_authorized": False,
            "winner_selected": False,
        },
    }
