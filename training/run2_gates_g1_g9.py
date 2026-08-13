"""Pure, fail-closed evaluator for GRPO Run 2 universal Gates G1-G9.

This module consumes already-aggregated D3 evidence.  It does not read files,
rank candidates, select a reward, or authorize GPU training.  Gate G10 remains
a separate full-training-scope result and is intentionally not merged here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


VERSION = "grpo-run2-gates-g1-g9-decision-v1"
CANDIDATES = ("U", "UA", "CB")
EXPECTED_GATE_IDS = tuple(f"G{i}_" for i in range(1, 10))
EXPECTED_VERIFICATIONS = {
    "all_directional_alignments",
    "all_paired_bootstraps",
    "all_segment_memberships_and_summaries",
    "field_and_class_contribution_dominance",
    "group_and_completion_denominators",
    "harmful_coverage_preferences",
    "no_selection_boundary",
    "reward_shapes_and_distributions",
    "source_identities",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _gate_contracts(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = _sequence(contract.get("universal_acceptance_gates"), "gate contracts")
    gates: dict[str, Mapping[str, Any]] = {}
    for position, item in enumerate(raw):
        gate = _mapping(item, f"gate contract {position}")
        gate_id = gate.get("id")
        if not isinstance(gate_id, str) or gate_id in gates:
            raise ValueError("gate IDs must be unique strings")
        gates[gate_id] = gate
    selected: dict[str, Mapping[str, Any]] = {}
    for prefix in EXPECTED_GATE_IDS:
        matches = [gate_id for gate_id in gates if gate_id.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one locked {prefix.rstrip('_')} contract")
        selected[matches[0]] = gates[matches[0]]
    return selected


def _result(
    gate: Mapping[str, Any],
    *,
    observed: Any,
    passed: bool | None,
    checks: Mapping[str, bool] | None = None,
    applicable: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "applicable": applicable,
        "gate_id": gate["id"],
        "metric": gate["metric"],
        "observed": observed,
        "operator": gate["operator"],
        "passed": passed,
    }
    for key in (
        "threshold",
        "point_threshold",
        "ci_lower_threshold",
        "ci_upper_threshold",
        "minimum_groups",
    ):
        if key in gate:
            result[key] = gate[key]
    if checks is not None:
        result["checks"] = dict(checks)
    return result


def _paired_delta(
    paired: Mapping[str, Any], candidate: str, metric: str
) -> Mapping[str, Any]:
    candidate_paired = _mapping(paired.get(candidate), f"{candidate} paired metrics")
    metric_result = _mapping(candidate_paired.get(metric), f"{candidate} {metric}")
    return _mapping(
        metric_result.get("delta_candidate_minus_baseline"),
        f"{candidate} {metric} delta",
    )


def evaluate_gates_g1_g9(
    *,
    comparison_contract: Mapping[str, Any],
    d3_analysis: Mapping[str, Any],
    independent_verification: Mapping[str, Any],
    cpu_test_evidence: Mapping[str, Any],
    input_identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply locked Gates G1-G9 without ranking or selecting candidates."""

    contract = _mapping(comparison_contract, "comparison contract")
    d3 = _mapping(d3_analysis, "D3 analysis")
    verification = _mapping(independent_verification, "independent verification")
    tests = _mapping(cpu_test_evidence, "CPU test evidence")
    identities = _mapping(input_identities, "input identities")
    gates = _gate_contracts(contract)
    gate_by_number = {
        int(gate_id[1 : gate_id.index("_")]): gate
        for gate_id, gate in gates.items()
    }

    if d3.get("status") != "aggregate_candidate_analysis_completed_pending_gates":
        raise ValueError("D3 analysis is not in the locked pending-gates state")
    d3_boundary = _mapping(d3.get("selection_boundary"), "D3 selection boundary")
    if d3_boundary.get("acceptance_gates_applied") is not False:
        raise ValueError("D3 source already claims gates were applied")
    if d3_boundary.get("candidate_rankings_calculated") is not False:
        raise ValueError("D3 source already claims candidate ranking")
    if d3_boundary.get("winner_selected") is not False:
        raise ValueError("D3 source already claims a winner")

    analysis = _mapping(d3.get("analysis_core"), "D3 analysis core")
    groups = _integer(analysis.get("groups"), "D3 groups")
    completions = _integer(analysis.get("completions"), "D3 completions")
    if groups != 1_438 or completions != 11_504:
        raise ValueError("D3 active-pool denominators drifted")
    summaries = _mapping(analysis.get("reward_summaries"), "reward summaries")
    paired = _mapping(
        analysis.get("paired_candidate_minus_original"), "paired comparisons"
    )
    diagnostics = _mapping(
        d3.get("contribution_and_segment_diagnostics"), "D3 diagnostics"
    )
    dominance = _mapping(diagnostics.get("dominance"), "dominance diagnostics")
    fields = _mapping(dominance.get("field_contributions"), "field contributions")
    cb_classes = _mapping(
        dominance.get("cb_class_contributions"), "CB class contributions"
    )

    verified = _mapping(verification.get("verified"), "verification flags")
    if set(verified) != EXPECTED_VERIFICATIONS:
        raise ValueError("independent verification flag set drifted")
    verification_passed = (
        verification.get("status") == "independent_d3_verification_passed"
        and all(value is True for value in verified.values())
    )
    test_count = _integer(tests.get("passed_tests"), "passed test count")
    cpu_tests_passed = tests.get("exit_code") == 0 and tests.get("passed") is True
    if test_count <= 0:
        raise ValueError("passed test count must be positive")
    g1_checks = {
        "cpu_tests_passed": cpu_tests_passed,
        "independent_rebuild_passed": verification_passed,
        "input_identities_verified": identities.get("all_verified") is True,
    }
    g1_passed = all(g1_checks.values())

    candidate_results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        summary = _mapping(summaries.get(candidate), f"{candidate} reward summary")
        if _integer(summary.get("groups"), f"{candidate} groups") != groups:
            raise ValueError(f"{candidate} group denominator drifted")

        zero_share = _finite_number(
            summary.get("zero_variance_share"), f"{candidate} zero-variance share"
        )
        distinct_share = _finite_number(
            summary.get("groups_with_at_least_three_levels_share"),
            f"{candidate} distinct-level share",
        )
        large_tie_share = _finite_number(
            summary.get("groups_with_largest_tie_at_least_six_share"),
            f"{candidate} large-tie share",
        )

        g5_delta = _paired_delta(paired, candidate, "pairwise_discrimination")
        g6_delta = _paired_delta(
            paired, candidate, "canonical_known_utility_net_alignment"
        )
        g7_delta = _paired_delta(paired, candidate, "harmful_coverage_preference")

        def delta_values(
            delta: Mapping[str, Any], label: str
        ) -> tuple[float, float, float, int]:
            point = _finite_number(delta.get("point"), f"{label} point")
            ci = _sequence(delta.get("ci"), f"{label} interval")
            if len(ci) != 2:
                raise ValueError(f"{label} interval must contain two values")
            lower = _finite_number(ci[0], f"{label} lower bound")
            upper = _finite_number(ci[1], f"{label} upper bound")
            metric_name = {
                id(g5_delta): "pairwise_discrimination",
                id(g6_delta): "canonical_known_utility_net_alignment",
                id(g7_delta): "harmful_coverage_preference",
            }[id(delta)]
            metric_parent = _mapping(
                _mapping(paired[candidate], candidate)[metric_name], metric_name
            )
            n_groups = _integer(
                metric_parent.get("groups_per_replicate"), f"{label} groups"
            )
            return point, lower, upper, n_groups

        g5_point, g5_lower, g5_upper, g5_groups = delta_values(g5_delta, "G5")
        g6_point, g6_lower, g6_upper, g6_groups = delta_values(g6_delta, "G6")
        g7_point, g7_lower, g7_upper, g7_groups = delta_values(g7_delta, "G7")

        field_result = _mapping(fields.get(candidate), f"{candidate} field dominance")
        field_share = _finite_number(
            field_result.get("largest_field_share"), f"{candidate} largest field share"
        )

        results: dict[str, Any] = {}
        results[gate_by_number[1]["id"]] = _result(
            gate_by_number[1], observed=g1_checks, checks=g1_checks, passed=g1_passed
        )
        results[gate_by_number[2]["id"]] = _result(
            gate_by_number[2],
            observed=zero_share,
            passed=zero_share == float(gate_by_number[2]["threshold"]),
        )
        results[gate_by_number[3]["id"]] = _result(
            gate_by_number[3],
            observed=distinct_share,
            passed=distinct_share >= float(gate_by_number[3]["threshold"]),
        )
        results[gate_by_number[4]["id"]] = _result(
            gate_by_number[4],
            observed=large_tie_share,
            passed=large_tie_share <= float(gate_by_number[4]["threshold"]),
        )

        g5_checks = {
            "minimum_groups_met": g5_groups >= int(gate_by_number[5]["minimum_groups"]),
            "point_threshold_met": g5_point >= float(gate_by_number[5]["point_threshold"]),
            "ci_lower_strictly_above_threshold": g5_lower
            > float(gate_by_number[5]["ci_lower_threshold"]),
        }
        results[gate_by_number[5]["id"]] = _result(
            gate_by_number[5],
            observed={"ci": [g5_lower, g5_upper], "groups": g5_groups, "point": g5_point},
            checks=g5_checks,
            passed=all(g5_checks.values()),
        )
        g6_checks = {
            "minimum_groups_met": g6_groups >= int(gate_by_number[6]["minimum_groups"]),
            "point_threshold_met": g6_point >= float(gate_by_number[6]["point_threshold"]),
            "ci_lower_noninferiority_met": g6_lower
            >= float(gate_by_number[6]["ci_lower_threshold"]),
        }
        results[gate_by_number[6]["id"]] = _result(
            gate_by_number[6],
            observed={"ci": [g6_lower, g6_upper], "groups": g6_groups, "point": g6_point},
            checks=g6_checks,
            passed=all(g6_checks.values()),
        )
        g7_checks = {
            "minimum_groups_met": g7_groups >= int(gate_by_number[7]["minimum_groups"]),
            "point_threshold_met": g7_point <= float(gate_by_number[7]["point_threshold"]),
            "ci_upper_safety_met": g7_upper
            <= float(gate_by_number[7]["ci_upper_threshold"]),
        }
        results[gate_by_number[7]["id"]] = _result(
            gate_by_number[7],
            observed={"ci": [g7_lower, g7_upper], "groups": g7_groups, "point": g7_point},
            checks=g7_checks,
            passed=all(g7_checks.values()),
        )
        results[gate_by_number[8]["id"]] = _result(
            gate_by_number[8],
            observed={
                "largest_field": field_result.get("largest_field"),
                "share": field_share,
            },
            passed=field_share <= float(gate_by_number[8]["threshold"]),
        )
        if candidate == "CB":
            class_share = _finite_number(
                cb_classes.get("largest_class_share"), "CB largest class share"
            )
            results[gate_by_number[9]["id"]] = _result(
                gate_by_number[9],
                observed={
                    "largest_class": cb_classes.get("largest_class"),
                    "share": class_share,
                },
                passed=class_share <= float(gate_by_number[9]["threshold"]),
            )
        else:
            results[gate_by_number[9]["id"]] = _result(
                gate_by_number[9], observed=None, passed=None, applicable=False
            )

        applicable = [result for result in results.values() if result["applicable"]]
        candidate_results[candidate] = {
            "all_applicable_gates_passed": all(
                result["passed"] is True for result in applicable
            ),
            "applicable_gate_count": len(applicable),
            "failed_gate_ids": [
                result["gate_id"] for result in applicable if result["passed"] is False
            ],
            "gate_results": results,
        }

    return {
        "version": VERSION,
        "status": "gates_g1_g9_applied_pending_g10_merge_and_selection",
        "role": "training_only_offline_reward_gate_decision",
        "candidate_order": list(CANDIDATES),
        "lineage": {"completions": completions, "groups": groups},
        "inputs": dict(identities),
        "cpu_test_evidence": dict(tests),
        "candidate_results": candidate_results,
        "summary": {
            candidate: {
                "all_applicable_gates_passed": result["all_applicable_gates_passed"],
                "failed_gate_ids": result["failed_gate_ids"],
            }
            for candidate, result in candidate_results.items()
        },
        "selection_boundary": {
            "candidate_rankings_calculated": False,
            "complexity_aware_selection_applied": False,
            "gate_g10_merged": False,
            "gates_g1_through_g9_applied": True,
            "gpu_training_authorized": False,
            "winner_selected": False,
        },
    }
