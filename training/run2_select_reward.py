"""Apply the locked GRPO Run 2 complexity-aware reward selection rule.

This is an offline reward-design decision.  It consumes training-only evidence,
does not evaluate a trained policy, and never authorizes GPU execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from training.analyze_run2_candidates import paired_group_bootstrap


VERSION = "grpo-run2-complexity-aware-selection-v1"
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


def _cb_selection_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    hierarchy = _mapping(contract.get("selection_hierarchy"), "selection hierarchy")
    steps = hierarchy.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise TypeError("selection steps must be a sequence")
    matches = [
        _mapping(step, "selection step")
        for step in steps
        if isinstance(step, Mapping) and step.get("order") == 3
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one locked CB selection step")
    step = matches[0]
    if step.get("comparison") != "CB versus the surviving uniform candidate":
        raise ValueError("CB comparison identity drifted")
    return _mapping(step.get("choose_more_complex_if"), "CB selection thresholds")


def _delta(result: Mapping[str, Any], label: str) -> tuple[float, float, float]:
    delta = _mapping(result.get("delta_candidate_minus_baseline"), f"{label} delta")
    point = delta.get("point")
    interval = delta.get("ci")
    if not isinstance(point, (int, float)) or isinstance(point, bool):
        raise TypeError(f"{label} point must be numeric")
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
        raise TypeError(f"{label} interval must be a sequence")
    if len(interval) != 2:
        raise ValueError(f"{label} interval must have two values")
    values = (float(point), float(interval[0]), float(interval[1]))
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError(f"{label} values must be finite")
    return values


def evaluate_cb_upgrade(
    *,
    comparisons: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    dominance_gates_pass: bool,
) -> dict[str, Any]:
    """Evaluate whether CB earns its extra complexity over eligible UA."""

    required = {
        "canonical_known_utility_alignment",
        "class_balanced_known_utility_alignment",
        "harmful_coverage",
        "pairwise_discrimination",
    }
    if set(comparisons) != required:
        raise ValueError("CB-versus-UA comparison metric set drifted")
    cb_point, cb_lower, cb_upper = _delta(
        comparisons["class_balanced_known_utility_alignment"],
        "class-balanced alignment",
    )
    known_point, known_lower, known_upper = _delta(
        comparisons["canonical_known_utility_alignment"], "known alignment"
    )
    resolution_point, resolution_lower, resolution_upper = _delta(
        comparisons["pairwise_discrimination"], "pairwise discrimination"
    )
    harmful_point, harmful_lower, harmful_upper = _delta(
        comparisons["harmful_coverage"], "harmful coverage"
    )

    checks = {
        "canonical_known_utility_ci_lower_noninferior": known_lower
        >= float(thresholds["canonical_known_utility_alignment_ci_lower_noninferiority"]),
        "class_balanced_alignment_ci_lower_above_zero": cb_lower
        > float(thresholds["class_balanced_known_utility_alignment_ci_lower_minimum"]),
        "class_balanced_alignment_point_gain_met": cb_point
        >= float(thresholds["class_balanced_known_utility_alignment_point_delta_minimum"]),
        "field_and_class_dominance_gates_pass": dominance_gates_pass
        is bool(thresholds["field_and_class_dominance_gates_pass"]),
        "harmful_coverage_ci_upper_noninferior": harmful_upper
        <= float(thresholds["harmful_coverage_ci_upper_noninferiority"]),
        "pairwise_discrimination_ci_lower_noninferior": resolution_lower
        >= float(thresholds["pairwise_discrimination_ci_lower_noninferiority"]),
    }
    return {
        "all_upgrade_conditions_passed": all(checks.values()),
        "checks": checks,
        "observed_deltas_cb_minus_ua": {
            "canonical_known_utility_alignment": {
                "ci": [known_lower, known_upper],
                "point": known_point,
            },
            "class_balanced_known_utility_alignment": {
                "ci": [cb_lower, cb_upper],
                "point": cb_point,
            },
            "harmful_coverage": {
                "ci": [harmful_lower, harmful_upper],
                "point": harmful_point,
            },
            "pairwise_discrimination": {
                "ci": [resolution_lower, resolution_upper],
                "point": resolution_point,
            },
        },
        "thresholds": dict(thresholds),
    }


def _alignment_values(
    analysis: Mapping[str, Any], candidate: str, target: str
) -> Mapping[str, float]:
    alignments = _mapping(analysis.get("directional_alignments"), "alignments")
    candidate_values = _mapping(alignments.get(candidate), f"{candidate} alignments")
    metric = _mapping(candidate_values.get(target), f"{candidate} {target}")
    return _mapping(
        metric.get("group_values_for_paired_analysis"),
        f"{candidate} {target} paired values",
    )


def _resolution_values(
    analysis: Mapping[str, Any], candidate: str
) -> dict[str, float]:
    summaries = _mapping(analysis.get("reward_summaries"), "reward summaries")
    summary = _mapping(summaries.get(candidate), f"{candidate} reward summary")
    values = _mapping(
        summary.get("group_values_for_paired_analysis"),
        f"{candidate} reward paired values",
    )
    return {
        group_id: float(
            _mapping(value, f"{candidate} {group_id} reward value").get(
                "pairwise_discrimination_rate"
            )
        )
        for group_id, value in values.items()
    }


def _harmful_values(
    analysis: Mapping[str, Any], candidate: str
) -> Mapping[str, float]:
    harmful = _mapping(analysis.get("harmful_coverage"), "harmful coverage")
    candidate_values = _mapping(harmful.get(candidate), f"{candidate} harmful coverage")
    return _mapping(
        candidate_values.get("group_values_for_paired_analysis"),
        f"{candidate} harmful paired values",
    )


def _paired(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("direct selection comparison group IDs must match exactly")
    return paired_group_bootstrap(
        baseline,
        candidate,
        seed=seed,
        replicates=replicates,
        confidence=confidence,
    )


def select_run2_reward(
    *,
    comparison_contract: Mapping[str, Any],
    universal_gate_decision: Mapping[str, Any],
    d3_analysis: Mapping[str, Any],
    input_identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Select UA or CB using only the locked training-only decision contract."""

    contract = _mapping(comparison_contract, "comparison contract")
    universal = _mapping(universal_gate_decision, "universal gate decision")
    d3 = _mapping(d3_analysis, "D3 analysis")
    identities = _mapping(input_identities, "input identities")
    if identities.get("all_verified") is not True:
        raise ValueError("input identities were not verified")
    if universal.get("status") != "all_universal_gates_merged_pending_complexity_selection":
        raise ValueError("universal gate decision status drifted")
    _candidate_order(universal.get("candidate_order"), "universal candidate order")
    universal_boundary = _mapping(
        universal.get("selection_boundary"), "universal selection boundary"
    )
    if universal_boundary.get("all_universal_gates_merged") is not True:
        raise ValueError("all universal gates were not merged")
    for key in (
        "candidate_rankings_calculated",
        "complexity_aware_selection_applied",
        "gpu_training_authorized",
        "winner_selected",
    ):
        if universal_boundary.get(key) is not False:
            raise ValueError(f"universal selection boundary {key} drifted")

    results = _mapping(universal.get("candidate_results"), "universal results")
    if set(results) != set(CANDIDATES):
        raise ValueError("universal candidate result set drifted")
    eligibility = {
        candidate: _mapping(results[candidate], f"{candidate} universal result").get(
            "all_ten_universal_gates_passed"
        )
        for candidate in CANDIDATES
    }
    if any(not isinstance(value, bool) for value in eligibility.values()):
        raise TypeError("universal eligibility values must be boolean")
    if eligibility != {"U": False, "UA": True, "CB": True}:
        raise ValueError("locked real decision path eligibility drifted")

    if d3.get("status") != "aggregate_candidate_analysis_completed_pending_gates":
        raise ValueError("D3 analysis status drifted")
    settings = _mapping(d3.get("settings"), "D3 settings")
    if settings.get("settings_locked_to_production_contract") is not True:
        raise ValueError("D3 settings are not locked")
    seed = settings.get("bootstrap_seed")
    replicates = settings.get("bootstrap_replicates")
    confidence = settings.get("confidence")
    if seed != 20260812 or replicates != 10_000 or confidence != 0.95:
        raise ValueError("selection bootstrap settings drifted")
    analysis = _mapping(d3.get("analysis_core"), "D3 analysis core")

    comparisons = {
        "class_balanced_known_utility_alignment": _paired(
            _alignment_values(analysis, "UA", "class_balanced_known_utility"),
            _alignment_values(analysis, "CB", "class_balanced_known_utility"),
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        ),
        "canonical_known_utility_alignment": _paired(
            _alignment_values(analysis, "UA", "canonical_known_utility"),
            _alignment_values(analysis, "CB", "canonical_known_utility"),
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        ),
        "pairwise_discrimination": _paired(
            _resolution_values(analysis, "UA"),
            _resolution_values(analysis, "CB"),
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        ),
        "harmful_coverage": _paired(
            _harmful_values(analysis, "UA"),
            _harmful_values(analysis, "CB"),
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        ),
    }
    thresholds = _cb_selection_contract(contract)
    decision = evaluate_cb_upgrade(
        comparisons=comparisons,
        thresholds=thresholds,
        dominance_gates_pass=True,
    )
    selected = "CB" if decision["all_upgrade_conditions_passed"] else "UA"

    return {
        "version": VERSION,
        "status": "run2_offline_reward_selected_pending_run_contract",
        "role": "training_only_offline_reward_design_selection",
        "candidate_order": list(CANDIDATES),
        "inputs": dict(identities),
        "universal_eligibility": eligibility,
        "selection_steps": {
            "1_universal_gate_filter": {
                "eligible": [candidate for candidate in CANDIDATES if eligibility[candidate]],
                "excluded": [candidate for candidate in CANDIDATES if not eligibility[candidate]],
                "U_failed_gate_ids": results["U"].get("failed_gate_ids"),
            },
            "2_ua_versus_u": {
                "applied": False,
                "reason": "U was already excluded by the universal gate filter; UA is the surviving uniform candidate",
                "surviving_uniform_candidate": "UA",
            },
            "3_cb_versus_ua": {
                "applied": True,
                "baseline_candidate": "UA",
                "more_complex_candidate": "CB",
                "decision": decision,
                "paired_bootstraps": comparisons,
            },
        },
        "selected_candidate": selected,
        "selection_reason": (
            "CB earned its extra class-balanced complexity under every locked condition"
            if selected == "CB"
            else "CB did not satisfy every locked upgrade condition, so the simpler eligible UA policy wins"
        ),
        "claim_boundary": (
            "offline reward-design selection only; this does not establish that GRPO "
            "will improve model quality"
        ),
        "selection_boundary": {
            "complexity_aware_selection_applied": True,
            "gpu_training_authorized": False,
            "run_contract_locked": False,
            "winner_selected": True,
        },
    }
