from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_select_reward import evaluate_cb_upgrade, select_run2_reward


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> dict:
    return {
        "comparison_contract": json.loads(
            (ROOT / "runs/grpo-run2-comparison-contract.json").read_text()
        ),
        "universal_gate_decision": json.loads(
            (ROOT / "runs/grpo-run2-d4-universal-gate-decision.json").read_text()
        ),
        "d3_analysis": json.loads(
            (ROOT / "runs/grpo-run2-d3-candidate-analysis.json").read_text()
        ),
        "input_identities": {"all_verified": True},
    }


def _comparison(point: float, lower: float, upper: float) -> dict:
    return {"delta_candidate_minus_baseline": {"point": point, "ci": [lower, upper]}}


def _thresholds() -> dict:
    contract = _inputs()["comparison_contract"]
    return contract["selection_hierarchy"]["steps"][2]["choose_more_complex_if"]


def test_real_evidence_selects_simpler_eligible_ua() -> None:
    result = select_run2_reward(**_inputs())
    decision = result["selection_steps"]["3_cb_versus_ua"]["decision"]

    assert result["selected_candidate"] == "UA"
    assert decision["all_upgrade_conditions_passed"] is False
    assert decision["checks"]["class_balanced_alignment_point_gain_met"] is False
    assert decision["checks"]["harmful_coverage_ci_upper_noninferior"] is False
    assert result["selection_boundary"] == {
        "complexity_aware_selection_applied": True,
        "gpu_training_authorized": False,
        "run_contract_locked": False,
        "winner_selected": True,
    }


def test_cb_upgrade_requires_every_condition() -> None:
    comparisons = {
        "canonical_known_utility_alignment": _comparison(0.0, -0.01, 0.01),
        "class_balanced_known_utility_alignment": _comparison(0.03, 0.001, 0.05),
        "harmful_coverage": _comparison(0.0, -0.01, 0.02),
        "pairwise_discrimination": _comparison(0.0, -0.02, 0.01),
    }
    result = evaluate_cb_upgrade(
        comparisons=comparisons,
        thresholds=_thresholds(),
        dominance_gates_pass=True,
    )

    assert result["all_upgrade_conditions_passed"] is True
    assert all(result["checks"].values())


def test_cb_point_gain_below_three_points_fails() -> None:
    comparisons = {
        "canonical_known_utility_alignment": _comparison(0.0, 0.0, 0.0),
        "class_balanced_known_utility_alignment": _comparison(0.0299, 0.001, 0.04),
        "harmful_coverage": _comparison(0.0, 0.0, 0.0),
        "pairwise_discrimination": _comparison(0.0, 0.0, 0.0),
    }
    result = evaluate_cb_upgrade(
        comparisons=comparisons,
        thresholds=_thresholds(),
        dominance_gates_pass=True,
    )

    assert result["checks"]["class_balanced_alignment_point_gain_met"] is False
    assert result["all_upgrade_conditions_passed"] is False


def test_selection_fails_closed_if_universal_eligibility_drifts() -> None:
    inputs = _inputs()
    inputs["universal_gate_decision"]["candidate_results"]["U"][
        "all_ten_universal_gates_passed"
    ] = True

    with pytest.raises(ValueError, match="eligibility drifted"):
        select_run2_reward(**inputs)


def test_direct_comparison_group_mismatch_fails_closed() -> None:
    inputs = _inputs()
    values = inputs["d3_analysis"]["analysis_core"]["harmful_coverage"]["CB"][
        "group_values_for_paired_analysis"
    ]
    values.pop(next(iter(values)))

    with pytest.raises(ValueError, match="group IDs must match exactly"):
        select_run2_reward(**inputs)
