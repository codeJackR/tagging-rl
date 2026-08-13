from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_gates_g1_g9 import evaluate_gates_g1_g9


ROOT = Path(__file__).resolve().parents[1]


def _real_inputs() -> dict:
    return {
        "comparison_contract": json.loads(
            (ROOT / "runs/grpo-run2-comparison-contract.json").read_text()
        ),
        "d3_analysis": json.loads(
            (ROOT / "runs/grpo-run2-d3-candidate-analysis.json").read_text()
        ),
        "independent_verification": json.loads(
            (ROOT / "runs/grpo-run2-d3-independent-verification.json").read_text()
        ),
        "cpu_test_evidence": {
            "command": ".venv/bin/python -m pytest -q",
            "exit_code": 0,
            "passed": True,
            "passed_tests": 1,
        },
        "input_identities": {"all_verified": True},
    }


def test_real_evidence_applies_gates_without_ranking_or_selection() -> None:
    result = evaluate_gates_g1_g9(**_real_inputs())

    assert result["summary"] == {
        "U": {
            "all_applicable_gates_passed": False,
            "failed_gate_ids": ["G3_distinct_levels"],
        },
        "UA": {"all_applicable_gates_passed": True, "failed_gate_ids": []},
        "CB": {"all_applicable_gates_passed": True, "failed_gate_ids": []},
    }
    assert result["selection_boundary"] == {
        "candidate_rankings_calculated": False,
        "complexity_aware_selection_applied": False,
        "gate_g10_merged": False,
        "gates_g1_through_g9_applied": True,
        "gpu_training_authorized": False,
        "winner_selected": False,
    }


def test_g5_interval_lower_bound_is_strictly_above_zero() -> None:
    inputs = _real_inputs()
    delta = inputs["d3_analysis"]["analysis_core"][
        "paired_candidate_minus_original"
    ]["UA"]["pairwise_discrimination"]["delta_candidate_minus_baseline"]
    delta["ci"][0] = 0.0

    result = evaluate_gates_g1_g9(**inputs)
    gate = result["candidate_results"]["UA"]["gate_results"][
        "G5_pairwise_resolution"
    ]
    assert gate["passed"] is False
    assert gate["checks"]["ci_lower_strictly_above_threshold"] is False


def test_g9_is_only_applicable_to_cb() -> None:
    result = evaluate_gates_g1_g9(**_real_inputs())

    for candidate in ("U", "UA"):
        gate = result["candidate_results"][candidate]["gate_results"][
            "G9_cb_class_dominance"
        ]
        assert gate["applicable"] is False
        assert gate["passed"] is None
    assert (
        result["candidate_results"]["CB"]["gate_results"][
            "G9_cb_class_dominance"
        ]["applicable"]
        is True
    )


def test_integrity_gate_fails_closed_on_failed_independent_check() -> None:
    inputs = _real_inputs()
    inputs["independent_verification"]["verified"]["source_identities"] = False

    result = evaluate_gates_g1_g9(**inputs)
    for candidate in ("U", "UA", "CB"):
        assert result["candidate_results"][candidate]["gate_results"][
            "G1_integrity_and_tests"
        ]["passed"] is False


def test_missing_gate_contract_fails_closed() -> None:
    inputs = _real_inputs()
    gates = inputs["comparison_contract"]["universal_acceptance_gates"]
    inputs["comparison_contract"]["universal_acceptance_gates"] = [
        gate for gate in gates if gate["id"] != "G7_harmful_coverage"
    ]

    with pytest.raises(ValueError, match="exactly one locked G7 contract"):
        evaluate_gates_g1_g9(**inputs)


def test_known_alignment_point_threshold_is_nonnegative() -> None:
    inputs = _real_inputs()
    delta = inputs["d3_analysis"]["analysis_core"][
        "paired_candidate_minus_original"
    ]["CB"]["canonical_known_utility_net_alignment"][
        "delta_candidate_minus_baseline"
    ]
    delta["point"] = -0.0001

    result = evaluate_gates_g1_g9(**inputs)
    assert result["candidate_results"]["CB"]["gate_results"][
        "G6_known_utility_alignment"
    ]["passed"] is False
