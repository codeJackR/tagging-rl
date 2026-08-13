from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_merge_universal_gates import merge_universal_gates


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> dict:
    return {
        "comparison_contract": json.loads(
            (ROOT / "runs/grpo-run2-comparison-contract.json").read_text()
        ),
        "gates_g1_g9": json.loads(
            (ROOT / "runs/grpo-run2-d4-gates-g1-g9.json").read_text()
        ),
        "gate_g10": json.loads(
            (ROOT / "runs/grpo-run2-gate-g10-result.json").read_text()
        ),
        "input_identities": {"all_verified": True},
    }


def test_real_evidence_merges_without_ranking_or_selection() -> None:
    result = merge_universal_gates(**_inputs())

    assert result["summary"] == {
        "U": {
            "all_ten_universal_gates_passed": False,
            "failed_gate_ids": ["G3_distinct_levels"],
        },
        "UA": {"all_ten_universal_gates_passed": True, "failed_gate_ids": []},
        "CB": {"all_ten_universal_gates_passed": True, "failed_gate_ids": []},
    }
    assert result["selection_boundary"] == {
        "all_universal_gates_merged": True,
        "candidate_rankings_calculated": False,
        "complexity_aware_selection_applied": False,
        "gpu_training_authorized": False,
        "winner_selected": False,
    }


def test_scopes_remain_separate() -> None:
    result = merge_universal_gates(**_inputs())
    guardrail = result["scope_guardrail"]

    assert guardrail["active_and_full_scope_denominators_combined"] is False
    assert guardrail["active_scope"]["groups"] == 1_438
    assert guardrail["full_training_scope"]["groups"] == 3_240
    assert guardrail["one_to_one_candidate_join"] is True


def test_candidate_identity_mismatch_fails_closed() -> None:
    inputs = _inputs()
    inputs["gate_g10"]["candidate_results"]["UA"]["candidate"] = "U"

    with pytest.raises(ValueError, match="UA G10 identity drifted"):
        merge_universal_gates(**inputs)


def test_locked_g10_threshold_mismatch_fails_closed() -> None:
    inputs = _inputs()
    inputs["gate_g10"]["gate_contract"]["threshold"] = 0.41

    with pytest.raises(ValueError, match="threshold disagrees"):
        merge_universal_gates(**inputs)


def test_failed_g10_is_added_to_failed_gate_ids() -> None:
    inputs = _inputs()
    inputs["gate_g10"]["candidate_results"]["UA"]["comparison"]["passes"] = False

    result = merge_universal_gates(**inputs)
    assert result["summary"]["UA"] == {
        "all_ten_universal_gates_passed": False,
        "failed_gate_ids": ["G10_full_training_variation"],
    }


def test_preexisting_ranking_claim_fails_closed() -> None:
    inputs = _inputs()
    inputs["gates_g1_g9"]["selection_boundary"][
        "candidate_rankings_calculated"
    ] = True

    with pytest.raises(ValueError, match="G1-G9 selection boundary drifted"):
        merge_universal_gates(**inputs)
