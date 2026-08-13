from __future__ import annotations

import pytest

from training.run2_confirmation_contract import LOCKED_INPUTS, build_contract


def _inputs() -> dict:
    return {name: dict(value) for name, value in LOCKED_INPUTS.items()}


def test_contract_locks_selection_before_labeling() -> None:
    result = build_contract(inputs=_inputs())

    assert result["selection"]["target_rows"] == 400
    assert result["acquisition"]["minimum_family_clean_candidate_rows"] == 800
    assert result["selection"]["membership_locked_before_labeling"] is True
    assert result["selection"]["uses_frontier_labels"] is False
    assert result["exclusions"]["normalized_family_overlap_allowed"] == 0


def test_complete_review_and_independent_audit_are_locked() -> None:
    result = build_contract(inputs=_inputs())

    assert result["human_review"]["cells_required"] == 6_000
    assert result["human_review"]["all_cells_reviewed"] is True
    assert result["human_review"]["independent_second_review"]["rows"] == 40
    assert result["human_review"]["unresolved_cells_allowed"] == 0


def test_support_shortfalls_cannot_change_membership() -> None:
    result = build_contract(inputs=_inputs())

    assert result["support_policy"]["membership_may_change_after_labels"] is False
    assert "do not replace" in result["support_policy"]["shortfalls"]


def test_contract_does_not_authorize_execution() -> None:
    boundary = build_contract(inputs=_inputs())["execution_boundary"]

    assert boundary
    assert all(value is False for value in boundary.values())


def test_missing_locked_input_fails_closed() -> None:
    inputs = _inputs()
    inputs.pop("pack_rules")

    with pytest.raises(ValueError, match="input set drifted"):
        build_contract(inputs=inputs)
