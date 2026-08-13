from __future__ import annotations

import json
import math

import pytest

from training.reward_scale_contract import (
    ABSTAIN,
    CLASS_WEIGHT_MAX,
    CLASS_WEIGHT_MIN,
    CORRECT,
    KNOWN_CELLS,
    MALFORMED_FLOOR,
    RULE_MAXIMUM_COST,
    RULE_VIOLATION_CAP,
    TOTAL_CELLS,
    UNKNOWN_CELLS,
    WRONG,
    build_contract,
    class_weight,
    combine_known_unknown,
    details_payoff,
    normalized_mean,
    rule_adjustment,
    valid_total,
)


def test_symbolic_order_is_preserved_on_the_numeric_scale():
    assert CORRECT > ABSTAIN > WRONG > MALFORMED_FLOOR
    assert details_payoff(0.0) == WRONG
    assert details_payoff(0.5) == ABSTAIN
    assert details_payoff(1.0) == CORRECT
    assert details_payoff(0.25) < details_payoff(0.75)
    assert combine_known_unknown(0.0, 1.0) > combine_known_unknown(0.0, -1.0)


def test_field_normalization_preserves_local_direction_at_every_density():
    for count in range(1, 16):
        baseline = normalized_mean([ABSTAIN] * count)
        assert normalized_mean([CORRECT] + [ABSTAIN] * (count - 1)) > baseline
        assert normalized_mean([WRONG] + [ABSTAIN] * (count - 1)) < baseline

    weights = [CLASS_WEIGHT_MIN, 1.0, CLASS_WEIGHT_MAX]
    baseline = normalized_mean([ABSTAIN] * 3, weights)
    for index in range(3):
        correct = [ABSTAIN] * 3
        wrong = [ABSTAIN] * 3
        correct[index] = CORRECT
        wrong[index] = WRONG
        assert normalized_mean(correct, weights) > baseline
        assert normalized_mean(wrong, weights) < baseline


def test_class_weights_are_mild_and_rule_cost_is_bounded():
    assert class_weight(1, 100) == CLASS_WEIGHT_MAX
    assert class_weight(100, 100) == 1.0
    assert class_weight(10_000, 100) == CLASS_WEIGHT_MIN
    assert rule_adjustment(0) == 0.0
    for count in range(RULE_VIOLATION_CAP):
        assert rule_adjustment(count) > rule_adjustment(count + 1)
    assert rule_adjustment(RULE_VIOLATION_CAP) == -RULE_MAXIMUM_COST
    assert rule_adjustment(100) == -RULE_MAXIMUM_COST
    assert valid_total(WRONG, 100) > MALFORMED_FLOOR


def test_scale_helpers_fail_closed_on_out_of_contract_values():
    with pytest.raises(ValueError, match="between 0 and 1"):
        details_payoff(1.01)
    with pytest.raises(ValueError, match="positive"):
        class_weight(0, 1)
    with pytest.raises(ValueError, match="same length"):
        normalized_mean([0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="cannot be negative"):
        rule_adjustment(-1)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        valid_total(1.1, 0)


def test_real_training_structure_and_proofs_are_locked():
    contract = build_contract(repo_root=".")

    assert contract["status"] == "passed"
    assert contract["selection_boundary"]["candidate_completion_rewards_calculated"] is False
    assert all(contract["proofs"].values())
    structure = contract["training_structure"]
    assert structure["active_products"] == 1_438
    assert structure["fields_per_product"] == 15
    assert structure["known_cells"] == KNOWN_CELLS == 12_533
    assert structure["unknown_cells"] == UNKNOWN_CELLS == 9_037
    assert KNOWN_CELLS + UNKNOWN_CELLS == TOTAL_CELLS == 21_570
    assert structure["known_fields_per_product"]["minimum"] == 2
    assert structure["known_fields_per_product"]["median"] == 9
    assert structure["known_fields_per_product"]["maximum"] == 15
    assert structure["details_labeled_set_sizes"]["histogram"] == {1: 917, 2: 38}
    assert structure["class_support"]["observed_attribute_class_pairs"] == 116
    assert structure["class_support"]["classes_below_five"] == 17
    assert structure["starting_policy_rule_violations"]["histogram"] == {
        "0": 11_418,
        "1": 81,
        "2": 1,
        "3": 4,
    }
    assert math.isclose(
        contract["numeric_contract"]["known_unknown_combination"]["known_weight"]
        + contract["numeric_contract"]["known_unknown_combination"]["unknown_weight"],
        1.0,
    )
