from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_gate_g10 import (
    GATE_ID,
    ORIGINAL_ZERO_VARIANCE_GROUPS,
    ORIGINAL_ZERO_VARIANCE_SHARE,
    THRESHOLD,
    GateG10Group,
    calculate_gate_g10,
)


def _group(group_id: str, *, zero_variance: bool) -> GateG10Group:
    rewards = [0.25] * 8 if zero_variance else [0.25] * 7 + [0.5]
    return GateG10Group(group_id=group_id, rewards=rewards)


def _groups(total: int, zero_variance: int) -> list[GateG10Group]:
    return [
        _group(f"group-{index}", zero_variance=index < zero_variance)
        for index in range(total)
    ]


def test_g10_counts_each_product_once_and_reports_locked_boundaries():
    result = calculate_gate_g10(_groups(5, 2), candidate="U", expected_groups=5)

    assert result["groups"] == 5
    assert result["completions"] == 40
    assert result["zero_variance_groups"] == 2
    assert result["varying_groups"] == 3
    assert result["zero_variance_share"] == 0.4
    assert result["unique_reward_levels_per_group_histogram"] == {"1": 2, "2": 3}
    assert len(result["ordered_group_id_sha256"]) == 64
    assert result["boundary"] == {
        "input_kind": "already-materialized in-memory candidate reward groups",
        "file_io_performed": False,
        "real_full_training_replay_opened_by_calculator": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }


def test_exactly_forty_percent_passes_and_one_more_group_fails():
    passing = calculate_gate_g10(_groups(5, 2), candidate="UA", expected_groups=5)
    failing = calculate_gate_g10(_groups(5, 3), candidate="UA", expected_groups=5)

    assert passing["comparison"] == {
        "canonical_decimal_places": 12,
        "operator": "less_than_or_equal",
        "threshold": 0.4,
        "threshold_exact_fraction": "2/5",
        "maximum_allowed_zero_variance_groups": 2,
        "passes": True,
        "margin_groups": 0,
        "margin_share": 0.0,
    }
    assert failing["comparison"]["passes"] is False
    assert failing["comparison"]["margin_groups"] == -1
    assert failing["comparison"]["margin_share"] == pytest.approx(-0.2)


def test_twelve_decimal_canonicalization_controls_zero_variance():
    rounded_tie = GateG10Group(
        group_id="rounded-tie",
        rewards=[1.0] * 7 + [1.0000000000004],
    )
    meaningful_difference = GateG10Group(
        group_id="meaningful-difference",
        rewards=[1.0] * 7 + [1.0000000000006],
    )
    result = calculate_gate_g10(
        [rounded_tie, meaningful_difference],
        candidate="CB",
        expected_groups=2,
    )

    assert result["zero_variance_groups"] == 1
    assert result["unique_reward_levels_per_group_histogram"] == {"1": 1, "2": 1}


def test_denominator_identity_shape_and_numbers_fail_closed():
    with pytest.raises(ValueError, match="denominator mismatch"):
        calculate_gate_g10(_groups(4, 1), candidate="U", expected_groups=5)
    with pytest.raises(ValueError, match="positive integer"):
        calculate_gate_g10(_groups(1, 0), candidate="U", expected_groups=True)
    with pytest.raises(ValueError, match="candidate must be"):
        calculate_gate_g10(_groups(1, 0), candidate="original", expected_groups=1)

    duplicate = [_group("same", zero_variance=True)] * 2
    with pytest.raises(ValueError, match="duplicate"):
        calculate_gate_g10(duplicate, candidate="U", expected_groups=2)
    with pytest.raises(ValueError, match="exactly 8"):
        calculate_gate_g10(
            [GateG10Group("short", [0.0] * 7)],
            candidate="U",
            expected_groups=1,
        )
    with pytest.raises(ValueError, match="finite"):
        calculate_gate_g10(
            [GateG10Group("nan", [0.0] * 7 + [float("nan")])],
            candidate="U",
            expected_groups=1,
        )


def test_constants_match_locked_contract_and_original_baseline():
    repo_root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (repo_root / "runs/grpo-run2-comparison-contract.json").read_text(
            encoding="utf-8"
        )
    )
    gate = next(
        item
        for item in contract["universal_acceptance_gates"]
        if item["id"] == GATE_ID
    )
    original = json.loads(
        (repo_root / "runs/grpo-run2-original-reward-training-replay.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = original["scopes"]["authoritative_sft_train"]["channels"][
        "weighted_total"
    ]

    assert gate["threshold"] == THRESHOLD == 0.4
    assert gate["baseline"] == ORIGINAL_ZERO_VARIANCE_SHARE
    assert baseline["zero_variance_groups"] == ORIGINAL_ZERO_VARIANCE_GROUPS == 1571
    assert baseline["zero_variance_share"] == ORIGINAL_ZERO_VARIANCE_SHARE
