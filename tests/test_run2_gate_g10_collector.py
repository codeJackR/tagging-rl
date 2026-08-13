from __future__ import annotations

from copy import deepcopy

import pytest

import training.run2_gate_g10_collector as collector
from training.run2_gate_g10 import calculate_gate_g10
from training.run2_gate_g10_collector import (
    VERSION,
    collect_and_calculate_gate_g10,
)


CANDIDATES = ("U", "UA", "CB")


def _rewards(*, zero_variance: bool, offset: float) -> list[float]:
    if zero_variance:
        return [offset] * 8
    return [offset] * 7 + [offset + 0.25]


def _group(
    position: int,
    *,
    u_zero: bool,
    ua_zero: bool,
    cb_zero: bool,
) -> dict:
    sku_id = f"synthetic-collector-{position}"
    candidate_rewards = {
        "U": _rewards(zero_variance=u_zero, offset=0.1),
        "UA": _rewards(zero_variance=ua_zero, offset=0.2),
        "CB": _rewards(zero_variance=cb_zero, offset=0.3),
    }
    return {
        "group_position": position,
        "sku_id": sku_id,
        "completions": [
            {
                "rollout_index": rollout_index,
                "source_rollout": {
                    "sku_id": sku_id,
                    "rollout_index": rollout_index,
                },
                "candidates": {
                    candidate: {
                        "candidate": candidate,
                        "eligible": True,
                        "reward": candidate_rewards[candidate][rollout_index],
                    }
                    for candidate in CANDIDATES
                },
            }
            for rollout_index in range(8)
        ],
    }


def _five_groups() -> list[dict]:
    return [
        _group(
            position,
            u_zero=position < 2,
            ua_zero=position < 3,
            cb_zero=False,
        )
        for position in range(5)
    ]


def test_collector_uses_one_shared_ordered_denominator_for_all_candidates():
    result = collect_and_calculate_gate_g10(_five_groups(), expected_groups=5)

    assert result["version"] == VERSION
    assert result["status"] == (
        "in_memory_gate_g10_completed_unverified_source_scope"
    )
    assert result["candidate_order"] == ["U", "UA", "CB"]
    assert result["lineage"]["groups"] == 5
    assert result["lineage"]["completions"] == 40
    assert result["lineage"]["unique_skus"] == 5
    assert result["lineage"]["contiguous_group_positions"] is True
    assert result["lineage"]["all_candidates_share_ordered_denominator"] is True
    assert result["lineage"]["groups_adapted_once"] is True
    assert len(set(result["lineage"]["candidate_ordered_group_sha256"].values())) == 1

    u = result["candidate_results"]["U"]
    ua = result["candidate_results"]["UA"]
    cb = result["candidate_results"]["CB"]
    assert (u["zero_variance_groups"], u["comparison"]["passes"]) == (2, True)
    assert (ua["zero_variance_groups"], ua["comparison"]["passes"]) == (3, False)
    assert (cb["zero_variance_groups"], cb["comparison"]["passes"]) == (0, True)
    assert result["boundary"] == {
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
    }


def test_collector_calls_calculator_once_per_candidate(monkeypatch):
    calls: list[str] = []

    def counting_calculator(groups, *, candidate, expected_groups):
        calls.append(candidate)
        return calculate_gate_g10(
            groups,
            candidate=candidate,
            expected_groups=expected_groups,
        )

    monkeypatch.setattr(collector, "calculate_gate_g10", counting_calculator)
    result = collect_and_calculate_gate_g10(_five_groups(), expected_groups=5)

    assert calls == ["U", "UA", "CB"]
    assert result["lineage"]["calculator_calls"] == 3


def test_collector_is_deterministic_and_performs_no_file_io(monkeypatch):
    groups = _five_groups()
    first = collect_and_calculate_gate_g10(groups, expected_groups=5)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("in-memory collector must not open files")

    monkeypatch.setattr("builtins.open", forbidden_open)
    second = collect_and_calculate_gate_g10(groups, expected_groups=5)

    assert second == first


def test_denominator_positions_and_duplicate_skus_fail_closed():
    with pytest.raises(ValueError, match="denominator mismatch"):
        collect_and_calculate_gate_g10(_five_groups(), expected_groups=4)
    with pytest.raises(ValueError, match="positive integer"):
        collect_and_calculate_gate_g10(_five_groups(), expected_groups=True)
    with pytest.raises(TypeError, match="ordered sequence"):
        collect_and_calculate_gate_g10("not-groups", expected_groups=1)

    position_gap = _five_groups()
    position_gap[2]["group_position"] = 3
    with pytest.raises(ValueError, match="contiguous from zero"):
        collect_and_calculate_gate_g10(position_gap, expected_groups=5)

    duplicate = _five_groups()
    duplicate[1]["sku_id"] = duplicate[0]["sku_id"]
    for completion in duplicate[1]["completions"]:
        completion["source_rollout"]["sku_id"] = duplicate[0]["sku_id"]
    with pytest.raises(ValueError, match="duplicate Gate G10 SKU"):
        collect_and_calculate_gate_g10(duplicate, expected_groups=5)


def test_nested_adapter_corruption_still_fails_before_calculation(monkeypatch):
    groups = deepcopy(_five_groups())
    groups[3]["completions"][4]["source_rollout"]["rollout_index"] = 5

    def forbidden_calculator(*args, **kwargs):
        raise AssertionError("calculator must not run after adapter failure")

    monkeypatch.setattr(collector, "calculate_gate_g10", forbidden_calculator)
    with pytest.raises(ValueError, match="source rollout key differs"):
        collect_and_calculate_gate_g10(groups, expected_groups=5)
