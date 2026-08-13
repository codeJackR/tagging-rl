from __future__ import annotations

from copy import deepcopy

import pytest

from training.run2_gate_g10 import calculate_gate_g10
from training.run2_gate_g10_adapter import (
    VERSION,
    adapt_gate_g10_group,
)


CANDIDATES = ("U", "UA", "CB")


def _completion(index: int) -> dict:
    return {
        "rollout_index": index,
        "source_rollout": {
            "sku_id": "synthetic-g10-sku",
            "rollout_index": index,
        },
        "candidates": {
            "U": {
                "candidate": "U",
                "eligible": True,
                "reward": index / 10,
            },
            "UA": {
                "candidate": "UA",
                "eligible": True,
                "reward": index / 20,
            },
            "CB": {
                "candidate": "CB",
                "eligible": True,
                "reward": 0.25,
            },
        },
    }


def _group() -> dict:
    return {
        "group_position": 7,
        "sku_id": "synthetic-g10-sku",
        "completions": [_completion(index) for index in range(8)],
    }


def test_adapter_preserves_product_candidate_and_rollout_alignment():
    adapted = adapt_gate_g10_group(_group())

    assert adapted.version == VERSION
    assert adapted.group_position == 7
    assert adapted.sku_id == "synthetic-g10-sku"
    assert tuple(group.group_id for group in adapted.candidate_groups) == (
        "synthetic-g10-sku",
    ) * 3
    assert tuple(group.rewards for group in adapted.candidate_groups) == (
        tuple(index / 10 for index in range(8)),
        tuple(index / 20 for index in range(8)),
        (0.25,) * 8,
    )
    assert len(adapted.ordered_rollout_key_sha256) == 64
    assert adapted.boundary == {
        "input_was_one_in_memory_replay_group": True,
        "file_io_performed": False,
        "real_full_training_replay_opened": False,
        "gate_g10_calculated": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }


def test_adapter_outputs_feed_the_pure_calculator_without_reordering():
    adapted = adapt_gate_g10_group(_group())
    results = {
        candidate: calculate_gate_g10(
            [candidate_group], candidate=candidate, expected_groups=1
        )
        for candidate, candidate_group in zip(
            CANDIDATES, adapted.candidate_groups, strict=True
        )
    }

    assert results["U"]["zero_variance_groups"] == 0
    assert results["UA"]["zero_variance_groups"] == 0
    assert results["CB"]["zero_variance_groups"] == 1
    assert all(
        result["ordered_group_id_sha256"]
        == results["U"]["ordered_group_id_sha256"]
        for result in results.values()
    )


def test_candidate_set_identity_and_eligibility_fail_closed():
    wrong_set = _group()
    del wrong_set["completions"][0]["candidates"]["CB"]
    wrong_set["completions"][0]["candidates"]["other"] = {
        "candidate": "other",
        "eligible": True,
        "reward": 0.0,
    }
    with pytest.raises(ValueError, match="candidate ledger set"):
        adapt_gate_g10_group(wrong_set)

    physically_sorted = _group()
    candidates = physically_sorted["completions"][0]["candidates"]
    physically_sorted["completions"][0]["candidates"] = {
        "CB": candidates["CB"],
        "U": candidates["U"],
        "UA": candidates["UA"],
    }
    adapted = adapt_gate_g10_group(physically_sorted)
    assert tuple(group.rewards for group in adapted.candidate_groups)[0] == tuple(
        index / 10 for index in range(8)
    )

    wrong_identity = _group()
    wrong_identity["completions"][0]["candidates"]["U"]["candidate"] = "UA"
    with pytest.raises(ValueError, match="candidate identity drifted"):
        adapt_gate_g10_group(wrong_identity)

    eligibility_drift = _group()
    eligibility_drift["completions"][0]["candidates"]["CB"]["eligible"] = False
    with pytest.raises(ValueError, match="eligibility alignment drifted"):
        adapt_gate_g10_group(eligibility_drift)


def test_completion_and_source_rollout_lineage_fail_closed():
    short = _group()
    short["completions"].pop()
    with pytest.raises(ValueError, match="exactly 8 completions"):
        adapt_gate_g10_group(short)

    reordered = _group()
    reordered["completions"][0], reordered["completions"][1] = (
        reordered["completions"][1],
        reordered["completions"][0],
    )
    with pytest.raises(ValueError, match="ordered 0 through 7"):
        adapt_gate_g10_group(reordered)

    wrong_source_sku = _group()
    wrong_source_sku["completions"][3]["source_rollout"]["sku_id"] = "other"
    with pytest.raises(ValueError, match="source rollout key differs"):
        adapt_gate_g10_group(wrong_source_sku)

    wrong_source_index = _group()
    wrong_source_index["completions"][3]["source_rollout"]["rollout_index"] = 4
    with pytest.raises(ValueError, match="source rollout key differs"):
        adapt_gate_g10_group(wrong_source_index)


def test_invalid_group_candidate_ledgers_and_rewards_fail_closed():
    with pytest.raises(TypeError, match="must be an object"):
        adapt_gate_g10_group([])

    missing_candidates = _group()
    del missing_candidates["completions"][0]["candidates"]["UA"]
    with pytest.raises(ValueError, match="candidate ledger set"):
        adapt_gate_g10_group(missing_candidates)

    non_boolean = _group()
    non_boolean["completions"][0]["candidates"]["U"]["eligible"] = 1
    with pytest.raises(TypeError, match="eligible must be boolean"):
        adapt_gate_g10_group(non_boolean)

    for invalid in (None, True, "not-a-number", float("nan"), float("inf")):
        bad_reward = deepcopy(_group())
        bad_reward["completions"][0]["candidates"]["U"]["reward"] = invalid
        with pytest.raises((TypeError, ValueError), match="numeric|finite"):
            adapt_gate_g10_group(bad_reward)
