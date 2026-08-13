from __future__ import annotations

from dataclasses import replace

import pytest

from labeling.records import read_jsonl
from training.replay_original_reward import (
    EXPECTED_ROLLOUT_RECORDS,
    EXPECTED_ROLLOUT_SHA256,
    _group_rollouts,
    load_locked_inputs,
    replay_reward_channels,
    summarize_scope,
)
from training.score_difficulty import RolloutRecord
from verifier import load_pack


def _record(sku: str, index: int, *, passed: bool = False) -> RolloutRecord:
    return RolloutRecord(
        sku_id=sku,
        rollout_index=index,
        raw_output="{}",
        passed=passed,
        schema_valid=True,
        vocab_valid=True,
        rule_violations=[],
        errors=[],
        correct_labels=0,
        scorable_labels=1,
        incorrect_labels=["garment_category"],
        excluded_gold_unknown_labels=[],
        generation_seed=1,
        completion_tokens=1,
    )


def test_scope_summary_keeps_group_variance_and_ties_separate():
    skus = ["constant", "mixed"]
    rewards = {}
    for sku in skus:
        for index in range(8):
            golden = 0.0 if sku == "constant" or index < 4 else 1.0
            rewards[(sku, index)] = {
                "format_validity_reward": 1.0,
                "vocab_rule_compliance_reward": 1.0,
                "golden_agreement_reward": golden,
                "weighted_total": 2.0 + 2.0 * golden,
            }

    summary = summarize_scope(sku_ids=skus, rewards_by_key=rewards)

    assert summary["groups"] == 2
    assert summary["difficulty_band_group_counts"] == {
        "always_failed": 1,
        "mixed": 1,
        "always_passed": 0,
    }
    total = summary["channels"]["weighted_total"]
    assert total["zero_variance_groups"] == 1
    assert total["groups_with_variation"] == 1
    assert total["completion_distribution"]["mean"] == 2.5
    assert total["group_mean_distribution"]["median"] == 2.5
    assert total["unique_reward_values_per_group_histogram"] == {"1": 1, "2": 1}
    assert total["largest_tie_size_per_group_histogram"] == {"4": 1, "8": 1}


def test_group_loader_rejects_incomplete_or_noncanonical_groups():
    complete = [_record("a", index) for index in range(8)]
    assert len(_group_rollouts(complete, expected_skus={"a"})["a"]) == 8

    with pytest.raises(ValueError, match="indices 0 through 7"):
        _group_rollouts(complete[:-1], expected_skus={"a"})
    with pytest.raises(ValueError, match="canonically ordered"):
        _group_rollouts(list(reversed(complete)), expected_skus={"a"})


def test_exact_reward_replay_rejects_durable_grade_drift():
    pack = load_pack("packs/vastraa_taste_v1")
    row = read_jsonl("data/train_weak.jsonl")[0]
    gold = row.to_verifier_record(pack)
    import json

    clean = _record(row.sku_id, 0)
    clean = replace(
        clean,
        raw_output=json.dumps(gold),
        passed=False,
        correct_labels=15,
        incorrect_labels=[],
    )
    records = [replace(clean, rollout_index=index) for index in range(8)]

    with pytest.raises(RuntimeError, match="durable rollout grades"):
        replay_reward_channels(
            sku_ids=[row.sku_id],
            records_by_sku={row.sku_id: records},
            rows_by_sku={row.sku_id: row},
            pack=pack,
        )


def test_real_d1_inputs_are_hash_locked_and_validation_excluded():
    inputs = load_locked_inputs(repo_root=".")

    assert inputs.metadata["rollouts"]["sha256"] == EXPECTED_ROLLOUT_SHA256
    assert inputs.metadata["physical_rollout_records"] == EXPECTED_ROLLOUT_RECORDS
    assert len(inputs.records_by_sku) == 3_600
    assert len(inputs.authoritative_train_skus) == 3_240
    assert len(inputs.active_pool_skus) == 1_438
    assert len(inputs.validation_skus) == 360
    assert set(inputs.active_pool_skus) <= set(inputs.authoritative_train_skus)
    assert not set(inputs.active_pool_skus) & set(inputs.validation_skus)
