from __future__ import annotations

import pytest

from labeling.records import read_jsonl
from training.audit_grpo_pool import build_audit


def test_real_retained_pool_audit_is_complete_and_deterministic():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    kwargs = {
        "scored_path": "data/train_weak_sft_scored.jsonl",
        "difficulty_manifest_path": "runs/sft-difficulty-k8/manifest.json",
        "created_at_utc": "2026-08-02T00:00:00+00:00",
    }
    first = build_audit(rows, **kwargs)
    second = build_audit(rows, **kwargs)

    assert first == second
    assert first["selection"]["full_rows"] == 3600
    assert first["selection"]["retained_rows"] == 1702
    assert first["composition"]["stores"]["full_group_count"] == 14
    assert first["composition"]["stores"]["retained_group_count"] == 14
    assert first["families"]["full_families"] == 2255
    assert first["families"]["retained_families"] == 1150
    assert first["families"]["maximum_retained_rows_one_family"] == 40
    assert first["gold_density"]["retained"]["rows_with_at_most_2_scorable"] == 6
    scenarios = {
        item["policy"]: item
        for item in first["sampling_policy_comparison"]["scenarios"]
    }
    assert scenarios["row_uniform"]["active_rows"] == 1702
    assert scenarios["deterministic_family_cap_2"]["active_rows"] == 1390
    assert scenarios["deterministic_family_cap_4"]["active_rows"] == 1565
    assert scenarios["deterministic_family_cap_8"]["active_rows"] == 1657
    assert scenarios["family_uniform"]["active_rows"] == 1702
    assert scenarios["family_uniform"]["effective_family_count_inverse_hhi"] == pytest.approx(1150)


def test_retained_pool_audit_records_known_composition_shifts():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    audit = build_audit(
        rows,
        scored_path="data/train_weak_sft_scored.jsonl",
        difficulty_manifest_path="runs/sft-difficulty-k8/manifest.json",
        created_at_utc="2026-08-02T00:00:00+00:00",
    )
    findings = audit["headline_findings"]

    assert findings["all_stores_represented"] is True
    assert findings["missing_categories"] == ["jumpsuit"]
    assert findings["largest_supported_category_increase"]["key"] == "shoe"
    assert findings["largest_supported_category_decrease"]["key"] == "dress"
    assert findings["largest_supported_store_increase"]["key"] == "www.thursdayboots.com"
    assert findings["largest_supported_store_decrease"]["key"] == "www.marinelayer.com"
    assert findings["largest_family"]["family_title"] == "everyday tote"


def test_family_cap_four_reduces_concentration_without_large_distribution_shift():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    audit = build_audit(
        rows,
        scored_path="data/train_weak_sft_scored.jsonl",
        difficulty_manifest_path="runs/sft-difficulty-k8/manifest.json",
        created_at_utc="2026-08-02T00:00:00+00:00",
    )
    scenarios = {
        item["policy"]: item
        for item in audit["sampling_policy_comparison"]["scenarios"]
    }
    baseline = scenarios["row_uniform"]
    capped = scenarios["deterministic_family_cap_4"]

    assert capped["active_row_share"] == pytest.approx(1565 / 1702)
    assert capped["maximum_family_probability"] < 0.003
    assert capped["top_10_family_probability"] < 0.026
    assert capped["effective_family_count_inverse_hhi"] > 850
    assert capped["category_tvd_vs_full"] < baseline["category_tvd_vs_full"]
    assert capped["store_tvd_vs_full"] < baseline["store_tvd_vs_full"]
    assert capped["difficulty_tvd_vs_eligible"] < 0.015
