from pathlib import Path

import pytest

from labeling.records import read_jsonl
from training.build_run2_causal_schedule import (
    DEFAULT_OUTPUT,
    SCHEDULE_ROWS,
    build_manifest,
    select_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_schedule_is_deterministic_unique_and_training_only():
    selected, manifest = build_manifest(
        root=ROOT,
        source="data/train_weak_grpo_cap4_sft_train_v1.jsonl",
        pool_manifest="runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json",
        output=DEFAULT_OUTPUT,
    )
    assert len(selected) == SCHEDULE_ROWS == 300
    assert len({row.sku_id for row in selected}) == 300
    assert all(row.split == "train" for row in selected)
    assert all(0 < row.difficulty.sft_pass_rate < 1 for row in selected)
    assert manifest["selection"]["shuffle_in_trainer"] is False
    assert manifest["invariants"]["validation_or_confirmation_rows_used"] is False
    rebuilt, rebuilt_manifest = build_manifest(
        root=ROOT,
        source="data/train_weak_grpo_cap4_sft_train_v1.jsonl",
        pool_manifest="runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json",
        output=DEFAULT_OUTPUT,
    )
    assert [row.sku_id for row in selected] == [row.sku_id for row in rebuilt]
    assert manifest == rebuilt_manifest


def test_published_schedule_matches_rebuild_when_present():
    output = ROOT / DEFAULT_OUTPUT
    if not output.exists():
        pytest.skip("causal schedule has not been published")
    published = read_jsonl(output)
    rebuilt, _ = build_manifest(
        root=ROOT,
        source="data/train_weak_grpo_cap4_sft_train_v1.jsonl",
        pool_manifest="runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json",
        output=DEFAULT_OUTPUT,
    )
    assert [row.model_dump(mode="json") for row in published] == [
        row.model_dump(mode="json") for row in rebuilt
    ]


def test_schedule_rejects_wrong_source_size():
    rows = read_jsonl(ROOT / "data/train_weak_grpo_cap4_sft_train_v1.jsonl")
    with pytest.raises(ValueError, match="1,438"):
        select_schedule(rows[:-1])
