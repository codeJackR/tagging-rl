from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from training.replay_original_reward import load_locked_inputs, replay_reward_channels
from training.replay_run2_candidates import (
    CANDIDATES,
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_RECORDS_OUTPUT,
    EXPECTED_ACTIVE_COMPLETIONS,
    EXPECTED_ACTIVE_GROUPS,
    VERSION,
    build_manifest,
    build_replay_group,
    write_replay_groups,
)
from training.run2_rewards import load_cb_class_weight_lookup
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def real_replay_inputs():
    return load_locked_inputs(repo_root=ROOT)


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def cb_lookup(pack):
    return load_cb_class_weight_lookup(pack)


def _one_real_group(real_replay_inputs, pack, cb_lookup):
    sku_id = real_replay_inputs.active_pool_skus[0]
    records = real_replay_inputs.records_by_sku[sku_id]
    original = replay_reward_channels(
        sku_ids=[sku_id],
        records_by_sku=real_replay_inputs.records_by_sku,
        rows_by_sku=real_replay_inputs.rows_by_sku,
        pack=pack,
    )
    return build_replay_group(
        group_position=0,
        row=real_replay_inputs.rows_by_sku[sku_id],
        records=records,
        original_rewards=original,
        pack=pack,
        class_weights=cb_lookup,
    )


def test_one_real_group_preserves_same_eight_keys_and_all_candidate_ledgers(
    real_replay_inputs, pack, cb_lookup
):
    group = _one_real_group(real_replay_inputs, pack, cb_lookup)
    assert group["group_position"] == 0
    assert group["sku_id"] == real_replay_inputs.active_pool_skus[0]
    assert group["gold_known_fields"] + group["gold_unknown_fields"] == 15
    assert [item["rollout_index"] for item in group["completions"]] == list(range(8))

    for completion in group["completions"]:
        assert set(completion["candidates"]) == set(CANDIDATES)
        assert completion["source_rollout"]["sku_id"] == group["sku_id"]
        assert completion["source_rollout"]["rollout_index"] == completion[
            "rollout_index"
        ]
        assert len(completion["raw_output_sha256"]) == 64
        assert completion["raw_output_sha256"] == hashlib.sha256(
            completion["source_rollout"]["raw_output"].encode("utf-8")
        ).hexdigest()
        assert set(completion["original_reward"]) == {
            "format_validity_reward",
            "vocab_rule_compliance_reward",
            "golden_agreement_reward",
            "weighted_total",
        }
        assert all(
            isinstance(completion["candidates"][candidate]["reward"], float)
            for candidate in CANDIDATES
        )


def test_group_builder_rejects_wrong_sku_incomplete_or_reordered_source(
    real_replay_inputs, pack, cb_lookup
):
    sku_id = real_replay_inputs.active_pool_skus[0]
    row = real_replay_inputs.rows_by_sku[sku_id]
    records = real_replay_inputs.records_by_sku[sku_id]
    original = replay_reward_channels(
        sku_ids=[sku_id],
        records_by_sku=real_replay_inputs.records_by_sku,
        rows_by_sku=real_replay_inputs.rows_by_sku,
        pack=pack,
    )
    common = {
        "group_position": 0,
        "row": row,
        "original_rewards": original,
        "pack": pack,
        "class_weights": cb_lookup,
    }
    with pytest.raises(ValueError, match="exactly 8"):
        build_replay_group(records=records[:-1], **common)
    with pytest.raises(ValueError, match="ordered 0 through 7"):
        build_replay_group(records=list(reversed(records)), **common)
    wrong_sku = replace(records[0], sku_id="different")
    with pytest.raises(ValueError, match="different SKU"):
        build_replay_group(records=[wrong_sku, *records[1:]], **common)


def test_group_writer_is_deterministic_and_rejects_duplicate_groups(
    real_replay_inputs, pack, cb_lookup, tmp_path
):
    group = _one_real_group(real_replay_inputs, pack, cb_lookup)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_meta = write_replay_groups(first, [group], repo_root=tmp_path)
    second_meta = write_replay_groups(second, [group], repo_root=tmp_path)

    assert first.read_bytes() == second.read_bytes()
    assert first_meta["sha256"] == second_meta["sha256"]
    assert first_meta["jsonl_group_records"] == 1
    assert first_meta["completion_records"] == 8
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == json.loads(json.dumps(group))
        assert handle.readline() == ""

    duplicate = dict(group)
    duplicate["group_position"] = 1
    with pytest.raises(ValueError, match="duplicate candidate replay group"):
        write_replay_groups(
            tmp_path / "duplicate.jsonl.gz",
            [group, duplicate],
            repo_root=tmp_path,
        )


def test_manifest_has_raw_lineage_counts_and_explicit_no_selection_boundary(
    real_replay_inputs
):
    published = json.loads(
        (ROOT / DEFAULT_MANIFEST_OUTPUT).read_text(encoding="utf-8")
    )
    assert published["version"] == VERSION
    assert published["status"] == "raw_evidence_published"
    assert published["output"]["path"] == DEFAULT_RECORDS_OUTPUT
    assert published["integrity"]["active_groups"] == EXPECTED_ACTIVE_GROUPS
    assert published["integrity"]["completion_records"] == (
        EXPECTED_ACTIVE_COMPLETIONS
    )
    assert published["integrity"]["active_pool_validation_sku_overlap"] == 0
    boundary = published["selection_boundary"]
    assert boundary["candidate_rewards_calculated"] is True
    assert boundary["aggregate_candidate_comparison_calculated"] is False
    assert boundary["candidate_rankings_calculated"] is False
    assert boundary["acceptance_thresholds_applied"] is False
    assert boundary["winner_selected"] is False
    assert published["interpretation_guardrails"][
        "candidate_superiority_claim_allowed"
    ] is False

    rebuilt = build_manifest(
        repo_root=ROOT,
        inputs=real_replay_inputs,
        records_metadata=published["output"],
        implementation_path=ROOT / "training" / "replay_run2_candidates.py",
    )
    assert rebuilt == published
