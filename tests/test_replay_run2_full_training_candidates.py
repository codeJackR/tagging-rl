from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import training.replay_run2_full_training_candidates as full_replay
from training.replay_original_reward import ReplayInputs
from training.replay_run2_full_training_candidates import (
    CB_ZERO_ACTIVE_SUPPORT_WEIGHT,
    CB_DIAGNOSTIC_EXTENSION_VERSION,
    FULL_TRAINING_ROLE,
    PRODUCTION_CB_EXTENSION_CONTRACT,
    FullTrainingCBExtension,
    FullTrainingScopeContract,
    _publish_pair_exclusively,
    build_full_training_manifest,
    build_full_training_cb_extension,
    iter_full_training_replay_groups,
    ordered_sha256,
    publish_full_training_replay,
    validate_cb_extension_for_publication,
    validate_full_training_scope,
)
from training.run2_rewards import (
    CBClassWeightLookup,
    load_cb_class_weight_lookup,
    score_class_balanced_known_fields,
)
from training.build_cb_class_weights import label_class_keys
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


def _row(sku_id: str, *, brand: str, title: str):
    return SimpleNamespace(
        sku_id=sku_id,
        input=SimpleNamespace(brand=brand, title=title),
    )


def _records(sku_id: str):
    return [
        SimpleNamespace(sku_id=sku_id, rollout_index=index)
        for index in range(8)
    ]


def _contract(train: list[str], *, validation=1, active=1):
    keys = [f"{sku_id}\t{index}" for sku_id in train for index in range(8)]
    return FullTrainingScopeContract(
        training_groups=len(train),
        validation_groups=validation,
        active_groups=active,
        ordered_sku_sha256=ordered_sha256(train),
        ordered_rollout_key_sha256=ordered_sha256(keys),
    )


def _inputs():
    rows = {
        "train-b": _row("train-b", brand="Beta", title="Jacket"),
        "train-a": _row("train-a", brand="Alpha", title="Dress"),
        "val-c": _row("val-c", brand="Gamma", title="Coat"),
    }
    return ReplayInputs(
        rows_by_sku=rows,
        records_by_sku={sku: _records(sku) for sku in rows},
        authoritative_train_skus=["train-b", "train-a"],
        active_pool_skus=["train-a"],
        validation_skus=["val-c"],
        metadata={},
    )


def _synthetic_cb_extension():
    base = CBClassWeightLookup(
        artifact_version="synthetic-base-v1",
        field_names=("field",),
        unknown_token="unknown",
        weights=MappingProxyType(
            {"field": MappingProxyType({"base_class": 1.0})}
        ),
    )
    derived = CBClassWeightLookup(
        artifact_version=(
            f"synthetic-base-v1+{CB_DIAGNOSTIC_EXTENSION_VERSION}"
        ),
        field_names=("field",),
        unknown_token="unknown",
        weights=MappingProxyType(
            {"field": MappingProxyType({"base_class": 1.0, "new_class": 2.0})}
        ),
    )
    entries = [
        {
            "field_name": "field",
            "class_name": "new_class",
            "active_pool_support": 0,
            "full_training_observations": 1,
            "affected_products": 1,
            "weight": 2.0,
        }
    ]
    entry_hash = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "version": CB_DIAGNOSTIC_EXTENSION_VERSION,
        "role": "gold-only full-training diagnostic extension to active CB lookup",
        "base_artifact_sha256": full_replay.CB_CLASS_WEIGHT_ARTIFACT_SHA256,
        "policy": {
            "trigger": "valid gold class has zero support in the active-pool map",
            "weight": 2.0,
            "rationale": "limit of the existing rare-class formula after clipping",
            "active_weights_changed": False,
            "full_training_support_used_to_retune_weights": False,
        },
        "missing_attribute_class_pairs": 1,
        "missing_class_observations": 1,
        "affected_training_products": 1,
        "affected_active_products": 0,
        "ordered_entry_ledger_sha256": entry_hash,
        "entries": entries,
        "candidate_completion_rewards_calculated": False,
        "candidate_aggregates_calculated": False,
    }
    return FullTrainingCBExtension(
        base_lookup=base,
        lookup=derived,
        audit=MappingProxyType(audit),
    )


@pytest.fixture(scope="module")
def production_inputs():
    from training.replay_original_reward import load_locked_inputs

    return load_locked_inputs(repo_root=ROOT)


@pytest.fixture(scope="module")
def production_pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def base_cb_lookup(production_pack):
    return load_cb_class_weight_lookup(production_pack)


@pytest.fixture(scope="module")
def diagnostic_cb_extension(production_inputs, production_pack, base_cb_lookup):
    return build_full_training_cb_extension(
        inputs=production_inputs,
        pack=production_pack,
        base_lookup=base_cb_lookup,
    )


def test_synthetic_scope_preserves_manifest_order_and_excludes_validation():
    inputs = _inputs()
    contract = _contract(inputs.authoritative_train_skus)

    result = validate_full_training_scope(inputs, contract=contract)

    assert result == {
        "version": full_replay.VERSION,
        "role": FULL_TRAINING_ROLE,
        "training_groups": 2,
        "validation_groups_excluded": 1,
        "active_groups_included": 1,
        "additional_training_groups": 1,
        "completion_records": 16,
        "ordered_sku_sha256": contract.ordered_sku_sha256,
        "ordered_rollout_key_sha256": contract.ordered_rollout_key_sha256,
        "training_validation_sku_overlap": 0,
        "training_validation_family_overlap": 0,
        "model_generation_performed": False,
        "candidate_aggregates_calculated": False,
    }


def test_iterator_delegates_in_manifest_order_to_shared_group_builder(monkeypatch):
    inputs = _inputs()
    contract = _contract(inputs.authoritative_train_skus)
    calls = []

    def fake_group_builder(**kwargs):
        calls.append(kwargs)
        return {
            "group_position": kwargs["group_position"],
            "sku_id": kwargs["row"].sku_id,
        }

    monkeypatch.setattr(full_replay, "build_replay_group", fake_group_builder)
    groups = list(
        iter_full_training_replay_groups(
            inputs=inputs,
            original_rewards={},
            pack=object(),
            class_weights=object(),
            contract=contract,
        )
    )

    assert groups == [
        {"group_position": 0, "sku_id": "train-b"},
        {"group_position": 1, "sku_id": "train-a"},
    ]
    assert [call["row"].sku_id for call in calls] == ["train-b", "train-a"]
    assert all(call["original_rewards"] == {} for call in calls)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_train", "duplicate SKUs"),
        ("sku_overlap", "overlap"),
        ("incomplete_coverage", "exactly cover source rows"),
        ("family_overlap", "families overlap"),
        ("missing_rollout_group", "exactly cover source rows"),
        ("reordered_rollouts", "ordered 0 through 7"),
    ],
)
def test_synthetic_scope_fails_closed_on_boundary_or_k8_drift(mutation, message):
    inputs = _inputs()
    contract = _contract(inputs.authoritative_train_skus)

    if mutation == "duplicate_train":
        inputs.authoritative_train_skus[1] = "train-b"
    elif mutation == "sku_overlap":
        inputs.validation_skus[0] = "train-b"
    elif mutation == "incomplete_coverage":
        inputs.validation_skus[0] = "missing"
    elif mutation == "family_overlap":
        inputs.rows_by_sku["val-c"].input.brand = "Alpha"
        inputs.rows_by_sku["val-c"].input.title = "Dress - Red"
    elif mutation == "missing_rollout_group":
        inputs.records_by_sku.pop("val-c")
    elif mutation == "reordered_rollouts":
        inputs.records_by_sku["train-a"] = list(
            reversed(inputs.records_by_sku["train-a"])
        )

    with pytest.raises(ValueError, match=message):
        validate_full_training_scope(inputs, contract=contract)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("ordered_sku_sha256", "SKU order hash drifted"),
        ("ordered_rollout_key_sha256", "rollout-key order hash drifted"),
        ("role", "role drifted"),
    ],
)
def test_synthetic_scope_rejects_contract_identity_drift(field, message):
    inputs = _inputs()
    contract = _contract(inputs.authoritative_train_skus)
    drifted_value = "different-role" if field == "role" else "0" * 64
    drifted = replace(contract, **{field: drifted_value})

    with pytest.raises(ValueError, match=message):
        validate_full_training_scope(inputs, contract=drifted)


def _fake_scored_group(**kwargs):
    return {
        "group_position": kwargs["group_position"],
        "sku_id": kwargs["row"].sku_id,
        "completions": [
            {"rollout_index": index, "candidates": {"U": {}, "UA": {}, "CB": {}}}
            for index in range(8)
        ],
    }


def _publish_synthetic(root: Path, monkeypatch):
    root.mkdir(parents=True)
    inputs = _inputs()
    contract = _contract(inputs.authoritative_train_skus)
    monkeypatch.setattr(full_replay, "build_replay_group", _fake_scored_group)
    manifest = publish_full_training_replay(
        repo_root=root,
        inputs=inputs,
        original_rewards={},
        pack=object(),
        cb_extension=_synthetic_cb_extension(),
        contract=contract,
    )
    return manifest


def test_synthetic_publication_is_deterministic_and_boundary_explicit(
    tmp_path, monkeypatch
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _publish_synthetic(first_root, monkeypatch)
    second = _publish_synthetic(second_root, monkeypatch)
    records_relative = Path(full_replay.DEFAULT_RECORDS_OUTPUT)
    manifest_relative = Path(full_replay.DEFAULT_MANIFEST_OUTPUT)

    assert (first_root / records_relative).read_bytes() == (
        second_root / records_relative
    ).read_bytes()
    assert (first_root / manifest_relative).read_bytes() == (
        second_root / manifest_relative
    ).read_bytes()
    assert first == second
    assert first["role"] == FULL_TRAINING_ROLE
    assert first["output"]["jsonl_group_records"] == 2
    assert first["output"]["completion_records"] == 16
    assert first["integrity"]["additional_training_groups"] == 1
    boundary = first["selection_boundary"]
    assert boundary["model_generation_performed"] is False
    assert boundary["validation_data_used"] is False
    assert boundary["candidate_rewards_calculated"] is True
    assert boundary["aggregate_candidate_comparison_calculated"] is False
    assert boundary["acceptance_thresholds_applied"] is False
    assert boundary["winner_selected"] is False
    assert first["interpretation_guardrails"]["gate_g10_result_available"] is False
    extension = first["cb_diagnostic_extension"]
    assert extension["ordered_entry_ledger_sha256"] == (
        first["integrity"]["cb_extension_ledger_sha256"]
    )
    assert extension["entries"][0]["class_name"] == "new_class"
    assert first["integrity"]["cb_active_weights_changed"] is False

    with gzip.open(first_root / records_relative, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert [record["sku_id"] for record in records] == ["train-b", "train-a"]


def test_collision_fails_before_scoring_and_preserves_existing_file(
    tmp_path, monkeypatch
):
    records = tmp_path / full_replay.DEFAULT_RECORDS_OUTPUT
    records.parent.mkdir(parents=True)
    records.write_bytes(b"existing")

    def forbidden_scorer(**kwargs):
        raise AssertionError("collision must fail before scoring")

    monkeypatch.setattr(full_replay, "build_replay_group", forbidden_scorer)
    inputs = _inputs()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_full_training_replay(
            repo_root=tmp_path,
            inputs=inputs,
            original_rewards={},
            pack=object(),
            cb_extension=_synthetic_cb_extension(),
            contract=_contract(inputs.authoritative_train_skus),
        )

    assert records.read_bytes() == b"existing"
    assert not (tmp_path / full_replay.DEFAULT_MANIFEST_OUTPUT).exists()


def test_late_scoring_failure_publishes_neither_file(tmp_path, monkeypatch):
    inputs = _inputs()

    def fail_on_second_group(**kwargs):
        if kwargs["group_position"] == 1:
            raise RuntimeError("synthetic late scoring failure")
        return _fake_scored_group(**kwargs)

    monkeypatch.setattr(full_replay, "build_replay_group", fail_on_second_group)
    with pytest.raises(RuntimeError, match="late scoring failure"):
        publish_full_training_replay(
            repo_root=tmp_path,
            inputs=inputs,
            original_rewards={},
            pack=object(),
            cb_extension=_synthetic_cb_extension(),
            contract=_contract(inputs.authoritative_train_skus),
        )

    assert not (tmp_path / full_replay.DEFAULT_RECORDS_OUTPUT).exists()
    assert not (tmp_path / full_replay.DEFAULT_MANIFEST_OUTPUT).exists()


def test_second_link_failure_rolls_back_first_published_link(tmp_path, monkeypatch):
    staged_records = tmp_path / "staged.jsonl.gz"
    staged_manifest = tmp_path / "staged.json"
    records_output = tmp_path / "final.jsonl.gz"
    manifest_output = tmp_path / "final.json"
    staged_records.write_bytes(b"records")
    staged_manifest.write_bytes(b"manifest")
    real_link = os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic manifest publication failure")
        real_link(source, destination)

    monkeypatch.setattr(full_replay.os, "link", fail_second_link)
    with pytest.raises(OSError, match="manifest publication failure"):
        _publish_pair_exclusively(
            staged_records=staged_records,
            records_output=records_output,
            staged_manifest=staged_manifest,
            manifest_output=manifest_output,
        )

    assert not records_output.exists()
    assert not manifest_output.exists()
    assert staged_records.read_bytes() == b"records"
    assert staged_manifest.read_bytes() == b"manifest"


@pytest.mark.parametrize(
    ("records_output", "manifest_output"),
    [
        (full_replay.ACTIVE_RECORDS_OUTPUT, full_replay.DEFAULT_MANIFEST_OUTPUT),
        (full_replay.DEFAULT_RECORDS_OUTPUT, full_replay.ACTIVE_MANIFEST_OUTPUT),
    ],
)
def test_full_scope_outputs_cannot_alias_active_replay(
    tmp_path, records_output, manifest_output
):
    inputs = _inputs()
    with pytest.raises(ValueError, match="cannot alias active replay"):
        publish_full_training_replay(
            repo_root=tmp_path,
            inputs=inputs,
            original_rewards={},
            pack=object(),
            cb_extension=_synthetic_cb_extension(),
            records_output=records_output,
            manifest_output=manifest_output,
            contract=_contract(inputs.authoritative_train_skus),
        )


def test_production_cb_extension_locks_zero_support_entries_without_retuning(
    production_inputs,
    production_pack,
    base_cb_lookup,
    diagnostic_cb_extension,
):
    audit = diagnostic_cb_extension.audit
    assert audit["missing_attribute_class_pairs"] == 13
    assert audit["missing_class_observations"] == 53
    assert audit["affected_training_products"] == 50
    assert audit["affected_active_products"] == 0
    assert audit["policy"]["weight"] == CB_ZERO_ACTIVE_SUPPORT_WEIGHT == 2.0
    assert audit["policy"]["active_weights_changed"] is False
    assert audit["policy"]["full_training_support_used_to_retune_weights"] is False
    assert len(audit["ordered_entry_ledger_sha256"]) == 64

    for field_name in production_pack.field_names:
        for class_name, weight in base_cb_lookup.weights[field_name].items():
            assert diagnostic_cb_extension.lookup.weights[field_name][class_name] == weight
    for entry in audit["entries"]:
        assert entry["active_pool_support"] == 0
        assert entry["weight"] == 2.0
        assert (
            diagnostic_cb_extension.lookup.weights[entry["field_name"]][
                entry["class_name"]
            ]
            == 2.0
        )

    with pytest.raises(TypeError):
        diagnostic_cb_extension.lookup.weights["details"]["new"] = 1.0


def test_extension_scores_active_unseen_gold_but_base_lookup_still_fails_closed(
    production_inputs,
    production_pack,
    base_cb_lookup,
    diagnostic_cb_extension,
):
    sku_id = next(
        sku
        for sku in production_inputs.authoritative_train_skus
        if "gathered"
        in label_class_keys(
            field_name="details",
            label=production_inputs.rows_by_sku[sku].labels["details"],
            pack=production_pack,
        )
    )
    assert sku_id not in set(production_inputs.active_pool_skus)
    gold = production_inputs.rows_by_sku[sku_id].to_verifier_record(production_pack)

    with pytest.raises(ValueError, match="no weight for details/gathered"):
        score_class_balanced_known_fields(
            gold,
            gold,
            production_pack,
            base_cb_lookup,
        )

    result = score_class_balanced_known_fields(
        gold,
        gold,
        production_pack,
        diagnostic_cb_extension.lookup,
    )
    details = next(
        outcome for outcome in result.field_outcomes if outcome.field_name == "details"
    )
    gathered_position = details.gold_class_keys.index("gathered")
    assert details.gold_class_weights[gathered_position] == 2.0


def test_diagnostic_extension_leaves_controlled_active_score_identical(
    production_inputs,
    production_pack,
    base_cb_lookup,
    diagnostic_cb_extension,
):
    sku_id = production_inputs.active_pool_skus[0]
    gold = production_inputs.rows_by_sku[sku_id].to_verifier_record(production_pack)
    base = score_class_balanced_known_fields(
        gold, gold, production_pack, base_cb_lookup
    )
    extended = score_class_balanced_known_fields(
        gold, gold, production_pack, diagnostic_cb_extension.lookup
    )
    assert extended == base


def test_diagnostic_extension_rejects_entry_contract_drift(
    production_inputs, production_pack, base_cb_lookup
):
    drifted = replace(
        PRODUCTION_CB_EXTENSION_CONTRACT,
        expected_entries=PRODUCTION_CB_EXTENSION_CONTRACT.expected_entries[:-1],
    )
    with pytest.raises(ValueError, match="entry contract drifted"):
        build_full_training_cb_extension(
            inputs=production_inputs,
            pack=production_pack,
            base_lookup=base_cb_lookup,
            contract=drifted,
        )


def test_diagnostic_extension_never_repairs_missing_active_class(
    production_inputs, production_pack, base_cb_lookup
):
    active_sku = production_inputs.active_pool_skus[0]
    active_row = production_inputs.rows_by_sku[active_sku]
    field_name, class_name = next(
        (field_name, class_name)
        for field_name in production_pack.field_names
        for class_name in label_class_keys(
            field_name=field_name,
            label=active_row.labels[field_name],
            pack=production_pack,
        )
    )
    copied = {
        field: dict(base_cb_lookup.weights[field])
        for field in production_pack.field_names
    }
    copied[field_name].pop(class_name)
    broken = CBClassWeightLookup(
        artifact_version=base_cb_lookup.artifact_version,
        field_names=base_cb_lookup.field_names,
        unknown_token=base_cb_lookup.unknown_token,
        weights=MappingProxyType(
            {field: MappingProxyType(weights) for field, weights in copied.items()}
        ),
    )

    with pytest.raises(ValueError, match="incomplete on the active pool"):
        build_full_training_cb_extension(
            inputs=production_inputs,
            pack=production_pack,
            base_lookup=broken,
        )


def test_production_manifest_embeds_complete_locked_extension_without_scoring(
    production_inputs,
    diagnostic_cb_extension,
):
    scope = validate_full_training_scope(production_inputs)
    records_metadata = {
        "path": full_replay.DEFAULT_RECORDS_OUTPUT,
        "bytes": 1,
        "sha256": "0" * 64,
        "jsonl_group_records": 3_240,
        "completion_records": 25_920,
        "ordered_sku_sha256": full_replay.EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": full_replay.EXPECTED_ORDERED_KEY_SHA256,
    }

    manifest = build_full_training_manifest(
        repo_root=ROOT,
        inputs=production_inputs,
        records_metadata=records_metadata,
        scope_validation=scope,
        cb_extension=diagnostic_cb_extension,
    )

    extension = manifest["cb_diagnostic_extension"]
    assert extension == dict(diagnostic_cb_extension.audit)
    assert len(extension["entries"]) == 13
    assert extension["ordered_entry_ledger_sha256"] == (
        "aeb089a1081d7efd1a99ccb2124e7b7412ec71f2362509f3df20dc2aa5837416"
    )
    assert manifest["integrity"]["cb_extension_ledger_sha256"] == (
        extension["ordered_entry_ledger_sha256"]
    )
    assert manifest["integrity"]["cb_active_weights_changed"] is False
    assert manifest["record_contract"]["cb_diagnostic_extension_included"] is True


@pytest.mark.parametrize("mutation", ["ledger", "lookup"])
def test_publication_rejects_extension_lookup_ledger_disagreement_before_scoring(
    tmp_path,
    monkeypatch,
    mutation,
):
    extension = _synthetic_cb_extension()
    if mutation == "ledger":
        audit = dict(extension.audit)
        audit["entries"] = []
        broken = replace(extension, audit=MappingProxyType(audit))
    else:
        broken_lookup = CBClassWeightLookup(
            artifact_version=extension.lookup.artifact_version,
            field_names=extension.lookup.field_names,
            unknown_token=extension.lookup.unknown_token,
            weights=MappingProxyType(
                {"field": MappingProxyType({"base_class": 1.0})}
            ),
        )
        broken = replace(extension, lookup=broken_lookup)

    def forbidden_scorer(**kwargs):
        raise AssertionError("extension mismatch must fail before scoring")

    monkeypatch.setattr(full_replay, "build_replay_group", forbidden_scorer)
    inputs = _inputs()
    with pytest.raises(ValueError, match="entry ledger is missing|additions differ"):
        publish_full_training_replay(
            repo_root=tmp_path,
            inputs=inputs,
            original_rewards={},
            pack=object(),
            cb_extension=broken,
            contract=_contract(inputs.authoritative_train_skus),
        )

    assert not (tmp_path / full_replay.DEFAULT_RECORDS_OUTPUT).exists()
    assert not (tmp_path / full_replay.DEFAULT_MANIFEST_OUTPUT).exists()
