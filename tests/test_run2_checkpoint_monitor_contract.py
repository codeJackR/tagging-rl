from copy import deepcopy
from pathlib import Path

import pytest

import training.run2_checkpoint_monitor_contract as contract


ROOT = Path(__file__).resolve().parent.parent


def test_production_contract_locks_population_decoding_and_boundaries():
    result = contract.build_contract(ROOT)

    assert result["development"]["rows"] == 360
    assert {name: view["rows"] for name, view in result["development"]["views"].items()} == {
        "representative_all": 360,
        "difficult_0_to_2_of_8": 204,
        "middle_3_to_5_of_8": 46,
        "easy_retention_6_to_8_of_8": 110,
    }
    assert result["decoding"]["sampled"] == {
        "repetitions": 8,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "seeds": list(range(20260813, 20260821)),
    }
    assert result["abort_policy"]["quality_abort_enabled"] is False
    assert result["boundaries"] == {
        "confirmation_paths_allowed": False,
        "legacy_frozen_300_allowed": False,
        "full_grpo_training_authorized": False,
        "checkpoint_quality_claim_authorized_by_smoke": False,
    }


def test_smoke_membership_covers_fixed_views_without_becoming_quality_evidence():
    result = contract.build_contract(ROOT)
    smoke = result["smoke"]
    views = result["development"]["views"]

    assert smoke["rows"] == 4
    assert smoke["quality_evidence"] is False
    assert smoke["sku_ids"][0] in views["difficult_0_to_2_of_8"]["sku_ids_in_source_order"]
    assert smoke["sku_ids"][1] in views["middle_3_to_5_of_8"]["sku_ids_in_source_order"]
    assert smoke["sku_ids"][2] in views["easy_retention_6_to_8_of_8"]["sku_ids_in_source_order"]


def test_view_membership_drift_fails_closed(monkeypatch):
    real_loads = contract.json.loads

    def drifted_loads(text):
        value = real_loads(text)
        if value.get("version") == "grpo-run2-data-role-manifest-v1":
            value = deepcopy(value)
            value["development"]["views"]["representative_all"]["rows"] = 359
        return value

    monkeypatch.setattr(contract.json, "loads", drifted_loads)
    with pytest.raises(ValueError, match="row count drifted"):
        contract.build_contract(ROOT)


def test_selected_reward_drift_fails_closed(monkeypatch):
    real_loads = contract.json.loads

    def drifted_loads(text):
        value = real_loads(text)
        if value.get("version") == "grpo-run2-complexity-aware-selection-v1":
            value = deepcopy(value)
            value["selected_candidate"] = "CB"
        return value

    monkeypatch.setattr(contract.json, "loads", drifted_loads)
    with pytest.raises(ValueError, match="Candidate UA"):
        contract.build_contract(ROOT)
