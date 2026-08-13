from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.records import read_jsonl
from training.run2_checkpoint_monitor import (
    load_development_rows,
    score_monitor_outputs,
    score_prediction_set,
    validate_contract_inputs,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "runs/grpo-run2-checkpoint-monitor-contract.json").read_text()
)


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs/vastraa_taste_v1")


def _record(row, pack):
    return json.dumps(row.to_verifier_record(pack), sort_keys=True)


def test_locked_development_loader_returns_exact_production_and_smoke_views():
    production, views = load_development_rows(root=ROOT, contract=CONTRACT, mode="production")
    smoke, smoke_views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")

    assert len(production) == 360
    assert [row.sku_id for row in production] == CONTRACT["development"]["views"]["representative_all"]["sku_ids_in_source_order"]
    assert {name: len(value) for name, value in views.items()} == {
        "difficult_0_to_2_of_8": 204,
        "easy_retention_6_to_8_of_8": 110,
        "middle_3_to_5_of_8": 46,
        "representative_all": 360,
    }
    assert [row.sku_id for row in smoke] == CONTRACT["smoke"]["sku_ids"]
    assert all(smoke_views.values())


def test_contract_input_byte_drift_fails_closed(tmp_path):
    contract = json.loads(json.dumps(CONTRACT))
    contract["inputs"]["pack_rules"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="pack_rules"):
        validate_contract_inputs(contract, ROOT)


def test_perfect_raw_predictions_score_all_attempts_and_both_rewards(pack):
    rows, _views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")
    predictions = [{"sku_id": row.sku_id, "raw": _record(row, pack)} for row in rows]

    result = score_prediction_set(rows=rows, predictions=predictions, pack=pack)

    assert result["attempted"] == result["parsed"] == 4
    assert result["unparseable"] == 0
    assert result["scalars"]["macro_f1"] == pytest.approx(1.0)
    assert result["scalars"]["schema_validity"] == 1.0
    assert result["scalars"]["vocab_validity"] == 1.0
    assert result["scalars"]["original_reward_mean"] == 4.0
    assert result["rewards"]["candidate_ua"]["values"]


def test_unparseable_output_stays_in_validity_denominator(pack):
    rows, _views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")
    predictions = [{"sku_id": row.sku_id, "raw": _record(row, pack)} for row in rows]
    predictions[0]["raw"] = "not json"

    result = score_prediction_set(rows=rows, predictions=predictions, pack=pack)

    assert result["attempted"] == 4
    assert result["parsed"] == 3
    assert result["unparseable"] == 1
    assert result["scalars"]["schema_validity"] == 0.75
    assert result["primary_metrics_conditional_on_parseable_outputs"] is True
    assert len(result["rewards"]["original_1_1_2"]["values"]) == 4


def test_prediction_membership_order_and_duplicates_fail_closed(pack):
    rows, _views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")
    predictions = [{"sku_id": row.sku_id, "raw": _record(row, pack)} for row in rows]
    with pytest.raises(ValueError, match="order or membership"):
        score_prediction_set(rows=rows, predictions=list(reversed(predictions)), pack=pack)
    predictions[1]["sku_id"] = predictions[0]["sku_id"]
    with pytest.raises(ValueError, match="duplicate"):
        score_prediction_set(rows=rows, predictions=predictions, pack=pack)


def test_greedy_and_sampled_outputs_aggregate_every_fixed_view(pack):
    rows, views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")
    greedy = [{"sku_id": row.sku_id, "raw": _record(row, pack)} for row in rows]
    sampled = []
    seeds = [11, 12]
    for repeat, seed in enumerate(seeds):
        sampled.extend(
            {"sku_id": row.sku_id, "raw": _record(row, pack), "repeat": repeat, "seed": seed}
            for row in rows
        )

    result = score_monitor_outputs(
        rows=rows,
        views=views,
        greedy_predictions=greedy,
        sampled_predictions=sampled,
        sampled_seeds=seeds,
        pack=pack,
    )

    assert result["status"] == "checkpoint_outputs_scored"
    assert set(result["views"]) == set(views)
    representative = result["views"]["representative_all"]
    assert representative["greedy"]["scalars"]["macro_f1"] == 1.0
    assert representative["sampled"]["aggregate"]["macro_f1"] == {
        "values": [1.0, 1.0],
        "mean": 1.0,
        "population_stddev": 0.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }


def test_sample_count_or_lineage_drift_fails_closed(pack):
    rows, views = load_development_rows(root=ROOT, contract=CONTRACT, mode="smoke")
    greedy = [{"sku_id": row.sku_id, "raw": _record(row, pack)} for row in rows]
    sampled = [
        {"sku_id": row.sku_id, "raw": _record(row, pack), "repeat": repeat, "seed": seed}
        for repeat, seed in enumerate((11, 12))
        for row in rows
    ]
    with pytest.raises(ValueError, match="count"):
        score_monitor_outputs(
            rows=rows,
            views=views,
            greedy_predictions=greedy,
            sampled_predictions=sampled[:-1],
            sampled_seeds=[11, 12],
            pack=pack,
        )
    sampled[0]["seed"] = 99
    with pytest.raises(ValueError, match="lineage"):
        score_monitor_outputs(
            rows=rows,
            views=views,
            greedy_predictions=greedy,
            sampled_predictions=sampled,
            sampled_seeds=[11, 12],
            pack=pack,
        )
