"""W1 Step 4 — the eval harness.

The load-bearing test here is `test_macro_f1_punishes_the_majority_guesser`. The
plan's stated reason for choosing macro-F1 is that accuracy "will flatter a model
that only predicts crew neck, cotton, casual". If the harness cannot be shown to
punish exactly that model, the headline metric is not doing the job it was picked
for.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evalharness import predictions as preds_mod
from evalharness.metrics import ABSTAIN, NA_CLASS, evaluate, score_attribute
from evalharness.report import markdown_row
from labeling.records import (
    AttributeLabel,
    LabelStatus,
    Provenance,
    Row,
    RowInput,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def L(value=None, status=LabelStatus.LABELED):
    return AttributeLabel(value=value, status=status)


def mkrow(sku, labels):
    return Row(
        sku_id=sku,
        source="test",
        split="eval",
        input=RowInput(title="t"),
        labels=labels,
        provenance=Provenance(labeler="t", prompt_version="v1"),
    )


# --- the reason macro-F1 was chosen ------------------------------------------


def test_macro_f1_punishes_the_majority_guesser(pack):
    """90 crew necks and 10 rare ones. Answer "crew" every time.

    Accuracy says 90%. Macro-F1 must not, because the model has learned nothing
    except which answer is common — which is precisely the failure mode the plan
    picked this metric to expose.
    """
    gold = [L("crew")] * 90 + [L("cowl")] * 5 + [L("halter")] * 5
    pairs = [(g, "crew") for g in gold]
    score = score_attribute("neckline", pairs, pack.unknown_token)

    assert score.exact_match == pytest.approx(0.90)
    assert score.macro_f1 < 0.35, "macro-F1 must not reward a constant predictor"
    assert score.per_class["cowl"].f1 == 0.0
    assert score.per_class["halter"].f1 == 0.0


def test_macro_f1_rewards_getting_the_rare_classes_right(pack):
    gold = [L("crew")] * 90 + [L("cowl")] * 5 + [L("halter")] * 5
    pairs = [(g, g.value) for g in gold]
    score = score_attribute("neckline", pairs, pack.unknown_token)
    assert score.macro_f1 == pytest.approx(1.0)


# --- gold `unknown` is not ground truth --------------------------------------


def test_gold_unknown_cells_are_excluded_not_scored(pack):
    """The listing never said, so there is nothing to be right or wrong about."""
    pairs = [
        (L(status=LabelStatus.UNKNOWN), "cotton"),
        (L(status=LabelStatus.UNKNOWN), "silk"),
        (L("linen"), "linen"),
    ]
    score = score_attribute("material", pairs, pack.unknown_token)
    assert score.n_gold_unknown == 2
    assert score.n_scorable == 1
    assert score.exact_match == 1.0, "the two unscorable cells must not drag it down"


# --- abstention is reported twice, never folded in ---------------------------


def test_abstention_counts_as_a_miss_in_the_headline(pack):
    pairs = [(L("crew"), pack.unknown_token), (L("cowl"), "cowl")]
    score = score_attribute("neckline", pairs, pack.unknown_token)
    assert score.n_abstained == 1
    assert score.exact_match == pytest.approx(0.5)
    assert score.per_class["crew"].fn == 1
    assert ABSTAIN not in score.per_class, "abstaining is not a class you can be right about"


def test_selective_metrics_exclude_the_abstained_cells(pack):
    pairs = [(L("crew"), pack.unknown_token), (L("cowl"), "cowl")]
    score = score_attribute("neckline", pairs, pack.unknown_token)
    assert score.coverage == pytest.approx(0.5)
    assert score.selective_accuracy == pytest.approx(1.0)
    assert score.selective_macro_f1 == pytest.approx(1.0)


def test_abstaining_everywhere_scores_zero_not_one(pack):
    """The non-gameability check: silence must not be a winning strategy."""
    pairs = [(L("crew"), pack.unknown_token) for _ in range(10)]
    score = score_attribute("neckline", pairs, pack.unknown_token)
    assert score.exact_match == 0.0
    assert score.macro_f1 == 0.0
    assert score.coverage == 0.0


# --- not_applicable is a real class ------------------------------------------


def test_not_applicable_is_scored_as_a_class(pack):
    pairs = [
        (L(status=LabelStatus.NOT_APPLICABLE), None),
        (L(status=LabelStatus.NOT_APPLICABLE), "long"),
        (L("long"), None),
    ]
    score = score_attribute("sleeve_length", pairs, pack.unknown_token)
    assert NA_CLASS in score.per_class
    assert score.per_class[NA_CLASS].support == 2
    assert score.per_class[NA_CLASS].tp == 1
    assert score.exact_match == pytest.approx(1 / 3)


# --- multi-valued fields ------------------------------------------------------


def test_multi_field_scores_per_value_not_per_set(pack):
    gold = L(["lined", "pleated"])
    score = score_attribute("details", [(gold, ["lined", "slit"])], pack.unknown_token)
    assert score.per_class["lined"].tp == 1
    assert score.per_class["pleated"].fn == 1
    assert score.per_class["slit"].fp == 1
    assert score.exact_match == 0.0, "the set did not match exactly"


def test_multi_field_order_does_not_matter(pack):
    gold = L(["lined", "pleated"])
    score = score_attribute("details", [(gold, ["pleated", "lined"])], pack.unknown_token)
    assert score.exact_match == 1.0


# --- prediction loading ------------------------------------------------------


def test_raw_predictions_measure_schema_validity(pack, tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"sku_id": "a", "raw": json.dumps({"garment_category": "dress"})}),
                json.dumps({"sku_id": "b", "raw": "Sure! Here are the tags:"}),
                json.dumps({"sku_id": "c", "raw": "```json\n{}\n```"}),
            ]
        )
        + "\n"
    )
    loaded = preds_mod.load(p, pack)
    assert loaded.raw_mode
    assert loaded.schema_valid == 1
    assert loaded.unparseable == 2


def test_validity_denominator_is_attempts_not_survivors(pack, tmp_path):
    """Regression: 1 good line + 2 garbage lines is 33% valid, not 100%.

    Dividing by the rows that parsed excludes exactly the rows that failed. The
    first real run printed "100.0% valid (20 unparseable)" — a format failure
    hiding inside its own metric.
    """
    p = tmp_path / "preds.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"sku_id": "a", "raw": json.dumps({"garment_category": "dress"})}),
                json.dumps({"sku_id": "b", "raw": "nope"}),
                json.dumps({"sku_id": "c", "raw": "also nope"}),
            ]
        )
        + "\n"
    )
    loaded = preds_mod.load(p, pack)
    assert loaded.n_attempted == 3
    gold = [mkrow("a", {"garment_category": L("dress")})]
    rep = evaluate(
        gold,
        loaded.records,
        pack,
        schema_valid=loaded.schema_valid,
        unparseable=loaded.unparseable,
        n_attempted=loaded.n_attempted,
    )
    assert rep.schema_validity == pytest.approx(1 / 3)


def test_preparsed_predictions_report_validity_as_unknown(pack, tmp_path):
    """A pre-parsed file has already thrown its format errors away. 100% would lie."""
    p = tmp_path / "preds.jsonl"
    p.write_text(json.dumps({"sku_id": "a", "prediction": {"garment_category": "dress"}}) + "\n")
    loaded = preds_mod.load(p, pack)
    assert loaded.schema_valid is None
    assert loaded.raw_mode is False


def test_unparseable_predictions_are_dropped_not_zero_filled(pack, tmp_path):
    """Recording garbage as an empty record turns one format failure into 15 wrong
    answers, and flatters the format metric by counting a row it never produced."""
    p = tmp_path / "preds.jsonl"
    p.write_text(json.dumps({"sku_id": "a", "raw": "not json"}) + "\n")
    loaded = preds_mod.load(p, pack)
    assert "a" not in loaded.records


def test_frontier_predictions_come_from_the_snapshot(pack):
    row = mkrow("x", {"garment_category": L("dress")})
    row.provenance.frontier_labels = {"garment_category": L("top")}
    loaded = preds_mod.from_frontier([row], pack)
    assert loaded.records["x"]["garment_category"] == "top"


def test_frontier_abstention_respects_arity(pack):
    row = mkrow("x", {"details": L(["lined"])})
    row.provenance.frontier_labels = {"details": L(status=LabelStatus.UNKNOWN)}
    loaded = preds_mod.from_frontier([row], pack)
    assert loaded.records["x"]["details"] == [pack.unknown_token]


# --- report assembly ---------------------------------------------------------


def test_missing_rows_are_reported_not_scored_as_wrong(pack):
    gold = [mkrow("a", {"garment_category": L("dress")}), mkrow("b", {"garment_category": L("top")})]
    rep = evaluate(gold, {"a": {"garment_category": "dress"}}, pack)
    assert rep.n_missing == ["b"]
    assert rep.attributes["garment_category"].n_scorable == 1
    assert rep.macro_f1 == pytest.approx(1.0)


def test_trusted_macro_excludes_untrustworthy_gold(pack):
    gold = [
        mkrow(f"r{i}", {"garment_category": L("dress"), "material": L("cotton")})
        for i in range(4)
    ]
    preds = {f"r{i}": {"garment_category": "dress", "material": "silk"} for i in range(4)}
    rep = evaluate(gold, preds, pack, reward_weights={"garment_category": 1.0, "material": 0.0})
    assert rep.macro_f1 < rep.trusted_macro_f1
    assert rep.trusted_macro_f1 == pytest.approx(1.0)


def test_rule_violations_are_counted_per_rule(pack):
    hist = Counter({"sleeveless_has_no_sleeve_style": 3, "pants_length_subset": 1})
    rep = evaluate([], {}, pack, rule_histogram=hist)
    assert rep.rule_violations == 4
    assert len(rep.rule_histogram) == 2


def test_markdown_row_shape(pack):
    gold = [mkrow("a", {"garment_category": L("dress")})]
    rep = evaluate(gold, {"a": {"garment_category": "dress"}}, pack)
    row = markdown_row(rep, "frontier (ceiling)", "$0.004")
    assert row.startswith("| frontier (ceiling) | 1.0000 |")
    assert row.endswith("| $0.004 |")
    assert row.count("|") == 6


def test_headline_is_the_mean_of_per_attribute_macros(pack):
    """Every attribute counts once — a model cannot buy the headline with colour."""
    gold = [
        mkrow(f"r{i}", {"garment_category": L("dress"), "colour_primary": L("black")})
        for i in range(4)
    ]
    preds = {f"r{i}": {"garment_category": "top", "colour_primary": "black"} for i in range(4)}
    rep = evaluate(gold, preds, pack)
    a = rep.attributes["garment_category"].macro_f1
    b = rep.attributes["colour_primary"].macro_f1
    assert rep.macro_f1 == pytest.approx((a + b) / 2)
