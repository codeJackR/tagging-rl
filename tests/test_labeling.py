"""W1 Step 3 — the dataset pipeline.

The load-bearing test in this file is `test_reliability_detects_planted_bias`.
A reliability table that cannot be shown to catch a known-bad attribute is not
evidence of anything, and the whole point of Step 3 is that this table is what
breaks the eval/train circularity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from labeling import consensus, freeze, lengths, reliability, review, splits
from labeling.records import (
    AttributeLabel,
    LabelStatus,
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    from_verifier_record,
    read_jsonl,
    write_jsonl,
)
from tools.make_synthetic_feed import build
from verifier import load_pack, verify_record

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def corpus(pack):
    """400 synthetic rows with a deliberate bias planted in two attributes."""
    rows, truth = build(
        pack, 400, seed=7, k=5, bias={"material": 0.35, "occasion": 0.30}
    )
    return rows, truth


def _correct(rows, truth):
    """Simulate the human pass: snapshot the frontier, then write ground truth."""
    for r in rows:
        r.provenance.frontier_labels = {
            k: v.model_copy(deep=True) for k, v in r.labels.items()
        }
        r.labels = {k: v.model_copy(deep=True) for k, v in truth[r.sku_id].items()}
        r.mark_corrected("2026-07-27")


# --- records: the three-state contract ---------------------------------------


def test_labeled_requires_a_value():
    with pytest.raises(ValueError):
        AttributeLabel(status=LabelStatus.LABELED)


def test_non_labeled_must_not_carry_a_value():
    with pytest.raises(ValueError):
        AttributeLabel(value="crew", status=LabelStatus.UNKNOWN)


def test_not_applicable_and_unknown_are_distinct():
    na = AttributeLabel(status=LabelStatus.NOT_APPLICABLE)
    unk = AttributeLabel(status=LabelStatus.UNKNOWN)
    assert na.key() != unk.key(), "collapsing these is the failure the schema prevents"


def test_multi_value_comparison_is_order_insensitive():
    a = AttributeLabel(value=["lined", "pleated"], status=LabelStatus.LABELED)
    b = AttributeLabel(value=["pleated", "lined"], status=LabelStatus.LABELED)
    assert a.key() == b.key()


# --- records: the bridge to the Step 2 verifier ------------------------------


def test_labels_grade_clean_against_the_verifier(corpus, pack):
    rows, _ = corpus
    for row in rows[:120]:
        res = verify_record(row.to_verifier_record(pack), pack)
        assert res.schema_valid, res.errors
        assert res.vocab_valid, res.errors


def test_multi_field_abstention_respects_arity(pack):
    """Regression: a multi-valued field declines with ["unknown"], not "unknown".

    Emitting the bare string produced a structurally invalid record — caught by the
    Step 2 verifier the first time this pipeline ran end to end.
    """
    multi = next(n for n, s in pack.specs.items() if s.kind == "multi")
    single = next(n for n, s in pack.specs.items() if s.kind == "single")
    row = Row(
        sku_id="X",
        source="test",
        split="train",
        input=RowInput(title="t"),
        labels={
            multi: AttributeLabel(status=LabelStatus.UNKNOWN),
            single: AttributeLabel(status=LabelStatus.UNKNOWN),
        },
        provenance=Provenance(labeler="t", prompt_version="v1"),
    )
    rec = row.to_verifier_record(pack)
    assert rec[multi] == [pack.unknown_token]
    assert rec[single] == pack.unknown_token
    assert verify_record(rec, pack).schema_valid


def test_verifier_record_round_trips(corpus, pack):
    rows, _ = corpus
    row = rows[0]
    back = from_verifier_record(row.to_verifier_record(pack), pack)
    assert {k: v.key() for k, v in back.items()} == {
        k: v.key() for k, v in row.labels.items()
    }


# --- consensus ---------------------------------------------------------------


def test_agreement_counts_the_modal_answer():
    s = [
        {"a": AttributeLabel(value="x", status=LabelStatus.LABELED)},
        {"a": AttributeLabel(value="x", status=LabelStatus.LABELED)},
        {"a": AttributeLabel(value="y", status=LabelStatus.LABELED)},
        {"a": AttributeLabel(value="x", status=LabelStatus.LABELED)},
    ]
    c = consensus.consensus(s)["a"]
    assert c.label.value == "x"
    assert c.agreement == 0.75
    assert c.n_variants == 2


def test_review_queue_skips_unanimous_cells(corpus):
    rows, _ = corpus
    saving = consensus.queue_savings(rows)
    assert saving["cells_to_review"] < saving["cells_total"]
    assert saving["reduction"] > 0.5, "consensus should remove most of the queue"


def test_always_review_overrides_unanimity(corpus):
    rows, _ = corpus
    row = next(
        r for r in rows if all(v == 1.0 for v in r.provenance.self_consistency.agreement.values())
    )
    assert consensus.review_cells(row) == []
    forced = consensus.review_cells(row, always_review=("material",))
    assert [c.attribute for c in forced] == ["material"]


def test_row_and_cell_escalation_differ(corpus):
    """Both numbers are honest; quoting only the flattering one is not."""
    rows, _ = corpus
    assert consensus.escalation_rate(rows) > consensus.queue_savings(rows)[
        "cells_to_review"
    ] / consensus.queue_savings(rows)["cells_total"]


# --- reliability: the centerpiece --------------------------------------------


def test_reliability_detects_planted_bias(corpus):
    """The table must find the attributes we deliberately corrupted, and only those.

    `material` (35% error) and `occasion` (30%) were planted; everything else runs
    at the 2% floor. If this test ever passes vacuously — e.g. by marking
    everything unsafe — the reliability table is not doing its job.
    """
    rows, truth = corpus
    rows = [r.model_copy(deep=True) for r in rows[:300]]
    _correct(rows, truth)

    table = reliability.reliability_table(rows)
    weights = reliability.reward_weights(rows)

    assert table["material"].accuracy < 0.90
    assert table["occasion"].accuracy < 0.95
    assert weights["material"] < 1.0, "a 35%-error attribute must not carry full weight"

    clean = [n for n in table if n not in ("material", "occasion")]
    assert all(table[n].accuracy > 0.95 for n in clean), "false positives on clean attrs"
    assert all(weights[n] == 1.0 for n in clean)


def test_reliability_needs_the_frontier_snapshot(corpus):
    """Correcting in place without snapshotting destroys the evidence."""
    rows, truth = corpus
    rows = [r.model_copy(deep=True) for r in rows[:50]]
    for r in rows:
        r.labels = {k: v.model_copy(deep=True) for k, v in truth[r.sku_id].items()}
        r.mark_corrected()  # no snapshot taken
    assert reliability.reliability_table(rows) == {}
    assert "frontier_labels" in reliability.format_report(rows)


def test_verdict_uses_the_lower_confidence_bound():
    """A point estimate on the threshold must not earn the verdict — the bound must.

    285/300 is exactly 95%, the `safe` cut-off, but its lower bound is ~0.92: the
    true rate could plausibly be well under 95%. Calling that `safe` would restore
    the false confidence this table exists to remove, so it lands on `usable`.
    Reaching `safe` takes evidence, not a lucky point estimate.
    """
    lo_10, hi_10 = reliability.wilson(9, 10)
    assert lo_10 < 0.85 < hi_10, "9/10 is consistent with far worse than 85%"
    assert reliability._verdict(lo_10, 10) == "insufficient_data"

    on_threshold = reliability.wilson(285, 300)[0]
    assert on_threshold < reliability.SAFE
    assert reliability._verdict(on_threshold, 300) == "usable"

    convincing = reliability.wilson(297, 300)[0]
    assert reliability._verdict(convincing, 300) == "safe"


def test_frontier_baseline_is_reported(corpus):
    rows, truth = corpus
    rows = [r.model_copy(deep=True) for r in rows[:300]]
    _correct(rows, truth)
    base = reliability.frontier_baseline(rows)
    assert 0.0 < base["macro_accuracy"] <= 1.0
    assert base["n_rows"] == 300


# --- splits ------------------------------------------------------------------


def test_stratification_hits_coverage_targets(corpus):
    rows, _ = corpus
    plan = splits.stratify(
        [r.model_copy(deep=True) for r in rows],
        eval_size=200,
        probe_size=60,
        min_per_value=5,
        seed=3,
    )
    assert len(plan.eval_rows) == 200
    assert len(plan.probe_rows) == 60
    assert plan.coverage["values_at_target"] == plan.coverage["values_tracked"]
    assert plan.shortfalls == []


def test_splits_are_deterministic(corpus):
    rows, _ = corpus
    a = splits.stratify([r.model_copy(deep=True) for r in rows], eval_size=100, probe_size=50, seed=11)
    b = splits.stratify([r.model_copy(deep=True) for r in rows], eval_size=100, probe_size=50, seed=11)
    assert [r.sku_id for r in a.eval_rows] == [r.sku_id for r in b.eval_rows]


def test_splits_are_disjoint(corpus):
    rows, _ = corpus
    p = splits.stratify([r.model_copy(deep=True) for r in rows], eval_size=150, probe_size=50, seed=5)
    ids = [{r.sku_id for r in g} for g in (p.eval_rows, p.probe_rows, p.train_rows)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(rows)


def test_shortfalls_are_reported_not_hidden(corpus):
    """A silent cap reads as full coverage. Under-covered values must surface."""
    rows, _ = corpus
    plan = splits.stratify(
        [r.model_copy(deep=True) for r in rows], eval_size=40, probe_size=10, min_per_value=25, seed=2
    )
    assert plan.shortfalls, "40 rows cannot cover every value 25 times"
    assert "BELOW TARGET" in splits.format_coverage(plan, min_per_value=25)


# --- lengths -----------------------------------------------------------------


def test_heuristic_tokenizer_is_never_mistaken_for_measured(corpus):
    rows, _ = corpus
    rep = lengths.length_report(rows[:50], lengths.HeuristicTokenizer())
    assert rep.measured is False
    assert "WARNING" in rep.recommend()
    assert "heuristic" in rep.tokenizer


def test_budget_exceeds_the_observed_tail(corpus):
    """The smoke-test failure was a budget below p99. Never recommend one again."""
    rows, _ = corpus
    rep = lengths.length_report(rows[:100], lengths.HeuristicTokenizer())
    rec = rep.recommend()
    assert rec["max_completion_length"] > rep.target["p99"]
    assert rec["max_prompt_length"] > rep.prompt["p95"]


def test_reasoning_block_widens_the_completion_budget(corpus):
    rows, _ = corpus
    rep = lengths.length_report(rows[:50], lengths.HeuristicTokenizer())
    assert (
        rep.recommend(reasoning_block=True)["max_completion_length"]
        > rep.recommend()["max_completion_length"]
    )


def test_stamped_rows_record_the_tokenizer(corpus):
    rows, _ = corpus
    sample = [r.model_copy(deep=True) for r in rows[:20]]
    lengths.stamp_rows(sample, lengths.HeuristicTokenizer())
    assert all(r.length_stats.tokenizer and r.length_stats.target_tokens for r in sample)


# --- review round-trip -------------------------------------------------------


def test_review_csv_round_trip(corpus, pack, tmp_path):
    rows, _ = corpus
    rows = [r.model_copy(deep=True) for r in rows[:40]]
    csv_path = tmp_path / "queue.csv"
    summary = review.export_review_csv(rows, csv_path)
    assert summary["cells_to_review"] > 0

    # Blank corrections mean "the proposal is right" — still counts as reviewed.
    result = review.import_review_csv(rows, csv_path, pack)
    assert result["cells_changed"] == 0
    assert result["cells_accepted"] == summary["cells_to_review"]
    assert all(r.provenance.frontier_labels is not None for r in rows if r.provenance.human_corrected)


def test_import_snapshots_before_editing(corpus, pack, tmp_path):
    rows, _ = corpus
    rows = [r.model_copy(deep=True) for r in rows[:30]]
    csv_path = tmp_path / "q.csv"
    review.export_review_csv(rows, csv_path)

    lines = list(csv.DictReader(csv_path.open()))
    target = lines[0]
    attr = target["attribute"]
    before = next(r for r in rows if r.sku_id == target["sku_id"]).labels[attr].key()
    new_value = next(v for v in pack.specs[attr].values if v != target["proposed_value"])
    target["corrected_value"] = new_value
    target["corrected_status"] = "labeled"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=review.COLUMNS)
        w.writeheader()
        w.writerows(lines)

    review.import_review_csv(rows, csv_path, pack)
    row = next(r for r in rows if r.sku_id == target["sku_id"])
    assert row.provenance.frontier_labels[attr].key() == before, "original must survive"
    assert row.labels[attr].value == new_value


def test_import_rejects_out_of_vocab_corrections(corpus, pack, tmp_path):
    rows, _ = corpus
    rows = [r.model_copy(deep=True) for r in rows[:20]]
    csv_path = tmp_path / "q.csv"
    review.export_review_csv(rows, csv_path)
    lines = list(csv.DictReader(csv_path.open()))
    lines[0]["corrected_value"] = "not_a_real_value"
    lines[0]["corrected_status"] = "labeled"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=review.COLUMNS)
        w.writeheader()
        w.writerows(lines)
    result = review.import_review_csv(rows, csv_path, pack)
    assert result["errors"] and "controlled vocabulary" in result["errors"][0]


# --- freeze ------------------------------------------------------------------


def test_checksum_is_stable_across_reserialization(corpus, tmp_path):
    rows, _ = corpus
    rows = rows[:25]
    first = freeze.checksum(rows)
    write_jsonl(rows, tmp_path / "a.jsonl")
    assert freeze.checksum(read_jsonl(tmp_path / "a.jsonl")) == first


def test_checksum_ignores_row_order(corpus):
    rows, _ = corpus
    rows = rows[:25]
    assert freeze.checksum(rows) == freeze.checksum(list(reversed(rows)))


def test_freeze_detects_drift(corpus, tmp_path):
    rows, _ = corpus
    rows = [r.model_copy(deep=True) for r in rows[:25]]
    path = tmp_path / "eval.jsonl"
    freeze.freeze(rows, path)
    assert freeze.verify(path)["ok"]

    rows[0].labels[next(iter(rows[0].labels))] = AttributeLabel(
        status=LabelStatus.UNKNOWN
    )
    write_jsonl(rows, path)
    result = freeze.verify(path)
    assert not result["ok"]
    assert "MISMATCH" in result["reason"]
    assert "no longer comparable" in result["consequence"]


def test_verify_reports_a_missing_sidecar(tmp_path, corpus):
    rows, _ = corpus
    write_jsonl(rows[:5], tmp_path / "eval.jsonl")
    assert not freeze.verify(tmp_path / "eval.jsonl")["ok"]


# --- the column that stays empty ---------------------------------------------


def test_difficulty_is_left_unset(corpus):
    """sft_pass_rate needs a model to measure against — it arrives in W2, not here."""
    rows, _ = corpus
    assert all(r.difficulty.sft_pass_rate is None for r in rows)


def test_empty_list_maps_onto_the_vocabulary_none_value(pack):
    """A real 20,000-request run returned `details: []` 178 times.

    Not an error and not an abstention — the model answered "nothing applies", and
    the vocabulary already has a value meaning that. Normalize onto it rather than
    inventing a fourth state.
    """
    labels = from_verifier_record({"details": []}, pack)
    assert labels["details"].status is LabelStatus.LABELED
    assert labels["details"].value == ["none"]


def test_empty_value_without_a_none_in_vocab_becomes_unknown(pack):
    """Conservative: asserting not_applicable would be a claim the model never made."""
    labels = from_verifier_record({"colour_primary": ""}, pack)
    assert labels["colour_primary"].status is LabelStatus.UNKNOWN


def test_converter_never_raises_on_wellformed_json(pack):
    """A converter that crashes on one row discards every row parsed alongside it."""
    for junk in ({"details": []}, {"neckline": ""}, {"details": ["unknown"]},
                 {"garment_category": None}, {}):
        from_verifier_record(junk, pack)
