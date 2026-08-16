"""Self-consistency confidence and the escalation queue.

The escalation rate is the unit economics of the product: the share of cells a
human must look at. These tests pin the arithmetic, the three-state label
mapping, and the two ways the rate can be silently corrupted — by counting an
unmeasurable product as confident, and by hiding one bad attribute inside a
healthy average.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from production.escalation import (
    DEFAULT_K,
    escalate,
    label_from_value,
    samples_to_labels,
    threshold_curve,
    write_queue,
)
from labeling.records import LabelStatus
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def record(**overrides):
    base = {name: "unknown" for name in
            ("closure", "colour_primary", "fit", "garment_category", "garment_length",
             "material", "occasion", "pattern", "silhouette", "waistline")}
    base.update({"collar_type": None, "neckline": None, "sleeve_length": None,
                 "sleeve_style": None, "details": ["unknown"]})
    base.update(overrides)
    return base


# --- the three-state mapping --------------------------------------------------


def test_null_and_unknown_are_not_collapsed(pack):
    """`null` means the field cannot apply; unknown means the listing did not
    say. Collapsing them would make an abstention look like an inapplicability
    and drop it out of the queue, which is where a human most needs it."""
    assert label_from_value(None, pack).status is LabelStatus.NOT_APPLICABLE
    assert label_from_value("unknown", pack).status is LabelStatus.UNKNOWN
    assert label_from_value("cotton", pack).status is LabelStatus.LABELED
    assert label_from_value(["unknown"], pack).status is LabelStatus.UNKNOWN


def test_multi_valued_fields_compare_order_insensitively(pack):
    samples = samples_to_labels(
        [record(details=["lined", "pleated"]), record(details=["pleated", "lined"])],
        pack,
    )
    _cells, _conf, report = escalate(
        [("sku", [record(details=["lined", "pleated"]),
                  record(details=["pleated", "lined"])])],
        pack=pack, threshold=1.0, k=2,
    )
    assert report.per_attribute["details"]["escalated"] == 0, "reorder is agreement"


# --- the rate itself ----------------------------------------------------------


def test_unanimous_samples_escalate_nothing(pack):
    samples = [record(material="cotton")] * DEFAULT_K
    cells, _conf, report = escalate(
        [("sku-1", samples)], pack=pack, threshold=1.0, k=DEFAULT_K
    )
    assert cells == [] and report.escalation_rate == 0.0
    assert report.products == 1 and report.cells == 15


def test_one_disagreeing_sample_escalates_only_that_cell(pack):
    samples = [record(material="cotton")] * 4 + [record(material="silk")]
    cells, _conf, report = escalate(
        [("sku-1", samples)], pack=pack, threshold=1.0, k=DEFAULT_K
    )
    assert [c.attribute for c in cells] == ["material"]
    assert cells[0].agreement == 0.8 and cells[0].n_variants == 2
    assert report.escalated_cells == 1
    assert report.escalation_rate == pytest.approx(1 / 15)


def test_a_lower_threshold_accepts_a_majority(pack):
    """4 of 5 agreeing passes a 0.8 threshold and fails a 1.0 threshold."""
    samples = [record(material="cotton")] * 4 + [record(material="silk")]
    _cells, _conf, strict = escalate([("s", samples)], pack=pack, threshold=1.0, k=5)
    _cells2, _conf2, loose = escalate([("s", samples)], pack=pack, threshold=0.8, k=5)
    assert strict.escalated_cells == 1
    assert loose.escalated_cells == 0


def test_a_product_whose_samples_all_failed_is_excluded_not_counted_confident(pack):
    """It is unmeasured, not confident. Counting it either way corrupts the rate."""
    _cells, _conf, report = escalate(
        [("dead", []), ("alive", [record()] * 3)], pack=pack, threshold=1.0, k=3
    )
    assert report.products == 1, "only the measurable product counts"


def test_per_attribute_rates_expose_a_single_bad_field(pack):
    """One attribute at 100% escalation inside 15 gives a 6.7% headline. The
    per-attribute breakdown is what stops that from looking healthy."""
    samples = [record(closure="pullover")] * 2 + [record(closure="zip")] * 3
    _cells, _conf, report = escalate(
        [("s", samples)], pack=pack, threshold=1.0, k=5
    )
    summary = report.summary()
    assert summary["escalation_rate"] == pytest.approx(1 / 15, abs=1e-3)
    worst = next(iter(summary["per_attribute"]))
    assert worst == "closure"
    assert summary["per_attribute"]["closure"]["escalation_rate"] == 1.0


def test_product_touch_rate_is_reported_beside_the_cell_rate(pack):
    """A reviewer opens products, not cells. 1 cell in 15 is a small cell rate
    and a 100% chance of having to open the product."""
    samples = [record(material="cotton")] * 4 + [record(material="silk")]
    _cells, _conf, report = escalate([("s", samples)], pack=pack, threshold=1.0, k=5)
    summary = report.summary()
    assert summary["escalation_rate"] < 0.07
    assert summary["product_touch_rate"] == 1.0


# --- the curve ----------------------------------------------------------------


def test_the_curve_is_monotonic_and_uses_achievable_thresholds(pack):
    """Escalation can only rise as the threshold rises, and only multiples of
    1/k are achievable, so intermediate thresholds would add false resolution."""
    samples = [
        ("a", [record(material="cotton")] * 5),
        ("b", [record(material="cotton")] * 4 + [record(material="silk")]),
        ("c", [record(material="cotton")] * 3 + [record(material="silk")] * 2),
    ]
    curve = threshold_curve(samples, pack=pack, k=5)
    assert [row["threshold"] for row in curve] == [0.2, 0.4, 0.6, 0.8, 1.0]
    rates = [row["escalation_rate"] for row in curve]
    assert rates == sorted(rates), "a higher bar cannot escalate less"
    assert rates[-1] > rates[0]


# --- the queue ----------------------------------------------------------------


def test_the_queue_is_a_csv_sorted_worst_first(tmp_path, pack):
    """CSV because the W1 review round was actually completed on a phone
    spreadsheet. Worst-first because a reviewer with limited time should spend
    it where the model is least sure."""
    samples = ([("a", [record(material="cotton")] * 3 + [record(material="silk")] * 2)]
               + [("b", [record(fit="loose")] * 4 + [record(fit="tight")])])
    cells, _conf, _report = escalate(samples, pack=pack, threshold=1.0, k=5)
    path = tmp_path / "queue" / "review.csv"
    write_queue(path, cells)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["attribute"] == "material", "lowest agreement first"
    assert float(rows[0]["agreement"]) <= float(rows[-1]["agreement"])
    assert set(rows[0]) == {
        "sku_id", "attribute", "agreement", "n_variants", "proposed", "status"
    }


def test_an_empty_report_does_not_divide_by_zero(pack):
    _cells, _conf, report = escalate([], pack=pack, threshold=1.0, k=5)
    assert report.summary()["escalation_rate"] == 0.0
    assert report.summary()["product_touch_rate"] == 0.0


# --- shapes only real output produces ----------------------------------------


def test_an_empty_multi_valued_field_does_not_crash_the_mapping(pack):
    """`details: []` is a legal record and the model emits it. Hand-built test
    records never did, so this went undetected until a real run."""
    label = label_from_value([], pack)
    assert label.status is LabelStatus.NOT_APPLICABLE
    assert label.value == []


def test_the_four_spellings_of_absence_stay_four_distinct_answers(pack):
    """For `details` the model says nothing in four ways: `unknown`, `none`,
    `null` and `[]`. Collapsing any two would manufacture agreement between
    answers that were not the same answer, and agreement is what decides
    whether a human ever sees the cell."""
    keys = {
        label_from_value(value, pack).key()
        for value in ([pack.unknown_token], ["none"], None, [])
    }
    assert len(keys) == 4, f"absence spellings collapsed: {keys}"


def test_every_real_production_record_maps_without_raising(pack):
    """The regression guard. Runs the mapping over committed production output
    rather than records this file invented, because the invented ones are what
    hid the defect."""
    run_dir = ROOT / "runs" / "production-demo"
    paths = sorted(run_dir.glob("pass-*.jsonl"))
    if not paths:
        pytest.skip("no production run committed yet")

    parsed = [
        json.loads(line)["parsed"]
        for path in paths
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("parsed") is not None
    ]
    assert parsed, "committed run has no parsed records"
    # Raises on any shape the mapping cannot represent.
    labels = samples_to_labels(parsed, pack)
    assert len(labels) == len(parsed)


def test_the_worst_first_ranking_survives_serialisation(pack):
    """`summary()` builds per_attribute worst-first, but the report is written
    with sort_keys=True and JSON objects are unordered anyway. Without an
    explicit ranking a reader takes the alphabetically first attribute for the
    worst one."""
    samples = [
        ("a", [record(material="cotton")] * 3 + [record(material="silk")] * 2),
        ("b", [record(fit="loose")] * 5),
    ]
    _cells, _conf, report = escalate(samples, pack=pack, threshold=1.0, k=5)
    summary = report.summary()

    ranking = summary["per_attribute_worst_first"]
    assert set(ranking) == set(summary["per_attribute"])
    rates = [summary["per_attribute"][name]["escalation_rate"] for name in ranking]
    assert rates == sorted(rates, reverse=True), "ranking is not worst-first"

    # The property that actually matters: it round-trips through the writer.
    reloaded = json.loads(json.dumps(summary, sort_keys=True))
    assert reloaded["per_attribute_worst_first"] == ranking
