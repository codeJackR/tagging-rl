"""The blind spot in the confidence signal.

The escalation rate says how much a human must review. It does not say that the
unreviewed remainder is correct, and these tests pin the distinction: a record
can be unanimous across every sample and still be rejected by the verifier,
because agreement measures stability and the gate measures correctness.

The load-bearing test is `test_a_unanimous_failure_is_counted_as_invisible`.
If that ever passes vacuously, the run report's escalation number goes back to
implying something it cannot support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.confidence_audit import audit, load_run
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
K = 5


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def record(**overrides):
    base = {
        name: "unknown"
        for name in (
            "closure", "colour_primary", "fit", "garment_category", "garment_length",
            "material", "occasion", "pattern", "silhouette", "waistline",
        )
    }
    base.update({
        "collar_type": None, "neckline": None, "sleeve_length": None,
        "sleeve_style": None, "details": ["unknown"],
    })
    base.update(overrides)
    return base


def failing(*violations):
    return {"gate_passed": False, "rule_violations": list(violations)}


PASSING = {"gate_passed": True, "rule_violations": []}


# --- the finding this module exists for --------------------------------------


def test_a_unanimous_failure_is_counted_as_invisible(pack):
    """Every sample agrees, and the verifier still rejects it. No threshold,
    including 1.0, can route this to a human."""
    samples = [record(garment_category="top", waistline="none")] * K
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.gate_failed == 1
    assert report.gate_failed_and_unanimous == 1
    assert report.invisible_share == 1.0


def test_a_split_failure_is_not_invisible(pack):
    """Disagreement on any cell means escalation can surface the product, so it
    is a failure the confidence signal *can* see."""
    samples = [record(garment_category="top", waistline="none")] * (K - 1)
    samples.append(record(garment_category="top", waistline=None))
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.gate_failed == 1
    assert report.gate_failed_and_unanimous == 0
    assert report.invisible_share == 0.0


def test_disagreement_on_an_unrelated_cell_still_makes_it_visible(pack):
    """The product is the review unit. A human pulled in by any low-agreement
    cell sees the whole record, including the cell that actually failed."""
    samples = [record(garment_category="top", waistline="none", material="cotton")] * (K - 1)
    samples.append(record(garment_category="top", waistline="none", material="linen"))
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.gate_failed_and_unanimous == 0


def test_a_passing_product_is_never_counted_as_a_failure(pack):
    report = audit(
        [("sku-1", [record()] * K)], {"sku-1": PASSING}, pack=pack, k=K
    )
    assert report.products == 1
    assert report.gate_failed == 0
    assert report.gate_failed_and_unanimous == 0


# --- the two ways the number could be silently corrupted ---------------------


def test_a_product_with_fewer_than_k_samples_is_skipped(pack):
    """Agreement over 3 samples is not comparable to agreement over 5, and
    mixing the two would make the share meaningless."""
    report = audit(
        [("sku-1", [record(garment_category="top", waistline="none")] * 3)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.products == 0
    assert report.gate_failed == 0


def test_a_product_with_no_verdict_is_skipped_not_assumed(pack):
    """An unscored product is unmeasured. Counting it as either passing or
    failing would be an invention."""
    report = audit([("sku-1", [record()] * K)], {}, pack=pack, k=K)
    assert report.products == 0


def test_unparseable_samples_do_not_count_toward_k(pack):
    samples = [record(garment_category="top", waistline="none")] * 4 + [None]
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.products == 0, "4 usable samples is not 5"


# --- attribution --------------------------------------------------------------


def test_applies_to_violations_are_attributed_to_their_field(pack):
    samples = [record(garment_category="top", waistline="none")] * K
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("auto:applies_to:waistline")},
        pack=pack, k=K,
    )
    assert report.by_field["waistline"] == {"violations": 1, "unanimous": 1}
    assert report.summary()["by_field"]["waistline"]["unanimous_share"] == 1.0


def test_cross_field_rules_are_not_attributed_to_one_field(pack):
    """`solid_is_not_multicolour` implicates pattern and colour together. Naming
    one would invent a precision the violation id does not carry."""
    samples = [record(pattern="solid", colour_primary="multicolour")] * K
    report = audit(
        [("sku-1", samples)],
        {"sku-1": failing("solid_is_not_multicolour")},
        pack=pack, k=K,
    )
    assert report.gate_failed == 1
    assert report.by_field == {}, "unattributable rules must stay unattributed"


def test_shares_are_zero_rather_than_undefined_on_an_empty_run(pack):
    report = audit([], {}, pack=pack, k=K)
    summary = report.summary()
    assert summary["invisible_share_of_gate_failures"] == 0.0
    assert summary["invisible_share_of_products"] == 0.0


def test_the_summary_reports_both_denominators(pack):
    """A share of gate failures and a share of all products answer different
    questions, and quoting only one invites the wrong reading."""
    unanimous = [record(garment_category="top", waistline="none")] * K
    report = audit(
        [("bad", unanimous), ("good", [record()] * K)],
        {"bad": failing("auto:applies_to:waistline"), "good": PASSING},
        pack=pack, k=K,
    )
    summary = report.summary()
    assert summary["invisible_share_of_gate_failures"] == 1.0
    assert summary["invisible_share_of_products"] == 0.5


# --- reading a real run -------------------------------------------------------


def test_load_run_refuses_a_missing_pass(tmp_path):
    (tmp_path / "pass-1.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path, K)


def test_load_run_folds_passes_by_sku_and_takes_verdicts_from_the_first(tmp_path):
    """The shipped record is pass 1. Verdicts folded across all five would
    describe a pipeline nobody runs."""
    for index in range(1, K + 1):
        rows = [{
            "sku_id": "a",
            "parsed": {"waistline": "none" if index == 1 else None},
            "gate_passed": index != 1,
            "rule_violations": ["auto:applies_to:waistline"] if index == 1 else [],
            "error": None,
        }]
        (tmp_path / f"pass-{index}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    per_product, gate = load_run(tmp_path, K)
    assert len(per_product) == 1 and len(per_product[0][1]) == K
    assert gate["a"]["gate_passed"] is False
    assert gate["a"]["rule_violations"] == ["auto:applies_to:waistline"]
