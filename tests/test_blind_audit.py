"""Blind labelling, and the one property that makes it worth doing.

The existing review pass showed reviewers `proposed_value` before they answered,
so its 78 cells measure agreement with the model rather than the model's
correctness. If this export ever regains a column carrying the stored label,
it becomes that same instrument again and every number derived from it is
quietly wrong.

`test_the_export_cannot_leak_the_stored_label` is that guard, and it checks the
written bytes rather than the column list, because a leak would most likely
arrive inside `evidence` or `note` rather than as an honest new column.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from labeling.blind_audit import (
    EXPORT_COLUMNS,
    evidence_for,
    export,
    score,
    select_cells,
)
from labeling.records import read_jsonl
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval_300" / "eval-labels.jsonl"


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def rows():
    return read_jsonl(GOLD)


# --- the property the module exists for --------------------------------------


def test_the_export_cannot_leak_the_stored_label(rows, pack, tmp_path):
    """No column carries it, and no free-text field smuggles it either."""
    out = tmp_path / "blind.csv"
    export(rows, pack, out, per_attribute_floor=4)

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "proposed" not in header
    assert set(next(csv.reader([header]))) == set(EXPORT_COLUMNS)

    index = {row.sku_id: row for row in rows}
    for record in csv.DictReader(out.open(encoding="utf-8")):
        stored = index[record["sku_id"]].labels[record["attribute"]]
        if stored.status.value != "labeled" or not stored.value:
            continue
        values = stored.value if isinstance(stored.value, list) else [stored.value]
        blob = f"{record['human_value']} {record['human_status']} {record['note']}"
        for value in values:
            assert value not in blob, f"stored label leaked into {record['attribute']}"


def test_the_answer_columns_start_empty(rows, pack, tmp_path):
    out = tmp_path / "blind.csv"
    export(rows, pack, out, per_attribute_floor=3)
    for record in csv.DictReader(out.open(encoding="utf-8")):
        assert record["human_value"] == ""
        assert record["human_status"] == ""


# --- sampling -----------------------------------------------------------------


def test_every_attribute_reaches_the_floor(rows, pack):
    cells, plan = select_cells(rows, pack, per_attribute_floor=10)
    assert set(plan.attributes) == set(
        name for name in pack.field_names if name in rows[0].labels
    )
    assert all(count == 10 for count in plan.attributes.values())
    assert len(cells) == sum(plan.attributes.values())


def test_the_sample_is_reproducible_from_its_seed(rows, pack):
    a, _ = select_cells(rows, pack, seed=7, per_attribute_floor=5)
    b, _ = select_cells(rows, pack, seed=7, per_attribute_floor=5)
    c, _ = select_cells(rows, pack, seed=8, per_attribute_floor=5)
    assert a == b
    assert a != c, "different seeds must draw different cells"


def test_cells_are_shuffled_rather_than_grouped_by_attribute(rows, pack):
    """A reviewer answering forty `fit` cells in a row anchors on their own
    previous answers, which is the same failure in a different coat."""
    cells, _ = select_cells(rows, pack, seed=3, per_attribute_floor=10)
    attributes = [attribute for _sku, attribute in cells]
    runs = sum(1 for i in range(1, len(attributes)) if attributes[i] == attributes[i - 1])
    assert runs < len(attributes) * 0.25


def test_evidence_is_trimmed_so_a_reviewer_will_actually_read_it(rows, pack):
    assert all(len(evidence_for(row)) <= 400 for row in rows)


# --- scoring ------------------------------------------------------------------


def write_answers(path: Path, answers: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for answer in answers:
            writer.writerow({column: answer.get(column, "") for column in EXPORT_COLUMNS})


def test_agreement_and_disagreement_are_counted(rows, pack, tmp_path):
    row = next(r for r in rows if r.labels["garment_category"].status.value == "labeled")
    stored = row.labels["garment_category"].value

    path = tmp_path / "done.csv"
    write_answers(path, [
        {"sku_id": row.sku_id, "attribute": "garment_category",
         "human_value": stored, "human_status": "labeled"},
        {"sku_id": row.sku_id, "attribute": "garment_category",
         "human_value": "definitely-not-this", "human_status": "labeled"},
    ])
    result = score(path, rows, pack)
    assert result["per_attribute"]["garment_category"] == {
        "n": 2, "agree": 1, "accuracy": 0.5
    }


def test_an_abstention_is_not_agreement_with_a_value(rows, pack, tmp_path):
    """`unknown` and `not_applicable` are answers. A human writing one where the
    label carries a value has disagreed, and collapsing that would inflate the
    number this whole module exists to produce."""
    row = next(r for r in rows if r.labels["garment_category"].status.value == "labeled")
    path = tmp_path / "done.csv"
    write_answers(path, [
        {"sku_id": row.sku_id, "attribute": "garment_category", "human_status": "unknown"},
        {"sku_id": row.sku_id, "attribute": "garment_category",
         "human_status": "not_applicable"},
    ])
    result = score(path, rows, pack)
    assert result["per_attribute"]["garment_category"]["agree"] == 0


def test_unanswered_cells_are_skipped_not_counted_wrong(rows, pack, tmp_path):
    row = rows[0]
    path = tmp_path / "done.csv"
    write_answers(path, [{"sku_id": row.sku_id, "attribute": "garment_category"}])
    result = score(path, rows, pack)
    assert result["cells_scored"] == 0
    assert result["cells_unanswered"] == 1


def test_multi_valued_answers_ignore_order(rows, pack, tmp_path):
    row = next(
        (r for r in rows
         if r.labels.get("details") and isinstance(r.labels["details"].value, list)
         and len(r.labels["details"].value) > 1),
        None,
    )
    if row is None:
        pytest.skip("no multi-valued details cell in the gold set")
    reversed_values = ", ".join(reversed(row.labels["details"].value))
    path = tmp_path / "done.csv"
    write_answers(path, [{"sku_id": row.sku_id, "attribute": "details",
                          "human_value": reversed_values, "human_status": "labeled"}])
    assert score(path, rows, pack)["per_attribute"]["details"]["agree"] == 1


def test_the_result_states_what_it_bounds(rows, pack, tmp_path):
    """The number is only useful if a reader knows it caps every model claim."""
    path = tmp_path / "done.csv"
    write_answers(path, [])
    result = score(path, rows, pack)
    assert result["blind"] is True
    assert "no model" in result["interpretation"].lower()
