"""Spreadsheet round-trip for the hand-correction pass.

Exports one line per *cell that needs a human*, not per row — that is the whole
saving. Blank `corrected_*` means "the proposal is right"; fill them in only where
it is wrong. Re-importing snapshots the frontier's original answer before applying
anything, because the reliability table cannot be rebuilt from corrected data.

`vocab_reference.csv` is written alongside so the review sheet can bind a
data-validation dropdown per attribute. Typing free text into a controlled-vocab
column is the fastest way to poison the one dataset that has to be trustworthy.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .consensus import review_cells
from .records import AttributeLabel, LabelStatus, Row

COLUMNS = [
    "sku_id",
    "attribute",
    "title",
    "evidence",
    "proposed_value",
    "proposed_status",
    "agreement",
    "alternatives",
    "corrected_value",  # <- you fill these two, only where the proposal is wrong
    "corrected_status",
    "note",
]


def export_review_csv(
    rows: list[Row],
    path: str | Path,
    *,
    threshold: float = 1.0,
    always_review: tuple[str, ...] = (),
    evidence_chars: int = 240,
) -> dict:
    """Write the review queue. Returns a summary you should log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_sku = {r.sku_id: r for r in rows}
    written = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            for cell in review_cells(
                row, threshold=threshold, always_review=always_review
            ):
                src = by_sku[cell.sku_id]
                w.writerow(
                    {
                        "sku_id": cell.sku_id,
                        "attribute": cell.attribute,
                        "title": src.input.title,
                        "evidence": (src.input.description or "")[:evidence_chars],
                        "proposed_value": _flat(cell.proposed),
                        "proposed_status": cell.proposed.status.value,
                        "agreement": f"{cell.agreement:.2f}",
                        "alternatives": " | ".join(_flat(a) for a in cell.alternatives),
                        "corrected_value": "",
                        "corrected_status": "",
                        "note": "",
                    }
                )
                written += 1

    total_cells = sum(len(r.labels) for r in rows)
    return {
        "path": str(path),
        "cells_to_review": written,
        "cells_total": total_cells,
        "reduction": round(1 - written / total_cells, 4) if total_cells else 0.0,
        "rows": len(rows),
    }


def export_vocab_reference(pack, path: str | Path) -> str:
    """One column per attribute, its allowed values below — for sheet dropdowns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(pack.specs)
    cols = [
        [n] + list(pack.specs[n].values) + [pack.unknown_token] for n in names
    ]
    height = max(len(c) for c in cols)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for i in range(height):
            w.writerow([c[i] if i < len(c) else "" for c in cols])
    return str(path)


def _flat(label: AttributeLabel) -> str:
    if label.status is not LabelStatus.LABELED:
        return ""
    if isinstance(label.value, list):
        return "|".join(label.value)
    return str(label.value or "")


def _parse(value: str, status: str, *, multi: bool) -> AttributeLabel:
    status = (status or "").strip().lower()
    value = (value or "").strip()
    if status in ("not_applicable", "na", "n/a"):
        return AttributeLabel(status=LabelStatus.NOT_APPLICABLE)
    if status == "unknown":
        return AttributeLabel(status=LabelStatus.UNKNOWN)
    if not value:
        raise ValueError("a 'labeled' correction needs a value")
    parsed = [v.strip() for v in value.split("|") if v.strip()] if multi else value
    return AttributeLabel(value=parsed, status=LabelStatus.LABELED)


def import_review_csv(
    rows: list[Row], path: str | Path, pack=None, *, when: date | str | None = None
) -> dict:
    """Apply corrections. Snapshots the frontier labels first, then edits in place.

    A blank `corrected_value` *and* `corrected_status` means the proposal stands —
    the row is still marked human_corrected, because a human did look at it and
    that agreement is exactly what the reliability table needs to count.
    """
    by_sku = {r.sku_id: r for r in rows}
    when = str(when or date.today())
    changed = accepted = 0
    touched: set[str] = set()
    errors: list[str] = []

    with Path(path).open(newline="", encoding="utf-8") as fh:
        for i, rec in enumerate(csv.DictReader(fh), start=2):
            sku, attr = rec.get("sku_id", ""), rec.get("attribute", "")
            row = by_sku.get(sku)
            if row is None:
                errors.append(f"line {i}: unknown sku_id {sku!r}")
                continue
            if attr not in row.labels:
                errors.append(f"line {i}: {sku} has no attribute {attr!r}")
                continue

            # Snapshot before the first edit to this row, never after.
            if row.provenance.frontier_labels is None:
                row.provenance.frontier_labels = {
                    k: v.model_copy(deep=True) for k, v in row.labels.items()
                }
            touched.add(sku)

            val, stat = rec.get("corrected_value", ""), rec.get("corrected_status", "")
            if not val.strip() and not stat.strip():
                accepted += 1
                continue

            multi = bool(pack and pack.specs.get(attr) and pack.specs[attr].kind == "multi")
            try:
                new = _parse(val, stat or "labeled", multi=multi)
            except ValueError as exc:
                errors.append(f"line {i}: {sku}/{attr}: {exc}")
                continue

            if pack is not None and new.status is LabelStatus.LABELED:
                allowed = set(pack.specs[attr].values)
                vals = new.value if isinstance(new.value, list) else [new.value]
                bad = [v for v in vals if v not in allowed]
                if bad:
                    errors.append(
                        f"line {i}: {sku}/{attr}: {bad} not in the controlled vocabulary"
                    )
                    continue

            if row.labels[attr].key() != new.key():
                changed += 1
            row.labels[attr] = new

    for sku in touched:
        by_sku[sku].mark_corrected(when)

    return {
        "rows_touched": len(touched),
        "cells_changed": changed,
        "cells_accepted": accepted,
        "errors": errors,
    }
