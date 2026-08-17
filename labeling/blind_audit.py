#!/usr/bin/env python3
"""Blind labelling, so an accuracy number can exist at all.

Every label in this project came from `gpt-5.6-luna`, and the one human pass
was run through `labeling.review`, whose CSV shows the reviewer
`proposed_value` and `proposed_status` before they answer. A reviewer shown the
model's answer mostly agrees with it, so those 78 cells measure how persuasive
the model is, not how right it is. `data/reliability.json` says as much by
carrying `usable: false`.

That is fine as a research position and fatal as a commercial one: the first
question any catalog owner asks is how accurate it is.

This module exports the same cells with the model's answer **removed**. The
reviewer sees the product and the permitted vocabulary and nothing else, and
answers from scratch. Scoring the result against the stored labels then
measures the labels, which is the quantity everything else is expressed
relative to.

Two things it deliberately does not do:

- **It does not sample only contested cells.** The existing 78 were the cells
  where two model families disagreed, which is why 72% is a lower bound chosen
  for difficulty rather than an estimate. A headline number needs a sample that
  is representative, so this stratifies across attributes and otherwise draws at
  random from a seed.
- **It does not let an abstention pass as agreement.** `unknown` and
  `not_applicable` are distinct answers here, and a human who writes one where
  the model wrote the other has disagreed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from labeling.records import AttributeLabel, LabelStatus, Row

VERSION = "blind-audit-v1"

# What the reviewer fills in. There is no proposed_* column on purpose; adding
# one back turns this file into `labeling.review` and re-creates the anchoring
# it exists to avoid.
EXPORT_COLUMNS = (
    "sku_id",
    "attribute",
    "title",
    "evidence",
    "permitted_values",
    "human_value",
    "human_status",
    "note",
)

STATUS_HELP = "labeled | unknown | not_applicable"


@dataclass
class AuditPlan:
    """Which cells were drawn, and by what rule, recorded so it is auditable."""

    seed: int
    per_attribute_floor: int
    total_cells: int
    attributes: dict[str, int] = field(default_factory=dict)
    source_sha256: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "seed": self.seed,
            "per_attribute_floor": self.per_attribute_floor,
            "total_cells": self.total_cells,
            "cells_per_attribute": dict(sorted(self.attributes.items())),
            "source_sha256": self.source_sha256,
            "blind": True,
            "note": (
                "The reviewer was not shown the stored label. Scoring these "
                "answers measures the labels, not the reviewer's agreement "
                "with them."
            ),
        }


def evidence_for(row: Row, limit: int = 400) -> str:
    """The listing text a human needs to answer, trimmed to stay readable.

    Title is carried separately, so this is description and tags. A reviewer
    working on a phone will not scroll 2,000 characters, and a cell they answer
    carelessly is worse than one they skip.
    """
    parts = [row.input.description or ""]
    if row.input.raw_tags:
        parts.append("Tags: " + ", ".join(row.input.raw_tags))
    text = " ".join(p.strip() for p in parts if p.strip())
    return text[:limit].strip()


def select_cells(
    rows: Sequence[Row],
    pack,
    *,
    seed: int = 20260817,
    per_attribute_floor: int = 12,
) -> tuple[list[tuple[str, str]], AuditPlan]:
    """Draw a representative sample of (sku_id, attribute) pairs.

    Stratified by attribute rather than drawn uniformly over cells, because a
    uniform draw would leave the rare attributes with one or two cells and no
    usable per-attribute reading. Every attribute gets the same floor; the
    resulting headline is therefore a macro average over attributes, which is
    the same shape as the macro-F1 the rest of the project reports.
    """
    rng = random.Random(seed)
    by_attribute: dict[str, list[str]] = {}
    for row in rows:
        for name in pack.field_names:
            if name in row.labels:
                by_attribute.setdefault(name, []).append(row.sku_id)

    plan = AuditPlan(seed=seed, per_attribute_floor=per_attribute_floor, total_cells=0)
    cells: list[tuple[str, str]] = []
    for name in sorted(by_attribute):
        pool = sorted(set(by_attribute[name]))
        take = min(per_attribute_floor, len(pool))
        drawn = rng.sample(pool, take)
        cells.extend((sku, name) for sku in drawn)
        plan.attributes[name] = take

    rng.shuffle(cells)  # so the reviewer does not answer one attribute in a block
    plan.total_cells = len(cells)
    return cells, plan


def export(
    rows: Sequence[Row],
    pack,
    out_path: Path,
    *,
    seed: int = 20260817,
    per_attribute_floor: int = 12,
) -> AuditPlan:
    """Write the blind CSV and return the plan that produced it."""
    index = {row.sku_id: row for row in rows}
    cells, plan = select_cells(
        rows, pack, seed=seed, per_attribute_floor=per_attribute_floor
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for sku_id, attribute in cells:
            row = index[sku_id]
            spec = pack.specs[attribute]
            writer.writerow(
                {
                    "sku_id": sku_id,
                    "attribute": attribute,
                    "title": row.input.title,
                    "evidence": evidence_for(row),
                    "permitted_values": " | ".join(spec.values),
                    "human_value": "",
                    "human_status": "",
                    "note": "",
                }
            )
    return plan


def _key(label: AttributeLabel) -> tuple:
    """Comparable identity, order-insensitive for multi-valued fields."""
    if isinstance(label.value, list):
        return (label.status.value, tuple(sorted(label.value)))
    return (label.status.value, label.value)


def _parse_human(value: str, status: str, *, multi: bool) -> AttributeLabel | None:
    status = (status or "").strip().lower()
    value = (value or "").strip()
    if not status and not value:
        return None  # not answered
    if status in ("unknown", "u"):
        return AttributeLabel(value=None, status=LabelStatus.UNKNOWN)
    if status in ("not_applicable", "na", "n/a"):
        return AttributeLabel(value=None, status=LabelStatus.NOT_APPLICABLE)
    if not value:
        return None
    parsed: Any = [v.strip() for v in value.split(",") if v.strip()] if multi else value
    return AttributeLabel(value=parsed, status=LabelStatus.LABELED)


def score(
    completed_csv: Path,
    rows: Sequence[Row],
    pack,
) -> dict[str, Any]:
    """Compare blind human answers against the stored labels.

    The headline is a macro average over attributes, matching how the rest of
    the project reports quality. A micro figure is also given, since the two
    diverge when attributes have uneven support and quoting only one hides that.
    """
    index = {row.sku_id: row for row in rows}
    per_attribute: dict[str, dict[str, int]] = {}
    unanswered = 0

    with completed_csv.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            attribute = record["attribute"]
            spec = pack.specs.get(attribute)
            if spec is None:
                continue
            human = _parse_human(
                record.get("human_value", ""),
                record.get("human_status", ""),
                multi=spec.kind == "multi",
            )
            if human is None:
                unanswered += 1
                continue
            row = index.get(record["sku_id"])
            if row is None or attribute not in row.labels:
                continue
            stats = per_attribute.setdefault(attribute, {"n": 0, "agree": 0})
            stats["n"] += 1
            if _key(human) == _key(row.labels[attribute]):
                stats["agree"] += 1

    scored = {
        name: {
            "n": s["n"],
            "agree": s["agree"],
            "accuracy": round(s["agree"] / s["n"], 4) if s["n"] else 0.0,
        }
        for name, s in sorted(per_attribute.items())
    }
    total_n = sum(s["n"] for s in per_attribute.values())
    total_agree = sum(s["agree"] for s in per_attribute.values())
    macro = (
        sum(v["accuracy"] for v in scored.values()) / len(scored) if scored else 0.0
    )

    return {
        "version": VERSION,
        "blind": True,
        "cells_scored": total_n,
        "cells_unanswered": unanswered,
        "label_accuracy_macro": round(macro, 4),
        "label_accuracy_micro": round(total_agree / total_n, 4) if total_n else 0.0,
        "per_attribute": scored,
        "interpretation": (
            "This is the accuracy of the stored labels against blind human "
            "judgement. Every model score in this project is agreement with "
            "those labels, so no model can be claimed to exceed this number."
        ),
    }
