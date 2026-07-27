"""Loading predictions, and grading their shape on the way in.

Schema validity is only measurable from **raw model output**. A prediction file of
pre-parsed dicts has already had its JSON errors thrown away, so validity is
trivially 100% and the number is a lie. W2's entire story is watching that number
climb as the model learns to emit clean JSON, so the loader keeps raw text when it
is offered and reports `None` — not 1.0 — when it is not.

Accepted shapes, one JSON object per line:

    {"sku_id": "...", "raw": "<the model's literal output>"}   <- preferred
    {"sku_id": "...", "prediction": {...}}                     <- pre-parsed
    {"sku_id": "...", "garment_category": "dress", ...}        <- bare record
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeling.records import LabelStatus, Row
from verifier import verify, verify_record


@dataclass
class LoadedPredictions:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_valid: int | None = None
    vocab_valid: int = 0
    rule_histogram: Counter = field(default_factory=Counter)
    unparseable: int = 0
    raw_mode: bool = False
    # Every line the model produced, including the ones too malformed to keep.
    # Validity must be scored against attempts, not survivors — dividing by the
    # survivors reported "100% valid, 20 unparseable" in the first real run.
    n_attempted: int = 0


def load(path: str | Path, pack) -> LoadedPredictions:
    out = LoadedPredictions()
    schema_ok = 0
    saw_raw = False

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.n_attempted += 1
        sku = obj.get("sku_id") or obj.get("sku") or obj.get("id")
        if sku is None:
            raise ValueError(f"prediction line has no sku_id: {line[:120]}")

        if "raw" in obj:
            saw_raw = True
            result = verify(obj["raw"], pack)
            if result.schema_valid:
                schema_ok += 1
            if result.parsed is None:
                out.unparseable += 1
                # An unparseable prediction is absent, not empty. Recording it as
                # an empty record would silently convert a format failure into a
                # full sweep of wrong answers and flatter the format metric.
                continue
            record = result.parsed
        else:
            record = obj.get("prediction") or {
                k: v for k, v in obj.items() if k in pack.specs
            }
            result = verify_record(record, pack)

        if result.vocab_valid:
            out.vocab_valid += 1
        for rule_id in result.rule_violations:
            out.rule_histogram[rule_id] += 1
        out.records[sku] = record

    out.raw_mode = saw_raw
    out.schema_valid = schema_ok if saw_raw else None
    return out


def from_frontier(gold: list[Row], pack) -> LoadedPredictions:
    """Build the ceiling run from the snapshots Step 3 already stored.

    The plan says to run the harness once against the frontier model's own labels
    to get the ceiling. No API call is needed: `provenance.frontier_labels` holds
    exactly that, captured before any human touched the row.

    Note what this measures. Cells the reviewer never opened (the consensus pass
    skipped them as unanimous) have gold == frontier by construction, so the
    ceiling is optimistic on those. That is a property of the sampling shortcut,
    not a bug — but it is why the number is a ceiling rather than a score.
    """
    out = LoadedPredictions(raw_mode=False)
    for row in gold:
        snapshot = row.provenance.frontier_labels
        if snapshot is None:
            continue
        record: dict[str, Any] = {}
        for name, label in snapshot.items():
            spec = pack.specs.get(name)
            if label.status is LabelStatus.LABELED:
                record[name] = label.value
            elif label.status is LabelStatus.NOT_APPLICABLE:
                record[name] = None
            else:
                multi = spec is not None and spec.kind == "multi"
                record[name] = [pack.unknown_token] if multi else pack.unknown_token
        result = verify_record(record, pack)
        if result.vocab_valid:
            out.vocab_valid += 1
        for rule_id in result.rule_violations:
            out.rule_histogram[rule_id] += 1
        out.records[row.sku_id] = record
    return out
