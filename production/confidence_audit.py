#!/usr/bin/env python3
"""What the confidence signal cannot see.

The escalation rate in the run report answers "how many cells does a human have
to look at?". Read on its own it invites a wrong inference: that the cells left
unreviewed are the cells that are right. They are not. Self-consistency measures
whether the model gives the same answer k times, and a model can be perfectly
consistent about a convention it has consistently misunderstood.

This module measures the size of that blind spot directly, by crossing the two
signals the pipeline already produces:

- **agreement**, from `production.escalation` (same `labeling.consensus` code the
  W1 review round used), and
- **the gate**, from the verifier, which is the same code the RL reward calls.

The number it exists to produce is `gate_failed_and_unanimous`: records the
verifier rejects while every one of their cells is unanimous across all k
samples. No confidence threshold surfaces those, including threshold 1.0, which
is already the most aggressive setting available. They are the records that
would ship wrong and unreviewed.

This is not a repair path. Nothing here changes an answer; it only counts what
one signal misses and the other catches. The fix, if one is wanted, belongs
upstream in the prompt or the schema, and is out of W3's scope.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from labeling.consensus import consensus
from production.escalation import samples_to_labels
from verifier import Pack

VERSION = "production-confidence-audit-v1"

# `auto:applies_to:<field>` is the one violation id that names the single field
# responsible for it, so it is the only class this module attributes to a cell.
# Cross-field rules (a solid pattern that is also multicolour) implicate two
# fields and attributing them to one would invent a precision the id does not
# carry; they are counted in the product-level total and left unattributed.
APPLIES_TO = re.compile(r"^auto:applies_to:(?P<field>.+)$")


@dataclass
class AuditReport:
    k: int
    products: int = 0
    gate_failed: int = 0
    gate_failed_and_unanimous: int = 0
    # Per attribute, for the attributable rule class only.
    by_field: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def invisible_share(self) -> float:
        """Share of gate failures no threshold can surface."""
        return (
            self.gate_failed_and_unanimous / self.gate_failed if self.gate_failed else 0.0
        )

    def summary(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "k": self.k,
            "products": self.products,
            "gate_failed": self.gate_failed,
            "gate_failed_and_unanimous": self.gate_failed_and_unanimous,
            "invisible_share_of_gate_failures": round(self.invisible_share, 4),
            "invisible_share_of_products": (
                round(self.gate_failed_and_unanimous / self.products, 4)
                if self.products
                else 0.0
            ),
            "by_field": {
                name: {
                    **stats,
                    "unanimous_share": (
                        round(stats["unanimous"] / stats["violations"], 4)
                        if stats["violations"]
                        else 0.0
                    ),
                }
                for name, stats in sorted(
                    self.by_field.items(), key=lambda kv: -kv[1]["violations"]
                )
            },
        }


def audit(
    per_product_samples: Iterable[tuple[str, Sequence[dict[str, Any]]]],
    gate: dict[str, dict[str, Any]],
    *,
    pack: Pack,
    k: int,
) -> AuditReport:
    """Cross agreement against the gate.

    `gate` maps sku_id to the shipped record's verdict, with keys `gate_passed`
    and `rule_violations`. It is the **first pass** rather than a consensus of
    all k, because the first pass is the record production actually emits; a
    verdict computed from a fold of all five would describe a pipeline nobody
    runs.
    """
    report = AuditReport(k=k)

    for sku_id, samples in per_product_samples:
        usable = [s for s in samples if isinstance(s, dict)]
        verdict = gate.get(sku_id)
        # A product with no verdict was never scored; counting it as passing or
        # failing would both be inventions.
        if not usable or verdict is None:
            continue
        if len(usable) < k:
            # Fewer than k usable samples means agreement is measured over a
            # smaller denominator than the escalation rate assumes. Skipping
            # keeps the two numbers comparable.
            continue

        agreed = consensus(samples_to_labels(usable, pack))
        if not agreed:
            continue

        report.products += 1
        violations = list(verdict.get("rule_violations") or [])
        if verdict.get("gate_passed"):
            continue

        report.gate_failed += 1
        if all(cell.unanimous for cell in agreed.values()):
            report.gate_failed_and_unanimous += 1

        for violation in violations:
            match = APPLIES_TO.match(violation)
            if match is None:
                continue
            name = match.group("field")
            stats = report.by_field.setdefault(name, {"violations": 0, "unanimous": 0})
            stats["violations"] += 1
            cell = agreed.get(name)
            if cell is not None and cell.unanimous:
                stats["unanimous"] += 1

    return report


def load_run(out_dir: Path, k: int) -> tuple[list[tuple[str, list[dict]]], dict[str, dict]]:
    """Rebuild samples and first-pass verdicts from a completed run directory."""
    passes = []
    for index in range(1, k + 1):
        path = out_dir / f"pass-{index}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        passes.append(
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        )

    by_sku: dict[str, list[dict]] = {}
    for rows in passes:
        for row in rows:
            if row.get("parsed") is not None:
                by_sku.setdefault(row["sku_id"], []).append(row["parsed"])

    gate = {
        row["sku_id"]: {
            "gate_passed": row["gate_passed"],
            "rule_violations": row["rule_violations"],
        }
        for row in passes[0]
        if row.get("error") is None
    }
    return list(by_sku.items()), gate


def main() -> None:
    import argparse

    from verifier import load_pack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/production-demo"))
    parser.add_argument("--pack", type=Path, default=Path("packs/vastraa_taste_v1"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    pack = load_pack(str(args.pack))
    per_product, gate = load_run(args.run_dir, args.k)
    report = audit(per_product, gate, pack=pack, k=args.k)
    summary = report.summary()

    out = args.run_dir / "confidence-audit.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
