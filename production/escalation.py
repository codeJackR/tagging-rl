#!/usr/bin/env python3
"""Confidence from self-consistency, and the escalation queue it feeds.

The plan's revision 2 is explicit about why confidence does not come from
logprobs: generative models have no well-calibrated confidence over structured
outputs, and token probabilities do not reliably track whether a value is right.
So confidence is measured behaviourally instead. Ask the same question k times
and see whether the answers agree.

The number this module exists to produce is the **escalation rate**: the share
of cells a human must look at. That is the unit economics of the whole product.
Accuracy without it is meaningless, because a pipeline can reach any accuracy
you like by escalating everything, and a pipeline that escalates nothing is only
as good as its worst cell.

The consensus mathematics is not reimplemented here. `labeling.consensus`
already computes modal labels, per-attribute agreement and review queues, and
was used for the W1 labelling round; this module samples the production path k
times and hands the results to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from labeling.consensus import CellConsensus, consensus
from labeling.records import AttributeLabel, LabelStatus
from verifier import Pack

VERSION = "production-escalation-v1"

# k=5 follows the plan. It is the smallest odd count that can distinguish
# unanimous (5/5) from strong (4/5) from split (3/5), and cost is linear in k.
DEFAULT_K = 5


def label_from_value(value: Any, pack: Pack) -> AttributeLabel:
    """Map one raw field value onto the three-state label the consensus uses.

    The distinction that matters is `null` versus the unknown token. `null`
    means the field cannot apply to this kind of product; unknown means it could
    apply and the listing did not say. Collapsing them would make an abstention
    look like an inapplicability and quietly remove it from the escalation
    queue, which is the one place a human most needs to see it.
    """
    if value is None:
        return AttributeLabel(value=None, status=LabelStatus.NOT_APPLICABLE)
    if value == pack.unknown_token or value == [pack.unknown_token]:
        return AttributeLabel(value=None, status=LabelStatus.UNKNOWN)
    return AttributeLabel(value=value, status=LabelStatus.LABELED)


def samples_to_labels(
    parsed_samples: Sequence[dict[str, Any]], pack: Pack
) -> list[dict[str, AttributeLabel]]:
    """Convert k parsed records into the shape `labeling.consensus` expects."""
    return [
        {
            name: label_from_value(record.get(name), pack)
            for name in pack.field_names
            if name in record
        }
        for record in parsed_samples
    ]


@dataclass
class EscalatedCell:
    sku_id: str
    attribute: str
    agreement: float
    n_variants: int
    proposed: Any
    proposed_status: str


@dataclass
class ProductConfidence:
    sku_id: str
    cells: int
    escalated: int
    mean_agreement: float
    unanimous_cells: int


@dataclass
class EscalationReport:
    threshold: float
    k: int
    products: int = 0
    cells: int = 0
    escalated_cells: int = 0
    products_with_any_escalation: int = 0
    per_attribute: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def escalation_rate(self) -> float:
        return self.escalated_cells / self.cells if self.cells else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "k": self.k,
            "threshold": self.threshold,
            "products": self.products,
            "cells": self.cells,
            "escalated_cells": self.escalated_cells,
            "escalation_rate": round(self.escalation_rate, 4),
            "products_with_any_escalation": self.products_with_any_escalation,
            "product_touch_rate": round(
                self.products_with_any_escalation / self.products, 4
            )
            if self.products
            else 0.0,
            # Per attribute as well as overall, because one bad attribute can
            # dominate the headline and hide that the rest are fine.
            "per_attribute": {
                name: {
                    "escalation_rate": round(stats["escalated"] / stats["cells"], 4),
                    "escalated": int(stats["escalated"]),
                    "cells": int(stats["cells"]),
                }
                for name, stats in sorted(
                    self.per_attribute.items(),
                    key=lambda kv: -kv[1]["escalated"] / max(kv[1]["cells"], 1),
                )
            },
        }


def escalate(
    per_product_samples: Iterable[tuple[str, Sequence[dict[str, Any]]]],
    *,
    pack: Pack,
    threshold: float,
    k: int = DEFAULT_K,
) -> tuple[list[EscalatedCell], list[ProductConfidence], EscalationReport]:
    """Route low-agreement cells to review.

    A cell is escalated when the fraction of samples agreeing on the modal
    answer falls **below** the threshold. At threshold 1.0 anything short of
    unanimity is escalated, which is the most conservative setting and the one
    W1 used for its review queue.
    """
    escalated: list[EscalatedCell] = []
    confidences: list[ProductConfidence] = []
    report = EscalationReport(threshold=threshold, k=k)

    for sku_id, samples in per_product_samples:
        usable = [s for s in samples if isinstance(s, dict)]
        if not usable:
            # A product whose every sample failed cannot be scored for
            # agreement. It is not "confident", it is unmeasured, and counting
            # it as either would corrupt the rate.
            continue
        agreed: dict[str, CellConsensus] = consensus(samples_to_labels(usable, pack))
        if not agreed:
            continue

        product_escalated = 0
        agreements = []
        unanimous = 0
        for name, cell in agreed.items():
            report.cells += 1
            stats = report.per_attribute.setdefault(name, {"cells": 0, "escalated": 0})
            stats["cells"] += 1
            agreements.append(cell.agreement)
            if cell.unanimous:
                unanimous += 1
            if cell.agreement < threshold:
                report.escalated_cells += 1
                stats["escalated"] += 1
                product_escalated += 1
                escalated.append(
                    EscalatedCell(
                        sku_id=sku_id,
                        attribute=name,
                        agreement=round(cell.agreement, 4),
                        n_variants=cell.n_variants,
                        proposed=cell.label.value,
                        proposed_status=cell.label.status.value,
                    )
                )

        report.products += 1
        if product_escalated:
            report.products_with_any_escalation += 1
        confidences.append(
            ProductConfidence(
                sku_id=sku_id,
                cells=len(agreed),
                escalated=product_escalated,
                mean_agreement=round(sum(agreements) / len(agreements), 4),
                unanimous_cells=unanimous,
            )
        )

    return escalated, confidences, report


def threshold_curve(
    per_product_samples: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    pack: Pack,
    k: int = DEFAULT_K,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Escalation rate across candidate thresholds.

    Reported as a curve rather than a single number because the threshold is a
    business decision, not a measurement. One point hides the trade-off it was
    chosen from; the curve makes the cost of each choice visible.
    """
    if thresholds is None:
        # The achievable agreements with k samples are the multiples of 1/k, so
        # any threshold between two of them behaves identically. Testing values
        # that cannot differ would pad the curve with false resolution.
        thresholds = [round((i + 1) / k, 4) for i in range(k)]
    curve = []
    for threshold in thresholds:
        _cells, _conf, report = escalate(
            per_product_samples, pack=pack, threshold=threshold, k=k
        )
        curve.append(
            {
                "threshold": threshold,
                "escalation_rate": round(report.escalation_rate, 4),
                "escalated_cells": report.escalated_cells,
                "cells": report.cells,
                "product_touch_rate": round(
                    report.products_with_any_escalation / report.products, 4
                )
                if report.products
                else 0.0,
            }
        )
    return curve


def write_queue(path, cells: Sequence[EscalatedCell]) -> None:
    """Write the review queue as CSV, the same round-trip W1 used.

    A CSV opens in a spreadsheet on a phone, which is how the W1 review round
    was actually completed. A review interface was not built then and is not
    built now.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sku_id", "attribute", "agreement", "n_variants", "proposed", "status"]
        )
        for cell in sorted(cells, key=lambda c: (c.agreement, c.sku_id, c.attribute)):
            writer.writerow(
                [
                    cell.sku_id,
                    cell.attribute,
                    cell.agreement,
                    cell.n_variants,
                    json.dumps(cell.proposed) if isinstance(cell.proposed, list) else cell.proposed,
                    cell.proposed_status,
                ]
            )
