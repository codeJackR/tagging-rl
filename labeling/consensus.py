"""Self-consistency: sample the labeler k times, route disagreement to a human.

Built once, used twice — this is the same machinery W3 Step 3 needs for the
production escalation queue. Here the "reviewer" is you clearing the 300; there
it is the review table. Nothing in this module knows which.

Why it exists here
------------------
300 rows x 15 attributes = 4,500 cells. At 3 seconds a cell that is nearly four
hours of clicking, and the 6-hour timebox is gone. But cells where k independent
samples all agree are very likely already correct — reviewing them buys almost
nothing. Reviewing only the disagreements typically cuts the queue by 60-80%.

The by-products are worth as much as the saving:
  - per-cell agreement becomes the label-confidence column on the weak train set,
    so noisy rows can be downweighted in the W2 reward
  - per-attribute disagreement rate is an early preview of the reliability table
    in `reliability.py`, available before any hand-correction

Where the k samples come from
-----------------------------
Deliberately not this module's concern — it takes k already-produced label dicts.
That keeps it usable by the W1 batch labeler and the W3 online path alike.

One caveat that shaped the caller: `temperature` is REMOVED on Claude Opus 5 and
Sonnet 5 (the API returns 400). "Sample k=5 at temperature" is not available on
the frontier models. `scripts/prelabel.py` gets its diversity from prompt
perturbation instead — see the note there.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .records import AttributeLabel, LabelStatus, Row


@dataclass(frozen=True)
class CellConsensus:
    attribute: str
    label: AttributeLabel  # the modal sample
    agreement: float  # fraction of samples matching it, 0..1
    n_samples: int
    n_variants: int  # how many distinct answers appeared

    @property
    def unanimous(self) -> bool:
        return self.agreement >= 1.0


@dataclass
class ReviewCell:
    """One (row, attribute) pair a human should look at."""

    sku_id: str
    attribute: str
    proposed: AttributeLabel
    agreement: float
    alternatives: list[AttributeLabel] = field(default_factory=list)


def consensus(samples: list[dict[str, AttributeLabel]]) -> dict[str, CellConsensus]:
    """Per-attribute modal label plus the agreement fraction behind it.

    Multi-valued fields compare order-insensitively (see `AttributeLabel.key`), so
    ["lined","pleated"] and ["pleated","lined"] count as agreement, not conflict.
    """
    if not samples:
        raise ValueError("no samples")

    attributes: list[str] = []
    for s in samples:
        for name in s:
            if name not in attributes:
                attributes.append(name)

    out: dict[str, CellConsensus] = {}
    for name in attributes:
        present = [s[name] for s in samples if name in s]
        if not present:
            continue
        counts = Counter(lab.key() for lab in present)
        winning_key, n_winning = counts.most_common(1)[0]
        modal = next(lab for lab in present if lab.key() == winning_key)
        out[name] = CellConsensus(
            attribute=name,
            label=modal,
            agreement=n_winning / len(present),
            n_samples=len(present),
            n_variants=len(counts),
        )
    return out


def consensus_labels(
    samples: list[dict[str, AttributeLabel]],
) -> tuple[dict[str, AttributeLabel], dict[str, float]]:
    """Convenience: (modal labels, per-attribute agreement) — the shape a Row wants."""
    result = consensus(samples)
    return (
        {name: c.label for name, c in result.items()},
        {name: round(c.agreement, 4) for name, c in result.items()},
    )


def review_cells(
    row: Row,
    samples: list[dict[str, AttributeLabel]] | None = None,
    *,
    threshold: float = 1.0,
    always_review: tuple[str, ...] = (),
) -> list[ReviewCell]:
    """Cells whose agreement is below `threshold` — the human queue for this row.

    `threshold=1.0` (default) means "review anything not unanimous". Loosen it to
    0.8 to review only real splits once you trust the labeler on this pack.

    `always_review` forces attributes into the queue regardless of agreement. Use
    it for fields the reliability table has shown the labeler is confidently wrong
    about — unanimity across k samples of one biased model is agreement, not
    correctness, and that is exactly the blind spot this whole step exists to close.
    """
    if samples is None:
        agree = (
            row.provenance.self_consistency.agreement
            if row.provenance.self_consistency
            else {}
        )
        return [
            ReviewCell(row.sku_id, name, lab, agree.get(name, 0.0))
            for name, lab in row.labels.items()
            if agree.get(name, 0.0) < threshold or name in always_review
        ]

    result = consensus(samples)
    queue: list[ReviewCell] = []
    for name, c in result.items():
        if c.agreement >= threshold and name not in always_review:
            continue
        alts = []
        seen = {c.label.key()}
        for s in samples:
            lab = s.get(name)
            if lab is not None and lab.key() not in seen:
                seen.add(lab.key())
                alts.append(lab)
        queue.append(ReviewCell(row.sku_id, name, c.label, c.agreement, alts))
    return queue


def queue_savings(rows: list[Row], *, threshold: float = 1.0) -> dict:
    """How much hand-review the consensus pass actually saved. Report this.

    A silent cap reads as 'we reviewed everything' when it isn't — so the number
    of skipped cells belongs in the run log, not just the reviewed ones.
    """
    total = flagged = 0
    for row in rows:
        agree = (
            row.provenance.self_consistency.agreement
            if row.provenance.self_consistency
            else {}
        )
        for name in row.labels:
            total += 1
            if agree.get(name, 0.0) < threshold:
                flagged += 1
    return {
        "cells_total": total,
        "cells_to_review": flagged,
        "cells_skipped": total - flagged,
        "reduction": round(1 - flagged / total, 4) if total else 0.0,
        "threshold": threshold,
    }


def attribute_disagreement(rows: list[Row]) -> dict[str, dict]:
    """Per-attribute disagreement rate — an early preview of the reliability table.

    Available before any hand-correction. It measures the labeler's *instability*,
    not its accuracy: an attribute the model is confidently and consistently wrong
    about scores 1.0 here. Treat a low score as a definite problem and a high score
    as no evidence either way.
    """
    stats: dict[str, list[float]] = {}
    for row in rows:
        sc = row.provenance.self_consistency
        if not sc:
            continue
        for name, agree in sc.agreement.items():
            stats.setdefault(name, []).append(agree)

    out = {}
    for name, vals in sorted(stats.items()):
        n = len(vals)
        out[name] = {
            "n": n,
            "mean_agreement": round(sum(vals) / n, 4),
            "unstable_rate": round(sum(1 for v in vals if v < 1.0) / n, 4),
        }
    return out


def escalation_rate(rows: list[Row], *, threshold: float = 1.0) -> float:
    """Fraction of ROWS with at least one below-threshold cell.

    W3 Step 3's done-when is "escalation % is a number you can quote". This is
    that number, computed by the same code path that builds the W1 review queue.

    Quote it next to the cell-level figure from `queue_savings`, never alone. With
    15 attributes the two diverge hard — ~14% of cells uncertain can still mean
    ~92% of rows containing at least one uncertain cell. Row-level is the honest
    number for "how many SKUs need a human to open them"; cell-level is the honest
    number for "how much work that human actually does". Reporting only the
    flattering one is the kind of thing an interviewer checks.
    """
    if not rows:
        return 0.0
    escalated = sum(1 for r in rows if review_cells(r, threshold=threshold))
    return round(escalated / len(rows), 4)


def unlabeled_share(rows: list[Row]) -> dict[str, dict]:
    """Per-attribute split of labeled / not_applicable / unknown.

    Sanity check on the three-state distinction: an attribute that is ~100%
    `unknown` is not extractable from this feed's text and should be reconsidered
    before it costs 300 hand-corrections.
    """
    counts: dict[str, Counter] = {}
    for row in rows:
        for name, lab in row.labels.items():
            counts.setdefault(name, Counter())[lab.status.value] += 1
    return {
        name: {
            "n": sum(c.values()),
            **{s.value: c.get(s.value, 0) for s in LabelStatus},
            "labeled_rate": round(c.get("labeled", 0) / sum(c.values()), 4),
        }
        for name, c in sorted(counts.items())
    }
