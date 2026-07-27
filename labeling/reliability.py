"""Per-attribute frontier reliability — which attributes are safe to reward.

This is the highest-value artifact of W1 Step 3, and it is not in the plan's
"done when". Build it anyway.

The circularity it breaks
-------------------------
The eval set and the training labels both start from the same frontier model. If
that model is systematically wrong about fabric, then:

  - the training labels teach the wrong answer, and
  - the eval set, seeded the same way, agrees that the wrong answer is right.

GRPO will faithfully optimize toward the labeler's bias, and the scoreboard will
say everything is fine. In RLVR the labels *are* the reward function, so label
noise is not an accuracy problem — it is a reward-specification problem.

Hand-correcting the 300 is what breaks the loop. Diffing the corrections against
the frontier's original output is what tells you *where* the loop was broken, per
attribute. It costs nothing extra: you are correcting those rows anyway.

Reading the output
------------------
An attribute below ~85% should be excluded from the reward or heavily
downweighted. Otherwise you spend GPU hours teaching the model to reproduce a
labeling bias that your eval cannot see.

Verdicts key off the **Wilson lower bound**, not the point estimate. With 300 rows
and a conditional attribute, some columns have only 20-30 comparisons — a 90%
point estimate on n=20 is consistent with a true 70%, and calling that "safe"
would reintroduce exactly the false confidence this table exists to remove.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .records import LabelStatus, Row

Z95 = 1.959963985

SAFE = 0.95
USABLE = 0.85
MIN_COMPARISONS = 20


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval — well-behaved at small n and at p near 0 or 1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class AttributeReliability:
    attribute: str
    n_compared: int
    n_agree: int
    accuracy: float
    ci_low: float
    ci_high: float
    verdict: str
    # Where the disagreements live. `status_confusion` counts frontier-status ->
    # corrected-status flips; the not_applicable/unknown cell is the specific
    # noise the three-state schema exists to catch.
    status_confusion: dict[str, int]
    top_errors: list[tuple[str, int]]

    @property
    def reward_weight(self) -> float:
        """Suggested multiplier for this attribute in the W2 reward. 0 = exclude."""
        if self.verdict == "exclude":
            return 0.0
        if self.verdict == "downweight":
            return 0.5
        if self.verdict == "insufficient_data":
            return 0.0
        return 1.0


def _verdict(ci_low: float, n: int) -> str:
    if n < MIN_COMPARISONS:
        return "insufficient_data"
    if ci_low >= SAFE:
        return "safe"
    if ci_low >= USABLE:
        return "usable"
    if ci_low >= 0.70:
        return "downweight"
    return "exclude"


def reliability_table(rows: list[Row]) -> dict[str, AttributeReliability]:
    """Compare each corrected row's frontier snapshot against its final labels.

    Only rows with `human_corrected=True` AND a `frontier_labels` snapshot are
    counted — an uncorrected row proves nothing about the frontier, and a
    corrected row without a snapshot has already lost the evidence.
    """
    agree: Counter[str] = Counter()
    total: Counter[str] = Counter()
    confusion: dict[str, Counter] = {}
    errors: dict[str, Counter] = {}

    for row in rows:
        snapshot = row.provenance.frontier_labels
        if not row.provenance.human_corrected or not snapshot:
            continue
        for name, truth in row.labels.items():
            before = snapshot.get(name)
            if before is None:
                continue
            total[name] += 1
            if before.key() == truth.key():
                agree[name] += 1
            else:
                confusion.setdefault(name, Counter())[
                    f"{before.status.value}->{truth.status.value}"
                ] += 1
                errors.setdefault(name, Counter())[
                    f"{_render(before)} -> {_render(truth)}"
                ] += 1

    out: dict[str, AttributeReliability] = {}
    for name in sorted(total):
        n, k = total[name], agree[name]
        lo, hi = wilson(k, n)
        out[name] = AttributeReliability(
            attribute=name,
            n_compared=n,
            n_agree=k,
            accuracy=round(k / n, 4),
            ci_low=round(lo, 4),
            ci_high=round(hi, 4),
            verdict=_verdict(lo, n),
            status_confusion=dict(confusion.get(name, Counter())),
            top_errors=errors.get(name, Counter()).most_common(3),
        )
    return out


def _render(lab) -> str:
    if lab.status is LabelStatus.LABELED:
        return str(lab.value)
    return f"<{lab.status.value}>"


def frontier_baseline(rows: list[Row]) -> dict:
    """The number W1 Step 4 asks for, computed as a by-product of the corrections.

    macro = mean of per-attribute accuracies (every attribute counts equally)
    micro = pooled over all cells (common attributes dominate)

    Quote the macro figure. It is the one comparable to the macro-F1 the eval
    harness reports, and the one a lazy model cannot inflate by being good at
    `colour_primary` and useless everywhere else.
    """
    table = reliability_table(rows)
    if not table:
        return {"macro_accuracy": None, "micro_accuracy": None, "n_rows": 0}
    accs = [r.accuracy for r in table.values()]
    tot = sum(r.n_compared for r in table.values())
    agr = sum(r.n_agree for r in table.values())
    corrected = sum(
        1 for r in rows if r.provenance.human_corrected and r.provenance.frontier_labels
    )
    return {
        "macro_accuracy": round(sum(accs) / len(accs), 4),
        "micro_accuracy": round(agr / tot, 4) if tot else None,
        "n_rows": corrected,
        "n_cells": tot,
        "n_attributes": len(table),
    }


def reward_weights(rows: list[Row]) -> dict[str, float]:
    """Attribute -> suggested W2 reward weight. Feed this to the reward function."""
    return {name: r.reward_weight for name, r in reliability_table(rows).items()}


def format_report(rows: list[Row]) -> str:
    table = reliability_table(rows)
    base = frontier_baseline(rows)
    lines: list[str] = []

    if not table:
        return (
            "No comparable rows.\n"
            "  Need rows with human_corrected=True and a frontier_labels snapshot.\n"
            "  If corrections were applied without snapshotting the original, the\n"
            "  evidence is gone — re-label those rows before correcting again.\n"
        )

    lines.append(
        f"Frontier reliability — {base['n_rows']} corrected rows, "
        f"{base['n_cells']} cells, {base['n_attributes']} attributes\n"
    )
    lines.append(
        f"  {'attribute':<20} {'n':>5} {'acc':>7} {'95% CI':>16}  {'verdict':<18} w"
    )
    lines.append(f"  {'-' * 20} {'-' * 5} {'-' * 7} {'-' * 16}  {'-' * 18} ---")
    for name, r in sorted(table.items(), key=lambda kv: kv[1].ci_low):
        ci = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        lines.append(
            f"  {name:<20} {r.n_compared:>5} {r.accuracy:>7.3f} {ci:>16}  "
            f"{r.verdict:<18} {r.reward_weight:.1f}"
        )

    lines.append(
        f"\n  FRONTIER BASELINE   macro {base['macro_accuracy']:.4f}"
        f"   micro {base['micro_accuracy']:.4f}"
    )
    lines.append("  (macro is the figure to quote — it is what W2 has to beat)")

    excluded = [n for n, r in table.items() if r.reward_weight == 0.0]
    if excluded:
        lines.append(
            f"\n  EXCLUDE FROM REWARD ({len(excluded)}): {', '.join(sorted(excluded))}"
        )
        lines.append(
            "  Rewarding these teaches the model the labeler's bias, and the eval\n"
            "  set — seeded by the same labeler — will not show it."
        )

    thin = [n for n, r in table.items() if r.verdict == "insufficient_data"]
    if thin:
        lines.append(
            f"\n  TOO FEW COMPARISONS (<{MIN_COMPARISONS}): {', '.join(sorted(thin))}"
        )
        lines.append(
            "  Not a verdict of 'bad' — a verdict of 'unmeasured'. Either stratify\n"
            "  more rows containing these attributes, or leave them out of the reward."
        )

    interesting = [
        (n, r) for n, r in table.items() if r.status_confusion or r.top_errors
    ]
    if interesting:
        lines.append("\n  where the disagreements are:")
        for name, r in sorted(interesting, key=lambda kv: kv[1].ci_low)[:8]:
            conf = ", ".join(f"{k} x{v}" for k, v in sorted(r.status_confusion.items()))
            lines.append(f"    {name}: {conf}")
            for err, count in r.top_errors:
                lines.append(f"        {err}  x{count}")

    return "\n".join(lines) + "\n"
