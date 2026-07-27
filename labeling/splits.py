"""Three splits, on purpose — and an honest account of what each one costs you.

    eval  (~300, stratified)   measuring macro-F1
    probe (~100, random)       measuring cost/SKU, escalation %, throughput in W3
    train (the rest, weak)     SFT baseline + GRPO, never scored against

Why eval is deliberately unrepresentative
-----------------------------------------
Apparel attribute values are Zipf-distributed. Draw 300 rows at random and a rare
value lands 0-3 times. Macro-F1 averages per class, so a class seen twice
contributes a term with enormous variance — a genuine 5-point improvement
disappears into the noise and you cannot tell a real gain from a coin flip.

Stratifying to >=N instances per value fixes the measurement and breaks the
distribution. That is a real trade, not a free win: an eval set built this way
over-represents rare garments, so any per-SKU cost or escalation rate computed on
it is wrong for the actual catalog.

Hence the probe. 100 random rows, no stratification, whose only job is to carry
the W3 numbers that need a representative distribution. Two small sets beat one
compromise set that is silently wrong for both jobs.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .records import LabelStatus, Row

NA_KEY = "<not_applicable>"
UNK_KEY = "<unknown>"


def coverage_keys(row: Row) -> set[tuple[str, str]]:
    """(attribute, value) pairs this row provides evidence for.

    Statuses are coverage targets too: predicting `not_applicable` correctly on a
    sleeveless dress is a real skill the eval must be able to measure, so it needs
    instances the same way any value does.
    """
    keys: set[tuple[str, str]] = set()
    for name, lab in row.labels.items():
        if lab.status is LabelStatus.NOT_APPLICABLE:
            keys.add((name, NA_KEY))
        elif lab.status is LabelStatus.UNKNOWN:
            keys.add((name, UNK_KEY))
        elif isinstance(lab.value, list):
            for v in lab.value:
                keys.add((name, str(v)))
        elif lab.value is not None:
            keys.add((name, str(lab.value)))
    return keys


@dataclass
class SplitPlan:
    eval_rows: list[Row]
    probe_rows: list[Row]
    train_rows: list[Row]
    coverage: dict
    shortfalls: list[tuple[str, str, int]]  # (attribute, value, achieved)


def stratify(
    rows: list[Row],
    *,
    eval_size: int = 300,
    probe_size: int = 100,
    min_per_value: int = 8,
    seed: int = 0,
) -> SplitPlan:
    """Greedy coverage selection for eval, random for probe, remainder to train.

    Greedy because exact set-cover is NP-hard and the approximation is more than
    good enough here: at each step take the row that most reduces the largest
    remaining deficits, weighting scarce values above plentiful ones.

    Deterministic for a given `seed` — the eval set must be reproducible or
    "frozen in git" means nothing.
    """
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)  # break ties reproducibly rather than by input order

    if eval_size + probe_size > len(pool):
        raise ValueError(
            f"need at least {eval_size + probe_size} rows, got {len(pool)}"
        )

    row_keys = {id(r): coverage_keys(r) for r in pool}
    demand: Counter[tuple[str, str]] = Counter()
    for r in pool:
        for key in row_keys[id(r)]:
            demand[key] += 1
    # A value that appears in fewer rows than the target is scarce; the target for
    # it is whatever the corpus can actually supply.
    targets = {key: min(min_per_value, n) for key, n in demand.items()}

    have: Counter[tuple[str, str]] = Counter()
    chosen: list[Row] = []
    remaining = list(pool)

    def gain(row: Row) -> float:
        score = 0.0
        for key in row_keys[id(row)]:
            deficit = targets[key] - have[key]
            if deficit > 0:
                # 1/sqrt(supply) so a value with 12 candidates outranks one with 900
                score += deficit / max(1.0, demand[key] ** 0.5)
        return score

    while len(chosen) < eval_size and remaining:
        best = max(remaining, key=gain)
        if gain(best) <= 0:
            break  # every target met — fill the rest randomly below
        remaining.remove(best)
        chosen.append(best)
        for key in row_keys[id(best)]:
            have[key] += 1

    while len(chosen) < eval_size and remaining:
        chosen.append(remaining.pop())

    probe = [remaining.pop() for _ in range(min(probe_size, len(remaining)))]

    for r in chosen:
        r.split = "eval"
    for r in probe:
        r.split = "probe"
    for r in remaining:
        r.split = "train"

    shortfalls = sorted(
        (attr, val, have[(attr, val)])
        for (attr, val), target in targets.items()
        if have[(attr, val)] < target
    )

    by_attr: dict[str, dict[str, int]] = defaultdict(dict)
    for (attr, val), n in have.items():
        by_attr[attr][val] = n

    return SplitPlan(
        eval_rows=chosen,
        probe_rows=probe,
        train_rows=remaining,
        coverage={
            "min_per_value": min_per_value,
            "values_tracked": len(targets),
            "values_at_target": sum(
                1 for k, t in targets.items() if have[k] >= t
            ),
            "by_attribute": {k: dict(sorted(v.items())) for k, v in sorted(by_attr.items())},
        },
        shortfalls=shortfalls,
    )


def format_coverage(plan: SplitPlan, *, min_per_value: int = 8) -> str:
    lines = [
        f"eval {len(plan.eval_rows)}  ·  probe {len(plan.probe_rows)}  ·  "
        f"train {len(plan.train_rows)}",
        f"values at target (>={min_per_value}): "
        f"{plan.coverage['values_at_target']}/{plan.coverage['values_tracked']}",
    ]
    if plan.shortfalls:
        lines.append(
            f"\n  BELOW TARGET — {len(plan.shortfalls)} values. Macro-F1 on these is\n"
            "  high-variance; a real improvement will not be distinguishable from noise.\n"
            "  Either accept they are unmeasurable, or drop them from the scored set."
        )
        for attr, val, got in plan.shortfalls[:25]:
            lines.append(f"    {attr:<20} {val:<24} n={got}")
        if len(plan.shortfalls) > 25:
            lines.append(f"    ... and {len(plan.shortfalls) - 25} more")
    else:
        lines.append("  every tracked value met its target")
    return "\n".join(lines) + "\n"


def distribution(rows: list[Row]) -> dict[str, dict[str, float]]:
    """Marginal value distribution per attribute — the Step 1 check, on real data.

    Any attribute whose top value exceeds ~0.85 will flatter the model: predicting
    the majority everywhere scores well on accuracy and tells you nothing. Merge
    the field, cut it, or accept that only its macro-F1 term is informative.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for attr, val in coverage_keys(row):
            counts[attr][val] += 1
    out = {}
    for attr, c in sorted(counts.items()):
        total = sum(c.values())
        out[attr] = {
            "n": total,
            "distinct": len(c),
            "top_value": c.most_common(1)[0][0],
            "top_share": round(c.most_common(1)[0][1] / total, 4),
            "flatters_model": c.most_common(1)[0][1] / total > 0.85,
        }
    return out
