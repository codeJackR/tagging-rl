"""Scoring a prediction file against the frozen eval set.

Three decisions that shape every number below
---------------------------------------------

**1. A gold cell of `unknown` is not scorable.** `unknown` means the listing never
said, so there is no ground truth to compare against. Scoring a prediction against
it is scoring against our own ignorance — the model could be right and be marked
wrong. Those cells are excluded and counted separately, not silently folded in.

**2. Model abstention is neither right nor wrong, so it is reported twice.**
  headline macro-F1   abstention counts as a miss. Non-gameable: a model that
                      abstains everywhere scores 0.
  selective macro-F1  computed only over cells where the model committed, and
      + coverage       always quoted with the coverage that produced it.
A single number cannot express "declined to answer". The pair can, and it is the
same accuracy-vs-coverage frontier W4's +1/0/-lambda sweep traces out — so the
harness that measures W2 already speaks W4's language.

**3. `not_applicable` is a real class, not a gap.** Predicting that a sleeveless
dress has no sleeve length is a skill worth measuring, so it gets a class in the
F1 like any value.

Macro over micro, throughout. Apparel attribute values are Zipf-distributed;
accuracy and micro-F1 both reward a model that answers "crew, cotton, casual" to
everything. Macro-F1 averages per class, so being useless on rare values costs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from labeling.records import LabelStatus, Row

NA_CLASS = "<not_applicable>"
ABSTAIN = "<abstain>"

# A class with fewer than this many gold instances produces an F1 term with
# enormous variance — a real 5-point gain is indistinguishable from a coin flip.
LOW_SUPPORT = 5


@dataclass
class ClassScore:
    label: str
    support: int  # gold instances
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class AttributeScore:
    attribute: str
    n_rows: int
    n_gold_unknown: int  # unscorable — no ground truth exists
    n_scorable: int
    n_abstained: int
    n_correct: int  # over scorable cells; abstention is not correct
    per_class: dict[str, ClassScore] = field(default_factory=dict)
    selective_per_class: dict[str, ClassScore] = field(default_factory=dict)
    hallucinated: Counter = field(default_factory=Counter)  # predicted, never gold

    @property
    def coverage(self) -> float:
        return (
            (self.n_scorable - self.n_abstained) / self.n_scorable
            if self.n_scorable
            else 0.0
        )

    @property
    def exact_match(self) -> float:
        return self.n_correct / self.n_scorable if self.n_scorable else 0.0

    @property
    def selective_accuracy(self) -> float:
        committed = self.n_scorable - self.n_abstained
        return self.n_correct / committed if committed else 0.0

    @property
    def macro_f1(self) -> float:
        scores = [c.f1 for c in self.per_class.values() if c.support > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def selective_macro_f1(self) -> float:
        scores = [c.f1 for c in self.selective_per_class.values() if c.support > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def low_support_classes(self) -> list[str]:
        return sorted(
            c.label
            for c in self.per_class.values()
            if 0 < c.support < LOW_SUPPORT
        )


@dataclass
class Report:
    n_gold: int
    n_predicted: int
    n_missing: list[str]
    n_unparseable: int
    n_attempted: int  # lines the model produced, incl. those too malformed to keep
    schema_valid: int | None  # None when predictions arrived pre-parsed
    vocab_valid: int
    rule_violations: int
    rule_histogram: Counter
    attributes: dict[str, AttributeScore]
    reward_weights: dict[str, float] = field(default_factory=dict)
    freeze: dict[str, Any] = field(default_factory=dict)

    @property
    def macro_f1(self) -> float:
        """Headline. Mean of per-attribute macro-F1 — every attribute counts once."""
        vals = [a.macro_f1 for a in self.attributes.values() if a.n_scorable]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def selective_macro_f1(self) -> float:
        vals = [a.selective_macro_f1 for a in self.attributes.values() if a.n_scorable]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def trusted_macro_f1(self) -> float:
        """Headline restricted to attributes the frontier was reliable on.

        Step 3's reliability table gives each attribute a weight; a 0 means the
        labeler was too unreliable there for the gold to be trustworthy. Scoring a
        model against untrustworthy gold measures agreement with a bias, so this
        figure drops those attributes. Quote it next to the headline, never instead.
        """
        vals = [
            a.macro_f1
            for name, a in self.attributes.items()
            if a.n_scorable and self.reward_weights.get(name, 1.0) > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def coverage(self) -> float:
        scorable = sum(a.n_scorable for a in self.attributes.values())
        abstained = sum(a.n_abstained for a in self.attributes.values())
        return (scorable - abstained) / scorable if scorable else 0.0

    @property
    def schema_validity(self) -> float | None:
        """Valid over ATTEMPTED, not over survivors.

        Dividing by the rows that parsed excludes exactly the rows that failed, and
        prints "100% valid, 20 unparseable" — which is how a format failure hides.
        """
        if self.schema_valid is None:
            return None
        denom = self.n_attempted or self.n_predicted
        return self.schema_valid / denom if denom else 0.0


def _gold_classes(label) -> set[str] | None:
    """Gold cell -> the set of classes it asserts, or None if unscorable."""
    if label.status is LabelStatus.UNKNOWN:
        return None  # no ground truth; the listing never said
    if label.status is LabelStatus.NOT_APPLICABLE:
        return {NA_CLASS}
    if isinstance(label.value, list):
        return set(label.value)
    return {str(label.value)}


def _pred_classes(value: Any, unknown_token: str) -> set[str]:
    """Predicted cell -> the set of classes it asserts. Abstention is its own class."""
    if value is None:
        return {NA_CLASS}
    if isinstance(value, list):
        vals = {str(v) for v in value}
        return {ABSTAIN} if vals == {unknown_token} else (vals - {unknown_token} or {ABSTAIN})
    if value == unknown_token:
        return {ABSTAIN}
    return {str(value)}


def score_attribute(
    attribute: str,
    pairs: list[tuple[Any, Any]],  # (gold AttributeLabel, predicted raw value)
    unknown_token: str,
) -> AttributeScore:
    out = AttributeScore(
        attribute=attribute, n_rows=len(pairs), n_gold_unknown=0, n_scorable=0,
        n_abstained=0, n_correct=0,
    )
    tally: dict[str, ClassScore] = {}
    sel: dict[str, ClassScore] = {}

    def bump(store, label, *, tp=0, fp=0, fn=0, support=0):
        cs = store.setdefault(label, ClassScore(label, 0, 0, 0, 0))
        cs.tp += tp
        cs.fp += fp
        cs.fn += fn
        cs.support += support

    for gold_label, pred_value in pairs:
        gold = _gold_classes(gold_label)
        if gold is None:
            out.n_gold_unknown += 1
            continue
        out.n_scorable += 1
        pred = _pred_classes(pred_value, unknown_token)
        abstained = pred == {ABSTAIN}
        if abstained:
            out.n_abstained += 1
        if pred == gold:
            out.n_correct += 1

        for cls in gold:
            bump(tally, cls, support=1, tp=1 if cls in pred else 0,
                 fn=0 if cls in pred else 1)
        for cls in pred - gold:
            if cls == ABSTAIN:
                continue  # abstention is a miss on the gold class, not a false class
            bump(tally, cls, fp=1)
            out.hallucinated[cls] += 1

        if abstained:
            continue
        for cls in gold:
            bump(sel, cls, support=1, tp=1 if cls in pred else 0,
                 fn=0 if cls in pred else 1)
        for cls in pred - gold:
            bump(sel, cls, fp=1)

    out.per_class = tally
    out.selective_per_class = sel
    return out


def evaluate(
    gold: list[Row],
    predictions: dict[str, dict[str, Any]],
    pack,
    *,
    reward_weights: dict[str, float] | None = None,
    schema_valid: int | None = None,
    vocab_valid: int = 0,
    rule_histogram: Counter | None = None,
    unparseable: int = 0,
    n_attempted: int = 0,
) -> Report:
    by_attr: dict[str, list[tuple[Any, Any]]] = {name: [] for name in pack.specs}
    missing: list[str] = []

    for row in gold:
        pred = predictions.get(row.sku_id)
        if pred is None:
            missing.append(row.sku_id)
            continue
        for name, label in row.labels.items():
            if name in by_attr:
                by_attr[name].append((label, pred.get(name, pack.unknown_token)))

    rule_histogram = rule_histogram or Counter()
    return Report(
        n_gold=len(gold),
        n_predicted=len(predictions),
        n_missing=missing,
        n_unparseable=unparseable,
        n_attempted=n_attempted or len(predictions),
        schema_valid=schema_valid,
        vocab_valid=vocab_valid,
        rule_violations=sum(rule_histogram.values()),
        rule_histogram=rule_histogram,
        attributes={
            name: score_attribute(name, pairs, pack.unknown_token)
            for name, pairs in by_attr.items()
            if pairs
        },
        reward_weights=reward_weights or {},
    )
