#!/usr/bin/env python3
"""Deterministic attribute, class and rule decomposition for two predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evalharness import predictions as preds_mod
from evalharness.metrics import AttributeScore, evaluate
from labeling import freeze
from labeling.records import Row, read_jsonl
from verifier import load_pack, verify


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_sku_sets(
    gold: Sequence[Row],
    baseline: Mapping[str, dict],
    candidate: Mapping[str, dict],
) -> list[str]:
    gold_ids = [row.sku_id for row in gold]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("gold dataset contains duplicate SKU IDs")
    gold_set = set(gold_ids)
    for name, predictions in (("baseline", baseline), ("candidate", candidate)):
        missing = gold_set - set(predictions)
        extra = set(predictions) - gold_set
        if missing or extra:
            raise ValueError(
                f"{name} SKU set differs from gold: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    return gold_ids


def _score_payload(score: AttributeScore) -> dict[str, Any]:
    return {
        "macro_f1": score.macro_f1,
        "selective_macro_f1": score.selective_macro_f1,
        "coverage": score.coverage,
        "exact_match": score.exact_match,
    }


def _class_payload(score: AttributeScore, label: str) -> dict[str, Any]:
    class_score = score.per_class.get(label)
    if class_score is None:
        return {"support": 0, "tp": 0, "fp": 0, "fn": 0, "f1": 0.0}
    return {
        "support": class_score.support,
        "tp": class_score.tp,
        "fp": class_score.fp,
        "fn": class_score.fn,
        "f1": class_score.f1,
    }


def _compare_attribute(
    baseline: AttributeScore, candidate: AttributeScore
) -> dict[str, Any]:
    baseline_values = _score_payload(baseline)
    candidate_values = _score_payload(candidate)
    classes: dict[str, Any] = {}
    for label in sorted(set(baseline.per_class) | set(candidate.per_class)):
        baseline_class = _class_payload(baseline, label)
        candidate_class = _class_payload(candidate, label)
        classes[label] = {
            "baseline": baseline_class,
            "candidate": candidate_class,
            "delta_f1": candidate_class["f1"] - baseline_class["f1"],
        }

    return {
        "n_scorable": baseline.n_scorable,
        "n_gold_unknown": baseline.n_gold_unknown,
        "baseline": baseline_values,
        "candidate": candidate_values,
        "delta_candidate_minus_baseline": {
            name: candidate_values[name] - baseline_values[name]
            for name in baseline_values
        },
        "classes": classes,
    }


def summarize_rule_transitions(
    sku_ids: Sequence[str],
    baseline_rules: Mapping[str, Sequence[str]],
    candidate_rules: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    baseline_histogram: Counter[str] = Counter()
    candidate_histogram: Counter[str] = Counter()
    transition_counts = Counter()
    added_by_rule: dict[str, list[str]] = {}
    removed_by_rule: dict[str, list[str]] = {}

    for sku_id in sku_ids:
        baseline_set = set(baseline_rules.get(sku_id, ()))
        candidate_set = set(candidate_rules.get(sku_id, ()))
        baseline_histogram.update(baseline_set)
        candidate_histogram.update(candidate_set)

        if not baseline_set and not candidate_set:
            transition_counts["both_clean"] += 1
        elif baseline_set and not candidate_set:
            transition_counts["baseline_only_violation"] += 1
        elif not baseline_set and candidate_set:
            transition_counts["candidate_only_violation"] += 1
        else:
            transition_counts["both_have_violations"] += 1

        for rule_id in sorted(candidate_set - baseline_set):
            added_by_rule.setdefault(rule_id, []).append(sku_id)
        for rule_id in sorted(baseline_set - candidate_set):
            removed_by_rule.setdefault(rule_id, []).append(sku_id)

    rule_ids = sorted(set(baseline_histogram) | set(candidate_histogram))
    return {
        "baseline_total_violations": sum(baseline_histogram.values()),
        "candidate_total_violations": sum(candidate_histogram.values()),
        "delta_total_violations": (
            sum(candidate_histogram.values()) - sum(baseline_histogram.values())
        ),
        "baseline_rows_with_violations": sum(bool(baseline_rules.get(sku)) for sku in sku_ids),
        "candidate_rows_with_violations": sum(bool(candidate_rules.get(sku)) for sku in sku_ids),
        "row_transitions": dict(sorted(transition_counts.items())),
        "rules": {
            rule_id: {
                "baseline": baseline_histogram[rule_id],
                "candidate": candidate_histogram[rule_id],
                "delta": candidate_histogram[rule_id] - baseline_histogram[rule_id],
                "added_skus": added_by_rule.get(rule_id, []),
                "removed_skus": removed_by_rule.get(rule_id, []),
            }
            for rule_id in rule_ids
        },
    }


def compare_predictions(
    gold: Sequence[Row],
    baseline_predictions: Mapping[str, dict],
    candidate_predictions: Mapping[str, dict],
    baseline_rules: Mapping[str, Sequence[str]],
    candidate_rules: Mapping[str, Sequence[str]],
    pack,
) -> dict[str, Any]:
    gold = list(gold)
    sku_ids = _validate_sku_sets(gold, baseline_predictions, candidate_predictions)
    baseline_report = evaluate(gold, dict(baseline_predictions), pack)
    candidate_report = evaluate(gold, dict(candidate_predictions), pack)

    attributes = {
        name: _compare_attribute(
            baseline_report.attributes[name], candidate_report.attributes[name]
        )
        for name in sorted(baseline_report.attributes)
    }
    macro_deltas = {
        name: values["delta_candidate_minus_baseline"]["macro_f1"]
        for name, values in attributes.items()
    }
    ranked = sorted(macro_deltas, key=lambda name: (macro_deltas[name], name))

    return {
        "headline": {
            "baseline_macro_f1": baseline_report.macro_f1,
            "candidate_macro_f1": candidate_report.macro_f1,
            "delta_macro_f1": candidate_report.macro_f1 - baseline_report.macro_f1,
            "baseline_selective_macro_f1": baseline_report.selective_macro_f1,
            "candidate_selective_macro_f1": candidate_report.selective_macro_f1,
            "delta_selective_macro_f1": (
                candidate_report.selective_macro_f1
                - baseline_report.selective_macro_f1
            ),
            "baseline_coverage": baseline_report.coverage,
            "candidate_coverage": candidate_report.coverage,
            "delta_coverage": candidate_report.coverage - baseline_report.coverage,
        },
        "attribute_direction_counts": {
            "improved": sum(delta > 0.0 for delta in macro_deltas.values()),
            "unchanged": sum(delta == 0.0 for delta in macro_deltas.values()),
            "regressed": sum(delta < 0.0 for delta in macro_deltas.values()),
        },
        "attributes_ranked_worst_to_best": ranked,
        "attributes": attributes,
        "rule_transitions": summarize_rule_transitions(
            sku_ids, baseline_rules, candidate_rules
        ),
    }


def _raw_rule_sets(path: str | Path, pack) -> dict[str, list[str]]:
    rule_sets: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        sku_id = obj.get("sku_id")
        if not isinstance(sku_id, str) or not sku_id:
            raise ValueError("prediction line has no nonempty sku_id")
        if sku_id in rule_sets:
            raise ValueError(f"duplicate prediction SKU ID: {sku_id}")
        if "raw" not in obj:
            raise ValueError("rule decomposition requires raw prediction strings")
        result = verify(obj["raw"], pack)
        rule_sets[sku_id] = sorted(set(result.rule_violations))
    return rule_sets


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = Path(args.gold)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    pack_path = Path(args.pack)
    pack = load_pack(pack_path)

    freeze_result = freeze.verify(gold_path)
    if not freeze_result.get("ok"):
        raise ValueError(f"frozen evaluation verification failed: {freeze_result}")

    gold = read_jsonl(gold_path)
    baseline = preds_mod.load(baseline_path, pack)
    candidate = preds_mod.load(candidate_path, pack)
    comparison = compare_predictions(
        gold,
        baseline.records,
        candidate.records,
        _raw_rule_sets(baseline_path, pack),
        _raw_rule_sets(candidate_path, pack),
        pack,
    )

    return {
        "version": "frozen-prediction-decomposition-v1",
        "status": "completed",
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "inputs": {
            "gold": {
                "path": str(gold_path),
                "sha256": _sha256_file(gold_path),
                "rows": len(gold),
                "freeze": freeze_result,
            },
            "baseline_predictions": {
                "path": str(baseline_path),
                "sha256": _sha256_file(baseline_path),
            },
            "candidate_predictions": {
                "path": str(candidate_path),
                "sha256": _sha256_file(candidate_path),
            },
            "pack": {
                "path": str(pack_path),
                "vocab_sha256": _sha256_file(pack_path / "vocab.yaml"),
                "rules_sha256": _sha256_file(pack_path / "rules.yaml"),
            },
        },
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/eval_300/eval.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--pack", default="packs/vastraa_taste_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(artifact["comparison"]["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
