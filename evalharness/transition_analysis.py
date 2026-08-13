#!/usr/bin/env python3
"""Paired SFT-to-GRPO cell transitions on the disclosed legacy frozen set.

This module is diagnostic tooling, not a model-selection evaluator.  It uses the
same gold-known, abstention, not-applicable and multi-value semantics as
``evalharness.metrics`` and preserves every analyzed cell in a hash-pinned JSON
artifact so aggregate claims can be audited back to individual SKUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evalharness.metrics import ABSTAIN, _gold_classes, _pred_classes
from labeling import freeze
from labeling.records import Row, read_jsonl
from verifier import load_pack, verify


STATES = ("abstain", "correct", "wrong")
STATE_TRANSITIONS = tuple(
    f"{baseline}_to_{candidate}"
    for baseline in STATES
    for candidate in STATES
)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_raw_predictions(path: str | Path, pack) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        obj = json.loads(line)
        sku_id = obj.get("sku_id")
        if not isinstance(sku_id, str) or not sku_id:
            raise ValueError(f"prediction line {line_number} has no nonempty sku_id")
        if sku_id in records:
            raise ValueError(f"duplicate prediction SKU ID: {sku_id}")
        raw = obj.get("raw")
        if not isinstance(raw, str):
            raise ValueError(
                f"prediction line {line_number} requires a raw model-output string"
            )
        result = verify(raw, pack)
        if result.parsed is None:
            raise ValueError(f"unparseable prediction for SKU {sku_id}")
        records[sku_id] = result.parsed
    return records


def _validate_pairing(
    gold: Sequence[Row],
    baseline: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, dict[str, Any]],
) -> None:
    gold_ids = [row.sku_id for row in gold]
    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError("gold dataset contains duplicate SKU IDs")
    gold_set = set(gold_ids)
    for label, predictions in (("baseline", baseline), ("candidate", candidate)):
        missing = sorted(gold_set - set(predictions))
        extra = sorted(set(predictions) - gold_set)
        if missing or extra:
            raise ValueError(
                f"{label} SKU set differs from gold: "
                f"missing={len(missing)}, extra={len(extra)}"
            )


def _state(gold_classes: set[str], predicted_classes: set[str]) -> str:
    if predicted_classes == {ABSTAIN}:
        return "abstain"
    if predicted_classes == gold_classes:
        return "correct"
    return "wrong"


def _answer_key(classes: set[str]) -> str:
    """Canonical exact-answer key; multi-value order is deliberately ignored."""
    return json.dumps(sorted(classes), ensure_ascii=False, separators=(",", ":"))


def _detailed_transition(
    baseline_state: str,
    candidate_state: str,
    baseline_classes: set[str],
    candidate_classes: set[str],
) -> str:
    if baseline_state == candidate_state == "correct":
        return "unchanged_correct"
    if baseline_state == candidate_state == "wrong":
        if baseline_classes == candidate_classes:
            return "unchanged_wrong"
        return "wrong_to_different_wrong"
    return f"{baseline_state}_to_{candidate_state}"


def _frequency_payload(
    baseline: Counter[str], candidate: Counter[str]
) -> dict[str, dict[str, float | int]]:
    baseline_total = sum(baseline.values())
    candidate_total = sum(candidate.values())
    return {
        value: {
            "baseline_count": baseline[value],
            "candidate_count": candidate[value],
            "delta_count": candidate[value] - baseline[value],
            "baseline_share": (
                baseline[value] / baseline_total if baseline_total else 0.0
            ),
            "candidate_share": (
                candidate[value] / candidate_total if candidate_total else 0.0
            ),
            "delta_share": (
                (candidate[value] / candidate_total if candidate_total else 0.0)
                - (baseline[value] / baseline_total if baseline_total else 0.0)
            ),
        }
        for value in sorted(set(baseline) | set(candidate))
    }


def _concentration(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    top_value, top_count = ranked[0] if ranked else (None, 0)
    return {
        "denominator": total,
        "unique_values": len(counter),
        "top_value": top_value,
        "top_count": top_count,
        "top_share": top_count / total if total else 0.0,
    }


def _summarize_cells(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(cells)
    state_counts = Counter(cell["state_transition"] for cell in cells)
    detailed_counts = Counter(cell["transition"] for cell in cells)
    baseline_states = Counter(cell["baseline_state"] for cell in cells)
    candidate_states = Counter(cell["candidate_state"] for cell in cells)

    baseline_answers = Counter(cell["baseline_answer_key"] for cell in cells)
    candidate_answers = Counter(cell["candidate_answer_key"] for cell in cells)
    baseline_classes: Counter[str] = Counter()
    candidate_classes: Counter[str] = Counter()
    for cell in cells:
        baseline_classes.update(cell["baseline_classes"])
        candidate_classes.update(cell["candidate_classes"])

    baseline_committed_classes = baseline_classes.copy()
    candidate_committed_classes = candidate_classes.copy()
    baseline_committed_classes.pop(ABSTAIN, None)
    candidate_committed_classes.pop(ABSTAIN, None)

    exits = state_counts["abstain_to_correct"] + state_counts["abstain_to_wrong"]
    harmful_exits = state_counts["abstain_to_wrong"]
    helpful_exits = state_counts["abstain_to_correct"]
    reverse_exits = (
        state_counts["correct_to_abstain"] + state_counts["wrong_to_abstain"]
    )

    if exits == 0:
        h1_direction = "inconclusive_no_baseline_abstain_exits"
    elif harmful_exits > helpful_exits:
        h1_direction = "strengthens_overcommitment_hypothesis"
    elif helpful_exits > harmful_exits:
        h1_direction = "weakens_overcommitment_hypothesis"
    else:
        h1_direction = "mixed_equal_helpful_and_harmful_exits"

    return {
        "gold_known_cells": total,
        "baseline_state_counts": {
            state: baseline_states[state] for state in STATES
        },
        "candidate_state_counts": {
            state: candidate_states[state] for state in STATES
        },
        "state_transition_counts": {
            transition: state_counts[transition] for transition in STATE_TRANSITIONS
        },
        "state_transition_rates": {
            transition: state_counts[transition] / total if total else 0.0
            for transition in STATE_TRANSITIONS
        },
        "detailed_transition_counts": dict(sorted(detailed_counts.items())),
        "baseline_abstain_exits": {
            "total": exits,
            "to_correct": helpful_exits,
            "to_wrong": harmful_exits,
            "to_wrong_minus_to_correct": harmful_exits - helpful_exits,
            "correct_share": helpful_exits / exits if exits else 0.0,
            "wrong_share": harmful_exits / exits if exits else 0.0,
        },
        "candidate_abstain_entries": reverse_exits,
        "net_abstentions_candidate_minus_baseline": (
            candidate_states["abstain"] - baseline_states["abstain"]
        ),
        "h1_descriptive_direction": h1_direction,
        "answer_frequency_shifts": _frequency_payload(
            baseline_answers, candidate_answers
        ),
        "class_frequency_shifts": _frequency_payload(
            baseline_classes, candidate_classes
        ),
        "common_committed_class_concentration": {
            "baseline": _concentration(baseline_committed_classes),
            "candidate": _concentration(candidate_committed_classes),
        },
        "exact_answer_concentration": {
            "baseline": _concentration(baseline_answers),
            "candidate": _concentration(candidate_answers),
        },
    }


def _examples(
    cells: Sequence[dict[str, Any]], *, limit: int
) -> dict[str, list[dict[str, Any]]]:
    counts = Counter(cell["transition"] for cell in cells)
    ranked_transitions = sorted(counts, key=lambda name: (-counts[name], name))
    examples: dict[str, list[dict[str, Any]]] = {}
    for transition in ranked_transitions:
        examples[transition] = [
            {
                key: cell[key]
                for key in (
                    "sku_id",
                    "title",
                    "attribute",
                    "gold_status",
                    "gold_value",
                    "baseline_value",
                    "candidate_value",
                )
            }
            for cell in cells
            if cell["transition"] == transition
        ][:limit]
    return examples


def analyze_transitions(
    gold: Sequence[Row],
    baseline_predictions: Mapping[str, dict[str, Any]],
    candidate_predictions: Mapping[str, dict[str, Any]],
    pack,
    *,
    examples_per_transition: int = 5,
) -> dict[str, Any]:
    """Analyze paired predictions over gold-known cells in deterministic order."""
    gold = list(gold)
    _validate_pairing(gold, baseline_predictions, candidate_predictions)
    cells: list[dict[str, Any]] = []

    for row in gold:
        baseline_record = baseline_predictions[row.sku_id]
        candidate_record = candidate_predictions[row.sku_id]
        for attribute in sorted(pack.specs):
            label = row.labels.get(attribute)
            if label is None:
                raise ValueError(
                    f"gold SKU {row.sku_id} is missing attribute {attribute}"
                )
            gold_classes = _gold_classes(label)
            if gold_classes is None:
                continue
            baseline_value = baseline_record.get(attribute, pack.unknown_token)
            candidate_value = candidate_record.get(attribute, pack.unknown_token)
            baseline_classes = _pred_classes(baseline_value, pack.unknown_token)
            candidate_classes = _pred_classes(candidate_value, pack.unknown_token)
            baseline_state = _state(gold_classes, baseline_classes)
            candidate_state = _state(gold_classes, candidate_classes)
            cells.append(
                {
                    "sku_id": row.sku_id,
                    "title": row.input.title,
                    "attribute": attribute,
                    "multi_value": pack.specs[attribute].kind == "multi",
                    "gold_status": label.status.value,
                    "gold_value": label.value,
                    "gold_classes": sorted(gold_classes),
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "baseline_classes": sorted(baseline_classes),
                    "candidate_classes": sorted(candidate_classes),
                    "baseline_answer_key": _answer_key(baseline_classes),
                    "candidate_answer_key": _answer_key(candidate_classes),
                    "baseline_state": baseline_state,
                    "candidate_state": candidate_state,
                    "state_transition": f"{baseline_state}_to_{candidate_state}",
                    "transition": _detailed_transition(
                        baseline_state,
                        candidate_state,
                        baseline_classes,
                        candidate_classes,
                    ),
                }
            )

    attributes = {
        attribute: _summarize_cells(
            [cell for cell in cells if cell["attribute"] == attribute]
        )
        for attribute in sorted(pack.specs)
    }
    multi_value_attributes = {
        attribute: attributes[attribute]
        for attribute in sorted(pack.specs)
        if pack.specs[attribute].kind == "multi"
    }
    overall = _summarize_cells(cells)

    return {
        "definitions": {
            "analysis_universe": (
                "gold-known cells only; gold status unknown is excluded"
            ),
            "abstain": f"normalized prediction is only {ABSTAIN}",
            "correct": "normalized predicted class set equals the gold class set",
            "wrong": "a committed normalized class set that does not equal gold",
            "not_applicable": "a scorable class represented as <not_applicable>",
            "multi_value": (
                "compared as order-insensitive sets; class frequencies count each "
                "emitted class, while answer frequencies count the whole set once"
            ),
            "h1_limit": (
                "direction is descriptive mechanism evidence, not a causal test; "
                "this disclosed set cannot select Run 2"
            ),
        },
        "overall": overall,
        "collar_type": attributes["collar_type"],
        "outside_collar_type": _summarize_cells(
            [cell for cell in cells if cell["attribute"] != "collar_type"]
        ),
        "attributes": attributes,
        "multi_value_attributes": multi_value_attributes,
        "examples_by_transition_largest_first": _examples(
            cells, limit=examples_per_transition
        ),
        "cells": cells,
    }


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
    baseline = _load_raw_predictions(baseline_path, pack)
    candidate = _load_raw_predictions(candidate_path, pack)

    return {
        "version": "paired-cell-transition-analysis-v1",
        "status": "completed",
        "role": "diagnosis_only_disclosed_legacy_frozen_set",
        "prohibited_uses": [
            "reward selection",
            "checkpoint selection",
            "beta selection",
            "confirmatory evaluation",
        ],
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
                "rows": len(baseline),
            },
            "candidate_predictions": {
                "path": str(candidate_path),
                "sha256": _sha256_file(candidate_path),
                "rows": len(candidate),
            },
            "pack": {
                "path": str(pack_path),
                "vocab_sha256": _sha256_file(pack_path / "vocab.yaml"),
                "rules_sha256": _sha256_file(pack_path / "rules.yaml"),
            },
        },
        "analysis": analyze_transitions(
            gold,
            baseline,
            candidate,
            pack,
            examples_per_transition=args.examples_per_transition,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/eval_300/eval.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="sft-combined-checkpoint-406")
    parser.add_argument("--candidate-label", default="grpo-first-300")
    parser.add_argument("--pack", default="packs/vastraa_taste_v1")
    parser.add_argument("--examples-per-transition", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.examples_per_transition < 1:
        parser.error("--examples-per-transition must be at least 1")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    overall = artifact["analysis"]["overall"]
    print(
        json.dumps(
            {
                "gold_known_cells": overall["gold_known_cells"],
                "baseline_state_counts": overall["baseline_state_counts"],
                "candidate_state_counts": overall["candidate_state_counts"],
                "state_transition_counts": overall["state_transition_counts"],
                "baseline_abstain_exits": overall["baseline_abstain_exits"],
                "h1_descriptive_direction": overall[
                    "h1_descriptive_direction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
