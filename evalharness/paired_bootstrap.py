#!/usr/bin/env python3
"""Deterministic paired row bootstrap for two saved prediction files.

Both models are evaluated on the same resampled product indices in every
replicate. That pairing preserves product difficulty: a hard product selected
three times is selected three times for both models, so the interval measures
uncertainty in their difference rather than mixing two independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from evalharness import predictions as preds_mod
from evalharness.metrics import evaluate
from labeling import freeze
from labeling.records import Row, read_jsonl
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("macro_f1", "selective_macro_f1", "coverage")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear_percentile(values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated percentile at ``(n - 1) * p``."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _metric_values(gold: list[Row], predictions: Mapping[str, dict], pack) -> dict[str, float]:
    report = evaluate(gold, dict(predictions), pack)
    return {
        "macro_f1": report.macro_f1,
        "selective_macro_f1": report.selective_macro_f1,
        "coverage": report.coverage,
    }


def _validate_pairing(
    gold: Sequence[Row],
    baseline_predictions: Mapping[str, dict],
    candidate_predictions: Mapping[str, dict],
) -> list[str]:
    gold_ids = [row.sku_id for row in gold]
    if not gold_ids:
        raise ValueError("gold dataset is empty")
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("gold dataset contains duplicate SKU IDs")

    gold_set = set(gold_ids)
    for name, predictions in (
        ("baseline", baseline_predictions),
        ("candidate", candidate_predictions),
    ):
        prediction_set = set(predictions)
        missing = gold_set - prediction_set
        extra = prediction_set - gold_set
        if missing or extra:
            raise ValueError(
                f"{name} SKU set differs from gold: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    return gold_ids


def _summarize_distribution(
    values: Sequence[float], confidence: float
) -> dict[str, Any]:
    alpha = 1.0 - confidence
    return {
        "mean": sum(values) / len(values),
        "median": linear_percentile(values, 0.5),
        "ci": [
            linear_percentile(values, alpha / 2.0),
            linear_percentile(values, 1.0 - alpha / 2.0),
        ],
    }


def paired_bootstrap(
    gold: Sequence[Row],
    baseline_predictions: Mapping[str, dict],
    candidate_predictions: Mapping[str, dict],
    pack,
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute paired percentile intervals for headline behavior metrics."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    gold = list(gold)
    _validate_pairing(gold, baseline_predictions, candidate_predictions)
    baseline_point = _metric_values(gold, baseline_predictions, pack)
    candidate_point = _metric_values(gold, candidate_predictions, pack)

    distributions = {
        metric: {"baseline": [], "candidate": [], "delta": []}
        for metric in METRICS
    }
    stream_hash = hashlib.sha256()
    rng = random.Random(seed)

    for replicate in range(replicates):
        sampled_gold = [gold[rng.randrange(len(gold))] for _ in range(len(gold))]
        baseline_values = _metric_values(sampled_gold, baseline_predictions, pack)
        candidate_values = _metric_values(sampled_gold, candidate_predictions, pack)
        stream_row: list[float | int] = [replicate]

        for metric in METRICS:
            baseline_value = baseline_values[metric]
            candidate_value = candidate_values[metric]
            delta = candidate_value - baseline_value
            distributions[metric]["baseline"].append(baseline_value)
            distributions[metric]["candidate"].append(candidate_value)
            distributions[metric]["delta"].append(delta)
            stream_row.extend((baseline_value, candidate_value, delta))

        stream_hash.update(
            (json.dumps(stream_row, separators=(",", ":")) + "\n").encode("utf-8")
        )

    results: dict[str, Any] = {}
    for metric in METRICS:
        baseline_values = distributions[metric]["baseline"]
        candidate_values = distributions[metric]["candidate"]
        deltas = distributions[metric]["delta"]
        delta_summary = _summarize_distribution(deltas, confidence)
        delta_summary.update(
            {
                "point": candidate_point[metric] - baseline_point[metric],
                "fraction_below_zero": sum(value < 0.0 for value in deltas) / replicates,
                "fraction_equal_zero": sum(value == 0.0 for value in deltas) / replicates,
                "fraction_above_zero": sum(value > 0.0 for value in deltas) / replicates,
            }
        )
        results[metric] = {
            "baseline": {
                "point": baseline_point[metric],
                **_summarize_distribution(baseline_values, confidence),
            },
            "candidate": {
                "point": candidate_point[metric],
                **_summarize_distribution(candidate_values, confidence),
            },
            "delta_candidate_minus_baseline": delta_summary,
        }

    return {
        "method": "paired nonparametric row bootstrap, percentile interval",
        "pairing_unit": "frozen product row",
        "seed": seed,
        "replicates": replicates,
        "rows_per_replicate": len(gold),
        "confidence": confidence,
        "percentile_method": "linear interpolation at sorted index (n - 1) * p",
        "replicate_stream_sha256": stream_hash.hexdigest(),
        "metrics": results,
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    pack = load_pack(args.pack)
    gold_path = Path(args.gold)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    freeze_result = freeze.verify(gold_path)
    if not freeze_result.get("ok"):
        raise ValueError(f"frozen evaluation verification failed: {freeze_result}")

    gold = read_jsonl(gold_path)
    baseline = preds_mod.load(baseline_path, pack)
    candidate = preds_mod.load(candidate_path, pack)
    result = paired_bootstrap(
        gold,
        baseline.records,
        candidate.records,
        pack,
        seed=args.seed,
        replicates=args.replicates,
        confidence=args.confidence,
    )

    return {
        "version": "paired-frozen-eval-bootstrap-v1",
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
                "attempted": baseline.n_attempted,
                "parsed": len(baseline.records),
            },
            "candidate_predictions": {
                "path": str(candidate_path),
                "sha256": _sha256_file(candidate_path),
                "attempted": candidate.n_attempted,
                "parsed": len(candidate.records),
            },
            "pack": {
                "path": str(Path(args.pack)),
                "vocab_sha256": _sha256_file(Path(args.pack) / "vocab.yaml"),
                "rules_sha256": _sha256_file(Path(args.pack) / "rules.yaml"),
            },
        },
        "bootstrap": result,
        "interpretation": {
            "primary_metric": "macro_f1",
            "decision_rule": "A percentile interval containing zero does not establish a directional difference at the declared confidence level.",
            "caveat": "The interval measures row-sampling uncertainty against the same weak frozen labels; it does not repair label reliability limitations.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/eval_300/eval.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--pack", default="packs/vastraa_taste_v1")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(artifact["bootstrap"]["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
