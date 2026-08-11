#!/usr/bin/env python3
"""Replay the locked GRPO reward functions on two frozen prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from labeling import freeze
from labeling.records import Row, read_jsonl
from training.rewards import (
    FIRST_RUN_REWARD_FUNCTIONS,
    FIRST_RUN_REWARD_WEIGHTS,
    format_validity_reward,
    golden_agreement_reward,
    vocab_rule_compliance_reward,
)
from verifier import load_pack


COMPONENTS = ("format_validity", "vocab_rule_compliance", "golden_agreement")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw_predictions(path: str | Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        sku_id = obj.get("sku_id")
        raw = obj.get("raw")
        if not isinstance(sku_id, str) or not sku_id:
            raise ValueError("prediction line has no nonempty sku_id")
        if not isinstance(raw, str):
            raise TypeError(f"prediction {sku_id} has no raw string")
        if sku_id in predictions:
            raise ValueError(f"duplicate prediction SKU ID: {sku_id}")
        predictions[sku_id] = raw
    return predictions


def _validate_pairing(
    gold: Sequence[Row],
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
) -> list[str]:
    sku_ids = [row.sku_id for row in gold]
    if len(set(sku_ids)) != len(sku_ids):
        raise ValueError("gold dataset contains duplicate SKU IDs")
    gold_set = set(sku_ids)
    for name, predictions in (("baseline", baseline), ("candidate", candidate)):
        missing = gold_set - set(predictions)
        extra = set(predictions) - gold_set
        if missing or extra:
            raise ValueError(
                f"{name} SKU set differs from gold: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    return sku_ids


def _model_rewards(
    completions: list[str], gold_records: list[dict], pack
) -> dict[str, list[float]]:
    return {
        "format_validity": format_validity_reward(completions, pack=pack),
        "vocab_rule_compliance": vocab_rule_compliance_reward(
            completions, pack=pack
        ),
        "golden_agreement": golden_agreement_reward(
            completions, gold=gold_records, pack=pack
        ),
    }


def _summarize_model(rewards: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    row_count = len(rewards[COMPONENTS[0]])
    weighted = [
        sum(
            rewards[name][index] * FIRST_RUN_REWARD_WEIGHTS[position]
            for position, name in enumerate(COMPONENTS)
        )
        for index in range(row_count)
    ]
    return {
        "rows": row_count,
        "components": {
            name: {
                "passes": int(sum(rewards[name])),
                "rate": sum(rewards[name]) / row_count if row_count else 0.0,
            }
            for name in COMPONENTS
        },
        "weighted_total": {
            "mean": sum(weighted) / row_count if row_count else 0.0,
            "maximum": sum(FIRST_RUN_REWARD_WEIGHTS),
            "histogram": {
                str(score): count
                for score, count in sorted(Counter(weighted).items())
            },
        },
    }


def _transitions(
    baseline: Sequence[float], candidate: Sequence[float]
) -> dict[str, int]:
    counts = Counter()
    for baseline_value, candidate_value in zip(baseline, candidate, strict=True):
        if baseline_value == 1.0 and candidate_value == 1.0:
            counts["both_pass"] += 1
        elif baseline_value == 1.0:
            counts["baseline_only_pass"] += 1
        elif candidate_value == 1.0:
            counts["candidate_only_pass"] += 1
        else:
            counts["both_fail"] += 1
    return {
        name: counts[name]
        for name in (
            "both_pass",
            "baseline_only_pass",
            "candidate_only_pass",
            "both_fail",
        )
    }


def replay_rewards(
    gold: Sequence[Row],
    baseline_predictions: Mapping[str, str],
    candidate_predictions: Mapping[str, str],
    pack,
) -> dict[str, Any]:
    gold = list(gold)
    sku_ids = _validate_pairing(gold, baseline_predictions, candidate_predictions)
    gold_by_sku = {row.sku_id: row.to_verifier_record(pack) for row in gold}
    gold_records = [gold_by_sku[sku_id] for sku_id in sku_ids]
    baseline_completions = [baseline_predictions[sku_id] for sku_id in sku_ids]
    candidate_completions = [candidate_predictions[sku_id] for sku_id in sku_ids]

    baseline_rewards = _model_rewards(baseline_completions, gold_records, pack)
    candidate_rewards = _model_rewards(candidate_completions, gold_records, pack)
    baseline_summary = _summarize_model(baseline_rewards)
    candidate_summary = _summarize_model(candidate_rewards)

    return {
        "reward_functions": [function.__name__ for function in FIRST_RUN_REWARD_FUNCTIONS],
        "reward_weights": list(FIRST_RUN_REWARD_WEIGHTS),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_baseline": {
            "component_pass_rates": {
                name: (
                    candidate_summary["components"][name]["rate"]
                    - baseline_summary["components"][name]["rate"]
                )
                for name in COMPONENTS
            },
            "weighted_total_mean": (
                candidate_summary["weighted_total"]["mean"]
                - baseline_summary["weighted_total"]["mean"]
            ),
        },
        "paired_component_transitions": {
            name: _transitions(baseline_rewards[name], candidate_rewards[name])
            for name in COMPONENTS
        },
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = Path(args.gold)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    pack_path = Path(args.pack)

    freeze_result = freeze.verify(gold_path)
    if not freeze_result.get("ok"):
        raise ValueError(f"frozen evaluation verification failed: {freeze_result}")

    pack = load_pack(pack_path)
    gold = read_jsonl(gold_path)
    replay = replay_rewards(
        gold,
        load_raw_predictions(baseline_path),
        load_raw_predictions(candidate_path),
        pack,
    )
    return {
        "version": "frozen-grpo-reward-replay-v1",
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
            "reward_code": {
                "path": "training/rewards.py",
                "sha256": _sha256_file("training/rewards.py"),
            },
        },
        "replay": replay,
        "interpretation_guardrails": {
            "training_objective_recreated": False,
            "reason": "This is greedy frozen-set replay. GRPO trained on sampled completions from training prompts and group-relative advantages, not the absolute mean reward of these evaluation rows.",
            "use": "A diagnostic of reward/metric alignment on fixed outputs, not an off-policy estimate of training return."
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(artifact["replay"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
