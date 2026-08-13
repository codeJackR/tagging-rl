"""Pure scoring and input validation for Run 2 checkpoint monitoring."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evalharness.metrics import evaluate
from evalharness.predictions import LoadedPredictions
from labeling.records import Row, read_jsonl
from training.rewards import (
    FIRST_RUN_REWARD_WEIGHTS,
    format_validity_reward,
    golden_agreement_reward,
    vocab_rule_compliance_reward,
)
from training.run2_checkpoint_monitor_contract import VERSION as CONTRACT_VERSION
from training.run2_rewards import score_candidate_ua
from verifier import Pack, verify


VERSION = "grpo-run2-checkpoint-monitor-score-v1"
SCALAR_METRICS = (
    "macro_f1",
    "selective_macro_f1",
    "coverage",
    "schema_validity",
    "vocab_validity",
    "rule_violations",
    "rule_violation_rate",
    "original_reward_mean",
    "dense_ua_reward_mean",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_contract_inputs(contract: Mapping[str, Any], root: str | Path) -> None:
    root = Path(root).resolve()
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError("unexpected checkpoint-monitor contract version")
    if contract.get("status") != "locked_before_run2_checkpoint_evaluation":
        raise ValueError("checkpoint-monitor contract is not locked")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("checkpoint-monitor contract has no input identities")
    for name, identity in inputs.items():
        if not isinstance(identity, Mapping):
            raise ValueError(f"contract input {name} is not an identity")
        path_text = identity.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError(f"contract input {name} has no path")
        path = (root / path_text).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"contract input {name} escapes repository root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"contract input {name} is missing: {path}")
        if path.stat().st_size != identity.get("bytes") or _sha256(path) != identity.get("sha256"):
            raise RuntimeError(f"checkpoint-monitor contract input drifted: {name}")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("confirmation_paths_allowed") is not False:
        raise ValueError("checkpoint-monitor contract does not forbid confirmation data")
    if boundaries.get("legacy_frozen_300_allowed") is not False:
        raise ValueError("checkpoint-monitor contract does not forbid legacy frozen data")


def load_development_rows(
    *,
    root: str | Path,
    contract: Mapping[str, Any],
    mode: str,
) -> tuple[list[Row], dict[str, list[str]]]:
    """Load exact production/smoke membership without reading confirmation data."""

    root = Path(root).resolve()
    validate_contract_inputs(contract, root)
    source_identity = contract["development"]["source"]
    source_path = (root / source_identity["path"]).resolve()
    if "confirmation" in source_path.parts or "eval_300" in source_path.parts:
        raise ValueError("checkpoint monitor source violates development-only boundary")
    rows = read_jsonl(source_path)
    by_sku = {row.sku_id: row for row in rows}
    if len(by_sku) != len(rows):
        raise ValueError("development source contains duplicate SKU IDs")
    views = {
        name: list(value["sku_ids_in_source_order"])
        for name, value in contract["development"]["views"].items()
    }
    representative = views["representative_all"]
    missing = [sku for sku in representative if sku not in by_sku]
    if missing:
        raise ValueError(f"development source is missing {len(missing)} locked SKUs")
    if _ordered_hash(representative) != contract["development"]["views"]["representative_all"]["ordered_sku_sha256"]:
        raise RuntimeError("representative development order drifted")
    if mode == "production":
        selected_skus = representative
    elif mode == "smoke":
        selected_skus = list(contract["smoke"]["sku_ids"])
    else:
        raise ValueError("monitor mode must be production or smoke")
    selected = [by_sku[sku] for sku in selected_skus]
    selected_set = set(selected_skus)
    active_views = {
        name: [sku for sku in members if sku in selected_set]
        for name, members in views.items()
    }
    if any(not members for members in active_views.values()):
        raise ValueError("monitor mode leaves a required development view empty")
    return selected, active_views


def _prediction_map(
    rows: Sequence[Row], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    expected = [row.sku_id for row in rows]
    observed: list[str] = []
    raw_by_sku: dict[str, str] = {}
    for index, prediction in enumerate(predictions):
        sku = prediction.get("sku_id")
        raw = prediction.get("raw")
        if not isinstance(sku, str) or not sku:
            raise ValueError(f"prediction {index} has no SKU")
        if not isinstance(raw, str):
            raise ValueError(f"prediction {sku} has no raw text")
        if sku in raw_by_sku:
            raise ValueError(f"duplicate prediction SKU: {sku}")
        observed.append(sku)
        raw_by_sku[sku] = raw
    if observed != expected:
        raise ValueError("prediction SKU order or membership differs from fixed development rows")
    return raw_by_sku


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("distribution requires at least one value")
    checked = [_finite(value, "distribution value") for value in values]
    return {
        "values": checked,
        "mean": statistics.fmean(checked),
        "population_stddev": statistics.pstdev(checked),
        "minimum": min(checked),
        "maximum": max(checked),
    }


def score_prediction_set(
    *,
    rows: Sequence[Row],
    predictions: Sequence[Mapping[str, Any]],
    pack: Pack,
) -> dict[str, Any]:
    """Score one exact greedy or sampled repetition on one fixed view."""

    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty view")
    raw_by_sku = _prediction_map(rows, predictions)
    loaded = LoadedPredictions(raw_mode=True, schema_valid=0)
    schema_valid = 0
    for row in rows:
        raw = raw_by_sku[row.sku_id]
        loaded.n_attempted += 1
        result = verify(raw, pack)
        if result.schema_valid:
            schema_valid += 1
        if result.parsed is None:
            loaded.unparseable += 1
            continue
        if result.vocab_valid:
            loaded.vocab_valid += 1
        for rule_id in result.rule_violations:
            loaded.rule_histogram[rule_id] += 1
        loaded.records[row.sku_id] = result.parsed
    loaded.schema_valid = schema_valid
    report = evaluate(
        list(rows),
        loaded.records,
        pack,
        schema_valid=loaded.schema_valid,
        vocab_valid=loaded.vocab_valid,
        rule_histogram=loaded.rule_histogram,
        unparseable=loaded.unparseable,
        n_attempted=loaded.n_attempted,
    )
    raws = [raw_by_sku[row.sku_id] for row in rows]
    gold = [row.to_verifier_record(pack) for row in rows]
    format_reward = format_validity_reward(raws, pack=pack)
    compliance_reward = vocab_rule_compliance_reward(raws, pack=pack)
    golden_reward = golden_agreement_reward(raws, gold=gold, pack=pack)
    original = [
        format_reward[index] * FIRST_RUN_REWARD_WEIGHTS[0]
        + compliance_reward[index] * FIRST_RUN_REWARD_WEIGHTS[1]
        + golden_reward[index] * FIRST_RUN_REWARD_WEIGHTS[2]
        for index in range(len(rows))
    ]
    dense = [
        score_candidate_ua(raw, answer, pack).reward
        for raw, answer in zip(raws, gold, strict=True)
    ]
    attempted = len(rows)
    vocab_validity = loaded.vocab_valid / attempted
    rule_violations = sum(loaded.rule_histogram.values())
    scalars = {
        "macro_f1": report.macro_f1,
        "selective_macro_f1": report.selective_macro_f1,
        "coverage": report.coverage,
        "schema_validity": schema_valid / attempted,
        "vocab_validity": vocab_validity,
        "rule_violations": float(rule_violations),
        "rule_violation_rate": rule_violations / attempted,
        "original_reward_mean": statistics.fmean(original),
        "dense_ua_reward_mean": statistics.fmean(dense),
    }
    if set(scalars) != set(SCALAR_METRICS):
        raise RuntimeError("checkpoint-monitor scalar metric set drifted")
    return {
        "version": VERSION,
        "rows": attempted,
        "attempted": attempted,
        "parsed": len(loaded.records),
        "unparseable": loaded.unparseable,
        "primary_metrics_conditional_on_parseable_outputs": bool(loaded.unparseable),
        "scalars": scalars,
        "validity": {
            "schema_valid": schema_valid,
            "vocab_valid": loaded.vocab_valid,
            "attempted_denominator": attempted,
        },
        "rule_histogram": dict(sorted(loaded.rule_histogram.items())),
        "rewards": {
            "original_1_1_2": {
                "weights": list(FIRST_RUN_REWARD_WEIGHTS),
                **_distribution(original),
            },
            "candidate_ua": _distribution(dense),
        },
        "attributes": {
            name: {
                "n_scorable": score.n_scorable,
                "n_gold_unknown": score.n_gold_unknown,
                "coverage": score.coverage,
                "macro_f1": score.macro_f1,
                "selective_macro_f1": score.selective_macro_f1,
            }
            for name, score in sorted(report.attributes.items())
        },
    }


def score_monitor_outputs(
    *,
    rows: Sequence[Row],
    views: Mapping[str, Sequence[str]],
    greedy_predictions: Sequence[Mapping[str, Any]],
    sampled_predictions: Sequence[Mapping[str, Any]],
    sampled_seeds: Sequence[int],
    pack: Pack,
) -> dict[str, Any]:
    """Score greedy plus exact repeat-major sampled outputs across fixed views."""

    rows = list(rows)
    by_sku = {row.sku_id: row for row in rows}
    if len(by_sku) != len(rows):
        raise ValueError("monitor rows contain duplicate SKU IDs")
    _prediction_map(rows, greedy_predictions)
    repetitions = len(sampled_seeds)
    if repetitions <= 1:
        raise ValueError("sampled monitoring requires repeated decoding")
    if len(sampled_predictions) != len(rows) * repetitions:
        raise ValueError("sampled prediction count differs from rows x repetitions")
    sampled_by_repeat = []
    for repeat, seed in enumerate(sampled_seeds):
        start = repeat * len(rows)
        values = list(sampled_predictions[start : start + len(rows)])
        for item in values:
            if item.get("repeat") != repeat or item.get("seed") != seed:
                raise ValueError("sampled repeat/seed lineage drifted")
        _prediction_map(rows, values)
        sampled_by_repeat.append(values)

    output_views = {}
    for view_name, sku_ids in views.items():
        sku_ids = list(sku_ids)
        if len(sku_ids) != len(set(sku_ids)) or any(sku not in by_sku for sku in sku_ids):
            raise ValueError(f"monitor view {view_name} has invalid membership")
        view_rows = [by_sku[sku] for sku in sku_ids]
        greedy_by_sku = {item["sku_id"]: item for item in greedy_predictions}
        view_greedy = [greedy_by_sku[sku] for sku in sku_ids]
        greedy_score = score_prediction_set(rows=view_rows, predictions=view_greedy, pack=pack)
        repetition_scores = []
        for repeat, values in enumerate(sampled_by_repeat):
            values_by_sku = {item["sku_id"]: item for item in values}
            score = score_prediction_set(
                rows=view_rows,
                predictions=[values_by_sku[sku] for sku in sku_ids],
                pack=pack,
            )
            repetition_scores.append(
                {"repeat": repeat, "seed": sampled_seeds[repeat], "score": score}
            )
        output_views[view_name] = {
            "rows": len(view_rows),
            "ordered_sku_sha256": _ordered_hash(sku_ids),
            "greedy": greedy_score,
            "sampled": {
                "repetitions": repetitions,
                "seeds": list(sampled_seeds),
                "per_repetition": repetition_scores,
                "aggregate": {
                    metric: _distribution(
                        [entry["score"]["scalars"][metric] for entry in repetition_scores]
                    )
                    for metric in SCALAR_METRICS
                },
            },
        }
    return {
        "version": VERSION,
        "status": "checkpoint_outputs_scored",
        "rows": len(rows),
        "sampled_repetitions": repetitions,
        "view_order": list(views),
        "views": output_views,
        "interpretation": {
            "sampled_dispersion_is_across_fixed_seed_repetitions": True,
            "greedy_sampled_gap_is_descriptive_not_causal_by_itself": True,
            "quality_abort_threshold_applied": False,
            "confirmation_data_used": False,
        },
    }
