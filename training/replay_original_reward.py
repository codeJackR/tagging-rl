#!/usr/bin/env python3
"""Replay Run 1's original 1:1:2 reward on locked training rollouts.

The primary scope is the corrected 1,438-product Run 2 pool.  A secondary
3,240-product scope retains every authoritative SFT-training product so reward
behavior can also be inspected on always-fail, mixed and always-pass groups.
No validation product is replayed into either scope.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from labeling.records import Row, read_jsonl
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.grpo_run2_preflight import verify_run2_pool
from training.rewards import (
    FIRST_RUN_REWARD_FUNCTIONS,
    FIRST_RUN_REWARD_WEIGHTS,
    format_validity_reward,
    golden_agreement_reward,
    vocab_rule_compliance_reward,
)
from training.score_difficulty import ROLLOUTS_PER_PROMPT, RolloutRecord
from verifier import load_pack


VERSION = "grpo-run2-original-reward-replay-v1"
COMPONENTS = (
    "format_validity_reward",
    "vocab_rule_compliance_reward",
    "golden_agreement_reward",
)
TOTAL = "weighted_total"
DEFAULT_DIFFICULTY_MANIFEST = "runs/sft-difficulty-k8/manifest.json"
DEFAULT_ROLLOUTS = "runs/sft-difficulty-k8/rollouts.jsonl.gz"
DEFAULT_SFT_SPLIT = "data/splits/sft-v1.json"
DEFAULT_POOL_DATA = "data/train_weak_grpo_cap4_sft_train_v1.jsonl"
DEFAULT_POOL_MANIFEST = (
    "runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json"
)
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_OUTPUT = "runs/grpo-run2-original-reward-training-replay.json"
EXPECTED_DIFFICULTY_VERSION = "sft-difficulty-v2"
EXPECTED_ROLLOUT_SHA256 = (
    "f17360b157287caaea8d0f8e907f0a4bf4fd107977452442e2e447628e95bf8b"
)
EXPECTED_ROLLOUT_RECORDS = 28_800


@dataclass(frozen=True)
class ReplayInputs:
    rows_by_sku: dict[str, Row]
    records_by_sku: dict[str, list[RolloutRecord]]
    authoritative_train_skus: list[str]
    active_pool_skus: list[str]
    validation_skus: list[str]
    metadata: dict[str, Any]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def _file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _ordered_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_rollouts(path: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"rollout line {line_number} is not an object")
            try:
                records.append(RolloutRecord(**value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid rollout line {line_number}: {exc}") from exc
    return records


def _group_rollouts(
    records: Sequence[RolloutRecord], *, expected_skus: set[str]
) -> dict[str, list[RolloutRecord]]:
    if not records:
        raise ValueError("rollout artifact is empty")
    observed_order = [(record.sku_id, record.rollout_index) for record in records]
    if observed_order != sorted(observed_order):
        raise ValueError("rollout artifact is not canonically ordered by SKU/index")

    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.sku_id, record.rollout_index)
        if key in seen:
            raise ValueError(f"duplicate rollout key: {key}")
        seen.add(key)
        grouped[record.sku_id].append(record)

    if set(grouped) != expected_skus:
        raise ValueError(
            "rollout SKU set differs from difficulty source: "
            f"missing={len(expected_skus - set(grouped))}, "
            f"extra={len(set(grouped) - expected_skus)}"
        )
    expected_indices = list(range(ROLLOUTS_PER_PROMPT))
    incomplete = sorted(
        sku
        for sku, group in grouped.items()
        if [record.rollout_index for record in group] != expected_indices
    )
    if incomplete:
        raise ValueError(
            "every rollout group must contain ordered indices 0 through 7: "
            f"{incomplete[:5]}"
        )
    return dict(grouped)


def load_locked_inputs(
    *,
    repo_root: str | Path,
    difficulty_manifest_path: str | Path = DEFAULT_DIFFICULTY_MANIFEST,
    rollouts_path: str | Path = DEFAULT_ROLLOUTS,
    sft_split_path: str | Path = DEFAULT_SFT_SPLIT,
    pool_data_path: str | Path = DEFAULT_POOL_DATA,
    pool_manifest_path: str | Path = DEFAULT_POOL_MANIFEST,
    expected_rollout_sha256: str = EXPECTED_ROLLOUT_SHA256,
    expected_rollout_records: int = EXPECTED_ROLLOUT_RECORDS,
) -> ReplayInputs:
    """Verify lineage, split membership and complete k=8 rollout structure."""
    repo_root = Path(repo_root).resolve()
    difficulty_path = _resolve(repo_root, difficulty_manifest_path)
    rollouts_path = _resolve(repo_root, rollouts_path)
    split_path = _resolve(repo_root, sft_split_path)
    pool_data_path = _resolve(repo_root, pool_data_path)
    pool_manifest_path = _resolve(repo_root, pool_manifest_path)

    pool_preflight = verify_run2_pool(
        repo_root=repo_root,
        data_path=pool_data_path,
        manifest_path=pool_manifest_path,
        split_manifest_path=split_path,
    )
    difficulty = _read_json(difficulty_path)
    if difficulty.get("version") != EXPECTED_DIFFICULTY_VERSION:
        raise RuntimeError("unexpected difficulty manifest version")
    if difficulty.get("mode") != "full":
        raise RuntimeError("difficulty artifact is not a full run")

    recorded_rollouts = _resolve(
        repo_root, difficulty.get("artifacts", {}).get("rollouts", "")
    )
    if recorded_rollouts != rollouts_path:
        raise RuntimeError("difficulty manifest references a different rollout file")
    actual_rollout_sha = sha256_file(rollouts_path)
    if actual_rollout_sha != expected_rollout_sha256:
        raise RuntimeError("locked difficulty rollout checksum mismatch")
    if actual_rollout_sha != difficulty["artifacts"].get("rollouts_sha256"):
        raise RuntimeError("rollout checksum disagrees with difficulty manifest")
    if difficulty["artifacts"].get("rollout_records") != expected_rollout_records:
        raise RuntimeError("difficulty manifest rollout count drifted")
    if difficulty.get("generation", {}).get("rollouts_per_product") != 8:
        raise RuntimeError("difficulty manifest is not a k=8 rollout run")

    split = _read_json(split_path)
    source_path = _resolve(repo_root, split.get("source", ""))
    source_hash = sha256_file(source_path)
    if source_hash != split.get("source_sha256"):
        raise RuntimeError("SFT split source checksum mismatch")
    difficulty_source = _resolve(repo_root, difficulty.get("data", {}).get("source", ""))
    if difficulty_source != source_path:
        raise RuntimeError("difficulty and SFT split use different source datasets")
    if source_hash != difficulty["data"].get("source_sha256"):
        raise RuntimeError("difficulty source checksum disagrees with SFT split")

    rows = read_jsonl(source_path)
    rows_by_sku = {row.sku_id: row for row in rows}
    if len(rows_by_sku) != len(rows):
        raise ValueError("difficulty source contains duplicate SKUs")
    train_skus = split.get("train")
    validation_skus = split.get("validation")
    if not isinstance(train_skus, list) or not isinstance(validation_skus, list):
        raise ValueError("SFT split assignments must be lists")
    if len(set(train_skus)) != len(train_skus) or len(set(validation_skus)) != len(
        validation_skus
    ):
        raise ValueError("SFT split assignments contain duplicates")
    train_set = set(train_skus)
    validation_set = set(validation_skus)
    if train_set & validation_set:
        raise ValueError("SFT train and validation assignments overlap")
    if train_set | validation_set != set(rows_by_sku):
        raise ValueError("SFT split assignments do not cover the source")

    pool_manifest = _read_json(pool_manifest_path)
    active_skus = pool_manifest.get("selection", {}).get(
        "active_skus_in_source_order"
    )
    if not isinstance(active_skus, list) or len(set(active_skus)) != len(active_skus):
        raise ValueError("Run 2 active SKU order is invalid")
    if not set(active_skus) <= train_set:
        raise ValueError("Run 2 active pool contains a non-training SKU")
    if set(active_skus) & validation_set:
        raise ValueError("Run 2 active pool contains an SFT-validation SKU")

    records = _read_rollouts(rollouts_path)
    if len(records) != expected_rollout_records:
        raise RuntimeError("physical difficulty rollout count drifted")
    records_by_sku = _group_rollouts(records, expected_skus=set(rows_by_sku))

    return ReplayInputs(
        rows_by_sku=rows_by_sku,
        records_by_sku=records_by_sku,
        authoritative_train_skus=list(train_skus),
        active_pool_skus=list(active_skus),
        validation_skus=list(validation_skus),
        metadata={
            "difficulty_manifest": _file_metadata(difficulty_path),
            "difficulty_manifest_version": difficulty["version"],
            "difficulty_generation": dict(difficulty["generation"]),
            "rollouts": _file_metadata(rollouts_path),
            "sft_split_manifest": _file_metadata(split_path),
            "sft_split_manifest_version": split.get("version"),
            "source_dataset": _file_metadata(source_path),
            "run2_pool_dataset": _file_metadata(pool_data_path),
            "run2_pool_manifest": _file_metadata(pool_manifest_path),
            "run2_pool_preflight": pool_preflight,
            "physical_rollout_records": len(records),
            "physical_rollout_groups": len(records_by_sku),
        },
    )


def replay_reward_channels(
    *,
    sku_ids: Sequence[str],
    records_by_sku: Mapping[str, Sequence[RolloutRecord]],
    rows_by_sku: Mapping[str, Row],
    pack,
    batch_size: int = 1024,
) -> dict[tuple[str, int], dict[str, float]]:
    """Run the exact locked reward functions and verify durable grades agree."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered_records = [
        record for sku in sku_ids for record in records_by_sku[sku]
    ]
    rewards_by_key: dict[tuple[str, int], dict[str, float]] = {}
    mismatch_counts = Counter()
    mismatch_examples: list[dict[str, Any]] = []

    for start in range(0, len(ordered_records), batch_size):
        batch = ordered_records[start : start + batch_size]
        completions = [record.raw_output for record in batch]
        gold = [rows_by_sku[record.sku_id].to_verifier_record(pack) for record in batch]
        components = {
            "format_validity_reward": format_validity_reward(
                completions, pack=pack
            ),
            "vocab_rule_compliance_reward": vocab_rule_compliance_reward(
                completions, pack=pack
            ),
            "golden_agreement_reward": golden_agreement_reward(
                completions, gold=gold, pack=pack
            ),
        }
        for index, record in enumerate(batch):
            values = {name: components[name][index] for name in COMPONENTS}
            values[TOTAL] = sum(
                values[name] * FIRST_RUN_REWARD_WEIGHTS[position]
                for position, name in enumerate(COMPONENTS)
            )
            key = (record.sku_id, record.rollout_index)
            rewards_by_key[key] = values

            durable = {
                "format_validity_reward": float(record.schema_valid),
                "vocab_rule_compliance_reward": float(
                    record.schema_valid
                    and record.vocab_valid
                    and not record.rule_violations
                    and not record.errors
                ),
                "golden_agreement_reward": float(record.passed),
            }
            for name in COMPONENTS:
                if values[name] != durable[name]:
                    mismatch_counts[name] += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            {
                                "sku_id": record.sku_id,
                                "rollout_index": record.rollout_index,
                                "component": name,
                                "replayed": values[name],
                                "durable": durable[name],
                            }
                        )

    if mismatch_counts:
        raise RuntimeError(
            "replayed rewards disagree with durable rollout grades: "
            f"counts={dict(mismatch_counts)}, examples={mismatch_examples}"
        )
    return rewards_by_key


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _number_key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.12g}"


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "population_std": statistics.pstdev(values) if values else 0.0,
        "minimum": ordered[0] if ordered else 0.0,
        "maximum": ordered[-1] if ordered else 0.0,
        "quantiles_linear": {
            label: _quantile(ordered, probability)
            for label, probability in (
                ("p05", 0.05),
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p95", 0.95),
            )
        },
        "histogram": {
            _number_key(value): count
            for value, count in sorted(Counter(values).items())
        },
    }


def _channel_summary(group_values: Sequence[Sequence[float]]) -> dict[str, Any]:
    completion_values = [value for group in group_values for value in group]
    group_means = [statistics.fmean(group) for group in group_values]
    group_variances = [statistics.pvariance(group) for group in group_values]
    unique_counts = [len(set(group)) for group in group_values]
    largest_ties = [max(Counter(group).values()) for group in group_values]
    zero_variance = sum(variance == 0.0 for variance in group_variances)
    return {
        "completion_distribution": _distribution(completion_values),
        "group_mean_distribution": _distribution(group_means),
        "within_group_variance_distribution": _distribution(group_variances),
        "zero_variance_groups": zero_variance,
        "zero_variance_share": zero_variance / len(group_values) if group_values else 0.0,
        "groups_with_variation": len(group_values) - zero_variance,
        "unique_reward_values_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(unique_counts).items())
        },
        "largest_tie_size_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(largest_ties).items())
        },
    }


def summarize_scope(
    *,
    sku_ids: Sequence[str],
    rewards_by_key: Mapping[tuple[str, int], Mapping[str, float]],
) -> dict[str, Any]:
    """Summarize reward signal without treating completions as independent groups."""
    channels = COMPONENTS + (TOTAL,)
    values = {
        channel: [
            [rewards_by_key[(sku, index)][channel] for index in range(8)]
            for sku in sku_ids
        ]
        for channel in channels
    }
    pass_rates = [statistics.fmean(group) for group in values["golden_agreement_reward"]]
    bands = {
        "always_failed": [i for i, rate in enumerate(pass_rates) if rate == 0.0],
        "mixed": [i for i, rate in enumerate(pass_rates) if 0.0 < rate < 1.0],
        "always_passed": [i for i, rate in enumerate(pass_rates) if rate == 1.0],
    }

    return {
        "groups": len(sku_ids),
        "completions": len(sku_ids) * 8,
        "ordered_sku_sha256": _ordered_sha256(list(sku_ids)),
        "ordered_rollout_key_sha256": _ordered_sha256(
            [f"{sku}\t{index}" for sku in sku_ids for index in range(8)]
        ),
        "difficulty_band_group_counts": {
            name: len(indices) for name, indices in bands.items()
        },
        "channels": {
            channel: _channel_summary(values[channel]) for channel in channels
        },
        "difficulty_bands": {
            band: {
                "groups": len(indices),
                "weighted_total": _channel_summary(
                    [values[TOTAL][index] for index in indices]
                ),
                "component_groups_with_variation": {
                    component: _channel_summary(
                        [values[component][index] for index in indices]
                    )["groups_with_variation"]
                    for component in COMPONENTS
                },
            }
            for band, indices in bands.items()
        },
    }


def build_artifact(*, inputs: ReplayInputs, pack, implementation_path: Path) -> dict:
    reward_function_names = [function.__name__ for function in FIRST_RUN_REWARD_FUNCTIONS]
    if reward_function_names != list(COMPONENTS):
        raise RuntimeError("first-run reward function order drifted")
    if tuple(FIRST_RUN_REWARD_WEIGHTS) != (1.0, 1.0, 2.0):
        raise RuntimeError("first-run reward weights drifted")

    rewards = replay_reward_channels(
        sku_ids=inputs.authoritative_train_skus,
        records_by_sku=inputs.records_by_sku,
        rows_by_sku=inputs.rows_by_sku,
        pack=pack,
    )
    authoritative = summarize_scope(
        sku_ids=inputs.authoritative_train_skus,
        rewards_by_key=rewards,
    )
    active = summarize_scope(
        sku_ids=inputs.active_pool_skus,
        rewards_by_key=rewards,
    )

    return {
        "version": VERSION,
        "status": "completed",
        "role": "training_only_offline_original_reward_baseline",
        "cuda_imports_performed": False,
        "reward_contract": {
            "functions_in_order": reward_function_names,
            "weights_in_order": list(FIRST_RUN_REWARD_WEIGHTS),
            "weighted_total_range": [0.0, sum(FIRST_RUN_REWARD_WEIGHTS)],
            "group_size": 8,
        },
        "scope_contract": {
            "primary": (
                "run2_active_pool: exact corrected product distribution eligible "
                "for Run 2 sampling"
            ),
            "secondary": (
                "authoritative_sft_train: all training-only groups, including "
                "always-fail, mixed and always-pass starting-policy groups"
            ),
            "excluded": f"all {len(inputs.validation_skus)} SFT-validation SKUs",
            "candidate_comparison_rule": (
                "all later reward candidates must reuse these ordered SKU and "
                "rollout-key hashes"
            ),
        },
        "inputs": inputs.metadata,
        "code": {
            "implementation_file": "training/replay_original_reward.py",
            "implementation_file_sha256": sha256_file(implementation_path),
            "reward_file": "training/rewards.py",
            "reward_file_sha256": sha256_file(implementation_path.parent / "rewards.py"),
        },
        "integrity": {
            "reward_replay_matches_all_durable_rollout_grades": True,
            "replay_grade_mismatches": 0,
            "authoritative_training_groups": len(inputs.authoritative_train_skus),
            "active_pool_groups": len(inputs.active_pool_skus),
            "validation_groups_excluded": len(inputs.validation_skus),
            "active_pool_is_authoritative_training_subset": (
                set(inputs.active_pool_skus) <= set(inputs.authoritative_train_skus)
            ),
            "active_pool_validation_overlap": len(
                set(inputs.active_pool_skus) & set(inputs.validation_skus)
            ),
        },
        "scopes": {
            "run2_active_pool": active,
            "authoritative_sft_train": authoritative,
        },
        "interpretation_guardrails": {
            "direct_run1_32_7_percent_comparison_allowed": False,
            "reason": (
                "this artifact replays one fixed starting-policy sample; Run 1's "
                "32.7% came from a policy changing over 300 optimizer steps"
            ),
            "off_policy_return_estimate": False,
            "completion_independence_assumed": False,
            "selection_data": "training-only; no validation labels used",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--difficulty-manifest", default=DEFAULT_DIFFICULTY_MANIFEST)
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--sft-split-manifest", default=DEFAULT_SFT_SPLIT)
    parser.add_argument("--pool-data", default=DEFAULT_POOL_DATA)
    parser.add_argument("--pool-manifest", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    implementation = Path(__file__).resolve()
    inputs = load_locked_inputs(
        repo_root=repo_root,
        difficulty_manifest_path=args.difficulty_manifest,
        rollouts_path=args.rollouts,
        sft_split_path=args.sft_split_manifest,
        pool_data_path=args.pool_data,
        pool_manifest_path=args.pool_manifest,
    )
    pack = load_pack(_resolve(repo_root, args.pack))
    artifact = build_artifact(
        inputs=inputs,
        pack=pack,
        implementation_path=implementation,
    )
    output = _resolve(repo_root, args.output)
    write_exclusive_atomic_json(output, artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "output": str(output),
                "active_pool": {
                    "groups": artifact["scopes"]["run2_active_pool"]["groups"],
                    "zero_variance_groups": artifact["scopes"][
                        "run2_active_pool"
                    ]["channels"][TOTAL]["zero_variance_groups"],
                },
                "authoritative_sft_train": {
                    "groups": artifact["scopes"]["authoritative_sft_train"][
                        "groups"
                    ],
                    "zero_variance_groups": artifact["scopes"][
                        "authoritative_sft_train"
                    ]["channels"][TOTAL]["zero_variance_groups"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
