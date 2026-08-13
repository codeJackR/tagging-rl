#!/usr/bin/env python3
"""Publish raw training-only U/UA/CB replay evidence on identical k=8 groups.

This D3 layer deliberately does not aggregate candidate rewards, compare
rankings, apply acceptance thresholds, or select a winner. One gzip JSONL line
contains one active-pool product and its exactly eight ordered starting-policy
completions, including original 1:1:2 channels and auditable U/UA/CB results.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from labeling.records import LabelStatus, Row
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.replay_original_reward import (
    DEFAULT_DIFFICULTY_MANIFEST,
    DEFAULT_PACK,
    DEFAULT_POOL_DATA,
    DEFAULT_POOL_MANIFEST,
    DEFAULT_ROLLOUTS,
    DEFAULT_SFT_SPLIT,
    ReplayInputs,
    load_locked_inputs,
    replay_reward_channels,
)
from training.run2_rewards import (
    CB_CLASS_WEIGHT_ARTIFACT_SHA256,
    VERSION as REWARD_IMPLEMENTATION_VERSION,
    CBClassWeightLookup,
    load_cb_class_weight_lookup,
    score_candidate_cb,
    score_candidate_u,
    score_candidate_ua,
)
from training.score_difficulty import ROLLOUTS_PER_PROMPT, RolloutRecord
from verifier import Pack, load_pack


VERSION = "grpo-run2-candidate-replay-records-v1"
DEFAULT_RECORDS_OUTPUT = "runs/grpo-run2-candidate-replay-records.jsonl.gz"
DEFAULT_MANIFEST_OUTPUT = "runs/grpo-run2-candidate-replay-manifest.json"
EXPECTED_ACTIVE_GROUPS = 1_438
EXPECTED_ACTIVE_COMPLETIONS = EXPECTED_ACTIVE_GROUPS * ROLLOUTS_PER_PROMPT
CANDIDATES = ("U", "UA", "CB")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _file_metadata(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _portable(value: Any, *, repo_root: Path) -> Any:
    """Replace absolute paths under the clone root with portable relative paths."""
    if isinstance(value, dict):
        return {
            key: _portable(item, repo_root=repo_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable(item, repo_root=repo_root) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return str(Path(value).resolve().relative_to(repo_root))
        except ValueError:
            return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _known_unknown_counts(row: Row) -> tuple[int, int]:
    known = sum(
        label.status is not LabelStatus.UNKNOWN for label in row.labels.values()
    )
    return known, len(row.labels) - known


def build_replay_group(
    *,
    group_position: int,
    row: Row,
    records: Sequence[RolloutRecord],
    original_rewards: Mapping[tuple[str, int], Mapping[str, float]],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> dict[str, Any]:
    """Build one auditable product group without aggregate candidate analysis."""
    if group_position < 0:
        raise ValueError("group_position cannot be negative")
    if len(records) != ROLLOUTS_PER_PROMPT:
        raise ValueError(f"{row.sku_id}: replay group must contain exactly 8 records")
    indices = [record.rollout_index for record in records]
    if indices != list(range(ROLLOUTS_PER_PROMPT)):
        raise ValueError(f"{row.sku_id}: rollout indices must be ordered 0 through 7")
    if any(record.sku_id != row.sku_id for record in records):
        raise ValueError(f"{row.sku_id}: replay group contains a different SKU")

    gold = row.to_verifier_record(pack)
    known_fields, unknown_fields = _known_unknown_counts(row)
    if known_fields <= 0 or known_fields + unknown_fields != len(pack.field_names):
        raise ValueError(f"{row.sku_id}: invalid gold-known/unknown accounting")

    completions: list[dict[str, Any]] = []
    for record in records:
        key = (row.sku_id, record.rollout_index)
        if key not in original_rewards:
            raise ValueError(f"missing original replay channels for {key}")
        original = dict(original_rewards[key])
        if set(original) != {
            "format_validity_reward",
            "vocab_rule_compliance_reward",
            "golden_agreement_reward",
            "weighted_total",
        }:
            raise ValueError(f"unexpected original reward channels for {key}")

        candidate_u = score_candidate_u(record.raw_output, gold, pack)
        candidate_ua = score_candidate_ua(record.raw_output, gold, pack)
        candidate_cb = score_candidate_cb(
            record.raw_output,
            gold,
            pack,
            class_weights,
        )
        completions.append(
            {
                "rollout_index": record.rollout_index,
                "raw_output_sha256": _sha256_text(record.raw_output),
                "source_rollout": asdict(record),
                "original_reward": original,
                "candidates": {
                    "U": asdict(candidate_u),
                    "UA": asdict(candidate_ua),
                    "CB": asdict(candidate_cb),
                },
            }
        )

    return {
        "group_position": group_position,
        "sku_id": row.sku_id,
        "difficulty_sft_pass_rate": row.difficulty.sft_pass_rate,
        "gold_record": gold,
        "gold_record_sha256": _sha256_text(_canonical_json(gold)),
        "gold_known_fields": known_fields,
        "gold_unknown_fields": unknown_fields,
        "completions": completions,
    }


def iter_replay_groups(
    *,
    inputs: ReplayInputs,
    original_rewards: Mapping[tuple[str, int], Mapping[str, float]],
    pack: Pack,
    class_weights: CBClassWeightLookup,
) -> Iterable[dict[str, Any]]:
    for position, sku_id in enumerate(inputs.active_pool_skus):
        yield build_replay_group(
            group_position=position,
            row=inputs.rows_by_sku[sku_id],
            records=inputs.records_by_sku[sku_id],
            original_rewards=original_rewards,
            pack=pack,
            class_weights=class_weights,
        )


def write_replay_groups(
    path: str | Path,
    groups: Iterable[Mapping[str, Any]],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Write canonical deterministic gzip JSONL with collision protection."""
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    if not str(path).endswith(".jsonl.gz"):
        raise ValueError("candidate replay records path must end in .jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    group_count = 0
    completion_count = 0
    sku_ids: list[str] = []
    rollout_keys: list[str] = []
    seen_skus: set[str] = set()
    seen_keys: set[tuple[str, int]] = set()
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                for expected_position, group in enumerate(groups):
                    group = dict(group)
                    if group.get("group_position") != expected_position:
                        raise ValueError("candidate replay groups are not canonically ordered")
                    sku_id = group.get("sku_id")
                    if not isinstance(sku_id, str) or not sku_id:
                        raise ValueError("candidate replay group has an invalid SKU")
                    if sku_id in seen_skus:
                        raise ValueError(f"duplicate candidate replay group: {sku_id}")
                    seen_skus.add(sku_id)
                    completions = group.get("completions")
                    if not isinstance(completions, list) or len(completions) != 8:
                        raise ValueError(f"{sku_id}: output group must contain 8 completions")
                    indices = [completion.get("rollout_index") for completion in completions]
                    if indices != list(range(8)):
                        raise ValueError(f"{sku_id}: output indices must be 0 through 7")
                    for index in indices:
                        key = (sku_id, index)
                        if key in seen_keys:
                            raise ValueError(f"duplicate candidate replay key: {key}")
                        seen_keys.add(key)
                        rollout_keys.append(f"{sku_id}\t{index}")
                    sku_ids.append(sku_id)
                    group_count += 1
                    completion_count += len(completions)
                    compressed.write((_canonical_json(group) + "\n").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        **_file_metadata(path, repo_root=repo_root),
        "jsonl_group_records": group_count,
        "completion_records": completion_count,
        "ordered_sku_sha256": _ordered_hash(sku_ids),
        "ordered_rollout_key_sha256": _ordered_hash(rollout_keys),
    }


def build_manifest(
    *,
    repo_root: str | Path,
    inputs: ReplayInputs,
    records_metadata: Mapping[str, Any],
    implementation_path: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    groups = records_metadata.get("jsonl_group_records")
    completions = records_metadata.get("completion_records")
    if groups != EXPECTED_ACTIVE_GROUPS or completions != EXPECTED_ACTIVE_COMPLETIONS:
        raise RuntimeError(
            "candidate replay output count drifted: "
            f"groups={groups}, completions={completions}"
        )
    if set(inputs.active_pool_skus) & set(inputs.validation_skus):
        raise RuntimeError("candidate replay active pool overlaps SFT validation")

    return {
        "version": VERSION,
        "status": "raw_evidence_published",
        "role": "training_only_identical_group_candidate_replay_records",
        "cuda_imports_performed": False,
        "selection_boundary": {
            "allowed": "corrected active training pool and its locked k=8 rollouts",
            "excluded": f"all {len(inputs.validation_skus)} SFT-validation SKUs",
            "legacy_frozen_300_used": False,
            "probe_100_used": False,
            "candidate_rewards_calculated": True,
            "aggregate_candidate_comparison_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_thresholds_applied": False,
            "winner_selected": False,
        },
        "record_contract": {
            "grain": "one JSONL line per product group; eight ordered completions",
            "group_size": ROLLOUTS_PER_PROMPT,
            "candidate_order": list(CANDIDATES),
            "original_reward_included": True,
            "raw_completion_text_included": True,
            "candidate_component_ledgers_included": True,
        },
        "inputs": _portable(inputs.metadata, repo_root=repo_root),
        "code": {
            "implementation": _file_metadata(
                implementation_path, repo_root=repo_root
            ),
            "candidate_rewards": _file_metadata(
                implementation_path.parent / "run2_rewards.py",
                repo_root=repo_root,
            ),
            "candidate_reward_version": REWARD_IMPLEMENTATION_VERSION,
            "original_replay_loader": _file_metadata(
                implementation_path.parent / "replay_original_reward.py",
                repo_root=repo_root,
            ),
            "cb_class_weight_artifact_sha256": CB_CLASS_WEIGHT_ARTIFACT_SHA256,
        },
        "output": dict(records_metadata),
        "integrity": {
            "active_groups": groups,
            "completion_records": completions,
            "unique_rollout_keys": completions,
            "all_groups_have_exactly_eight_ordered_completions": True,
            "same_source_completion_scored_by_all_candidates": True,
            "active_pool_is_authoritative_training_subset": (
                set(inputs.active_pool_skus) <= set(inputs.authoritative_train_skus)
            ),
            "active_pool_validation_sku_overlap": 0,
            "ordered_sku_sha256": records_metadata["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": records_metadata[
                "ordered_rollout_key_sha256"
            ],
        },
        "interpretation_guardrails": {
            "off_policy_return_estimate": False,
            "completion_independence_assumed": False,
            "candidate_superiority_claim_allowed": False,
            "next_step": (
                "aggregate these fixed records under predeclared D3 metrics; "
                "do not regenerate completions"
            ),
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
    parser.add_argument("--records-output", default=DEFAULT_RECORDS_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    records_output = _resolve(repo_root, args.records_output)
    manifest_output = _resolve(repo_root, args.manifest_output)
    if records_output.exists() or manifest_output.exists():
        existing = [str(path) for path in (records_output, manifest_output) if path.exists()]
        raise FileExistsError(f"candidate replay output already exists: {existing}")

    inputs = load_locked_inputs(
        repo_root=repo_root,
        difficulty_manifest_path=args.difficulty_manifest,
        rollouts_path=args.rollouts,
        sft_split_path=args.sft_split_manifest,
        pool_data_path=args.pool_data,
        pool_manifest_path=args.pool_manifest,
    )
    pack = load_pack(_resolve(repo_root, args.pack))
    class_weights = load_cb_class_weight_lookup(pack)
    original_rewards = replay_reward_channels(
        sku_ids=inputs.active_pool_skus,
        records_by_sku=inputs.records_by_sku,
        rows_by_sku=inputs.rows_by_sku,
        pack=pack,
    )
    groups = iter_replay_groups(
        inputs=inputs,
        original_rewards=original_rewards,
        pack=pack,
        class_weights=class_weights,
    )
    records_metadata = write_replay_groups(
        records_output,
        groups,
        repo_root=repo_root,
    )
    implementation_path = Path(__file__).resolve()
    manifest = build_manifest(
        repo_root=repo_root,
        inputs=inputs,
        records_metadata=records_metadata,
        implementation_path=implementation_path,
    )
    write_exclusive_atomic_json(manifest_output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": records_metadata,
                "manifest": _display_path(manifest_output, repo_root),
                "aggregate_candidate_comparison_calculated": False,
                "winner_selected": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
