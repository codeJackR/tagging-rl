"""Executable contract for Run 2 checkpoint monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import write_exclusive_atomic_json


VERSION = "grpo-run2-checkpoint-monitor-contract-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-checkpoint-monitor-contract.json"
PRODUCTION_ROWS = 360
PRODUCTION_SAMPLED_REPETITIONS = 8
SAMPLED_SEEDS = tuple(range(20260813, 20260821))
CHECKPOINT_STEPS = (100, 200, 300)
SMOKE_ROWS = 4
SMOKE_SAMPLED_REPETITIONS = 2
INPUTS = {
    "contract_document": "W2_GRPO_RUN2_CHECKPOINT_MONITOR_CONTRACT.md",
    "data_roles": "runs/grpo-run2-data-role-manifest.json",
    "development_source": "data/train_weak.jsonl",
    "sft_split": "data/splits/sft-v1.json",
    "reward_selection": "runs/grpo-run2-d4-reward-selection.json",
    "sft_selection": "runs/sft-selection.json",
    "pack_vocab": "packs/vastraa_taste_v1/vocab.yaml",
    "pack_rules": "packs/vastraa_taste_v1/rules.yaml",
    "original_reward_code": "training/rewards.py",
    "dense_reward_code": "training/run2_rewards.py",
}
VIEW_ORDER = (
    "representative_all",
    "difficult_0_to_2_of_8",
    "middle_3_to_5_of_8",
    "easy_retention_6_to_8_of_8",
)


def _identity(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def build_contract(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    inputs = {name: _identity(root / path, root) for name, path in INPUTS.items()}
    roles = json.loads((root / INPUTS["data_roles"]).read_text(encoding="utf-8"))
    if roles.get("status") != "development_roles_locked_confirmation_required":
        raise ValueError("Run 2 data roles are not locked for development monitoring")
    development = roles.get("development", {})
    if development.get("source_rows") != PRODUCTION_ROWS:
        raise ValueError("development population must contain exactly 360 rows")
    views = development.get("views", {})
    if set(views) != set(VIEW_ORDER):
        raise ValueError("development view membership names drifted")
    expected_rows = {
        "representative_all": 360,
        "difficult_0_to_2_of_8": 204,
        "middle_3_to_5_of_8": 46,
        "easy_retention_6_to_8_of_8": 110,
    }
    locked_views = {}
    for name in VIEW_ORDER:
        view = views[name]
        skus = view.get("sku_ids_in_source_order")
        if not isinstance(skus, list) or view.get("rows") != expected_rows[name]:
            raise ValueError(f"development view {name} row count drifted")
        if len(skus) != len(set(skus)) or len(skus) != expected_rows[name]:
            raise ValueError(f"development view {name} membership is invalid")
        observed_hash = _ordered_hash(skus)
        if observed_hash != view.get("ordered_sku_sha256"):
            raise ValueError(f"development view {name} ordered hash drifted")
        locked_views[name] = {
            "rows": len(skus),
            "ordered_sku_sha256": observed_hash,
            "sku_ids_in_source_order": skus,
        }
    representative = locked_views["representative_all"]["sku_ids_in_source_order"]
    difficult = locked_views["difficult_0_to_2_of_8"]["sku_ids_in_source_order"]
    middle = locked_views["middle_3_to_5_of_8"]["sku_ids_in_source_order"]
    easy = locked_views["easy_retention_6_to_8_of_8"]["sku_ids_in_source_order"]
    if set(difficult) | set(middle) | set(easy) != set(representative):
        raise ValueError("difficulty views do not cover the representative population")
    if set(difficult) & set(middle) or set(difficult) & set(easy) or set(middle) & set(easy):
        raise ValueError("difficulty views are not disjoint")
    smoke_skus = [difficult[0], middle[0], easy[0], representative[-1]]
    if len(set(smoke_skus)) != SMOKE_ROWS:
        raise ValueError("smoke SKU selection is not unique")

    reward_selection = json.loads(
        (root / INPUTS["reward_selection"]).read_text(encoding="utf-8")
    )
    if reward_selection.get("selected_candidate") != "UA":
        raise ValueError("checkpoint monitor requires locked Candidate UA")
    sft_selection = json.loads(
        (root / INPUTS["sft_selection"]).read_text(encoding="utf-8")
    )
    selected_checkpoint = sft_selection.get("selected_checkpoint", {})
    if selected_checkpoint.get("adapter_weights", {}).get("sha256") != (
        "00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af"
    ):
        raise ValueError("locked SFT adapter identity drifted")

    return {
        "version": VERSION,
        "status": "locked_before_run2_checkpoint_evaluation",
        "role": "nonfrozen_development_checkpoint_monitor",
        "inputs": inputs,
        "development": {
            "source": inputs["development_source"],
            "rows": PRODUCTION_ROWS,
            "views": locked_views,
            "training_sku_overlap": 0,
            "training_family_overlap": 0,
            "limitations": development.get("limitations"),
        },
        "checkpoints": {
            "required_steps": list(CHECKPOINT_STEPS),
            "baseline_adapter": selected_checkpoint,
            "checkpoint_identity_required": True,
        },
        "decoding": {
            "max_prompt_length": 600,
            "max_completion_length": 170,
            "batch_size": 8,
            "greedy": {"repetitions": 1, "do_sample": False},
            "sampled": {
                "repetitions": PRODUCTION_SAMPLED_REPETITIONS,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "seeds": list(SAMPLED_SEEDS),
            },
        },
        "smoke": {
            "quality_evidence": False,
            "rows": SMOKE_ROWS,
            "sku_ids": smoke_skus,
            "sampled_repetitions": SMOKE_SAMPLED_REPETITIONS,
            "sampled_seeds": list(SAMPLED_SEEDS[:SMOKE_SAMPLED_REPETITIONS]),
            "same_temperature_top_p_and_lengths_as_production": True,
        },
        "metrics": {
            "primary": [
                "macro_f1",
                "selective_macro_f1",
                "coverage",
                "schema_validity",
                "vocab_validity",
                "rule_violations",
            ],
            "reward_replays": ["original_1_1_2", "candidate_ua"],
            "sampled_distribution": ["values", "mean", "population_stddev", "minimum", "maximum"],
            "views": list(VIEW_ORDER),
        },
        "runtime": {
            "synchronous_after_checkpoint_save": True,
            "timeout_seconds_per_checkpoint": 3600,
            "terminate_then_kill_on_timeout": True,
            "failure_artifact_required": True,
            "atomic_success_bundle_required": True,
            "resource_cleanup_evidence_required": True,
        },
        "abort_policy": {
            "monitor_failure_aborts_training": True,
            "quality_abort_enabled": False,
            "reason": "baseline variability and Phase G material/repeated guardrail are not yet locked",
        },
        "boundaries": {
            "confirmation_paths_allowed": False,
            "legacy_frozen_300_allowed": False,
            "full_grpo_training_authorized": False,
            "checkpoint_quality_claim_authorized_by_smoke": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    contract = build_contract(root)
    write_exclusive_atomic_json(output, contract)
    print(json.dumps({"output": args.output, "status": contract["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
