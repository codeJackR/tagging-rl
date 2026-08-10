"""Fail-closed persistence contract for the five-step GRPO integration smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

SMOKE_BUNDLE_VERSION = "grpo-five-step-smoke-bundle-v1"
EXPECTED_STEPS = 5
GENERATIONS_PER_STEP = 8
EXPECTED_ROLLOUTS = EXPECTED_STEPS * GENERATIONS_PER_STEP
EXPECTED_TRAINABLE_TENSORS = 392
EXPECTED_TRAINABLE_PARAMETERS = 18_464_768
EXPECTED_REWARD_NAMES = (
    "format_validity_reward",
    "vocab_rule_compliance_reward",
    "golden_agreement_reward",
)
EXPECTED_REWARD_WEIGHTS = (1.0, 1.0, 2.0)
EXPECTED_LEARNING_RATE = 5e-6
EXPECTED_MAX_COMPLETION_LENGTH = 170
EXPECTED_ADAPTER_MODEL_BYTES = 73_911_112
MINIMUM_FREE_DISK_BYTES = 3 * 1024**3
EXPECTED_ADAPTER_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
FORBIDDEN_BASENAMES = {
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
    "pytorch_model.bin",
    "model.safetensors",
}
FINAL_ROOT_ENTRIES = {
    "adapter",
    "manifest.json",
    "rollouts.jsonl",
    "trainer-log.json",
}
MAX_BUNDLE_BYTES = 256 * 1024**2


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one regular file."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe_finite(value: object, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} is not finite")
    return numeric


def validate_rollout_records(
    records: Sequence[dict],
    *,
    expected_sku_ids: Sequence[str],
) -> dict:
    """Validate the complete ordered 5 x 8 raw-rollout audit trail."""
    if len(expected_sku_ids) != EXPECTED_STEPS:
        raise ValueError("smoke manifest must declare exactly five ordered SKUs")
    if len(set(expected_sku_ids)) != EXPECTED_STEPS:
        raise ValueError("smoke manifest contains duplicate SKUs")
    if len(records) != EXPECTED_ROLLOUTS:
        raise ValueError(f"smoke must preserve exactly {EXPECTED_ROLLOUTS} rollouts")

    reward_variance_steps = 0
    nonzero_advantages = 0
    total_completion_tokens = 0
    weighted_totals_by_step = []
    for step in range(1, EXPECTED_STEPS + 1):
        group = records[(step - 1) * GENERATIONS_PER_STEP : step * GENERATIONS_PER_STEP]
        if [record.get("step") for record in group] != [step] * GENERATIONS_PER_STEP:
            raise ValueError(f"rollout records are not contiguous for step {step}")
        if [record.get("rollout_index") for record in group] != list(
            range(GENERATIONS_PER_STEP)
        ):
            raise ValueError(f"rollout indices drifted at step {step}")
        expected_sku = expected_sku_ids[step - 1]
        if [record.get("sku_id") for record in group] != [
            expected_sku
        ] * GENERATIONS_PER_STEP:
            raise ValueError(f"rollout SKU drifted at step {step}")

        group_totals = []
        for record in group:
            raw_output = record.get("raw_output")
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise ValueError("rollout record has no raw model output")
            rewards = record.get("component_rewards")
            if not isinstance(rewards, dict) or set(rewards) != set(
                EXPECTED_REWARD_NAMES
            ):
                raise ValueError("rollout reward components drifted")
            component_values = [
                _json_safe_finite(rewards[name], label=f"reward {name}")
                for name in EXPECTED_REWARD_NAMES
            ]
            expected_total = sum(
                value * weight
                for value, weight in zip(
                    component_values, EXPECTED_REWARD_WEIGHTS
                )
            )
            stored_total = _json_safe_finite(
                record.get("weighted_total"), label="weighted total"
            )
            if not math.isclose(stored_total, expected_total, abs_tol=1e-9):
                raise ValueError("rollout weighted total disagrees with components")
            group_totals.append(stored_total)
            advantage = _json_safe_finite(
                record.get("advantage"), label="advantage"
            )
            if not math.isclose(advantage, 0.0, abs_tol=1e-8):
                nonzero_advantages += 1
            completion_tokens = record.get("effective_completion_tokens")
            if not isinstance(completion_tokens, int) or not (
                0 < completion_tokens <= EXPECTED_MAX_COMPLETION_LENGTH
            ):
                raise ValueError("rollout completion-token count is outside bounds")
            total_completion_tokens += completion_tokens
            if record.get("truncated_and_masked") is not False:
                raise ValueError("smoke rollout was truncated and masked")

        weighted_totals_by_step.append(group_totals)
        if len(set(group_totals)) > 1:
            reward_variance_steps += 1

    if reward_variance_steps == 0 or nonzero_advantages == 0:
        raise ValueError("smoke produced no usable group-relative reward signal")
    return {
        "records": len(records),
        "steps": EXPECTED_STEPS,
        "generations_per_step": GENERATIONS_PER_STEP,
        "reward_variance_steps": reward_variance_steps,
        "zero_variance_steps": EXPECTED_STEPS - reward_variance_steps,
        "nonzero_advantages": nonzero_advantages,
        "mean_completion_tokens": total_completion_tokens / len(records),
        "weighted_totals_by_step": weighted_totals_by_step,
        "all_rollouts_untruncated": True,
        "ordered_sku_mapping_verified": True,
    }


def _expected_step_learning_rates() -> list[float]:
    return [
        EXPECTED_LEARNING_RATE
        * 0.5
        * (1.0 + math.cos(math.pi * (step - 1) / EXPECTED_STEPS))
        for step in range(1, EXPECTED_STEPS + 1)
    ]


def validate_trainer_log_history(log_history: Sequence[dict]) -> dict:
    """Require one complete finite scalar training record for each smoke step."""
    step_logs = [entry for entry in log_history if "loss" in entry]
    if [entry.get("step") for entry in step_logs] != list(
        range(1, EXPECTED_STEPS + 1)
    ):
        raise ValueError("trainer log does not contain exactly steps 1 through 5")

    required_metrics = {
        "loss",
        "grad_norm",
        "learning_rate",
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "completions/mean_length",
        "completions/max_length",
        "completions/clipped_ratio",
        "clip_ratio/low_mean",
        "clip_ratio/high_mean",
        "rewards/format_validity_reward/mean",
        "rewards/format_validity_reward/std",
        "rewards/vocab_rule_compliance_reward/mean",
        "rewards/vocab_rule_compliance_reward/std",
        "rewards/golden_agreement_reward/mean",
        "rewards/golden_agreement_reward/std",
    }
    expected_lrs = _expected_step_learning_rates()
    losses = []
    gradient_norms = []
    for index, entry in enumerate(step_logs):
        missing = required_metrics - set(entry)
        if missing:
            raise ValueError(f"trainer step log is missing metrics: {sorted(missing)}")
        for key in required_metrics:
            _json_safe_finite(entry[key], label=f"trainer metric {key}")
        loss = float(entry["loss"])
        gradient_norm = float(entry["grad_norm"])
        if gradient_norm <= 0:
            raise ValueError("trainer logged a non-positive gradient norm")
        if not math.isclose(
            float(entry["learning_rate"]),
            expected_lrs[index],
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("trainer learning-rate schedule drifted")
        if float(entry["completions/clipped_ratio"]) != 0.0:
            raise ValueError("trainer logged a clipped completion")
        losses.append(loss)
        gradient_norms.append(gradient_norm)

    return {
        "step_logs": EXPECTED_STEPS,
        "expected_learning_rates": expected_lrs,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "all_metrics_finite": True,
        "all_gradient_norms_positive": True,
        "all_completion_clip_ratios_zero": True,
    }


def _inventory_files(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"smoke bundle may not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def validate_saved_adapter(
    adapter_dir: str | Path,
    *,
    starting_adapter_sha256: str,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Reject incomplete, resumable or full-model output masquerading as LoRA."""
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise FileNotFoundError(f"saved adapter directory is absent: {adapter_dir}")
    root_names = {path.name for path in adapter_dir.iterdir()}
    missing = EXPECTED_ADAPTER_FILES - root_names
    if missing:
        raise ValueError(f"saved adapter is missing files: {sorted(missing)}")
    if any(
        path.is_dir() and path.name.startswith("checkpoint-")
        for path in adapter_dir.rglob("*")
    ):
        raise ValueError("saved adapter contains an intermediate checkpoint")
    forbidden = [
        path
        for path in adapter_dir.rglob("*")
        if path.is_file()
        and (
            path.name in FORBIDDEN_BASENAMES
            or path.name.startswith("model-")
            or path.name.startswith("pytorch_model-")
        )
    ]
    if forbidden:
        raise ValueError(f"saved adapter contains forbidden state: {forbidden[0].name}")
    unexpected = root_names - EXPECTED_ADAPTER_FILES
    if unexpected:
        raise ValueError(
            f"saved adapter contains unexpected files: {sorted(unexpected)}"
        )

    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "base_model_name_or_path": "unsloth/Qwen2.5-1.5B-Instruct",
        "r": 16,
        "lora_alpha": 16,
        "peft_type": "LORA",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"saved adapter config drifted for {key}")
    if set(config.get("target_modules", ())) != {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }:
        raise ValueError("saved adapter target modules drifted")

    weights_path = adapter_dir / "adapter_model.safetensors"
    if weights_path.stat().st_size != expected_adapter_model_bytes:
        raise ValueError("saved adapter weight footprint drifted")
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 == starting_adapter_sha256:
        raise ValueError(
            "saved smoke adapter is byte-identical to its starting adapter"
        )
    inventory = _inventory_files(adapter_dir)
    total_bytes = sum(item["bytes"] for item in inventory)
    if total_bytes <= 0 or total_bytes > MAX_BUNDLE_BYTES:
        raise ValueError("saved adapter size is outside the smoke safety bound")
    return {
        "directory": "adapter",
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": total_bytes,
        "adapter_model_bytes": weights_path.stat().st_size,
        "adapter_model_sha256": weights_sha256,
        "adapter_config_sha256": sha256_file(config_path),
        "differs_from_starting_adapter": True,
        "contains_optimizer_state": False,
        "contains_full_model": False,
        "config_matches_locked_lora": True,
    }


def create_staging_output(final_output_dir: str | Path) -> Path:
    """Create a same-filesystem staging directory after collision checks."""
    final_output_dir = Path(final_output_dir).resolve()
    if final_output_dir.exists():
        raise FileExistsError(f"final smoke output already exists: {final_output_dir}")
    if not final_output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"final smoke output parent does not exist: {final_output_dir.parent}"
        )
    return Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-",
            dir=final_output_dir.parent,
        )
    ).resolve()


def validate_smoke_context(context: dict) -> None:
    """Validate the immutable code, source, config, runtime and resource record."""
    git = context.get("git", {})
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
    ):
        raise ValueError("smoke context has no full Git commit")
    if git.get("tracked_worktree_dirty") or git.get("index_dirty"):
        raise ValueError("smoke context Git state is dirty")
    source = context.get("source_lock", {})
    required_hashes = {
        "starting_adapter_sha256",
        "fixture_data_sha256",
        "fixture_manifest_sha256",
        "selection_manifest_sha256",
    }
    for key in required_hashes:
        value = source.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef" for character in value.lower()
            )
        ):
            raise ValueError(f"smoke context is missing source hash {key}")
    config = context.get("config", {})
    expected_config = {
        "max_steps": EXPECTED_STEPS,
        "num_generations": GENERATIONS_PER_STEP,
        "per_device_train_batch_size": GENERATIONS_PER_STEP,
        "gradient_accumulation_steps": 1,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "optim": "adamw_8bit",
        "save_strategy": "no",
        "save_only_model": True,
        "use_vllm": False,
        "beta": 0.0,
        "reward_weights": list(EXPECTED_REWARD_WEIGHTS),
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"smoke context config drifted for {key}")
    runtime = context.get("runtime", {})
    if runtime.get("global_step") != EXPECTED_STEPS:
        raise ValueError("smoke runtime did not complete exactly five steps")
    if runtime.get("trainable_tensors") != EXPECTED_TRAINABLE_TENSORS:
        raise ValueError("smoke runtime trainable tensor count drifted")
    if runtime.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("smoke runtime trainable parameter count drifted")
    if not runtime.get("source_adapter_unchanged"):
        raise ValueError("smoke runtime did not preserve the source adapter")
    runtime_hashes = [
        runtime.get("starting_lora_sha256"),
        runtime.get("final_lora_sha256"),
    ]
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
        for value in runtime_hashes
    ):
        raise ValueError("smoke runtime has an invalid LoRA fingerprint")
    if runtime_hashes[0] == runtime_hashes[1]:
        raise ValueError("smoke runtime LoRA fingerprint did not change")
    if runtime.get("optimizer_steps") != EXPECTED_STEPS:
        raise ValueError("smoke runtime optimizer-step count drifted")
    if runtime.get("rollout_records") != EXPECTED_ROLLOUTS:
        raise ValueError("smoke runtime rollout-record count drifted")
    resources = context.get("resources", {})
    peak_allocated = resources.get("peak_allocated_bytes")
    peak_reserved = resources.get("peak_reserved_bytes")
    disk_free_after = resources.get("disk_free_after_bytes")
    if not isinstance(peak_allocated, int) or peak_allocated <= 0:
        raise ValueError("smoke resources have no peak allocation")
    if not isinstance(peak_reserved, int) or peak_reserved < peak_allocated:
        raise ValueError("smoke resources have an invalid reserved-memory peak")
    if (
        not isinstance(disk_free_after, int)
        or disk_free_after < MINIMUM_FREE_DISK_BYTES
    ):
        raise ValueError("smoke resources violate the post-run disk floor")


def write_and_publish_smoke_bundle(
    *,
    staging_dir: str | Path,
    final_output_dir: str | Path,
    records: Sequence[dict],
    trainer_log_history: Sequence[dict],
    expected_sku_ids: Sequence[str],
    context: dict,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Validate, write and atomically publish one completed smoke bundle."""
    staging_dir = Path(staging_dir).resolve()
    final_output_dir = Path(final_output_dir).resolve()
    if final_output_dir.exists():
        raise FileExistsError(f"final smoke output already exists: {final_output_dir}")
    expected_prefix = f".{final_output_dir.name}.staging-"
    if (
        staging_dir.parent != final_output_dir.parent
        or not staging_dir.name.startswith(expected_prefix)
    ):
        raise ValueError("smoke staging directory is not bound to the final output")
    if not staging_dir.is_dir() or staging_dir.is_symlink():
        raise FileNotFoundError("smoke staging directory is absent")
    if {path.name for path in staging_dir.iterdir()} != {"adapter"}:
        raise ValueError(
            "staging root must contain only the saved adapter before finalization"
        )

    validate_smoke_context(context)
    rollout_summary = validate_rollout_records(
        records, expected_sku_ids=expected_sku_ids
    )
    trainer_log_summary = validate_trainer_log_history(trainer_log_history)
    starting_sha = context["source_lock"]["starting_adapter_sha256"]
    adapter_summary = validate_saved_adapter(
        staging_dir / "adapter",
        starting_adapter_sha256=starting_sha,
        expected_adapter_model_bytes=expected_adapter_model_bytes,
    )

    rollouts_path = staging_dir / "rollouts.jsonl"
    trainer_log_path = staging_dir / "trainer-log.json"
    manifest_path = staging_dir / "manifest.json"
    with rollouts_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    with trainer_log_path.open("x", encoding="utf-8") as handle:
        json.dump(
            list(trainer_log_history),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")

    manifest = {
        "version": SMOKE_BUNDLE_VERSION,
        "status": "completed",
        "run_type": "five_step_grpo_integration_smoke",
        "code": context["git"],
        "source_lock": context["source_lock"],
        "config": context["config"],
        "runtime": context["runtime"],
        "resources": context["resources"],
        "selection": {
            "ordered_sku_ids": list(expected_sku_ids),
            "steps": EXPECTED_STEPS,
            "generations_per_step": GENERATIONS_PER_STEP,
        },
        "rollouts": {
            **rollout_summary,
            "path": "rollouts.jsonl",
            "bytes": rollouts_path.stat().st_size,
            "sha256": sha256_file(rollouts_path),
        },
        "trainer_log": {
            **trainer_log_summary,
            "path": "trainer-log.json",
            "bytes": trainer_log_path.stat().st_size,
            "sha256": sha256_file(trainer_log_path),
        },
        "adapter": adapter_summary,
        "invariants": {
            "exactly_five_steps": True,
            "exactly_forty_rollouts": True,
            "ordered_sku_mapping_verified": True,
            "usable_reward_variance_observed": True,
            "all_rollouts_untruncated": True,
            "all_logged_metrics_finite": True,
            "exact_locked_lora_scope": True,
            "source_adapter_unchanged": True,
            "saved_adapter_differs_from_start": True,
            "no_optimizer_or_full_model_saved": True,
            "final_output_published_atomically": True,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    root_entries = {path.name for path in staging_dir.iterdir()}
    if root_entries != FINAL_ROOT_ENTRIES:
        raise ValueError("completed smoke staging root has unexpected entries")
    inventory = _inventory_files(staging_dir)
    bundle_bytes = sum(item["bytes"] for item in inventory)
    if bundle_bytes > MAX_BUNDLE_BYTES:
        raise ValueError("completed smoke bundle exceeds the disk safety bound")
    os.rename(staging_dir, final_output_dir)
    if not final_output_dir.is_dir() or staging_dir.exists():
        raise RuntimeError("atomic smoke publication did not complete")
    return manifest
