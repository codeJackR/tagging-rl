from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from training.grpo_smoke_artifacts import (
    EXPECTED_ADAPTER_FILES,
    EXPECTED_LEARNING_RATE,
    EXPECTED_REWARD_NAMES,
    EXPECTED_STEPS,
    FINAL_ROOT_ENTRIES,
    SMOKE_BUNDLE_VERSION,
    create_staging_output,
    sha256_file,
    validate_rollout_records,
    validate_saved_adapter,
    validate_smoke_context,
    validate_trainer_log_history,
    write_and_publish_smoke_bundle,
)


SKUS = [f"sku-{index}" for index in range(1, 6)]
STARTING_SHA = "a" * 64
FIXTURE_ADAPTER_WEIGHTS = b"changed LoRA weights"


def make_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir()
    for name in EXPECTED_ADAPTER_FILES:
        path = adapter_dir / name
        if name == "adapter_config.json":
            path.write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "unsloth/Qwen2.5-1.5B-Instruct",
                        "r": 16,
                        "lora_alpha": 16,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        "peft_type": "LORA",
                        "bias": "none",
                        "task_type": "CAUSAL_LM",
                    }
                ),
                encoding="utf-8",
            )
        elif name == "adapter_model.safetensors":
            path.write_bytes(FIXTURE_ADAPTER_WEIGHTS)
        else:
            path.write_text(f"fixture for {name}\n", encoding="utf-8")


def make_records() -> list[dict]:
    records = []
    for step, sku_id in enumerate(SKUS, start=1):
        for rollout_index in range(8):
            agreement = float(rollout_index % 2)
            records.append(
                {
                    "step": step,
                    "sku_id": sku_id,
                    "rollout_index": rollout_index,
                    "raw_output": json.dumps({"step": step, "i": rollout_index}),
                    "component_rewards": {
                        EXPECTED_REWARD_NAMES[0]: 1.0,
                        EXPECTED_REWARD_NAMES[1]: 1.0,
                        EXPECTED_REWARD_NAMES[2]: agreement,
                    },
                    "weighted_total": 2.0 + 2.0 * agreement,
                    "advantage": -1.0 if agreement == 0 else 1.0,
                    "effective_completion_tokens": 100 + rollout_index,
                    "truncated_and_masked": False,
                }
            )
    return records


def make_log_history() -> list[dict]:
    history = []
    for step in range(1, EXPECTED_STEPS + 1):
        learning_rate = EXPECTED_LEARNING_RATE * 0.5 * (
            1.0 + math.cos(math.pi * (step - 1) / EXPECTED_STEPS)
        )
        history.append(
            {
                "step": step,
                "loss": -0.01 * step,
                "grad_norm": 0.5,
                "learning_rate": learning_rate,
                "reward": 3.0,
                "reward_std": 1.0,
                "frac_reward_zero_std": 0.0,
                "completions/mean_length": 103.5,
                "completions/max_length": 107.0,
                "completions/clipped_ratio": 0.0,
                "clip_ratio/low_mean": 0.0,
                "clip_ratio/high_mean": 0.0,
                "rewards/format_validity_reward/mean": 1.0,
                "rewards/format_validity_reward/std": 0.0,
                "rewards/vocab_rule_compliance_reward/mean": 1.0,
                "rewards/vocab_rule_compliance_reward/std": 0.0,
                "rewards/golden_agreement_reward/mean": 0.5,
                "rewards/golden_agreement_reward/std": 0.5,
            }
        )
    history.append({"step": 5, "train_runtime": 90.0, "train_loss": -0.03})
    return history


def make_context() -> dict:
    return {
        "git": {
            "commit": "b" * 40,
            "tracked_worktree_dirty": False,
            "index_dirty": False,
        },
        "source_lock": {
            "starting_adapter_sha256": STARTING_SHA,
            "fixture_data_sha256": "c" * 64,
            "fixture_manifest_sha256": "d" * 64,
            "selection_manifest_sha256": "e" * 64,
        },
        "config": {
            "max_steps": 5,
            "num_generations": 8,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-6,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "optim": "adamw_8bit",
            "save_strategy": "no",
            "save_only_model": True,
            "use_vllm": False,
            "beta": 0.0,
            "reward_weights": [1.0, 1.0, 2.0],
        },
        "runtime": {
            "global_step": 5,
            "trainable_tensors": 392,
            "trainable_parameters": 18_464_768,
            "starting_lora_sha256": "f" * 64,
            "final_lora_sha256": "0" * 64,
            "source_adapter_unchanged": True,
            "optimizer_steps": 5,
            "rollout_records": 40,
        },
        "resources": {
            "peak_allocated_bytes": 4_300_000_000,
            "peak_reserved_bytes": 4_400_000_000,
            "disk_free_after_bytes": 4_500_000_000,
        },
    }


def test_rollout_records_require_complete_ordered_untruncated_groups():
    summary = validate_rollout_records(make_records(), expected_sku_ids=SKUS)

    assert summary["records"] == 40
    assert summary["reward_variance_steps"] == 5
    assert summary["nonzero_advantages"] == 40
    assert summary["all_rollouts_untruncated"]

    misordered = make_records()
    misordered[8]["sku_id"] = SKUS[0]
    with pytest.raises(ValueError, match="SKU drifted at step 2"):
        validate_rollout_records(misordered, expected_sku_ids=SKUS)

    truncated = make_records()
    truncated[-1]["truncated_and_masked"] = True
    with pytest.raises(ValueError, match="truncated and masked"):
        validate_rollout_records(truncated, expected_sku_ids=SKUS)


def test_trainer_log_requires_five_complete_finite_scheduled_steps():
    summary = validate_trainer_log_history(make_log_history())

    assert summary["step_logs"] == 5
    assert summary["all_metrics_finite"]
    assert summary["expected_learning_rates"][0] == EXPECTED_LEARNING_RATE

    wrong_lr = make_log_history()
    wrong_lr[2]["learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="learning-rate schedule drifted"):
        validate_trainer_log_history(wrong_lr)

    missing = make_log_history()
    del missing[0]["reward_std"]
    with pytest.raises(ValueError, match="missing metrics"):
        validate_trainer_log_history(missing)


def test_saved_adapter_rejects_optimizer_state_and_unchanged_weights(tmp_path):
    adapter = tmp_path / "adapter"
    make_adapter(adapter)
    summary = validate_saved_adapter(
        adapter,
        starting_adapter_sha256=STARTING_SHA,
        expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
    )

    assert summary["config_matches_locked_lora"]
    assert summary["differs_from_starting_adapter"]
    assert not summary["contains_optimizer_state"]

    (adapter / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="forbidden state"):
        validate_saved_adapter(
            adapter,
            starting_adapter_sha256=STARTING_SHA,
            expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
        )


def test_saved_adapter_must_change_and_match_locked_footprint(tmp_path):
    adapter = tmp_path / "adapter"
    make_adapter(adapter)

    unchanged_sha = sha256_file(adapter / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="byte-identical"):
        validate_saved_adapter(
            adapter,
            starting_adapter_sha256=unchanged_sha,
            expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
        )

    with pytest.raises(ValueError, match="footprint drifted"):
        validate_saved_adapter(
            adapter,
            starting_adapter_sha256=STARTING_SHA,
            expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS) + 1,
        )


def test_smoke_context_rejects_config_drift_and_low_disk():
    context = make_context()
    validate_smoke_context(context)

    context["config"]["warmup_ratio"] = 0.1
    with pytest.raises(ValueError, match="config drifted for warmup_ratio"):
        validate_smoke_context(context)

    context = make_context()
    context["resources"]["disk_free_after_bytes"] = 3 * 1024**3 - 1
    with pytest.raises(ValueError, match="post-run disk floor"):
        validate_smoke_context(context)


def test_smoke_bundle_is_collision_safe_and_published_atomically(tmp_path):
    final = tmp_path / "grpo-first-smoke"
    staging = create_staging_output(final)
    assert staging.parent == final.parent
    assert staging.name.startswith(".grpo-first-smoke.staging-")
    make_adapter(staging / "adapter")

    manifest = write_and_publish_smoke_bundle(
        staging_dir=staging,
        final_output_dir=final,
        records=make_records(),
        trainer_log_history=make_log_history(),
        expected_sku_ids=SKUS,
        context=make_context(),
        expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
    )

    assert manifest["version"] == SMOKE_BUNDLE_VERSION
    assert manifest["status"] == "completed"
    assert all(manifest["invariants"].values())
    assert final.is_dir()
    assert not staging.exists()
    assert {path.name for path in final.iterdir()} == FINAL_ROOT_ENTRIES
    assert sum(1 for _ in (final / "rollouts.jsonl").open()) == 40
    assert json.loads((final / "manifest.json").read_text())["status"] == (
        "completed"
    )

    with pytest.raises(FileExistsError, match="already exists"):
        create_staging_output(final)


def test_bundle_refuses_unbound_or_incomplete_staging(tmp_path):
    final = tmp_path / "grpo-first-smoke"
    unbound = tmp_path / "wrong-staging"
    unbound.mkdir()
    make_adapter(unbound / "adapter")
    with pytest.raises(ValueError, match="not bound"):
        write_and_publish_smoke_bundle(
            staging_dir=unbound,
            final_output_dir=final,
            records=make_records(),
            trainer_log_history=make_log_history(),
            expected_sku_ids=SKUS,
            context=make_context(),
            expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
        )

    staging = create_staging_output(final)
    (staging / "unexpected.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="only the saved adapter"):
        write_and_publish_smoke_bundle(
            staging_dir=staging,
            final_output_dir=final,
            records=make_records(),
            trainer_log_history=make_log_history(),
            expected_sku_ids=SKUS,
            context=make_context(),
            expected_adapter_model_bytes=len(FIXTURE_ADAPTER_WEIGHTS),
        )
