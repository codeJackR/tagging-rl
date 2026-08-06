from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_smoke_artifacts import (
    EXPECTED_ADAPTER_FILES,
    EXPECTED_REWARD_NAMES,
    FINAL_ROOT_ENTRIES,
    create_staging_output,
)
from training.train_grpo import (
    LOCKED_TARGET_MODULES,
    LOCKED_TRAINABLE_PARAMETERS,
    LOCKED_TRAINABLE_TENSORS,
    build_completed_smoke_context,
    grpo_smoke_config_kwargs,
    save_and_publish_completed_smoke,
)


SKUS = [f"sku-{index}" for index in range(1, 6)]
ADAPTER_WEIGHTS = b"updated smoke LoRA"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_preflight(source_adapter_file: Path) -> dict:
    return {
        "git": {
            "commit": "a" * 40,
            "tracked_worktree_dirty": False,
            "index_dirty": False,
        },
        "fixture": {
            "data_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "sft_lock": {
            "adapter_sha256": sha256_bytes(source_adapter_file.read_bytes()),
            "selection_manifest_sha256": "d" * 64,
        },
    }


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
                    "raw_output": f'{{"step": {step}, "rollout": {rollout_index}}}',
                    "component_rewards": {
                        EXPECTED_REWARD_NAMES[0]: 1.0,
                        EXPECTED_REWARD_NAMES[1]: 1.0,
                        EXPECTED_REWARD_NAMES[2]: agreement,
                    },
                    "weighted_total": 2.0 + 2.0 * agreement,
                    "advantage": -1.0 if agreement == 0 else 1.0,
                    "effective_completion_tokens": 80 + rollout_index,
                    "truncated_and_masked": False,
                }
            )
    return records


def make_log_history() -> list[dict]:
    history = []
    for step in range(1, 6):
        history.append(
            {
                "step": step,
                "loss": -0.01 * step,
                "grad_norm": 0.5,
                "learning_rate": 5e-6
                * 0.5
                * (1.0 + math.cos(math.pi * (step - 1) / 5)),
                "reward": 3.0,
                "reward_std": 1.0,
                "frac_reward_zero_std": 0.0,
                "completions/mean_length": 83.5,
                "completions/max_length": 87.0,
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
    return history


class FakeAdapterModel:
    def __init__(self, *, source_to_corrupt: Path | None = None):
        self.safe_serialization = None
        self.source_to_corrupt = source_to_corrupt

    def save_pretrained(self, adapter_dir: Path, *, safe_serialization: bool):
        self.safe_serialization = safe_serialization
        adapter_dir.mkdir()
        (adapter_dir / "README.md").write_text("smoke adapter\n", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(ADAPTER_WEIGHTS)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": (
                        "unsloth/Qwen2.5-1.5B-Instruct"
                    ),
                    "r": 16,
                    "lora_alpha": 16,
                    "target_modules": sorted(LOCKED_TARGET_MODULES),
                    "peft_type": "LORA",
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
            ),
            encoding="utf-8",
        )
        if self.source_to_corrupt is not None:
            self.source_to_corrupt.write_bytes(b"corrupted source")


class FakeTokenizer:
    def save_pretrained(self, adapter_dir: Path):
        model_files = {
            "README.md",
            "adapter_model.safetensors",
            "adapter_config.json",
        }
        for name in EXPECTED_ADAPTER_FILES - model_files:
            (adapter_dir / name).write_text(f"fixture {name}\n", encoding="utf-8")


def make_trainability() -> dict:
    return {
        "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
        "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
    }


def publish_kwargs(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-adapter.safetensors"
    source.write_bytes(b"immutable starting adapter")
    final = tmp_path / "grpo-first-smoke"
    staging = create_staging_output(final)
    return {
        "model": FakeAdapterModel(),
        "tokenizer": FakeTokenizer(),
        "source_adapter_file": source,
        "staging_dir": staging,
        "final_output_dir": final,
        "records": make_records(),
        "trainer_log_history": make_log_history(),
        "expected_sku_ids": SKUS,
        "preflight_report": make_preflight(source),
        "config_settings": grpo_smoke_config_kwargs(output_dir=staging),
        "trainability": make_trainability(),
        "global_step": 5,
        "optimizer_steps": 5,
        "starting_lora_sha256": "e" * 64,
        "final_lora_sha256": "f" * 64,
        "peak_allocated_bytes": 4_200_000_000,
        "peak_reserved_bytes": 4_300_000_000,
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
        "expected_adapter_model_bytes": len(ADAPTER_WEIGHTS),
    }


def test_completed_context_maps_preflight_and_runtime_evidence(tmp_path):
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    preflight = make_preflight(source)
    config = grpo_smoke_config_kwargs(output_dir=tmp_path)

    context = build_completed_smoke_context(
        preflight_report=preflight,
        config_settings=config,
        trainability=make_trainability(),
        global_step=5,
        optimizer_steps=5,
        rollout_records=40,
        starting_lora_sha256="e" * 64,
        final_lora_sha256="f" * 64,
        peak_allocated_bytes=4_200_000_000,
        peak_reserved_bytes=4_300_000_000,
        disk_free_after_bytes=4 * 1024**3,
    )

    assert context["git"] == preflight["git"]
    assert context["source_lock"]["starting_adapter_sha256"] == (
        preflight["sft_lock"]["adapter_sha256"]
    )
    assert context["runtime"]["rollout_records"] == 40
    assert context["resources"]["disk_free_after_bytes"] == 4 * 1024**3


def test_save_handoff_publishes_exact_adapter_and_manifest(tmp_path):
    kwargs = publish_kwargs(tmp_path)
    source_before = Path(kwargs["source_adapter_file"]).read_bytes()

    manifest = save_and_publish_completed_smoke(**kwargs)

    final = Path(kwargs["final_output_dir"])
    assert kwargs["model"].safe_serialization is True
    assert Path(kwargs["source_adapter_file"]).read_bytes() == source_before
    assert {path.name for path in final.iterdir()} == FINAL_ROOT_ENTRIES
    assert manifest["runtime"]["global_step"] == 5
    assert manifest["runtime"]["optimizer_steps"] == 5
    assert manifest["runtime"]["rollout_records"] == 40
    assert manifest["adapter"]["adapter_model_bytes"] == len(ADAPTER_WEIGHTS)
    assert manifest["resources"]["disk_free_after_bytes"] == 4 * 1024**3


def test_save_handoff_refuses_source_mutation_and_low_post_save_disk(tmp_path):
    corrupt = publish_kwargs(tmp_path / "corrupt")
    corrupt["model"].source_to_corrupt = Path(corrupt["source_adapter_file"])
    with pytest.raises(RuntimeError, match="changed while saving"):
        save_and_publish_completed_smoke(**corrupt)
    assert not Path(corrupt["final_output_dir"]).exists()
    assert Path(corrupt["staging_dir"]).is_dir()

    low_disk = publish_kwargs(tmp_path / "low-disk")
    low_disk["disk_usage_fn"] = lambda _: SimpleNamespace(free=3 * 1024**3 - 1)
    with pytest.raises(ValueError, match="post-run disk floor"):
        save_and_publish_completed_smoke(**low_disk)
    assert not Path(low_disk["final_output_dir"]).exists()
    assert Path(low_disk["staging_dir"]).is_dir()


def test_save_handoff_rejects_unbound_staging_before_model_save(tmp_path):
    kwargs = publish_kwargs(tmp_path)
    unbound = tmp_path / "unbound"
    unbound.mkdir()
    kwargs["staging_dir"] = unbound

    with pytest.raises(ValueError, match="not bound"):
        save_and_publish_completed_smoke(**kwargs)

    assert kwargs["model"].safe_serialization is None
    assert not any(unbound.iterdir())
