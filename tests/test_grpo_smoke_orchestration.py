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
)
from training.train_grpo import (
    LOCKED_TARGET_MODULES,
    LOCKED_TRAINABLE_PARAMETERS,
    LOCKED_TRAINABLE_TENSORS,
    SmokeRolloutCollector,
    grpo_smoke_config_kwargs,
    make_rollout_capturing_trainer_class,
    run_five_step_smoke_orchestration,
)


SKUS = [f"sku-{index}" for index in range(1, 6)]
SAVED_WEIGHTS = b"orchestrated updated LoRA"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step_log(step: int) -> dict:
    return {
        "step": step,
        "loss": -0.01 * step,
        "grad_norm": 0.4 + step / 100,
        "learning_rate": 5e-6
        * 0.5
        * (1.0 + math.cos(math.pi * (step - 1) / 5)),
        "reward": 3.0,
        "reward_std": 1.0,
        "frac_reward_zero_std": 0.0,
        "completions/mean_length": 93.5,
        "completions/max_length": 97.0,
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


class FakeTrainableAdapter:
    def __init__(self):
        self.updated = False
        self.save_called = False

    def save_pretrained(self, adapter_dir: Path, *, safe_serialization: bool):
        assert safe_serialization is True
        self.save_called = True
        adapter_dir.mkdir()
        (adapter_dir / "README.md").write_text("adapter\n", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(SAVED_WEIGHTS)
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


class FakeTokenizer:
    def save_pretrained(self, adapter_dir: Path):
        model_files = {
            "README.md",
            "adapter_model.safetensors",
            "adapter_config.json",
        }
        for name in EXPECTED_ADAPTER_FILES - model_files:
            (adapter_dir / name).write_text(f"tokenizer {name}\n", encoding="utf-8")


class FakeFiveStepTrainer:
    def __init__(
        self,
        *,
        model: FakeTrainableAdapter,
        steps_to_run: int = 5,
        generations_to_run: int = 5,
    ):
        self.model = model
        self.steps_to_run = steps_to_run
        self.generations_to_run = generations_to_run
        self.state = SimpleNamespace(global_step=0, log_history=[], epoch=0.0)
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_func_names = list(EXPECTED_REWARD_NAMES)
        self.reward_weights = [1.0, 1.0, 2.0]
        self.optimizer = None
        self.lr_scheduler = None
        self._logs = {}

    def _generate_and_score_completions(self, inputs):
        step = self.state.global_step + 1
        agreement = [float(index % 2) for index in range(8)]
        self._logs = {
            "completion": [
                f"step-{step}-completion-{index}" for index in range(8)
            ],
            "advantages": [-1.0 if value == 0 else 1.0 for value in agreement],
            "rewards": {
                EXPECTED_REWARD_NAMES[0]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[1]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[2]: agreement,
            },
        }
        return {"completion_mask": [[1] * (90 + index) for index in range(8)]}

    def train(self):
        for step in range(1, self.steps_to_run + 1):
            if step <= self.generations_to_run:
                inputs = [{"sku_id": SKUS[step - 1]} for _ in range(8)]
                self._generate_and_score_completions(inputs)
            self.model.updated = True
            self.state.global_step = step
            self.state.epoch = step / 5
            self.state.log_history.append(step_log(step))
        self.optimizer = SimpleNamespace(optimizer=object())
        self.lr_scheduler = object()
        return SimpleNamespace(
            global_step=self.state.global_step,
            metrics={"train_loss": -0.03, "train_runtime": 10.0},
        )


def make_preflight(source_file: Path) -> dict:
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
            "adapter_sha256": sha256_file(source_file),
            "selection_manifest_sha256": "d" * 64,
        },
    }


def make_orchestration(tmp_path: Path, **trainer_kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-adapter.safetensors"
    source.write_bytes(b"immutable source adapter")
    final = tmp_path / "grpo-first-smoke"
    model = FakeTrainableAdapter()
    collector = SmokeRolloutCollector(SKUS)
    trainer_class = make_rollout_capturing_trainer_class(FakeFiveStepTrainer)
    trainer = trainer_class(
        model=model,
        smoke_rollout_collector=collector,
        **trainer_kwargs,
    )

    def inspect_trainability(_model):
        return {
            "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
            "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
        }

    def fingerprint(active_model):
        return "f" * 64 if active_model.updated else "e" * 64

    def inspect_parameter_values(_model):
        return {
            "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
            "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
            "nonfinite_parameters": 0,
            "all_trainable_values_finite": True,
        }

    def inspect_optimizer(_optimizer, _model, *, expected_step):
        assert expected_step == 5
        return {
            "state_step_values": [expected_step],
            "state_covers_exact_locked_lora": True,
            "state_is_finite": True,
        }

    cuda_readings = iter(
        [
            {
                "torch_peak_allocated_bytes": 3_200_000_000,
                "torch_peak_reserved_bytes": 3_300_000_000,
            },
            {
                "torch_peak_allocated_bytes": 4_200_000_000,
                "torch_peak_reserved_bytes": 4_300_000_000,
            },
            {
                "torch_peak_allocated_bytes": 4_250_000_000,
                "torch_peak_reserved_bytes": 4_350_000_000,
            },
        ]
    )

    kwargs = {
        "trainer": trainer,
        "tokenizer": FakeTokenizer(),
        "source_adapter_file": source,
        "final_output_dir": final,
        "expected_sku_ids": SKUS,
        "preflight_report": make_preflight(source),
        "config_settings": grpo_smoke_config_kwargs(output_dir=final),
        "cuda_snapshot_fn": lambda: next(cuda_readings),
        "trainability_fn": inspect_trainability,
        "parameter_values_fn": inspect_parameter_values,
        "fingerprint_fn": fingerprint,
        "optimizer_inspector_fn": inspect_optimizer,
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
        "expected_adapter_model_bytes": len(SAVED_WEIGHTS),
    }
    return kwargs, model, collector


def test_orchestration_runs_five_steps_and_atomically_publishes(tmp_path):
    kwargs, model, collector = make_orchestration(tmp_path)

    report = run_five_step_smoke_orchestration(**kwargs)

    final = Path(kwargs["final_output_dir"])
    assert report["status"] == "passed"
    assert report["global_step"] == report["optimizer_steps"] == 5
    assert report["rollout_validation"]["records"] == 40
    assert report["trainer_log_validation"]["step_logs"] == 5
    assert report["optimizer_state"]["state_step_values"] == [5]
    assert report["parameter_values"]["all_trainable_values_finite"]
    assert report["manifest"]["status"] == "completed"
    assert report["manifest"]["resources"]["peak_allocated_bytes"] == (
        4_250_000_000
    )
    assert report["published"]
    assert model.save_called
    assert collector.captured_steps == 5
    assert {path.name for path in final.iterdir()} == FINAL_ROOT_ENTRIES


def test_orchestration_refuses_short_training_before_creating_output(tmp_path):
    kwargs, model, _collector = make_orchestration(tmp_path, steps_to_run=4)

    with pytest.raises(RuntimeError, match="finish exactly five updates"):
        run_five_step_smoke_orchestration(**kwargs)

    assert not Path(kwargs["final_output_dir"]).exists()
    assert not model.save_called
    assert not list(tmp_path.glob(".grpo-first-smoke.staging-*"))


def test_orchestration_refuses_missing_rollout_before_saving(tmp_path):
    kwargs, model, collector = make_orchestration(
        tmp_path,
        generations_to_run=4,
    )

    with pytest.raises(RuntimeError, match="does not contain all five groups"):
        run_five_step_smoke_orchestration(**kwargs)

    assert collector.captured_steps == 4
    assert not Path(kwargs["final_output_dir"]).exists()
    assert not model.save_called


def test_orchestration_refuses_unchanged_lora_before_saving(tmp_path):
    kwargs, model, _collector = make_orchestration(tmp_path)
    kwargs["fingerprint_fn"] = lambda _model: "e" * 64

    with pytest.raises(RuntimeError, match="changed no trainable LoRA bytes"):
        run_five_step_smoke_orchestration(**kwargs)

    assert not Path(kwargs["final_output_dir"]).exists()
    assert not model.save_called


def test_orchestration_refuses_nonfinite_lora_before_saving(tmp_path):
    kwargs, model, _collector = make_orchestration(tmp_path)

    def reject_nonfinite(_model):
        raise RuntimeError("five-step LoRA contains NaN or infinity")

    kwargs["parameter_values_fn"] = reject_nonfinite
    with pytest.raises(RuntimeError, match="contains NaN or infinity"):
        run_five_step_smoke_orchestration(**kwargs)

    assert not Path(kwargs["final_output_dir"]).exists()
    assert not model.save_called
