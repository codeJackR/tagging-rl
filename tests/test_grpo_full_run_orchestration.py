"""CPU-only end-to-end tests for the guarded 300-step orchestration shell."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_full_run_evidence import expected_full_run_learning_rates
from training.grpo_smoke_artifacts import EXPECTED_ADAPTER_FILES
from training.train_grpo import (
    LOCKED_TARGET_MODULES,
    run_full_run_300_orchestration,
)

SOURCE_BYTES = b"immutable SFT source adapter"
SOURCE_SHA = hashlib.sha256(SOURCE_BYTES).hexdigest()
FINAL_WEIGHTS = b"final GRPO adapter after 300 steps"


def format_validity_reward(*args, **kwargs):
    return [1.0]


def vocab_rule_compliance_reward(*args, **kwargs):
    return [1.0]


def golden_agreement_reward(*args, **kwargs):
    return [1.0]


REWARD_FUNCTIONS = (
    format_validity_reward,
    vocab_rule_compliance_reward,
    golden_agreement_reward,
)


class FakeDataset:
    column_names = ["prompt", "gold", "sku_id"]

    def __init__(self, rows=1_565):
        self.sku_ids = [f"sku-{index:04d}" for index in range(rows)]

    def __len__(self):
        return len(self.sku_ids)

    def __getitem__(self, key):
        if key == "sku_id":
            return list(self.sku_ids)
        raise KeyError(key)


class FakeGRPOConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.generation_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )


class FakeTrainerCallback:
    pass


class FakeModel:
    def __init__(self, *, final_weights=FINAL_WEIGHTS):
        self.final_weights = final_weights
        self.save_calls = 0
        self.updated = False

    def save_pretrained(self, path: Path, *, safe_serialization: bool):
        assert safe_serialization is True
        self.save_calls += 1
        path.mkdir()
        config = {
            "base_model_name_or_path": "unsloth/Qwen2.5-1.5B-Instruct",
            "r": 16,
            "lora_alpha": 16,
            "peft_type": "LORA",
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": sorted(LOCKED_TARGET_MODULES),
        }
        (path / "adapter_model.safetensors").write_bytes(self.final_weights)
        (path / "adapter_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (path / "README.md").write_text("adapter\n", encoding="utf-8")


class FakeTokenizer:
    def save_pretrained(self, path: Path):
        model_files = {
            "README.md",
            "adapter_model.safetensors",
            "adapter_config.json",
        }
        for name in EXPECTED_ADAPTER_FILES - model_files:
            (path / name).write_text(f"tokenizer {name}\n", encoding="utf-8")


def step_log(step: int) -> dict:
    return {
        "step": step,
        "loss": -0.01 * step,
        "grad_norm": 0.0 if step == 2 else 0.5,
        "learning_rate": expected_full_run_learning_rates()[step - 1],
        "reward": 3.0,
        "reward_std": 0.0 if step == 2 else 1.0,
        "frac_reward_zero_std": 1.0 if step == 2 else 0.0,
        "completions/mean_length": 103.5,
        "completions/max_length": 170.0 if step == 3 else 107.0,
        "completions/clipped_ratio": 0.125 if step == 3 else 0.0,
        "clip_ratio/low_mean": 0.01,
        "clip_ratio/high_mean": 0.02,
        "rewards/format_validity_reward/mean": 1.0,
        "rewards/format_validity_reward/std": 0.0,
        "rewards/vocab_rule_compliance_reward/mean": 1.0,
        "rewards/vocab_rule_compliance_reward/std": 0.0,
        "rewards/golden_agreement_reward/mean": 1.0 if step == 2 else 0.5,
        "rewards/golden_agreement_reward/std": 0.0 if step == 2 else 0.5,
    }


class FakeGRPOTrainer:
    steps_to_run = 300
    emit_save_callbacks = True

    def __init__(
        self,
        *,
        model,
        reward_funcs,
        args,
        train_dataset,
        processing_class,
    ):
        self.model = model
        self.reward_funcs = reward_funcs
        self.args = args
        self.train_dataset = train_dataset
        self.processing_class = processing_class
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_func_names = [reward.__name__ for reward in reward_funcs]
        self.reward_weights = list(args.reward_weights)
        self.optimizer = None
        self.lr_scheduler = None
        self._logs = {}
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _generate_and_score_completions(self, inputs):
        step = self.state.global_step + 1
        agreement = [float(index % 2) for index in range(8)]
        if step == 2:
            agreement = [1.0] * 8
        self._logs = {
            "completion": [f"step-{step}-completion-{index}" for index in range(8)],
            "advantages": [
                0.0 if step == 2 else (-1.0 if value == 0 else 1.0)
                for value in agreement
            ],
            "rewards": {
                "format_validity_reward": [1.0] * 8,
                "vocab_rule_compliance_reward": [1.0] * 8,
                "golden_agreement_reward": agreement,
            },
        }
        masks = [[1] * (80 + index) for index in range(8)]
        if step == 3:
            masks[4] = [0] * 170
        return {"completion_mask": masks}

    def _write_checkpoint(self, step: int):
        checkpoint = Path(self.args.output_dir) / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(
            f"checkpoint adapter {step}".encode()
        )
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"r": 16, "step": step}), encoding="utf-8"
        )
        (checkpoint / "training_args.bin").write_bytes(b"metadata")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "max_steps": 300,
                    "num_train_epochs": 1,
                    "train_batch_size": 8,
                    "logging_steps": 1,
                    "save_steps": 100,
                    "is_local_process_zero": True,
                    "is_world_process_zero": True,
                    "epoch": step / 1_565,
                    "log_history": [
                        step_log(index) for index in range(1, step + 1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        if step == 300:
            shutil.rmtree(Path(self.args.output_dir) / "checkpoint-100")

    def train(self):
        control = SimpleNamespace()
        for callback in self.callbacks:
            callback.on_train_begin(self.args, self.state, control)
        for step in range(1, self.steps_to_run + 1):
            sku_id = self.train_dataset.sku_ids[step - 1]
            self._generate_and_score_completions(
                [{"sku_id": sku_id} for _ in range(8)]
            )
            self.state.global_step = step
            self.state.log_history.append(step_log(step))
            for callback in self.callbacks:
                callback.on_log(self.args, self.state, control)
            if self.emit_save_callbacks and step in (100, 200, 300):
                self._write_checkpoint(step)
                for callback in self.callbacks:
                    callback.on_save(self.args, self.state, control)
        self.state.log_history.append(
            {"step": self.state.global_step, "train_runtime": 90.0}
        )
        for callback in self.callbacks:
            callback.on_log(self.args, self.state, control)
        self.model.updated = self.steps_to_run > 0
        self.optimizer = object()
        self.lr_scheduler = object()
        for callback in self.callbacks:
            callback.on_train_end(self.args, self.state, control)
        return SimpleNamespace(
            global_step=self.state.global_step,
            metrics={"train_loss": -1.5, "train_runtime": 90.0},
        )


class ShortFakeGRPOTrainer(FakeGRPOTrainer):
    steps_to_run = 299


class NoSaveFakeGRPOTrainer(FakeGRPOTrainer):
    emit_save_callbacks = False


def ordered_sku_sha256(sku_ids):
    return hashlib.sha256(("\n".join(sku_ids) + "\n").encode()).hexdigest()


def cuda_snapshot():
    return {
        "device_index": 0,
        "device_name": "Fake GPU",
        "device_total_bytes": 1_000,
        "driver_free_bytes": 800,
        "driver_used_bytes": 200,
        "torch_allocated_bytes": 10,
        "torch_reserved_bytes": 20,
        "torch_peak_allocated_bytes": 30,
        "torch_peak_reserved_bytes": 40,
    }


def orchestration_kwargs(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-adapter.safetensors"
    source.write_bytes(SOURCE_BYTES)
    output = tmp_path / "grpo-first-300"
    dataset = FakeDataset()
    values = {
        "base_trainer_class": FakeGRPOTrainer,
        "base_callback_class": FakeTrainerCallback,
        "config_class": FakeGRPOConfig,
        "model": FakeModel(),
        "tokenizer": FakeTokenizer(),
        "dataset": dataset,
        "reward_functions": REWARD_FUNCTIONS,
        "reward_weights": (1.0, 1.0, 2.0),
        "source_adapter_file": source,
        "final_output_dir": output,
        "preflight_report": {
            "status": "passed",
            "cuda_imports_performed": False,
            "output": {"path": str(output.resolve())},
            "pool": {
                "ordered_sku_sha256": ordered_sku_sha256(dataset.sku_ids)
            },
            "sft_lock": {"adapter_sha256": SOURCE_SHA},
            "disk": {"free_bytes": 4 * 1024**3},
        },
        "trainability_fn": lambda model: {
            "trainable_parameters": 18_464_768,
            "trainable_tensors": 392,
            "trainable_parameter_names": ["fake.lora.weight"],
        },
        "parameter_values_fn": lambda model: {
            "all_trainable_values_finite": True,
            "nonfinite_parameters": 0,
        },
        "fingerprint_fn": lambda model: (
            "b" * 64 if model.updated else "a" * 64
        ),
        "runtime_context": {
            "version": "grpo-full-run-runtime-context-v1",
            "runtime": {"torch_version": "fake"},
            "cuda_before_load": cuda_snapshot(),
            "cuda_after_load": cuda_snapshot(),
        },
        "cuda_snapshot_fn": cuda_snapshot,
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
        "expected_adapter_model_bytes": len(FINAL_WEIGHTS),
    }
    values.update(overrides)
    return values


def test_orchestration_constructs_trains_and_atomically_publishes(tmp_path):
    kwargs = orchestration_kwargs(tmp_path)

    report = run_full_run_300_orchestration(**kwargs)

    final = Path(kwargs["final_output_dir"])
    assert report["status"] == "passed"
    assert report["global_step"] == report["optimizer_steps"] == 300
    assert report["expected_rollouts"] == 2_400
    assert report["dataset_rows"] == 1_565
    assert report["reward_weights"] == [1.0, 1.0, 2.0]
    assert report["rollout_validation"]["records"] == 2_400
    assert report["trainer_log_validation"]["step_logs"] == 300
    assert set(report["checkpoint_evidence"]) == {100, 200, 300}
    assert report["lifecycle"]["event_count"] == 10
    assert report["lifecycle"]["status"] == "completed_and_published"
    assert report["publication"]["published_atomically"]
    assert report["published"]
    assert final.is_dir()
    assert not list(tmp_path.glob(".grpo-first-300.staging-*"))
    assert {path.name for path in (final / "trainer").iterdir()} == {
        "checkpoint-200",
        "checkpoint-300",
    }
    assert len((final / "rollouts.jsonl").read_text().splitlines()) == 2_400
    assert kwargs["model"].save_calls == 1


def test_orchestration_forwards_all_300_authoritative_progress_steps(tmp_path):
    progress = []
    kwargs = orchestration_kwargs(
        tmp_path,
        progress_callback=lambda **values: progress.append(values),
    )

    report = run_full_run_300_orchestration(**kwargs)

    assert report["status"] == "passed"
    assert len(progress) == 300
    assert progress[0] == {
        "optimizer_step": 1,
        "rollout_records": 8,
        "scalar_logs": 1,
    }
    assert progress[-1] == {
        "optimizer_step": 300,
        "rollout_records": 2_400,
        "scalar_logs": 300,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda values: values["preflight_report"].update(status="failed"), "passed preflight"),
        (lambda values: values.update(reward_weights=(1.0, 1.0, 1.0)), "weights drifted"),
        (lambda values: values["dataset"].sku_ids.pop(), "1,565 rows"),
        (
            lambda values: values["preflight_report"]["pool"].update(
                ordered_sku_sha256="0" * 64
            ),
            "SKU order disagrees",
        ),
    ],
)
def test_orchestration_rejects_contract_drift_before_staging(tmp_path, change, message):
    kwargs = orchestration_kwargs(tmp_path)
    change(kwargs)

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_full_run_300_orchestration(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()
    assert not list(tmp_path.glob(".grpo-first-300.staging-*"))


@pytest.mark.parametrize(
    ("trainer_class", "message"),
    [
        (ShortFakeGRPOTrainer, "did not end at step 300"),
        (NoSaveFakeGRPOTrainer, "did not process all checkpoints"),
    ],
)
def test_orchestration_refuses_incomplete_training_without_publication(
    tmp_path, trainer_class, message
):
    kwargs = orchestration_kwargs(tmp_path, base_trainer_class=trainer_class)

    with pytest.raises(RuntimeError, match=message):
        run_full_run_300_orchestration(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()
    assert kwargs["model"].save_calls == 0
    assert len(list(tmp_path.glob(".grpo-first-300.staging-*"))) == 1


def test_orchestration_refuses_unsafe_final_adapter_without_publication(tmp_path):
    model = FakeModel(final_weights=SOURCE_BYTES)
    kwargs = orchestration_kwargs(tmp_path, model=model)
    kwargs["expected_adapter_model_bytes"] = len(SOURCE_BYTES)

    with pytest.raises(ValueError, match="byte-identical"):
        run_full_run_300_orchestration(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()
    assert model.save_calls == 1
    assert len(list(tmp_path.glob(".grpo-first-300.staging-*"))) == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {
                "parameter_values_fn": lambda model: {
                    "all_trainable_values_finite": False,
                    "nonfinite_parameters": 1,
                }
            },
            "non-finite value",
        ),
        ({"fingerprint_fn": lambda model: "a" * 64}, "changed no trainable"),
    ],
)
def test_orchestration_audits_live_lora_before_adapter_save(
    tmp_path, override, message
):
    kwargs = orchestration_kwargs(tmp_path, **override)

    with pytest.raises(RuntimeError, match=message):
        run_full_run_300_orchestration(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()
    assert kwargs["model"].save_calls == 0
    assert len(list(tmp_path.glob(".grpo-first-300.staging-*"))) == 1
