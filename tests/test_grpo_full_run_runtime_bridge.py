"""CPU-only tests for the real-runtime-to-orchestration bridge."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.train_grpo import (
    FULL_RUN_RUNTIME_BRIDGE_VERSION,
    run_full_run_300_gate,
)

SOURCE_BYTES = b"locked runtime-bridge adapter"
SOURCE_SHA = hashlib.sha256(SOURCE_BYTES).hexdigest()


def format_validity_reward(*args, **kwargs):
    return [1.0]


def vocab_rule_compliance_reward(*args, **kwargs):
    return [1.0]


def golden_agreement_reward(*args, **kwargs):
    return [1.0]


REWARDS = (
    format_validity_reward,
    vocab_rule_compliance_reward,
    golden_agreement_reward,
)


class FakeDataset:
    column_names = ["prompt", "gold", "sku_id"]

    def __init__(self):
        self.skus = [f"sku-{index:04d}" for index in range(1_565)]

    def __len__(self):
        return len(self.skus)

    def __getitem__(self, key):
        if key == "sku_id":
            return list(self.skus)
        raise KeyError(key)


class FakeModel:
    def __init__(self):
        self.updated = False


class FakeCuda:
    def __init__(self):
        self.empty_cache_calls = 0
        self.synchronize_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self):
        pass

    def synchronize(self):
        self.synchronize_calls += 1


class FakeGlobalOptimManager:
    instance = SimpleNamespace(module_weight_config_triple=[])

    @classmethod
    def get_instance(cls):
        return cls.instance


class FakeTrainer:
    pass


class FakeCallback:
    pass


class FakeConfig:
    pass


def ordered_sku_sha256(skus):
    return hashlib.sha256(("\n".join(skus) + "\n").encode()).hexdigest()


def make_bridge(tmp_path, *, orchestration_status="passed"):
    training_data = tmp_path / "train.jsonl"
    training_data.write_text("fixture\n", encoding="utf-8")
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_file.write_bytes(SOURCE_BYTES)
    final_output = tmp_path / "grpo-first-300"
    dataset = FakeDataset()
    model = FakeModel()
    torch = SimpleNamespace(cuda=FakeCuda())
    observed = {}
    FakeGlobalOptimManager.instance.module_weight_config_triple[:] = ["existing"]

    def runtime_loader():
        return {
            "FastLanguageModel": object(),
            "torch": torch,
            "TrainerCallback": FakeCallback,
            "GRPOConfig": FakeConfig,
            "GRPOTrainer": FakeTrainer,
            "GlobalOptimManager": FakeGlobalOptimManager,
            "load_grpo_prompts": lambda pack, path, require_pass_rate_band: dataset,
            "load_pack": lambda name: {"name": name},
            "reward_functions": REWARDS,
            "reward_weights": (1.0, 1.0, 2.0),
        }

    def orchestration_fn(**kwargs):
        observed.update(kwargs)
        FakeGlobalOptimManager.instance.module_weight_config_triple.append("new")
        if orchestration_status == "raises":
            raise RuntimeError("synthetic orchestration failure")
        if orchestration_status == "passed":
            model.updated = True
            final_output.mkdir()
            return {
                "status": "passed",
                "published": True,
                "global_step": 300,
            }
        return {"status": orchestration_status, "published": False}

    preflight = {
        "status": "passed",
        "cuda_imports_performed": False,
        "pool": {
            "data_path": str(training_data.resolve()),
            "ordered_sku_sha256": ordered_sku_sha256(dataset.skus),
        },
        "sft_lock": {
            "adapter_path": str(adapter_path.resolve()),
            "adapter_file": str(adapter_file.resolve()),
            "adapter_sha256": SOURCE_SHA,
        },
        "output": {"path": str(final_output.resolve())},
        "disk": {"free_bytes": 4 * 1024**3},
    }
    kwargs = {
        "preflight_report": preflight,
        "training_data_path": training_data,
        "adapter_path": adapter_path,
        "adapter_file": adapter_file,
        "final_output_dir": final_output,
        "runtime_loader": runtime_loader,
        "policy_loader_fn": lambda fast, torch_module, path: (model, object()),
        "trainability_fn": lambda active_model: {
            "trainable_parameters": 18_464_768,
            "trainable_tensors": 392,
        },
        "parameter_values_fn": lambda active_model: {
            "all_trainable_values_finite": True,
            "nonfinite_parameters": 0,
        },
        "fingerprint_fn": lambda active_model: (
            "b" * 64 if active_model.updated else "a" * 64
        ),
        "cuda_snapshot_fn": lambda torch_module: {
            "torch_allocated_bytes": 1_000,
            "torch_reserved_bytes": 2_000,
        },
        "orchestration_fn": orchestration_fn,
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
        "expected_adapter_model_bytes": 123,
    }
    return kwargs, observed, model, torch


def test_runtime_bridge_loads_real_interfaces_and_invokes_orchestration(tmp_path):
    kwargs, observed, model, torch = make_bridge(tmp_path)
    progress_callback = lambda **values: values
    kwargs["progress_callback"] = progress_callback

    report = run_full_run_300_gate(**kwargs)

    assert report["version"] == FULL_RUN_RUNTIME_BRIDGE_VERSION
    assert report["status"] == "passed"
    assert report["dataset_rows"] == 1_565
    assert report["training_dispatched"]
    assert report["published"]
    assert report["trainable_lora_changed"]
    assert report["source_adapter_unchanged"]
    assert report["global_optimizer_manager_overrides_before"] == 1
    assert report["global_optimizer_manager_overrides_removed"]
    assert FakeGlobalOptimManager.instance.module_weight_config_triple == ["existing"]
    assert report["model_retained"] is False
    assert report["trainer_retained"] is False
    assert torch.cuda.empty_cache_calls == 2
    assert torch.cuda.synchronize_calls == 1

    assert observed["base_trainer_class"] is FakeTrainer
    assert observed["base_callback_class"] is FakeCallback
    assert observed["config_class"] is FakeConfig
    assert observed["model"] is model
    assert observed["dataset"] is not None
    assert observed["reward_functions"] == REWARDS
    assert observed["reward_weights"] == (1.0, 1.0, 2.0)
    assert observed["trainability_fn"] is kwargs["trainability_fn"]
    assert observed["parameter_values_fn"] is kwargs["parameter_values_fn"]
    assert observed["fingerprint_fn"] is kwargs["fingerprint_fn"]
    assert observed["runtime_context"]["version"] == (
        "grpo-full-run-runtime-context-v1"
    )
    assert callable(observed["cuda_snapshot_fn"])
    assert observed["progress_callback"] is progress_callback
    assert observed["expected_adapter_model_bytes"] == 123


def test_runtime_bridge_releases_global_state_when_orchestration_fails(tmp_path):
    kwargs, _observed, _model, torch = make_bridge(
        tmp_path, orchestration_status="raises"
    )

    with pytest.raises(RuntimeError, match="synthetic orchestration failure"):
        run_full_run_300_gate(**kwargs)
    assert FakeGlobalOptimManager.instance.module_weight_config_triple == ["existing"]
    assert torch.cuda.empty_cache_calls == 2
    assert torch.cuda.synchronize_calls == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values["preflight_report"].update(status="failed"), "passed preflight"),
        (
            lambda values: values["preflight_report"]["output"].update(
                path="/tmp/wrong-output"
            ),
            "output path disagrees",
        ),
        (
            lambda values: values["preflight_report"]["pool"].update(
                ordered_sku_sha256="0" * 64
            ),
            "dataset order drifted",
        ),
    ],
)
def test_runtime_bridge_rejects_preflight_lineage_drift(tmp_path, mutate, message):
    kwargs, _observed, _model, _torch = make_bridge(tmp_path)
    mutate(kwargs)

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_full_run_300_gate(**kwargs)


def test_runtime_bridge_refuses_false_orchestration_success(tmp_path):
    kwargs, _observed, _model, _torch = make_bridge(
        tmp_path, orchestration_status="failed"
    )

    with pytest.raises(RuntimeError, match="did not publish success"):
        run_full_run_300_gate(**kwargs)
