"""CPU fakes for the real-stack 300-step no-training construction gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.train_grpo import (
    FULL_RUN_CONSTRUCTION_GATE_VERSION,
    run_full_run_300_construction_gate,
)

SOURCE_BYTES = b"locked construction-gate adapter"
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


class FakeParameter:
    def __init__(self):
        self.value = np.array([1.0, 2.0], dtype=np.float32)
        self.requires_grad = True
        self.dtype = "float32"
        self.shape = self.value.shape
        self.grad = None

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.parameter = FakeParameter()

    def named_parameters(self):
        return [("layer.q_proj.lora_A.weight", self.parameter)]

    def parameters(self):
        return [self.parameter]


class FakeDataset:
    column_names = ["prompt", "gold", "sku_id"]

    def __init__(self, rows=1_565):
        self.skus = [f"sku-{index:04d}" for index in range(rows)]

    def __len__(self):
        return len(self.skus)

    def __getitem__(self, key):
        if key == "sku_id":
            return list(self.skus)
        raise KeyError(key)


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.generation_batch_size = 8


class FakeCallback:
    pass


class FakeAccelerator:
    num_processes = 1

    def backward(self, loss):
        raise AssertionError("construction gate must not run backward")


class FakeTrainer:
    construct_optimizer = False

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
        self.reward_func_names = [reward.__name__ for reward in reward_funcs]
        self.reward_weights = list(args.reward_weights)
        self.optimizer = object() if self.construct_optimizer else None
        self.lr_scheduler = None
        self.ref_model = None
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.accelerator = FakeAccelerator()
        self.callback_handler = SimpleNamespace(callbacks=[])

    def add_callback(self, callback):
        self.callback_handler.callbacks.append(callback)

    def _generate(self):
        raise AssertionError("construction gate must not generate")

    def _calculate_rewards(self):
        raise AssertionError("construction gate must not score rewards")

    def compute_loss(self):
        raise AssertionError("construction gate must not compute loss")

    def _generate_and_score_completions(self, inputs):
        raise AssertionError("construction gate must not generate")


class OptimizerCreatingTrainer(FakeTrainer):
    construct_optimizer = True


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


def ordered_sku_sha256(skus):
    return hashlib.sha256(("\n".join(skus) + "\n").encode()).hexdigest()


def make_inputs(tmp_path, *, trainer_class=FakeTrainer, rows=1_565):
    training_data = tmp_path / "train.jsonl"
    training_data.write_text("fixture\n", encoding="utf-8")
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_file.write_bytes(SOURCE_BYTES)
    dataset = FakeDataset(rows=rows)
    torch = SimpleNamespace(cuda=FakeCuda())
    model = FakeModel()
    tokenizer = object()

    def runtime_loader():
        return {
            "FastLanguageModel": object(),
            "torch": torch,
            "TrainerCallback": FakeCallback,
            "GRPOConfig": FakeConfig,
            "GRPOTrainer": trainer_class,
            "load_grpo_prompts": lambda pack, path, require_pass_rate_band: dataset,
            "load_pack": lambda name: {"name": name},
            "reward_functions": REWARDS,
            "reward_weights": (1.0, 1.0, 2.0),
        }

    values = {
        "training_data_path": training_data,
        "adapter_path": adapter_path,
        "adapter_file": adapter_file,
        "expected_ordered_sku_sha256": ordered_sku_sha256(dataset.skus),
        "expected_adapter_sha256": SOURCE_SHA,
        "runtime_loader": runtime_loader,
        "policy_loader_fn": lambda fast, torch_module, path: (model, tokenizer),
        "trainability_fn": lambda active_model: {
            "trainable_parameters": 18_464_768,
            "trainable_tensors": 392,
            "only_lora_parameters_trainable": True,
        },
        "cuda_snapshot_fn": lambda torch_module: {
            "torch_allocated_bytes": 1_000,
            "torch_reserved_bytes": 2_000,
        },
    }
    return values, dataset, torch


def test_full_run_construction_gate_builds_everything_without_training(tmp_path):
    kwargs, dataset, torch = make_inputs(tmp_path)

    report = run_full_run_300_construction_gate(**kwargs)

    assert report["version"] == FULL_RUN_CONSTRUCTION_GATE_VERSION
    assert report["status"] == "passed"
    assert report["dataset_rows"] == 1_565
    assert report["dataset_ordered_sku_sha256"] == ordered_sku_sha256(dataset.skus)
    assert report["config"]["settings_match_locked_contract"]
    assert report["collector_attached"]
    assert report["checkpoint_callback_attached"]
    assert report["phase_profiler_attached"]
    assert report["phase_profiler_steps"] == 0
    assert report["lifecycle_writer_constructed"]
    assert report["lifecycle_events"] == 0
    assert report["global_step"] == report["rollouts_generated"] == 0
    assert not report["optimizer_constructed"]
    assert not report["scheduler_constructed"]
    assert not report["training_dispatched"]
    assert report["trainable_lora_unchanged"]
    assert report["temporary_output_removed"]
    assert not Path(report["temporary_output_root"]).exists()
    assert not report["model_retained"]
    assert not report["trainer_retained"]
    assert torch.cuda.empty_cache_calls == 2
    assert torch.cuda.synchronize_calls == 2


def test_construction_gate_rejects_dataset_order_drift(tmp_path):
    kwargs, dataset, _torch = make_inputs(tmp_path)
    kwargs["expected_ordered_sku_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="dataset order drifted"):
        run_full_run_300_construction_gate(**kwargs)


def test_construction_gate_rejects_wrong_dataset_size(tmp_path):
    kwargs, _dataset, _torch = make_inputs(tmp_path, rows=1_564)

    with pytest.raises(RuntimeError, match="has 1564 rows"):
        run_full_run_300_construction_gate(**kwargs)


def test_construction_gate_rejects_eager_optimizer_creation(tmp_path):
    kwargs, _dataset, _torch = make_inputs(
        tmp_path, trainer_class=OptimizerCreatingTrainer
    )

    with pytest.raises(RuntimeError, match="created an optimizer"):
        run_full_run_300_construction_gate(**kwargs)


def test_construction_gate_rejects_source_adapter_drift_before_imports(tmp_path):
    kwargs, _dataset, _torch = make_inputs(tmp_path)
    kwargs["adapter_file"].write_bytes(b"drifted")
    called = False

    def forbidden_runtime():
        nonlocal called
        called = True
        raise AssertionError("runtime should not load")

    kwargs["runtime_loader"] = forbidden_runtime
    with pytest.raises(RuntimeError, match="source adapter disagrees"):
        run_full_run_300_construction_gate(**kwargs)
    assert not called


def test_construction_gate_rejects_runtime_reward_drift(tmp_path):
    kwargs, _dataset, _torch = make_inputs(tmp_path)
    original_loader = kwargs["runtime_loader"]

    def drifted_runtime():
        runtime = original_loader()
        runtime["reward_weights"] = (1.0, 1.0, 1.0)
        return runtime

    kwargs["runtime_loader"] = drifted_runtime
    with pytest.raises(RuntimeError, match="reward weights drifted"):
        run_full_run_300_construction_gate(**kwargs)
