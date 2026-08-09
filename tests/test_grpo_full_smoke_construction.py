from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_smoke_artifacts import EXPECTED_REWARD_NAMES
from training.train_grpo import (
    LOCKED_REWARD_WEIGHTS,
    LOCKED_TRAINABLE_PARAMETERS,
    LOCKED_TRAINABLE_TENSORS,
    SmokeRolloutCollector,
    construct_capturing_grpo_trainer,
    parse_args,
    run_five_step_smoke_gate,
)


SKUS = [f"sku-{index}" for index in range(1, 6)]


def format_validity_reward(*args, **kwargs):
    return []


def vocab_rule_compliance_reward(*args, **kwargs):
    return []


def golden_agreement_reward(*args, **kwargs):
    return []


REWARD_FUNCTIONS = (
    format_validity_reward,
    vocab_rule_compliance_reward,
    golden_agreement_reward,
)


class FakeDataset:
    def __init__(self, sku_ids=None, columns=None):
        self.sku_ids = list(SKUS if sku_ids is None else sku_ids)
        self.column_names = list(
            ["prompt", "gold", "sku_id"] if columns is None else columns
        )

    def __len__(self):
        return len(self.sku_ids)

    def __getitem__(self, key):
        if key == "sku_id":
            return list(self.sku_ids)
        raise KeyError(key)


class FakeGRPOConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.report_to = [] if self.report_to == "none" else self.report_to
        self.generation_batch_size = 8


class FakeGRPOTrainer:
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
        self.optimizer = None
        self.lr_scheduler = None
        self.ref_model = None
        self.state = SimpleNamespace(global_step=0)

    def _generate_and_score_completions(self, inputs):
        raise AssertionError("construction test must not generate")


class FakeCuda:
    def empty_cache(self):
        return None

    def reset_peak_memory_stats(self):
        return None

    def synchronize(self):
        return None

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, _index):
        return SimpleNamespace(name="Fake RTX 3090", total_memory=24 * 1024**3)

    def mem_get_info(self, _index):
        return 20 * 1024**3, 24 * 1024**3

    def memory_allocated(self, _index):
        return 3_000_000_000

    def memory_reserved(self, _index):
        return 3_100_000_000

    def max_memory_allocated(self, _index):
        return 4_000_000_000

    def max_memory_reserved(self, _index):
        return 4_100_000_000


class FakeGlobalOptimManager:
    instance = SimpleNamespace(module_weight_config_triple=["existing"])

    @classmethod
    def get_instance(cls):
        return cls.instance


def runtime_stack(dataset: FakeDataset | None = None) -> dict:
    active_dataset = dataset or FakeDataset()

    def load_pack(path):
        assert path == "packs/vastraa_taste_v1"
        return "fake-pack"

    def load_grpo_prompts(pack, path, *, require_pass_rate_band):
        assert pack == "fake-pack"
        assert Path(path).name == "fixture.jsonl"
        assert require_pass_rate_band is True
        return active_dataset

    return {
        "FastLanguageModel": object(),
        "torch": SimpleNamespace(cuda=FakeCuda()),
        "GRPOConfig": FakeGRPOConfig,
        "GRPOTrainer": FakeGRPOTrainer,
        "GlobalOptimManager": FakeGlobalOptimManager,
        "load_grpo_prompts": load_grpo_prompts,
        "load_pack": load_pack,
        "reward_functions": REWARD_FUNCTIONS,
        "reward_weights": LOCKED_REWARD_WEIGHTS,
    }


def make_gate_kwargs(tmp_path: Path):
    adapter = tmp_path / "checkpoint-406"
    adapter.mkdir(parents=True)
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"locked source adapter")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("fixture\n", encoding="utf-8")
    final = tmp_path / "grpo-first-smoke"
    preflight = {
        "fixture": {"sku_ids_in_step_order": list(SKUS)},
        "sft_lock": {
            "adapter_sha256": hashlib.sha256(
                adapter_file.read_bytes()
            ).hexdigest()
        },
    }
    model = object()
    tokenizer = object()
    observed = {}

    def policy_loader(fast_language_model, torch_module, adapter_path):
        assert fast_language_model is not None
        assert torch_module.cuda.is_available()
        assert adapter_path == adapter.resolve()
        return model, tokenizer

    def trainability(active_model):
        assert active_model is model
        return {
            "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
            "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
        }

    def orchestrate(**kwargs):
        observed.update(kwargs)
        assert isinstance(
            kwargs["trainer"].smoke_rollout_collector,
            SmokeRolloutCollector,
        )
        assert kwargs["trainer"].state.global_step == 0
        assert kwargs["expected_sku_ids"] == SKUS
        assert kwargs["config_settings"]["max_steps"] == 5
        assert kwargs["config_settings"]["save_strategy"] == "no"
        kwargs["final_output_dir"].mkdir()
        return {
            "status": "passed",
            "published": True,
            "manifest": {"status": "completed"},
        }

    kwargs = {
        "preflight_report": preflight,
        "fixture_data_path": fixture,
        "adapter_path": adapter,
        "adapter_file": adapter_file,
        "final_output_dir": final,
        "expected_sku_ids": SKUS,
        "runtime_loader": runtime_stack,
        "policy_loader_fn": policy_loader,
        "trainability_fn": trainability,
        "orchestration_fn": orchestrate,
    }
    return kwargs, observed


def test_construct_capturing_trainer_locks_config_rewards_and_dataset(tmp_path):
    trainer, report = construct_capturing_grpo_trainer(
        base_trainer_class=FakeGRPOTrainer,
        config_class=FakeGRPOConfig,
        model=object(),
        tokenizer=object(),
        dataset=FakeDataset(),
        expected_sku_ids=SKUS,
        reward_functions=REWARD_FUNCTIONS,
        reward_weights=LOCKED_REWARD_WEIGHTS,
        temporary_output_dir=tmp_path,
    )

    assert isinstance(trainer.smoke_rollout_collector, SmokeRolloutCollector)
    assert report["collector_attached"]
    assert report["reward_names"] == list(EXPECTED_REWARD_NAMES)
    assert report["reward_weights"] == [1.0, 1.0, 2.0]
    assert report["sku_ids_in_step_order"] == SKUS
    assert report["config"]["settings"]["max_steps"] == 5
    assert report["matches_locked_contract"]

    with pytest.raises(RuntimeError, match="reward function names drifted"):
        construct_capturing_grpo_trainer(
            base_trainer_class=FakeGRPOTrainer,
            config_class=FakeGRPOConfig,
            model=object(),
            tokenizer=object(),
            dataset=FakeDataset(),
            expected_sku_ids=SKUS,
            reward_functions=(lambda: None, *REWARD_FUNCTIONS[1:]),
            reward_weights=LOCKED_REWARD_WEIGHTS,
            temporary_output_dir=tmp_path,
        )


def test_real_gate_bridge_constructs_then_calls_orchestration_cpu_only(tmp_path):
    kwargs, observed = make_gate_kwargs(tmp_path)

    report = run_five_step_smoke_gate(**kwargs)

    assert report["status"] == "passed"
    assert report["version"] == "grpo-five-step-smoke-gate-v1"
    assert report["construction"]["collector_attached"]
    assert report["total_gate_seconds_before_release"] >= (
        report["construction_seconds_including_model_load"]
    )
    assert report["temporary_output_removed"]
    assert report["global_optimizer_manager_overrides_removed"]
    assert report["trainer_retained"] is False
    assert report["model_retained"] is False
    assert observed["tokenizer"] is not None
    assert Path(kwargs["final_output_dir"]).is_dir()


def test_real_gate_bridge_rejects_dataset_and_source_drift(tmp_path):
    kwargs, _observed = make_gate_kwargs(tmp_path / "order")
    kwargs["runtime_loader"] = lambda: runtime_stack(
        FakeDataset(sku_ids=list(reversed(SKUS)))
    )
    with pytest.raises(RuntimeError, match="dataset SKU order drifted"):
        run_five_step_smoke_gate(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()

    kwargs, _observed = make_gate_kwargs(tmp_path / "source")

    def corrupting_policy_loader(_fast, _torch, adapter_path):
        (adapter_path / "adapter_model.safetensors").write_bytes(b"changed")
        return object(), object()

    kwargs["policy_loader_fn"] = corrupting_policy_loader
    kwargs["trainability_fn"] = lambda _model: {
        "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
        "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
    }
    with pytest.raises(RuntimeError, match="source adapter changed"):
        run_five_step_smoke_gate(**kwargs)
    assert not Path(kwargs["final_output_dir"]).exists()


def test_full_smoke_cli_mode_is_mutually_exclusive():
    assert parse_args(["--five-step-smoke"]).five_step_smoke
    with pytest.raises(SystemExit):
        parse_args(["--five-step-smoke", "--one-update-only"])
