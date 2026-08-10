#!/usr/bin/env python3
"""Guarded entry point for staged GRPO integration gates.

Importing this module deliberately performs no Torch, Transformers, TRL, PEFT,
Unsloth or vLLM import. GPU-capable modes import their stack only after
``run_preflight`` succeeds. The five-step mode additionally requires its exact
commit, output-path, disk-floor and atomic-publication launch controls. The
300-step mode currently validates its launch arguments and then stops before
preflight or CUDA; training dispatch is intentionally a later gate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

from training.grpo_smoke_artifacts import (
    EXPECTED_ADAPTER_MODEL_BYTES,
    EXPECTED_REWARD_NAMES,
    GENERATIONS_PER_STEP,
    EXPECTED_ROLLOUTS,
    EXPECTED_STEPS,
    create_staging_output,
    validate_smoke_context,
    validate_rollout_records,
    validate_trainer_log_history,
    write_and_publish_smoke_bundle,
)
from training.grpo_full_run_artifacts import (
    FULL_RUN_LIFECYCLE_VERSION,
    FULL_RUN_SUMMARY_VERSION,
    FullRunCheckpointLifecycleWriter,
    build_full_run_lifecycle_plan,
    create_full_run_staging_output,
)
from training.grpo_full_run_evidence import (
    validate_full_run_rollout_records,
    validate_full_run_trainer_log_history,
)
from training.grpo_phase_profiler import (
    FullRunPhaseProfiler,
    PROFILE_PHASES,
    make_phase_profiler_callback_class,
)

PREFLIGHT_VERSION = "grpo-smoke-preflight-v1"
MODEL_LOAD_VERSION = "grpo-smoke-model-load-v1"
TRAINER_CONSTRUCTION_VERSION = "grpo-smoke-trainer-construction-v1"
ROLLOUT_GATE_VERSION = "grpo-smoke-rollout-gate-v1"
GRADIENT_GATE_VERSION = "grpo-smoke-gradient-gate-v1"
OPTIMIZER_CONSTRUCTION_VERSION = "grpo-smoke-optimizer-construction-v1"
ONE_UPDATE_GATE_VERSION = "grpo-smoke-one-update-gate-v1"
FIVE_STEP_SMOKE_GATE_VERSION = "grpo-five-step-smoke-gate-v1"
FULL_RUN_300_CONTRACT_VERSION = "grpo-full-run-300-contract-v1"
FULL_RUN_CONSTRUCTION_GATE_VERSION = "grpo-full-run-construction-gate-v1"
FULL_RUN_RUNTIME_BRIDGE_VERSION = "grpo-full-run-runtime-bridge-v1"
FULL_RUN_RUNTIME_CONTEXT_VERSION = "grpo-full-run-runtime-context-v1"
DEFAULT_FIXTURE_DATA = "data/train_weak_grpo_smoke_v1.jsonl"
DEFAULT_FIXTURE_MANIFEST = "data/splits/grpo-smoke-v1.json"
DEFAULT_FULL_RUN_DATA = "data/train_weak_grpo_cap4.jsonl"
DEFAULT_FULL_RUN_MANIFEST = (
    "runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json"
)
DEFAULT_SELECTION_MANIFEST = "runs/sft-selection.json"
DEFAULT_ADAPTER = "runs/sft-combined-2epoch/checkpoint-406"
DEFAULT_OUTPUT_DIR = "runs/grpo-first-smoke"
DEFAULT_FULL_RUN_OUTPUT_DIR = "runs/grpo-first-300"
DEFAULT_MINIMUM_FREE_GIB = 3.0
MODEL_MAX_SEQUENCE_LENGTH = 896
LOCKED_REWARD_WEIGHTS = (1.0, 1.0, 2.0)
LOCKED_OPTIMIZER_NAME = "adamw_8bit"
LOCKED_LEARNING_RATE = 5e-6
LOCKED_WEIGHT_DECAY = 0.001
LOCKED_ADAM_BETAS = (0.9, 0.999)
LOCKED_ADAM_EPSILON = 1e-8
LOCKED_WARMUP_RATIO = 0.0
LOCKED_MAX_GRAD_NORM = 1.0
FULL_RUN_STEPS = 300
FULL_RUN_WARMUP_RATIO = 0.1
FULL_RUN_SAVE_STEPS = 100
FULL_RUN_SAVE_TOTAL_LIMIT = 2
FULL_RUN_CHECKPOINT_STEPS = (100, 200, 300)
FULL_RUN_DATA_ROWS = 1_565
FULL_RUN_DATA_BYTES = 3_467_347
FULL_RUN_MAX_IDLE_GPU_MEMORY_MIB = 1_024
FULL_RUN_MAX_IDLE_GPU_UTILIZATION_PERCENT = 5

LOCKED_FIXTURE_DATA_SHA256 = (
    "268373ceb08c53125976493340d972a47c90e10911e919002716590f75ca4084"
)
LOCKED_FIXTURE_MANIFEST_SHA256 = (
    "e898510534d967b9a35367e0aba5a564e6cb564e2c326d45804d0319d528dd05"
)
LOCKED_FULL_RUN_DATA_SHA256 = (
    "3e378187a8147923bae1e0753a750d6e252336e911fa8c91cd57a4a8ddc3a102"
)
LOCKED_FULL_RUN_MANIFEST_SHA256 = (
    "d166325a0c4ef3d78023ba492881fb3971e290b1b3606ee4ac8cd6aa733175e0"
)
LOCKED_SELECTION_MANIFEST_SHA256 = (
    "e425635d323b3ffe9e7350fb61a2d9e1848345a95abab6b92032bf64d2718299"
)
LOCKED_ADAPTER_SHA256 = (
    "00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af"
)
LOCKED_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
LOCKED_TRAINABLE_PARAMETERS = 18_464_768
LOCKED_TRAINABLE_TENSORS = 392
LOCKED_LORA_RANK = 16
LOCKED_LORA_ALPHA = 16
LOCKED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def grpo_smoke_config_kwargs(
    *,
    output_dir: str | Path,
    reward_weights: Sequence[float] = LOCKED_REWARD_WEIGHTS,
) -> dict:
    """Return the complete, auditable five-step smoke configuration contract."""
    normalized_weights = tuple(float(weight) for weight in reward_weights)
    if normalized_weights != LOCKED_REWARD_WEIGHTS:
        raise ValueError(
            f"reward weights must remain locked at {LOCKED_REWARD_WEIGHTS}"
        )
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "run_name": "grpo-first-smoke",
        "seed": 42,
        "data_seed": 42,
        "max_prompt_length": 600,
        "max_completion_length": 170,
        "num_generations": 8,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": LOCKED_MAX_GRAD_NORM,
        "steps_per_generation": 1,
        "max_steps": 5,
        "shuffle_dataset": False,
        "remove_unused_columns": False,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "use_vllm": False,
        "learning_rate": LOCKED_LEARNING_RATE,
        "weight_decay": LOCKED_WEIGHT_DECAY,
        "adam_beta1": LOCKED_ADAM_BETAS[0],
        "adam_beta2": LOCKED_ADAM_BETAS[1],
        "adam_epsilon": LOCKED_ADAM_EPSILON,
        "warmup_ratio": LOCKED_WARMUP_RATIO,
        "lr_scheduler_type": "cosine",
        "optim": LOCKED_OPTIMIZER_NAME,
        "beta": 0.0,
        "num_iterations": 1,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "scale_rewards": "group",
        "loss_type": "dapo",
        "mask_truncated_completions": True,
        "reward_weights": list(normalized_weights),
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "log_completions": True,
        "num_completions_to_print": 8,
        "report_to": "none",
        "save_strategy": "no",
        "save_only_model": True,
    }


def grpo_full_run_300_config_kwargs(
    *,
    output_dir: str | Path,
    reward_weights: Sequence[float] = LOCKED_REWARD_WEIGHTS,
) -> dict:
    """Return the locked trainer settings for the first 300-step GRPO run.

    The proven smoke contract remains unchanged.  This longer-run contract
    deliberately changes only duration, prompt shuffling, warmup, completion
    table printing, and checkpoint persistence.
    """
    config = grpo_smoke_config_kwargs(
        output_dir=output_dir,
        reward_weights=reward_weights,
    )
    config.update(
        {
            "run_name": "grpo-first-300",
            "max_steps": FULL_RUN_STEPS,
            "shuffle_dataset": True,
            "warmup_ratio": FULL_RUN_WARMUP_RATIO,
            # Scalar metrics still log every step. Avoid printing 2,400 full
            # completions into the detached process log; rollout artifacts are
            # captured separately by the audited collector.
            "log_completions": False,
            "save_strategy": "steps",
            "save_steps": FULL_RUN_SAVE_STEPS,
            "save_total_limit": FULL_RUN_SAVE_TOTAL_LIMIT,
        }
    )
    return config


def full_run_300_contract(*, output_dir: str | Path) -> dict:
    """Describe the CPU-auditable launch contract without enabling training."""
    resolved_output = Path(output_dir).resolve()
    config = grpo_full_run_300_config_kwargs(output_dir=resolved_output)
    checkpoint_steps = tuple(
        range(FULL_RUN_SAVE_STEPS, FULL_RUN_STEPS + 1, FULL_RUN_SAVE_STEPS)
    )
    if checkpoint_steps != FULL_RUN_CHECKPOINT_STEPS:
        raise RuntimeError("300-step checkpoint schedule drifted")

    return {
        "version": FULL_RUN_300_CONTRACT_VERSION,
        "status": "locked_not_launchable",
        "training": {
            "optimizer_steps": FULL_RUN_STEPS,
            "generations_per_step": config["num_generations"],
            "expected_rollouts": FULL_RUN_STEPS * config["num_generations"],
            "prompt_data": DEFAULT_FULL_RUN_DATA,
            "prompt_order": "seeded_shuffle",
            "seed": config["seed"],
            "data_seed": config["data_seed"],
            "warmup_steps": math.ceil(
                FULL_RUN_STEPS * FULL_RUN_WARMUP_RATIO
            ),
        },
        "checkpointing": {
            "lifecycle_version": FULL_RUN_LIFECYCLE_VERSION,
            "events": list(checkpoint_steps),
            "save_total_limit": FULL_RUN_SAVE_TOTAL_LIMIT,
            "save_only_model": config["save_only_model"],
            "optimizer_state_saved": not config["save_only_model"],
            "step_100_evidence_required_before_eviction": True,
            "final_adapter_directory": str(resolved_output / "final-adapter"),
            "final_adapter_must_be_retained": True,
        },
        "reporting": {
            "scalar_logging_steps": config["logging_steps"],
            "completion_tables_printed": config["log_completions"],
            "local_trainer_log_required": True,
            "external_reporting_required": False,
            "report_to": config["report_to"],
            "phase_timing_required": True,
            "phase_timing_boundaries": list(PROFILE_PHASES),
            "phase_timing_cuda_synchronized": True,
            "phase_timing_durable_steps": list(FULL_RUN_CHECKPOINT_STEPS),
            "phase_timing_observer_effect_recorded": True,
        },
        "resources": {
            "minimum_free_gib_before_launch": DEFAULT_MINIMUM_FREE_GIB,
            "minimum_free_bytes_before_launch": int(
                DEFAULT_MINIMUM_FREE_GIB * 1024**3
            ),
            "detached_launch_required": True,
        },
        "config": config,
    }


def inspect_grpo_config(config: object) -> dict:
    """Assert TRL preserved the locked smoke settings after normalization."""
    expected = grpo_smoke_config_kwargs(output_dir=getattr(config, "output_dir"))
    normalized = {}
    for key, expected_value in expected.items():
        actual = getattr(config, key, None)
        if key == "report_to":
            # Transformers normalizes the user-facing "none" value to [].
            if actual not in ("none", [], ()):  # pragma: no branch - explicit forms
                raise RuntimeError(
                    f"GRPO config normalized {key} unexpectedly: {actual}"
                )
            normalized[key] = [] if actual != "none" else "none"
            continue
        if key == "reward_weights":
            actual = list(actual) if actual is not None else None
        if actual != expected_value:
            raise RuntimeError(
                f"GRPO config drift for {key}: {actual!r} != {expected_value!r}"
            )
        normalized[key] = actual

    generation_batch_size = getattr(config, "generation_batch_size", None)
    if generation_batch_size != 8:
        raise RuntimeError(
            f"generation batch size must be 8, found {generation_batch_size}"
        )
    return {
        "settings": normalized,
        "generation_batch_size": generation_batch_size,
        "prompts_per_generation_batch": generation_batch_size
        // expected["num_generations"],
        "settings_match_locked_contract": True,
    }


def inspect_grpo_full_run_config(config: object) -> dict:
    """Assert TRL preserved every normalized 300-step trainer setting."""
    expected = grpo_full_run_300_config_kwargs(
        output_dir=getattr(config, "output_dir")
    )
    normalized = {}
    for key, expected_value in expected.items():
        actual = getattr(config, key, None)
        if key == "report_to":
            if actual not in ("none", [], ()):
                raise RuntimeError(
                    f"full-run config normalized {key} unexpectedly: {actual}"
                )
            normalized[key] = [] if actual != "none" else "none"
            continue
        if key == "reward_weights":
            actual = list(actual) if actual is not None else None
        if actual != expected_value:
            raise RuntimeError(
                f"full-run config drift for {key}: "
                f"{actual!r} != {expected_value!r}"
            )
        normalized[key] = actual

    generation_batch_size = getattr(config, "generation_batch_size", None)
    if generation_batch_size != GENERATIONS_PER_STEP:
        raise RuntimeError(
            "full-run generation batch size must be 8, "
            f"found {generation_batch_size}"
        )
    return {
        "settings": normalized,
        "generation_batch_size": generation_batch_size,
        "settings_match_locked_contract": True,
    }


def build_rollout_evidence(
    *,
    sku_id: str,
    completions: Sequence[str],
    reward_names: Sequence[str],
    component_rewards: dict[str, Sequence[float]],
    reward_weights: Sequence[float],
    advantages: Sequence[float],
    effective_completion_tokens: Sequence[int],
    truncated_and_masked: Sequence[bool],
) -> dict:
    """Validate and assemble one auditable eight-completion rollout group."""
    expected = GENERATIONS_PER_STEP
    aligned = {
        "completions": len(completions),
        "reward_names": len(reward_names),
        "reward_weights": len(reward_weights),
        "advantages": len(advantages),
        "effective_completion_tokens": len(effective_completion_tokens),
        "truncated_and_masked": len(truncated_and_masked),
    }
    if aligned["completions"] != expected:
        raise RuntimeError(f"rollout must contain eight completions: {aligned}")
    if aligned["reward_names"] != aligned["reward_weights"]:
        raise RuntimeError(f"reward names and weights are misaligned: {aligned}")
    if any(
        aligned[key] != expected
        for key in (
            "advantages",
            "effective_completion_tokens",
            "truncated_and_masked",
        )
    ):
        raise RuntimeError(f"rollout evidence arrays are misaligned: {aligned}")
    if set(component_rewards) != set(reward_names):
        raise RuntimeError("component reward names disagree with trainer order")
    if any(len(component_rewards[name]) != expected for name in reward_names):
        raise RuntimeError("component reward arrays must each contain eight values")
    if any(not isinstance(completion, str) for completion in completions):
        raise TypeError("every logged completion must be text")

    weighted_totals = [
        sum(
            float(component_rewards[name][index]) * float(reward_weights[position])
            for position, name in enumerate(reward_names)
        )
        for index in range(expected)
    ]
    normalized_advantages = [float(value) for value in advantages]
    numeric_values = weighted_totals + normalized_advantages + [
        float(value)
        for name in reward_names
        for value in component_rewards[name]
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError("rollout produced a non-finite reward or advantage")

    records = []
    for index, completion in enumerate(completions):
        records.append(
            {
                "sku_id": sku_id,
                "rollout_index": index,
                "raw_output": completion,
                "component_rewards": {
                    name: float(component_rewards[name][index])
                    for name in reward_names
                },
                "weighted_total": weighted_totals[index],
                "advantage": normalized_advantages[index],
                "effective_completion_tokens": int(
                    effective_completion_tokens[index]
                ),
                "truncated_and_masked": bool(truncated_and_masked[index]),
            }
        )

    return {
        "component_reward_names": list(reward_names),
        "weighted_totals": weighted_totals,
        "weighted_total_unique_count": len(set(weighted_totals)),
        "weighted_total_has_variance": len(set(weighted_totals)) > 1,
        "advantages": normalized_advantages,
        "nonzero_advantage_count": sum(
            not math.isclose(value, 0.0, abs_tol=1e-8)
            for value in normalized_advantages
        ),
        "effective_completion_tokens": [
            int(value) for value in effective_completion_tokens
        ],
        "truncated_and_masked_count": sum(
            bool(value) for value in truncated_and_masked
        ),
        "records": records,
    }


def _tensor_like_to_list(value: object, *, label: str) -> list:
    """Copy a tensor-like value to ordinary CPU-owned Python data."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} is not list- or tensor-like")
    return list(value)


class SmokeRolloutCollector:
    """Freeze every generated group before TRL's latest-group deque overwrites it."""

    def __init__(self, expected_sku_ids: Sequence[str]):
        self.expected_sku_ids = tuple(expected_sku_ids)
        if len(self.expected_sku_ids) != EXPECTED_STEPS:
            raise ValueError("rollout collector requires exactly five ordered SKUs")
        if len(set(self.expected_sku_ids)) != EXPECTED_STEPS:
            raise ValueError("rollout collector SKU order contains duplicates")
        self._groups: list[dict] = []

    @property
    def captured_steps(self) -> int:
        return len(self._groups)

    @property
    def records(self) -> list[dict]:
        return copy.deepcopy(
            [record for group in self._groups for record in group["records"]]
        )

    def capture_from_trainer(
        self,
        *,
        trainer: object,
        inputs: Sequence[dict],
        prepared: dict,
    ) -> dict:
        """Capture the just-generated group from one single-GPU TRL trainer."""
        if not isinstance(prepared, dict):
            raise TypeError("prepared rollout must be a dictionary")
        accelerator = getattr(trainer, "accelerator", None)
        if int(getattr(accelerator, "num_processes", 1)) != 1:
            raise RuntimeError("smoke rollout capture supports exactly one process")
        state = getattr(trainer, "state", None)
        step = int(getattr(state, "global_step", -1)) + 1
        expected_step = len(self._groups) + 1
        if step != expected_step or not 1 <= step <= EXPECTED_STEPS:
            raise RuntimeError(
                f"rollout capture step drifted: {step} != {expected_step}"
            )
        if len(inputs) != GENERATIONS_PER_STEP or any(
            not isinstance(row, dict) for row in inputs
        ):
            raise RuntimeError("rollout capture requires eight input rows")
        expected_sku = self.expected_sku_ids[step - 1]
        input_skus = [row.get("sku_id") for row in inputs]
        if input_skus != [expected_sku] * GENERATIONS_PER_STEP:
            raise RuntimeError(f"rollout input SKU drifted at step {step}")

        logs = getattr(trainer, "_logs", None)
        if not isinstance(logs, dict):
            raise RuntimeError("trainer has no completion logs to capture")
        completions = list(logs.get("completion", ()))
        advantages = list(logs.get("advantages", ()))
        reward_names = list(getattr(trainer, "reward_func_names", ()))
        logged_rewards = logs.get("rewards", {})
        component_rewards = {
            name: list(logged_rewards.get(name, ())) for name in reward_names
        }
        reward_weights = _tensor_like_to_list(
            getattr(trainer, "reward_weights", None), label="trainer reward weights"
        )

        completion_mask = prepared.get("completion_mask")
        mask_rows = _tensor_like_to_list(
            completion_mask, label="prepared completion mask"
        )
        if len(mask_rows) != GENERATIONS_PER_STEP:
            raise RuntimeError("prepared completion mask must contain eight rows")
        effective_completion_tokens = []
        truncated_and_masked = []
        for row in mask_rows:
            values = _tensor_like_to_list(row, label="completion-mask row")
            if any(value not in (0, 1, False, True) for value in values):
                raise RuntimeError("completion mask contains a non-binary value")
            effective_tokens = sum(int(value) for value in values)
            effective_completion_tokens.append(effective_tokens)
            truncated_and_masked.append(effective_tokens == 0)

        group = build_rollout_evidence(
            sku_id=expected_sku,
            completions=completions,
            reward_names=reward_names,
            component_rewards=component_rewards,
            reward_weights=reward_weights,
            advantages=advantages,
            effective_completion_tokens=effective_completion_tokens,
            truncated_and_masked=truncated_and_masked,
        )
        group["step"] = step
        group["records"] = [
            {"step": step, **record} for record in group["records"]
        ]
        self._groups.append(group)
        return {
            "step": step,
            "sku_id": expected_sku,
            "records_captured": len(group["records"]),
            "total_records_captured": len(self.records),
            "truncated_and_masked_count": group["truncated_and_masked_count"],
        }

    def finalize(self) -> dict:
        """Require and return the complete immutable 5 x 8 rollout evidence."""
        records = self.records
        if len(self._groups) != EXPECTED_STEPS or len(records) != EXPECTED_ROLLOUTS:
            raise RuntimeError("rollout collector does not contain all five groups")
        validation = validate_rollout_records(
            records, expected_sku_ids=self.expected_sku_ids
        )
        return {
            "groups": copy.deepcopy(self._groups),
            "records": records,
            "validation": validation,
            "all_groups_captured_before_overwrite": True,
        }


def make_rollout_capturing_trainer_class(base_trainer_class: type) -> type:
    """Wrap GRPOTrainer so every generated group is copied before training resumes."""
    if not isinstance(base_trainer_class, type):
        raise TypeError("base trainer must be a class")

    class RolloutCapturingTrainer(base_trainer_class):
        def __init__(
            self,
            *args,
            smoke_rollout_collector: SmokeRolloutCollector,
            **kwargs,
        ):
            if not isinstance(smoke_rollout_collector, SmokeRolloutCollector):
                raise TypeError("trainer requires a SmokeRolloutCollector")
            self.smoke_rollout_collector = smoke_rollout_collector
            super().__init__(*args, **kwargs)

        def _generate_and_score_completions(self, inputs):
            prepared = super()._generate_and_score_completions(inputs)
            self.smoke_rollout_collector.capture_from_trainer(
                trainer=self,
                inputs=inputs,
                prepared=prepared,
            )
            return prepared

    RolloutCapturingTrainer.__name__ = f"RolloutCapturing{base_trainer_class.__name__}"
    return RolloutCapturingTrainer


class FullRunRolloutCollector:
    """Preserve every shuffled 300-step rollout group before TRL overwrites it."""

    def __init__(self, *, expected_steps: int = FULL_RUN_STEPS):
        if not isinstance(expected_steps, int) or not 1 <= expected_steps <= FULL_RUN_STEPS:
            raise ValueError("expected_steps must be between 1 and 300")
        self.expected_steps = expected_steps
        self._groups: list[dict] = []
        self._seen_skus: set[str] = set()

    @property
    def captured_steps(self) -> int:
        return len(self._groups)

    @property
    def records(self) -> list[dict]:
        return copy.deepcopy(
            [record for group in self._groups for record in group["records"]]
        )

    def capture_from_trainer(
        self,
        *,
        trainer: object,
        inputs: Sequence[dict],
        prepared: dict,
    ) -> dict:
        """Capture one generated eight-completion group from a single GPU."""
        if not isinstance(prepared, dict):
            raise TypeError("prepared rollout must be a dictionary")
        accelerator = getattr(trainer, "accelerator", None)
        if int(getattr(accelerator, "num_processes", 1)) != 1:
            raise RuntimeError("full-run rollout capture supports exactly one process")
        state = getattr(trainer, "state", None)
        step = int(getattr(state, "global_step", -1)) + 1
        expected_step = len(self._groups) + 1
        if step != expected_step or not 1 <= step <= self.expected_steps:
            raise RuntimeError(
                f"rollout capture step drifted: {step} != {expected_step}"
            )
        if len(inputs) != GENERATIONS_PER_STEP or any(
            not isinstance(row, dict) for row in inputs
        ):
            raise RuntimeError("rollout capture requires eight input rows")
        input_skus = [row.get("sku_id") for row in inputs]
        sku_id = input_skus[0]
        if not isinstance(sku_id, str) or not sku_id.strip():
            raise RuntimeError(f"rollout input has no SKU at step {step}")
        if input_skus != [sku_id] * GENERATIONS_PER_STEP:
            raise RuntimeError(f"rollout input SKU drifted at step {step}")
        if sku_id in self._seen_skus:
            raise RuntimeError(f"rollout input SKU repeated across steps: {sku_id}")

        reward_names = list(getattr(trainer, "reward_func_names", ()))
        if reward_names != list(EXPECTED_REWARD_NAMES):
            raise RuntimeError("trainer reward names or order drifted")
        reward_weights = _tensor_like_to_list(
            getattr(trainer, "reward_weights", None), label="trainer reward weights"
        )
        if tuple(float(value) for value in reward_weights) != LOCKED_REWARD_WEIGHTS:
            raise RuntimeError("trainer reward weights drifted")

        logs = getattr(trainer, "_logs", None)
        if not isinstance(logs, dict):
            raise RuntimeError("trainer has no completion logs to capture")
        logged_rewards = logs.get("rewards", {})
        if not isinstance(logged_rewards, dict):
            raise RuntimeError("trainer has no component reward logs to capture")
        component_rewards = {
            name: list(logged_rewards.get(name, ())) for name in reward_names
        }

        mask_rows = _tensor_like_to_list(
            prepared.get("completion_mask"), label="prepared completion mask"
        )
        if len(mask_rows) != GENERATIONS_PER_STEP:
            raise RuntimeError("prepared completion mask must contain eight rows")
        effective_completion_tokens = []
        truncated_and_masked = []
        for row in mask_rows:
            values = _tensor_like_to_list(row, label="completion-mask row")
            if any(value not in (0, 1, False, True) for value in values):
                raise RuntimeError("completion mask contains a non-binary value")
            effective_tokens = sum(int(value) for value in values)
            effective_completion_tokens.append(effective_tokens)
            truncated_and_masked.append(effective_tokens == 0)

        group = build_rollout_evidence(
            sku_id=sku_id,
            completions=list(logs.get("completion", ())),
            reward_names=reward_names,
            component_rewards=component_rewards,
            reward_weights=reward_weights,
            advantages=list(logs.get("advantages", ())),
            effective_completion_tokens=effective_completion_tokens,
            truncated_and_masked=truncated_and_masked,
        )
        group["step"] = step
        group["records"] = [
            {"step": step, **record} for record in group["records"]
        ]
        self._groups.append(group)
        self._seen_skus.add(sku_id)
        return {
            "step": step,
            "sku_id": sku_id,
            "records_captured": GENERATIONS_PER_STEP,
            "total_records_captured": len(self._groups) * GENERATIONS_PER_STEP,
            "truncated_and_masked_count": group["truncated_and_masked_count"],
        }

    def snapshot(self, *, expected_step: int) -> dict:
        """Return a validated immutable prefix for a checkpoint handoff."""
        if expected_step != len(self._groups):
            raise RuntimeError(
                f"rollout snapshot step drifted: {expected_step} != {len(self._groups)}"
            )
        records = self.records
        validation = validate_full_run_rollout_records(
            records, expected_steps=expected_step
        )
        return {
            "groups": copy.deepcopy(self._groups),
            "records": records,
            "validation": validation,
            "all_groups_captured_before_overwrite": True,
        }

    def finalize(self) -> dict:
        """Return the complete validated long-run rollout evidence."""
        if len(self._groups) != self.expected_steps:
            raise RuntimeError(
                f"rollout collector has {len(self._groups)} of {self.expected_steps} groups"
            )
        return self.snapshot(expected_step=self.expected_steps)


def make_full_run_rollout_capturing_trainer_class(
    base_trainer_class: type,
) -> type:
    """Wrap GRPOTrainer so all long-run groups are copied before reuse."""
    if not isinstance(base_trainer_class, type):
        raise TypeError("base trainer must be a class")

    class FullRunRolloutCapturingTrainer(base_trainer_class):
        def __init__(
            self,
            *args,
            full_run_rollout_collector: FullRunRolloutCollector,
            **kwargs,
        ):
            if not isinstance(full_run_rollout_collector, FullRunRolloutCollector):
                raise TypeError("trainer requires a FullRunRolloutCollector")
            self.full_run_rollout_collector = full_run_rollout_collector
            super().__init__(*args, **kwargs)

        def _generate_and_score_completions(self, inputs):
            prepared = super()._generate_and_score_completions(inputs)
            self.full_run_rollout_collector.capture_from_trainer(
                trainer=self,
                inputs=inputs,
                prepared=prepared,
            )
            return prepared

    FullRunRolloutCapturingTrainer.__name__ = (
        f"FullRunRolloutCapturing{base_trainer_class.__name__}"
    )
    return FullRunRolloutCapturingTrainer


class FullRunCheckpointHandoff:
    """Bridge Trainer checkpoint events to validated evidence persistence.

    This coordinator is deliberately CPU-only and independent of Transformers.
    A small factory below mixes it into the installed ``TrainerCallback`` only
    after the guarded GPU path has imported that class.
    """

    def __init__(
        self,
        *,
        lifecycle_writer: FullRunCheckpointLifecycleWriter,
        rollout_collector: FullRunRolloutCollector,
        phase_timing_snapshot_fn: Callable[[int], dict],
        progress_callback: Callable[..., object] | None = None,
    ):
        if not isinstance(lifecycle_writer, FullRunCheckpointLifecycleWriter):
            raise TypeError("handoff requires a FullRunCheckpointLifecycleWriter")
        if not isinstance(rollout_collector, FullRunRolloutCollector):
            raise TypeError("handoff requires a FullRunRolloutCollector")
        if rollout_collector.expected_steps != FULL_RUN_STEPS:
            raise ValueError("handoff collector must require exactly 300 steps")
        if not callable(phase_timing_snapshot_fn):
            raise TypeError("handoff requires a phase-timing snapshot function")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("handoff progress callback must be callable")
        self.lifecycle_writer = lifecycle_writer
        self.rollout_collector = rollout_collector
        self.phase_timing_snapshot_fn = phase_timing_snapshot_fn
        self.progress_callback = progress_callback
        self._begun = False
        self._ended = False
        self._reported_progress_step = 0
        self._processed_checkpoint_steps: list[int] = []
        self._checkpoint_evidence: dict[int, dict] = {}
        self._final_evidence: dict | None = None

    @staticmethod
    def _locked_argument_values(args: object) -> dict:
        return {
            "max_steps": int(getattr(args, "max_steps", -1)),
            "save_steps": int(getattr(args, "save_steps", -1)),
            "save_total_limit": int(getattr(args, "save_total_limit", -1)),
            "save_only_model": getattr(args, "save_only_model", None),
        }

    def on_train_begin(self, args: object, state: object) -> dict:
        """Verify the callback starts from the untouched locked run state."""
        if self._begun:
            raise RuntimeError("full-run checkpoint handoff began more than once")
        expected_arguments = {
            "max_steps": FULL_RUN_STEPS,
            "save_steps": FULL_RUN_SAVE_STEPS,
            "save_total_limit": FULL_RUN_SAVE_TOTAL_LIMIT,
            "save_only_model": True,
        }
        actual_arguments = self._locked_argument_values(args)
        if actual_arguments != expected_arguments:
            raise RuntimeError(
                "full-run callback trainer arguments drifted: "
                f"{actual_arguments} != {expected_arguments}"
            )
        actual_output = Path(getattr(args, "output_dir", "")).resolve()
        expected_output = Path(
            self.lifecycle_writer.plan["trainer_output_dir"]
        ).resolve()
        if actual_output != expected_output:
            raise RuntimeError(
                f"full-run callback output directory drifted: {actual_output}"
            )
        if int(getattr(state, "global_step", -1)) != 0:
            raise RuntimeError("full-run callback must begin at global step zero")
        if self.rollout_collector.captured_steps != 0:
            raise RuntimeError("full-run callback began with captured rollouts")
        if self.lifecycle_writer.events:
            raise RuntimeError("full-run callback began with lifecycle events")
        self._begun = True
        return {
            "status": "ready",
            "trainer_output_dir": str(expected_output),
            "checkpoint_steps": list(FULL_RUN_CHECKPOINT_STEPS),
        }

    @staticmethod
    def _log_history(state: object) -> list[dict]:
        history = getattr(state, "log_history", None)
        if not isinstance(history, (list, tuple)) or not all(
            isinstance(entry, dict) for entry in history
        ):
            raise TypeError("trainer state log_history must contain dictionaries")
        return list(history)

    def on_save(self, state: object) -> dict:
        """Validate one checkpoint boundary and persist its lifecycle event."""
        if not self._begun or self._ended:
            raise RuntimeError("checkpoint save occurred outside active training")
        step = int(getattr(state, "global_step", -1))
        expected_index = len(self._processed_checkpoint_steps)
        expected_step = (
            FULL_RUN_CHECKPOINT_STEPS[expected_index]
            if expected_index < len(FULL_RUN_CHECKPOINT_STEPS)
            else None
        )
        if step != expected_step:
            raise RuntimeError(
                f"full-run checkpoint callback expected step {expected_step}, found {step}"
            )

        # Validate all evidence before mutating the lifecycle writer. This keeps
        # a malformed rollout or scalar log from being recorded as a good save.
        rollout_snapshot = self.rollout_collector.snapshot(expected_step=step)
        log_validation = validate_full_run_trainer_log_history(
            self._log_history(state), expected_steps=step
        )
        phase_timing_report = self.phase_timing_snapshot_fn(step)
        checkpoint_event = self.lifecycle_writer.record_checkpoint_saved(step)
        milestone_event = None
        retention_report = None
        if step == 100:
            milestone_event = self.lifecycle_writer.export_step_100(
                rollout_records=rollout_snapshot["records"],
                trainer_step_logs=log_validation["step_records"],
                phase_timing_report=phase_timing_report,
            )
        else:
            milestone_event = self.lifecycle_writer.export_rolling_evidence(
                step=step,
                rollout_records=rollout_snapshot["records"],
                trainer_step_logs=log_validation["step_records"],
                phase_timing_report=phase_timing_report,
            )
        if step == 300:
            retention_report = (
                self.lifecycle_writer.verify_retention_after_step_300()
            )

        report = {
            "step": step,
            "checkpoint_event": checkpoint_event,
            "milestone_event": milestone_event,
            "retention_report": retention_report,
            "rollout_validation": copy.deepcopy(rollout_snapshot["validation"]),
            "trainer_log_validation": {
                key: copy.deepcopy(value)
                for key, value in log_validation.items()
                if key != "step_records"
            },
            "phase_timing_steps": phase_timing_report["steps"],
        }
        self._processed_checkpoint_steps.append(step)
        self._checkpoint_evidence[step] = report
        return copy.deepcopy(report)

    def on_log(self, state: object) -> dict | None:
        """Validate and forward one authoritative optimizer-step heartbeat."""
        if self.progress_callback is None:
            return None
        if not self._begun or self._ended:
            raise RuntimeError("full-run progress occurred outside active training")
        step = int(getattr(state, "global_step", -1))
        history = self._log_history(state)
        latest_log = history[-1] if history else {}
        if "loss" not in latest_log:
            if (
                step == FULL_RUN_STEPS
                and self._reported_progress_step == FULL_RUN_STEPS
            ):
                return None
            raise RuntimeError(
                "full-run non-step log arrived before progress completion"
            )
        expected_step = self._reported_progress_step + 1
        if step != expected_step:
            raise RuntimeError(
                f"full-run progress expected step {expected_step}, found {step}"
            )
        if self.rollout_collector.captured_steps != step:
            raise RuntimeError(
                "full-run progress disagrees with captured rollout groups"
            )
        log_validation = validate_full_run_trainer_log_history(
            history, expected_steps=step
        )
        if log_validation.get("step_records") is None or len(
            log_validation["step_records"]
        ) != step:
            raise RuntimeError("full-run progress scalar-log prefix drifted")
        result = self.progress_callback(
            optimizer_step=step,
            rollout_records=step * GENERATIONS_PER_STEP,
            scalar_logs=step,
        )
        self._reported_progress_step = step
        return {
            "optimizer_step": step,
            "rollout_records": step * GENERATIONS_PER_STEP,
            "scalar_logs": step,
            "callback_result": result,
        }

    def on_train_end(self, state: object) -> dict:
        """Freeze complete evidence for the later final-adapter publication."""
        if not self._begun or self._ended:
            raise RuntimeError("full-run checkpoint handoff ended out of order")
        if int(getattr(state, "global_step", -1)) != FULL_RUN_STEPS:
            raise RuntimeError("full-run callback did not end at step 300")
        if self._processed_checkpoint_steps != list(FULL_RUN_CHECKPOINT_STEPS):
            raise RuntimeError("full-run callback did not process all checkpoints")
        if (
            self.progress_callback is not None
            and self._reported_progress_step != FULL_RUN_STEPS
        ):
            raise RuntimeError("full-run callback did not report all progress steps")
        if (
            self.lifecycle_writer.snapshot().get("status")
            != "checkpoints_ready_for_final_handoff"
        ):
            raise RuntimeError("checkpoint lifecycle is not ready for final handoff")

        rollout_snapshot = self.rollout_collector.finalize()
        log_validation = validate_full_run_trainer_log_history(
            self._log_history(state)
        )
        self._final_evidence = {
            "rollout_records": rollout_snapshot["records"],
            "trainer_step_logs": log_validation["step_records"],
            "rollout_validation": copy.deepcopy(rollout_snapshot["validation"]),
            "trainer_log_validation": {
                key: copy.deepcopy(value)
                for key, value in log_validation.items()
                if key != "step_records"
            },
            "phase_timing_report": self.phase_timing_snapshot_fn(FULL_RUN_STEPS),
            "checkpoint_evidence": copy.deepcopy(self._checkpoint_evidence),
            "lifecycle": self.lifecycle_writer.snapshot(),
        }
        self._ended = True
        return self.final_evidence()

    def final_evidence(self) -> dict:
        """Return immutable complete evidence only after successful train end."""
        if self._final_evidence is None or not self._ended:
            raise RuntimeError("full-run final evidence is not ready")
        return copy.deepcopy(self._final_evidence)


def make_full_run_checkpoint_callback_class(base_callback_class: type) -> type:
    """Create a Transformers-compatible callback around the CPU handoff."""
    if not isinstance(base_callback_class, type):
        raise TypeError("base callback must be a class")

    class FullRunCheckpointCallback(base_callback_class):
        def __init__(
            self,
            *,
            lifecycle_writer: FullRunCheckpointLifecycleWriter,
            rollout_collector: FullRunRolloutCollector,
            phase_timing_snapshot_fn: Callable[[int], dict],
            progress_callback: Callable[..., object] | None = None,
        ):
            super().__init__()
            self.checkpoint_handoff = FullRunCheckpointHandoff(
                lifecycle_writer=lifecycle_writer,
                rollout_collector=rollout_collector,
                phase_timing_snapshot_fn=phase_timing_snapshot_fn,
                progress_callback=progress_callback,
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self.checkpoint_handoff.on_train_begin(args, state)
            return control

        def on_save(self, args, state, control, **kwargs):
            self.checkpoint_handoff.on_save(state)
            return control

        def on_log(self, args, state, control, **kwargs):
            self.checkpoint_handoff.on_log(state)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.checkpoint_handoff.on_train_end(state)
            return control

        def final_evidence(self) -> dict:
            return self.checkpoint_handoff.final_evidence()

    FullRunCheckpointCallback.__name__ = (
        f"FullRunCheckpoint{base_callback_class.__name__}"
    )
    return FullRunCheckpointCallback


def run_full_run_300_orchestration(
    *,
    base_trainer_class: type,
    base_callback_class: type,
    config_class: type,
    model: object,
    tokenizer: object,
    dataset: object,
    reward_functions: Sequence[Callable],
    reward_weights: Sequence[float],
    source_adapter_file: str | Path,
    final_output_dir: str | Path,
    preflight_report: dict,
    trainability_fn: Callable[[object], dict] | None = None,
    parameter_values_fn: Callable[[object], dict] | None = None,
    fingerprint_fn: Callable[[object], str] | None = None,
    runtime_context: dict | None = None,
    cuda_snapshot_fn: Callable[[], dict] | None = None,
    phase_synchronize_fn: Callable[[], object] | None = None,
    disk_usage_fn: Callable[[Path], object] | None = None,
    progress_callback: Callable[..., object] | None = None,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Construct, run and publish the locked full run via injected runtime types.

    This function has no CLI caller. Its dependencies are injected so the
    complete orchestration can be proven with CPU fakes before GPU dispatch is
    enabled.
    """
    if not all(
        isinstance(value, type)
        for value in (base_trainer_class, base_callback_class, config_class)
    ):
        raise TypeError("trainer, callback and config dependencies must be classes")
    if preflight_report.get("status") != "passed":
        raise ValueError("full-run orchestration requires a passed preflight")
    if preflight_report.get("cuda_imports_performed") is not False:
        raise ValueError("full-run preflight CUDA-import evidence drifted")
    trainability_fn = trainability_fn or inspect_model_trainability
    parameter_values_fn = (
        parameter_values_fn or inspect_trainable_parameter_values
    )
    fingerprint_fn = fingerprint_fn or _trainable_parameter_sha256
    if not isinstance(runtime_context, dict) or runtime_context.get(
        "version"
    ) != FULL_RUN_RUNTIME_CONTEXT_VERSION:
        raise ValueError("full-run orchestration requires locked runtime context")
    if not callable(cuda_snapshot_fn):
        raise TypeError("full-run orchestration requires CUDA snapshots")
    if not callable(phase_synchronize_fn):
        raise TypeError("full-run orchestration requires phase synchronization")

    final_output = Path(final_output_dir).resolve()
    preflight_output = Path(
        preflight_report.get("output", {}).get("path", "")
    ).resolve()
    if preflight_output != final_output:
        raise RuntimeError("full-run output disagrees with preflight")
    if final_output.exists():
        raise FileExistsError(f"final full-run output already exists: {final_output}")

    source_adapter = Path(source_adapter_file).resolve()
    expected_source_sha = preflight_report.get("sft_lock", {}).get(
        "adapter_sha256"
    )
    if not isinstance(expected_source_sha, str) or len(expected_source_sha) != 64:
        raise ValueError("preflight has no locked SFT adapter SHA-256")
    if _sha256_file(source_adapter) != expected_source_sha:
        raise RuntimeError("source adapter disagrees with full-run preflight")

    if len(dataset) != FULL_RUN_DATA_ROWS:
        raise RuntimeError("full-run dataset must contain exactly 1,565 rows")
    required_columns = {"prompt", "gold", "sku_id"}
    if set(getattr(dataset, "column_names", ())) != required_columns:
        raise RuntimeError("full-run dataset columns drifted")
    dataset_skus = list(dataset["sku_id"])
    if len(dataset_skus) != FULL_RUN_DATA_ROWS or any(
        not isinstance(sku_id, str) or not sku_id.strip()
        for sku_id in dataset_skus
    ):
        raise RuntimeError("full-run dataset contains an invalid SKU")
    if len(set(dataset_skus)) != FULL_RUN_DATA_ROWS:
        raise RuntimeError("full-run dataset contains duplicate SKUs")
    expected_order_sha = preflight_report.get("pool", {}).get(
        "ordered_sku_sha256"
    )
    if _ordered_sku_sha256(dataset_skus) != expected_order_sha:
        raise RuntimeError("full-run dataset SKU order disagrees with preflight")

    normalized_weights = tuple(float(value) for value in reward_weights)
    if normalized_weights != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("full-run reward weights drifted")
    reward_names = [
        getattr(reward, "__name__", type(reward).__name__)
        for reward in reward_functions
    ]
    if tuple(reward_names) != EXPECTED_REWARD_NAMES:
        raise RuntimeError("full-run reward function names or order drifted")
    if not all(callable(reward) for reward in reward_functions):
        raise TypeError("every full-run reward function must be callable")

    model_save = getattr(model, "save_pretrained", None)
    tokenizer_save = getattr(tokenizer, "save_pretrained", None)
    if not callable(model_save) or not callable(tokenizer_save):
        raise TypeError("model and tokenizer must expose save_pretrained()")
    trainability_before = trainability_fn(model)
    lora_sha_before = fingerprint_fn(model)

    staging = create_full_run_staging_output(final_output)
    plan = build_full_run_lifecycle_plan(
        final_output_dir=final_output,
        staging_dir=staging,
    )
    writer = FullRunCheckpointLifecycleWriter(
        plan=plan,
        starting_adapter_sha256=expected_source_sha,
    )
    config_settings = grpo_full_run_300_config_kwargs(
        output_dir=plan["trainer_output_dir"],
        reward_weights=normalized_weights,
    )
    config = config_class(**config_settings)
    config_report = inspect_grpo_full_run_config(config)

    collector = FullRunRolloutCollector()
    capturing_trainer_class = make_full_run_rollout_capturing_trainer_class(
        base_trainer_class
    )
    trainer = capturing_trainer_class(
        model=model,
        reward_funcs=list(reward_functions),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        full_run_rollout_collector=collector,
    )
    if getattr(trainer, "full_run_rollout_collector", None) is not collector:
        raise RuntimeError("full-run trainer did not retain its rollout collector")
    if int(getattr(getattr(trainer, "state", None), "global_step", -1)) != 0:
        raise RuntimeError("full-run trainer did not start at global step zero")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("full-run trainer unexpectedly started with an optimizer")
    if getattr(trainer, "lr_scheduler", None) is not None:
        raise RuntimeError("full-run trainer unexpectedly started with a scheduler")
    if list(getattr(trainer, "reward_func_names", ())) != reward_names:
        raise RuntimeError("full-run trainer reward names drifted")
    actual_weights = _tensor_like_to_list(
        getattr(trainer, "reward_weights", None),
        label="full-run trainer reward weights",
    )
    if tuple(float(value) for value in actual_weights) != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("full-run trainer reward weights drifted after construction")

    phase_profiler = FullRunPhaseProfiler(
        expected_steps=FULL_RUN_STEPS,
        synchronize_fn=phase_synchronize_fn,
    )
    phase_profiler.instrument_trainer(trainer)
    profiler_callback_class = make_phase_profiler_callback_class(
        base_callback_class
    )
    profiler_callback = profiler_callback_class(phase_profiler=phase_profiler)
    callback_class = make_full_run_checkpoint_callback_class(base_callback_class)
    callback = callback_class(
        lifecycle_writer=writer,
        rollout_collector=collector,
        phase_timing_snapshot_fn=(
            lambda step: phase_profiler.snapshot(expected_steps=step)
        ),
        progress_callback=progress_callback,
    )
    add_callback = getattr(trainer, "add_callback", None)
    if not callable(add_callback):
        raise TypeError("full-run trainer does not expose add_callback()")
    add_callback(profiler_callback)
    add_callback(callback)

    cuda_before_train = cuda_snapshot_fn()
    train_started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_started
    phase_profile_summary = phase_profiler.finalize(train_seconds=train_seconds)
    cuda_after_train = cuda_snapshot_fn()
    completed_step = int(getattr(trainer.state, "global_step", -1))
    if completed_step != FULL_RUN_STEPS:
        raise RuntimeError("full-run trainer did not finish exactly 300 updates")
    if int(getattr(train_result, "global_step", -1)) != FULL_RUN_STEPS:
        raise RuntimeError("full-run result disagrees with completed global step")
    evidence = callback.final_evidence()
    if len(evidence["rollout_records"]) != FULL_RUN_STEPS * GENERATIONS_PER_STEP:
        raise RuntimeError("full-run callback returned incomplete rollout evidence")
    if len(evidence["trainer_step_logs"]) != FULL_RUN_STEPS:
        raise RuntimeError("full-run callback returned incomplete trainer logs")

    trainability_after = trainability_fn(model)
    for key in ("trainable_parameters", "trainable_tensors"):
        if trainability_after.get(key) != trainability_before.get(key):
            raise RuntimeError(f"full-run trainability changed for {key}")
    before_names = trainability_before.get("trainable_parameter_names")
    if before_names is not None and trainability_after.get(
        "trainable_parameter_names"
    ) != before_names:
        raise RuntimeError("full-run trainable parameter names changed")
    parameter_values = parameter_values_fn(model)
    if parameter_values.get("all_trainable_values_finite") is not True:
        raise RuntimeError("full-run LoRA contains a non-finite value")
    lora_sha_after = fingerprint_fn(model)
    if lora_sha_after == lora_sha_before:
        raise RuntimeError("full-run changed no trainable LoRA bytes")
    cuda_after_model_audit = cuda_snapshot_fn()
    train_metrics = dict(getattr(train_result, "metrics", {}))
    run_summary = {
        "version": FULL_RUN_SUMMARY_VERSION,
        "status": "completed",
        "training": {
            "optimizer_steps": completed_step,
            "rollout_records": len(evidence["rollout_records"]),
            "train_seconds": train_seconds,
            "train_metrics": train_metrics,
        },
        "model_audit": {
            "trainability_before": trainability_before,
            "trainability_after": trainability_after,
            "parameter_values_after": parameter_values,
            "lora_sha256_before": lora_sha_before,
            "lora_sha256_after": lora_sha_after,
            "trainable_lora_changed": True,
        },
        "resources": {
            "runtime": copy.deepcopy(runtime_context["runtime"]),
            "preflight_disk_free_bytes": preflight_report["disk"]["free_bytes"],
            "cuda_before_load": copy.deepcopy(runtime_context["cuda_before_load"]),
            "cuda_after_load": copy.deepcopy(runtime_context["cuda_after_load"]),
            "cuda_before_train": cuda_before_train,
            "cuda_after_train": cuda_after_train,
            "cuda_after_model_audit": cuda_after_model_audit,
        },
        "profiling": phase_profile_summary,
    }

    def save_live_adapter(adapter_output: Path) -> None:
        model_save(adapter_output, safe_serialization=True)
        tokenizer_save(adapter_output)

    final_adapter = writer.save_and_validate_final_adapter(
        save_adapter_fn=save_live_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=expected_adapter_model_bytes,
    )
    publication = writer.publish_completed_bundle(
        rollout_records=evidence["rollout_records"],
        trainer_step_logs=evidence["trainer_step_logs"],
        phase_timing_report=evidence["phase_timing_report"],
        preflight_report=preflight_report,
        config_settings=config_settings,
        run_summary=run_summary,
        disk_usage_fn=disk_usage_fn,
    )
    if writer.snapshot().get("status") != "completed_and_published":
        raise RuntimeError("full-run lifecycle did not finish publication")

    return {
        "status": "passed",
        "global_step": completed_step,
        "optimizer_steps": completed_step,
        "expected_rollouts": FULL_RUN_STEPS * GENERATIONS_PER_STEP,
        "train_seconds": train_seconds,
        "train_metrics": train_metrics,
        "phase_profile": phase_profile_summary,
        "phase_timing_report": evidence["phase_timing_report"],
        "run_summary": run_summary,
        "trainability_before": trainability_before,
        "trainability_after": trainability_after,
        "parameter_values_after": parameter_values,
        "lora_sha256_before": lora_sha_before,
        "lora_sha256_after": lora_sha_after,
        "trainable_lora_changed": True,
        "config": config_report,
        "dataset_rows": len(dataset),
        "dataset_order_sha256": expected_order_sha,
        "reward_names": reward_names,
        "reward_weights": list(normalized_weights),
        "rollout_validation": evidence["rollout_validation"],
        "trainer_log_validation": evidence["trainer_log_validation"],
        "checkpoint_evidence": evidence["checkpoint_evidence"],
        "final_adapter": final_adapter,
        "publication": publication,
        "lifecycle": writer.snapshot(),
        "final_output_dir": str(final_output),
        "published": True,
    }


def inspect_model_trainability(
    model: object,
    *,
    expected_trainable_parameters: int = LOCKED_TRAINABLE_PARAMETERS,
    expected_target_modules: set[str] = LOCKED_TARGET_MODULES,
) -> dict:
    """Fail unless the loaded policy exposes exactly the locked LoRA for training."""
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("loaded model does not expose named_parameters()")

    total_parameters = 0
    trainable_parameters = 0
    trainable_tensors = 0
    trainable_names = []
    trainable_dtypes: set[str] = set()
    trainable_devices: set[str] = set()
    observed_target_modules: set[str] = set()

    for name, parameter in named_parameters():
        if not isinstance(name, str) or not name:
            raise RuntimeError("loaded model contains an invalid parameter name")
        numel_fn = getattr(parameter, "numel", None)
        if not callable(numel_fn):
            raise TypeError(f"parameter {name} does not expose numel()")
        numel = int(numel_fn())
        if numel < 0:
            raise RuntimeError(f"parameter {name} has a negative size")
        total_parameters += numel

        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        trainable_parameters += numel
        trainable_tensors += 1
        trainable_names.append(name)
        trainable_dtypes.add(str(getattr(parameter, "dtype", "unknown")))
        trainable_devices.add(str(getattr(parameter, "device", "unknown")))
        if "lora_" not in name.lower():
            raise RuntimeError(f"non-LoRA parameter is unexpectedly trainable: {name}")
        matched_targets = {
            module
            for module in expected_target_modules
            if f".{module}." in f".{name}."
        }
        if len(matched_targets) != 1:
            raise RuntimeError(
                f"trainable parameter does not map to one locked target module: {name}"
            )
        observed_target_modules.update(matched_targets)

    if trainable_parameters != expected_trainable_parameters:
        raise RuntimeError(
            "runtime trainable-parameter count mismatch: "
            f"{trainable_parameters} != {expected_trainable_parameters}"
        )
    if observed_target_modules != expected_target_modules:
        raise RuntimeError(
            "runtime LoRA target modules mismatch: "
            f"{sorted(observed_target_modules)} != {sorted(expected_target_modules)}"
        )
    if total_parameters <= trainable_parameters:
        raise RuntimeError("loaded model does not contain a frozen base model")

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": 100 * trainable_parameters / total_parameters,
        "trainable_tensors": trainable_tensors,
        "trainable_parameter_names": trainable_names,
        "trainable_dtypes": sorted(trainable_dtypes),
        "trainable_devices": sorted(trainable_devices),
        "target_modules_observed": sorted(observed_target_modules),
        "only_lora_parameters_trainable": True,
        "matches_locked_trainable_count": True,
    }


def _trainable_parameter_sha256(model: object) -> str:
    """Hash trainable tensor names, metadata and bytes without writing a checkpoint."""
    digest = hashlib.sha256()
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(parameter.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = parameter.detach().contiguous().cpu().numpy().tobytes()
        digest.update(raw)
        digest.update(b"\0")
    if trainable_tensors == 0:
        raise RuntimeError("cannot fingerprint a model with no trainable tensors")
    return digest.hexdigest()


def inspect_trainable_parameter_values(
    model: object,
    *,
    expected_trainable_parameters: int = LOCKED_TRAINABLE_PARAMETERS,
    expected_trainable_tensors: int = LOCKED_TRAINABLE_TENSORS,
) -> dict:
    """Require every value in the exact trainable LoRA footprint to be finite."""
    trainable_tensors = 0
    trainable_parameters = 0
    nonfinite_parameters = 0
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        values = parameter.detach()
        elements = int(values.numel())
        trainable_parameters += elements
        finite_elements = int(values.isfinite().sum().item())
        nonfinite_parameters += elements - finite_elements
    if trainable_tensors != expected_trainable_tensors:
        raise RuntimeError("finite-value audit has an unexpected tensor count")
    if trainable_parameters != expected_trainable_parameters:
        raise RuntimeError("finite-value audit has an unexpected parameter count")
    if nonfinite_parameters != 0:
        raise RuntimeError("five-step LoRA contains NaN or infinity")
    return {
        "trainable_tensors": trainable_tensors,
        "trainable_parameters": trainable_parameters,
        "nonfinite_parameters": 0,
        "all_trainable_values_finite": True,
    }


def validate_gradient_evidence(stats: dict) -> dict:
    """Fail closed unless backward produced complete, finite, nonzero LoRA gradients."""
    if stats.get("trainable_tensors") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("gradient evidence has an unexpected trainable tensor count")
    if stats.get("gradient_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("gradient evidence has an unexpected element count")
    if stats.get("tensors_with_gradient") != stats.get("trainable_tensors"):
        raise RuntimeError("at least one trainable LoRA tensor has no gradient")
    if stats.get("nonfinite_gradient_elements") != 0:
        raise RuntimeError("gradient evidence contains NaN or infinity")
    if stats.get("nonzero_gradient_tensors", 0) <= 0:
        raise RuntimeError("all LoRA gradient tensors are zero")
    if stats.get("nonzero_gradient_elements", 0) <= 0:
        raise RuntimeError("all LoRA gradient elements are zero")
    global_l2_norm = float(stats.get("global_l2_norm", float("nan")))
    if not math.isfinite(global_l2_norm) or global_l2_norm <= 0:
        raise RuntimeError("global LoRA gradient norm is not finite and positive")
    return {
        **stats,
        "all_trainable_tensors_have_gradients": True,
        "all_gradients_finite": True,
        "has_nonzero_gradient": True,
        "matches_locked_gradient_footprint": True,
    }


def validate_optimizer_evidence(stats: dict) -> dict:
    """Fail closed unless the fresh optimizer controls only the locked LoRA."""
    if stats.get("optimizer_class") != "AdamW":
        raise RuntimeError("configured optimizer is not bitsandbytes AdamW")
    if not str(stats.get("optimizer_module", "")).startswith("bitsandbytes.optim"):
        raise RuntimeError("configured optimizer did not come from bitsandbytes")
    if stats.get("optimizer_bits") != 8:
        raise RuntimeError("configured bitsandbytes AdamW is not using 8-bit state")
    if stats.get("is_paged"):
        raise RuntimeError("configured optimizer unexpectedly uses paged state")
    if stats.get("optimizer_initialized_flag"):
        raise RuntimeError("fresh optimizer unexpectedly reports initialized state")
    parameter_groups = stats.get("parameter_groups")
    if not isinstance(parameter_groups, list) or len(parameter_groups) != 2:
        raise RuntimeError("optimizer parameter-group structure drifted")
    for group in parameter_groups:
        if float(group.get("lr", float("nan"))) != LOCKED_LEARNING_RATE:
            raise RuntimeError("optimizer learning rate drifted")
        if tuple(group.get("betas", ())) != LOCKED_ADAM_BETAS:
            raise RuntimeError("optimizer beta values drifted")
        if float(group.get("eps", float("nan"))) != LOCKED_ADAM_EPSILON:
            raise RuntimeError("optimizer epsilon drifted")
    nonempty_groups = [
        group for group in parameter_groups if group.get("parameter_tensors", 0) > 0
    ]
    if len(nonempty_groups) != 1:
        raise RuntimeError("optimizer nonempty parameter-group structure drifted")
    if float(nonempty_groups[0].get("weight_decay", float("nan"))) != (
        LOCKED_WEIGHT_DECAY
    ):
        raise RuntimeError("optimizer weight decay drifted")
    if stats.get("trainable_model_tensors") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("optimizer audit has an unexpected trainable tensor count")
    if stats.get("trainable_model_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("optimizer audit has an unexpected trainable element count")
    if stats.get("unique_optimizer_parameter_tensors") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("optimizer does not control every trainable LoRA tensor")
    if stats.get("unique_optimizer_parameter_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("optimizer does not control every trainable LoRA element")
    if stats.get("missing_trainable_tensors") != 0:
        raise RuntimeError("optimizer is missing at least one trainable LoRA tensor")
    if stats.get("frozen_optimizer_tensors") != 0:
        raise RuntimeError("optimizer controls at least one frozen tensor")
    if stats.get("duplicate_optimizer_references") != 0:
        raise RuntimeError("optimizer contains a duplicate parameter reference")
    if stats.get("optimizer_state_entries") != 0:
        raise RuntimeError("fresh optimizer unexpectedly initialized parameter state")
    if stats.get("optimizer_state_tensor_count") != 0:
        raise RuntimeError("fresh optimizer unexpectedly owns state tensors")
    if stats.get("optimizer_state_tensor_bytes") != 0:
        raise RuntimeError("fresh optimizer unexpectedly allocated state bytes")
    if stats.get("gradients_attached") != 0:
        raise RuntimeError("optimizer-construction gate unexpectedly has gradients")
    if not stats.get("trainable_lora_unchanged"):
        raise RuntimeError("LoRA weights changed during optimizer construction")
    if stats.get("global_step") != 0:
        raise RuntimeError("optimizer construction advanced global_step")
    return {
        **stats,
        "controls_exact_locked_lora": True,
        "optimizer_state_is_lazy": True,
        "no_parameter_update": True,
    }


def inspect_lora_gradients(model: object) -> dict:
    """Measure gradients on every trainable LoRA tensor after one backward pass."""
    trainable_tensors = 0
    tensors_with_gradient = 0
    gradient_elements = 0
    nonfinite_gradient_elements = 0
    nonzero_gradient_tensors = 0
    nonzero_gradient_elements = 0
    global_squared_l2 = 0.0
    global_max_abs = 0.0
    global_sum_abs = 0.0
    gradient_dtypes: set[str] = set()
    by_target = {
        target: {
            "trainable_tensors": 0,
            "tensors_with_gradient": 0,
            "nonzero_gradient_tensors": 0,
            "gradient_elements": 0,
            "squared_l2": 0.0,
            "max_abs": 0.0,
        }
        for target in sorted(LOCKED_TARGET_MODULES)
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        matched_targets = [
            target
            for target in LOCKED_TARGET_MODULES
            if f".{target}." in f".{name}."
        ]
        if len(matched_targets) != 1 or "lora_" not in name.lower():
            raise RuntimeError(f"unexpected trainable tensor during gradient audit: {name}")
        target_stats = by_target[matched_targets[0]]
        target_stats["trainable_tensors"] += 1

        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.detach()
        tensors_with_gradient += 1
        target_stats["tensors_with_gradient"] += 1
        elements = int(gradient.numel())
        gradient_elements += elements
        target_stats["gradient_elements"] += elements
        gradient_dtypes.add(str(gradient.dtype))

        finite_mask = gradient.isfinite()
        finite_count = int(finite_mask.sum().item())
        nonfinite_gradient_elements += elements - finite_count
        nonzero_elements = int(gradient.count_nonzero().item())
        nonzero_gradient_elements += nonzero_elements
        if nonzero_elements > 0:
            nonzero_gradient_tensors += 1
            target_stats["nonzero_gradient_tensors"] += 1

        gradient_float = gradient.float()
        gradient_abs = gradient_float.abs()
        tensor_l2 = float(gradient_float.norm(2).item())
        tensor_max_abs = float(gradient_abs.max().item())
        tensor_sum_abs = float(gradient_abs.sum().item())
        global_squared_l2 += tensor_l2**2
        global_max_abs = max(global_max_abs, tensor_max_abs)
        global_sum_abs += tensor_sum_abs
        target_stats["squared_l2"] += tensor_l2**2
        target_stats["max_abs"] = max(target_stats["max_abs"], tensor_max_abs)
        del finite_mask, gradient_float, gradient_abs

    for target_stats in by_target.values():
        target_stats["l2_norm"] = math.sqrt(target_stats.pop("squared_l2"))

    stats = {
        "trainable_tensors": trainable_tensors,
        "tensors_with_gradient": tensors_with_gradient,
        "gradient_elements": gradient_elements,
        "nonfinite_gradient_elements": nonfinite_gradient_elements,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "nonzero_gradient_elements": nonzero_gradient_elements,
        "zero_gradient_elements": gradient_elements - nonzero_gradient_elements,
        "global_l2_norm": math.sqrt(global_squared_l2),
        "global_max_abs": global_max_abs,
        "global_mean_abs": global_sum_abs / gradient_elements
        if gradient_elements
        else 0.0,
        "gradient_dtypes": sorted(gradient_dtypes),
        "by_target_module": by_target,
    }
    return validate_gradient_evidence(stats)


def inspect_fresh_optimizer(optimizer: object, model: object, global_step: int) -> dict:
    """Audit a newly constructed optimizer before gradients or a first step."""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    optimizer_parameters = []
    parameter_groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        optimizer_parameters.extend(parameters)
        parameter_groups.append(
            {
                "index": index,
                "parameter_tensors": len(parameters),
                "parameter_elements": sum(
                    int(parameter.numel()) for parameter in parameters
                ),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
            }
        )

    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    unique_optimizer_by_id = {
        id(parameter): parameter for parameter in optimizer_parameters
    }
    optimizer_id_set = set(optimizer_ids)

    state_tensor_count = 0
    state_tensor_bytes = 0
    state_tensor_devices: set[str] = set()
    state_tensor_dtypes: set[str] = set()
    for parameter_state in optimizer.state.values():
        if not isinstance(parameter_state, dict):
            continue
        for value in parameter_state.values():
            if not hasattr(value, "numel") or not hasattr(value, "element_size"):
                continue
            state_tensor_count += 1
            state_tensor_bytes += int(value.numel()) * int(value.element_size())
            state_tensor_devices.add(str(value.device))
            state_tensor_dtypes.add(str(value.dtype))

    quantization_maps = []
    for name, value in sorted(getattr(optimizer, "name2qmap", {}).items()):
        quantization_maps.append(
            {
                "name": str(name),
                "elements": int(value.numel()),
                "bytes": int(value.numel()) * int(value.element_size()),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )

    stats = {
        "optimizer_class": type(optimizer).__name__,
        "optimizer_module": type(optimizer).__module__,
        "optimizer_bits": int(getattr(optimizer.args, "optim_bits")),
        "is_paged": bool(getattr(optimizer, "is_paged", False)),
        "optimizer_initialized_flag": bool(
            getattr(optimizer, "initialized", False)
        ),
        "parameter_group_count": len(parameter_groups),
        "parameter_groups": parameter_groups,
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "optimizer_parameter_elements": sum(
            int(parameter.numel()) for parameter in optimizer_parameters
        ),
        "unique_optimizer_parameter_tensors": len(unique_optimizer_by_id),
        "unique_optimizer_parameter_elements": sum(
            int(parameter.numel())
            for parameter in unique_optimizer_by_id.values()
        ),
        "trainable_model_tensors": len(trainable_parameters),
        "trainable_model_elements": sum(
            int(parameter.numel()) for parameter in trainable_parameters
        ),
        "missing_trainable_tensors": len(trainable_ids - optimizer_id_set),
        "frozen_optimizer_tensors": sum(
            not parameter.requires_grad for parameter in optimizer_parameters
        ),
        "duplicate_optimizer_references": len(optimizer_ids)
        - len(unique_optimizer_by_id),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_tensor_count": state_tensor_count,
        "optimizer_state_tensor_bytes": state_tensor_bytes,
        "optimizer_state_tensor_devices": sorted(state_tensor_devices),
        "optimizer_state_tensor_dtypes": sorted(state_tensor_dtypes),
        "quantization_maps": quantization_maps,
        "quantization_map_bytes": sum(item["bytes"] for item in quantization_maps),
        "gradients_attached": sum(
            parameter.grad is not None for parameter in trainable_parameters
        ),
        "trainable_lora_unchanged": True,
        "global_step": int(global_step),
    }
    return validate_optimizer_evidence(stats)


def build_logged_reward_evidence(
    *,
    sku_id: str,
    completions: Sequence[str],
    reward_names: Sequence[str],
    component_rewards: dict[str, Sequence[float]],
    reward_weights: Sequence[float],
    advantages: Sequence[float],
) -> dict:
    """Preserve the trainer's post-step completion/reward log without retokenizing."""
    expected = 8
    if len(completions) != expected or len(advantages) != expected:
        raise RuntimeError("one-update log must contain exactly eight completions")
    if len(reward_names) != len(reward_weights):
        raise RuntimeError("one-update reward names and weights are misaligned")
    if set(component_rewards) != set(reward_names):
        raise RuntimeError("one-update reward components are incomplete")
    if any(len(component_rewards[name]) != expected for name in reward_names):
        raise RuntimeError("one-update reward component lengths are misaligned")

    weighted_totals = [
        sum(
            float(component_rewards[name][index]) * float(weight)
            for name, weight in zip(reward_names, reward_weights)
        )
        for index in range(expected)
    ]
    normalized_advantages = [float(value) for value in advantages]
    numeric_values = weighted_totals + normalized_advantages + [
        float(value)
        for name in reward_names
        for value in component_rewards[name]
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError("one-update log contains a non-finite reward or advantage")

    return {
        "component_reward_names": list(reward_names),
        "weighted_totals": weighted_totals,
        "weighted_total_unique_count": len(set(weighted_totals)),
        "weighted_total_has_variance": len(set(weighted_totals)) > 1,
        "advantages": normalized_advantages,
        "nonzero_advantage_count": sum(
            not math.isclose(value, 0.0, abs_tol=1e-8)
            for value in normalized_advantages
        ),
        "records": [
            {
                "sku_id": sku_id,
                "rollout_index": index,
                "raw_output": completion,
                "component_rewards": {
                    name: float(component_rewards[name][index])
                    for name in reward_names
                },
                "weighted_total": weighted_totals[index],
                "advantage": normalized_advantages[index],
            }
            for index, completion in enumerate(completions)
        ],
    }


def snapshot_trainable_parameters(model: object) -> dict:
    """Copy the live trainable LoRA to CPU for a one-update delta audit."""
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(snapshot) != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("one-update snapshot has an unexpected tensor count")
    if sum(int(tensor.numel()) for tensor in snapshot.values()) != (
        LOCKED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("one-update snapshot has an unexpected element count")
    return snapshot


def validate_parameter_update_evidence(stats: dict) -> dict:
    """Fail unless exactly one update made a finite change to the locked LoRA."""
    if stats.get("trainable_tensors") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("update evidence has an unexpected tensor count")
    if stats.get("trainable_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("update evidence has an unexpected element count")
    if stats.get("changed_tensors") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("one update did not change every LoRA tensor")
    if stats.get("changed_elements", 0) <= 0:
        raise RuntimeError("one update changed no LoRA elements")
    if stats.get("nonfinite_after_elements") != 0:
        raise RuntimeError("one update produced a non-finite LoRA value")
    if stats.get("nonfinite_delta_elements") != 0:
        raise RuntimeError("one update produced a non-finite LoRA delta")
    delta_l2 = float(stats.get("global_delta_l2_norm", float("nan")))
    if not math.isfinite(delta_l2) or delta_l2 <= 0:
        raise RuntimeError("one-update LoRA delta norm is not finite and positive")
    return {
        **stats,
        "all_lora_tensors_changed": True,
        "all_updated_weights_finite": True,
        "has_finite_nonzero_update": True,
    }


def inspect_parameter_update(model: object, before: dict) -> dict:
    """Measure the exact CPU delta between pre-step and post-step LoRA tensors."""
    names_seen = set()
    changed_tensors = 0
    changed_elements = 0
    nonfinite_after_elements = 0
    nonfinite_delta_elements = 0
    global_delta_squared_l2 = 0.0
    global_before_squared_l2 = 0.0
    global_max_abs_delta = 0.0
    global_sum_abs_delta = 0.0
    trainable_elements = 0
    by_target = {
        target: {
            "trainable_tensors": 0,
            "changed_tensors": 0,
            "trainable_elements": 0,
            "changed_elements": 0,
            "squared_delta_l2": 0.0,
            "max_abs_delta": 0.0,
        }
        for target in sorted(LOCKED_TARGET_MODULES)
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names_seen.add(name)
        if name not in before:
            raise RuntimeError(f"post-update LoRA tensor was absent before step: {name}")
        old = before[name]
        current = parameter.detach().cpu()
        if current.shape != old.shape or current.dtype != old.dtype:
            raise RuntimeError(f"LoRA metadata changed during update: {name}")
        matched_targets = [
            target
            for target in LOCKED_TARGET_MODULES
            if f".{target}." in f".{name}."
        ]
        if len(matched_targets) != 1:
            raise RuntimeError(f"updated LoRA tensor has an unknown target: {name}")
        target_stats = by_target[matched_targets[0]]
        elements = int(current.numel())
        trainable_elements += elements
        target_stats["trainable_tensors"] += 1
        target_stats["trainable_elements"] += elements

        old_float = old.float()
        current_float = current.float()
        delta = current_float - old_float
        changed = int(delta.count_nonzero().item())
        changed_elements += changed
        target_stats["changed_elements"] += changed
        if changed > 0:
            changed_tensors += 1
            target_stats["changed_tensors"] += 1
        nonfinite_after_elements += elements - int(current.isfinite().sum().item())
        nonfinite_delta_elements += elements - int(delta.isfinite().sum().item())
        delta_l2 = float(delta.norm(2).item())
        before_l2 = float(old_float.norm(2).item())
        max_abs_delta = float(delta.abs().max().item())
        sum_abs_delta = float(delta.abs().sum().item())
        global_delta_squared_l2 += delta_l2**2
        global_before_squared_l2 += before_l2**2
        global_max_abs_delta = max(global_max_abs_delta, max_abs_delta)
        global_sum_abs_delta += sum_abs_delta
        target_stats["squared_delta_l2"] += delta_l2**2
        target_stats["max_abs_delta"] = max(
            target_stats["max_abs_delta"], max_abs_delta
        )
        del old_float, current_float, delta

    if names_seen != set(before):
        raise RuntimeError("one-update LoRA tensor names changed")
    for target_stats in by_target.values():
        target_stats["delta_l2_norm"] = math.sqrt(
            target_stats.pop("squared_delta_l2")
        )
    global_delta_l2 = math.sqrt(global_delta_squared_l2)
    global_before_l2 = math.sqrt(global_before_squared_l2)
    stats = {
        "trainable_tensors": len(names_seen),
        "trainable_elements": trainable_elements,
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "unchanged_elements": trainable_elements - changed_elements,
        "nonfinite_after_elements": nonfinite_after_elements,
        "nonfinite_delta_elements": nonfinite_delta_elements,
        "global_delta_l2_norm": global_delta_l2,
        "global_before_l2_norm": global_before_l2,
        "relative_delta_l2": global_delta_l2 / global_before_l2,
        "global_max_abs_delta": global_max_abs_delta,
        "global_mean_abs_delta": global_sum_abs_delta / trainable_elements,
        "by_target_module": by_target,
    }
    return validate_parameter_update_evidence(stats)


def validate_initialized_optimizer_evidence(
    stats: dict,
    *,
    expected_step: int = 1,
) -> dict:
    """Fail unless an update initialized complete finite 8-bit optimizer state."""
    if expected_step <= 0:
        raise ValueError("expected optimizer step must be positive")
    if stats.get("optimizer_class") != "AdamW":
        raise RuntimeError("updated optimizer is not bitsandbytes AdamW")
    if stats.get("optimizer_bits") != 8 or stats.get("is_paged"):
        raise RuntimeError("updated optimizer is not non-paged 8-bit AdamW")
    if not stats.get("optimizer_initialized_flag"):
        raise RuntimeError("optimizer did not initialize during the first step")
    if stats.get("state_parameter_entries") != LOCKED_TRAINABLE_TENSORS:
        raise RuntimeError("optimizer state does not cover every LoRA tensor")
    if stats.get("missing_trainable_state_entries") != 0:
        raise RuntimeError("at least one trainable LoRA tensor has no optimizer state")
    if stats.get("foreign_state_entries") != 0:
        raise RuntimeError("optimizer state includes a frozen or foreign tensor")
    if stats.get("state_step_values") != [expected_step]:
        step_label = "one" if expected_step == 1 else str(expected_step)
        raise RuntimeError(
            f"optimizer state is not exactly at step {step_label}"
        )
    if stats.get("state1_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("first-moment optimizer state has the wrong size")
    if stats.get("state2_elements") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("second-moment optimizer state has the wrong size")
    if stats.get("state1_dtypes") != ["torch.uint8"]:
        raise RuntimeError("first-moment optimizer state is not fully 8-bit")
    if stats.get("state2_dtypes") != ["torch.uint8"]:
        raise RuntimeError("second-moment optimizer state is not fully 8-bit")
    if stats.get("absmax1_elements") != stats.get("expected_quantization_blocks"):
        raise RuntimeError("first-moment quantization scales have the wrong size")
    if stats.get("absmax2_elements") != stats.get("expected_quantization_blocks"):
        raise RuntimeError("second-moment quantization scales have the wrong size")
    if stats.get("nonfinite_state_elements") != 0:
        raise RuntimeError("optimizer state contains NaN or infinity")
    if stats.get("unique_state_tensor_bytes", 0) <= 0:
        raise RuntimeError("optimizer state allocated no tensor storage")
    return {
        **stats,
        "state_covers_exact_locked_lora": True,
        "state_initialized_at_expected_step": True,
        "state_initialized_at_step_one": expected_step == 1,
        "state_is_finite": True,
    }


def inspect_initialized_optimizer(
    optimizer: object,
    model: object,
    *,
    expected_step: int = 1,
) -> dict:
    """Measure bitsandbytes state after an exact expected optimizer step."""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    state_ids = {id(parameter) for parameter in optimizer.state}
    state_key_counts: dict[str, int] = {}
    state_step_values: set[int] = set()
    unique_state_tensors = {}
    state_tensor_references = 0
    nonfinite_state_elements = 0
    specialized_elements = {
        "state1": 0,
        "state2": 0,
        "absmax1": 0,
        "absmax2": 0,
    }
    specialized_dtypes = {"state1": set(), "state2": set()}

    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            state_key_counts[key] = state_key_counts.get(key, 0) + 1
            if key == "step":
                state_step_values.add(int(value))
                continue
            if not hasattr(value, "numel") or not hasattr(value, "element_size"):
                continue
            state_tensor_references += 1
            unique_state_tensors[id(value)] = value
            if key in specialized_elements:
                specialized_elements[key] += int(value.numel())
            if key in specialized_dtypes:
                specialized_dtypes[key].add(str(value.dtype))
            if value.is_floating_point():
                nonfinite_state_elements += int(value.numel()) - int(
                    value.isfinite().sum().item()
                )

    expected_blocks = sum(
        (int(parameter.numel()) + 255) // 256
        for parameter in trainable_parameters
    )
    stats = {
        "optimizer_class": type(optimizer).__name__,
        "optimizer_module": type(optimizer).__module__,
        "optimizer_bits": int(getattr(optimizer.args, "optim_bits")),
        "is_paged": bool(getattr(optimizer, "is_paged", False)),
        "optimizer_initialized_flag": bool(
            getattr(optimizer, "initialized", False)
        ),
        "state_parameter_entries": len(optimizer.state),
        "missing_trainable_state_entries": len(trainable_ids - state_ids),
        "foreign_state_entries": len(state_ids - trainable_ids),
        "state_step_values": sorted(state_step_values),
        "state_key_counts": dict(sorted(state_key_counts.items())),
        "state_tensor_references": state_tensor_references,
        "unique_state_tensors": len(unique_state_tensors),
        "unique_state_tensor_bytes": sum(
            int(value.numel()) * int(value.element_size())
            for value in unique_state_tensors.values()
        ),
        "unique_state_tensor_devices": sorted(
            {str(value.device) for value in unique_state_tensors.values()}
        ),
        "unique_state_tensor_dtypes": sorted(
            {str(value.dtype) for value in unique_state_tensors.values()}
        ),
        "state1_elements": specialized_elements["state1"],
        "state2_elements": specialized_elements["state2"],
        "absmax1_elements": specialized_elements["absmax1"],
        "absmax2_elements": specialized_elements["absmax2"],
        "expected_quantization_blocks": expected_blocks,
        "state1_dtypes": sorted(specialized_dtypes["state1"]),
        "state2_dtypes": sorted(specialized_dtypes["state2"]),
        "nonfinite_state_elements": nonfinite_state_elements,
        "gradients_attached_after_step": sum(
            parameter.grad is not None for parameter in trainable_parameters
        ),
    }
    if stats["gradients_attached_after_step"] != 0:
        raise RuntimeError("trainer left gradients attached after the first step")
    return validate_initialized_optimizer_evidence(
        stats,
        expected_step=expected_step,
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sku_sha256(sku_ids: Sequence[str]) -> str:
    payload = "\n".join(sku_ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sku_set_sha256(sku_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(sku_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def build_completed_smoke_context(
    *,
    preflight_report: dict,
    config_settings: dict,
    trainability: dict,
    global_step: int,
    optimizer_steps: int,
    rollout_records: int,
    starting_lora_sha256: str,
    final_lora_sha256: str,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    disk_free_after_bytes: int,
) -> dict:
    """Translate measured trainer state into the publisher's locked schema."""
    context = {
        "git": dict(preflight_report["git"]),
        "source_lock": {
            "starting_adapter_sha256": preflight_report["sft_lock"][
                "adapter_sha256"
            ],
            "fixture_data_sha256": preflight_report["fixture"]["data_sha256"],
            "fixture_manifest_sha256": preflight_report["fixture"][
                "manifest_sha256"
            ],
            "selection_manifest_sha256": preflight_report["sft_lock"][
                "selection_manifest_sha256"
            ],
        },
        "config": dict(config_settings),
        "runtime": {
            "global_step": int(global_step),
            "trainable_tensors": int(trainability["trainable_tensors"]),
            "trainable_parameters": int(trainability["trainable_parameters"]),
            "starting_lora_sha256": starting_lora_sha256,
            "final_lora_sha256": final_lora_sha256,
            "source_adapter_unchanged": True,
            "optimizer_steps": int(optimizer_steps),
            "rollout_records": int(rollout_records),
        },
        "resources": {
            "peak_allocated_bytes": int(peak_allocated_bytes),
            "peak_reserved_bytes": int(peak_reserved_bytes),
            "disk_free_after_bytes": int(disk_free_after_bytes),
        },
    }
    validate_smoke_context(context)
    return context


def save_and_publish_completed_smoke(
    *,
    model: object,
    tokenizer: object,
    source_adapter_file: str | Path,
    staging_dir: str | Path,
    final_output_dir: str | Path,
    records: Sequence[dict],
    trainer_log_history: Sequence[dict],
    expected_sku_ids: Sequence[str],
    preflight_report: dict,
    config_settings: dict,
    trainability: dict,
    global_step: int,
    optimizer_steps: int,
    starting_lora_sha256: str,
    final_lora_sha256: str,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    disk_usage_fn: Callable[[Path], object] | None = None,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Save only the live LoRA/tokenizer and atomically publish its evidence."""
    staging_dir = Path(staging_dir).resolve()
    final_output_dir = Path(final_output_dir).resolve()
    source_adapter_file = Path(source_adapter_file).resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"final smoke output already exists: {final_output_dir}"
        )
    expected_staging_prefix = f".{final_output_dir.name}.staging-"
    if (
        staging_dir.parent != final_output_dir.parent
        or not staging_dir.name.startswith(expected_staging_prefix)
    ):
        raise ValueError("smoke staging directory is not bound to final output")
    if not staging_dir.is_dir() or staging_dir.is_symlink():
        raise FileNotFoundError("smoke staging directory is absent")
    if any(staging_dir.iterdir()):
        raise ValueError("smoke staging directory must be empty before adapter save")

    expected_source_sha256 = preflight_report["sft_lock"]["adapter_sha256"]
    if _sha256_file(source_adapter_file) != expected_source_sha256:
        raise RuntimeError("source adapter changed before final save")
    model_save = getattr(model, "save_pretrained", None)
    tokenizer_save = getattr(tokenizer, "save_pretrained", None)
    if not callable(model_save) or not callable(tokenizer_save):
        raise TypeError("model and tokenizer must expose save_pretrained()")

    adapter_dir = staging_dir / "adapter"
    model_save(adapter_dir, safe_serialization=True)
    tokenizer_save(adapter_dir)
    if _sha256_file(source_adapter_file) != expected_source_sha256:
        raise RuntimeError("source adapter changed while saving smoke output")

    disk_usage = (disk_usage_fn or shutil.disk_usage)(staging_dir)
    context = build_completed_smoke_context(
        preflight_report=preflight_report,
        config_settings=config_settings,
        trainability=trainability,
        global_step=global_step,
        optimizer_steps=optimizer_steps,
        rollout_records=len(records),
        starting_lora_sha256=starting_lora_sha256,
        final_lora_sha256=final_lora_sha256,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        disk_free_after_bytes=int(disk_usage.free),
    )
    return write_and_publish_smoke_bundle(
        staging_dir=staging_dir,
        final_output_dir=final_output_dir,
        records=records,
        trainer_log_history=trainer_log_history,
        expected_sku_ids=expected_sku_ids,
        context=context,
        expected_adapter_model_bytes=expected_adapter_model_bytes,
    )


def run_five_step_smoke_orchestration(
    *,
    trainer: object,
    tokenizer: object,
    source_adapter_file: str | Path,
    final_output_dir: str | Path,
    expected_sku_ids: Sequence[str],
    preflight_report: dict,
    config_settings: dict,
    cuda_snapshot_fn: Callable[[], dict],
    trainability_fn: Callable[[object], dict] = inspect_model_trainability,
    parameter_values_fn: Callable[[object], dict] = (
        inspect_trainable_parameter_values
    ),
    fingerprint_fn: Callable[[object], str] = _trainable_parameter_sha256,
    optimizer_inspector_fn: Callable[..., dict] = inspect_initialized_optimizer,
    disk_usage_fn: Callable[[Path], object] | None = None,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Run, validate and publish one already-constructed capturing trainer."""
    final_output_dir = Path(final_output_dir).resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"final smoke output already exists: {final_output_dir}"
        )
    collector = getattr(trainer, "smoke_rollout_collector", None)
    if not isinstance(collector, SmokeRolloutCollector):
        raise TypeError("five-step trainer has no SmokeRolloutCollector")
    if collector.expected_sku_ids != tuple(expected_sku_ids):
        raise RuntimeError("trainer collector SKU order drifted")
    state = getattr(trainer, "state", None)
    if int(getattr(state, "global_step", -1)) != 0:
        raise RuntimeError("five-step trainer did not start at global_step zero")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("five-step trainer started with an optimizer")
    if getattr(trainer, "lr_scheduler", None) is not None:
        raise RuntimeError("five-step trainer started with an LR scheduler")
    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("five-step trainer has no model")

    trainability_before = trainability_fn(model)
    starting_lora_sha256 = fingerprint_fn(model)
    if not callable(cuda_snapshot_fn):
        raise TypeError("five-step orchestration requires CUDA measurement")
    cuda_before_train = cuda_snapshot_fn()
    train_started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_started
    cuda_after_train = cuda_snapshot_fn()

    completed_step = int(getattr(trainer.state, "global_step", -1))
    if completed_step != EXPECTED_STEPS:
        raise RuntimeError("five-step trainer did not finish exactly five updates")
    if int(getattr(train_result, "global_step", -1)) != EXPECTED_STEPS:
        raise RuntimeError("trainer result disagrees with completed global step")
    if getattr(trainer, "optimizer", None) is None:
        raise RuntimeError("five-step trainer did not construct an optimizer")
    if getattr(trainer, "lr_scheduler", None) is None:
        raise RuntimeError("five-step trainer did not construct an LR scheduler")

    rollout_evidence = collector.finalize()
    log_history = [dict(entry) for entry in trainer.state.log_history]
    trainer_log_validation = validate_trainer_log_history(log_history)
    trainability_after = trainability_fn(model)
    if trainability_after["trainable_tensors"] != trainability_before[
        "trainable_tensors"
    ]:
        raise RuntimeError("trainable tensor count changed during five-step smoke")
    if trainability_after["trainable_parameters"] != trainability_before[
        "trainable_parameters"
    ]:
        raise RuntimeError("trainable parameter count changed during five-step smoke")
    before_names = trainability_before.get("trainable_parameter_names")
    after_names = trainability_after.get("trainable_parameter_names")
    if before_names is not None and after_names != before_names:
        raise RuntimeError("trainable parameter names changed during five-step smoke")
    parameter_values = parameter_values_fn(model)
    final_lora_sha256 = fingerprint_fn(model)
    if final_lora_sha256 == starting_lora_sha256:
        raise RuntimeError("five-step smoke changed no trainable LoRA bytes")

    wrapped_optimizer = trainer.optimizer
    base_optimizer = getattr(wrapped_optimizer, "optimizer", wrapped_optimizer)
    optimizer_state = optimizer_inspector_fn(
        base_optimizer,
        model,
        expected_step=EXPECTED_STEPS,
    )
    cuda_after_audit = cuda_snapshot_fn()
    peak_allocated_bytes = int(
        cuda_after_audit.get("torch_peak_allocated_bytes", 0)
    )
    peak_reserved_bytes = int(
        cuda_after_audit.get("torch_peak_reserved_bytes", 0)
    )
    if peak_allocated_bytes <= 0 or peak_reserved_bytes < peak_allocated_bytes:
        raise RuntimeError("five-step trainer produced invalid CUDA peak evidence")

    staging_dir = create_staging_output(final_output_dir)
    manifest = save_and_publish_completed_smoke(
        model=model,
        tokenizer=tokenizer,
        source_adapter_file=source_adapter_file,
        staging_dir=staging_dir,
        final_output_dir=final_output_dir,
        records=rollout_evidence["records"],
        trainer_log_history=log_history,
        expected_sku_ids=expected_sku_ids,
        preflight_report=preflight_report,
        config_settings=config_settings,
        trainability=trainability_after,
        global_step=completed_step,
        optimizer_steps=completed_step,
        starting_lora_sha256=starting_lora_sha256,
        final_lora_sha256=final_lora_sha256,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        disk_usage_fn=disk_usage_fn,
        expected_adapter_model_bytes=expected_adapter_model_bytes,
    )
    train_metrics = dict(getattr(train_result, "metrics", {}))
    return {
        "status": "passed",
        "global_step": completed_step,
        "optimizer_steps": completed_step,
        "train_seconds": train_seconds,
        "train_metrics": train_metrics,
        "trainability_before": trainability_before,
        "trainability_after": trainability_after,
        "parameter_values": parameter_values,
        "starting_lora_sha256": starting_lora_sha256,
        "final_lora_sha256": final_lora_sha256,
        "rollout_validation": rollout_evidence["validation"],
        "trainer_log_validation": trainer_log_validation,
        "optimizer_state": optimizer_state,
        "cuda_before_train": cuda_before_train,
        "cuda_after_train": cuda_after_train,
        "cuda_after_audit": cuda_after_audit,
        "manifest": manifest,
        "final_output_dir": str(final_output_dir),
        "published": True,
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def inspect_git_state(repo_root: Path) -> dict:
    """Resolve the exact tracked code state while allowing untracked run files."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_dirty = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            check=False,
        ).returncode != 0
        index_dirty = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
            check=False,
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not resolve Git state") from exc
    return {
        "commit": commit,
        "tracked_worktree_dirty": worktree_dirty,
        "index_dirty": index_dirty,
    }


def verify_fixture(
    *,
    fixture_data_path: Path,
    fixture_manifest_path: Path,
    expected_data_sha256: str,
    expected_manifest_sha256: str,
) -> dict:
    actual_manifest_sha = _sha256_file(fixture_manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise RuntimeError("locked smoke fixture manifest checksum mismatch")
    manifest = _read_json(fixture_manifest_path)
    if manifest.get("version") != "grpo-smoke-v1":
        raise RuntimeError("unexpected smoke fixture manifest version")
    if not all(manifest.get("invariants", {}).values()):
        raise RuntimeError("smoke fixture manifest contains a failed invariant")

    actual_data_sha = _sha256_file(fixture_data_path)
    if actual_data_sha != expected_data_sha256:
        raise RuntimeError("locked smoke fixture data checksum mismatch")
    if actual_data_sha != manifest.get("output", {}).get("smoke_dataset_sha256"):
        raise RuntimeError("smoke fixture data disagrees with its manifest")

    rows = _read_jsonl_objects(fixture_data_path)
    expected_rows = manifest.get("selection", {}).get("selected_rows")
    if len(rows) != expected_rows or len(rows) != 5:
        raise RuntimeError("smoke fixture must contain exactly five rows")
    sku_ids = [row.get("sku_id") for row in rows]
    if any(not isinstance(sku_id, str) or not sku_id for sku_id in sku_ids):
        raise RuntimeError("smoke fixture contains a missing SKU ID")
    if len(set(sku_ids)) != len(sku_ids):
        raise RuntimeError("smoke fixture contains duplicate SKU IDs")
    if sku_ids != manifest["selection"]["selected_skus_in_step_order"]:
        raise RuntimeError("smoke fixture SKU order disagrees with its manifest")
    if _ordered_sku_sha256(sku_ids) != manifest["selection"][
        "selected_sku_order_sha256"
    ]:
        raise RuntimeError("smoke fixture ordered-SKU checksum mismatch")
    if any(row.get("split") != "train" for row in rows):
        raise RuntimeError("smoke fixture contains a non-training row")
    if any(row.get("difficulty", {}).get("sft_pass_rate") != 0.5 for row in rows):
        raise RuntimeError("smoke fixture contains a row outside pass rate 0.5")

    return {
        "data_path": str(fixture_data_path),
        "data_sha256": actual_data_sha,
        "manifest_path": str(fixture_manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "rows": len(rows),
        "ordered_sku_sha256": manifest["selection"][
            "selected_sku_order_sha256"
        ],
        "sku_ids_in_step_order": sku_ids,
    }


def verify_full_run_pool(
    *,
    repo_root: Path,
    data_path: Path,
    manifest_path: Path,
    expected_data_sha256: str = LOCKED_FULL_RUN_DATA_SHA256,
    expected_manifest_sha256: str = LOCKED_FULL_RUN_MANIFEST_SHA256,
    expected_rows: int = FULL_RUN_DATA_ROWS,
    expected_bytes: int = FULL_RUN_DATA_BYTES,
) -> dict:
    """Verify the complete cap-four prompt pool and its committed lineage."""
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise RuntimeError("locked full-run pool manifest checksum mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("version") != "grpo-pool-cap4-v1":
        raise RuntimeError("unexpected full-run pool manifest version")
    invariants = manifest.get("invariants")
    if not isinstance(invariants, dict) or not invariants or not all(
        invariants.values()
    ):
        raise RuntimeError("full-run pool manifest contains a failed invariant")

    output = manifest.get("output", {})
    manifest_data_path = _resolve(repo_root, output.get("active_dataset", ""))
    if manifest_data_path != data_path:
        raise RuntimeError("full-run data path disagrees with pool manifest")
    actual_data_sha = _sha256_file(data_path)
    if actual_data_sha != expected_data_sha256:
        raise RuntimeError("locked full-run data checksum mismatch")
    if actual_data_sha != output.get("active_dataset_sha256"):
        raise RuntimeError("full-run data checksum disagrees with pool manifest")
    if data_path.stat().st_size != expected_bytes:
        raise RuntimeError("locked full-run data byte size mismatch")
    if data_path.stat().st_size != output.get("active_dataset_bytes"):
        raise RuntimeError("full-run data byte size disagrees with pool manifest")

    rows = _read_jsonl_objects(data_path)
    if len(rows) != expected_rows or len(rows) != output.get("active_dataset_rows"):
        raise RuntimeError("full-run data must contain exactly 1,565 rows")
    sku_ids = [row.get("sku_id") for row in rows]
    if any(not isinstance(sku_id, str) or not sku_id for sku_id in sku_ids):
        raise RuntimeError("full-run data contains a missing SKU ID")
    if len(set(sku_ids)) != len(sku_ids):
        raise RuntimeError("full-run data contains duplicate SKU IDs")
    if any(row.get("split") != "train" for row in rows):
        raise RuntimeError("full-run data contains a non-training row")
    pass_rates = [row.get("difficulty", {}).get("sft_pass_rate") for row in rows]
    if any(
        not isinstance(value, (int, float)) or not 0 < float(value) < 1
        for value in pass_rates
    ):
        raise RuntimeError("full-run data contains an ineligible pass rate")

    selection = manifest.get("selection", {})
    manifest_skus = selection.get("active_skus_in_source_order")
    if sku_ids != manifest_skus:
        raise RuntimeError("full-run SKU order disagrees with pool manifest")
    actual_sku_set_sha = _sku_set_sha256(sku_ids)
    if actual_sku_set_sha != selection.get("active_sku_set_sha256"):
        raise RuntimeError("full-run SKU set disagrees with pool manifest")
    if selection.get("active_rows") != expected_rows:
        raise RuntimeError("full-run active-row count disagrees with pool manifest")

    policy = manifest.get("policy", {})
    if (
        policy.get("family_cap") != 4
        or policy.get("selection_seed") != 42
        or policy.get("eligibility_rule") != "0 < sft_pass_rate < 1"
    ):
        raise RuntimeError("full-run pool policy drifted")

    return {
        "data_path": str(data_path),
        "data_sha256": actual_data_sha,
        "data_bytes": data_path.stat().st_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "rows": len(rows),
        "ordered_sku_sha256": _ordered_sku_sha256(sku_ids),
        "sku_set_sha256": actual_sku_set_sha,
        "family_cap": policy["family_cap"],
        "selection_seed": policy["selection_seed"],
        "manifest_invariants": dict(invariants),
    }


def verify_sft_lock(
    *,
    repo_root: Path,
    selection_manifest_path: Path,
    adapter_path: Path,
    expected_selection_sha256: str,
    expected_adapter_sha256: str,
) -> dict:
    actual_selection_sha = _sha256_file(selection_manifest_path)
    if actual_selection_sha != expected_selection_sha256:
        raise RuntimeError("SFT selection manifest checksum mismatch")
    selection = _read_json(selection_manifest_path)
    if selection.get("status") != "locked_before_frozen_eval":
        raise RuntimeError("SFT selection manifest is not in the locked state")

    selected = selection.get("selected_checkpoint", {})
    locked_adapter_path = _resolve(repo_root, selected.get("remote_path", ""))
    if locked_adapter_path != adapter_path:
        raise RuntimeError("adapter path disagrees with the SFT selection lock")
    if selected.get("base_model") != LOCKED_BASE_MODEL:
        raise RuntimeError("base model disagrees with the SFT selection lock")

    lora = selected.get("lora", {})
    if lora.get("rank") != LOCKED_LORA_RANK:
        raise RuntimeError("LoRA rank disagrees with the SFT selection lock")
    if lora.get("alpha") != LOCKED_LORA_ALPHA:
        raise RuntimeError("LoRA alpha disagrees with the SFT selection lock")
    if set(lora.get("target_modules", [])) != LOCKED_TARGET_MODULES:
        raise RuntimeError("LoRA targets disagree with the SFT selection lock")
    if lora.get("trainable_parameters") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("trainable-parameter expectation disagrees with lock")

    weights = selected.get("adapter_weights", {})
    adapter_file = adapter_path / weights.get("file", "adapter_model.safetensors")
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    actual_adapter_sha = _sha256_file(adapter_file)
    if actual_adapter_sha != expected_adapter_sha256:
        raise RuntimeError("locked SFT adapter checksum mismatch")
    if actual_adapter_sha != weights.get("sha256"):
        raise RuntimeError("adapter checksum disagrees with SFT selection manifest")
    if adapter_file.stat().st_size != weights.get("bytes"):
        raise RuntimeError("adapter byte size disagrees with SFT selection manifest")

    adapter_config_path = adapter_path / "adapter_config.json"
    config = _read_json(adapter_config_path)
    if config.get("base_model_name_or_path") != LOCKED_BASE_MODEL:
        raise RuntimeError("adapter config names an unexpected base model")
    if config.get("r") != LOCKED_LORA_RANK:
        raise RuntimeError("adapter config has an unexpected LoRA rank")
    if config.get("lora_alpha") != LOCKED_LORA_ALPHA:
        raise RuntimeError("adapter config has an unexpected LoRA alpha")
    if config.get("lora_dropout") != 0 or config.get("bias") != "none":
        raise RuntimeError("adapter config has unexpected dropout or bias")
    if set(config.get("target_modules", [])) != LOCKED_TARGET_MODULES:
        raise RuntimeError("adapter config has unexpected target modules")

    return {
        "selection_manifest": str(selection_manifest_path),
        "selection_manifest_sha256": actual_selection_sha,
        "adapter_path": str(adapter_path),
        "adapter_file": str(adapter_file),
        "adapter_bytes": adapter_file.stat().st_size,
        "adapter_sha256": actual_adapter_sha,
        "adapter_config": str(adapter_config_path),
        "base_model": LOCKED_BASE_MODEL,
        "lora_rank": LOCKED_LORA_RANK,
        "lora_alpha": LOCKED_LORA_ALPHA,
        "target_modules": sorted(LOCKED_TARGET_MODULES),
        "trainable_parameters_expected": LOCKED_TRAINABLE_PARAMETERS,
        "runtime_trainable_parameter_assertion_required": True,
    }


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"no existing parent for disk check: {path}")
    return candidate


def run_preflight(
    *,
    repo_root: str | Path,
    fixture_data: str | Path,
    fixture_manifest: str | Path,
    selection_manifest: str | Path,
    adapter: str | Path,
    output_dir: str | Path,
    minimum_free_bytes: int,
    expected_commit: str | None = None,
    expected_fixture_data_sha256: str = LOCKED_FIXTURE_DATA_SHA256,
    expected_fixture_manifest_sha256: str = LOCKED_FIXTURE_MANIFEST_SHA256,
    expected_selection_manifest_sha256: str = LOCKED_SELECTION_MANIFEST_SHA256,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
    git_state_fn: Callable[[Path], dict] | None = None,
    disk_usage_fn: Callable[[Path], object] | None = None,
) -> dict:
    """Validate every CPU-visible smoke lock and return a read-only report."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    fixture_data_path = _resolve(repo_root, fixture_data)
    fixture_manifest_path = _resolve(repo_root, fixture_manifest)
    selection_manifest_path = _resolve(repo_root, selection_manifest)
    adapter_path = _resolve(repo_root, adapter)
    output_path = _resolve(repo_root, output_dir)

    git = (git_state_fn or inspect_git_state)(repo_root)
    if git.get("tracked_worktree_dirty") or git.get("index_dirty"):
        raise RuntimeError("tracked Git state must be clean before GRPO")
    if expected_commit is not None and git.get("commit") != expected_commit:
        raise RuntimeError("Git commit disagrees with expected smoke commit")
    if output_path.exists():
        raise FileExistsError(f"GRPO smoke output already exists: {output_path}")

    fixture = verify_fixture(
        fixture_data_path=fixture_data_path,
        fixture_manifest_path=fixture_manifest_path,
        expected_data_sha256=expected_fixture_data_sha256,
        expected_manifest_sha256=expected_fixture_manifest_sha256,
    )
    sft_lock = verify_sft_lock(
        repo_root=repo_root,
        selection_manifest_path=selection_manifest_path,
        adapter_path=adapter_path,
        expected_selection_sha256=expected_selection_manifest_sha256,
        expected_adapter_sha256=expected_adapter_sha256,
    )

    if minimum_free_bytes <= 0:
        raise ValueError("minimum free disk must be positive")
    disk_probe = _existing_parent(output_path.parent)
    usage = (disk_usage_fn or shutil.disk_usage)(disk_probe)
    free_bytes = int(usage.free)
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} < {minimum_free_bytes} bytes"
        )

    return {
        "version": PREFLIGHT_VERSION,
        "status": "passed",
        "git": git,
        "fixture": fixture,
        "sft_lock": sft_lock,
        "output": {
            "path": str(output_path),
            "collision_free": True,
            "created": False,
        },
        "disk": {
            "probe_path": str(disk_probe),
            "free_bytes": free_bytes,
            "minimum_free_bytes": minimum_free_bytes,
            "passes": True,
        },
        "cuda_imports_performed": False,
        "model_loaded": False,
        "trainer_constructed": False,
    }


def inspect_gpu_idle_state(
    *,
    command_runner: Callable[..., object] | None = None,
) -> dict:
    """Read one GPU's idle state through nvidia-smi without importing CUDA."""
    runner = command_runner or subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not inspect GPU state with nvidia-smi") from exc
    lines = [line.strip() for line in str(result.stdout).splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("full-run preflight requires exactly one visible GPU")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError("unexpected nvidia-smi output")
    try:
        index = int(parts[0])
        memory_used_mib = int(parts[2])
        utilization_percent = int(parts[3])
        temperature_c = int(parts[4])
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned non-numeric GPU measurements") from exc
    if min(index, memory_used_mib, utilization_percent, temperature_c) < 0:
        raise RuntimeError("nvidia-smi returned a negative GPU measurement")

    memory_idle = memory_used_mib <= FULL_RUN_MAX_IDLE_GPU_MEMORY_MIB
    utilization_idle = (
        utilization_percent <= FULL_RUN_MAX_IDLE_GPU_UTILIZATION_PERCENT
    )
    return {
        "source": "nvidia-smi",
        "device_index": index,
        "device_name": parts[1],
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "temperature_c": temperature_c,
        "maximum_idle_memory_mib": FULL_RUN_MAX_IDLE_GPU_MEMORY_MIB,
        "maximum_idle_utilization_percent": (
            FULL_RUN_MAX_IDLE_GPU_UTILIZATION_PERCENT
        ),
        "memory_idle": memory_idle,
        "utilization_idle": utilization_idle,
        "idle": memory_idle and utilization_idle,
        "cuda_imports_performed": False,
    }


def run_full_run_300_preflight(
    *,
    repo_root: str | Path,
    training_data: str | Path,
    pool_manifest: str | Path,
    selection_manifest: str | Path,
    adapter: str | Path,
    output_dir: str | Path,
    minimum_free_bytes: int,
    expected_commit: str,
    expected_data_sha256: str = LOCKED_FULL_RUN_DATA_SHA256,
    expected_pool_manifest_sha256: str = LOCKED_FULL_RUN_MANIFEST_SHA256,
    expected_selection_manifest_sha256: str = LOCKED_SELECTION_MANIFEST_SHA256,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
    expected_rows: int = FULL_RUN_DATA_ROWS,
    expected_bytes: int = FULL_RUN_DATA_BYTES,
    git_state_fn: Callable[[Path], dict] | None = None,
    disk_usage_fn: Callable[[Path], object] | None = None,
    gpu_state_fn: Callable[[], dict] | None = None,
    sft_lock_fn: Callable[..., dict] | None = None,
) -> dict:
    """Verify all read-only 300-step launch prerequisites before CUDA import."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    training_data_path = _resolve(repo_root, training_data)
    pool_manifest_path = _resolve(repo_root, pool_manifest)
    selection_manifest_path = _resolve(repo_root, selection_manifest)
    adapter_path = _resolve(repo_root, adapter)
    output_path = _resolve(repo_root, output_dir)

    git = (git_state_fn or inspect_git_state)(repo_root)
    if git.get("tracked_worktree_dirty") or git.get("index_dirty"):
        raise RuntimeError("tracked Git state must be clean before full GRPO")
    if git.get("commit") != expected_commit:
        raise RuntimeError("Git commit disagrees with expected full-run commit")
    if output_path.exists():
        raise FileExistsError(f"full GRPO output already exists: {output_path}")
    staging_pattern = f".{output_path.name}.staging-*"
    staging_collisions = sorted(
        str(path) for path in output_path.parent.glob(staging_pattern)
    )
    if staging_collisions:
        raise FileExistsError(
            "full GRPO staging output already exists: "
            + ", ".join(staging_collisions)
        )

    pool = verify_full_run_pool(
        repo_root=repo_root,
        data_path=training_data_path,
        manifest_path=pool_manifest_path,
        expected_data_sha256=expected_data_sha256,
        expected_manifest_sha256=expected_pool_manifest_sha256,
        expected_rows=expected_rows,
        expected_bytes=expected_bytes,
    )
    sft_lock = (sft_lock_fn or verify_sft_lock)(
        repo_root=repo_root,
        selection_manifest_path=selection_manifest_path,
        adapter_path=adapter_path,
        expected_selection_sha256=expected_selection_manifest_sha256,
        expected_adapter_sha256=expected_adapter_sha256,
    )

    if minimum_free_bytes <= 0:
        raise ValueError("minimum free disk must be positive")
    disk_probe = _existing_parent(output_path.parent)
    usage = (disk_usage_fn or shutil.disk_usage)(disk_probe)
    free_bytes = int(usage.free)
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} < {minimum_free_bytes} bytes"
        )

    gpu = (gpu_state_fn or inspect_gpu_idle_state)()
    if not gpu.get("idle"):
        raise RuntimeError("GPU is not idle enough for full GRPO launch")
    if gpu.get("cuda_imports_performed"):
        raise RuntimeError("GPU preflight unexpectedly imported CUDA")

    contract = full_run_300_contract(output_dir=output_path)
    return {
        "version": "grpo-full-run-300-preflight-v1",
        "status": "passed",
        "git": git,
        "pool": pool,
        "sft_lock": sft_lock,
        "output": {
            "path": str(output_path),
            "collision_free": True,
            "staging_pattern": staging_pattern,
            "staging_collision_free": True,
            "created": False,
        },
        "disk": {
            "probe_path": str(disk_probe),
            "free_bytes": free_bytes,
            "minimum_free_bytes": minimum_free_bytes,
            "passes": True,
        },
        "gpu": gpu,
        "contract_version": contract["version"],
        "contract_status": contract["status"],
        "cuda_imports_performed": False,
        "model_loaded": False,
        "trainer_constructed": False,
        "training_dispatch_enabled": False,
    }


def _cuda_memory_snapshot(torch_module: object) -> dict:
    """Capture one JSON-safe CUDA memory reading from the active device."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the GRPO model-load gate")
    device_index = int(cuda.current_device())
    properties = cuda.get_device_properties(device_index)
    free_bytes, total_bytes = cuda.mem_get_info(device_index)
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_bytes": int(properties.total_memory),
        "driver_free_bytes": int(free_bytes),
        "driver_used_bytes": int(total_bytes - free_bytes),
        "torch_allocated_bytes": int(cuda.memory_allocated(device_index)),
        "torch_reserved_bytes": int(cuda.memory_reserved(device_index)),
        "torch_peak_allocated_bytes": int(cuda.max_memory_allocated(device_index)),
        "torch_peak_reserved_bytes": int(cuda.max_memory_reserved(device_index)),
    }


def _load_locked_policy(
    FastLanguageModel: object,
    torch_module: object,
    adapter_path: Path,
):
    """Load the selected SFT adapter as trainable PEFT weights, never a fresh LoRA."""
    return FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),  # Locked PEFT checkpoint, not fresh Qwen.
        max_seq_length=MODEL_MAX_SEQUENCE_LENGTH,  # Measured SFT/GRPO ceiling.
        dtype=torch_module.bfloat16,  # Match the bf16 SFT policy on the RTX 3090.
        load_in_4bit=False,  # Continue the unquantized SFT adapter unchanged.
        local_files_only=True,  # Refuse downloads or remote revision drift.
        use_gradient_checkpointing="unsloth",  # Match planned GRPO memory mode.
        fast_inference=False,  # Do not start the colocated vLLM path yet.
    )


def run_model_load_gate(
    *,
    adapter_path: str | Path,
    adapter_file: str | Path,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
) -> dict:
    """Load and inspect the locked policy, then release it without training."""
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()

    # Unsloth must patch the model stack before Torch/Transformers/TRL paths are
    # used. No heavyweight import occurs unless the CPU-only preflight passed.
    from unsloth import FastLanguageModel

    import torch

    model = None
    tokenizer = None
    report = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_memory_snapshot(torch)
    started = time.perf_counter()
    try:
        model, tokenizer = _load_locked_policy(
            FastLanguageModel, torch, adapter_path
        )
        torch.cuda.synchronize()
        trainability = inspect_model_trainability(model)
        after_load = _cuda_memory_snapshot(torch)
        adapter_sha_after_load = _sha256_file(adapter_file)
        if adapter_sha_after_load != expected_adapter_sha256:
            raise RuntimeError("source adapter changed during model loading")
        report = {
            "version": MODEL_LOAD_VERSION,
            "status": "passed",
            "adapter_path": str(adapter_path),
            "adapter_file": str(adapter_file),
            "adapter_sha256_after_load": adapter_sha_after_load,
            "source_adapter_unchanged": True,
            "base_model": LOCKED_BASE_MODEL,
            "max_sequence_length": MODEL_MAX_SEQUENCE_LENGTH,
            "dtype_requested": "bfloat16",
            "load_in_4bit": False,
            "local_files_only": True,
            "gradient_checkpointing": "unsloth",
            "fast_inference": False,
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "load_seconds": time.perf_counter() - started,
            "trainability": trainability,
            "cuda_before_load": before,
            "cuda_after_load": after_load,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "generation_performed": False,
            "training_steps": 0,
        }
    finally:
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("model-load gate ended without a report")
    report["cuda_after_release"] = _cuda_memory_snapshot(torch)
    report["model_retained"] = False
    return report


def _run_trainer_gate(
    *,
    fixture_data_path: str | Path,
    adapter_path: str | Path,
    adapter_file: str | Path,
    expected_sku_ids: Sequence[str],
    perform_rollout: bool,
    perform_backward: bool,
    construct_optimizer: bool,
    perform_one_update: bool,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
) -> dict:
    """Construct the exact trainer and cross one explicitly selected gate."""
    if perform_backward and not perform_rollout:
        raise ValueError("backward requires a prepared rollout")
    if construct_optimizer and (perform_rollout or perform_backward):
        raise ValueError("optimizer construction must remain an isolated gate")
    if perform_one_update and (
        perform_rollout or perform_backward or construct_optimizer
    ):
        raise ValueError("one-update training must remain an isolated gate")
    fixture_data_path = Path(fixture_data_path).resolve()
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()

    # Import order is part of the remote dependency contract: Unsloth patches
    # the installed TRL/vLLM compatibility path before GRPOTrainer is imported.
    from unsloth import FastLanguageModel

    import torch
    from trl import GRPOConfig, GRPOTrainer

    from training.dataset import load_grpo_prompts
    from training.rewards import (
        FIRST_RUN_REWARD_FUNCTIONS,
        FIRST_RUN_REWARD_WEIGHTS,
    )
    from verifier import load_pack

    if tuple(FIRST_RUN_REWARD_WEIGHTS) != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("reward implementation weights drifted from GRPO lock")

    model = None
    tokenizer = None
    trainer = None
    dataset = None
    generation_batch = None
    prepared = None
    loss = None
    optimizer = None
    parameter_snapshot = None
    train_result = None
    report = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_memory_snapshot(torch)
    started = time.perf_counter()
    temporary_output_path = None
    try:
        model, tokenizer = _load_locked_policy(
            FastLanguageModel, torch, adapter_path
        )
        trainability_before = inspect_model_trainability(model)

        pack = load_pack("packs/vastraa_taste_v1")
        dataset = load_grpo_prompts(
            pack,
            fixture_data_path,
            require_pass_rate_band=True,
        )
        actual_sku_ids = list(dataset["sku_id"])
        if actual_sku_ids != list(expected_sku_ids):
            raise RuntimeError("trainer dataset SKU order drifted from smoke manifest")
        if len(dataset) != 5:
            raise RuntimeError(
                f"trainer dataset must contain five rows, found {len(dataset)}"
            )
        required_columns = {"prompt", "gold", "sku_id"}
        if set(dataset.column_names) != required_columns:
            raise RuntimeError(
                f"trainer dataset columns drifted: {dataset.column_names}"
            )

        # Trainer construction may initialize logging internals. A temporary
        # output keeps this no-training gate isolated from the reserved smoke path.
        with tempfile.TemporaryDirectory(prefix="grpo-trainer-construction-") as temp:
            temporary_output_path = Path(temp).resolve()
            config = GRPOConfig(
                **grpo_smoke_config_kwargs(
                    output_dir=temporary_output_path,
                    reward_weights=FIRST_RUN_REWARD_WEIGHTS,
                )
            )
            config_report = inspect_grpo_config(config)
            trainer = GRPOTrainer(
                model=model,  # Locked SFT policy with its existing trainable LoRA.
                reward_funcs=list(FIRST_RUN_REWARD_FUNCTIONS),  # Three plain rewards.
                args=config,  # Fully asserted five-step smoke configuration.
                train_dataset=dataset,  # Five deterministic pass-rate-0.5 prompts.
                processing_class=tokenizer,  # Qwen chat template and tokenization.
            )
            torch.cuda.synchronize()

            trainer_reward_names = [
                getattr(reward, "__name__", type(reward).__name__)
                for reward in trainer.reward_funcs
            ]
            expected_reward_names = [
                reward.__name__ for reward in FIRST_RUN_REWARD_FUNCTIONS
            ]
            if trainer_reward_names != expected_reward_names:
                raise RuntimeError("trainer reward function order drifted")
            raw_reward_weights = trainer.reward_weights
            if hasattr(raw_reward_weights, "tolist"):
                raw_reward_weights = raw_reward_weights.tolist()
            runtime_reward_weights = [
                float(weight) for weight in raw_reward_weights
            ]
            if runtime_reward_weights != list(LOCKED_REWARD_WEIGHTS):
                raise RuntimeError("trainer reward weights drifted")
            if trainer.optimizer is not None:
                raise RuntimeError(
                    "trainer construction unexpectedly created an optimizer"
                )
            if trainer.lr_scheduler is not None:
                raise RuntimeError(
                    "trainer construction unexpectedly created an LR scheduler"
                )
            if trainer.ref_model is not None:
                raise RuntimeError(
                    "beta=0 trainer construction unexpectedly created a reference model"
                )
            if int(trainer.state.global_step) != 0:
                raise RuntimeError(
                    "trainer construction unexpectedly advanced global_step"
                )
            trainer_dataset_columns = list(trainer.train_dataset.column_names)
            trainer_sku_ids = list(trainer.train_dataset["sku_id"])
            if set(trainer_dataset_columns) != required_columns:
                raise RuntimeError("trainer dropped a hidden reward/audit column")
            if trainer_sku_ids != actual_sku_ids:
                raise RuntimeError("trainer changed the deterministic SKU order")

            rollout_report = None
            gradient_report = None
            optimizer_report = None
            if construct_optimizer:
                if any(
                    parameter.grad is not None
                    for parameter in trainer.model.parameters()
                    if parameter.requires_grad
                ):
                    raise RuntimeError(
                        "LoRA gradients existed before optimizer construction"
                    )

                from bitsandbytes.optim import GlobalOptimManager

                global_optim_manager = GlobalOptimManager.get_instance()
                overrides_before = len(
                    global_optim_manager.module_weight_config_triple
                )
                lora_sha_before_optimizer = _trainable_parameter_sha256(
                    trainer.model
                )
                torch.cuda.reset_peak_memory_stats()
                cuda_before_optimizer = _cuda_memory_snapshot(torch)
                optimizer_started = time.perf_counter()
                optimizer = trainer.create_optimizer()
                torch.cuda.synchronize()
                optimizer_construction_seconds = (
                    time.perf_counter() - optimizer_started
                )
                cuda_after_optimizer = _cuda_memory_snapshot(torch)
                if optimizer is None or trainer.optimizer is not optimizer:
                    raise RuntimeError("trainer did not retain the created optimizer")
                lora_sha_after_optimizer = _trainable_parameter_sha256(
                    trainer.model
                )
                if lora_sha_after_optimizer != lora_sha_before_optimizer:
                    raise RuntimeError(
                        "LoRA weights changed during optimizer construction"
                    )
                optimizer_stats = inspect_fresh_optimizer(
                    optimizer,
                    trainer.model,
                    int(trainer.state.global_step),
                )
                if trainer.lr_scheduler is not None:
                    raise RuntimeError(
                        "optimizer construction unexpectedly created an LR scheduler"
                    )
                overrides_after = len(
                    global_optim_manager.module_weight_config_triple
                )

                optimizer_report = {
                    "status": "passed",
                    "construction_seconds": optimizer_construction_seconds,
                    "stats": optimizer_stats,
                    "lora_sha256_before_optimizer": lora_sha_before_optimizer,
                    "lora_sha256_after_optimizer": lora_sha_after_optimizer,
                    "trainable_lora_unchanged": True,
                    "cuda_before_optimizer": cuda_before_optimizer,
                    "cuda_after_optimizer": cuda_after_optimizer,
                    "global_optimizer_manager_overrides_before": overrides_before,
                    "global_optimizer_manager_overrides_after": overrides_after,
                    "global_optimizer_manager_overrides_added": (
                        overrides_after - overrides_before
                    ),
                    "lr_scheduler_constructed": False,
                    "optimizer_step_performed": False,
                    "global_step": int(trainer.state.global_step),
                }

                optimizer.zero_grad(set_to_none=True)
                trainer.optimizer = None
                optimizer = None
                del global_optim_manager.module_weight_config_triple[
                    overrides_before:
                ]
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                optimizer_report["global_optimizer_manager_overrides_removed"] = (
                    len(global_optim_manager.module_weight_config_triple)
                    == overrides_before
                )
                optimizer_report["optimizer_released"] = True
                optimizer_report["cuda_after_optimizer_release"] = (
                    _cuda_memory_snapshot(torch)
                )

            one_update_report = None
            if perform_one_update:
                from bitsandbytes.optim import GlobalOptimManager
                from transformers import TrainerCallback

                class StopAfterFirstUpdateCallback(TrainerCallback):
                    def __init__(self):
                        self.step_end_calls = []

                    def on_step_end(self, args, state, control, **kwargs):
                        self.step_end_calls.append(int(state.global_step))
                        if int(state.global_step) >= 1:
                            control.should_training_stop = True
                        return control

                if trainer.optimizer is not None or trainer.lr_scheduler is not None:
                    raise RuntimeError(
                        "one-update gate started with optimization state"
                    )
                if any(
                    parameter.grad is not None
                    for parameter in trainer.model.parameters()
                    if parameter.requires_grad
                ):
                    raise RuntimeError("one-update gate started with LoRA gradients")

                stop_callback = StopAfterFirstUpdateCallback()
                trainer.add_callback(stop_callback)
                global_optim_manager = GlobalOptimManager.get_instance()
                overrides_before = len(
                    global_optim_manager.module_weight_config_triple
                )
                lora_sha_before_update = _trainable_parameter_sha256(trainer.model)
                parameter_snapshot = snapshot_trainable_parameters(trainer.model)
                torch.cuda.reset_peak_memory_stats()
                cuda_before_update = _cuda_memory_snapshot(torch)
                update_started = time.perf_counter()
                train_result = trainer.train()
                torch.cuda.synchronize()
                update_seconds = time.perf_counter() - update_started
                cuda_after_update = _cuda_memory_snapshot(torch)

                if stop_callback.step_end_calls != [1]:
                    raise RuntimeError(
                        "one-update callback did not stop after exactly one step"
                    )
                if int(trainer.state.global_step) != 1:
                    raise RuntimeError("one-update gate did not finish at global_step 1")
                if trainer.optimizer is None or trainer.lr_scheduler is None:
                    raise RuntimeError(
                        "one-update gate did not create optimizer and scheduler"
                    )
                wrapped_optimizer = trainer.optimizer
                base_optimizer = getattr(
                    wrapped_optimizer, "optimizer", wrapped_optimizer
                )
                optimizer_state = inspect_initialized_optimizer(
                    base_optimizer, trainer.model
                )
                parameter_update = inspect_parameter_update(
                    trainer.model, parameter_snapshot
                )
                lora_sha_after_update = _trainable_parameter_sha256(trainer.model)
                if lora_sha_after_update == lora_sha_before_update:
                    raise RuntimeError("one real optimizer step changed no LoRA bytes")

                log_history = [dict(entry) for entry in trainer.state.log_history]
                step_logs = [
                    entry
                    for entry in log_history
                    if int(entry.get("step", -1)) == 1 and "loss" in entry
                ]
                if len(step_logs) != 1:
                    raise RuntimeError("one-update gate has no unique step-one log")
                step_log = step_logs[0]
                train_loss = float(step_log.get("loss", float("nan")))
                gradient_norm = float(step_log.get("grad_norm", float("nan")))
                learning_rate_used = float(
                    step_log.get("learning_rate", float("nan"))
                )
                if not math.isfinite(train_loss):
                    raise RuntimeError("one-update training loss is not finite")
                if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise RuntimeError("one-update gradient norm is not finite and positive")
                if learning_rate_used != LOCKED_LEARNING_RATE:
                    raise RuntimeError("first optimizer step used the wrong learning rate")

                next_learning_rates = [
                    float(group["lr"]) for group in base_optimizer.param_groups
                ]
                expected_next_learning_rate = LOCKED_LEARNING_RATE * 0.5 * (
                    1.0 + math.cos(math.pi / 5)
                )
                if any(
                    not math.isclose(
                        value,
                        expected_next_learning_rate,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                    for value in next_learning_rates
                ):
                    raise RuntimeError("cosine scheduler produced an unexpected next LR")

                completions = list(trainer._logs["completion"])
                advantages = [float(value) for value in trainer._logs["advantages"]]
                component_rewards = {
                    name: [float(value) for value in trainer._logs["rewards"][name]]
                    for name in trainer.reward_func_names
                }
                reward_evidence = build_logged_reward_evidence(
                    sku_id=expected_sku_ids[0],
                    completions=completions,
                    reward_names=trainer.reward_func_names,
                    component_rewards=component_rewards,
                    reward_weights=runtime_reward_weights,
                    advantages=advantages,
                )
                if not reward_evidence["weighted_total_has_variance"]:
                    raise RuntimeError("one-update rollout had no reward variance")

                overrides_after = len(
                    global_optim_manager.module_weight_config_triple
                )
                train_metrics = {
                    key: float(value) if isinstance(value, (int, float)) else value
                    for key, value in train_result.metrics.items()
                }
                one_update_report = {
                    "status": "passed",
                    "update_seconds": update_seconds,
                    "callback_step_end_calls": stop_callback.step_end_calls,
                    "global_step": int(trainer.state.global_step),
                    "epoch": float(trainer.state.epoch),
                    "train_loss": train_loss,
                    "gradient_norm_before_clipping": gradient_norm,
                    "max_gradient_norm": float(config.max_grad_norm),
                    "learning_rate_used": learning_rate_used,
                    "next_learning_rates": next_learning_rates,
                    "expected_next_learning_rate": expected_next_learning_rate,
                    "optimizer_wrapper_class": type(wrapped_optimizer).__name__,
                    "optimizer_state": optimizer_state,
                    "parameter_update": parameter_update,
                    "reward_evidence": reward_evidence,
                    "train_metrics": train_metrics,
                    "trainer_log_history": log_history,
                    "lora_sha256_before_update": lora_sha_before_update,
                    "lora_sha256_after_update": lora_sha_after_update,
                    "trainable_lora_changed": True,
                    "optimizer_constructed": True,
                    "optimizer_state_initialized": True,
                    "lr_scheduler_constructed": True,
                    "lr_scheduler_class": type(trainer.lr_scheduler).__name__,
                    "optimizer_step_performed": True,
                    "cuda_before_update": cuda_before_update,
                    "cuda_after_update": cuda_after_update,
                    "global_optimizer_manager_overrides_before": overrides_before,
                    "global_optimizer_manager_overrides_after": overrides_after,
                    "global_optimizer_manager_overrides_added": (
                        overrides_after - overrides_before
                    ),
                }
                one_update_report["cuda_after_update_audit"] = (
                    _cuda_memory_snapshot(torch)
                )

                trainer.model.zero_grad(set_to_none=True)
                trainer.optimizer = None
                trainer.lr_scheduler = None
                wrapped_optimizer = None
                base_optimizer = None
                parameter_snapshot = None
                train_result = None
                del global_optim_manager.module_weight_config_triple[
                    overrides_before:
                ]
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                one_update_report["global_optimizer_manager_overrides_removed"] = (
                    len(global_optim_manager.module_weight_config_triple)
                    == overrides_before
                )
                one_update_report["optimizer_and_scheduler_detached_from_trainer"] = (
                    True
                )
                one_update_report["cuda_after_optimizer_detach"] = (
                    _cuda_memory_snapshot(torch)
                )

            if perform_rollout:
                generation_batch = next(iter(trainer.get_train_dataloader()))
                if not isinstance(generation_batch, list) or len(generation_batch) != 8:
                    raise RuntimeError(
                        "rollout generation batch must be a list of eight rows"
                    )
                generation_sku_ids = [row.get("sku_id") for row in generation_batch]
                expected_first_sku = expected_sku_ids[0]
                if generation_sku_ids != [expected_first_sku] * 8:
                    raise RuntimeError(
                        "first rollout batch must repeat the first locked SKU eight times"
                    )
                if any("gold" not in row for row in generation_batch):
                    raise RuntimeError("rollout generation batch lost hidden gold")

                trainer.model.train()
                lora_sha_before_rollout = _trainable_parameter_sha256(trainer.model)
                cuda_before_rollout = _cuda_memory_snapshot(torch)
                rollout_started = time.perf_counter()
                prepared = trainer._prepare_inputs(generation_batch)
                torch.cuda.synchronize()
                rollout_seconds = time.perf_counter() - rollout_started
                cuda_after_rollout = _cuda_memory_snapshot(torch)
                lora_sha_after_rollout = _trainable_parameter_sha256(trainer.model)
                if lora_sha_after_rollout != lora_sha_before_rollout:
                    raise RuntimeError("trainable LoRA changed during rollout-only gate")
                if int(trainer.state.global_step) != 0:
                    raise RuntimeError("rollout-only gate advanced global_step")
                if trainer.optimizer is not None or trainer.lr_scheduler is not None:
                    raise RuntimeError("rollout-only gate created optimization state")

                completions = list(trainer._logs["completion"])
                advantages = [float(value) for value in trainer._logs["advantages"]]
                component_rewards = {
                    name: [float(value) for value in trainer._logs["rewards"][name]]
                    for name in trainer.reward_func_names
                }
                completion_mask = prepared["completion_mask"].detach().cpu()
                effective_completion_tokens = [
                    int(row.sum().item()) for row in completion_mask
                ]
                truncated = [length == 0 for length in effective_completion_tokens]
                if int(prepared["completion_ids"].shape[0]) != 8:
                    raise RuntimeError("prepared rollout tensor batch is not eight")

                rollout_report = {
                    "status": "passed",
                    "sku_id": expected_first_sku,
                    "generation_batch_rows": len(generation_batch),
                    "unique_generation_batch_skus": sorted(set(generation_sku_ids)),
                    "rollout_seconds": rollout_seconds,
                    "prepared_tensor_keys": sorted(prepared),
                    "lora_sha256_before_rollout": lora_sha_before_rollout,
                    "lora_sha256_after_rollout": lora_sha_after_rollout,
                    "trainable_lora_unchanged": True,
                    "cuda_before_rollout": cuda_before_rollout,
                    "cuda_after_rollout": cuda_after_rollout,
                    **build_rollout_evidence(
                        sku_id=expected_first_sku,
                        completions=completions,
                        reward_names=trainer.reward_func_names,
                        component_rewards=component_rewards,
                        reward_weights=runtime_reward_weights,
                        advantages=advantages,
                        effective_completion_tokens=effective_completion_tokens,
                        truncated_and_masked=truncated,
                    ),
                }

            if perform_backward:
                if prepared is None or rollout_report is None:
                    raise RuntimeError("gradient gate has no prepared rollout")
                if any(
                    parameter.grad is not None
                    for parameter in trainer.model.parameters()
                    if parameter.requires_grad
                ):
                    raise RuntimeError("LoRA gradients existed before gradient gate")

                lora_sha_before_backward = _trainable_parameter_sha256(trainer.model)
                torch.cuda.reset_peak_memory_stats()
                cuda_before_backward = _cuda_memory_snapshot(torch)
                backward_started = time.perf_counter()
                loss = trainer.compute_loss(
                    trainer.model,
                    prepared,
                    return_outputs=False,
                )
                if int(loss.numel()) != 1:
                    raise RuntimeError("GRPO loss must be a scalar")
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError("GRPO loss is NaN or infinity")
                trainer.accelerator.backward(loss)
                torch.cuda.synchronize()
                backward_seconds = time.perf_counter() - backward_started
                cuda_after_backward = _cuda_memory_snapshot(torch)
                gradient_stats = inspect_lora_gradients(trainer.model)
                cuda_after_gradient_inspection = _cuda_memory_snapshot(torch)
                lora_sha_after_backward = _trainable_parameter_sha256(trainer.model)
                if lora_sha_after_backward != lora_sha_before_backward:
                    raise RuntimeError("LoRA weights changed during gradient-only gate")
                if trainer.optimizer is not None or trainer.lr_scheduler is not None:
                    raise RuntimeError("gradient-only gate created optimization state")
                if int(trainer.state.global_step) != 0:
                    raise RuntimeError("gradient-only gate advanced global_step")

                gradient_report = {
                    "status": "passed",
                    "loss": loss_value,
                    "forward_and_backward_seconds": backward_seconds,
                    "stats": gradient_stats,
                    "lora_sha256_before_backward": lora_sha_before_backward,
                    "lora_sha256_after_backward": lora_sha_after_backward,
                    "trainable_lora_unchanged": True,
                    "cuda_before_backward": cuda_before_backward,
                    "cuda_after_backward": cuda_after_backward,
                    "cuda_after_gradient_inspection": (
                        cuda_after_gradient_inspection
                    ),
                    "optimizer_constructed": False,
                    "lr_scheduler_constructed": False,
                    "optimizer_step_performed": False,
                    "global_step": int(trainer.state.global_step),
                }

                trainer.model.zero_grad(set_to_none=True)
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gradients_remaining = sum(
                    parameter.grad is not None
                    for parameter in trainer.model.parameters()
                    if parameter.requires_grad
                )
                if gradients_remaining != 0:
                    raise RuntimeError("gradient cleanup left attached LoRA gradients")
                gradient_report["gradients_remaining_after_clear"] = 0
                gradient_report["gradients_cleared"] = True
                gradient_report["cuda_after_gradient_clear"] = _cuda_memory_snapshot(
                    torch
                )

            trainability_after = inspect_model_trainability(trainer.model)
            adapter_sha_after = _sha256_file(adapter_file)
            if adapter_sha_after != expected_adapter_sha256:
                raise RuntimeError("source adapter changed during trainer construction")
            after_construction = _cuda_memory_snapshot(torch)
            report = {
                "version": (
                    ONE_UPDATE_GATE_VERSION
                    if perform_one_update
                    else OPTIMIZER_CONSTRUCTION_VERSION
                    if construct_optimizer
                    else GRADIENT_GATE_VERSION
                    if perform_backward
                    else ROLLOUT_GATE_VERSION
                    if perform_rollout
                    else TRAINER_CONSTRUCTION_VERSION
                ),
                "status": "passed",
                "trainer_class": type(trainer).__name__,
                "config_class": type(config).__name__,
                "model_class": type(trainer.model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "construction_seconds_including_model_load": time.perf_counter()
                - started,
                "dataset": {
                    "rows": len(dataset),
                    "columns_retained_by_trainer": trainer_dataset_columns,
                    "sku_ids_in_step_order": trainer_sku_ids,
                    "order_matches_manifest": True,
                    "hidden_gold_retained": "gold" in trainer_dataset_columns,
                    "hidden_sku_id_retained": "sku_id" in trainer_dataset_columns,
                },
                "rewards": {
                    "names_in_trainer_order": trainer_reward_names,
                    "weights_in_trainer": runtime_reward_weights,
                    "order_matches_contract": True,
                },
                "config": config_report,
                "trainability_before_trainer": trainability_before,
                "trainability_after_trainer": trainability_after,
                "adapter_sha256_after_construction": adapter_sha_after,
                "source_adapter_unchanged": True,
                "cuda_before_load": before,
                "cuda_after_trainer_construction": after_construction,
                "optimizer_constructed": (
                    optimizer_report is not None or one_update_report is not None
                ),
                "lr_scheduler_constructed": one_update_report is not None,
                "reference_model_constructed": False,
                "generation_performed": perform_rollout or perform_one_update,
                "loss_computed": perform_backward or perform_one_update,
                "backward_performed": perform_backward or perform_one_update,
                "training_steps": 1 if perform_one_update else 0,
                "global_step": int(trainer.state.global_step),
                "temporary_output_path": str(temporary_output_path),
            }
            if rollout_report is not None:
                report["rollout"] = rollout_report
            if gradient_report is not None:
                report["gradient"] = gradient_report
            if optimizer_report is not None:
                report["optimizer"] = optimizer_report
            if one_update_report is not None:
                report["one_update"] = one_update_report

            trainer = None
            config = None

        report["temporary_output_removed"] = not temporary_output_path.exists()
        if not report["temporary_output_removed"]:
            raise RuntimeError("temporary trainer-construction output was not removed")
    finally:
        trainer = None
        dataset = None
        generation_batch = None
        prepared = None
        loss = None
        optimizer = None
        parameter_snapshot = None
        train_result = None
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("trainer-construction gate ended without a report")
    report["cuda_after_release"] = _cuda_memory_snapshot(torch)
    report["trainer_retained"] = False
    report["model_retained"] = False
    report["optimizer_retained"] = False
    return report


def run_trainer_construction_gate(**kwargs) -> dict:
    """Construct and inspect the exact GRPO trainer without generating."""
    return _run_trainer_gate(
        perform_rollout=False,
        perform_backward=False,
        construct_optimizer=False,
        perform_one_update=False,
        **kwargs,
    )


def run_rollout_gate(**kwargs) -> dict:
    """Generate and reward one eight-completion group without computing a loss."""
    return _run_trainer_gate(
        perform_rollout=True,
        perform_backward=False,
        construct_optimizer=False,
        perform_one_update=False,
        **kwargs,
    )


def run_gradient_gate(**kwargs) -> dict:
    """Generate, score and backpropagate one group without an optimizer update."""
    return _run_trainer_gate(
        perform_rollout=True,
        perform_backward=True,
        construct_optimizer=False,
        perform_one_update=False,
        **kwargs,
    )


def run_optimizer_construction_gate(**kwargs) -> dict:
    """Construct and audit 8-bit AdamW without gradients or an update."""
    return _run_trainer_gate(
        perform_rollout=False,
        perform_backward=False,
        construct_optimizer=True,
        perform_one_update=False,
        **kwargs,
    )


def run_one_update_gate(**kwargs) -> dict:
    """Run the real trainer loop and stop after exactly one optimizer step."""
    return _run_trainer_gate(
        perform_rollout=False,
        perform_backward=False,
        construct_optimizer=False,
        perform_one_update=True,
        **kwargs,
    )


def _load_five_step_runtime() -> dict:
    """Import the real GPU stack only inside the guarded full-smoke gate."""
    from unsloth import FastLanguageModel

    import torch
    from bitsandbytes.optim import GlobalOptimManager
    from trl import GRPOConfig, GRPOTrainer

    from training.dataset import load_grpo_prompts
    from training.rewards import (
        FIRST_RUN_REWARD_FUNCTIONS,
        FIRST_RUN_REWARD_WEIGHTS,
    )
    from verifier import load_pack

    return {
        "FastLanguageModel": FastLanguageModel,
        "torch": torch,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "GlobalOptimManager": GlobalOptimManager,
        "load_grpo_prompts": load_grpo_prompts,
        "load_pack": load_pack,
        "reward_functions": tuple(FIRST_RUN_REWARD_FUNCTIONS),
        "reward_weights": tuple(FIRST_RUN_REWARD_WEIGHTS),
    }


def _load_full_run_construction_runtime() -> dict:
    """Import the real stack only inside the no-training full-run gate."""
    from unsloth import FastLanguageModel

    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from training.dataset import load_grpo_prompts
    from training.rewards import (
        FIRST_RUN_REWARD_FUNCTIONS,
        FIRST_RUN_REWARD_WEIGHTS,
    )
    from verifier import load_pack

    return {
        "FastLanguageModel": FastLanguageModel,
        "torch": torch,
        "TrainerCallback": TrainerCallback,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "load_grpo_prompts": load_grpo_prompts,
        "load_pack": load_pack,
        "reward_functions": tuple(FIRST_RUN_REWARD_FUNCTIONS),
        "reward_weights": tuple(FIRST_RUN_REWARD_WEIGHTS),
    }


def _load_full_run_runtime() -> dict:
    """Load construction dependencies plus optimizer-global cleanup support."""
    runtime = _load_full_run_construction_runtime()
    from bitsandbytes.optim import GlobalOptimManager

    return {**runtime, "GlobalOptimManager": GlobalOptimManager}


def run_full_run_300_construction_gate(
    *,
    training_data_path: str | Path,
    adapter_path: str | Path,
    adapter_file: str | Path,
    expected_ordered_sku_sha256: str,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
    runtime_loader: Callable[[], dict] = _load_full_run_construction_runtime,
    policy_loader_fn: Callable[..., tuple] = _load_locked_policy,
    trainability_fn: Callable[[object], dict] = inspect_model_trainability,
    cuda_snapshot_fn: Callable[[object], dict] = _cuda_memory_snapshot,
) -> dict:
    """Construct all real full-run components and release them without training."""
    training_data = Path(training_data_path).resolve()
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()
    if not training_data.is_file():
        raise FileNotFoundError(training_data)
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    if (
        not isinstance(expected_ordered_sku_sha256, str)
        or len(expected_ordered_sku_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_ordered_sku_sha256
        )
    ):
        raise ValueError("expected ordered-SKU SHA-256 is invalid")
    if _sha256_file(adapter_file) != expected_adapter_sha256:
        raise RuntimeError("source adapter disagrees before full construction")

    runtime = runtime_loader()
    required_runtime = {
        "FastLanguageModel",
        "torch",
        "TrainerCallback",
        "GRPOConfig",
        "GRPOTrainer",
        "load_grpo_prompts",
        "load_pack",
        "reward_functions",
        "reward_weights",
    }
    if set(runtime) != required_runtime:
        raise RuntimeError("full-run construction runtime is incomplete or unexpected")
    reward_functions = tuple(runtime["reward_functions"])
    reward_weights = tuple(float(value) for value in runtime["reward_weights"])
    reward_names = tuple(
        getattr(reward, "__name__", type(reward).__name__)
        for reward in reward_functions
    )
    if reward_weights != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("full-run runtime reward weights drifted")
    if reward_names != EXPECTED_REWARD_NAMES:
        raise RuntimeError("full-run runtime reward functions drifted")

    torch = runtime["torch"]
    model = None
    tokenizer = None
    dataset = None
    trainer = None
    collector = None
    writer = None
    callback = None
    phase_profiler = None
    profiler_callback = None
    report = None
    temporary_root = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_before = cuda_snapshot_fn(torch)
    started = time.perf_counter()
    try:
        model, tokenizer = policy_loader_fn(
            runtime["FastLanguageModel"], torch, adapter_path
        )
        trainability = trainability_fn(model)
        lora_sha_before = _trainable_parameter_sha256(model)

        pack = runtime["load_pack"]("packs/vastraa_taste_v1")
        dataset = runtime["load_grpo_prompts"](
            pack,
            training_data,
            require_pass_rate_band=True,
        )
        if len(dataset) != FULL_RUN_DATA_ROWS:
            raise RuntimeError(
                f"full-run construction dataset has {len(dataset)} rows"
            )
        required_columns = {"prompt", "gold", "sku_id"}
        if set(dataset.column_names) != required_columns:
            raise RuntimeError("full-run construction dataset columns drifted")
        dataset_skus = list(dataset["sku_id"])
        if len(set(dataset_skus)) != FULL_RUN_DATA_ROWS:
            raise RuntimeError("full-run construction dataset SKUs are not unique")
        observed_ordered_sha = _ordered_sku_sha256(dataset_skus)
        if observed_ordered_sha != expected_ordered_sku_sha256:
            raise RuntimeError("full-run construction dataset order drifted")

        with tempfile.TemporaryDirectory(
            prefix="grpo-full-run-construction-"
        ) as temporary:
            temporary_root = Path(temporary).resolve()
            probe_final = temporary_root / "grpo-first-300"
            staging = create_full_run_staging_output(probe_final)
            plan = build_full_run_lifecycle_plan(
                final_output_dir=probe_final,
                staging_dir=staging,
            )
            writer = FullRunCheckpointLifecycleWriter(
                plan=plan,
                starting_adapter_sha256=expected_adapter_sha256,
            )
            config = runtime["GRPOConfig"](
                **grpo_full_run_300_config_kwargs(
                    output_dir=plan["trainer_output_dir"],
                    reward_weights=reward_weights,
                )
            )
            config_report = inspect_grpo_full_run_config(config)
            collector = FullRunRolloutCollector()
            trainer_class = make_full_run_rollout_capturing_trainer_class(
                runtime["GRPOTrainer"]
            )
            trainer = trainer_class(
                model=model,
                reward_funcs=list(reward_functions),
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
                full_run_rollout_collector=collector,
            )
            phase_profiler = FullRunPhaseProfiler(
                expected_steps=FULL_RUN_STEPS,
                synchronize_fn=torch.cuda.synchronize,
            )
            phase_profiler.instrument_trainer(trainer)
            profiler_callback_class = make_phase_profiler_callback_class(
                runtime["TrainerCallback"]
            )
            profiler_callback = profiler_callback_class(
                phase_profiler=phase_profiler
            )
            trainer.add_callback(profiler_callback)
            callback_class = make_full_run_checkpoint_callback_class(
                runtime["TrainerCallback"]
            )
            callback = callback_class(
                lifecycle_writer=writer,
                rollout_collector=collector,
                phase_timing_snapshot_fn=(
                    lambda step: phase_profiler.snapshot(expected_steps=step)
                ),
            )
            trainer.add_callback(callback)
            torch.cuda.synchronize()

            actual_reward_names = list(getattr(trainer, "reward_func_names", ()))
            actual_reward_weights = _tensor_like_to_list(
                getattr(trainer, "reward_weights", None),
                label="constructed full-run reward weights",
            )
            if actual_reward_names != list(EXPECTED_REWARD_NAMES):
                raise RuntimeError("constructed full-run reward names drifted")
            if tuple(float(value) for value in actual_reward_weights) != (
                LOCKED_REWARD_WEIGHTS
            ):
                raise RuntimeError("constructed full-run reward weights drifted")
            if getattr(trainer, "optimizer", None) is not None:
                raise RuntimeError("full-run construction created an optimizer")
            if getattr(trainer, "lr_scheduler", None) is not None:
                raise RuntimeError("full-run construction created a scheduler")
            if getattr(trainer, "ref_model", None) is not None:
                raise RuntimeError("beta-zero full-run constructed a reference model")
            if int(getattr(trainer.state, "global_step", -1)) != 0:
                raise RuntimeError("full-run construction advanced global step")
            if collector.captured_steps != 0:
                raise RuntimeError("full-run construction generated rollouts")
            if writer.events:
                raise RuntimeError("full-run construction wrote lifecycle events")
            callback_list = list(
                getattr(getattr(trainer, "callback_handler", None), "callbacks", ())
            )
            if callback not in callback_list:
                raise RuntimeError("full-run checkpoint callback was not attached")
            if profiler_callback not in callback_list:
                raise RuntimeError("full-run profiling callback was not attached")
            if set(trainer.train_dataset.column_names) != required_columns:
                raise RuntimeError("constructed trainer dropped dataset columns")
            if list(trainer.train_dataset["sku_id"]) != dataset_skus:
                raise RuntimeError("constructed trainer changed dataset order")
            if any(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("full-run construction created gradients")

            lora_sha_after = _trainable_parameter_sha256(model)
            if lora_sha_after != lora_sha_before:
                raise RuntimeError("full-run construction changed LoRA weights")
            if _sha256_file(adapter_file) != expected_adapter_sha256:
                raise RuntimeError("source adapter changed during full construction")
            cuda_after = cuda_snapshot_fn(torch)
            report = {
                "version": FULL_RUN_CONSTRUCTION_GATE_VERSION,
                "status": "passed",
                "training_data_path": str(training_data),
                "dataset_rows": len(dataset),
                "dataset_columns": sorted(required_columns),
                "dataset_ordered_sku_sha256": observed_ordered_sha,
                "adapter_path": str(adapter_path),
                "adapter_file": str(adapter_file),
                "adapter_sha256": expected_adapter_sha256,
                "source_adapter_unchanged": True,
                "trainability": trainability,
                "lora_sha256_before_construction": lora_sha_before,
                "lora_sha256_after_construction": lora_sha_after,
                "trainable_lora_unchanged": True,
                "config": config_report,
                "trainer_class": type(trainer).__name__,
                "base_trainer_class": runtime["GRPOTrainer"].__name__,
                "callback_class": type(callback).__name__,
                "reward_names": actual_reward_names,
                "reward_weights": [
                    float(value) for value in actual_reward_weights
                ],
                "collector_attached": (
                    trainer.full_run_rollout_collector is collector
                ),
                "checkpoint_callback_attached": True,
                "phase_profiler_attached": True,
                "phase_profiler_steps": 0,
                "lifecycle_writer_constructed": True,
                "lifecycle_events": 0,
                "global_step": 0,
                "optimizer_constructed": False,
                "scheduler_constructed": False,
                "reference_model_constructed": False,
                "rollouts_generated": 0,
                "gradients_created": False,
                "training_dispatched": False,
                "temporary_output_root": str(temporary_root),
                "cuda_before": cuda_before,
                "cuda_after_construction": cuda_after,
                "construction_seconds": time.perf_counter() - started,
            }
        if temporary_root.exists():
            raise RuntimeError("temporary full-run construction output survived")
    finally:
        profiler_callback = None
        phase_profiler = None
        callback = None
        writer = None
        collector = None
        trainer = None
        dataset = None
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("full-run construction gate ended without a report")
    if _sha256_file(adapter_file) != expected_adapter_sha256:
        raise RuntimeError("source adapter changed after full construction release")
    report["temporary_output_removed"] = True
    report["cuda_after_release"] = cuda_snapshot_fn(torch)
    report["model_retained"] = False
    report["trainer_retained"] = False
    report["training_dispatched"] = False
    return report


def run_full_run_300_gate(
    *,
    preflight_report: dict,
    training_data_path: str | Path,
    adapter_path: str | Path,
    adapter_file: str | Path,
    final_output_dir: str | Path,
    runtime_loader: Callable[[], dict] = _load_full_run_runtime,
    policy_loader_fn: Callable[..., tuple] = _load_locked_policy,
    trainability_fn: Callable[[object], dict] = inspect_model_trainability,
    parameter_values_fn: Callable[[object], dict] = (
        inspect_trainable_parameter_values
    ),
    fingerprint_fn: Callable[[object], str] = _trainable_parameter_sha256,
    cuda_snapshot_fn: Callable[[object], dict] = _cuda_memory_snapshot,
    orchestration_fn: Callable[..., dict] = run_full_run_300_orchestration,
    disk_usage_fn: Callable[[Path], object] | None = None,
    progress_callback: Callable[..., object] | None = None,
    expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
) -> dict:
    """Load the real runtime and invoke guarded orchestration outside the CLI."""
    if preflight_report.get("status") != "passed":
        raise ValueError("full-run runtime bridge requires a passed preflight")
    if preflight_report.get("cuda_imports_performed") is not False:
        raise ValueError("full-run runtime bridge received invalid preflight state")

    training_data = Path(training_data_path).resolve()
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()
    final_output = Path(final_output_dir).resolve()
    expected_data = Path(preflight_report.get("pool", {}).get("data_path", "")).resolve()
    expected_adapter_path = Path(
        preflight_report.get("sft_lock", {}).get("adapter_path", "")
    ).resolve()
    expected_adapter_file = Path(
        preflight_report.get("sft_lock", {}).get("adapter_file", "")
    ).resolve()
    expected_output = Path(
        preflight_report.get("output", {}).get("path", "")
    ).resolve()
    if training_data != expected_data:
        raise RuntimeError("full-run bridge data path disagrees with preflight")
    if adapter_path != expected_adapter_path or adapter_file != expected_adapter_file:
        raise RuntimeError("full-run bridge adapter path disagrees with preflight")
    if final_output != expected_output:
        raise RuntimeError("full-run bridge output path disagrees with preflight")
    if final_output.exists():
        raise FileExistsError(f"final full-run output already exists: {final_output}")
    expected_adapter_sha = preflight_report.get("sft_lock", {}).get(
        "adapter_sha256"
    )
    if _sha256_file(adapter_file) != expected_adapter_sha:
        raise RuntimeError("source adapter disagrees before full-run bridge")

    runtime = runtime_loader()
    required_runtime = {
        "FastLanguageModel",
        "torch",
        "TrainerCallback",
        "GRPOConfig",
        "GRPOTrainer",
        "GlobalOptimManager",
        "load_grpo_prompts",
        "load_pack",
        "reward_functions",
        "reward_weights",
    }
    if set(runtime) != required_runtime:
        raise RuntimeError("full-run runtime bridge stack is incomplete or unexpected")
    reward_weights = tuple(float(value) for value in runtime["reward_weights"])
    reward_names = tuple(
        getattr(reward, "__name__", type(reward).__name__)
        for reward in runtime["reward_functions"]
    )
    if reward_weights != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("full-run bridge reward weights drifted")
    if reward_names != EXPECTED_REWARD_NAMES:
        raise RuntimeError("full-run bridge reward functions drifted")

    torch = runtime["torch"]
    global_optim_manager = runtime["GlobalOptimManager"].get_instance()
    overrides_before = len(global_optim_manager.module_weight_config_triple)
    model = None
    tokenizer = None
    dataset = None
    report = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_before_load = cuda_snapshot_fn(torch)
    started = time.perf_counter()
    try:
        model, tokenizer = policy_loader_fn(
            runtime["FastLanguageModel"], torch, adapter_path
        )
        trainability_before = trainability_fn(model)
        lora_sha_before = fingerprint_fn(model)
        cuda_after_load = cuda_snapshot_fn(torch)
        runtime_context = {
            "version": FULL_RUN_RUNTIME_CONTEXT_VERSION,
            "runtime": {
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "trainer_class": runtime["GRPOTrainer"].__name__,
                "trainer_module": runtime["GRPOTrainer"].__module__,
                "callback_class": runtime["TrainerCallback"].__name__,
                "config_class": runtime["GRPOConfig"].__name__,
            },
            "cuda_before_load": cuda_before_load,
            "cuda_after_load": cuda_after_load,
        }

        pack = runtime["load_pack"]("packs/vastraa_taste_v1")
        dataset = runtime["load_grpo_prompts"](
            pack,
            training_data,
            require_pass_rate_band=True,
        )
        if len(dataset) != FULL_RUN_DATA_ROWS:
            raise RuntimeError("full-run bridge dataset row count drifted")
        required_columns = {"prompt", "gold", "sku_id"}
        if set(dataset.column_names) != required_columns:
            raise RuntimeError("full-run bridge dataset columns drifted")
        dataset_skus = list(dataset["sku_id"])
        if len(set(dataset_skus)) != FULL_RUN_DATA_ROWS:
            raise RuntimeError("full-run bridge dataset SKUs are not unique")
        observed_order_sha = _ordered_sku_sha256(dataset_skus)
        if observed_order_sha != preflight_report["pool"].get(
            "ordered_sku_sha256"
        ):
            raise RuntimeError("full-run bridge dataset order drifted")

        orchestration = orchestration_fn(
            base_trainer_class=runtime["GRPOTrainer"],
            base_callback_class=runtime["TrainerCallback"],
            config_class=runtime["GRPOConfig"],
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_functions=runtime["reward_functions"],
            reward_weights=runtime["reward_weights"],
            source_adapter_file=adapter_file,
            final_output_dir=final_output,
            preflight_report=preflight_report,
            trainability_fn=trainability_fn,
            parameter_values_fn=parameter_values_fn,
            fingerprint_fn=fingerprint_fn,
            runtime_context=runtime_context,
            cuda_snapshot_fn=lambda: cuda_snapshot_fn(torch),
            phase_synchronize_fn=lambda: torch.cuda.synchronize(),
            disk_usage_fn=disk_usage_fn,
            progress_callback=progress_callback,
            expected_adapter_model_bytes=expected_adapter_model_bytes,
        )
        if (
            orchestration.get("status") != "passed"
            or not orchestration.get("published")
            or not final_output.is_dir()
        ):
            raise RuntimeError("full-run orchestration did not publish success")

        trainability_after = trainability_fn(model)
        if trainability_after.get("trainable_parameters") != trainability_before.get(
            "trainable_parameters"
        ):
            raise RuntimeError("full-run trainable-parameter count changed")
        if trainability_after.get("trainable_tensors") != trainability_before.get(
            "trainable_tensors"
        ):
            raise RuntimeError("full-run trainable-tensor count changed")
        parameter_values = parameter_values_fn(model)
        lora_sha_after = fingerprint_fn(model)
        if lora_sha_after == lora_sha_before:
            raise RuntimeError("full-run bridge changed no trainable LoRA bytes")
        if _sha256_file(adapter_file) != expected_adapter_sha:
            raise RuntimeError("source adapter changed during full-run bridge")
        cuda_after_orchestration = cuda_snapshot_fn(torch)
        report = {
            "version": FULL_RUN_RUNTIME_BRIDGE_VERSION,
            "status": "passed",
            "training_data_path": str(training_data),
            "dataset_rows": len(dataset),
            "dataset_columns": sorted(required_columns),
            "dataset_ordered_sku_sha256": observed_order_sha,
            "adapter_path": str(adapter_path),
            "adapter_file": str(adapter_file),
            "adapter_sha256": expected_adapter_sha,
            "source_adapter_unchanged": True,
            "trainability_before": trainability_before,
            "trainability_after": trainability_after,
            "parameter_values_after": parameter_values,
            "lora_sha256_before": lora_sha_before,
            "lora_sha256_after": lora_sha_after,
            "trainable_lora_changed": True,
            "orchestration": orchestration,
            "cuda_before_load": cuda_before_load,
            "cuda_after_load": cuda_after_load,
            "cuda_after_orchestration": cuda_after_orchestration,
            "bridge_seconds_before_release": time.perf_counter() - started,
            "global_optimizer_manager_overrides_before": overrides_before,
            "training_dispatched": True,
            "published": True,
        }
    finally:
        dataset = None
        model = None
        tokenizer = None
        del global_optim_manager.module_weight_config_triple[overrides_before:]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("full-run runtime bridge ended without a report")
    report["global_optimizer_manager_overrides_removed"] = (
        len(global_optim_manager.module_weight_config_triple) == overrides_before
    )
    report["cuda_after_release"] = cuda_snapshot_fn(torch)
    report["model_retained"] = False
    report["trainer_retained"] = False
    return report


def construct_capturing_grpo_trainer(
    *,
    base_trainer_class: type,
    config_class: type,
    model: object,
    tokenizer: object,
    dataset: object,
    expected_sku_ids: Sequence[str],
    reward_functions: Sequence[Callable],
    reward_weights: Sequence[float],
    temporary_output_dir: str | Path,
) -> tuple[object, dict]:
    """Construct and assert the real trainer shape used only by the full smoke."""
    expected_sku_ids = list(expected_sku_ids)
    if len(dataset) != EXPECTED_STEPS:
        raise RuntimeError("full-smoke dataset must contain exactly five rows")
    required_columns = {"prompt", "gold", "sku_id"}
    if set(dataset.column_names) != required_columns:
        raise RuntimeError("full-smoke dataset columns drifted")
    dataset_sku_ids = list(dataset["sku_id"])
    if dataset_sku_ids != expected_sku_ids:
        raise RuntimeError("full-smoke dataset SKU order drifted")
    normalized_weights = tuple(float(value) for value in reward_weights)
    if normalized_weights != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("full-smoke reward weights drifted")
    expected_reward_names = [
        getattr(reward, "__name__", type(reward).__name__)
        for reward in reward_functions
    ]
    if tuple(expected_reward_names) != EXPECTED_REWARD_NAMES:
        raise RuntimeError("full-smoke reward function names drifted")

    config = config_class(
        **grpo_smoke_config_kwargs(
            output_dir=temporary_output_dir,
            reward_weights=normalized_weights,
        )
    )
    config_report = inspect_grpo_config(config)
    collector = SmokeRolloutCollector(expected_sku_ids)
    capturing_class = make_rollout_capturing_trainer_class(base_trainer_class)
    trainer = capturing_class(
        model=model,
        reward_funcs=list(reward_functions),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        smoke_rollout_collector=collector,
    )

    actual_reward_names = list(getattr(trainer, "reward_func_names", ()))
    if actual_reward_names != expected_reward_names:
        raise RuntimeError("capturing trainer reward function order drifted")
    actual_weights = _tensor_like_to_list(
        getattr(trainer, "reward_weights", None),
        label="capturing trainer reward weights",
    )
    if [float(value) for value in actual_weights] != list(LOCKED_REWARD_WEIGHTS):
        raise RuntimeError("capturing trainer reward weights drifted")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("capturing trainer unexpectedly constructed an optimizer")
    if getattr(trainer, "lr_scheduler", None) is not None:
        raise RuntimeError("capturing trainer unexpectedly constructed a scheduler")
    if getattr(trainer, "ref_model", None) is not None:
        raise RuntimeError("beta-zero capturing trainer constructed a reference model")
    if int(getattr(trainer.state, "global_step", -1)) != 0:
        raise RuntimeError("capturing trainer advanced global_step during construction")
    trainer_columns = set(trainer.train_dataset.column_names)
    trainer_sku_ids = list(trainer.train_dataset["sku_id"])
    if trainer_columns != required_columns or trainer_sku_ids != expected_sku_ids:
        raise RuntimeError("capturing trainer changed its locked dataset")

    return trainer, {
        "trainer_class": type(trainer).__name__,
        "base_trainer_class": base_trainer_class.__name__,
        "config_class": type(config).__name__,
        "config": config_report,
        "dataset_rows": len(dataset),
        "dataset_columns": sorted(trainer_columns),
        "sku_ids_in_step_order": trainer_sku_ids,
        "reward_names": actual_reward_names,
        "reward_weights": [float(value) for value in actual_weights],
        "collector_attached": trainer.smoke_rollout_collector is collector,
        "optimizer_constructed": False,
        "scheduler_constructed": False,
        "reference_model_constructed": False,
        "global_step": 0,
        "matches_locked_contract": True,
    }


def run_five_step_smoke_gate(
    *,
    preflight_report: dict,
    fixture_data_path: str | Path,
    adapter_path: str | Path,
    adapter_file: str | Path,
    final_output_dir: str | Path,
    expected_sku_ids: Sequence[str],
    runtime_loader: Callable[[], dict] = _load_five_step_runtime,
    policy_loader_fn: Callable[..., tuple] = _load_locked_policy,
    trainability_fn: Callable[[object], dict] = inspect_model_trainability,
    orchestration_fn: Callable[..., dict] = run_five_step_smoke_orchestration,
) -> dict:
    """Build the live capturing trainer and invoke guarded orchestration."""
    fixture_data_path = Path(fixture_data_path).resolve()
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()
    final_output_dir = Path(final_output_dir).resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"final smoke output already exists: {final_output_dir}"
        )
    if list(expected_sku_ids) != preflight_report["fixture"][
        "sku_ids_in_step_order"
    ]:
        raise RuntimeError("full-smoke SKU order disagrees with preflight")

    runtime = runtime_loader()
    required_runtime = {
        "FastLanguageModel",
        "torch",
        "GRPOConfig",
        "GRPOTrainer",
        "GlobalOptimManager",
        "load_grpo_prompts",
        "load_pack",
        "reward_functions",
        "reward_weights",
    }
    if set(runtime) != required_runtime:
        raise RuntimeError("full-smoke runtime stack is incomplete or unexpected")
    if tuple(runtime["reward_weights"]) != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("runtime reward weights drifted")

    torch = runtime["torch"]
    model = None
    tokenizer = None
    dataset = None
    trainer = None
    report = None
    temporary_output_path = None
    global_optim_manager = runtime["GlobalOptimManager"].get_instance()
    overrides_before = len(global_optim_manager.module_weight_config_triple)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_before_load = _cuda_memory_snapshot(torch)
    started = time.perf_counter()
    try:
        model, tokenizer = policy_loader_fn(
            runtime["FastLanguageModel"],
            torch,
            adapter_path,
        )
        trainability_before = trainability_fn(model)
        pack = runtime["load_pack"]("packs/vastraa_taste_v1")
        dataset = runtime["load_grpo_prompts"](
            pack,
            fixture_data_path,
            require_pass_rate_band=True,
        )
        with tempfile.TemporaryDirectory(prefix="grpo-five-step-runtime-") as temp:
            temporary_output_path = Path(temp).resolve()
            trainer, construction = construct_capturing_grpo_trainer(
                base_trainer_class=runtime["GRPOTrainer"],
                config_class=runtime["GRPOConfig"],
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                expected_sku_ids=expected_sku_ids,
                reward_functions=runtime["reward_functions"],
                reward_weights=runtime["reward_weights"],
                temporary_output_dir=temporary_output_path,
            )
            torch.cuda.synchronize()
            cuda_after_construction = _cuda_memory_snapshot(torch)
            adapter_sha_after_construction = _sha256_file(adapter_file)
            expected_adapter_sha = preflight_report["sft_lock"]["adapter_sha256"]
            if adapter_sha_after_construction != expected_adapter_sha:
                raise RuntimeError(
                    "source adapter changed during full-smoke construction"
                )
            construction_seconds = time.perf_counter() - started

            orchestration = orchestration_fn(
                trainer=trainer,
                tokenizer=tokenizer,
                source_adapter_file=adapter_file,
                final_output_dir=final_output_dir,
                expected_sku_ids=expected_sku_ids,
                preflight_report=preflight_report,
                config_settings=construction["config"]["settings"],
                cuda_snapshot_fn=lambda: _cuda_memory_snapshot(torch),
                trainability_fn=trainability_fn,
            )
            if (
                orchestration.get("status") != "passed"
                or not orchestration.get("published")
                or orchestration.get("manifest", {}).get("status") != "completed"
                or not final_output_dir.is_dir()
            ):
                raise RuntimeError("full-smoke orchestration did not publish success")
            report = {
                "version": FIVE_STEP_SMOKE_GATE_VERSION,
                "status": "passed",
                "construction_seconds_including_model_load": construction_seconds,
                "total_gate_seconds_before_release": time.perf_counter() - started,
                "temporary_output_path": str(temporary_output_path),
                "trainability_before_trainer": trainability_before,
                "construction": construction,
                "orchestration": orchestration,
                "adapter_sha256_after_construction": (
                    adapter_sha_after_construction
                ),
                "source_adapter_unchanged_at_construction": True,
                "cuda_before_load": cuda_before_load,
                "cuda_after_construction": cuda_after_construction,
                "global_optimizer_manager_overrides_before": overrides_before,
            }

        report["temporary_output_removed"] = not temporary_output_path.exists()
        if not report["temporary_output_removed"]:
            raise RuntimeError("temporary full-smoke trainer output was not removed")
    finally:
        trainer = None
        dataset = None
        model = None
        tokenizer = None
        del global_optim_manager.module_weight_config_triple[overrides_before:]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("full-smoke gate ended without a report")
    report["global_optimizer_manager_overrides_removed"] = (
        len(global_optim_manager.module_weight_config_triple) == overrides_before
    )
    report["cuda_after_release"] = _cuda_memory_snapshot(torch)
    report["trainer_retained"] = False
    report["model_retained"] = False
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--model-load-only", action="store_true")
    mode.add_argument("--trainer-construction-only", action="store_true")
    mode.add_argument("--rollout-only", action="store_true")
    mode.add_argument("--gradient-only", action="store_true")
    mode.add_argument("--optimizer-construction-only", action="store_true")
    mode.add_argument("--one-update-only", action="store_true")
    mode.add_argument("--five-step-smoke", action="store_true")
    mode.add_argument("--full-run-300", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fixture-data", default=DEFAULT_FIXTURE_DATA)
    parser.add_argument("--fixture-manifest", default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--full-run-data", default=DEFAULT_FULL_RUN_DATA)
    parser.add_argument("--full-run-manifest", default=DEFAULT_FULL_RUN_MANIFEST)
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--report-file",
        help="new JSON evidence file; valid with rollout, gradient, optimizer or one-update mode",
    )
    return parser.parse_args(argv)


def validate_five_step_launch_args(args: argparse.Namespace) -> dict:
    """Fail before preflight unless the destructive launch surface is locked."""
    if not args.five_step_smoke:
        raise ValueError("five-step launch validation requires five-step mode")
    commit = args.expected_commit
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise SystemExit(
            "--five-step-smoke requires a full lowercase --expected-commit"
        )
    repo_root = Path(args.repo_root).resolve()
    requested_output = _resolve(repo_root, args.output_dir)
    reserved_output = _resolve(repo_root, DEFAULT_OUTPUT_DIR)
    if requested_output != reserved_output:
        raise SystemExit(
            "--five-step-smoke must use the reserved runs/grpo-first-smoke output"
        )
    if (
        not math.isfinite(args.minimum_free_gib)
        or args.minimum_free_gib < DEFAULT_MINIMUM_FREE_GIB
    ):
        raise SystemExit(
            "--five-step-smoke requires a preflight disk floor of at least 3 GiB"
        )
    if args.report_file is not None:
        raise SystemExit(
            "--five-step-smoke publishes only its atomic bundle; "
            "--report-file is forbidden"
        )
    return {
        "expected_commit": commit,
        "reserved_output": str(reserved_output),
        "minimum_free_gib": float(args.minimum_free_gib),
        "standalone_report_forbidden": True,
        "passed": True,
    }


def validate_full_run_300_launch_args(args: argparse.Namespace) -> dict:
    """Validate the fixed 300-step launch surface without running preflight."""
    if not args.full_run_300:
        raise ValueError("full-run launch validation requires --full-run-300")

    commit = args.expected_commit
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise SystemExit(
            "--full-run-300 requires a full lowercase --expected-commit"
        )

    repo_root = Path(args.repo_root).resolve()
    path_locks = {
        "training_data": (
            _resolve(repo_root, args.full_run_data),
            _resolve(repo_root, DEFAULT_FULL_RUN_DATA),
        ),
        "pool_manifest": (
            _resolve(repo_root, args.full_run_manifest),
            _resolve(repo_root, DEFAULT_FULL_RUN_MANIFEST),
        ),
        "selection_manifest": (
            _resolve(repo_root, args.selection_manifest),
            _resolve(repo_root, DEFAULT_SELECTION_MANIFEST),
        ),
        "adapter": (
            _resolve(repo_root, args.adapter),
            _resolve(repo_root, DEFAULT_ADAPTER),
        ),
        "output": (
            _resolve(repo_root, args.output_dir),
            _resolve(repo_root, DEFAULT_FULL_RUN_OUTPUT_DIR),
        ),
    }
    for name, (requested, locked) in path_locks.items():
        if requested != locked:
            raise SystemExit(
                f"--full-run-300 must use the locked {name} path: {locked}"
            )

    if (
        not math.isfinite(args.minimum_free_gib)
        or args.minimum_free_gib < DEFAULT_MINIMUM_FREE_GIB
    ):
        raise SystemExit(
            "--full-run-300 requires a preflight disk floor of at least 3 GiB"
        )
    if args.report_file is not None:
        raise SystemExit(
            "--full-run-300 publishes only its declared run artifacts; "
            "--report-file is forbidden"
        )

    contract = full_run_300_contract(output_dir=path_locks["output"][0])
    return {
        "expected_commit": commit,
        "repo_root": str(repo_root),
        "training_data": str(path_locks["training_data"][0]),
        "pool_manifest": str(path_locks["pool_manifest"][0]),
        "selection_manifest": str(path_locks["selection_manifest"][0]),
        "adapter": str(path_locks["adapter"][0]),
        "reserved_output": str(path_locks["output"][0]),
        "minimum_free_gib": float(args.minimum_free_gib),
        "standalone_report_forbidden": True,
        "contract_version": contract["version"],
        "contract_status": contract["status"],
        "training_dispatch_enabled": False,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.full_run_300:
        launch_control = validate_full_run_300_launch_args(args)
        report = run_full_run_300_preflight(
            repo_root=args.repo_root,
            training_data=args.full_run_data,
            pool_manifest=args.full_run_manifest,
            selection_manifest=args.selection_manifest,
            adapter=args.adapter,
            output_dir=args.output_dir,
            minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
            expected_commit=args.expected_commit,
        )
        report["launch_control"] = launch_control
        report["preflight_only"] = True
        report["stop_reason"] = (
            "full-run preflight passed; model loading and training dispatch "
            "remain intentionally unavailable"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not (
        args.preflight_only
        or args.model_load_only
        or args.trainer_construction_only
        or args.rollout_only
        or args.gradient_only
        or args.optimizer_construction_only
        or args.one_update_only
        or args.five_step_smoke
    ):
        raise SystemExit(
            "training is intentionally unavailable; pass --preflight-only or "
            "--model-load-only, --trainer-construction-only, --rollout-only, "
            "--gradient-only, --optimizer-construction-only, --one-update-only, "
            "or --five-step-smoke"
        )
    launch_control = None
    if args.five_step_smoke:
        launch_control = validate_five_step_launch_args(args)
    report_path = None
    if args.report_file is not None:
        if not (
            args.rollout_only
            or args.gradient_only
            or args.optimizer_construction_only
            or args.one_update_only
        ):
            raise SystemExit(
                "--report-file is valid only with --rollout-only, --gradient-only, "
                "--optimizer-construction-only, or --one-update-only"
            )
        report_path = _resolve(Path(args.repo_root).resolve(), args.report_file)
        if report_path.exists():
            raise FileExistsError(f"evidence report already exists: {report_path}")
        if not report_path.parent.is_dir():
            raise FileNotFoundError(
                f"evidence report parent does not exist: {report_path.parent}"
            )
    report = run_preflight(
        repo_root=args.repo_root,
        fixture_data=args.fixture_data,
        fixture_manifest=args.fixture_manifest,
        selection_manifest=args.selection_manifest,
        adapter=args.adapter,
        output_dir=args.output_dir,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
        expected_commit=args.expected_commit,
    )
    if args.five_step_smoke and report.get("status") != "passed":
        raise RuntimeError("five-step smoke cannot continue after failed preflight")
    if args.model_load_only:
        report["model_load"] = run_model_load_gate(
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.trainer_construction_only:
        report["trainer_construction"] = run_trainer_construction_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.rollout_only:
        report["rollout_gate"] = run_rollout_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = True
        report["optimizer_constructed"] = False
        report["training_steps"] = 0
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.gradient_only:
        report["gradient_gate"] = run_gradient_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = True
        report["loss_computed"] = True
        report["backward_performed"] = True
        report["optimizer_constructed"] = False
        report["optimizer_step_performed"] = False
        report["training_steps"] = 0
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.optimizer_construction_only:
        report["optimizer_construction_gate"] = run_optimizer_construction_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = False
        report["loss_computed"] = False
        report["backward_performed"] = False
        report["optimizer_constructed"] = True
        report["optimizer_state_initialized"] = False
        report["optimizer_step_performed"] = False
        report["training_steps"] = 0
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.one_update_only:
        report["one_update_gate"] = run_one_update_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = True
        report["loss_computed"] = True
        report["backward_performed"] = True
        report["optimizer_constructed"] = True
        report["optimizer_state_initialized"] = True
        report["lr_scheduler_constructed"] = True
        report["optimizer_step_performed"] = True
        report["training_steps"] = 1
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.five_step_smoke:
        report["five_step_smoke_gate"] = run_five_step_smoke_gate(
            preflight_report=report,
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            final_output_dir=report["output"]["path"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["launch_control"] = launch_control
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = True
        report["loss_computed"] = True
        report["backward_performed"] = True
        report["optimizer_constructed"] = True
        report["optimizer_state_initialized"] = True
        report["lr_scheduler_constructed"] = True
        report["optimizer_step_performed"] = True
        report["training_steps"] = EXPECTED_STEPS
        report["rollout_records"] = EXPECTED_ROLLOUTS
        report["final_adapter_saved"] = True
        report["atomic_bundle_published"] = True
        report["output"]["created"] = True
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if report_path is not None:
        report["report_artifact"] = {
            "path": str(report_path),
            "created": True,
            "overwrite_allowed": False,
        }
    serialized_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path is not None:
        with report_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized_report)
    print(serialized_report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
