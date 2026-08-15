#!/usr/bin/env python3
"""Executable lock, preflight and quality policy for the Run 2 causal GPU trial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import read_jsonl
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_checkpoint_monitor_control import CheckpointMonitorCoordinator


VERSION = "grpo-run2-causal-gpu-experiment-v1"
PREFLIGHT_VERSION = "grpo-run2-causal-gpu-preflight-v1"
CONSTRUCTION_VERSION = "grpo-run2-causal-config-construction-v1"
QUALITY_POLICY_VERSION = "grpo-run2-causal-quality-policy-v1"
DEFAULT_DOCUMENT = "W2_GRPO_RUN2_CAUSAL_EXPERIMENT_CONTRACT.md"
DEFAULT_SCHEDULE = "data/grpo_run2_causal_schedule_v1.jsonl"
DEFAULT_SCHEDULE_MANIFEST = "runs/grpo-run2-causal-schedule.json"
DEFAULT_BASELINE_DIR = "runs/grpo-run2-checkpoint-monitor-baseline-sft"
DEFAULT_BASELINE_RECEIPT = "runs/grpo-run2-checkpoint-monitor-baseline-sft.receipt.json"
DEFAULT_OUTPUT = "runs/grpo-run2-causal-experiment-contract.json"
DEFAULT_PREFLIGHT_OUTPUT = "runs/grpo-run2-causal-preflight.json"
DEFAULT_CONSTRUCTION_OUTPUT = "runs/grpo-run2-causal-construction.json"
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
STARTING_ADAPTER = "runs/sft-combined-2epoch/checkpoint-406"
STARTING_ADAPTER_SHA256 = "00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af"
CHECKPOINT_STEPS = (100, 200, 300)
TRAINING_STEPS = 300
GENERATIONS_PER_STEP = 8
SCHEDULE_ROWS = 300
TRAINING_SEED = 42
BOOTSTRAP_SEED = 20_260_821
BOOTSTRAP_REPLICATES = 10_000
MINIMUM_SUITE_START_BYTES = 3 * 1024**3
MINIMUM_ARM_B_START_BYTES = int(2.5 * 1024**3)
MINIMUM_POST_ARM_BYTES = 2 * 1024**3
MINIMUM_MONITOR_DRIVER_FREE_BYTES = 6 * 1024**3
EXPECTED_ENVIRONMENT = {
    "python": "3.12.12",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "datasets": "4.3.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "unsloth": "2026.7.5",
}
EXPECTED_GPU = {
    "name": "NVIDIA GeForce RTX 3090",
    "memory_mib": 24_576,
    "driver": "590.48.01",
}
EXECUTION_CODE_FILES = (
    "training/run2_causal_experiment.py",
    "training/build_run2_causal_schedule.py",
    "training/train_grpo.py",
    "training/dataset.py",
    "training/rewards.py",
    "training/run2_rewards.py",
    "training/grpo_full_run_artifacts.py",
    "training/grpo_full_run_evidence.py",
    "training/grpo_phase_profiler.py",
    "training/grpo_detached_control.py",
    "training/grpo_detached_runtime.py",
    "training/grpo_full_run_workload.py",
    "training/run2_checkpoint_monitor_contract.py",
    "training/run2_checkpoint_monitor.py",
    "training/run2_checkpoint_monitor_runtime.py",
    "training/run2_checkpoint_monitor_control.py",
    "training/predict.py",
    "training/audit_data_boundaries.py",
    "evalharness/metrics.py",
    "evalharness/predictions.py",
    "labeling/records.py",
    "verifier/__init__.py",
    "verifier/rules.py",
    "verifier/schema.py",
)
LOCKED_INPUTS = {
    "contract_document": DEFAULT_DOCUMENT,
    "schedule": DEFAULT_SCHEDULE,
    "schedule_manifest": DEFAULT_SCHEDULE_MANIFEST,
    "corrected_pool": "data/train_weak_grpo_cap4_sft_train_v1.jsonl",
    "corrected_pool_manifest": "runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json",
    "sft_selection": "runs/sft-selection.json",
    "reward_selection": "runs/grpo-run2-d4-reward-selection.json",
    "reward_gate_g1_g9": "runs/grpo-run2-d4-gates-g1-g9.json",
    "reward_gate_g10": "runs/grpo-run2-gate-g10-result.json",
    "monitor_contract": "runs/grpo-run2-checkpoint-monitor-contract.json",
    "monitor_smoke_manifest": "runs/grpo-run2-checkpoint-monitor-smoke-v3/manifest.json",
    "monitor_smoke_receipt": "runs/grpo-run2-checkpoint-monitor-smoke-v3.receipt.json",
    "baseline_manifest": f"{DEFAULT_BASELINE_DIR}/manifest.json",
    "baseline_report": f"{DEFAULT_BASELINE_DIR}/report.json",
    "baseline_resource": f"{DEFAULT_BASELINE_DIR}/resource.json",
    "baseline_receipt": DEFAULT_BASELINE_RECEIPT,
    "pack_vocab": "packs/vastraa_taste_v1/vocab.yaml",
    "pack_rules": "packs/vastraa_taste_v1/rules.yaml",
}
# Guardrails that may abort a run, versus guardrails that are recorded only.
#
# The original policy aborted on any repeated breach. Arm A's first production
# attempt was killed at step 200 by rule-violation rate and sampled vocabulary
# validity, while macro-F1 was slightly ABOVE the SFT baseline and rising.
#
# Measured against the same 360 dev prompts, the original reward raises the
# rule-violation rate to 2.0-2.5x its baseline; the threshold permits 1.72x.
# Run 1 used this reward, completed 300 steps, and finished at 2.33x, so it
# would have breached at every checkpoint too. The control arm was therefore
# disqualified by construction: its purpose is to reproduce the degradation the
# policy treats as a reason to stop.
#
# Compliance metrics remain measured, reported and warned on for both arms.
# They are simply not grounds for termination, because they are the phenomenon
# under study rather than a sign the run has gone wrong. For Arm B the same
# numbers carry the opposite meaning -- its dense reward prices each violation,
# so failing to hold them is that arm's headline result, not an abort trigger.
ABORTING_METRICS = frozenset({"macro_f1", "selective_macro_f1", "coverage"})

PRACTICAL_MARGINS = (
    ("representative_all", "macro_f1", "lower", 0.05),
    ("representative_all", "selective_macro_f1", "lower", 0.05),
    ("representative_all", "coverage", "lower", 0.10),
    ("representative_all", "schema_validity", "lower", 0.02),
    ("representative_all", "vocab_validity", "lower", 0.02),
    ("representative_all", "rule_violation_rate", "upper", 0.02),
    ("easy_retention_6_to_8_of_8", "macro_f1", "lower", 0.05),
    ("easy_retention_6_to_8_of_8", "coverage", "lower", 0.10),
)


class CausalQualityAbort(RuntimeError):
    """Raised after a repeated quality breach has durable evidence."""


def _identity(path: Path, root: Path) -> dict[str, Any]:
    try:
        display = str(path.relative_to(root))
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git_blob_identity(root: Path, commit: str, relative: str) -> dict[str, Any]:
    try:
        raw = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"execution file is absent from expected commit: {relative}") from exc
    current = (root / relative).read_bytes()
    if current != raw:
        raise RuntimeError(f"execution file differs from expected code commit: {relative}")
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _valid_commit(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def common_training_config(*, output_dir: str, run_name: str) -> dict[str, Any]:
    """Return every shared trainer setting without importing a GPU library."""
    return {
        "output_dir": output_dir,
        "run_name": run_name,
        "seed": TRAINING_SEED,
        "data_seed": TRAINING_SEED,
        "max_prompt_length": 600,
        "max_completion_length": 170,
        "num_generations": GENERATIONS_PER_STEP,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "steps_per_generation": 1,
        "max_steps": TRAINING_STEPS,
        "shuffle_dataset": False,
        "remove_unused_columns": False,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "use_vllm": False,
        "learning_rate": 5e-6,
        "weight_decay": 0.001,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "beta": 0.0,
        "num_iterations": 1,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "scale_rewards": "group",
        "loss_type": "dapo",
        "mask_truncated_completions": True,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "log_completions": False,
        "num_completions_to_print": 8,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "save_only_model": True,
    }


def arm_spec(arm: str) -> dict[str, Any]:
    if arm == "A":
        output = "runs/grpo-run2-arm-a-original"
        reward = {
            "policy": "original_1_1_2",
            "functions": [
                "format_validity_reward",
                "vocab_rule_compliance_reward",
                "golden_agreement_reward",
            ],
            "weights": [1.0, 1.0, 2.0],
            "implementation": "training/rewards.py",
        }
    elif arm == "B":
        output = "runs/grpo-run2-arm-b-ua"
        reward = {
            "policy": "candidate_ua",
            "functions": ["candidate_ua_reward"],
            "weights": [1.0],
            "implementation": "training/run2_rewards.py",
        }
    else:
        raise ValueError("causal arm must be A or B")
    return {
        "arm": arm,
        "role": "corrected_control" if arm == "A" else "dense_reward_treatment",
        "output_dir": output,
        "monitor_root": f"{output}-monitor",
        "quality_root": f"{output}-monitor-quality",
        "control_dir": f"{output}-control",
        "failure_dir": f"{output}-failed",
        "reward": reward,
        "config": common_training_config(
            output_dir=output,
            run_name="grpo-run2-arm-a-original" if arm == "A" else "grpo-run2-arm-b-ua",
        ),
    }


def _arm_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    config_diff = {
        key: {"A": a["config"].get(key), "B": b["config"].get(key)}
        for key in sorted(set(a["config"]) | set(b["config"]))
        if a["config"].get(key) != b["config"].get(key)
    }
    if set(config_diff) != {"output_dir", "run_name"}:
        raise RuntimeError(f"non-reward trainer configuration drifted: {config_diff}")
    return {
        "trainer_config_differences": config_diff,
        "reward_difference": {"A": a["reward"], "B": b["reward"]},
        "intended_causal_difference": "reward definition",
        "noncausal_bookkeeping_differences": ["output_dir", "run_name"],
        "all_other_trainer_settings_equal": True,
        "beta_equal_and_explicitly_zero": (
            a["config"]["beta"] == b["config"]["beta"] == 0.0
        ),
    }


def build_quality_policy(baseline_report: Mapping[str, Any]) -> dict[str, Any]:
    if baseline_report.get("status") != "checkpoint_outputs_scored":
        raise ValueError("baseline report is not completely scored")
    if baseline_report.get("rows") != 360 or baseline_report.get("sampled_repetitions") != 8:
        raise ValueError("quality baseline must contain 360 rows and eight repeats")
    guardrails = []
    for view, metric, direction, practical_margin in PRACTICAL_MARGINS:
        baseline_greedy = float(
            baseline_report["views"][view]["greedy"]["scalars"][metric]
        )
        distribution = baseline_report["views"][view]["sampled"]["aggregate"][metric]
        mean = float(distribution["mean"])
        std = float(distribution["population_stddev"])
        variability_allowance = max(2.0 * std, practical_margin)
        if direction == "lower":
            threshold_fn = lambda value: max(
                0.0, value - variability_allowance
            )
        else:
            threshold_fn = lambda value: min(
                1.0, value + variability_allowance
            )
        guardrails.append(
            {
                "key": f"{view}:{metric}",
                "view": view,
                "metric": metric,
                "direction": direction,
                "baseline_greedy": baseline_greedy,
                "baseline_sampled_mean": mean,
                "baseline_sampled_population_stddev": std,
                "practical_margin": practical_margin,
                "variability_allowance": variability_allowance,
                "thresholds": {
                    "greedy": round(threshold_fn(baseline_greedy), 12),
                    "sampled_mean": round(threshold_fn(mean), 12),
                },
            }
        )
    return {
        "version": QUALITY_POLICY_VERSION,
        "status": "baseline_derived_before_training",
        "guardrails": guardrails,
        "consecutive_breaches_to_abort": 2,
        "checkpoints": list(CHECKPOINT_STEPS),
        "one_breach_action": "warn_and_continue",
        "same_key_and_decoding_mode_required": True,
        "clean_checkpoint_resets_sequence": True,
        "monitor_failure_aborts_immediately": True,
        "reward_metrics_can_abort": False,
        "single_checkpoint_quality_abort": False,
    }


def quality_observations(
    report: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if report.get("status") != "checkpoint_outputs_scored":
        raise ValueError("checkpoint quality report is not completely scored")
    observations = []
    for guardrail in policy["guardrails"]:
        view = report["views"][guardrail["view"]]
        values = {
            "greedy": float(view["greedy"]["scalars"][guardrail["metric"]]),
            "sampled_mean": float(
                view["sampled"]["aggregate"][guardrail["metric"]]["mean"]
            ),
        }
        for mode, value in values.items():
            threshold = guardrail["thresholds"][mode]
            breached = (
                value < threshold
                if guardrail["direction"] == "lower"
                else value > threshold
            )
            observations.append(
                {
                    "key": f"{guardrail['key']}:{mode}",
                    "view": guardrail["view"],
                    "metric": guardrail["metric"],
                    "decoding_mode": mode,
                    "direction": guardrail["direction"],
                    "threshold": threshold,
                    "observed": value,
                    "breached": breached,
                }
            )
    return observations


class QualityBreachTracker:
    """Apply the locked repeated-breach rule across checkpoints."""

    def __init__(self, policy: Mapping[str, Any]):
        self.policy = copy.deepcopy(dict(policy))
        self.expected_steps = tuple(self.policy["checkpoints"])
        self._seen: list[int] = []
        self._previous_breaches: set[str] = set()
        self._previous_abortable_breaches: set[str] = set()

    def observe(self, *, step: int, report: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.expected_steps[len(self._seen)] if len(self._seen) < len(self.expected_steps) else None
        if step != expected:
            raise RuntimeError(f"quality tracker expected step {expected}, found {step}")
        observations = quality_observations(report, self.policy)
        current = {value["key"] for value in observations if value["breached"]}
        # Every breach is recorded; only quality metrics can end a run.
        abortable = {
            value["key"]
            for value in observations
            if value["breached"] and value["metric"] in ABORTING_METRICS
        }
        repeated = sorted(abortable & self._previous_abortable_breaches)
        decision = {
            "version": QUALITY_POLICY_VERSION,
            "step": step,
            "status": "abort" if repeated else "warn" if current else "pass",
            "observations": observations,
            "breached_keys": sorted(current),
            "abortable_breached_keys": sorted(abortable),
            "recorded_only_breached_keys": sorted(current - abortable),
            "repeated_consecutive_breach_keys": repeated,
            "abort_training": bool(repeated),
            "single_checkpoint_abort": False,
        }
        self._seen.append(step)
        self._previous_breaches = current
        self._previous_abortable_breaches = abortable
        return decision


class CausalCheckpointMonitorCoordinator:
    """Add the Phase G repeated quality rule around the Phase F coordinator."""

    def __init__(
        self,
        *,
        base: CheckpointMonitorCoordinator,
        quality_policy: Mapping[str, Any],
        quality_output_root: str | Path,
        gpu_free_bytes_fn: Callable[[], int] | None = None,
    ):
        if not isinstance(base, CheckpointMonitorCoordinator):
            raise TypeError("causal monitor requires a Phase F coordinator")
        self.base = base
        self.tracker = QualityBreachTracker(quality_policy)
        self.quality_output_root = Path(quality_output_root).resolve()
        self.gpu_free_bytes_fn = gpu_free_bytes_fn or _driver_free_bytes

    def on_train_begin(self, args: object, state: object) -> dict[str, Any]:
        result = self.base.on_train_begin(args, state)
        self.quality_output_root.mkdir(parents=True, exist_ok=False)
        return result

    def on_save(self, args: object, state: object) -> dict[str, Any]:
        step = int(getattr(state, "global_step", -1))
        free_bytes = int(self.gpu_free_bytes_fn())
        if free_bytes < MINIMUM_MONITOR_DRIVER_FREE_BYTES:
            failure = {
                "version": QUALITY_POLICY_VERSION,
                "status": "checkpoint_monitor_resource_block",
                "step": step,
                "driver_free_bytes": free_bytes,
                "minimum_driver_free_bytes": MINIMUM_MONITOR_DRIVER_FREE_BYTES,
                "training_must_abort": True,
            }
            output = self.quality_output_root / f"checkpoint-{step}.resource-block.json"
            write_exclusive_atomic_json(output, failure)
            raise CausalQualityAbort(
                f"insufficient GPU headroom for checkpoint monitor at step {step}: {output}"
            )
        receipt = self.base.on_save(args, state)
        report_path = self.base.monitor_root / f"checkpoint-{step}" / "report.json"
        decision = self.tracker.observe(step=step, report=_load_json(report_path))
        decision["monitor_receipt"] = receipt
        output = self.quality_output_root / f"checkpoint-{step}.quality.json"
        write_exclusive_atomic_json(output, decision)
        if decision["abort_training"]:
            raise CausalQualityAbort(
                f"repeated checkpoint quality breach at step {step}: {output}"
            )
        return {"monitor_receipt": receipt, "quality_decision": decision}

    def on_train_end(self, state: object) -> dict[str, Any]:
        return self.base.on_train_end(state)


def make_causal_monitor_callback_class(base_callback_class: type) -> type:
    if not isinstance(base_callback_class, type):
        raise TypeError("base callback must be a class")

    class CausalMonitorCallback(base_callback_class):
        def __init__(self, *, coordinator: CausalCheckpointMonitorCoordinator):
            super().__init__()
            if not isinstance(coordinator, CausalCheckpointMonitorCoordinator):
                raise TypeError("callback requires a causal monitor coordinator")
            self.causal_monitor = coordinator

        def on_train_begin(self, args, state, control, **kwargs):
            self.causal_monitor.on_train_begin(args, state)
            return control

        def on_save(self, args, state, control, **kwargs):
            self.causal_monitor.on_save(args, state)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.causal_monitor.on_train_end(state)
            return control

    CausalMonitorCallback.__name__ = f"Run2CausalMonitor{base_callback_class.__name__}"
    return CausalMonitorCallback


def _driver_free_bytes() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.splitlines() if value.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError("could not read one GPU's free memory")
    return int(values[0]) * 1024**2


def build_contract(*, root: str | Path, expected_code_commit: str) -> dict[str, Any]:
    root = Path(root).resolve()
    if not _valid_commit(expected_code_commit):
        raise ValueError("expected code commit must be a full lowercase Git SHA")
    inputs = {name: _identity(root / value, root) for name, value in LOCKED_INPUTS.items()}
    code = {
        relative: _git_blob_identity(root, expected_code_commit, relative)
        for relative in EXECUTION_CODE_FILES
    }
    schedule_manifest = _load_json(root / DEFAULT_SCHEDULE_MANIFEST)
    schedule_rows = read_jsonl(root / DEFAULT_SCHEDULE)
    schedule_skus = [row.sku_id for row in schedule_rows]
    if schedule_manifest.get("status") != "fixed_before_causal_gpu_dispatch":
        raise ValueError("causal schedule is not locked")
    if len(schedule_rows) != SCHEDULE_ROWS or len(set(schedule_skus)) != SCHEDULE_ROWS:
        raise ValueError("causal schedule must contain 300 unique rows")
    if schedule_manifest["selection"]["sku_ids_in_optimizer_step_order"] != schedule_skus:
        raise RuntimeError("causal schedule order drifted")
    if schedule_manifest["selection"]["ordered_sku_sha256"] != _ordered_hash(schedule_skus):
        raise RuntimeError("causal schedule ordered hash drifted")

    reward_selection = _load_json(root / LOCKED_INPUTS["reward_selection"])
    if (
        reward_selection.get("selected_candidate") != "UA"
        or reward_selection.get("status") != "run2_offline_reward_selected_pending_run_contract"
    ):
        raise ValueError("Candidate UA is not the locked offline winner")
    sft = _load_json(root / LOCKED_INPUTS["sft_selection"])["selected_checkpoint"]
    if sft["adapter_weights"]["sha256"] != STARTING_ADAPTER_SHA256:
        raise ValueError("starting SFT adapter drifted")

    baseline_manifest = _load_json(root / LOCKED_INPUTS["baseline_manifest"])
    baseline_receipt = _load_json(root / LOCKED_INPUTS["baseline_receipt"])
    baseline_report = _load_json(root / LOCKED_INPUTS["baseline_report"])
    baseline_resource = _load_json(root / LOCKED_INPUTS["baseline_resource"])
    if (
        baseline_manifest.get("status") != "checkpoint_monitor_complete"
        or baseline_manifest.get("mode") != "production"
        or baseline_manifest.get("quality_evidence") is not True
        or baseline_manifest.get("checkpoint", {}).get("adapter_model_sha256") != STARTING_ADAPTER_SHA256
        or baseline_receipt.get("status") != "checkpoint_monitor_accepted"
        or baseline_receipt.get("timed_out") is not False
    ):
        raise ValueError("starting-SFT production baseline is not accepted")
    quality_policy = build_quality_policy(baseline_report)
    monitor_peak_allocated = int(
        baseline_resource.get("cuda_peak", {}).get("max_allocated_bytes", -1)
    )
    if monitor_peak_allocated <= 0:
        raise ValueError("production baseline has no valid monitor peak memory")
    historical_training_peak_allocated = 4_914_862_080

    arm_a = arm_spec("A")
    arm_b = arm_spec("B")
    arm_difference = _arm_diff(arm_a, arm_b)
    paths = []
    for arm in (arm_a, arm_b):
        paths.extend(
            arm[key]
            for key in (
                "output_dir",
                "monitor_root",
                "quality_root",
                "control_dir",
                "failure_dir",
            )
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("causal arm evidence paths collide")

    contract = {
        "version": VERSION,
        "status": "locked_no_gpu_training_dispatched",
        "question": "effect of Candidate UA versus original 1:1:2 under corrected identical training",
        "expected_execution_code_commit": expected_code_commit,
        "inputs": inputs,
        "execution_code": code,
        "environment": {
            "required": EXPECTED_ENVIRONMENT,
            "gpu": EXPECTED_GPU,
            "offline_model_cache_required": True,
            "single_visible_gpu_required": True,
            "no_environment_change_between_arms": True,
        },
        "starting_policy": {
            "base_model": BASE_MODEL,
            "adapter_path": STARTING_ADAPTER,
            "adapter_weights": sft["adapter_weights"],
            "lora": sft["lora"],
            "same_bytes_for_both_arms": True,
        },
        "schedule": {
            "manifest": inputs["schedule_manifest"],
            "dataset": inputs["schedule"],
            "rows": SCHEDULE_ROWS,
            "ordered_sku_sha256": _ordered_hash(schedule_skus),
            "one_product_per_step": True,
            "trainer_shuffle": False,
        },
        "arm_order": ["A", "B"],
        "arms": {"A": arm_a, "B": arm_b},
        "causal_difference_audit": arm_difference,
        "monitoring": {
            "phase_f_contract": inputs["monitor_contract"],
            "callback": "CausalCheckpointMonitorCoordinator",
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "synchronous": True,
            "minimum_driver_free_bytes_before_child": MINIMUM_MONITOR_DRIVER_FREE_BYTES,
            "quality_policy": quality_policy,
        },
        "endpoint": {
            "primary": "checkpoint-300 representative greedy macro_f1, B minus A",
            "paired_unit": "development product/SKU",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "earlier_checkpoints_are_diagnostic_not_selection_candidates": True,
        },
        "treatment_acceptance": {
            "primary_minimum_improvement": 0.02,
            "primary_interval_lower_bound_must_exceed": 0.0,
            "sampled_macro_f1_mean_minimum_delta": 0.0,
            "maximum_negative_delta": {
                "greedy_coverage": -0.03,
                "greedy_schema_validity": -0.03,
                "greedy_vocab_validity": -0.03,
                "greedy_easy_retention_macro_f1": -0.03,
            },
            "maximum_rule_violation_rate_increase": 0.02,
            "all_required": True,
            "default_if_not_all_pass": "retain_arm_A",
            "missing_or_aborted_arm": "experiment_incomplete_no_winner",
        },
        "resources": {
            "minimum_suite_start_bytes": MINIMUM_SUITE_START_BYTES,
            "minimum_arm_B_start_bytes": MINIMUM_ARM_B_START_BYTES,
            "minimum_post_arm_bytes": MINIMUM_POST_ARM_BYTES,
            "save_total_limit": 2,
            "save_only_model": True,
            "exact_optimizer_resume_available": False,
            "historical_training_peak_allocated_bytes": historical_training_peak_allocated,
            "monitor_peak_allocated_bytes": monitor_peak_allocated,
            "monitor_peak_source": inputs["baseline_resource"],
            "conservative_additive_peak_bytes": (
                historical_training_peak_allocated + monitor_peak_allocated
            ),
        },
        "arm_B_gate": {
            "arm_A_success_manifest_required": True,
            "arm_A_checkpoint_300_monitor_receipt_required": True,
            "same_execution_code_and_environment_required": True,
            "disk_floor_required": True,
        },
        "smoke_to_full_diff": {
            "duration_steps": {"smoke": 5, "causal": 300},
            "dataset": {"smoke": "four-row fixture", "causal": DEFAULT_SCHEDULE},
            "warmup_ratio": {"smoke": 0.0, "causal": 0.1},
            "completion_table_logging": {"smoke": True, "causal": False},
            "checkpointing": {"smoke": "none", "causal": "steps 100/200/300; retain 2"},
            "monitoring": {"smoke": "none", "causal": "greedy plus 8 sampled development passes"},
            "inherited_generation_optimizer_lora_settings_rejustified": True,
        },
        "boundaries": {
            "legacy_run1_is_historical_not_control": True,
            "legacy_frozen_300_allowed": False,
            "confirmation_data_allowed": False,
            "kl_arm_included": False,
            "multi_seed_claim_allowed": False,
            "gpu_training_dispatched_by_contract_build": False,
        },
        "publication": {
            "detached_fail_closed_execution": True,
            "atomic_arm_publication": True,
            "raw_rollouts_and_step_logs_required": True,
            "failed_arms_retained": True,
            "existing_path_means_collision_not_resume": True,
        },
        "deferral_audit": {
            "unresolved_parameters": [],
            "unresolved_thresholds": [],
            "unresolved_output_paths": [],
            "deferred_decisions": [],
            "passed": True,
        },
    }
    serialized = json.dumps(contract, sort_keys=True)
    forbidden_markers = ("<tbd>", "todo", "choose later", "decide later")
    found = [marker for marker in forbidden_markers if marker in serialized.casefold()]
    if found:
        raise RuntimeError(f"causal contract contains unresolved markers: {found}")
    return contract


def _current_environment() -> dict[str, str]:
    observed = {"python": platform.python_version()}
    for package in EXPECTED_ENVIRONMENT:
        if package == "python":
            continue
        observed[package] = importlib.metadata.version(package)
    return observed


def _gpu_state() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("causal preflight requires exactly one visible GPU")
    name, driver, total, used, utilization = [value.strip() for value in lines[0].split(",")]
    return {
        "name": name,
        "driver": driver,
        "memory_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
    }


def run_preflight(
    *,
    root: str | Path,
    contract_path: str | Path,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
    environment_fn: Callable[[], dict[str, str]] = _current_environment,
    gpu_state_fn: Callable[[], dict[str, Any]] = _gpu_state,
) -> dict[str, Any]:
    """Validate the locked experiment without importing CUDA or creating arm paths."""
    root = Path(root).resolve()
    contract_path = (root / contract_path).resolve()
    contract = _load_json(contract_path)
    if contract.get("version") != VERSION or contract.get("status") != "locked_no_gpu_training_dispatched":
        raise ValueError("causal experiment contract is not locked")
    for identity in contract["inputs"].values():
        path = root / identity["path"]
        if not path.is_file() or _identity(path, root) != identity:
            raise RuntimeError(f"causal input drifted: {identity['path']}")
    expected_commit = contract["expected_execution_code_commit"]
    for relative, identity in contract["execution_code"].items():
        if _git_blob_identity(root, expected_commit, relative) != identity:
            raise RuntimeError(f"execution code identity drifted: {relative}")

    environment = environment_fn()
    if environment != contract["environment"]["required"]:
        raise RuntimeError(f"causal environment drifted: {environment}")
    gpu = gpu_state_fn()
    for key, expected in contract["environment"]["gpu"].items():
        if gpu.get(key) != expected:
            raise RuntimeError(f"causal GPU {key} drifted: {gpu.get(key)} != {expected}")
    if gpu["memory_used_mib"] > 1_024 or gpu["utilization_percent"] > 5:
        raise RuntimeError("causal GPU is not idle")

    adapter = root / contract["starting_policy"]["adapter_path"]
    adapter_file = adapter / "adapter_model.safetensors"
    if not adapter_file.is_file() or sha256_file(adapter_file) != STARTING_ADAPTER_SHA256:
        raise RuntimeError("causal starting adapter is missing or drifted")
    paths = []
    for arm in contract["arm_order"]:
        spec = contract["arms"][arm]
        for key in (
            "output_dir",
            "monitor_root",
            "quality_root",
            "control_dir",
            "failure_dir",
        ):
            path = root / spec[key]
            paths.append(str(path))
            if path.exists():
                raise FileExistsError(f"causal output path already exists: {path}")
    free_bytes = int(disk_usage_fn(root / "runs").free)
    if free_bytes < contract["resources"]["minimum_suite_start_bytes"]:
        raise RuntimeError("causal suite-start disk floor is not met")
    if contract["deferral_audit"].get("passed") is not True or any(
        contract["deferral_audit"][key]
        for key in (
            "unresolved_parameters",
            "unresolved_thresholds",
            "unresolved_output_paths",
            "deferred_decisions",
        )
    ):
        raise RuntimeError("causal contract still contains a deferred decision")
    return {
        "version": PREFLIGHT_VERSION,
        "status": "passed_read_only_no_training_dispatch",
        "contract": _identity(contract_path, root),
        "expected_execution_code_commit": expected_commit,
        "current_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "execution_code_matches_expected_commit": True,
        "inputs_verified": len(contract["inputs"]),
        "execution_files_verified": len(contract["execution_code"]),
        "arm_configuration_difference": contract["causal_difference_audit"],
        "environment": environment,
        "gpu": gpu,
        "adapter_sha256": STARTING_ADAPTER_SHA256,
        "schedule_rows": contract["schedule"]["rows"],
        "schedule_order_sha256": contract["schedule"]["ordered_sku_sha256"],
        "collision_checked_paths": paths,
        "disk_free_bytes": free_bytes,
        "minimum_disk_free_bytes": contract["resources"]["minimum_suite_start_bytes"],
        "deferred_decisions": 0,
        "cuda_imports_performed": False,
        "model_loaded": False,
        "trainer_constructed": False,
        "training_dispatched": False,
        "arm_output_paths_created": False,
    }


def _normalized_config_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def _inspect_constructed_config(config: object, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = {}
    for key, expected_value in expected.items():
        actual = _normalized_config_value(getattr(config, key, None))
        if key == "report_to" and actual in ([], (), "none"):
            observed[key] = []
            continue
        if key == "reward_weights":
            actual = list(actual) if actual is not None else None
        if actual != expected_value:
            raise RuntimeError(
                f"causal GRPOConfig normalized {key} unexpectedly: {actual!r} != {expected_value!r}"
            )
        observed[key] = actual
    generation_batch_size = getattr(config, "generation_batch_size", None)
    if generation_batch_size != 8:
        raise RuntimeError("causal GRPOConfig generation batch must be eight")
    return {
        "settings": observed,
        "generation_batch_size": generation_batch_size,
        "prompts_per_generation_batch": generation_batch_size // GENERATIONS_PER_STEP,
        "settings_match_contract": True,
    }


def _reward_callables(arm: str) -> tuple[Callable[..., list[float]], ...]:
    if arm == "A":
        from training.rewards import FIRST_RUN_REWARD_FUNCTIONS

        return FIRST_RUN_REWARD_FUNCTIONS
    if arm == "B":
        from training.run2_rewards import candidate_ua_reward

        return (candidate_ua_reward,)
    raise ValueError("causal arm must be A or B")


def run_construction(
    *,
    root: str | Path,
    contract_path: str | Path,
    preflight_path: str | Path,
    config_class: type | None = None,
) -> dict[str, Any]:
    """Construct both real config/reward surfaces without a model or Trainer."""
    root = Path(root).resolve()
    contract_path = (root / contract_path).resolve()
    preflight_path = (root / preflight_path).resolve()
    contract = _load_json(contract_path)
    preflight = _load_json(preflight_path)
    if preflight.get("status") != "passed_read_only_no_training_dispatch":
        raise ValueError("causal construction requires a passed preflight")
    if preflight.get("contract") != _identity(contract_path, root):
        raise RuntimeError("causal construction preflight names a different contract")
    if any(
        preflight.get(key) is not False
        for key in ("model_loaded", "trainer_constructed", "training_dispatched")
    ):
        raise RuntimeError("causal construction received a mutated preflight")
    if config_class is None:
        import torch
        from trl import GRPOConfig

        config_class = GRPOConfig
        cuda_initialized_before = torch.cuda.is_initialized()
    else:
        torch = None
        cuda_initialized_before = False

    arm_reports = {}
    for arm in contract["arm_order"]:
        spec = contract["arms"][arm]
        expected = {**spec["config"], "reward_weights": spec["reward"]["weights"]}
        config = config_class(**expected)
        callables = _reward_callables(arm)
        names = [getattr(value, "__name__", type(value).__name__) for value in callables]
        if names != spec["reward"]["functions"]:
            raise RuntimeError(f"causal arm {arm} reward callable binding drifted")
        arm_reports[arm] = {
            "config_class": config_class.__name__,
            "config_module": config_class.__module__,
            "config": _inspect_constructed_config(config, expected),
            "reward_callable_names": names,
            "reward_callable_modules": [value.__module__ for value in callables],
            "model_constructed": False,
            "trainer_constructed": False,
            "training_dispatched": False,
        }
    cuda_initialized_after = torch.cuda.is_initialized() if torch is not None else False
    torch_allocated_bytes = (
        int(torch.cuda.memory_allocated())
        if torch is not None and cuda_initialized_after
        else 0
    )
    if torch_allocated_bytes > 64 * 1024**2:
        raise RuntimeError("read-only causal config construction allocated material CUDA memory")
    return {
        "version": CONSTRUCTION_VERSION,
        "status": "both_arm_configs_constructed_no_trainer_no_dispatch",
        "contract": _identity(contract_path, root),
        "preflight": _identity(preflight_path, root),
        "arms": arm_reports,
        "causal_difference_audit": contract["causal_difference_audit"],
        "cuda_context": {
            "initialized_before_config_construction": cuda_initialized_before,
            "initialized_after_config_construction": cuda_initialized_after,
            "torch_allocated_bytes_after": torch_allocated_bytes,
            "maximum_allowed_allocated_bytes": 64 * 1024**2,
            "material_cuda_allocation": False,
        },
        "model_constructed": False,
        "trainer_constructed": False,
        "training_dispatched": False,
        "arm_output_paths_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--expected-code-commit", required=True)
    build.add_argument("--output", default=DEFAULT_OUTPUT)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--contract", default=DEFAULT_OUTPUT)
    preflight.add_argument("--output", default=DEFAULT_PREFLIGHT_OUTPUT)
    construction = sub.add_parser("construct")
    construction.add_argument("--contract", default=DEFAULT_OUTPUT)
    construction.add_argument("--preflight", default=DEFAULT_PREFLIGHT_OUTPUT)
    construction.add_argument("--output", default=DEFAULT_CONSTRUCTION_OUTPUT)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    if args.command == "build":
        result = build_contract(root=root, expected_code_commit=args.expected_code_commit)
        output = root / args.output
    elif args.command == "preflight":
        result = run_preflight(root=root, contract_path=args.contract)
        output = root / args.output
    else:
        result = run_construction(
            root=root,
            contract_path=args.contract,
            preflight_path=args.preflight,
        )
        output = root / args.output
    write_exclusive_atomic_json(output, result)
    print(json.dumps({"output": str(output.relative_to(root)), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
