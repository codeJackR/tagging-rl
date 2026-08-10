"""CPU-only evidence validation for the first 300-step GRPO run."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Sequence

from training.grpo_smoke_artifacts import (
    EXPECTED_MAX_COMPLETION_LENGTH,
    EXPECTED_REWARD_NAMES,
    EXPECTED_REWARD_WEIGHTS,
)

FULL_RUN_STEPS = 300
GENERATIONS_PER_STEP = 8
EXPECTED_ROLLOUTS = FULL_RUN_STEPS * GENERATIONS_PER_STEP
WARMUP_STEPS = 30
BASE_LEARNING_RATE = 5e-6


def _finite_float(value: object, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} is not finite")
    return numeric


def _sku_order_sha256(sku_ids: Sequence[str]) -> str:
    payload = json.dumps(list(sku_ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_full_run_rollout_records(
    records: Sequence[dict],
    *,
    expected_steps: int = FULL_RUN_STEPS,
) -> dict:
    """Validate an ordered prefix or all of the 300 eight-rollout groups."""
    if not isinstance(expected_steps, int) or not 1 <= expected_steps <= FULL_RUN_STEPS:
        raise ValueError("expected_steps must be between 1 and 300")
    expected_records = expected_steps * GENERATIONS_PER_STEP
    if len(records) != expected_records:
        raise ValueError(f"expected exactly {expected_records} rollout records")

    sku_order: list[str] = []
    weighted_totals_by_step: list[list[float]] = []
    reward_variance_steps = 0
    nonzero_advantages = 0
    total_completion_tokens = 0
    truncated_and_masked_records = 0

    for step in range(1, expected_steps + 1):
        start = (step - 1) * GENERATIONS_PER_STEP
        group = records[start : start + GENERATIONS_PER_STEP]
        if [record.get("step") for record in group] != [step] * GENERATIONS_PER_STEP:
            raise ValueError(f"rollout records are not contiguous for step {step}")
        if [record.get("rollout_index") for record in group] != list(
            range(GENERATIONS_PER_STEP)
        ):
            raise ValueError(f"rollout indices drifted at step {step}")

        group_skus = [record.get("sku_id") for record in group]
        sku_id = group_skus[0]
        if not isinstance(sku_id, str) or not sku_id.strip():
            raise ValueError(f"step {step} has no SKU")
        if group_skus != [sku_id] * GENERATIONS_PER_STEP:
            raise ValueError(f"rollout SKU drifted at step {step}")
        if sku_id in sku_order:
            raise ValueError(f"SKU repeated across full-run steps: {sku_id}")
        sku_order.append(sku_id)

        group_totals: list[float] = []
        for record in group:
            raw_output = record.get("raw_output")
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise ValueError("rollout record has no raw model output")
            rewards = record.get("component_rewards")
            if not isinstance(rewards, dict) or list(rewards) != list(
                EXPECTED_REWARD_NAMES
            ):
                raise ValueError("rollout reward components or order drifted")
            component_values = [
                _finite_float(rewards[name], label=f"reward {name}")
                for name in EXPECTED_REWARD_NAMES
            ]
            if any(value not in (0.0, 1.0) for value in component_values):
                raise ValueError("component rewards must remain binary")
            expected_total = sum(
                value * weight
                for value, weight in zip(
                    component_values, EXPECTED_REWARD_WEIGHTS
                )
            )
            stored_total = _finite_float(
                record.get("weighted_total"), label="weighted total"
            )
            if not math.isclose(stored_total, expected_total, abs_tol=1e-9):
                raise ValueError("rollout weighted total disagrees with components")
            group_totals.append(stored_total)

            advantage = _finite_float(record.get("advantage"), label="advantage")
            if not math.isclose(advantage, 0.0, abs_tol=1e-8):
                nonzero_advantages += 1

            completion_tokens = record.get("effective_completion_tokens")
            if not isinstance(completion_tokens, int) or not (
                0 <= completion_tokens <= EXPECTED_MAX_COMPLETION_LENGTH
            ):
                raise ValueError("rollout completion-token count is outside bounds")
            masked = record.get("truncated_and_masked")
            if not isinstance(masked, bool):
                raise ValueError("truncation marker must be boolean")
            if masked != (completion_tokens == 0):
                raise ValueError("truncation marker disagrees with completion mask")
            if masked:
                truncated_and_masked_records += 1
            total_completion_tokens += completion_tokens

        weighted_totals_by_step.append(group_totals)
        if len(set(group_totals)) > 1:
            reward_variance_steps += 1

    if reward_variance_steps == 0 or nonzero_advantages == 0:
        raise ValueError("full run produced no usable group-relative reward signal")
    return {
        "records": len(records),
        "steps": expected_steps,
        "generations_per_step": GENERATIONS_PER_STEP,
        "sku_order": sku_order,
        "sku_order_sha256": _sku_order_sha256(sku_order),
        "unique_skus": len(set(sku_order)),
        "reward_variance_steps": reward_variance_steps,
        "zero_variance_steps": expected_steps - reward_variance_steps,
        "nonzero_advantages": nonzero_advantages,
        "truncated_and_masked_records": truncated_and_masked_records,
        "mean_completion_tokens": total_completion_tokens / len(records),
        "weighted_totals_by_step": weighted_totals_by_step,
        "ordered_unique_sku_mapping_verified": True,
    }


def expected_full_run_learning_rates(
    *, expected_steps: int = FULL_RUN_STEPS
) -> list[float]:
    """Reproduce Transformers' logged warmup-plus-cosine schedule.

    A log labelled step N contains the scheduler value based on current step
    N - 1. Warmup and cosine duration remain locked to the complete 300-step
    run even when validating the first 100-step milestone.
    """
    if not isinstance(expected_steps, int) or not 1 <= expected_steps <= FULL_RUN_STEPS:
        raise ValueError("expected_steps must be between 1 and 300")
    rates = []
    decay_steps = FULL_RUN_STEPS - WARMUP_STEPS
    for logged_step in range(1, expected_steps + 1):
        current_step = logged_step - 1
        if current_step < WARMUP_STEPS:
            multiplier = current_step / WARMUP_STEPS
        else:
            progress = (current_step - WARMUP_STEPS) / decay_steps
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        rates.append(BASE_LEARNING_RATE * multiplier)
    return rates


def validate_full_run_trainer_log_history(
    log_history: Sequence[dict],
    *,
    expected_steps: int = FULL_RUN_STEPS,
) -> dict:
    """Require one finite scalar record per step and summarize health telemetry."""
    expected_lrs = expected_full_run_learning_rates(expected_steps=expected_steps)
    step_logs = [entry for entry in log_history if "loss" in entry]
    if [entry.get("step") for entry in step_logs] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError(
            f"trainer log does not contain exactly steps 1 through {expected_steps}"
        )

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
    }
    for reward_name in EXPECTED_REWARD_NAMES:
        required_metrics.update(
            {
                f"rewards/{reward_name}/mean",
                f"rewards/{reward_name}/std",
            }
        )

    losses: list[float] = []
    gradient_norms: list[float] = []
    clipping_ratios: list[float] = []
    zero_gradient_steps: list[int] = []
    clipped_steps: list[int] = []
    zero_std_steps: list[int] = []
    for index, entry in enumerate(step_logs):
        missing = required_metrics - set(entry)
        if missing:
            raise ValueError(f"trainer step log is missing metrics: {sorted(missing)}")
        metrics = {
            key: _finite_float(entry[key], label=f"trainer metric {key}")
            for key in required_metrics
        }
        step = index + 1
        if metrics["grad_norm"] < 0:
            raise ValueError("trainer logged a negative gradient norm")
        if metrics["grad_norm"] == 0:
            zero_gradient_steps.append(step)
        if not math.isclose(
            metrics["learning_rate"],
            expected_lrs[index],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("trainer learning-rate schedule drifted")
        for key in (
            "frac_reward_zero_std",
            "completions/clipped_ratio",
            "clip_ratio/low_mean",
            "clip_ratio/high_mean",
        ):
            if not 0.0 <= metrics[key] <= 1.0:
                raise ValueError(f"trainer ratio is outside [0, 1]: {key}")
        for key in (
            "reward_std",
            *(f"rewards/{name}/std" for name in EXPECTED_REWARD_NAMES),
        ):
            if metrics[key] < 0:
                raise ValueError(f"trainer standard deviation is negative: {key}")
        for reward_name in EXPECTED_REWARD_NAMES:
            key = f"rewards/{reward_name}/mean"
            if not 0.0 <= metrics[key] <= 1.0:
                raise ValueError(f"binary reward mean is outside [0, 1]: {key}")
        mean_length = metrics["completions/mean_length"]
        max_length = metrics["completions/max_length"]
        if not 0.0 <= mean_length <= max_length <= EXPECTED_MAX_COMPLETION_LENGTH:
            raise ValueError("trainer completion lengths are outside bounds")

        clipped_ratio = metrics["completions/clipped_ratio"]
        if clipped_ratio > 0:
            clipped_steps.append(step)
        if metrics["frac_reward_zero_std"] > 0:
            zero_std_steps.append(step)
        losses.append(metrics["loss"])
        gradient_norms.append(metrics["grad_norm"])
        clipping_ratios.append(clipped_ratio)

    if not any(value > 0 for value in gradient_norms):
        raise ValueError("full run logged no positive gradient norm")
    return {
        "step_logs": len(step_logs),
        "step_records": copy.deepcopy(step_logs),
        "expected_learning_rates": expected_lrs,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "zero_gradient_steps": zero_gradient_steps,
        "zero_gradient_step_count": len(zero_gradient_steps),
        "clipping_ratios": clipping_ratios,
        "clipped_steps": clipped_steps,
        "clipped_step_count": len(clipped_steps),
        "maximum_completion_clipped_ratio": max(clipping_ratios),
        "zero_reward_std_steps": zero_std_steps,
        "zero_reward_std_step_count": len(zero_std_steps),
        "all_metrics_finite": True,
        "at_least_one_positive_gradient_norm": True,
    }
