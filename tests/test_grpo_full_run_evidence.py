from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from training.grpo_full_run_evidence import (
    BASE_LEARNING_RATE,
    EXPECTED_ROLLOUTS,
    FULL_RUN_STEPS,
    WARMUP_STEPS,
    expected_full_run_learning_rates,
    validate_full_run_rollout_records,
    validate_full_run_trainer_log_history,
)
from training.grpo_smoke_artifacts import EXPECTED_REWARD_NAMES
from training.train_grpo import (
    FullRunRolloutCollector,
    make_full_run_rollout_capturing_trainer_class,
)


class FakeFullRunTrainer:
    """Mimic TRL retaining only the latest generated group."""

    def __init__(self):
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_func_names = list(EXPECTED_REWARD_NAMES)
        self.reward_weights = [1.0, 1.0, 2.0]
        self._logs = {}

    def _generate_and_score_completions(self, inputs):
        step = self.state.global_step + 1
        agreement = [float(index % 2) for index in range(8)]
        # Preserve an auditable but non-fatal zero-variance group.
        if step == 2:
            agreement = [1.0] * 8
        self._logs = {
            "completion": [f"step-{step}-completion-{index}" for index in range(8)],
            "advantages": [
                0.0 if step == 2 else (-1.0 if value == 0 else 1.0)
                for value in agreement
            ],
            "rewards": {
                EXPECTED_REWARD_NAMES[0]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[1]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[2]: agreement,
            },
        }
        masks = [[1] * (80 + index) for index in range(8)]
        # Preserve an auditable but non-fatal masked truncation.
        if step == 3:
            masks[4] = [0] * 170
        return {"completion_mask": masks, "latest_step": step}


def _run_fake_steps(trainer, count: int) -> None:
    for step in range(1, count + 1):
        inputs = [{"sku_id": f"sku-{step}"} for _ in range(8)]
        prepared = trainer._generate_and_score_completions(inputs)
        assert prepared["latest_step"] == step
        trainer.state.global_step += 1


def _make_records(steps: int = FULL_RUN_STEPS) -> list[dict]:
    records = []
    for step in range(1, steps + 1):
        for rollout_index in range(8):
            agreement = float(rollout_index % 2)
            records.append(
                {
                    "step": step,
                    "sku_id": f"sku-{step}",
                    "rollout_index": rollout_index,
                    "raw_output": f'{{"step":{step},"i":{rollout_index}}}',
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


def _make_log_history(steps: int = FULL_RUN_STEPS) -> list[dict]:
    rates = expected_full_run_learning_rates(expected_steps=steps)
    history = []
    for step, learning_rate in enumerate(rates, start=1):
        history.append(
            {
                "step": step,
                "loss": -0.01 * step,
                "grad_norm": 0.0 if step == 2 else 0.5,
                "learning_rate": learning_rate,
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
        )
    history.append({"step": steps, "train_runtime": 100.0})
    return history


def test_full_collector_preserves_all_2400_records_before_latest_logs_overwrite():
    collector = FullRunRolloutCollector()
    trainer_class = make_full_run_rollout_capturing_trainer_class(
        FakeFullRunTrainer
    )
    trainer = trainer_class(full_run_rollout_collector=collector)

    _run_fake_steps(trainer, FULL_RUN_STEPS)
    evidence = collector.finalize()

    assert collector.captured_steps == 300
    assert len(evidence["records"]) == EXPECTED_ROLLOUTS
    assert evidence["records"][0]["raw_output"] == "step-1-completion-0"
    assert evidence["records"][-1]["raw_output"] == "step-300-completion-7"
    assert trainer._logs["completion"][0] == "step-300-completion-0"
    assert evidence["validation"]["zero_variance_steps"] == 1
    assert evidence["validation"]["truncated_and_masked_records"] == 1
    assert evidence["all_groups_captured_before_overwrite"]


def test_step_100_snapshot_is_validated_and_isolated_from_later_mutation():
    collector = FullRunRolloutCollector()
    trainer_class = make_full_run_rollout_capturing_trainer_class(
        FakeFullRunTrainer
    )
    trainer = trainer_class(full_run_rollout_collector=collector)
    _run_fake_steps(trainer, 100)

    snapshot = collector.snapshot(expected_step=100)
    snapshot["records"][0]["raw_output"] = "mutated"

    assert len(snapshot["records"]) == 800
    assert snapshot["validation"]["steps"] == 100
    assert collector.records[0]["raw_output"] == "step-1-completion-0"


def test_full_collector_rejects_duplicate_sku_reward_drift_and_multiprocess():
    collector = FullRunRolloutCollector(expected_steps=2)
    trainer_class = make_full_run_rollout_capturing_trainer_class(
        FakeFullRunTrainer
    )
    trainer = trainer_class(full_run_rollout_collector=collector)
    trainer._generate_and_score_completions([{"sku_id": "same"}] * 8)
    trainer.state.global_step += 1
    with pytest.raises(RuntimeError, match="repeated across steps"):
        trainer._generate_and_score_completions([{"sku_id": "same"}] * 8)

    collector = FullRunRolloutCollector(expected_steps=1)
    trainer = trainer_class(full_run_rollout_collector=collector)
    trainer.reward_weights = [1.0, 2.0, 1.0]
    with pytest.raises(RuntimeError, match="reward weights drifted"):
        trainer._generate_and_score_completions([{"sku_id": "sku"}] * 8)

    collector = FullRunRolloutCollector(expected_steps=1)
    trainer = trainer_class(full_run_rollout_collector=collector)
    trainer.accelerator.num_processes = 2
    with pytest.raises(RuntimeError, match="exactly one process"):
        trainer._generate_and_score_completions([{"sku_id": "sku"}] * 8)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update(weighted_total=99.0), "weighted total"),
        (
            lambda rows: rows[0]["component_rewards"].update(
                {EXPECTED_REWARD_NAMES[0]: 0.5}
            ),
            "binary",
        ),
        (lambda rows: rows[0].update(advantage=float("nan")), "not finite"),
        (
            lambda rows: [row.update(sku_id="sku-1") for row in rows[8:16]],
            "repeated",
        ),
        (
            lambda rows: rows[0].update(
                effective_completion_tokens=0, truncated_and_masked=False
            ),
            "marker disagrees",
        ),
    ],
)
def test_rollout_validator_fails_closed_on_corrupt_evidence(mutate, message):
    records = _make_records(2)
    mutate(records)
    with pytest.raises(ValueError, match=message):
        validate_full_run_rollout_records(records, expected_steps=2)


def test_full_run_learning_rate_schedule_matches_transformers_step_alignment():
    rates = expected_full_run_learning_rates()

    assert rates[0] == 0.0
    assert math.isclose(rates[29], BASE_LEARNING_RATE * 29 / WARMUP_STEPS)
    assert rates[30] == BASE_LEARNING_RATE
    expected_last = BASE_LEARNING_RATE * 0.5 * (
        1.0 + math.cos(math.pi * (299 - WARMUP_STEPS) / 270)
    )
    assert math.isclose(rates[-1], expected_last)
    assert rates[-1] > 0.0


def test_trainer_log_validator_records_nonfatal_zero_signal_and_clipping():
    summary = validate_full_run_trainer_log_history(_make_log_history())

    assert summary["step_logs"] == 300
    assert summary["zero_gradient_steps"] == [2]
    assert summary["zero_reward_std_steps"] == [2]
    assert summary["clipped_steps"] == [3]
    assert summary["maximum_completion_clipped_ratio"] == 0.125
    assert len(summary["step_records"]) == 300


def test_step_100_log_validation_keeps_full_run_schedule_denominator():
    logs = _make_log_history(100)
    summary = validate_full_run_trainer_log_history(logs, expected_steps=100)

    assert summary["step_logs"] == 100
    assert summary["expected_learning_rates"][-1] == (
        expected_full_run_learning_rates()[99]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda logs: logs[10].pop("reward"), "missing metrics"),
        (lambda logs: logs[10].update(loss=float("inf")), "not finite"),
        (lambda logs: logs[10].update(learning_rate=9e-6), "schedule drifted"),
        (lambda logs: logs[10].update(grad_norm=-1.0), "negative gradient"),
        (
            lambda logs: logs[10].update(**{"completions/clipped_ratio": 1.1}),
            "outside \\[0, 1\\]",
        ),
        (
            lambda logs: logs[10].update(
                **{"rewards/golden_agreement_reward/mean": 1.1}
            ),
            "reward mean",
        ),
    ],
)
def test_trainer_log_validator_fails_closed_on_bad_metrics(mutate, message):
    logs = _make_log_history()
    mutate(logs)
    with pytest.raises(ValueError, match=message):
        validate_full_run_trainer_log_history(logs)


def test_trainer_log_validator_requires_some_positive_gradient_signal():
    logs = _make_log_history()
    for entry in logs:
        if "loss" in entry:
            entry["grad_norm"] = 0.0

    with pytest.raises(ValueError, match="no positive gradient"):
        validate_full_run_trainer_log_history(logs)


def test_rollout_validator_requires_some_group_relative_signal():
    records = _make_records(2)
    for record in records:
        record["component_rewards"][EXPECTED_REWARD_NAMES[2]] = 1.0
        record["weighted_total"] = 4.0
        record["advantage"] = 0.0

    with pytest.raises(ValueError, match="no usable group-relative"):
        validate_full_run_rollout_records(records, expected_steps=2)


def test_collector_wrapper_requires_exact_type():
    trainer_class = make_full_run_rollout_capturing_trainer_class(
        FakeFullRunTrainer
    )
    with pytest.raises(TypeError, match="requires a FullRunRolloutCollector"):
        trainer_class(full_run_rollout_collector=object())

    with pytest.raises(TypeError, match="base trainer must be a class"):
        make_full_run_rollout_capturing_trainer_class(object())
