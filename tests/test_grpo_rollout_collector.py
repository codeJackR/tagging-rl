from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.grpo_smoke_artifacts import EXPECTED_REWARD_NAMES
from training.train_grpo import (
    SmokeRolloutCollector,
    make_rollout_capturing_trainer_class,
)


SKUS = [f"sku-{index}" for index in range(1, 6)]


class FakeGRPOTrainer:
    """Mimic TRL's latest-eight logging behavior without importing Torch."""

    def __init__(self, *, truncate_at: tuple[int, int] | None = None):
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_func_names = list(EXPECTED_REWARD_NAMES)
        self.reward_weights = [1.0, 1.0, 2.0]
        self.truncate_at = truncate_at
        self._logs = {}

    def _generate_and_score_completions(self, inputs):
        step = self.state.global_step + 1
        completions = [f"step-{step}-completion-{index}" for index in range(8)]
        agreement = [float(index % 2) for index in range(8)]
        self._logs = {
            "completion": completions,
            "advantages": [-1.0 if value == 0 else 1.0 for value in agreement],
            "rewards": {
                EXPECTED_REWARD_NAMES[0]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[1]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[2]: agreement,
            },
        }
        masks = [[1] * (80 + index) for index in range(8)]
        if self.truncate_at is not None and self.truncate_at[0] == step:
            masks[self.truncate_at[1]] = [0] * 170
        return {
            "completion_mask": masks,
            "unrelated_tensor": f"prepared-step-{step}",
        }


def make_trainer(*, truncate_at: tuple[int, int] | None = None):
    collector = SmokeRolloutCollector(SKUS)
    trainer_class = make_rollout_capturing_trainer_class(FakeGRPOTrainer)
    trainer = trainer_class(
        smoke_rollout_collector=collector,
        truncate_at=truncate_at,
    )
    return trainer, collector


def run_five_fake_steps(trainer) -> None:
    for step, sku_id in enumerate(SKUS, start=1):
        inputs = [{"sku_id": sku_id, "gold": {}} for _ in range(8)]
        prepared = trainer._generate_and_score_completions(inputs)
        assert prepared["unrelated_tensor"] == f"prepared-step-{step}"
        trainer.state.global_step += 1


def test_capturing_trainer_preserves_all_five_groups_before_deque_overwrite():
    trainer, collector = make_trainer()

    run_five_fake_steps(trainer)
    evidence = collector.finalize()

    assert collector.captured_steps == 5
    assert len(evidence["records"]) == 40
    assert evidence["validation"]["records"] == 40
    assert evidence["validation"]["reward_variance_steps"] == 5
    assert evidence["records"][0]["raw_output"] == "step-1-completion-0"
    assert evidence["records"][31]["raw_output"] == "step-4-completion-7"
    assert evidence["records"][-1]["raw_output"] == "step-5-completion-7"
    assert list(trainer._logs["completion"])[0] == "step-5-completion-0"
    assert evidence["all_groups_captured_before_overwrite"]


def test_collector_returns_copies_not_mutable_internal_records():
    trainer, collector = make_trainer()
    run_five_fake_steps(trainer)

    records = collector.records
    records[0]["component_rewards"][EXPECTED_REWARD_NAMES[0]] = 99.0
    finalized = collector.finalize()

    assert finalized["records"][0]["component_rewards"][
        EXPECTED_REWARD_NAMES[0]
    ] == 1.0


def test_collector_rejects_duplicate_step_and_wrong_sku():
    trainer, collector = make_trainer()
    first_inputs = [{"sku_id": SKUS[0]} for _ in range(8)]
    trainer._generate_and_score_completions(first_inputs)

    with pytest.raises(RuntimeError, match="capture step drifted"):
        trainer._generate_and_score_completions(first_inputs)
    assert collector.captured_steps == 1

    trainer, collector = make_trainer()
    wrong_inputs = [{"sku_id": SKUS[1]} for _ in range(8)]
    with pytest.raises(RuntimeError, match="input SKU drifted at step 1"):
        trainer._generate_and_score_completions(wrong_inputs)
    assert collector.captured_steps == 0


def test_collector_preserves_truncation_but_refuses_completed_evidence():
    trainer, collector = make_trainer(truncate_at=(3, 4))
    run_five_fake_steps(trainer)

    truncated = collector.records[2 * 8 + 4]
    assert truncated["step"] == 3
    assert truncated["effective_completion_tokens"] == 0
    assert truncated["truncated_and_masked"] is True
    with pytest.raises(ValueError, match="completion-token count is outside bounds"):
        collector.finalize()


def test_capturing_trainer_requires_collector_and_single_process():
    trainer_class = make_rollout_capturing_trainer_class(FakeGRPOTrainer)
    with pytest.raises(TypeError, match="requires a SmokeRolloutCollector"):
        trainer_class(smoke_rollout_collector=object())

    trainer, collector = make_trainer()
    trainer.accelerator.num_processes = 2
    with pytest.raises(RuntimeError, match="exactly one process"):
        trainer._generate_and_score_completions(
            [{"sku_id": SKUS[0]} for _ in range(8)]
        )
    assert collector.captured_steps == 0
