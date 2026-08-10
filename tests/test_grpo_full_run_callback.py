"""CPU-only tests for the full-run Trainer callback handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_full_run_artifacts import (
    FullRunCheckpointLifecycleWriter,
    build_full_run_lifecycle_plan,
    create_full_run_staging_output,
)
from training.grpo_full_run_evidence import expected_full_run_learning_rates
from training.grpo_smoke_artifacts import EXPECTED_REWARD_NAMES
from training.train_grpo import (
    FullRunCheckpointHandoff,
    FullRunRolloutCollector,
    make_full_run_checkpoint_callback_class,
    make_full_run_rollout_capturing_trainer_class,
)

STARTING_BYTES = b"locked starting SFT adapter"
STARTING_SHA = hashlib.sha256(STARTING_BYTES).hexdigest()


class FakeTrainerCallback:
    def __init__(self):
        self.base_callback_initialized = True


class FakeGRPOTrainer:
    def __init__(self):
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_func_names = list(EXPECTED_REWARD_NAMES)
        self.reward_weights = [1.0, 1.0, 2.0]
        self._logs = {}

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
                EXPECTED_REWARD_NAMES[0]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[1]: [1.0] * 8,
                EXPECTED_REWARD_NAMES[2]: agreement,
            },
        }
        masks = [[1] * (80 + index) for index in range(8)]
        if step == 3:
            masks[4] = [0] * 170
        return {"completion_mask": masks}


def make_writer(tmp_path):
    final = tmp_path / "grpo-first-300"
    staging = create_full_run_staging_output(final)
    plan = build_full_run_lifecycle_plan(
        final_output_dir=final,
        staging_dir=staging,
    )
    writer = FullRunCheckpointLifecycleWriter(
        plan=plan,
        starting_adapter_sha256=STARTING_SHA,
    )
    return writer, plan


def make_checkpoint(plan: dict, step: int) -> Path:
    checkpoint = Path(plan["checkpoint_paths"][str(step)])
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(
        f"GRPO adapter step {step}".encode()
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"r": 16, "step": step}), encoding="utf-8"
    )
    (checkpoint / "training_args.bin").write_bytes(b"metadata")
    return checkpoint


def trainer_args(plan: dict):
    return SimpleNamespace(
        output_dir=plan["trainer_output_dir"],
        max_steps=300,
        save_steps=100,
        save_total_limit=2,
        save_only_model=True,
    )


def step_log(step: int) -> dict:
    learning_rate = expected_full_run_learning_rates()[step - 1]
    return {
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


def make_runtime(tmp_path, *, progress_callback=None):
    writer, plan = make_writer(tmp_path)
    collector = FullRunRolloutCollector()
    trainer_class = make_full_run_rollout_capturing_trainer_class(FakeGRPOTrainer)
    trainer = trainer_class(full_run_rollout_collector=collector)
    callback_class = make_full_run_checkpoint_callback_class(FakeTrainerCallback)
    callback = callback_class(
        lifecycle_writer=writer,
        rollout_collector=collector,
        progress_callback=progress_callback,
    )
    return writer, plan, trainer, callback


def run_generated_steps(trainer, end_step: int) -> None:
    start = trainer.state.global_step + 1
    for step in range(start, end_step + 1):
        trainer._generate_and_score_completions(
            [{"sku_id": f"sku-{step}"} for _ in range(8)]
        )
        trainer.state.global_step += 1
        trainer.state.log_history.append(step_log(step))


def test_callback_hands_validated_step_100_and_final_evidence_to_writer(tmp_path):
    writer, plan, trainer, callback = make_runtime(tmp_path)
    args = trainer_args(plan)
    control = SimpleNamespace(marker="same-object")

    assert callback.on_train_begin(args, trainer.state, control) is control
    assert callback.base_callback_initialized

    run_generated_steps(trainer, 100)
    make_checkpoint(plan, 100)
    assert callback.on_save(args, trainer.state, control) is control
    assert writer.snapshot()["event_count"] == 2
    milestone = Path(plan["step_100_export"]["directory"])
    assert len((milestone / "rollouts.jsonl").read_text().splitlines()) == 800
    assert len(json.loads((milestone / "trainer-log.json").read_text())) == 100

    run_generated_steps(trainer, 200)
    make_checkpoint(plan, 200)
    callback.on_save(args, trainer.state, control)
    assert writer.snapshot()["event_count"] == 3

    run_generated_steps(trainer, 300)
    make_checkpoint(plan, 300)
    # Transformers rotates checkpoint 100 before firing on_save at step 300.
    shutil.rmtree(Path(plan["checkpoint_paths"]["100"]))
    callback.on_save(args, trainer.state, control)
    assert writer.snapshot()["status"] == "checkpoints_ready_for_final_handoff"

    assert callback.on_train_end(args, trainer.state, control) is control
    evidence = callback.final_evidence()
    assert len(evidence["rollout_records"]) == 2_400
    assert len(evidence["trainer_step_logs"]) == 300
    assert evidence["rollout_validation"]["zero_variance_steps"] == 1
    assert evidence["rollout_validation"]["truncated_and_masked_records"] == 1
    assert evidence["trainer_log_validation"]["zero_gradient_steps"] == [2]
    assert evidence["trainer_log_validation"]["clipped_steps"] == [3]
    assert set(evidence["checkpoint_evidence"]) == {100, 200, 300}
    assert evidence["lifecycle"]["event_count"] == 6

    evidence["rollout_records"][0]["raw_output"] = "caller mutation"
    assert callback.final_evidence()["rollout_records"][0]["raw_output"] == (
        "step-1-completion-0"
    )


def test_callback_validates_evidence_before_recording_checkpoint(tmp_path):
    writer, plan, trainer, callback = make_runtime(tmp_path)
    args = trainer_args(plan)
    control = SimpleNamespace()
    callback.on_train_begin(args, trainer.state, control)
    run_generated_steps(trainer, 100)
    trainer.state.log_history[50]["learning_rate"] = 9e-6
    make_checkpoint(plan, 100)

    with pytest.raises(ValueError, match="learning-rate schedule drifted"):
        callback.on_save(args, trainer.state, control)
    assert writer.events == []
    assert not Path(plan["step_100_export"]["directory"]).exists()


def test_callback_forwards_only_validated_consecutive_progress(tmp_path):
    progress = []
    writer, plan, trainer, callback = make_runtime(
        tmp_path,
        progress_callback=lambda **values: progress.append(values),
    )
    args = trainer_args(plan)
    control = SimpleNamespace()
    callback.on_train_begin(args, trainer.state, control)

    run_generated_steps(trainer, 1)
    assert callback.on_log(args, trainer.state, control) is control
    assert progress == [
        {"optimizer_step": 1, "rollout_records": 8, "scalar_logs": 1}
    ]

    with pytest.raises(RuntimeError, match="expected step 2, found 1"):
        callback.on_log(args, trainer.state, control)
    assert len(progress) == 1


def test_callback_refuses_progress_before_matching_rollout_group(tmp_path):
    progress = []
    writer, plan, trainer, callback = make_runtime(
        tmp_path,
        progress_callback=lambda **values: progress.append(values),
    )
    callback.on_train_begin(trainer_args(plan), trainer.state, SimpleNamespace())
    trainer.state.global_step = 1
    trainer.state.log_history.append(step_log(1))

    with pytest.raises(RuntimeError, match="captured rollout groups"):
        callback.on_log(
            trainer_args(plan),
            trainer.state,
            SimpleNamespace(),
        )
    assert progress == []


def test_callback_rejects_checkpoint_when_rollout_prefix_is_incomplete(tmp_path):
    writer, plan, trainer, callback = make_runtime(tmp_path)
    args = trainer_args(plan)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 100
    trainer.state.log_history = [step_log(step) for step in range(1, 101)]
    make_checkpoint(plan, 100)

    with pytest.raises(RuntimeError, match="snapshot step drifted"):
        callback.on_save(args, trainer.state, SimpleNamespace())
    assert writer.events == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 301),
        ("save_steps", 50),
        ("save_total_limit", 3),
        ("save_only_model", False),
        ("output_dir", "/tmp/wrong-trainer-output"),
    ],
)
def test_callback_refuses_trainer_argument_drift(tmp_path, field, value):
    writer, plan, trainer, callback = make_runtime(tmp_path)
    args = trainer_args(plan)
    setattr(args, field, value)

    with pytest.raises(RuntimeError, match="arguments drifted|output directory drifted"):
        callback.on_train_begin(args, trainer.state, SimpleNamespace())
    assert writer.events == []


def test_callback_rejects_unexpected_save_order_and_early_train_end(tmp_path):
    writer, plan, trainer, callback = make_runtime(tmp_path)
    args = trainer_args(plan)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 200
    with pytest.raises(RuntimeError, match="expected step 100, found 200"):
        callback.on_save(args, trainer.state, SimpleNamespace())
    with pytest.raises(RuntimeError, match="did not end at step 300"):
        callback.on_train_end(args, trainer.state, SimpleNamespace())
    assert writer.events == []


def test_handoff_and_callback_factory_require_exact_dependencies(tmp_path):
    writer, _plan = make_writer(tmp_path)
    collector = FullRunRolloutCollector()
    with pytest.raises(TypeError, match="LifecycleWriter"):
        FullRunCheckpointHandoff(
            lifecycle_writer=object(), rollout_collector=collector
        )
    with pytest.raises(TypeError, match="FullRunRolloutCollector"):
        FullRunCheckpointHandoff(
            lifecycle_writer=writer, rollout_collector=object()
        )
    with pytest.raises(ValueError, match="exactly 300 steps"):
        FullRunCheckpointHandoff(
            lifecycle_writer=writer,
            rollout_collector=FullRunRolloutCollector(expected_steps=100),
        )
    with pytest.raises(TypeError, match="base callback must be a class"):
        make_full_run_checkpoint_callback_class(object())
