"""CPU-only proof of the Arm A workload: train, validate, publish.

The trainer is a fake that produces evidence on demand, so every failure path
is reachable without a GPU. The tests that matter are the ones where training
*succeeds* and publication must still be refused: a run that trained perfectly
but recorded nothing is not a usable arm, and Run 1 produced exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_arm_workload import (
    VERSION,
    WorkloadError,
    publish_atomically,
    run_arm,
    validate_training_result,
)
from tests.test_run2_arm_runtime_composition import (  # reuse the proven fakes
    FakeCallbackBase,
    FakeConfig,
    FakeTrainer,
    fake_command_builder,
    forbidden_runner,
    plenty_of_gpu,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "runs" / "grpo-run2-causal-experiment-contract.json").read_text()
)
MONITOR_CONTRACT = ROOT / "runs" / "grpo-run2-checkpoint-monitor-contract.json"


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def good_result(steps=1, *, checkpoints=()):
    return {
        "global_step": steps,
        "training_loss": 0.5,
        "log_history": [{"step": i} for i in range(1, steps + 1)],
        "phase_records": [
            {"step": i, "phase_calls": {"generation": 1, "reward": 1}}
            for i in range(1, steps + 1)
        ],
        "rollouts": [
            {"step": i, "index": j, "reward": float(j)}
            for i in range(1, steps + 1)
            for j in range(8)
        ],
        "checkpoints": list(checkpoints),
    }


# --- validation: training succeeded, evidence did not -------------------------


def test_complete_evidence_validates():
    summary = validate_training_result(good_result(2), steps=2, smoke=True)
    assert summary["steps"] == 2
    assert summary["rollouts"] == 16
    assert summary["phase_records"] == 2


def test_short_run_fails_closed():
    result = good_result(2) | {"global_step": 1}
    with pytest.raises(WorkloadError, match="stopped at step 1"):
        validate_training_result(result, steps=2, smoke=True)


def test_instrumented_but_never_fired_profiler_fails_closed():
    """Composition proves the phase methods are wrapped. Only a real run can
    prove the wrappers fired; zero phase calls mean they did not."""
    result = good_result(1)
    result["phase_records"] = [{"step": 1, "phase_calls": {}}]
    with pytest.raises(WorkloadError, match="no measured phases"):
        validate_training_result(result, steps=1, smoke=True)


def test_missing_rollouts_fail_closed():
    result = good_result(2)
    result["rollouts"] = result["rollouts"][:-1]
    with pytest.raises(WorkloadError, match="expected 16"):
        validate_training_result(result, steps=2, smoke=True)


def test_non_finite_reward_fails_closed():
    result = good_result(1)
    result["rollouts"][0]["reward"] = float("nan")
    with pytest.raises(WorkloadError, match="finite reward"):
        validate_training_result(result, steps=1, smoke=True)


def test_non_finite_loss_fails_closed():
    with pytest.raises(WorkloadError, match="not a finite number"):
        validate_training_result(
            good_result(1) | {"training_loss": float("nan")}, steps=1, smoke=True
        )


def test_production_run_requires_the_three_checkpoints():
    result = good_result(300, checkpoints=(100, 200))
    with pytest.raises(WorkloadError, match=r"do not match \[100, 200, 300\]"):
        validate_training_result(result, steps=300, smoke=False)


def test_smoke_must_not_checkpoint():
    with pytest.raises(WorkloadError, match="must not checkpoint"):
        validate_training_result(good_result(1, checkpoints=(1,)), steps=1, smoke=True)


# --- publication --------------------------------------------------------------


def test_publish_refuses_to_overwrite(tmp_path):
    staging, final = tmp_path / "staging", tmp_path / "final"
    staging.mkdir()
    final.mkdir()
    with pytest.raises(WorkloadError, match="refusing to overwrite"):
        publish_atomically(staging, final)


def test_publish_moves_the_tree(tmp_path):
    staging, final = tmp_path / "staging", tmp_path / "final"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}")
    assert publish_atomically(staging, final)["published"] is True
    assert (final / "manifest.json").exists() and not staging.exists()


# --- end to end with a fake trainer ------------------------------------------


class TrainingFakeTrainer(FakeTrainer):
    """A trainer that actually 'runs' and reports evidence."""

    steps = 1
    phases_fire = True
    n_rollouts = None

    def train(self):
        # Drive the registered callbacks the way Transformers does, so the
        # profiler is exercised through the real path rather than bypassed.
        control = object()
        optimizer = type("Opt", (), {"step": lambda self: None})()
        # The profiler instruments the optimizer from on_train_begin and then
        # requires one optimizer call per step, so the fake must supply both.
        for callback in self.callback_handler.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(self.args, self.state, control, optimizer=optimizer)
        for step in range(1, self.steps + 1):
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, "on_step_begin"):
                    callback.on_step_begin(self.args, self.state, control)
            if self.phases_fire:
                self._generate()
                self._calculate_rewards()
                self.compute_loss()
                self.accelerator.backward()
                optimizer.step()
            self.state.global_step = step
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, "on_step_end"):
                    callback.on_step_end(self.args, self.state, control)
        self.state.global_step = self.steps
        self.state.log_history = [{"step": i} for i in range(1, self.steps + 1)]
        n = self.n_rollouts if self.n_rollouts is not None else self.steps * 8
        self.collected_rollouts = [{"i": i, "reward": 1.0} for i in range(n)]
        self.saved_checkpoints = []
        return type("Out", (), {"training_loss": 0.42})()


def _run(pack, tmp_path, trainer_class=TrainingFakeTrainer, **kw):
    return run_arm(
        root=ROOT, arm="A", contract=CONTRACT, pack=pack,
        model_loader=lambda: (object(), object()),
        config_class=FakeConfig, trainer_class=trainer_class,
        callback_base_class=FakeCallbackBase, synchronize_fn=lambda: None,
        monitor_contract_path=MONITOR_CONTRACT,
        monitor_command_builder=fake_command_builder,
        monitor_runner=forbidden_runner, gpu_free_bytes_fn=plenty_of_gpu,
        scratch_root=tmp_path / "scratch", smoke_max_steps=1, **kw,
    )


def test_smoke_run_completes_and_stays_in_scratch(pack, tmp_path):
    manifest = _run(pack, tmp_path)
    assert manifest["version"] == VERSION
    assert manifest["status"] == "arm_smoke_completed"
    assert manifest["smoke_mode"] is True
    assert manifest["published_to"] is None
    assert manifest["evidence"]["rollouts"] == 8
    assert manifest["composition"]["status"] == "arm_runtime_smoke_composed"
    # a smoke must never claim the reserved production path
    assert not (ROOT / CONTRACT["arms"]["A"]["output_dir"]).exists()
    assert Path(manifest["manifest_path"]).exists()


def test_a_training_exception_is_reported_not_swallowed(pack, tmp_path):
    class ExplodingTrainer(TrainingFakeTrainer):
        def train(self):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(WorkloadError, match="training raised RuntimeError"):
        _run(pack, tmp_path, trainer_class=ExplodingTrainer)


def test_a_successful_run_with_dead_instrumentation_is_not_published(pack, tmp_path):
    class SilentProfilerTrainer(TrainingFakeTrainer):
        phases_fire = False

    # The profiler's own validator catches this first: a step whose wrapped
    # methods never ran has zero phase calls, which it rejects before our
    # phase-record check is reached. Either way nothing is published.
    with pytest.raises(WorkloadError, match="call counts drifted"):
        _run(pack, tmp_path, trainer_class=SilentProfilerTrainer)
    assert not (ROOT / CONTRACT["arms"]["A"]["output_dir"]).exists()


def test_a_successful_run_with_missing_rollouts_is_not_published(pack, tmp_path):
    class ForgetfulTrainer(TrainingFakeTrainer):
        n_rollouts = 3

    with pytest.raises(WorkloadError, match="expected 8"):
        _run(pack, tmp_path, trainer_class=ForgetfulTrainer)
