"""CPU-only tests for synchronized GRPO phase timing."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from training.grpo_phase_profiler import (
    FullRunPhaseProfiler,
    make_phase_profiler_callback_class,
    validate_phase_profile_summary,
    validate_phase_timing_report,
)


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeAccelerator:
    def __init__(self, clock):
        self.clock = clock

    def backward(self, loss):
        self.clock.advance(1.5)
        return loss


class FakeOptimizer:
    def __init__(self, clock):
        self.clock = clock

    def step(self):
        self.clock.advance(0.25)
        return "stepped"


class FakeTrainer:
    def __init__(self, clock):
        self.clock = clock
        self.accelerator = FakeAccelerator(clock)

    def _generate(self):
        self.clock.advance(2.0)
        return "generated"

    def _calculate_rewards(self):
        self.clock.advance(0.5)
        return "rewarded"

    def compute_loss(self):
        self.clock.advance(1.0)
        return "loss"


class FakeCallback:
    pass


def test_profiler_reconciles_exact_phase_and_other_time():
    clock = ManualClock()
    sync_calls = []
    profiler = FullRunPhaseProfiler(
        expected_steps=1,
        synchronize_fn=lambda: sync_calls.append(clock()),
        clock_fn=clock,
    )
    trainer = FakeTrainer(clock)
    optimizer = FakeOptimizer(clock)
    profiler.instrument_trainer(trainer)
    profiler.instrument_optimizer(optimizer)

    profiler.begin_step(1)
    assert trainer._generate() == "generated"
    assert trainer._calculate_rewards() == "rewarded"
    assert trainer.compute_loss() == "loss"
    assert trainer.accelerator.backward("loss") == "loss"
    assert optimizer.step() == "stepped"
    clock.advance(0.75)
    record = profiler.end_step(1)

    assert record["step_wall_seconds"] == 6.0
    assert record["accounted_seconds"] == 5.25
    assert record["other_seconds"] == 0.75
    assert record["phase_seconds"] == {
        "generation": 2.0,
        "reward": 0.5,
        "forward_loss": 1.0,
        "backward": 1.5,
        "optimizer": 0.25,
    }
    timing_report = profiler.snapshot(expected_steps=1)
    assert validate_phase_timing_report(timing_report, expected_steps=1)[
        "status"
    ] == "passed"
    summary = profiler.finalize(train_seconds=7.0)
    assert summary["phase_total_seconds"] == 5.25
    assert summary["other_within_steps_seconds"] == 0.75
    assert summary["outside_steps_seconds"] == 1.0
    assert summary["unattributed_total_seconds"] == 1.75
    assert summary["measurement"]["synchronization_calls"] == 12
    assert len(sync_calls) == 12
    assert validate_phase_profile_summary(
        summary, expected_steps=1, train_seconds=7.0
    )["status"] == "passed"


def test_callback_attaches_optimizer_and_brackets_step():
    clock = ManualClock()
    profiler = FullRunPhaseProfiler(
        expected_steps=1,
        synchronize_fn=lambda: None,
        clock_fn=clock,
    )
    trainer = FakeTrainer(clock)
    profiler.instrument_trainer(trainer)
    optimizer = FakeOptimizer(clock)
    callback_class = make_phase_profiler_callback_class(FakeCallback)
    callback = callback_class(phase_profiler=profiler)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace()

    assert callback.on_train_begin(None, state, control, optimizer=optimizer) is control
    callback.on_step_begin(None, state, control)
    trainer._generate()
    trainer._calculate_rewards()
    trainer.compute_loss()
    trainer.accelerator.backward(None)
    optimizer.step()
    state.global_step = 1
    callback.on_step_end(None, state, control)

    assert profiler.snapshot(expected_steps=1)["steps"] == 1


def test_profiler_fails_closed_on_missing_or_duplicate_phase_calls():
    clock = ManualClock()
    profiler = FullRunPhaseProfiler(
        expected_steps=1,
        synchronize_fn=lambda: None,
        clock_fn=clock,
    )
    trainer = FakeTrainer(clock)
    profiler.instrument_trainer(trainer)
    profiler.instrument_optimizer(FakeOptimizer(clock))
    profiler.begin_step(1)
    trainer._generate()
    trainer._calculate_rewards()
    trainer.compute_loss()
    trainer.accelerator.backward(None)

    with pytest.raises(RuntimeError, match="call counts drifted"):
        profiler.end_step(1)


def test_timing_report_rejects_nonfinite_or_nonreconciling_values():
    clock = ManualClock()
    profiler = FullRunPhaseProfiler(
        expected_steps=1,
        synchronize_fn=lambda: None,
        clock_fn=clock,
    )
    trainer = FakeTrainer(clock)
    optimizer = FakeOptimizer(clock)
    profiler.instrument_trainer(trainer)
    profiler.instrument_optimizer(optimizer)
    profiler.begin_step(1)
    trainer._generate()
    trainer._calculate_rewards()
    trainer.compute_loss()
    trainer.accelerator.backward(None)
    optimizer.step()
    profiler.end_step(1)
    report = profiler.snapshot(expected_steps=1)
    report["records"][0]["phase_seconds"]["generation"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        validate_phase_timing_report(report, expected_steps=1)


def test_profile_summary_rejects_a_residual_that_disagrees_with_step_records():
    clock = ManualClock()
    profiler = FullRunPhaseProfiler(
        expected_steps=1,
        synchronize_fn=lambda: None,
        clock_fn=clock,
    )
    trainer = FakeTrainer(clock)
    optimizer = FakeOptimizer(clock)
    profiler.instrument_trainer(trainer)
    profiler.instrument_optimizer(optimizer)
    profiler.begin_step(1)
    trainer._generate()
    trainer._calculate_rewards()
    trainer.compute_loss()
    trainer.accelerator.backward(None)
    optimizer.step()
    clock.advance(0.75)
    profiler.end_step(1)
    summary = profiler.finalize(train_seconds=7.0)
    tampered = copy.deepcopy(summary)
    tampered["other_within_steps_seconds"] += 0.25

    with pytest.raises(ValueError, match="within-step residual disagrees"):
        validate_phase_profile_summary(
            tampered, expected_steps=1, train_seconds=7.0
        )
