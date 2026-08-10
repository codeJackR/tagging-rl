"""Lightweight synchronized phase timing for the auditable GRPO full run."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

PHASE_PROFILER_VERSION = "grpo-phase-profiler-v1"
PHASE_TIMING_REPORT_VERSION = "grpo-phase-timings-v1"
PROFILE_PHASES = (
    "generation",
    "reward",
    "forward_loss",
    "backward",
    "optimizer",
)
PHASE_METHODS = {
    "generation": "_generate",
    "reward": "_calculate_rewards",
    "forward_loss": "compute_loss",
}


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_phase_timing_report(report: object, *, expected_steps: int) -> dict:
    """Validate one durable prefix of synchronized per-step phase timings."""
    if not isinstance(expected_steps, int) or expected_steps <= 0:
        raise ValueError("expected profiling steps must be a positive integer")
    if not isinstance(report, dict):
        raise TypeError("phase timing report must be a dictionary")
    if report.get("version") != PHASE_TIMING_REPORT_VERSION:
        raise ValueError("unexpected phase timing report version")
    if report.get("status") != "completed_prefix":
        raise ValueError("phase timing report is not a completed prefix")
    if report.get("synchronized_boundaries") is not True:
        raise ValueError("phase timing report did not synchronize boundaries")
    if report.get("clock") != "time.perf_counter":
        raise ValueError("phase timing report clock drifted")
    if report.get("steps") != expected_steps:
        raise ValueError("phase timing report step count drifted")
    if report.get("phases") != list(PROFILE_PHASES):
        raise ValueError("phase timing report phase inventory drifted")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != expected_steps:
        raise ValueError("phase timing report records are incomplete")

    phase_totals = {phase: 0.0 for phase in PROFILE_PHASES}
    step_wall_total = 0.0
    within_step_other_total = 0.0
    for expected_step, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("step") != expected_step:
            raise ValueError("phase timing report step order drifted")
        phase_seconds = record.get("phase_seconds")
        phase_calls = record.get("phase_calls")
        if not isinstance(phase_seconds, dict) or set(phase_seconds) != set(
            PROFILE_PHASES
        ):
            raise ValueError(f"phase timing step {expected_step} phases drifted")
        if not isinstance(phase_calls, dict) or phase_calls != {
            phase: 1 for phase in PROFILE_PHASES
        }:
            raise ValueError(f"phase timing step {expected_step} call counts drifted")
        if not all(_finite_nonnegative(value) for value in phase_seconds.values()):
            raise ValueError(f"phase timing step {expected_step} is non-finite")
        step_wall = record.get("step_wall_seconds")
        accounted = record.get("accounted_seconds")
        other = record.get("other_seconds")
        if not all(_finite_nonnegative(value) for value in (step_wall, accounted, other)):
            raise ValueError(f"phase timing step {expected_step} totals are invalid")
        calculated = sum(float(phase_seconds[phase]) for phase in PROFILE_PHASES)
        tolerance = max(1e-9, float(step_wall) * 1e-9)
        if abs(float(accounted) - calculated) > tolerance:
            raise ValueError(f"phase timing step {expected_step} accounted total drifted")
        if abs(float(step_wall) - float(accounted) - float(other)) > tolerance:
            raise ValueError(f"phase timing step {expected_step} does not reconcile")
        for phase in PROFILE_PHASES:
            phase_totals[phase] += float(phase_seconds[phase])
        step_wall_total += float(step_wall)
        within_step_other_total += float(other)

    return {
        "version": PHASE_TIMING_REPORT_VERSION,
        "status": "passed",
        "steps": expected_steps,
        "records": expected_steps,
        "phase_totals_seconds": phase_totals,
        "step_wall_total_seconds": step_wall_total,
        "within_step_other_total_seconds": within_step_other_total,
        "all_values_finite": True,
    }


def validate_phase_profile_summary(
    summary: object, *, expected_steps: int, train_seconds: float
) -> dict:
    """Validate aggregate phase statistics against authoritative train time."""
    if not isinstance(summary, dict):
        raise TypeError("phase profile summary must be a dictionary")
    if summary.get("version") != PHASE_PROFILER_VERSION:
        raise ValueError("unexpected phase profile summary version")
    if summary.get("status") != "completed" or summary.get("steps") != expected_steps:
        raise ValueError("phase profile summary is incomplete")
    if not _finite_nonnegative(train_seconds) or float(train_seconds) <= 0:
        raise ValueError("authoritative training time must be positive")
    tolerance = max(1e-9, float(train_seconds) * 1e-9)
    if abs(float(summary.get("train_seconds", -1)) - float(train_seconds)) > tolerance:
        raise ValueError("phase profile training time drifted")
    measurement = summary.get("measurement")
    if not isinstance(measurement, dict) or measurement.get(
        "cuda_synchronized_boundaries"
    ) is not True:
        raise ValueError("phase profile boundaries were not synchronized")
    if measurement.get("clock") != "time.perf_counter":
        raise ValueError("phase profile clock drifted")
    if not isinstance(measurement.get("observer_effect"), str) or not measurement[
        "observer_effect"
    ].strip():
        raise ValueError("phase profile observer effect is missing")
    expected_sync_calls = expected_steps * (2 * len(PROFILE_PHASES) + 2)
    if measurement.get("synchronization_calls") != expected_sync_calls:
        raise ValueError("phase profile synchronization count drifted")
    statistics = summary.get("phase_statistics")
    if not isinstance(statistics, dict) or set(statistics) != set(PROFILE_PHASES):
        raise ValueError("phase profile statistics inventory drifted")
    timing_validation = summary.get("timing_report_validation")
    if not isinstance(timing_validation, dict) or (
        timing_validation.get("status") != "passed"
        or timing_validation.get("steps") != expected_steps
    ):
        raise ValueError("phase profile timing validation is missing")
    timing_phase_totals = timing_validation.get("phase_totals_seconds")
    if not isinstance(timing_phase_totals, dict) or set(timing_phase_totals) != set(
        PROFILE_PHASES
    ):
        raise ValueError("phase profile timing totals inventory drifted")
    phase_total = 0.0
    for phase, values in statistics.items():
        if not isinstance(values, dict) or values.get("calls") != expected_steps:
            raise ValueError(f"phase profile {phase} call count drifted")
        numeric_fields = (
            "total_seconds",
            "mean_seconds",
            "min_seconds",
            "p50_seconds",
            "p95_seconds",
            "max_seconds",
            "percent_of_train",
        )
        if not all(_finite_nonnegative(values.get(field)) for field in numeric_fields):
            raise ValueError(f"phase profile {phase} contains invalid timing")
        if not _finite_nonnegative(timing_phase_totals.get(phase)) or abs(
            float(values["total_seconds"]) - float(timing_phase_totals[phase])
        ) > tolerance:
            raise ValueError(f"phase profile {phase} disagrees with step records")
        phase_total += float(values["total_seconds"])
    if abs(float(summary.get("phase_total_seconds", -1)) - phase_total) > tolerance:
        raise ValueError("phase profile aggregate phase total drifted")
    step_wall = summary.get("step_wall_total_seconds")
    within = summary.get("other_within_steps_seconds")
    outside = summary.get("outside_steps_seconds")
    unattributed = summary.get("unattributed_total_seconds")
    if not all(_finite_nonnegative(value) for value in (step_wall, within, outside, unattributed)):
        raise ValueError("phase profile residual timing is invalid")
    timing_step_wall = timing_validation.get("step_wall_total_seconds")
    timing_within = timing_validation.get("within_step_other_total_seconds")
    if not all(_finite_nonnegative(value) for value in (timing_step_wall, timing_within)):
        raise ValueError("phase profile timing validation totals are invalid")
    if abs(float(step_wall) - float(timing_step_wall)) > tolerance:
        raise ValueError("phase profile step wall disagrees with step records")
    if abs(float(within) - float(timing_within)) > tolerance:
        raise ValueError("phase profile within-step residual disagrees with step records")
    if abs(phase_total + float(within) - float(step_wall)) > tolerance:
        raise ValueError("phase profile within-step timing does not reconcile")
    if abs(float(step_wall) + float(outside) - float(train_seconds)) > tolerance:
        raise ValueError("phase profile step/outside timing does not reconcile")
    if abs(phase_total + float(unattributed) - float(train_seconds)) > tolerance:
        raise ValueError("phase profile attributed timing does not reconcile")
    if abs(float(within) + float(outside) - float(unattributed)) > tolerance:
        raise ValueError("phase profile residual buckets do not reconcile")
    accounted_percent = summary.get("accounted_percent")
    if not _finite_nonnegative(accounted_percent) or abs(
        float(accounted_percent) - phase_total / float(train_seconds) * 100.0
    ) > max(1e-9, tolerance * 100.0):
        raise ValueError("phase profile accounted percentage drifted")
    return {
        "version": PHASE_PROFILER_VERSION,
        "status": "passed",
        "steps": expected_steps,
        "train_seconds": float(train_seconds),
        "phase_total_seconds": phase_total,
        "unattributed_total_seconds": float(unattributed),
        "synchronization_calls": expected_sync_calls,
        "all_values_finite": True,
    }


class FullRunPhaseProfiler:
    """Measure synchronized wall time at explicit GRPO execution boundaries."""

    def __init__(
        self,
        *,
        expected_steps: int,
        synchronize_fn: Callable[[], object],
        clock_fn: Callable[[], float] = time.perf_counter,
    ):
        if not isinstance(expected_steps, int) or expected_steps <= 0:
            raise ValueError("phase profiler expected_steps must be positive")
        if not callable(synchronize_fn) or not callable(clock_fn):
            raise TypeError("phase profiler requires callable sync and clock functions")
        self.expected_steps = expected_steps
        self.synchronize_fn = synchronize_fn
        self.clock_fn = clock_fn
        self._active_step: int | None = None
        self._step_started_at: float | None = None
        self._active_phase_seconds: dict[str, float] = {}
        self._active_phase_calls: dict[str, int] = {}
        self._records: list[dict] = []
        self._trainer_instrumented = False
        self._optimizer_instrumented = False
        self._synchronization_calls = 0

    def _synchronize(self) -> None:
        self.synchronize_fn()
        self._synchronization_calls += 1

    def begin_step(self, step: int) -> None:
        expected = len(self._records) + 1
        if self._active_step is not None or step != expected:
            raise RuntimeError(
                f"phase profiler expected step {expected}, found {step}"
            )
        self._synchronize()
        self._active_step = step
        self._step_started_at = float(self.clock_fn())
        self._active_phase_seconds = {phase: 0.0 for phase in PROFILE_PHASES}
        self._active_phase_calls = {phase: 0 for phase in PROFILE_PHASES}

    def _measure(self, phase: str, function: Callable, *args, **kwargs):
        if phase not in PROFILE_PHASES:
            raise ValueError(f"unknown profiling phase: {phase}")
        if self._active_step is None:
            raise RuntimeError(f"profiling phase {phase} occurred outside a step")
        self._synchronize()
        started = float(self.clock_fn())
        try:
            return function(*args, **kwargs)
        finally:
            self._synchronize()
            duration = float(self.clock_fn()) - started
            if not _finite_nonnegative(duration):
                raise RuntimeError(f"profiling phase {phase} had invalid duration")
            self._active_phase_seconds[phase] += duration
            self._active_phase_calls[phase] += 1

    def _wrap_callable(self, owner: object, name: str, phase: str) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            raise TypeError(f"profiling target {name} is not callable")

        def measured(*args, **kwargs):
            return self._measure(phase, original, *args, **kwargs)

        setattr(owner, name, measured)

    def instrument_trainer(self, trainer: object) -> None:
        if self._trainer_instrumented:
            raise RuntimeError("phase profiler instrumented trainer more than once")
        for phase, method_name in PHASE_METHODS.items():
            self._wrap_callable(trainer, method_name, phase)
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is None:
            raise TypeError("phase profiler trainer has no accelerator")
        self._wrap_callable(accelerator, "backward", "backward")
        self._trainer_instrumented = True

    def instrument_optimizer(self, optimizer: object) -> None:
        if not self._trainer_instrumented:
            raise RuntimeError("phase profiler must instrument trainer first")
        if self._optimizer_instrumented:
            raise RuntimeError("phase profiler instrumented optimizer more than once")
        self._wrap_callable(optimizer, "step", "optimizer")
        self._optimizer_instrumented = True

    def end_step(self, step: int) -> dict:
        if self._active_step != step or self._step_started_at is None:
            raise RuntimeError(f"phase profiler cannot end unexpected step {step}")
        self._synchronize()
        step_wall = float(self.clock_fn()) - self._step_started_at
        if not _finite_nonnegative(step_wall):
            raise RuntimeError("phase profiler step wall time is invalid")
        if self._active_phase_calls != {phase: 1 for phase in PROFILE_PHASES}:
            raise RuntimeError(
                f"phase profiler step {step} call counts drifted: "
                f"{self._active_phase_calls}"
            )
        accounted = sum(self._active_phase_seconds.values())
        tolerance = max(1e-9, step_wall * 1e-9)
        if accounted > step_wall + tolerance:
            raise RuntimeError("phase profiler accounted time exceeds step wall time")
        record = {
            "step": step,
            "step_wall_seconds": step_wall,
            "phase_seconds": dict(self._active_phase_seconds),
            "phase_calls": dict(self._active_phase_calls),
            "accounted_seconds": accounted,
            "other_seconds": max(0.0, step_wall - accounted),
        }
        self._records.append(record)
        self._active_step = None
        self._step_started_at = None
        self._active_phase_seconds = {}
        self._active_phase_calls = {}
        return copy.deepcopy(record)

    def snapshot(self, *, expected_steps: int) -> dict:
        if self._active_step is not None:
            raise RuntimeError("cannot snapshot phase timings during an active step")
        report = {
            "version": PHASE_TIMING_REPORT_VERSION,
            "status": "completed_prefix",
            "steps": len(self._records),
            "phases": list(PROFILE_PHASES),
            "clock": "time.perf_counter",
            "synchronized_boundaries": True,
            "records": copy.deepcopy(self._records),
        }
        validate_phase_timing_report(report, expected_steps=expected_steps)
        return report

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        ordered = sorted(float(value) for value in values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    def finalize(self, *, train_seconds: float) -> dict:
        if not _finite_nonnegative(train_seconds) or float(train_seconds) <= 0:
            raise ValueError("phase profiler requires positive total training time")
        timing_report = self.snapshot(expected_steps=self.expected_steps)
        validation = validate_phase_timing_report(
            timing_report, expected_steps=self.expected_steps
        )
        records = timing_report["records"]
        phase_statistics = {}
        for phase in PROFILE_PHASES:
            values = [record["phase_seconds"][phase] for record in records]
            total = sum(values)
            phase_statistics[phase] = {
                "calls": len(values),
                "total_seconds": total,
                "mean_seconds": total / len(values),
                "min_seconds": min(values),
                "p50_seconds": self._percentile(values, 0.50),
                "p95_seconds": self._percentile(values, 0.95),
                "max_seconds": max(values),
                "percent_of_train": total / float(train_seconds) * 100.0,
            }
        phase_total = sum(
            statistics["total_seconds"] for statistics in phase_statistics.values()
        )
        step_wall_total = validation["step_wall_total_seconds"]
        tolerance = max(1e-9, float(train_seconds) * 1e-9)
        if step_wall_total > float(train_seconds) + tolerance:
            raise RuntimeError("profiled step wall time exceeds total training time")
        report = {
            "version": PHASE_PROFILER_VERSION,
            "status": "completed",
            "measurement": {
                "clock": "time.perf_counter",
                "cuda_synchronized_boundaries": True,
                "synchronization_calls": self._synchronization_calls,
                "observer_effect": (
                    "timings describe the synchronized instrumented run; "
                    "synchronization may reduce asynchronous overlap"
                ),
            },
            "phase_definitions": {
                "generation": "trainer._generate",
                "reward": "trainer._calculate_rewards",
                "forward_loss": "trainer.compute_loss",
                "backward": "trainer.accelerator.backward",
                "optimizer": "trainer optimizer.step",
                "other_within_steps": (
                    "step wall time outside the five measured call boundaries"
                ),
                "outside_steps": (
                    "trainer startup, checkpointing, logging, and shutdown"
                ),
            },
            "steps": self.expected_steps,
            "train_seconds": float(train_seconds),
            "phase_statistics": phase_statistics,
            "phase_total_seconds": phase_total,
            "step_wall_total_seconds": step_wall_total,
            "other_within_steps_seconds": validation[
                "within_step_other_total_seconds"
            ],
            "outside_steps_seconds": max(0.0, float(train_seconds) - step_wall_total),
            "unattributed_total_seconds": max(0.0, float(train_seconds) - phase_total),
            "accounted_percent": phase_total / float(train_seconds) * 100.0,
            "timing_report_validation": validation,
        }
        return report


def make_phase_profiler_callback_class(base_callback_class: type) -> type:
    """Create a Transformers callback that brackets steps and wraps optimizer."""
    if not isinstance(base_callback_class, type):
        raise TypeError("base callback must be a class")

    class FullRunPhaseProfilerCallback(base_callback_class):
        def __init__(self, *, phase_profiler: FullRunPhaseProfiler):
            super().__init__()
            if not isinstance(phase_profiler, FullRunPhaseProfiler):
                raise TypeError("profiling callback requires FullRunPhaseProfiler")
            self.phase_profiler = phase_profiler

        def on_train_begin(self, args, state, control, **kwargs):
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                raise RuntimeError("profiling callback received no optimizer")
            self.phase_profiler.instrument_optimizer(optimizer)
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            self.phase_profiler.begin_step(int(state.global_step) + 1)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.phase_profiler.end_step(int(state.global_step))
            return control

    FullRunPhaseProfilerCallback.__name__ = (
        f"FullRunPhaseProfiler{base_callback_class.__name__}"
    )
    return FullRunPhaseProfilerCallback
