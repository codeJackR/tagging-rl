#!/usr/bin/env python3
"""Run one Run 2 arm end to end, or prove it can be run without a GPU.

This is the only module in the Run 2 stack that calls `trainer.train()`. Every
GPU-bearing collaborator is still injected, so the whole control flow, including
the failure paths and the publication decision, is exercisable on CPU with
fakes. Production injects the real Unsloth loader, `GRPOConfig`, `GRPOTrainer`,
`TrainerCallback` and `torch.cuda.synchronize`.

The composition itself is not re-implemented here. `compose_arm_runtime` has
been through four adversarial reviews and a real-stack construction smoke; this
module's job is the part that composition deliberately stops short of:

1. run the training loop;
2. collect the evidence the loop produced;
3. decide whether that evidence is complete;
4. publish atomically, or leave a failure record and no partial bundle.

The ordering of 3 and 4 is the point. Run 1 produced a bundle that looked
complete and was not, and a later attempt trained successfully but died during
publication. Validation therefore happens against an in-memory result before
anything is linked into place, and a staging tree is renamed only once it has
passed.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import write_exclusive_atomic_json
from training.run2_arm_runtime_composition import (
    CompositionError,
    compose_arm_runtime,
)
from training.run2_causal_experiment import (
    CHECKPOINT_STEPS,
    GENERATIONS_PER_STEP,
    TRAINING_STEPS,
)

VERSION = "grpo-run2-arm-workload-v1"
STAGING_PREFIX = ".{name}.staging-"


class WorkloadError(RuntimeError):
    """Raised when an arm run cannot be completed or its evidence is incomplete."""


def _expected_rollouts(steps: int) -> int:
    return steps * GENERATIONS_PER_STEP


def validate_training_result(
    result: Mapping[str, Any],
    *,
    steps: int,
    smoke: bool,
) -> dict[str, Any]:
    """Decide whether a finished run produced a complete, usable record.

    Called on an in-memory result before anything is published. A run that
    trained perfectly but recorded nothing is not a usable arm, and neither is
    one whose phase records stop halfway: both would otherwise publish a bundle
    that reads as complete.
    """
    missing = sorted(
        key
        for key in ("global_step", "training_loss", "log_history", "phase_records")
        if key not in result
    )
    if missing:
        raise WorkloadError(f"training result is missing {missing}")

    observed_steps = int(result["global_step"])
    if observed_steps != steps:
        raise WorkloadError(
            f"training stopped at step {observed_steps}, expected {steps}"
        )

    loss = result["training_loss"]
    if not isinstance(loss, (int, float)) or isinstance(loss, bool) or loss != loss:
        raise WorkloadError(f"training loss is not a finite number: {loss!r}")

    phase_records = result["phase_records"]
    if len(phase_records) != steps:
        raise WorkloadError(
            f"phase profiler recorded {len(phase_records)} steps, expected {steps}"
        )
    # An instrumented-but-never-fired profiler yields records whose phases are
    # all empty. Composition can only prove the methods were wrapped; this is
    # where the wrappers are proven to have run.
    for index, record in enumerate(phase_records, start=1):
        phases = record.get("phases") if isinstance(record, Mapping) else None
        if not phases:
            raise WorkloadError(f"phase record {index} contains no measured phases")

    rollouts = result.get("rollouts", [])
    expected = _expected_rollouts(steps)
    if len(rollouts) != expected:
        raise WorkloadError(
            f"collected {len(rollouts)} rollouts, expected {expected}"
        )
    rewards = [entry.get("reward") for entry in rollouts if isinstance(entry, Mapping)]
    if len(rewards) != len(rollouts) or any(
        value is None or value != value for value in rewards
    ):
        raise WorkloadError("every rollout must carry a finite reward")

    checkpoints = sorted(int(value) for value in result.get("checkpoints", []))
    if smoke:
        if checkpoints:
            raise WorkloadError(f"smoke mode must not checkpoint, found {checkpoints}")
    elif checkpoints != list(CHECKPOINT_STEPS):
        raise WorkloadError(
            f"checkpoints {checkpoints} do not match {list(CHECKPOINT_STEPS)}"
        )

    return {
        "steps": observed_steps,
        "rollouts": len(rollouts),
        "phase_records": len(phase_records),
        "checkpoints": checkpoints,
        "training_loss": float(loss),
        "distinct_rewards": len({round(float(value), 12) for value in rewards}),
    }


def publish_atomically(staging: Path, final: Path) -> dict[str, Any]:
    """Rename a validated staging tree into place, or leave nothing behind.

    A partial bundle at the reserved path is worse than no bundle: the launch
    preflight requires that path to be absent, so a half-written directory
    blocks every future attempt until someone removes it by hand.
    """
    if final.exists():
        raise WorkloadError(f"refusing to overwrite an existing arm output: {final}")
    if not staging.is_dir():
        raise WorkloadError(f"staging tree is absent: {staging}")
    staging.rename(final)
    return {"published": True, "path": str(final)}


def run_arm(
    *,
    root: str | Path,
    arm: str,
    contract: Mapping[str, Any],
    pack: Any,
    model_loader: Callable[[], tuple[Any, Any]],
    config_class: type,
    trainer_class: type,
    callback_base_class: type,
    synchronize_fn: Callable[[], object],
    monitor_contract_path: str | Path,
    monitor_command_builder: Callable[[int, Path, Path], list[str]],
    monitor_runner: Callable[..., Mapping[str, Any]],
    gpu_free_bytes_fn: Callable[[], int],
    scratch_root: str | Path,
    smoke_max_steps: int | None = None,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Compose, train, validate and publish one arm."""
    root = Path(root).resolve()
    scratch_root = Path(scratch_root).resolve()
    spec = contract["arms"][arm]
    smoke = smoke_max_steps is not None
    steps = int(smoke_max_steps) if smoke else TRAINING_STEPS

    final = (root / spec["output_dir"]).resolve()
    staging = final.parent / (STAGING_PREFIX.format(name=final.name) + str(steps))
    if not smoke and staging.exists():
        raise WorkloadError(f"a previous staging tree survives: {staging}")

    report, trainer = compose_arm_runtime(
        root=root,
        arm=arm,
        contract=contract,
        pack=pack,
        model_loader=model_loader,
        config_class=config_class,
        trainer_class=trainer_class,
        callback_base_class=callback_base_class,
        synchronize_fn=synchronize_fn,
        monitor_contract_path=monitor_contract_path,
        monitor_command_builder=monitor_command_builder,
        monitor_runner=monitor_runner,
        gpu_free_bytes_fn=gpu_free_bytes_fn,
        trainer_scratch_dir=scratch_root / "trainer",
        smoke_max_steps=smoke_max_steps,
    )

    started = clock_fn()
    try:
        outcome = trainer.train()
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        raise WorkloadError(f"training raised {type(exc).__name__}: {exc}") from exc
    wall_seconds = clock_fn() - started

    result = {
        "global_step": int(getattr(trainer.state, "global_step", -1)),
        "training_loss": getattr(outcome, "training_loss", None),
        "log_history": list(getattr(trainer.state, "log_history", [])),
        "phase_records": list(getattr(trainer, "collected_phase_records", [])),
        "rollouts": list(getattr(trainer, "collected_rollouts", [])),
        "checkpoints": list(getattr(trainer, "saved_checkpoints", [])),
    }
    evidence = validate_training_result(result, steps=steps, smoke=smoke)

    manifest = {
        "version": VERSION,
        "status": "arm_smoke_completed" if smoke else "arm_run_completed",
        "arm": arm,
        "role": spec["role"],
        "smoke_mode": smoke,
        "composition": report,
        "evidence": evidence,
        "wall_seconds": wall_seconds,
        "published_to": None if smoke else str(final),
    }

    if smoke:
        # A smoke never claims a reserved path. Its evidence stays in scratch so
        # it cannot be mistaken for, or block, a production run.
        smoke_manifest = scratch_root / "smoke-manifest.json"
        write_exclusive_atomic_json(smoke_manifest, manifest)
        manifest["manifest_path"] = str(smoke_manifest)
        return manifest

    staging.mkdir(parents=True, exist_ok=False)
    try:
        write_exclusive_atomic_json(staging / "manifest.json", manifest)
        (staging / "rollouts.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in result["rollouts"]),
            encoding="utf-8",
        )
        (staging / "trainer-log.json").write_text(
            json.dumps(result["log_history"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "phase-timings.json").write_text(
            json.dumps(result["phase_records"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest |= publish_atomically(staging, final)
    except Exception:
        # Leave no plausible partial bundle at either path.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
