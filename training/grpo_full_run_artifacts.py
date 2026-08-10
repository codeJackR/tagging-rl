"""CPU-only checkpoint and publication contract for the 300-step GRPO run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from training.grpo_smoke_artifacts import (
    EXPECTED_ADAPTER_MODEL_BYTES,
    EXPECTED_ADAPTER_FILES,
    FORBIDDEN_BASENAMES,
    MINIMUM_FREE_DISK_BYTES,
    validate_saved_adapter,
)

FULL_RUN_LIFECYCLE_VERSION = "grpo-full-run-300-lifecycle-v1"
FULL_RUN_SUMMARY_VERSION = "grpo-full-run-summary-v1"
CHECKPOINT_STEPS = (100, 200, 300)
RETAINED_CHECKPOINT_STEPS = (200, 300)
EVICTED_CHECKPOINT_STEPS = (100,)
GENERATIONS_PER_STEP = 8
EXPECTED_FINAL_ROLLOUTS = 300 * GENERATIONS_PER_STEP
EXPECTED_STEP_100_ROLLOUTS = 100 * GENERATIONS_PER_STEP
MAX_FINAL_BUNDLE_BYTES = 512 * 1024**2
MAX_TRAINER_ROOT_README_BYTES = 64 * 1024
TRAINER_ROOT_REQUIRED_ENTRIES = {"checkpoint-200", "checkpoint-300"}
TRAINER_ROOT_ALLOWED_METADATA = {"README.md"}
CHECKPOINT_ALLOWED_METADATA = {"trainer_state.json", "training_args.bin"}
CHECKPOINT_FORBIDDEN_BASENAMES = (
    FORBIDDEN_BASENAMES - CHECKPOINT_ALLOWED_METADATA
)
STEP_100_EXPORT_FILES = {
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "manifest.json",
    "rollouts.jsonl",
    "trainer-log.json",
}
REQUIRED_EVENT_SEQUENCE = (
    "checkpoint_saved_100",
    "milestone_exported_100",
    "checkpoint_saved_200",
    "checkpoint_saved_300",
    "checkpoint_evicted_100",
    "retention_verified",
    "final_adapter_saved",
    "final_adapter_validated",
    "bundle_validated",
    "bundle_published",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_trainer_root_inventory(trainer: Path) -> dict:
    """Allow only retained checkpoints and a bounded trainer model card."""
    if not trainer.is_dir() or trainer.is_symlink():
        raise FileNotFoundError("trainer output must be a regular directory")
    entries = {path.name: path for path in trainer.iterdir()}
    entry_names = set(entries)
    if not TRAINER_ROOT_REQUIRED_ENTRIES.issubset(entry_names) or (
        entry_names
        - TRAINER_ROOT_REQUIRED_ENTRIES
        - TRAINER_ROOT_ALLOWED_METADATA
    ):
        raise RuntimeError("trainer checkpoint retention inventory drifted")
    for name in TRAINER_ROOT_REQUIRED_ENTRIES:
        checkpoint = entries[name]
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise RuntimeError("retained trainer checkpoint is not a directory")

    readme = entries.get("README.md")
    if readme is None:
        return {"readme_present": False}
    if not readme.is_file() or readme.is_symlink():
        raise ValueError("trainer README must be a regular non-symlink file")
    size = readme.stat().st_size
    if not 0 < size <= MAX_TRAINER_ROOT_README_BYTES:
        raise ValueError("trainer README is empty or exceeds its size bound")
    try:
        content = readme.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("trainer README must be valid UTF-8") from exc
    if not content.strip() or "\x00" in content:
        raise ValueError("trainer README contains invalid text content")
    return {
        "readme_present": True,
        "path": str(readme),
        "bytes": size,
        "sha256": _sha256_file(readme),
        "encoding": "utf-8",
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _validate_json_finite(value: object, *, label: str) -> None:
    """Fail unless a nested value can be persisted as strict finite JSON."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_finite(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{label} contains a non-string dictionary key")
        for key, item in value.items():
            _validate_json_finite(item, label=f"{label}.{key}")
        return
    raise TypeError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _validate_checkpoint_trainer_state(checkpoint: Path, *, step: int) -> dict:
    """Validate Transformers' non-resumable checkpoint metadata."""
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.is_file() or trainer_state.is_symlink():
        raise FileNotFoundError(
            f"checkpoint-{step} lacks regular trainer_state.json metadata"
        )
    duplicate_locations = [
        path
        for path in checkpoint.rglob("trainer_state.json")
        if path != trainer_state
    ]
    if duplicate_locations:
        raise ValueError(f"checkpoint-{step} has nested trainer-state metadata")
    try:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"checkpoint-{step} trainer_state.json is invalid JSON"
        ) from exc
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint-{step} trainer state must be a JSON object")
    _validate_json_finite(state, label=f"checkpoint-{step} trainer state")

    expected_fields = {
        "global_step": step,
        "max_steps": 300,
        "num_train_epochs": 1,
        "train_batch_size": 8,
        "logging_steps": 1,
        "save_steps": 100,
        "is_local_process_zero": True,
        "is_world_process_zero": True,
    }
    for field, expected in expected_fields.items():
        if state.get(field) != expected:
            raise ValueError(
                f"checkpoint-{step} trainer state {field} drifted: "
                f"{state.get(field)!r} != {expected!r}"
            )
    epoch = state.get("epoch")
    if (
        not isinstance(epoch, (int, float))
        or isinstance(epoch, bool)
        or not math.isfinite(float(epoch))
        or not 0.0 <= float(epoch) <= 1.0
    ):
        raise ValueError(f"checkpoint-{step} trainer state epoch is invalid")
    log_history = state.get("log_history")
    if not isinstance(log_history, list) or not all(
        isinstance(entry, dict) for entry in log_history
    ):
        raise TypeError(
            f"checkpoint-{step} trainer state log_history is invalid"
        )
    step_logs = [entry for entry in log_history if "loss" in entry]
    if len(log_history) != step or len(step_logs) != step:
        raise ValueError(
            f"checkpoint-{step} trainer state must contain exactly {step} "
            "step logs"
        )
    if [entry.get("step") for entry in step_logs] != list(range(1, step + 1)):
        raise ValueError(
            f"checkpoint-{step} trainer state step-log order drifted"
        )
    return {
        "path": str(trainer_state),
        "sha256": _sha256_file(trainer_state),
        "global_step": step,
        "max_steps": 300,
        "step_logs": step,
        "contains_optimizer_scheduler_or_rng_state": False,
    }


def _validate_cuda_snapshot(snapshot: object, *, label: str) -> dict:
    if not isinstance(snapshot, dict):
        raise TypeError(f"{label} must be a dictionary")
    required_integer_fields = {
        "device_index",
        "device_total_bytes",
        "driver_free_bytes",
        "driver_used_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
    }
    if required_integer_fields - set(snapshot):
        raise ValueError(f"{label} is missing CUDA byte measurements")
    if not isinstance(snapshot.get("device_name"), str) or not snapshot[
        "device_name"
    ]:
        raise ValueError(f"{label} has no CUDA device name")
    for field in required_integer_fields:
        value = snapshot[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{field} must be a nonnegative integer")
    if snapshot["device_total_bytes"] <= 0:
        raise ValueError(f"{label} has no device capacity")
    if snapshot["driver_free_bytes"] + snapshot["driver_used_bytes"] != snapshot[
        "device_total_bytes"
    ]:
        raise ValueError(f"{label} driver memory does not sum to device capacity")
    if snapshot["torch_reserved_bytes"] < snapshot["torch_allocated_bytes"]:
        raise ValueError(f"{label} reserved memory is below allocated memory")
    if snapshot["torch_peak_reserved_bytes"] < snapshot[
        "torch_peak_allocated_bytes"
    ]:
        raise ValueError(f"{label} peak reserved memory is below peak allocated")
    if snapshot["torch_peak_allocated_bytes"] < snapshot["torch_allocated_bytes"]:
        raise ValueError(f"{label} peak allocation is below current allocation")
    return dict(snapshot)


def validate_full_run_summary(summary: object) -> dict:
    """Validate the persisted training, model-health and resource evidence."""
    if not isinstance(summary, dict):
        raise TypeError("full-run summary must be a dictionary")
    if summary.get("version") != FULL_RUN_SUMMARY_VERSION:
        raise ValueError("unexpected full-run summary version")
    if summary.get("status") != "completed":
        raise ValueError("full-run summary is not completed")
    _validate_json_finite(summary, label="full-run summary")

    training = summary.get("training")
    if not isinstance(training, dict):
        raise TypeError("full-run summary has no training section")
    if training.get("optimizer_steps") != 300:
        raise ValueError("full-run summary has the wrong optimizer-step count")
    if training.get("rollout_records") != EXPECTED_FINAL_ROLLOUTS:
        raise ValueError("full-run summary has the wrong rollout count")
    train_seconds = training.get("train_seconds")
    if not isinstance(train_seconds, (int, float)) or train_seconds <= 0:
        raise ValueError("full-run summary has no positive training duration")
    if not isinstance(training.get("train_metrics"), dict):
        raise TypeError("full-run summary training metrics must be a dictionary")

    model_audit = summary.get("model_audit")
    if not isinstance(model_audit, dict):
        raise TypeError("full-run summary has no model audit")
    before_sha = model_audit.get("lora_sha256_before")
    after_sha = model_audit.get("lora_sha256_after")
    if not _is_sha256(before_sha) or not _is_sha256(after_sha):
        raise ValueError("full-run summary has an invalid LoRA SHA-256")
    if before_sha == after_sha or model_audit.get("trainable_lora_changed") is not True:
        raise ValueError("full-run summary did not record a LoRA change")
    before = model_audit.get("trainability_before")
    after = model_audit.get("trainability_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise TypeError("full-run summary trainability evidence is missing")
    for key, expected in (
        ("trainable_parameters", 18_464_768),
        ("trainable_tensors", 392),
    ):
        if before.get(key) != expected or after.get(key) != expected:
            raise ValueError(f"full-run summary trainability drifted for {key}")
    values = model_audit.get("parameter_values_after")
    if not isinstance(values, dict) or values.get(
        "all_trainable_values_finite"
    ) is not True:
        raise ValueError("full-run summary did not prove finite trainable values")
    if values.get("nonfinite_parameters") != 0:
        raise ValueError("full-run summary contains non-finite trainable values")

    resources = summary.get("resources")
    if not isinstance(resources, dict):
        raise TypeError("full-run summary has no resource section")
    snapshot_names = (
        "cuda_before_load",
        "cuda_after_load",
        "cuda_before_train",
        "cuda_after_train",
        "cuda_after_model_audit",
    )
    snapshots = {
        name: _validate_cuda_snapshot(resources.get(name), label=name)
        for name in snapshot_names
    }
    identity = {
        (
            snapshot["device_index"],
            snapshot["device_name"],
            snapshot["device_total_bytes"],
        )
        for snapshot in snapshots.values()
    }
    if len(identity) != 1:
        raise ValueError("full-run CUDA device identity drifted between snapshots")
    runtime = resources.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("full-run summary has no runtime metadata")
    if resources.get("preflight_disk_free_bytes", 0) < MINIMUM_FREE_DISK_BYTES:
        raise ValueError("full-run summary preflight disk was below the floor")

    return {
        "version": FULL_RUN_SUMMARY_VERSION,
        "status": "passed",
        "optimizer_steps": 300,
        "rollout_records": EXPECTED_FINAL_ROLLOUTS,
        "train_seconds": float(train_seconds),
        "lora_sha256_before": before_sha,
        "lora_sha256_after": after_sha,
        "cuda_device": next(iter(identity))[1],
        "peak_allocated_bytes": max(
            snapshot["torch_peak_allocated_bytes"]
            for snapshot in snapshots.values()
        ),
        "peak_reserved_bytes": max(
            snapshot["torch_peak_reserved_bytes"]
            for snapshot in snapshots.values()
        ),
        "preflight_disk_free_bytes": resources["preflight_disk_free_bytes"],
        "all_values_finite": True,
    }


def create_full_run_staging_output(final_output_dir: str | Path) -> Path:
    """Create the private run root only when final and stale staging are absent."""
    final_output = Path(final_output_dir).resolve()
    if final_output.exists():
        raise FileExistsError(f"final full-run output already exists: {final_output}")
    if not final_output.parent.is_dir():
        raise FileNotFoundError(
            f"final full-run output parent does not exist: {final_output.parent}"
        )
    staging_pattern = f".{final_output.name}.staging-*"
    collisions = sorted(final_output.parent.glob(staging_pattern))
    if collisions:
        raise FileExistsError(f"stale full-run staging output exists: {collisions[0]}")
    return Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.staging-",
            dir=final_output.parent,
        )
    ).resolve()


def build_full_run_lifecycle_plan(
    *,
    final_output_dir: str | Path,
    staging_dir: str | Path,
) -> dict:
    """Bind trainer checkpoints and durable evidence to one atomic staging root."""
    final_output = Path(final_output_dir).resolve()
    staging = Path(staging_dir).resolve()
    expected_prefix = f".{final_output.name}.staging-"
    if staging.parent != final_output.parent or not staging.name.startswith(
        expected_prefix
    ):
        raise ValueError("full-run staging directory is not bound to final output")
    if staging == final_output:
        raise ValueError("full-run staging and final output must differ")

    trainer = staging / "trainer"
    milestone = staging / "milestones" / "step-100"
    return {
        "version": FULL_RUN_LIFECYCLE_VERSION,
        "status": "planned_not_executed",
        "final_output_dir": str(final_output),
        "staging_dir": str(staging),
        "trainer_output_dir": str(trainer),
        "checkpoint_paths": {
            str(step): str(trainer / f"checkpoint-{step}")
            for step in CHECKPOINT_STEPS
        },
        "checkpoint_policy": {
            "save_steps": 100,
            "save_total_limit": 2,
            "save_only_model": True,
            "retained_steps": list(RETAINED_CHECKPOINT_STEPS),
            "evicted_steps": list(EVICTED_CHECKPOINT_STEPS),
        },
        "step_100_export": {
            "directory": str(milestone),
            "required_before_checkpoint_eviction": True,
            "required_files": sorted(STEP_100_EXPORT_FILES),
            "rollout_records": EXPECTED_STEP_100_ROLLOUTS,
            "trainer_step_logs": 100,
        },
        "final_adapter_dir": str(staging / "final-adapter"),
        "root_evidence": {
            "manifest": str(staging / "manifest.json"),
            "rollouts": str(staging / "rollouts.jsonl"),
            "trainer_log": str(staging / "trainer-log.json"),
            "rollout_records": EXPECTED_FINAL_ROLLOUTS,
            "trainer_step_logs": 300,
        },
        "publication": {
            "method": "same_filesystem_atomic_rename",
            "maximum_bundle_bytes": MAX_FINAL_BUNDLE_BYTES,
            "minimum_free_disk_after_bytes": MINIMUM_FREE_DISK_BYTES,
            "required_event_sequence": list(REQUIRED_EVENT_SEQUENCE),
        },
    }


class FullRunCheckpointLifecycleWriter:
    """Produce the durable step-100 export and audit bounded retention.

    A future Trainer callback will call these methods. This class deliberately
    knows nothing about Torch, TRL, a model, or training dispatch.
    """

    def __init__(self, *, plan: dict, starting_adapter_sha256: str):
        if plan.get("version") != FULL_RUN_LIFECYCLE_VERSION:
            raise ValueError("unexpected full-run lifecycle plan version")
        if not _is_sha256(starting_adapter_sha256):
            raise ValueError("starting adapter SHA-256 is invalid")
        staging = Path(plan["staging_dir"])
        if not staging.is_dir() or staging.is_symlink():
            raise FileNotFoundError("full-run staging root is absent")
        self.plan = plan
        self.starting_adapter_sha256 = starting_adapter_sha256
        self.events: list[dict] = []

    def _checkpoint_path(self, step: int) -> Path:
        try:
            return Path(self.plan["checkpoint_paths"][str(step)])
        except KeyError as exc:
            raise ValueError(f"unexpected checkpoint step: {step}") from exc

    def record_checkpoint_saved(self, step: int) -> dict:
        expected_steps = (100, 200, 300)
        recorded_steps = [
            event["step"]
            for event in self.events
            if event["event"].startswith("checkpoint_saved_")
        ]
        expected_next = expected_steps[len(recorded_steps)] if len(recorded_steps) < 3 else None
        if step != expected_next:
            raise RuntimeError(
                f"checkpoint save order drifted: expected {expected_next}, found {step}"
            )
        if step == 200 and [event["event"] for event in self.events] != [
            "checkpoint_saved_100",
            "milestone_exported_100",
        ]:
            raise RuntimeError("checkpoint 200 cannot precede the step-100 export")

        checkpoint = self._checkpoint_path(step)
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise FileNotFoundError(f"checkpoint-{step} is absent")
        required = {
            checkpoint / "adapter_config.json",
            checkpoint / "adapter_model.safetensors",
        }
        if not all(path.is_file() and not path.is_symlink() for path in required):
            raise FileNotFoundError(f"checkpoint-{step} lacks PEFT adapter files")
        forbidden = [
            path
            for path in checkpoint.rglob("*")
            if path.is_file()
            and (
                path.name in CHECKPOINT_FORBIDDEN_BASENAMES
                or path.name.startswith("rng_state")
                or path.name.startswith("model-")
                or path.name.startswith("pytorch_model-")
            )
        ]
        if forbidden:
            raise ValueError(
                f"checkpoint-{step} contains forbidden state: {forbidden[0].name}"
            )
        trainer_state = _validate_checkpoint_trainer_state(
            checkpoint,
            step=step,
        )
        event = {
            "event": f"checkpoint_saved_{step}",
            "step": step,
            "path": str(checkpoint),
            "save_only_model": True,
            "trainer_state_sha256": trainer_state["sha256"],
            "trainer_state_global_step": trainer_state["global_step"],
            "trainer_state_step_logs": trainer_state["step_logs"],
            "contains_optimizer_scheduler_or_rng_state": False,
        }
        self.events.append(event)
        return dict(event)

    def export_step_100(
        self,
        *,
        rollout_records: Sequence[dict],
        trainer_step_logs: Sequence[dict],
    ) -> dict:
        if [event["event"] for event in self.events] != ["checkpoint_saved_100"]:
            raise RuntimeError("step-100 export must immediately follow checkpoint 100")
        if len(rollout_records) != EXPECTED_STEP_100_ROLLOUTS:
            raise ValueError("step-100 export requires exactly 800 rollout records")
        if len(trainer_step_logs) != 100:
            raise ValueError("step-100 export requires exactly 100 trainer step logs")
        if not all(isinstance(row, dict) for row in rollout_records):
            raise TypeError("step-100 rollouts must be JSON objects")
        if not all(isinstance(row, dict) for row in trainer_step_logs):
            raise TypeError("step-100 trainer logs must be JSON objects")

        checkpoint = self._checkpoint_path(100)
        source_weights = checkpoint / "adapter_model.safetensors"
        source_config = checkpoint / "adapter_config.json"
        weights_sha = _sha256_file(source_weights)
        config_sha = _sha256_file(source_config)
        if weights_sha == self.starting_adapter_sha256:
            raise ValueError("step-100 adapter is unchanged from the SFT adapter")

        milestone = Path(self.plan["step_100_export"]["directory"])
        if milestone.exists():
            raise FileExistsError("step-100 milestone already exists")
        milestone.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".step-100.staging-", dir=milestone.parent)
        )
        adapter_output = temporary / "adapter"
        adapter_output.mkdir()
        shutil.copy2(source_weights, adapter_output / source_weights.name)
        shutil.copy2(source_config, adapter_output / source_config.name)
        _write_jsonl(temporary / "rollouts.jsonl", rollout_records)
        _write_json(temporary / "trainer-log.json", list(trainer_step_logs))
        manifest = {
            "version": "grpo-full-run-step-100-export-v1",
            "status": "completed",
            "step": 100,
            "source_checkpoint": str(checkpoint),
            "adapter_model_sha256": weights_sha,
            "adapter_config_sha256": config_sha,
            "rollout_records": len(rollout_records),
            "trainer_step_logs": len(trainer_step_logs),
            "files": sorted(STEP_100_EXPORT_FILES),
            "source_adapter_unchanged_by_export": True,
        }
        _write_json(temporary / "manifest.json", manifest)

        if _sha256_file(source_weights) != weights_sha or _sha256_file(
            source_config
        ) != config_sha:
            raise RuntimeError("checkpoint 100 changed during milestone export")
        if _sha256_file(adapter_output / source_weights.name) != weights_sha:
            raise RuntimeError("exported step-100 adapter weights differ from source")
        if _sha256_file(adapter_output / source_config.name) != config_sha:
            raise RuntimeError("exported step-100 adapter config differs from source")
        actual_files = {
            str(path.relative_to(temporary))
            for path in temporary.rglob("*")
            if path.is_file()
        }
        if actual_files != STEP_100_EXPORT_FILES:
            raise RuntimeError("step-100 temporary export inventory drifted")
        os.rename(temporary, milestone)
        if not milestone.is_dir() or temporary.exists():
            raise RuntimeError("step-100 milestone publication failed")

        event = {
            "event": "milestone_exported_100",
            "step": 100,
            "path": str(milestone),
            "files": sorted(actual_files),
            "rollout_records": len(rollout_records),
            "trainer_step_logs": len(trainer_step_logs),
            "adapter_model_sha256": weights_sha,
            "adapter_config_sha256": config_sha,
        }
        self.events.append(event)
        return dict(event)

    def verify_retention_after_step_300(self) -> dict:
        expected_prefix = [
            "checkpoint_saved_100",
            "milestone_exported_100",
            "checkpoint_saved_200",
            "checkpoint_saved_300",
        ]
        if [event["event"] for event in self.events] != expected_prefix:
            raise RuntimeError("checkpoint retention cannot be verified yet")
        checkpoint_100 = self._checkpoint_path(100)
        retained = [self._checkpoint_path(step) for step in RETAINED_CHECKPOINT_STEPS]
        if checkpoint_100.exists():
            raise RuntimeError("checkpoint 100 was not evicted at the retention limit")
        if not all(path.is_dir() and not path.is_symlink() for path in retained):
            raise RuntimeError("checkpoint 200 or 300 is missing after retention")
        milestone = Path(self.plan["step_100_export"]["directory"])
        if not milestone.is_dir() or milestone.is_symlink():
            raise RuntimeError("step-100 milestone is absent after checkpoint eviction")

        eviction = {
            "event": "checkpoint_evicted_100",
            "step": 100,
            "path": str(checkpoint_100),
            "milestone_export_verified": True,
        }
        retention = {
            "event": "retention_verified",
            "retained_steps": list(RETAINED_CHECKPOINT_STEPS),
            "absent_steps": list(EVICTED_CHECKPOINT_STEPS),
        }
        self.events.extend((eviction, retention))
        return {
            "eviction": dict(eviction),
            "retention": dict(retention),
        }

    def save_and_validate_final_adapter(
        self,
        *,
        save_adapter_fn: Callable[[Path], None],
        source_adapter_file: str | Path,
        expected_adapter_model_bytes: int = EXPECTED_ADAPTER_MODEL_BYTES,
    ) -> dict:
        """Save the step-300 PEFT adapter and verify source/final integrity."""
        if [event["event"] for event in self.events] != list(
            REQUIRED_EVENT_SEQUENCE[:6]
        ):
            raise RuntimeError("final adapter cannot precede checkpoint retention")
        source_adapter = Path(source_adapter_file).resolve()
        if _sha256_file(source_adapter) != self.starting_adapter_sha256:
            raise RuntimeError("source SFT adapter drifted before final save")
        final_adapter = Path(self.plan["final_adapter_dir"])
        if final_adapter.exists():
            raise FileExistsError("final adapter output already exists")

        save_adapter_fn(final_adapter)
        if not final_adapter.is_dir() or final_adapter.is_symlink():
            raise RuntimeError("final adapter saver did not create its directory")
        if _sha256_file(source_adapter) != self.starting_adapter_sha256:
            raise RuntimeError("source SFT adapter changed during final save")
        saved = {
            "event": "final_adapter_saved",
            "step": 300,
            "path": str(final_adapter),
        }
        self.events.append(saved)

        validation = validate_saved_adapter(
            final_adapter,
            starting_adapter_sha256=self.starting_adapter_sha256,
            expected_adapter_model_bytes=expected_adapter_model_bytes,
        )
        step_100_sha = self.events[1]["adapter_model_sha256"]
        final_sha = validation["adapter_model_sha256"]
        if final_sha == step_100_sha:
            raise RuntimeError("final adapter is unchanged from step 100")
        if validation.get("contains_optimizer_state") is not False:
            raise RuntimeError("final adapter validation found optimizer state")
        if validation.get("contains_full_model") is not False:
            raise RuntimeError("final adapter validation found full-model state")
        validated = {
            "event": "final_adapter_validated",
            "step": 300,
            "path": str(final_adapter),
            "files": sorted(item["path"] for item in validation["files"]),
            "adapter_model_sha256": final_sha,
            "contains_optimizer_state": False,
            "contains_full_model": False,
        }
        self.events.append(validated)
        self._final_adapter_validation = dict(validation)
        return {
            "saved": dict(saved),
            "validated": dict(validated),
            "validation": dict(validation),
            "source_adapter_unchanged": True,
        }

    def publish_completed_bundle(
        self,
        *,
        rollout_records: Sequence[dict],
        trainer_step_logs: Sequence[dict],
        preflight_report: dict,
        config_settings: dict,
        run_summary: dict,
        disk_usage_fn: Callable[[Path], object] | None = None,
        maximum_bundle_bytes: int = MAX_FINAL_BUNDLE_BYTES,
        minimum_free_bytes: int = MINIMUM_FREE_DISK_BYTES,
    ) -> dict:
        """Validate the completed staging tree and atomically publish the run."""
        if [event["event"] for event in self.events] != list(
            REQUIRED_EVENT_SEQUENCE[:8]
        ):
            raise RuntimeError("bundle publication requires a validated final adapter")
        if len(rollout_records) != EXPECTED_FINAL_ROLLOUTS:
            raise ValueError("completed bundle requires exactly 2,400 rollouts")
        if len(trainer_step_logs) != 300:
            raise ValueError("completed bundle requires exactly 300 trainer step logs")
        if not all(isinstance(row, dict) for row in rollout_records):
            raise TypeError("full-run rollouts must be JSON objects")
        if not all(isinstance(row, dict) for row in trainer_step_logs):
            raise TypeError("full-run trainer logs must be JSON objects")
        if preflight_report.get("status") != "passed":
            raise ValueError("completed bundle requires a passed preflight")
        required_config = {
            "max_steps": 300,
            "num_generations": 8,
            "save_steps": 100,
            "save_total_limit": 2,
            "save_only_model": True,
        }
        for key, expected in required_config.items():
            if config_settings.get(key) != expected:
                raise ValueError(f"completed bundle config drifted for {key}")
        run_summary_validation = validate_full_run_summary(run_summary)

        staging = Path(self.plan["staging_dir"])
        final_output = Path(self.plan["final_output_dir"])
        if not staging.is_dir() or staging.is_symlink():
            raise FileNotFoundError("full-run staging directory is absent")
        if final_output.exists():
            raise FileExistsError("final full-run output already exists")
        root_files = {
            "rollouts.jsonl": staging / "rollouts.jsonl",
            "trainer-log.json": staging / "trainer-log.json",
            "manifest.json": staging / "manifest.json",
        }
        if any(path.exists() for path in root_files.values()):
            raise FileExistsError("completed-bundle evidence already exists")

        trainer = Path(self.plan["trainer_output_dir"])
        trainer_root_metadata = _validate_trainer_root_inventory(trainer)
        milestones = staging / "milestones"
        if {path.name for path in milestones.iterdir()} != {"step-100"}:
            raise RuntimeError("milestone inventory drifted before publication")
        final_adapter = Path(self.plan["final_adapter_dir"])
        if not final_adapter.is_dir() or final_adapter.is_symlink():
            raise RuntimeError("validated final adapter disappeared")

        _write_jsonl(root_files["rollouts.jsonl"], rollout_records)
        _write_json(root_files["trainer-log.json"], list(trainer_step_logs))
        artifact_hashes = {
            "rollouts_sha256": _sha256_file(root_files["rollouts.jsonl"]),
            "trainer_log_sha256": _sha256_file(root_files["trainer-log.json"]),
        }
        manifest = {
            "version": "grpo-full-run-300-bundle-v1",
            "status": "completed",
            "steps": 300,
            "rollout_records": len(rollout_records),
            "trainer_step_logs": len(trainer_step_logs),
            "preflight": preflight_report,
            "config": config_settings,
            "run_summary": run_summary,
            "run_summary_validation": run_summary_validation,
            "checkpoint_lifecycle_version": FULL_RUN_LIFECYCLE_VERSION,
            "checkpoint_events_before_publication": [
                dict(event) for event in self.events
            ],
            "trainer_root_metadata": trainer_root_metadata,
            "final_adapter": self._final_adapter_validation,
            "artifacts": artifact_hashes,
            "publication_method": "same_filesystem_atomic_rename",
        }
        _write_json(root_files["manifest.json"], manifest)

        expected_root_entries = {
            "trainer",
            "milestones",
            "final-adapter",
            "rollouts.jsonl",
            "trainer-log.json",
            "manifest.json",
        }
        if {path.name for path in staging.iterdir()} != expected_root_entries:
            raise RuntimeError("completed full-run root inventory drifted")
        files = [path for path in staging.rglob("*") if path.is_file()]
        if any(path.is_symlink() for path in staging.rglob("*")):
            raise ValueError("completed full-run bundle contains a symlink")
        total_bytes = sum(path.stat().st_size for path in files)
        if (
            not isinstance(maximum_bundle_bytes, int)
            or maximum_bundle_bytes <= 0
            or total_bytes > maximum_bundle_bytes
        ):
            raise ValueError("completed full-run bundle exceeds its size bound")
        usage = (disk_usage_fn or shutil.disk_usage)(staging)
        free_bytes = int(usage.free)
        if free_bytes < minimum_free_bytes:
            raise ValueError("completed full-run bundle leaves insufficient free disk")

        bundle_event = {
            "event": "bundle_validated",
            "step": 300,
            "path": str(staging),
            "rollout_records": len(rollout_records),
            "trainer_step_logs": len(trainer_step_logs),
            "total_bytes": total_bytes,
            "disk_free_after_bytes": free_bytes,
        }
        publication_event = {
            "event": "bundle_published",
            "step": 300,
            "source": str(staging),
            "path": str(final_output),
            "atomic": True,
        }
        candidate_events = [
            *self.events,
            bundle_event,
            publication_event,
        ]
        lifecycle_validation = validate_full_run_lifecycle_events(
            candidate_events,
            plan=self.plan,
            starting_adapter_sha256=self.starting_adapter_sha256,
        )
        os.rename(staging, final_output)
        if not final_output.is_dir() or staging.exists():
            raise RuntimeError("atomic full-run publication failed")
        self.events = candidate_events
        manifest_sha256 = _sha256_file(final_output / "manifest.json")
        return {
            "version": "grpo-full-run-300-bundle-v1",
            "status": "completed",
            "final_output_dir": str(final_output),
            "manifest_sha256": manifest_sha256,
            "rollouts_sha256": artifact_hashes["rollouts_sha256"],
            "trainer_log_sha256": artifact_hashes["trainer_log_sha256"],
            "trainer_root_metadata": trainer_root_metadata,
            "run_summary_validation": run_summary_validation,
            "total_bytes": total_bytes,
            "disk_free_after_bytes": free_bytes,
            "lifecycle_validation": lifecycle_validation,
            "published_atomically": True,
        }

    def snapshot(self) -> dict:
        names = [event["event"] for event in self.events]
        ready = names == list(REQUIRED_EVENT_SEQUENCE[:6])
        completed = names == list(REQUIRED_EVENT_SEQUENCE)
        return {
            "version": FULL_RUN_LIFECYCLE_VERSION,
            "status": (
                "completed_and_published"
                if completed
                else (
                    "checkpoints_ready_for_final_handoff"
                    if ready
                    else "checkpoint_lifecycle_incomplete"
                )
            ),
            "events": [dict(event) for event in self.events],
            "event_count": len(self.events),
            "step_100_exported_before_eviction": ready or completed,
            "retained_checkpoint_steps": (
                list(RETAINED_CHECKPOINT_STEPS) if ready or completed else []
            ),
            "final_adapter_saved": completed,
            "bundle_published": completed,
        }


def validate_full_run_lifecycle_events(
    events: Sequence[dict],
    *,
    plan: dict,
    starting_adapter_sha256: str,
) -> dict:
    """Validate that eviction and publication occurred only after durable export."""
    if plan.get("version") != FULL_RUN_LIFECYCLE_VERSION:
        raise ValueError("unexpected full-run lifecycle plan version")
    if not _is_sha256(starting_adapter_sha256):
        raise ValueError("starting adapter SHA-256 is invalid")
    names = [event.get("event") for event in events]
    if names != list(REQUIRED_EVENT_SEQUENCE):
        raise ValueError("full-run lifecycle event sequence drifted")

    by_name = {event["event"]: event for event in events}
    checkpoint_paths = plan["checkpoint_paths"]
    for step in CHECKPOINT_STEPS:
        event = by_name[f"checkpoint_saved_{step}"]
        if event.get("step") != step:
            raise ValueError(f"checkpoint-{step} event has the wrong step")
        if event.get("path") != checkpoint_paths[str(step)]:
            raise ValueError(f"checkpoint-{step} path drifted")
        if event.get("save_only_model") is not True:
            raise ValueError(f"checkpoint-{step} is not model-only")
        if not _is_sha256(event.get("trainer_state_sha256")):
            raise ValueError(f"checkpoint-{step} trainer-state hash is invalid")
        if event.get("trainer_state_global_step") != step:
            raise ValueError(f"checkpoint-{step} trainer-state step drifted")
        if event.get("trainer_state_step_logs") != step:
            raise ValueError(f"checkpoint-{step} trainer-state logs drifted")
        if event.get("contains_optimizer_scheduler_or_rng_state") is not False:
            raise ValueError(f"checkpoint-{step} contains resumable state")

    milestone = by_name["milestone_exported_100"]
    expected_milestone = plan["step_100_export"]
    if milestone.get("step") != 100:
        raise ValueError("step-100 export has the wrong step")
    if milestone.get("path") != expected_milestone["directory"]:
        raise ValueError("step-100 export path drifted")
    if set(milestone.get("files", ())) != STEP_100_EXPORT_FILES:
        raise ValueError("step-100 export file inventory drifted")
    if milestone.get("rollout_records") != EXPECTED_STEP_100_ROLLOUTS:
        raise ValueError("step-100 export has the wrong rollout count")
    if milestone.get("trainer_step_logs") != 100:
        raise ValueError("step-100 export has the wrong trainer-log count")
    step_100_sha = milestone.get("adapter_model_sha256")
    if not _is_sha256(step_100_sha) or step_100_sha == starting_adapter_sha256:
        raise ValueError("step-100 adapter hash is invalid or unchanged")
    if not _is_sha256(milestone.get("adapter_config_sha256")):
        raise ValueError("step-100 adapter-config hash is invalid")

    eviction = by_name["checkpoint_evicted_100"]
    if eviction.get("step") != 100 or eviction.get("path") != checkpoint_paths["100"]:
        raise ValueError("checkpoint-100 eviction evidence drifted")
    if eviction.get("milestone_export_verified") is not True:
        raise ValueError("checkpoint-100 was evicted before export verification")

    retention = by_name["retention_verified"]
    if retention.get("retained_steps") != list(RETAINED_CHECKPOINT_STEPS):
        raise ValueError("retained checkpoint set drifted")
    if retention.get("absent_steps") != list(EVICTED_CHECKPOINT_STEPS):
        raise ValueError("evicted checkpoint set drifted")

    final_saved = by_name["final_adapter_saved"]
    if final_saved.get("step") != 300:
        raise ValueError("final adapter was not saved at step 300")
    if final_saved.get("path") != plan["final_adapter_dir"]:
        raise ValueError("final adapter save path drifted")

    final_validated = by_name["final_adapter_validated"]
    if final_validated.get("step") != 300:
        raise ValueError("final adapter validation has the wrong step")
    if final_validated.get("path") != plan["final_adapter_dir"]:
        raise ValueError("final adapter validation path drifted")
    if set(final_validated.get("files", ())) != EXPECTED_ADAPTER_FILES:
        raise ValueError("final adapter file inventory drifted")
    final_sha = final_validated.get("adapter_model_sha256")
    if (
        not _is_sha256(final_sha)
        or final_sha == starting_adapter_sha256
        or final_sha == step_100_sha
    ):
        raise ValueError("final adapter hash is invalid or unchanged")
    if final_validated.get("contains_optimizer_state") is not False:
        raise ValueError("final adapter contains optimizer state")
    if final_validated.get("contains_full_model") is not False:
        raise ValueError("final adapter contains full-model state")

    bundle = by_name["bundle_validated"]
    if bundle.get("step") != 300 or bundle.get("path") != plan["staging_dir"]:
        raise ValueError("final bundle validation path or step drifted")
    if bundle.get("rollout_records") != EXPECTED_FINAL_ROLLOUTS:
        raise ValueError("final bundle has the wrong rollout count")
    if bundle.get("trainer_step_logs") != 300:
        raise ValueError("final bundle has the wrong trainer-log count")
    total_bytes = bundle.get("total_bytes")
    if not isinstance(total_bytes, int) or not 0 < total_bytes <= MAX_FINAL_BUNDLE_BYTES:
        raise ValueError("final bundle exceeds its size bound")
    free_bytes = bundle.get("disk_free_after_bytes")
    if not isinstance(free_bytes, int) or free_bytes < MINIMUM_FREE_DISK_BYTES:
        raise ValueError("final bundle leaves insufficient free disk")

    published = by_name["bundle_published"]
    if published.get("step") != 300:
        raise ValueError("bundle publication has the wrong step")
    if published.get("source") != plan["staging_dir"]:
        raise ValueError("bundle publication source drifted")
    if published.get("path") != plan["final_output_dir"]:
        raise ValueError("bundle publication destination drifted")
    if published.get("atomic") is not True:
        raise ValueError("full-run bundle was not published atomically")

    return {
        "version": FULL_RUN_LIFECYCLE_VERSION,
        "status": "passed",
        "events": len(events),
        "step_100_exported_before_eviction": True,
        "retained_checkpoint_steps": list(RETAINED_CHECKPOINT_STEPS),
        "final_adapter_sha256": final_sha,
        "final_bundle_bytes": total_bytes,
        "disk_free_after_bytes": free_bytes,
        "published_atomically": True,
    }
