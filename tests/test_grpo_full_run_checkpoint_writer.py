"""Filesystem tests for the CPU-only full-run checkpoint lifecycle writer."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_full_run_artifacts import (
    EXPECTED_ADAPTER_FILES,
    FullRunCheckpointLifecycleWriter,
    build_full_run_lifecycle_plan,
    create_full_run_staging_output,
    validate_full_run_summary,
)

STARTING_SHA = hashlib.sha256(b"starting SFT adapter").hexdigest()


def trainer_state(step: int) -> dict:
    return {
        "global_step": step,
        "max_steps": 300,
        "num_train_epochs": 1,
        "train_batch_size": 8,
        "logging_steps": 1,
        "save_steps": 100,
        "is_local_process_zero": True,
        "is_world_process_zero": True,
        "epoch": step / 1_565,
        "log_history": [
            {"step": index, "loss": 0.0}
            for index in range(1, step + 1)
        ],
    }


def make_checkpoint(path: Path, step: int):
    path.mkdir(parents=True)
    weights = f"updated adapter at step {step}".encode()
    (path / "adapter_model.safetensors").write_bytes(weights)
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 16, "step": step}), encoding="utf-8"
    )
    # Transformers 4.57.6 writes this small metadata file even with
    # save_only_model=True; resumable optimizer/scheduler/RNG state stays absent.
    (path / "training_args.bin").write_bytes(b"metadata")
    (path / "trainer_state.json").write_text(
        json.dumps(trainer_state(step)),
        encoding="utf-8",
    )
    return hashlib.sha256(weights).hexdigest()


def make_writer(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def rollouts(count=800):
    return [{"step": index // 8 + 1, "rollout_index": index % 8} for index in range(count)]


def logs(count=100):
    return [{"step": index + 1, "loss": 0.0} for index in range(count)]


def make_ready_writer(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint_100 = Path(plan["checkpoint_paths"]["100"])
    checkpoint_200 = Path(plan["checkpoint_paths"]["200"])
    checkpoint_300 = Path(plan["checkpoint_paths"]["300"])
    make_checkpoint(checkpoint_100, 100)
    writer.record_checkpoint_saved(100)
    writer.export_step_100(rollout_records=rollouts(), trainer_step_logs=logs())
    make_checkpoint(checkpoint_200, 200)
    writer.record_checkpoint_saved(200)
    make_checkpoint(checkpoint_300, 300)
    writer.record_checkpoint_saved(300)
    shutil.rmtree(checkpoint_100)
    writer.verify_retention_after_step_300()
    source_adapter = tmp_path / "source-adapter.safetensors"
    source_adapter.write_bytes(b"starting SFT adapter")
    return writer, plan, source_adapter


def save_fake_final_adapter(path: Path, weights=b"final adapter at step 300"):
    path.mkdir()
    config = {
        "base_model_name_or_path": "unsloth/Qwen2.5-1.5B-Instruct",
        "r": 16,
        "lora_alpha": 16,
        "peft_type": "LORA",
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    for name in EXPECTED_ADAPTER_FILES:
        output = path / name
        if name == "adapter_model.safetensors":
            output.write_bytes(weights)
        elif name == "adapter_config.json":
            output.write_text(json.dumps(config), encoding="utf-8")
        else:
            output.write_text("{}", encoding="utf-8")


def locked_config():
    return {
        "max_steps": 300,
        "num_generations": 8,
        "save_steps": 100,
        "save_total_limit": 2,
        "save_only_model": True,
    }


def cuda_snapshot(**overrides):
    value = {
        "device_index": 0,
        "device_name": "Fake GPU",
        "device_total_bytes": 1_000,
        "driver_free_bytes": 800,
        "driver_used_bytes": 200,
        "torch_allocated_bytes": 10,
        "torch_reserved_bytes": 20,
        "torch_peak_allocated_bytes": 30,
        "torch_peak_reserved_bytes": 40,
    }
    value.update(overrides)
    return value


def run_summary():
    return {
        "version": "grpo-full-run-summary-v1",
        "status": "completed",
        "training": {
            "optimizer_steps": 300,
            "rollout_records": 2_400,
            "train_seconds": 90.0,
            "train_metrics": {"train_loss": -1.5},
        },
        "model_audit": {
            "trainability_before": {
                "trainable_parameters": 18_464_768,
                "trainable_tensors": 392,
            },
            "trainability_after": {
                "trainable_parameters": 18_464_768,
                "trainable_tensors": 392,
            },
            "parameter_values_after": {
                "all_trainable_values_finite": True,
                "nonfinite_parameters": 0,
            },
            "lora_sha256_before": "e" * 64,
            "lora_sha256_after": "f" * 64,
            "trainable_lora_changed": True,
        },
        "resources": {
            "runtime": {"torch_version": "fake"},
            "preflight_disk_free_bytes": 4 * 1024**3,
            "cuda_before_load": cuda_snapshot(),
            "cuda_after_load": cuda_snapshot(),
            "cuda_before_train": cuda_snapshot(),
            "cuda_after_train": cuda_snapshot(),
            "cuda_after_model_audit": cuda_snapshot(),
        },
    }


def test_step_100_is_exported_atomically_before_bounded_eviction(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint_100 = Path(plan["checkpoint_paths"]["100"])
    expected_sha = make_checkpoint(checkpoint_100, 100)

    writer.record_checkpoint_saved(100)
    export = writer.export_step_100(
        rollout_records=rollouts(),
        trainer_step_logs=logs(),
    )

    milestone = Path(plan["step_100_export"]["directory"])
    assert milestone.is_dir()
    assert export["adapter_model_sha256"] == expected_sha
    assert len((milestone / "rollouts.jsonl").read_text().splitlines()) == 800
    assert len(json.loads((milestone / "trainer-log.json").read_text())) == 100
    manifest = json.loads((milestone / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["source_adapter_unchanged_by_export"]
    assert not list(milestone.parent.glob(".step-100.staging-*"))

    checkpoint_200 = Path(plan["checkpoint_paths"]["200"])
    checkpoint_300 = Path(plan["checkpoint_paths"]["300"])
    make_checkpoint(checkpoint_200, 200)
    writer.record_checkpoint_saved(200)
    make_checkpoint(checkpoint_300, 300)
    writer.record_checkpoint_saved(300)
    shutil.rmtree(checkpoint_100)
    retention = writer.verify_retention_after_step_300()

    assert retention["retention"]["retained_steps"] == [200, 300]
    snapshot = writer.snapshot()
    assert snapshot["status"] == "checkpoints_ready_for_final_handoff"
    assert snapshot["event_count"] == 6
    assert snapshot["step_100_exported_before_eviction"]
    assert not snapshot["final_adapter_saved"]
    assert not snapshot["bundle_published"]


def test_staging_creation_refuses_final_or_stale_collision(tmp_path):
    final = tmp_path / "grpo-first-300"
    final.mkdir()
    with pytest.raises(FileExistsError, match="final full-run output"):
        create_full_run_staging_output(final)

    final.rmdir()
    (tmp_path / ".grpo-first-300.staging-stale").mkdir()
    with pytest.raises(FileExistsError, match="stale full-run staging"):
        create_full_run_staging_output(final)


def test_writer_requires_ordered_model_only_checkpoints(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint_200 = Path(plan["checkpoint_paths"]["200"])
    make_checkpoint(checkpoint_200, 200)
    with pytest.raises(RuntimeError, match="save order drifted"):
        writer.record_checkpoint_saved(200)

    checkpoint_100 = Path(plan["checkpoint_paths"]["100"])
    make_checkpoint(checkpoint_100, 100)
    (checkpoint_100 / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="forbidden state"):
        writer.record_checkpoint_saved(100)


def test_writer_accepts_real_transformers_model_only_inventory(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint = Path(plan["checkpoint_paths"]["100"])
    make_checkpoint(checkpoint, 100)
    for name in (
        "README.md",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (checkpoint / name).write_text("metadata\n", encoding="utf-8")

    event = writer.record_checkpoint_saved(100)

    assert event["trainer_state_global_step"] == 100
    assert event["trainer_state_step_logs"] == 100
    assert len(event["trainer_state_sha256"]) == 64
    assert event["contains_optimizer_scheduler_or_rng_state"] is False
    assert not (checkpoint / "optimizer.pt").exists()
    assert not (checkpoint / "scheduler.pt").exists()
    assert not (checkpoint / "rng_state.pth").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.update(global_step=99), "global_step drifted"),
        (lambda state: state.update(max_steps=301), "max_steps drifted"),
        (lambda state: state.update(train_batch_size=16), "train_batch_size drifted"),
        (lambda state: state.update(logging_steps=2), "logging_steps drifted"),
        (lambda state: state.update(save_steps=50), "save_steps drifted"),
        (
            lambda state: state["log_history"].pop(),
            "exactly 100 step logs",
        ),
        (
            lambda state: state["log_history"][-1].update(step=99),
            "step-log order drifted",
        ),
        (lambda state: state.update(epoch=float("nan")), "non-finite"),
    ],
)
def test_writer_strictly_validates_trainer_state_metadata(
    tmp_path,
    mutate,
    message,
):
    writer, plan = make_writer(tmp_path)
    checkpoint = Path(plan["checkpoint_paths"]["100"])
    make_checkpoint(checkpoint, 100)
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mutate(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        writer.record_checkpoint_saved(100)
    assert writer.events == []


def test_writer_requires_valid_regular_trainer_state_file(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint = Path(plan["checkpoint_paths"]["100"])
    make_checkpoint(checkpoint, 100)
    state_path = checkpoint / "trainer_state.json"
    state_path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        writer.record_checkpoint_saved(100)
    state_path.unlink()
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        writer.record_checkpoint_saved(100)


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "rng_state_0.pth",
        "model.safetensors",
        "pytorch_model-00001-of-00002.bin",
    ],
)
def test_writer_still_rejects_resumable_or_full_model_state(
    tmp_path,
    forbidden_name,
):
    writer, plan = make_writer(tmp_path)
    checkpoint = Path(plan["checkpoint_paths"]["100"])
    make_checkpoint(checkpoint, 100)
    (checkpoint / forbidden_name).write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="forbidden state"):
        writer.record_checkpoint_saved(100)


@pytest.mark.parametrize(
    ("records", "step_logs", "message"),
    [
        (rollouts(799), logs(), "800 rollout"),
        (rollouts(), logs(99), "100 trainer step"),
    ],
)
def test_step_100_export_rejects_incomplete_evidence(
    tmp_path, records, step_logs, message
):
    writer, plan = make_writer(tmp_path)
    make_checkpoint(Path(plan["checkpoint_paths"]["100"]), 100)
    writer.record_checkpoint_saved(100)
    with pytest.raises(ValueError, match=message):
        writer.export_step_100(
            rollout_records=records,
            trainer_step_logs=step_logs,
        )
    assert not Path(plan["step_100_export"]["directory"]).exists()


def test_retention_refuses_eviction_without_milestone_or_missing_survivor(tmp_path):
    writer, plan = make_writer(tmp_path)
    checkpoint_100 = Path(plan["checkpoint_paths"]["100"])
    checkpoint_200 = Path(plan["checkpoint_paths"]["200"])
    checkpoint_300 = Path(plan["checkpoint_paths"]["300"])
    make_checkpoint(checkpoint_100, 100)
    writer.record_checkpoint_saved(100)
    writer.export_step_100(rollout_records=rollouts(), trainer_step_logs=logs())
    make_checkpoint(checkpoint_200, 200)
    writer.record_checkpoint_saved(200)
    make_checkpoint(checkpoint_300, 300)
    writer.record_checkpoint_saved(300)

    with pytest.raises(RuntimeError, match="was not evicted"):
        writer.verify_retention_after_step_300()
    shutil.rmtree(checkpoint_100)
    shutil.rmtree(checkpoint_200)
    with pytest.raises(RuntimeError, match="200 or 300 is missing"):
        writer.verify_retention_after_step_300()


def test_final_adapter_and_complete_bundle_publish_atomically(tmp_path):
    writer, plan, source_adapter = make_ready_writer(tmp_path)
    final_weights = b"final adapter at step 300"
    handoff = writer.save_and_validate_final_adapter(
        save_adapter_fn=save_fake_final_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=len(final_weights),
    )

    assert handoff["source_adapter_unchanged"]
    assert handoff["validation"]["file_count"] == 10
    assert handoff["validation"]["differs_from_starting_adapter"]
    report = writer.publish_completed_bundle(
        rollout_records=rollouts(2_400),
        trainer_step_logs=logs(300),
        preflight_report={"status": "passed", "cuda_imports_performed": False},
        config_settings=locked_config(),
        run_summary=run_summary(),
        disk_usage_fn=lambda _: SimpleNamespace(free=4 * 1024**3),
    )

    final = Path(plan["final_output_dir"])
    assert report["status"] == "completed"
    assert report["published_atomically"]
    assert final.is_dir()
    assert not Path(plan["staging_dir"]).exists()
    assert {path.name for path in final.iterdir()} == {
        "trainer",
        "milestones",
        "final-adapter",
        "rollouts.jsonl",
        "trainer-log.json",
        "manifest.json",
    }
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["rollout_records"] == 2_400
    assert manifest["trainer_step_logs"] == 300
    assert len(manifest["checkpoint_events_before_publication"]) == 8
    assert manifest["run_summary"]["version"] == "grpo-full-run-summary-v1"
    assert manifest["run_summary_validation"]["status"] == "passed"
    assert manifest["run_summary_validation"]["peak_allocated_bytes"] == 30
    assert writer.snapshot()["status"] == "completed_and_published"
    assert writer.snapshot()["event_count"] == 10


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(version="wrong"), "unexpected.*version"),
        (
            lambda value: value["training"].update(train_seconds=0.0),
            "positive training duration",
        ),
        (
            lambda value: value["training"].update(train_seconds=float("nan")),
            "non-finite",
        ),
        (
            lambda value: value["model_audit"].update(
                lora_sha256_after=value["model_audit"]["lora_sha256_before"]
            ),
            "did not record a LoRA change",
        ),
        (
            lambda value: value["model_audit"]["parameter_values_after"].update(
                all_trainable_values_finite=False
            ),
            "did not prove finite",
        ),
        (
            lambda value: value["resources"]["cuda_after_train"].update(
                driver_free_bytes=799
            ),
            "does not sum",
        ),
        (
            lambda value: value["resources"]["cuda_after_train"].update(
                device_name="Other GPU"
            ),
            "identity drifted",
        ),
        (
            lambda value: value["resources"].update(
                preflight_disk_free_bytes=3 * 1024**3 - 1
            ),
            "disk was below",
        ),
        (
            lambda value: value["training"]["train_metrics"].update(
                bad=object()
            ),
            "non-JSON value",
        ),
    ],
)
def test_run_summary_validator_fails_closed(mutate, message):
    value = copy.deepcopy(run_summary())
    mutate(value)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_full_run_summary(value)


def test_final_handoff_rejects_source_drift_or_unchanged_final_adapter(tmp_path):
    writer, _plan, source_adapter = make_ready_writer(tmp_path)
    source_adapter.write_bytes(b"drifted source")
    with pytest.raises(RuntimeError, match="drifted before final save"):
        writer.save_and_validate_final_adapter(
            save_adapter_fn=save_fake_final_adapter,
            source_adapter_file=source_adapter,
            expected_adapter_model_bytes=len(b"final adapter at step 300"),
        )

    writer, _plan, source_adapter = make_ready_writer(tmp_path / "unchanged")

    def save_unchanged(path):
        save_fake_final_adapter(path, weights=b"starting SFT adapter")

    with pytest.raises(ValueError, match="byte-identical"):
        writer.save_and_validate_final_adapter(
            save_adapter_fn=save_unchanged,
            source_adapter_file=source_adapter,
            expected_adapter_model_bytes=len(b"starting SFT adapter"),
        )


@pytest.mark.parametrize(
    ("rollout_count", "log_count", "message"),
    [
        (2_399, 300, "2,400 rollouts"),
        (2_400, 299, "300 trainer step"),
    ],
)
def test_bundle_handoff_rejects_incomplete_evidence(
    tmp_path, rollout_count, log_count, message
):
    writer, _plan, source_adapter = make_ready_writer(tmp_path)
    writer.save_and_validate_final_adapter(
        save_adapter_fn=save_fake_final_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=len(b"final adapter at step 300"),
    )
    with pytest.raises(ValueError, match=message):
        writer.publish_completed_bundle(
            rollout_records=rollouts(rollout_count),
            trainer_step_logs=logs(log_count),
            preflight_report={"status": "passed"},
            config_settings=locked_config(),
            run_summary=run_summary(),
            disk_usage_fn=lambda _: SimpleNamespace(free=4 * 1024**3),
        )


def test_bundle_handoff_rejects_size_disk_or_config_drift(tmp_path):
    writer, _plan, source_adapter = make_ready_writer(tmp_path)
    writer.save_and_validate_final_adapter(
        save_adapter_fn=save_fake_final_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=len(b"final adapter at step 300"),
    )
    with pytest.raises(ValueError, match="size bound"):
        writer.publish_completed_bundle(
            rollout_records=rollouts(2_400),
            trainer_step_logs=logs(300),
            preflight_report={"status": "passed"},
            config_settings=locked_config(),
            run_summary=run_summary(),
            disk_usage_fn=lambda _: SimpleNamespace(free=4 * 1024**3),
            maximum_bundle_bytes=1,
        )
    assert not Path(writer.plan["final_output_dir"]).exists()

    writer, _plan, source_adapter = make_ready_writer(tmp_path / "disk")
    writer.save_and_validate_final_adapter(
        save_adapter_fn=save_fake_final_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=len(b"final adapter at step 300"),
    )
    with pytest.raises(ValueError, match="insufficient free disk"):
        writer.publish_completed_bundle(
            rollout_records=rollouts(2_400),
            trainer_step_logs=logs(300),
            preflight_report={"status": "passed"},
            config_settings=locked_config(),
            run_summary=run_summary(),
            disk_usage_fn=lambda _: SimpleNamespace(free=3 * 1024**3 - 1),
        )

    writer, _plan, source_adapter = make_ready_writer(tmp_path / "config")
    writer.save_and_validate_final_adapter(
        save_adapter_fn=save_fake_final_adapter,
        source_adapter_file=source_adapter,
        expected_adapter_model_bytes=len(b"final adapter at step 300"),
    )
    config = locked_config()
    config["save_total_limit"] = 3
    with pytest.raises(ValueError, match="config drifted for save_total_limit"):
        writer.publish_completed_bundle(
            rollout_records=rollouts(2_400),
            trainer_step_logs=logs(300),
            preflight_report={"status": "passed"},
            config_settings=config,
            run_summary=run_summary(),
            disk_usage_fn=lambda _: SimpleNamespace(free=4 * 1024**3),
        )
