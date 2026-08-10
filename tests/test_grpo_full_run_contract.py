"""CPU-only contract tests for the first 300-step GRPO run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import training.train_grpo as train_grpo_module

from training.train_grpo import (
    DEFAULT_ADAPTER,
    DEFAULT_FULL_RUN_DATA,
    DEFAULT_FULL_RUN_MANIFEST,
    DEFAULT_FULL_RUN_OUTPUT_DIR,
    DEFAULT_MINIMUM_FREE_GIB,
    DEFAULT_SELECTION_MANIFEST,
    FULL_RUN_300_CONTRACT_VERSION,
    FULL_RUN_CHECKPOINT_STEPS,
    FULL_RUN_SAVE_STEPS,
    FULL_RUN_SAVE_TOTAL_LIMIT,
    FULL_RUN_STEPS,
    FULL_RUN_WARMUP_RATIO,
    LOCKED_LEARNING_RATE,
    LOCKED_REWARD_WEIGHTS,
    full_run_300_contract,
    grpo_full_run_300_config_kwargs,
    inspect_gpu_idle_state,
    main,
    parse_args,
    run_full_run_300_preflight,
    validate_full_run_300_launch_args,
    verify_full_run_pool,
)
from training.grpo_full_run_artifacts import FULL_RUN_LIFECYCLE_VERSION

ROOT = Path(__file__).resolve().parent.parent


def test_full_run_config_locks_duration_warmup_and_seeded_shuffle(tmp_path):
    config = grpo_full_run_300_config_kwargs(output_dir=tmp_path)

    assert config["max_steps"] == FULL_RUN_STEPS == 300
    assert config["num_generations"] == 8
    assert config["per_device_train_batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 1
    assert config["shuffle_dataset"] is True
    assert config["seed"] == config["data_seed"] == 42
    assert config["warmup_ratio"] == FULL_RUN_WARMUP_RATIO == 0.1
    assert config["lr_scheduler_type"] == "cosine"
    assert config["learning_rate"] == LOCKED_LEARNING_RATE == 5e-6
    assert config["reward_weights"] == list(LOCKED_REWARD_WEIGHTS)


def test_full_run_config_logs_scalars_without_completion_table_flood(tmp_path):
    config = grpo_full_run_300_config_kwargs(output_dir=tmp_path)

    assert config["logging_strategy"] == "steps"
    assert config["logging_steps"] == 1
    assert config["logging_first_step"] is True
    assert config["log_completions"] is False
    assert config["report_to"] == "none"


def test_full_run_config_bounds_model_only_checkpoints(tmp_path):
    config = grpo_full_run_300_config_kwargs(output_dir=tmp_path)

    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == FULL_RUN_SAVE_STEPS == 100
    assert config["save_total_limit"] == FULL_RUN_SAVE_TOTAL_LIMIT == 2
    assert config["save_only_model"] is True


def test_full_run_contract_records_rollouts_retention_and_disk_gate(tmp_path):
    contract = full_run_300_contract(output_dir=tmp_path)

    assert contract["version"] == FULL_RUN_300_CONTRACT_VERSION
    assert contract["status"] == "locked_not_launchable"
    assert contract["training"] == {
        "optimizer_steps": 300,
        "generations_per_step": 8,
        "expected_rollouts": 2_400,
        "prompt_data": DEFAULT_FULL_RUN_DATA,
        "prompt_order": "seeded_shuffle",
        "seed": 42,
        "data_seed": 42,
        "warmup_steps": 30,
    }
    assert contract["checkpointing"]["events"] == list(
        FULL_RUN_CHECKPOINT_STEPS
    )
    assert contract["checkpointing"]["lifecycle_version"] == (
        FULL_RUN_LIFECYCLE_VERSION
    )
    assert contract["checkpointing"]["step_100_evidence_required_before_eviction"]
    assert contract["checkpointing"]["final_adapter_must_be_retained"]
    assert not contract["checkpointing"]["optimizer_state_saved"]
    assert contract["checkpointing"]["final_adapter_directory"] == str(
        (Path(tmp_path) / "final-adapter").resolve()
    )
    assert contract["reporting"]["local_trainer_log_required"]
    assert not contract["reporting"]["external_reporting_required"]
    assert contract["resources"]["minimum_free_gib_before_launch"] == (
        DEFAULT_MINIMUM_FREE_GIB
    )
    assert contract["resources"]["minimum_free_bytes_before_launch"] == 3 * 1024**3
    assert contract["resources"]["detached_launch_required"]


def test_full_run_rejects_reward_weight_drift(tmp_path):
    with pytest.raises(ValueError, match="reward weights must remain locked"):
        grpo_full_run_300_config_kwargs(
            output_dir=tmp_path,
            reward_weights=(1.0, 1.0, 1.0),
        )


def full_run_args(tmp_path, *extra):
    return parse_args(
        [
            "--full-run-300",
            "--repo-root",
            str(tmp_path),
            "--expected-commit",
            "a" * 40,
            "--output-dir",
            DEFAULT_FULL_RUN_OUTPUT_DIR,
            *extra,
        ]
    )


def test_full_run_launch_validator_accepts_only_locked_surface(tmp_path):
    report = validate_full_run_300_launch_args(full_run_args(tmp_path))

    assert report["passed"]
    assert report["expected_commit"] == "a" * 40
    assert report["training_data"] == str(
        (tmp_path / DEFAULT_FULL_RUN_DATA).resolve()
    )
    assert report["pool_manifest"] == str(
        (tmp_path / DEFAULT_FULL_RUN_MANIFEST).resolve()
    )
    assert report["selection_manifest"] == str(
        (tmp_path / DEFAULT_SELECTION_MANIFEST).resolve()
    )
    assert report["adapter"] == str((tmp_path / DEFAULT_ADAPTER).resolve())
    assert report["reserved_output"] == str(
        (tmp_path / DEFAULT_FULL_RUN_OUTPUT_DIR).resolve()
    )
    assert report["minimum_free_gib"] == 3.0
    assert report["standalone_report_forbidden"]
    assert report["contract_status"] == "locked_not_launchable"
    assert not report["training_dispatch_enabled"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--full-run-data", "data/other.jsonl"), "locked training_data path"),
        (("--full-run-manifest", "other.json"), "locked pool_manifest path"),
        (("--selection-manifest", "other.json"), "locked selection_manifest path"),
        (("--adapter", "other-adapter"), "locked adapter path"),
        (("--output-dir", "runs/other"), "locked output path"),
        (("--minimum-free-gib", "2.99"), "disk floor"),
        (("--minimum-free-gib", "nan"), "disk floor"),
        (("--report-file", "report.json"), "report-file is forbidden"),
    ],
)
def test_full_run_launch_validator_rejects_surface_drift(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        validate_full_run_300_launch_args(full_run_args(tmp_path, *extra))


@pytest.mark.parametrize("commit", ["abc123", "A" * 40, "g" * 40])
def test_full_run_launch_validator_requires_exact_lowercase_commit(tmp_path, commit):
    args = full_run_args(tmp_path)
    args.expected_commit = commit

    with pytest.raises(SystemExit, match="full lowercase"):
        validate_full_run_300_launch_args(args)


def test_full_run_mode_prints_preflight_then_stops_without_dispatch(
    tmp_path, monkeypatch, capsys
):
    observed = {}

    def fake_full_preflight(**kwargs):
        observed.update(kwargs)
        return {
            "version": "grpo-full-run-300-preflight-v1",
            "status": "passed",
            "cuda_imports_performed": False,
            "model_loaded": False,
            "trainer_constructed": False,
            "training_dispatch_enabled": False,
        }

    monkeypatch.setattr(
        train_grpo_module,
        "run_full_run_300_preflight",
        fake_full_preflight,
    )
    monkeypatch.setattr(
        train_grpo_module,
        "run_preflight",
        lambda **_: pytest.fail("smoke preflight must not run in full-run mode"),
    )

    result = main(
        [
            "--full-run-300",
            "--repo-root",
            str(tmp_path),
            "--expected-commit",
            "a" * 40,
            "--output-dir",
            DEFAULT_FULL_RUN_OUTPUT_DIR,
        ]
    )

    assert result == 0
    assert observed["repo_root"] == str(tmp_path)
    assert observed["training_data"] == DEFAULT_FULL_RUN_DATA
    assert observed["pool_manifest"] == DEFAULT_FULL_RUN_MANIFEST
    assert observed["minimum_free_bytes"] == 3 * 1024**3
    assert observed["expected_commit"] == "a" * 40
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["preflight_only"]
    assert report["launch_control"]["passed"]
    assert not report["launch_control"]["training_dispatch_enabled"]
    assert "remain intentionally unavailable" in report["stop_reason"]
    assert not report["cuda_imports_performed"]
    assert not report["model_loaded"]
    assert not report["trainer_constructed"]
    assert not report["training_dispatch_enabled"]


def test_full_run_mode_prints_nothing_when_preflight_fails(
    tmp_path, monkeypatch, capsys
):
    def fail_preflight(**_):
        raise RuntimeError("preflight failed closed")

    monkeypatch.setattr(
        train_grpo_module,
        "run_full_run_300_preflight",
        fail_preflight,
    )

    with pytest.raises(RuntimeError, match="failed closed"):
        main(
            [
                "--full-run-300",
                "--repo-root",
                str(tmp_path),
                "--expected-commit",
                "a" * 40,
                "--output-dir",
                DEFAULT_FULL_RUN_OUTPUT_DIR,
            ]
        )
    assert capsys.readouterr().out == ""


def test_full_run_and_smoke_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--full-run-300", "--five-step-smoke"])


def test_real_full_run_pool_matches_locked_lineage():
    report = verify_full_run_pool(
        repo_root=ROOT,
        data_path=ROOT / DEFAULT_FULL_RUN_DATA,
        manifest_path=ROOT / DEFAULT_FULL_RUN_MANIFEST,
    )

    assert report["rows"] == 1_565
    assert report["data_bytes"] == 3_467_347
    assert report["family_cap"] == 4
    assert report["selection_seed"] == 42
    assert len(report["manifest_invariants"]) == 5
    assert all(report["manifest_invariants"].values())


def test_full_run_pool_rejects_hash_drift():
    with pytest.raises(RuntimeError, match="data checksum"):
        verify_full_run_pool(
            repo_root=ROOT,
            data_path=ROOT / DEFAULT_FULL_RUN_DATA,
            manifest_path=ROOT / DEFAULT_FULL_RUN_MANIFEST,
            expected_data_sha256="0" * 64,
        )

    with pytest.raises(RuntimeError, match="manifest checksum"):
        verify_full_run_pool(
            repo_root=ROOT,
            data_path=ROOT / DEFAULT_FULL_RUN_DATA,
            manifest_path=ROOT / DEFAULT_FULL_RUN_MANIFEST,
            expected_manifest_sha256="0" * 64,
        )


def fake_nvidia_smi(stdout):
    def run(command, *, check, capture_output, text):
        assert command[0] == "nvidia-smi"
        assert check and capture_output and text
        return SimpleNamespace(stdout=stdout)

    return run


def test_gpu_idle_inspection_is_cpu_only_and_thresholded():
    report = inspect_gpu_idle_state(
        command_runner=fake_nvidia_smi(
            "0, NVIDIA GeForce RTX 3090, 264, 0, 53\n"
        )
    )

    assert report["idle"]
    assert report["memory_idle"]
    assert report["utilization_idle"]
    assert report["memory_used_mib"] == 264
    assert report["utilization_percent"] == 0
    assert not report["cuda_imports_performed"]

    busy = inspect_gpu_idle_state(
        command_runner=fake_nvidia_smi(
            "0, NVIDIA GeForce RTX 3090, 4096, 80, 60\n"
        )
    )
    assert not busy["idle"]
    assert not busy["memory_idle"]
    assert not busy["utilization_idle"]


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "0, GPU A, 0, 0, 30\n1, GPU B, 0, 0, 30\n",
        "0, GPU, unknown, 0, 30\n",
    ],
)
def test_gpu_idle_inspection_rejects_ambiguous_or_malformed_output(stdout):
    with pytest.raises(RuntimeError):
        inspect_gpu_idle_state(command_runner=fake_nvidia_smi(stdout))


def passing_full_preflight_kwargs(tmp_path):
    return {
        "repo_root": ROOT,
        "training_data": DEFAULT_FULL_RUN_DATA,
        "pool_manifest": DEFAULT_FULL_RUN_MANIFEST,
        "selection_manifest": "unused-selection.json",
        "adapter": "unused-adapter",
        "output_dir": tmp_path / "grpo-first-300",
        "minimum_free_bytes": 3 * 1024**3,
        "expected_commit": "a" * 40,
        "git_state_fn": lambda _: {
            "commit": "a" * 40,
            "tracked_worktree_dirty": False,
            "index_dirty": False,
        },
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
        "gpu_state_fn": lambda: {
            "idle": True,
            "memory_used_mib": 264,
            "utilization_percent": 0,
            "cuda_imports_performed": False,
        },
        "sft_lock_fn": lambda **_: {"status": "locked"},
    }


def test_full_run_preflight_passes_without_creating_or_importing_cuda(tmp_path):
    report = run_full_run_300_preflight(
        **passing_full_preflight_kwargs(tmp_path)
    )

    assert report["status"] == "passed"
    assert report["pool"]["rows"] == 1_565
    assert report["disk"]["passes"]
    assert report["gpu"]["idle"]
    assert report["output"]["collision_free"]
    assert report["output"]["staging_collision_free"]
    assert not Path(report["output"]["path"]).exists()
    assert not report["cuda_imports_performed"]
    assert not report["model_loaded"]
    assert not report["trainer_constructed"]
    assert not report["training_dispatch_enabled"]


def test_full_run_preflight_rejects_git_output_and_staging_drift(tmp_path):
    kwargs = passing_full_preflight_kwargs(tmp_path)
    kwargs["git_state_fn"] = lambda _: {
        "commit": "b" * 40,
        "tracked_worktree_dirty": False,
        "index_dirty": False,
    }
    with pytest.raises(RuntimeError, match="expected full-run commit"):
        run_full_run_300_preflight(**kwargs)

    kwargs = passing_full_preflight_kwargs(tmp_path / "output-collision")
    Path(kwargs["output_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="output already exists"):
        run_full_run_300_preflight(**kwargs)

    kwargs = passing_full_preflight_kwargs(tmp_path / "staging-collision")
    output = Path(kwargs["output_dir"])
    output.parent.mkdir(parents=True)
    (output.parent / f".{output.name}.staging-stale").mkdir()
    with pytest.raises(FileExistsError, match="staging output already exists"):
        run_full_run_300_preflight(**kwargs)


def test_full_run_preflight_rejects_low_disk_or_busy_gpu(tmp_path):
    kwargs = passing_full_preflight_kwargs(tmp_path)
    kwargs["disk_usage_fn"] = lambda _: SimpleNamespace(free=3 * 1024**3 - 1)
    with pytest.raises(RuntimeError, match="insufficient free disk"):
        run_full_run_300_preflight(**kwargs)

    kwargs = passing_full_preflight_kwargs(tmp_path / "busy")
    kwargs["gpu_state_fn"] = lambda: {
        "idle": False,
        "cuda_imports_performed": False,
    }
    with pytest.raises(RuntimeError, match="GPU is not idle"):
        run_full_run_300_preflight(**kwargs)


def test_full_run_preflight_rejects_gpu_probe_that_imported_cuda(tmp_path):
    kwargs = passing_full_preflight_kwargs(tmp_path)
    kwargs["gpu_state_fn"] = lambda: {
        "idle": True,
        "cuda_imports_performed": True,
    }
    with pytest.raises(RuntimeError, match="unexpectedly imported CUDA"):
        run_full_run_300_preflight(**kwargs)
