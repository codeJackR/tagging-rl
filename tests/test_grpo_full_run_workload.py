"""CPU-only tests for the detached production full-run workload boundary."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training.grpo_detached_control import (
    WORKLOAD_RESULT_FILE,
    create_detached_control,
)
from training.grpo_full_run_workload import (
    FULL_RUN_WORKLOAD_VERSION,
    run_production_full_run_workload,
)
from training.train_grpo import (
    DEFAULT_ADAPTER,
    DEFAULT_FULL_RUN_DATA,
    DEFAULT_FULL_RUN_MANIFEST,
    DEFAULT_FULL_RUN_OUTPUT_DIR,
    DEFAULT_SELECTION_MANIFEST,
)

COMMIT = "c" * 40
OTHER_COMMIT = "d" * 40
PREPARED = "2026-08-10T08:00:00Z"
STARTED = "2026-08-10T08:00:01Z"


def progress_clock():
    current = datetime(2026, 8, 10, 8, 0, 1, tzinfo=timezone.utc)

    def tick():
        nonlocal current
        current += timedelta(seconds=1)
        return current.isoformat().replace("+00:00", "Z")

    return tick


def make_control(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    final = repo / DEFAULT_FULL_RUN_OUTPUT_DIR
    control_dir = final.with_name(final.name + "-control")
    workload = [
        sys.executable,
        "-m",
        "training.grpo_full_run_workload",
        "--control-dir",
        str(control_dir),
        "--repo-root",
        str(repo),
        "--expected-commit",
        COMMIT,
    ]
    writer = create_detached_control(
        final_output_dir=final,
        expected_commit=COMMIT,
        worker_command=[
            sys.executable,
            "-m",
            "training.grpo_detached_runtime",
            "worker",
            "--control-dir",
            str(control_dir),
            "--",
            *workload,
        ],
        prepared_at=PREPARED,
        worker_cwd=repo,
    )
    writer.record_worker_started(
        pid=4242,
        process_start_token="test-token-4242",
        started_at=STARTED,
    )
    environment = {
        "GRPO_DETACHED_CONTROL_DIR": str(writer.control_dir),
        "GRPO_DETACHED_WORKLOAD_RESULT": str(
            writer.control_dir / WORKLOAD_RESULT_FILE
        ),
    }
    return repo, final, writer, environment


def make_fakes(repo: Path, final: Path, *, bridge_mode="complete"):
    observed = {"preflight_calls": 0, "bridge_calls": 0, "call_order": []}
    training_data = repo / DEFAULT_FULL_RUN_DATA
    training_data.parent.mkdir(parents=True, exist_ok=True)
    training_data.write_text("fake\n", encoding="utf-8")
    adapter_path = repo / DEFAULT_ADAPTER
    adapter_path.mkdir(parents=True)
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_file.write_bytes(b"locked adapter")

    def preflight_fn(**kwargs):
        observed["preflight_calls"] += 1
        observed["call_order"].append("preflight")
        observed["preflight_kwargs"] = kwargs
        return {
            "status": "passed",
            "cuda_imports_performed": False,
            "git": {"commit": COMMIT},
            "pool": {"data_path": str(training_data.resolve())},
            "sft_lock": {
                "adapter_path": str(adapter_path.resolve()),
                "adapter_file": str(adapter_file.resolve()),
            },
            "output": {"path": str(final.resolve())},
        }

    def bridge_fn(**kwargs):
        observed["bridge_calls"] += 1
        observed["call_order"].append("bridge")
        observed["bridge_kwargs"] = kwargs
        if bridge_mode == "failed":
            return {"status": "failed", "published": False}
        final_step = 299 if bridge_mode == "incomplete-progress" else 300
        for step in range(1, final_step + 1):
            kwargs["progress_callback"](
                optimizer_step=step,
                rollout_records=step * 8,
                scalar_logs=step,
            )
        final.mkdir()
        (final / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_summary_validation": {"status": "passed"},
                }
            ),
            encoding="utf-8",
        )
        report = {
            "version": "fake-runtime-bridge-v1",
            "status": "passed",
            "published": True,
        }
        if bridge_mode == "non-json":
            report["bad_metric"] = math.nan
        return report

    return preflight_fn, bridge_fn, observed


def test_workload_runs_preflight_bridge_progress_and_atomic_handoff(tmp_path):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, observed = make_fakes(repo, final)

    result = run_production_full_run_workload(
        control_dir=writer.control_dir,
        repo_root=repo,
        expected_commit=COMMIT,
        environment=environment,
        preflight_fn=preflight_fn,
        bridge_fn=bridge_fn,
        clock_fn=progress_clock(),
    )

    result_path = writer.control_dir / WORKLOAD_RESULT_FILE
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    snapshot = writer.snapshot()
    assert result == persisted
    assert result["detached_workload"]["version"] == FULL_RUN_WORKLOAD_VERSION
    assert result["detached_workload"]["preflight_passed_before_bridge"]
    assert result["final_output_dir"] == str(final.resolve())
    assert observed["preflight_calls"] == observed["bridge_calls"] == 1
    assert observed["call_order"] == ["preflight", "bridge"]
    assert observed["preflight_kwargs"]["training_data"] == DEFAULT_FULL_RUN_DATA
    assert observed["preflight_kwargs"]["pool_manifest"] == (
        DEFAULT_FULL_RUN_MANIFEST
    )
    assert observed["preflight_kwargs"]["selection_manifest"] == (
        DEFAULT_SELECTION_MANIFEST
    )
    assert observed["preflight_kwargs"]["adapter"] == DEFAULT_ADAPTER
    assert observed["preflight_kwargs"]["output_dir"] == (
        DEFAULT_FULL_RUN_OUTPUT_DIR
    )
    assert observed["preflight_kwargs"]["minimum_free_bytes"] == 3 * 1024**3
    assert observed["bridge_kwargs"]["preflight_report"]["status"] == "passed"
    assert snapshot["progress"]["optimizer_step"] == 300
    assert snapshot["progress"]["rollout_records"] == 2_400
    assert snapshot["exit"] is None
    assert not list(writer.control_dir.glob(".workload-result.json.*.tmp"))

    writer.record_worker_exit(
        exit_code=0,
        ended_at="2026-08-10T08:06:00Z",
        bridge_report=result,
    )
    assert writer.snapshot()["status"] == "completed"


def test_workload_rejects_commit_drift_before_preflight(tmp_path):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, observed = make_fakes(repo, final)

    with pytest.raises(RuntimeError, match="commit disagrees"):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=OTHER_COMMIT,
            environment=environment,
            preflight_fn=preflight_fn,
            bridge_fn=bridge_fn,
        )
    assert observed["preflight_calls"] == observed["bridge_calls"] == 0


@pytest.mark.parametrize(
    ("environment_key", "message"),
    [
        ("GRPO_DETACHED_CONTROL_DIR", "control environment path drifted"),
        ("GRPO_DETACHED_WORKLOAD_RESULT", "result environment path drifted"),
    ],
)
def test_workload_rejects_environment_path_drift_before_preflight(
    tmp_path,
    environment_key,
    message,
):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, observed = make_fakes(repo, final)
    environment[environment_key] = str(tmp_path / "wrong")

    with pytest.raises(RuntimeError, match=message):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=COMMIT,
            environment=environment,
            preflight_fn=preflight_fn,
            bridge_fn=bridge_fn,
        )
    assert observed["preflight_calls"] == observed["bridge_calls"] == 0


def test_workload_refuses_failed_preflight_before_bridge(tmp_path):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, observed = make_fakes(repo, final)

    def failed_preflight(**kwargs):
        observed["preflight_calls"] += 1
        return {"status": "failed", "cuda_imports_performed": False}

    with pytest.raises(RuntimeError, match="preflight did not pass"):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=COMMIT,
            environment=environment,
            preflight_fn=failed_preflight,
            bridge_fn=bridge_fn,
        )
    assert observed["bridge_calls"] == 0
    assert not (writer.control_dir / WORKLOAD_RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report, final: report.update(cuda_imports_performed=True),
            "preflight imported CUDA",
        ),
        (
            lambda report, final: report["git"].update(commit=OTHER_COMMIT),
            "preflight commit drifted",
        ),
        (
            lambda report, final: report["output"].update(
                path=str(final.with_name("wrong-output"))
            ),
            "preflight output drifted",
        ),
    ],
)
def test_workload_rejects_preflight_lineage_drift_before_bridge(
    tmp_path,
    mutation,
    message,
):
    repo, final, writer, environment = make_control(tmp_path)
    valid_preflight, bridge_fn, observed = make_fakes(repo, final)

    def drifted_preflight(**kwargs):
        report = valid_preflight(**kwargs)
        mutation(report, final)
        return report

    with pytest.raises(RuntimeError, match=message):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=COMMIT,
            environment=environment,
            preflight_fn=drifted_preflight,
            bridge_fn=bridge_fn,
        )
    assert observed["preflight_calls"] == 1
    assert observed["bridge_calls"] == 0
    assert not (writer.control_dir / WORKLOAD_RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("bridge_mode", "message"),
    [
        ("failed", "bridge did not publish success"),
        ("incomplete-progress", "lacks complete detached progress"),
        ("non-json", "Out of range float values"),
    ],
)
def test_workload_never_writes_handoff_for_invalid_bridge_result(
    tmp_path,
    bridge_mode,
    message,
):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, _observed = make_fakes(
        repo,
        final,
        bridge_mode=bridge_mode,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=COMMIT,
            environment=environment,
            preflight_fn=preflight_fn,
            bridge_fn=bridge_fn,
            clock_fn=progress_clock(),
        )
    assert not (writer.control_dir / WORKLOAD_RESULT_FILE).exists()
    assert not list(writer.control_dir.glob(".workload-result.json.*.tmp"))


def test_workload_refuses_existing_result_before_preflight(tmp_path):
    repo, final, writer, environment = make_control(tmp_path)
    preflight_fn, bridge_fn, observed = make_fakes(repo, final)
    (writer.control_dir / WORKLOAD_RESULT_FILE).write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="result already exists"):
        run_production_full_run_workload(
            control_dir=writer.control_dir,
            repo_root=repo,
            expected_commit=COMMIT,
            environment=environment,
            preflight_fn=preflight_fn,
            bridge_fn=bridge_fn,
        )
    assert observed["preflight_calls"] == observed["bridge_calls"] == 0
