"""CPU-only tests for detached launch evidence and monitoring semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.grpo_detached_control import (
    BRIDGE_REPORT_FILE,
    EXIT_FILE,
    PROGRESS_FILE,
    DetachedRunControlWriter,
    create_detached_control,
)

COMMIT = "a" * 40
PREPARED = "2026-08-10T05:00:00Z"
STARTED = "2026-08-10T05:00:01Z"


def make_control(tmp_path):
    final = tmp_path / "grpo-first-300"
    writer = create_detached_control(
        final_output_dir=final,
        expected_commit=COMMIT,
        worker_command=[
            "/venv/rl/bin/python",
            "-m",
            "training.full_run_worker",
            "--expected-commit",
            COMMIT,
        ],
        prepared_at=PREPARED,
    )
    return writer, final


def write_completed_final(final: Path):
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


def test_detached_control_records_complete_success_and_monitor_snapshot(tmp_path):
    writer, final = make_control(tmp_path)
    worker = writer.record_worker_started(
        pid=4242,
        process_start_token="linux-proc-start-123",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=1,
        rollout_records=8,
        scalar_logs=1,
        updated_at="2026-08-10T05:01:00Z",
    )
    writer.record_progress(
        optimizer_step=100,
        rollout_records=800,
        scalar_logs=100,
        updated_at="2026-08-10T05:30:00Z",
    )
    final_progress = writer.record_progress(
        optimizer_step=300,
        rollout_records=2_400,
        scalar_logs=300,
        updated_at="2026-08-10T06:30:00Z",
    )
    write_completed_final(final)
    bridge = {
        "status": "passed",
        "published": True,
        "final_output_dir": str(final.resolve()),
    }
    exit_evidence = writer.record_worker_exit(
        exit_code=0,
        ended_at="2026-08-10T06:31:00Z",
        bridge_report=bridge,
    )
    snapshot = writer.snapshot(
        process_identity_fn=lambda pid, token: pytest.fail(
            "completed workers must not be probed as running"
        )
    )

    assert worker["pid"] == 4242
    assert final_progress["status"] == "training_complete"
    assert exit_evidence["status"] == "completed"
    assert exit_evidence["exit_code"] == 0
    assert snapshot["status"] == "completed"
    assert snapshot["progress"]["optimizer_step"] == 300
    assert snapshot["process_identity_matches"] is False
    assert snapshot["final_output_exists"]
    assert not snapshot["missing_exit_is_success"]
    assert (writer.control_dir / BRIDGE_REPORT_FILE).is_file()
    assert (writer.control_dir / EXIT_FILE).is_file()


def test_monitor_distinguishes_running_from_missing_worker_without_exit(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=10,
        rollout_records=80,
        scalar_logs=10,
        updated_at="2026-08-10T05:05:00Z",
    )

    running = writer.snapshot(process_identity_fn=lambda pid, token: True)
    missing = writer.snapshot(process_identity_fn=lambda pid, token: False)
    unverified = writer.snapshot()

    assert running["status"] == "running"
    assert running["process_identity_matches"]
    assert missing["status"] == "worker_missing_without_exit_record"
    assert missing["process_identity_matches"] is False
    assert unverified["status"] == "started_unverified"
    assert unverified["process_identity_matches"] is None


def test_nonzero_exit_is_auditable_failure_and_may_preserve_staging(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    staging = final.parent / ".grpo-first-300.staging-diagnostic"
    staging.mkdir()

    exit_evidence = writer.record_worker_exit(
        exit_code=1,
        ended_at="2026-08-10T05:10:00Z",
    )

    assert exit_evidence["status"] == "failed"
    assert not exit_evidence["final_output_exists"]
    assert staging.is_dir()
    assert writer.snapshot()["status"] == "failed"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda values: values.update(expected_commit="abc"), "commit is invalid"),
        (lambda values: values.update(worker_command="python train.py"), "argument list"),
        (lambda values: values.update(worker_command=["python", "bad\narg"]), "control character"),
        (lambda values: values.update(prepared_at="not-a-time"), "UTC timestamp"),
        (
            lambda values: values.update(
                final_output_dir=Path(values["final_output_dir"]).with_name("other")
            ),
            "reserved full-run output",
        ),
    ],
)
def test_control_creation_rejects_invalid_launch_contract(tmp_path, change, message):
    values = {
        "final_output_dir": tmp_path / "grpo-first-300",
        "expected_commit": COMMIT,
        "worker_command": ["python", "worker.py"],
        "prepared_at": PREPARED,
    }
    change(values)

    with pytest.raises((TypeError, ValueError), match=message):
        create_detached_control(**values)


def test_control_creation_refuses_final_control_or_staging_collision(tmp_path):
    final = tmp_path / "grpo-first-300"
    final.mkdir()
    with pytest.raises(FileExistsError, match="final full-run output"):
        create_detached_control(
            final_output_dir=final,
            expected_commit=COMMIT,
            worker_command=["python", "worker.py"],
            prepared_at=PREPARED,
        )

    final.rmdir()
    control = tmp_path / "grpo-first-300-control"
    control.mkdir()
    with pytest.raises(FileExistsError, match="control directory"):
        create_detached_control(
            final_output_dir=final,
            expected_commit=COMMIT,
            worker_command=["python", "worker.py"],
            prepared_at=PREPARED,
        )

    control.rmdir()
    (tmp_path / ".grpo-first-300.staging-stale").mkdir()
    with pytest.raises(FileExistsError, match="stale full-run staging"):
        create_detached_control(
            final_output_dir=final,
            expected_commit=COMMIT,
            worker_command=["python", "worker.py"],
            prepared_at=PREPARED,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "optimizer_step": 2,
                "rollout_records": 8,
                "scalar_logs": 2,
                "updated_at": "2026-08-10T05:02:00Z",
            },
            "step times eight",
        ),
        (
            {
                "optimizer_step": 2,
                "rollout_records": 16,
                "scalar_logs": 1,
                "updated_at": "2026-08-10T05:02:00Z",
            },
            "scalar-log progress",
        ),
        (
            {
                "optimizer_step": 301,
                "rollout_records": 2_408,
                "scalar_logs": 301,
                "updated_at": "2026-08-10T05:02:00Z",
            },
            "outside 0..300",
        ),
    ],
)
def test_progress_rejects_count_and_step_drift(tmp_path, kwargs, message):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    with pytest.raises(ValueError, match=message):
        writer.record_progress(**kwargs)


def test_progress_is_strictly_monotonic_and_atomic(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=10,
        rollout_records=80,
        scalar_logs=10,
        updated_at="2026-08-10T05:05:00Z",
    )
    with pytest.raises(ValueError, match="not increasing"):
        writer.record_progress(
            optimizer_step=10,
            rollout_records=80,
            scalar_logs=10,
            updated_at="2026-08-10T05:06:00Z",
        )
    assert json.loads((writer.control_dir / PROGRESS_FILE).read_text())[
        "optimizer_step"
    ] == 10
    assert not list(writer.control_dir.glob(".progress.json.*.tmp"))


def test_zero_exit_requires_complete_progress_bridge_and_final_manifest(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=299,
        rollout_records=2_392,
        scalar_logs=299,
        updated_at="2026-08-10T06:29:00Z",
    )
    with pytest.raises(RuntimeError, match="lacks complete progress"):
        writer.record_worker_exit(
            exit_code=0,
            ended_at="2026-08-10T06:31:00Z",
            bridge_report={
                "status": "passed",
                "published": True,
                "final_output_dir": str(final),
            },
        )
    assert not (writer.control_dir / EXIT_FILE).exists()


def test_success_rejects_unvalidated_final_manifest(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=300,
        rollout_records=2_400,
        scalar_logs=300,
        updated_at="2026-08-10T06:30:00Z",
    )
    final.mkdir()
    (final / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="run-summary validation"):
        writer.record_worker_exit(
            exit_code=0,
            ended_at="2026-08-10T06:31:00Z",
            bridge_report={
                "status": "passed",
                "published": True,
                "final_output_dir": str(final.resolve()),
            },
        )
    assert not (writer.control_dir / EXIT_FILE).exists()
    assert not (writer.control_dir / BRIDGE_REPORT_FILE).exists()


def test_writer_reopens_existing_control_evidence(tmp_path):
    writer, _final = make_control(tmp_path)
    reopened = DetachedRunControlWriter(control_dir=writer.control_dir)

    assert reopened.launch == writer.launch
    assert reopened.snapshot()["status"] == "prepared"


def test_worker_start_cannot_predate_launch_preparation(tmp_path):
    writer, _final = make_control(tmp_path)

    with pytest.raises(ValueError, match="before launch preparation"):
        writer.record_worker_started(
            pid=4242,
            process_start_token="token",
            started_at="2026-08-10T04:59:59Z",
        )


def test_progress_requires_worker_and_must_follow_worker_start(tmp_path):
    writer, _final = make_control(tmp_path)
    with pytest.raises(FileNotFoundError, match="worker.json"):
        writer.record_progress(
            optimizer_step=1,
            rollout_records=8,
            scalar_logs=1,
            updated_at="2026-08-10T05:00:02Z",
        )

    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    with pytest.raises(ValueError, match="predates worker start"):
        writer.record_progress(
            optimizer_step=1,
            rollout_records=8,
            scalar_logs=1,
            updated_at=STARTED,
        )


def test_exit_must_follow_latest_progress(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=1,
        rollout_records=8,
        scalar_logs=1,
        updated_at="2026-08-10T05:05:00Z",
    )

    with pytest.raises(ValueError, match="predates latest progress"):
        writer.record_worker_exit(
            exit_code=1,
            ended_at="2026-08-10T05:04:00Z",
        )


def test_negative_signal_return_code_is_an_auditable_failure(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )

    exit_evidence = writer.record_worker_exit(
        exit_code=-15,
        ended_at="2026-08-10T05:02:00Z",
    )

    assert exit_evidence["status"] == "failed"
    assert exit_evidence["exit_code"] == -15
    assert writer.snapshot()["status"] == "failed"


def test_snapshot_rejects_tampered_progress_counts(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=10,
        rollout_records=80,
        scalar_logs=10,
        updated_at="2026-08-10T05:05:00Z",
    )
    progress_path = writer.control_dir / PROGRESS_FILE
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["rollout_records"] = 79
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="progress evidence drifted"):
        writer.snapshot()


def test_snapshot_rejects_tampered_exit_status(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_worker_exit(
        exit_code=1,
        ended_at="2026-08-10T05:02:00Z",
    )
    exit_path = writer.control_dir / EXIT_FILE
    exit_evidence = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_evidence["status"] = "completed"
    exit_path.write_text(json.dumps(exit_evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="exit evidence drifted"):
        writer.snapshot()


def test_snapshot_rejects_tampered_completed_bridge(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=300,
        rollout_records=2_400,
        scalar_logs=300,
        updated_at="2026-08-10T06:30:00Z",
    )
    write_completed_final(final)
    writer.record_worker_exit(
        exit_code=0,
        ended_at="2026-08-10T06:31:00Z",
        bridge_report={
            "status": "passed",
            "published": True,
            "final_output_dir": str(final.resolve()),
        },
    )
    bridge_path = writer.control_dir / BRIDGE_REPORT_FILE
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["published"] = False
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    with pytest.raises(ValueError, match="bridge evidence drifted"):
        writer.snapshot()


def test_snapshot_rejects_cross_file_timestamp_drift(tmp_path):
    writer, _final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=10,
        rollout_records=80,
        scalar_logs=10,
        updated_at="2026-08-10T05:05:00Z",
    )
    progress_path = writer.control_dir / PROGRESS_FILE
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["updated_at"] = PREPARED
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="progress timestamp drifted"):
        writer.snapshot()


def test_snapshot_requires_complete_progress_for_completed_exit(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=300,
        rollout_records=2_400,
        scalar_logs=300,
        updated_at="2026-08-10T06:30:00Z",
    )
    write_completed_final(final)
    writer.record_worker_exit(
        exit_code=0,
        ended_at="2026-08-10T06:31:00Z",
        bridge_report={
            "status": "passed",
            "published": True,
            "final_output_dir": str(final.resolve()),
        },
    )
    progress_path = writer.control_dir / PROGRESS_FILE
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        status="running",
        optimizer_step=299,
        rollout_records=2_392,
        scalar_logs=299,
    )
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="completed exit lacks complete progress"):
        writer.snapshot()


def test_snapshot_revalidates_completed_final_manifest(tmp_path):
    writer, final = make_control(tmp_path)
    writer.record_worker_started(
        pid=4242,
        process_start_token="token",
        started_at=STARTED,
    )
    writer.record_progress(
        optimizer_step=300,
        rollout_records=2_400,
        scalar_logs=300,
        updated_at="2026-08-10T06:30:00Z",
    )
    write_completed_final(final)
    writer.record_worker_exit(
        exit_code=0,
        ended_at="2026-08-10T06:31:00Z",
        bridge_report={
            "status": "passed",
            "published": True,
            "final_output_dir": str(final.resolve()),
        },
    )
    (final / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="final manifest drifted"):
        writer.snapshot()
