"""CPU-only tests for the Linux detached launcher and worker wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.grpo_detached_control import (
    PROCESS_LOG_FILE,
    WORKLOAD_RESULT_FILE,
    DetachedRunControlWriter,
    create_detached_control,
)
from training.grpo_detached_runtime import (
    WORKER_INTERNAL_ERROR_EXIT_CODE,
    launch_detached_worker,
    linux_process_identity_matches,
    linux_process_start_token,
    run_detached_worker,
)

COMMIT = "b" * 40
PREPARED = "2026-08-10T07:00:00Z"
STARTED = "2026-08-10T07:00:01Z"
ENDED = "2026-08-10T07:10:00Z"


def make_control(tmp_path: Path, workload_command=None):
    final = tmp_path / "grpo-first-300"
    workload = workload_command or [
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    ]
    control_dir = tmp_path / "grpo-first-300-control"
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
        worker_cwd=tmp_path,
    )
    return writer, final


def sequential_clock(*timestamps: str):
    values = iter(timestamps)
    return lambda: next(values)


def write_fake_proc_record(
    proc_root: Path,
    *,
    pid: int,
    boot_id: str,
    start_ticks: str,
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    boot_dir = proc_root / "sys/kernel/random"
    boot_dir.mkdir(parents=True)
    fields_after_name = ["S", *(["0"] * 18), start_ticks, "0"]
    (process_dir / "stat").write_text(
        f"{pid} (python worker) {' '.join(fields_after_name)}\n",
        encoding="utf-8",
    )
    (boot_dir / "boot_id").write_text(f"{boot_id}\n", encoding="utf-8")


def test_linux_process_token_binds_pid_start_tick_and_boot(tmp_path):
    proc_root = tmp_path / "proc"
    write_fake_proc_record(
        proc_root,
        pid=4242,
        boot_id="boot-abc",
        start_ticks="98765",
    )

    assert linux_process_start_token(4242, proc_root=proc_root) == (
        "linux-proc-v1:boot-abc:98765"
    )

    (proc_root / "4242/stat").write_text(
        "4242 malformed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed"):
        linux_process_start_token(4242, proc_root=proc_root)


def test_linux_identity_returns_false_for_missing_or_reused_process(monkeypatch):
    monkeypatch.setattr(
        "training.grpo_detached_runtime.linux_process_start_token",
        lambda pid: "linux-proc-v1:boot:222",
    )
    assert linux_process_identity_matches(4242, "linux-proc-v1:boot:222")
    assert not linux_process_identity_matches(4242, "linux-proc-v1:boot:111")

    def missing(_pid):
        raise FileNotFoundError

    monkeypatch.setattr(
        "training.grpo_detached_runtime.linux_process_start_token",
        missing,
    )
    assert not linux_process_identity_matches(4242, "any-token")


class FakeProcess:
    def __init__(self, *, pid: int = 4242, return_code=None):
        self.pid = pid
        self.return_code = return_code
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def wait(self, timeout):
        assert timeout == 5
        self.waited = True
        return self.return_code


def test_launcher_uses_detached_subprocess_contract_and_verifies_handshake(tmp_path):
    captured = {}
    process = FakeProcess()

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        control_dir = Path(command[command.index("--control-dir") + 1])
        DetachedRunControlWriter(control_dir=control_dir).record_worker_started(
            pid=process.pid,
            process_start_token="token-4242",
            started_at=STARTED,
        )
        return process

    report = launch_detached_worker(
        final_output_dir=tmp_path / "grpo-first-300",
        expected_commit=COMMIT,
        repo_root=tmp_path,
        workload_command=[sys.executable, "-c", "raise SystemExit(7)"],
        popen_fn=fake_popen,
        process_identity_fn=lambda pid, token: (pid, token)
        == (4242, "token-4242"),
        clock_fn=lambda: PREPARED,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: pytest.fail("handshake should be immediate"),
    )

    kwargs = captured["kwargs"]
    assert captured["command"][-4:] == [
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    ]
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["close_fds"] is True
    assert kwargs["cwd"] == str(tmp_path.resolve())
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert kwargs["stdout"].closed
    assert report["status"] == "launched"
    assert report["process_identity_matches_at_launch"] is True
    assert report["workload_dispatched"] is True
    assert Path(report["process_log"]).name == PROCESS_LOG_FILE


def test_launcher_terminates_worker_that_never_records_identity(tmp_path):
    process = FakeProcess()
    monotonic_values = iter([0.0, 1.0])

    with pytest.raises(TimeoutError, match="before timeout"):
        launch_detached_worker(
            final_output_dir=tmp_path / "grpo-first-300",
            expected_commit=COMMIT,
            repo_root=tmp_path,
            workload_command=[sys.executable, "-c", "pass"],
            startup_timeout_seconds=0.5,
            popen_fn=lambda command, **kwargs: process,
            clock_fn=lambda: PREPARED,
            monotonic_fn=lambda: next(monotonic_values),
            sleep_fn=lambda _seconds: None,
        )

    assert process.terminated
    assert process.waited


def test_launcher_rejects_worker_exit_before_identity_record(tmp_path):
    process = FakeProcess(return_code=9)

    with pytest.raises(RuntimeError, match="exited before recording"):
        launch_detached_worker(
            final_output_dir=tmp_path / "grpo-first-300",
            expected_commit=COMMIT,
            repo_root=tmp_path,
            workload_command=[sys.executable, "-c", "pass"],
            popen_fn=lambda command, **kwargs: process,
            clock_fn=lambda: PREPARED,
            monotonic_fn=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )


def test_worker_runs_harmless_real_child_and_records_nonzero_exit(tmp_path):
    writer, _final = make_control(tmp_path)

    return_code = run_detached_worker(
        control_dir=writer.control_dir,
        workload_command=[sys.executable, "-c", "raise SystemExit(7)"],
        process_start_token_fn=lambda pid: f"test-token-{pid}",
        clock_fn=sequential_clock(STARTED, ENDED),
    )

    snapshot = writer.snapshot()
    assert return_code == 7
    assert snapshot["status"] == "failed"
    assert snapshot["exit"]["exit_code"] == 7
    assert snapshot["worker"]["process_start_token"].startswith("test-token-")


def test_worker_converts_missing_success_evidence_to_internal_failure(
    tmp_path,
    capsys,
):
    successful_command = [sys.executable, "-c", "pass"]
    writer, _final = make_control(
        tmp_path,
        workload_command=successful_command,
    )

    return_code = run_detached_worker(
        control_dir=writer.control_dir,
        workload_command=successful_command,
        process_start_token_fn=lambda pid: f"test-token-{pid}",
        clock_fn=sequential_clock(STARTED, ENDED),
    )

    snapshot = writer.snapshot()
    assert return_code == WORKER_INTERNAL_ERROR_EXIT_CODE
    assert snapshot["status"] == "failed"
    assert snapshot["exit"]["exit_code"] == WORKER_INTERNAL_ERROR_EXIT_CODE
    assert "workload-result.json" in capsys.readouterr().err


def test_worker_rejects_workload_command_drift_before_start(tmp_path):
    writer, _final = make_control(tmp_path)

    with pytest.raises(RuntimeError, match="drifted from launch evidence"):
        run_detached_worker(
            control_dir=writer.control_dir,
            workload_command=[sys.executable, "-c", "pass"],
            process_start_token_fn=lambda pid: f"test-token-{pid}",
            clock_fn=sequential_clock(STARTED, ENDED),
        )

    assert writer.snapshot()["status"] == "prepared"


def test_worker_accepts_complete_fake_success_handoff(tmp_path):
    writer, final = make_control(tmp_path, workload_command=["fake-grpo"])

    def fake_success_runner(command, **kwargs):
        assert command == ["fake-grpo"]
        control_dir = Path(kwargs["env"]["GRPO_DETACHED_CONTROL_DIR"])
        live_writer = DetachedRunControlWriter(control_dir=control_dir)
        live_writer.record_progress(
            optimizer_step=300,
            rollout_records=2_400,
            scalar_logs=300,
            updated_at="2026-08-10T07:09:00Z",
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
        Path(kwargs["env"]["GRPO_DETACHED_WORKLOAD_RESULT"]).write_text(
            json.dumps(
                {
                    "status": "passed",
                    "published": True,
                    "final_output_dir": str(final.resolve()),
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    return_code = run_detached_worker(
        control_dir=writer.control_dir,
        workload_command=["fake-grpo"],
        command_runner=fake_success_runner,
        process_start_token_fn=lambda pid: f"test-token-{pid}",
        clock_fn=sequential_clock(STARTED, ENDED),
    )

    snapshot = writer.snapshot()
    assert return_code == 0
    assert snapshot["status"] == "completed"
    assert snapshot["progress"]["optimizer_step"] == 300
    assert (writer.control_dir / WORKLOAD_RESULT_FILE).is_file()
