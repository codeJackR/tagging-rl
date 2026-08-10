#!/usr/bin/env python3
"""Linux detached-process launcher and worker for the guarded GRPO run.

This module has no GPU imports and does not know how to start GRPO.  It only
executes an explicit argument-list workload under the evidence contract in
``training.grpo_detached_control``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from training.grpo_detached_control import (
    CONTROL_SUFFIX,
    EXIT_FILE,
    PROCESS_LOG_FILE,
    WORKER_FILE,
    WORKLOAD_RESULT_FILE,
    DetachedRunControlWriter,
    create_detached_control,
)

DETACHED_RUNTIME_VERSION = "grpo-full-run-detached-runtime-v1"
WORKER_INTERNAL_ERROR_EXIT_CODE = 70
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_STARTUP_POLL_SECONDS = 0.05


def utc_now() -> str:
    """Return a sortable UTC timestamp accepted by the control contract."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _normalize_command(command: Sequence[str], *, label: str) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError(f"{label} must be a nonempty argument list")
    normalized = list(command)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{label} contains an invalid argument")
    if any("\n" in value or "\x00" in value for value in normalized):
        raise ValueError(f"{label} contains a control character")
    return normalized


def _strip_argument_separator(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return _normalize_command(normalized, label="detached workload command")


def _declared_workload_command(worker_command: object) -> list[str]:
    command = _normalize_command(
        worker_command,
        label="declared detached worker command",
    )
    try:
        separator = command.index("--")
    except ValueError as exc:
        raise ValueError("declared detached worker command has no workload") from exc
    return _normalize_command(
        command[separator + 1 :],
        label="declared detached workload command",
    )


def _read_json_object(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected regular JSON workload result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("detached workload result must be a JSON object")
    return value


def linux_process_start_token(
    pid: int,
    *,
    proc_root: str | Path = "/proc",
) -> str:
    """Bind a Linux PID to its kernel start tick and current boot identity."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise ValueError("Linux process PID must be an integer greater than one")
    proc_root = Path(proc_root)
    stat_text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").strip()
    closing_parenthesis = stat_text.rfind(")")
    if not stat_text.startswith(f"{pid} (") or closing_parenthesis < 0:
        raise ValueError("Linux process stat record is malformed")
    fields_after_name = stat_text[closing_parenthesis + 1 :].split()
    if len(fields_after_name) <= 19:
        raise ValueError("Linux process stat record lacks start time")
    start_ticks = fields_after_name[19]
    if not start_ticks.isdigit():
        raise ValueError("Linux process start time is invalid")
    boot_id = (proc_root / "sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
    if not boot_id or ":" in boot_id:
        raise ValueError("Linux boot identity is invalid")
    return f"linux-proc-v1:{boot_id}:{start_ticks}"


def linux_process_identity_matches(pid: int, expected_token: str) -> bool:
    """Return false when the PID disappeared or now belongs to another process."""
    try:
        return linux_process_start_token(pid) == expected_token
    except (OSError, ValueError):
        return False


def run_detached_worker(
    *,
    control_dir: str | Path,
    workload_command: Sequence[str],
    command_runner: Callable[..., object] = subprocess.run,
    process_start_token_fn: Callable[[int], str] = linux_process_start_token,
    clock_fn: Callable[[], str] = utc_now,
) -> int:
    """Run one child workload and always try to leave terminal evidence."""
    command = _normalize_command(
        workload_command,
        label="detached workload command",
    )
    writer = DetachedRunControlWriter(control_dir=control_dir)
    if command != _declared_workload_command(writer.launch["worker_command"]):
        raise RuntimeError("detached workload command drifted from launch evidence")
    worker_recorded = False
    try:
        pid = os.getpid()
        writer.record_worker_started(
            pid=pid,
            process_start_token=process_start_token_fn(pid),
            started_at=clock_fn(),
        )
        worker_recorded = True
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["GRPO_DETACHED_CONTROL_DIR"] = str(writer.control_dir)
        environment["GRPO_DETACHED_WORKLOAD_RESULT"] = str(
            writer.control_dir / WORKLOAD_RESULT_FILE
        )
        completed = command_runner(
            command,
            cwd=writer.launch["subprocess_contract"]["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return_code = getattr(completed, "returncode", None)
        if not isinstance(return_code, int) or isinstance(return_code, bool):
            raise RuntimeError("detached workload returned no integer exit code")
        bridge_report = None
        if return_code == 0:
            bridge_report = _read_json_object(
                writer.control_dir / WORKLOAD_RESULT_FILE
            )
        writer.record_worker_exit(
            exit_code=return_code,
            ended_at=clock_fn(),
            bridge_report=bridge_report,
        )
        return return_code
    except Exception:
        traceback.print_exc()
        exit_path = writer.control_dir / EXIT_FILE
        if (
            worker_recorded
            and not exit_path.exists()
            and not writer.final_output.exists()
        ):
            try:
                writer.record_worker_exit(
                    exit_code=WORKER_INTERNAL_ERROR_EXIT_CODE,
                    ended_at=clock_fn(),
                )
            except Exception:
                traceback.print_exc()
        return WORKER_INTERNAL_ERROR_EXIT_CODE


def _terminate_failed_start(process: object) -> None:
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        terminate()
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def launch_detached_worker(
    *,
    final_output_dir: str | Path,
    expected_commit: str,
    repo_root: str | Path,
    workload_command: Sequence[str],
    python_executable: str = sys.executable,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    popen_fn: Callable[..., object] = subprocess.Popen,
    process_identity_fn: Callable[[int, str], bool] = (
        linux_process_identity_matches
    ),
    clock_fn: Callable[[], str] = utc_now,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """Create control evidence, detach the worker and verify its startup."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError("detached launcher repository root does not exist")
    if (
        not isinstance(startup_timeout_seconds, (int, float))
        or isinstance(startup_timeout_seconds, bool)
        or startup_timeout_seconds <= 0
    ):
        raise ValueError("detached startup timeout must be positive")
    workload = _normalize_command(
        workload_command,
        label="detached workload command",
    )
    final_output = Path(final_output_dir).resolve()
    control_dir = final_output.with_name(final_output.name + CONTROL_SUFFIX)
    worker_command = [
        python_executable,
        "-m",
        "training.grpo_detached_runtime",
        "worker",
        "--control-dir",
        str(control_dir),
        "--",
        *workload,
    ]
    writer = create_detached_control(
        final_output_dir=final_output,
        expected_commit=expected_commit,
        worker_command=worker_command,
        prepared_at=clock_fn(),
        worker_cwd=repo_root,
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process_log = writer.control_dir / PROCESS_LOG_FILE
    with process_log.open("xb", buffering=0) as log_handle:
        process = popen_fn(
            worker_command,
            cwd=str(repo_root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        _terminate_failed_start(process)
        raise RuntimeError("detached launcher received an invalid worker PID")

    deadline = monotonic_fn() + float(startup_timeout_seconds)
    worker_path = writer.control_dir / WORKER_FILE
    while not worker_path.exists():
        if getattr(process, "poll")() is not None:
            raise RuntimeError(
                "detached worker exited before recording its identity; "
                f"inspect {process_log}"
            )
        if monotonic_fn() >= deadline:
            _terminate_failed_start(process)
            raise TimeoutError(
                "detached worker did not record its identity before timeout"
            )
        sleep_fn(DEFAULT_STARTUP_POLL_SECONDS)

    snapshot = writer.snapshot()
    worker = snapshot["worker"]
    if worker["pid"] != pid:
        _terminate_failed_start(process)
        raise RuntimeError("detached worker PID disagrees with spawned process")
    process_alive = getattr(process, "poll")() is None
    identity_matches = process_identity_fn(
        worker["pid"], worker["process_start_token"]
    )
    if process_alive and not identity_matches:
        _terminate_failed_start(process)
        raise RuntimeError("detached worker process identity could not be verified")
    if not process_alive and snapshot["exit"] is None:
        raise RuntimeError(
            "detached worker disappeared without terminal evidence; "
            f"inspect {process_log}"
        )
    return {
        "version": DETACHED_RUNTIME_VERSION,
        "status": "launched",
        "pid": pid,
        "process_identity_matches_at_launch": identity_matches,
        "control_dir": str(writer.control_dir),
        "process_log": str(process_log),
        "worker_status_at_launch": snapshot["status"],
        "workload_dispatched": True,
    }


def monitor_detached_worker(*, control_dir: str | Path) -> dict:
    """Reopen control evidence and verify live Linux process identity."""
    writer = DetachedRunControlWriter(control_dir=control_dir)
    return writer.snapshot(process_identity_fn=linux_process_identity_matches)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--final-output-dir", required=True)
    launch.add_argument("--expected-commit", required=True)
    launch.add_argument("--repo-root", required=True)
    launch.add_argument("--python-executable", default=sys.executable)
    launch.add_argument("workload_command", nargs=argparse.REMAINDER)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--control-dir", required=True)
    worker.add_argument("workload_command", nargs=argparse.REMAINDER)

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--control-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "launch":
        report = launch_detached_worker(
            final_output_dir=args.final_output_dir,
            expected_commit=args.expected_commit,
            repo_root=args.repo_root,
            workload_command=_strip_argument_separator(args.workload_command),
            python_executable=args.python_executable,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return run_detached_worker(
            control_dir=args.control_dir,
            workload_command=_strip_argument_separator(args.workload_command),
        )
    if args.command == "monitor":
        report = monitor_detached_worker(control_dir=args.control_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise RuntimeError("unknown detached-runtime command")


if __name__ == "__main__":
    raise SystemExit(main())
