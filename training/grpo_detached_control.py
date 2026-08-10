"""CPU-only control-plane contract for the detached 300-step GRPO run."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

DETACHED_CONTROL_VERSION = "grpo-full-run-detached-control-v1"
EXPECTED_STEPS = 300
GENERATIONS_PER_STEP = 8
CONTROL_SUFFIX = "-control"
LAUNCH_FILE = "launch.json"
WORKER_FILE = "worker.json"
PROGRESS_FILE = "progress.json"
EXIT_FILE = "exit.json"
BRIDGE_REPORT_FILE = "bridge-report.json"
PROCESS_LOG_FILE = "process.log"
WORKLOAD_RESULT_FILE = "workload-result.json"


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must use UTC")
    return value


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _read_json_object(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected regular JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_replace_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("detached worker command must be a nonempty argument list")
    normalized = list(command)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("detached worker command contains an invalid argument")
    if any("\n" in value or "\x00" in value for value in normalized):
        raise ValueError("detached worker command contains a control character")
    return normalized


def create_detached_control(
    *,
    final_output_dir: str | Path,
    expected_commit: str,
    worker_command: Sequence[str],
    prepared_at: str,
    worker_cwd: str | Path | None = None,
) -> "DetachedRunControlWriter":
    """Reserve the control directory and write its immutable launch contract."""
    final_output = Path(final_output_dir).resolve()
    if final_output.name != "grpo-first-300":
        raise ValueError("detached control requires the reserved full-run output")
    if not final_output.parent.is_dir():
        raise FileNotFoundError("detached control output parent does not exist")
    control_dir = final_output.with_name(final_output.name + CONTROL_SUFFIX)
    if final_output.exists():
        raise FileExistsError("final full-run output already exists")
    if control_dir.exists():
        raise FileExistsError("detached full-run control directory already exists")
    staging_collisions = sorted(
        final_output.parent.glob(f".{final_output.name}.staging-*")
    )
    if staging_collisions:
        raise FileExistsError("stale full-run staging output already exists")
    if not _is_git_commit(expected_commit):
        raise ValueError("detached control expected commit is invalid")
    command = _validate_command(worker_command)
    prepared_at = _parse_utc_timestamp(prepared_at, label="prepared_at")
    worker_cwd = (
        Path.cwd().resolve()
        if worker_cwd is None
        else Path(worker_cwd).resolve()
    )
    if not worker_cwd.is_dir():
        raise FileNotFoundError("detached worker working directory does not exist")

    control_dir.mkdir(mode=0o700)
    launch = {
        "version": DETACHED_CONTROL_VERSION,
        "status": "prepared",
        "expected_commit": expected_commit,
        "prepared_at": prepared_at,
        "final_output_dir": str(final_output),
        "control_dir": str(control_dir),
        "worker_command": command,
        "subprocess_contract": {
            "shell": False,
            "start_new_session": True,
            "stdin": "DEVNULL",
            "stdout": str(control_dir / PROCESS_LOG_FILE),
            "stderr": "STDOUT",
            "unbuffered_python": True,
            "cwd": str(worker_cwd),
        },
        "evidence_paths": {
            "worker": str(control_dir / WORKER_FILE),
            "progress": str(control_dir / PROGRESS_FILE),
            "exit": str(control_dir / EXIT_FILE),
            "bridge_report": str(control_dir / BRIDGE_REPORT_FILE),
            "process_log": str(control_dir / PROCESS_LOG_FILE),
            "workload_result": str(control_dir / WORKLOAD_RESULT_FILE),
        },
        "success_requires": {
            "exit_code": 0,
            "optimizer_steps": EXPECTED_STEPS,
            "rollout_records": EXPECTED_STEPS * GENERATIONS_PER_STEP,
            "scalar_logs": EXPECTED_STEPS,
            "bridge_status": "passed",
            "bridge_published": True,
            "final_manifest_status": "completed",
        },
        "failure_policy": {
            "final_output_must_remain_absent": True,
            "private_staging_may_survive_for_diagnosis": True,
            "missing_exit_record_is_not_success": True,
        },
    }
    try:
        _write_exclusive_json(control_dir / LAUNCH_FILE, launch)
    except Exception:
        control_dir.rmdir()
        raise
    return DetachedRunControlWriter(control_dir=control_dir)


class DetachedRunControlWriter:
    """Write and validate detached worker identity, progress and terminal state."""

    def __init__(self, *, control_dir: str | Path):
        self.control_dir = Path(control_dir).resolve()
        self.launch = _read_json_object(self.control_dir / LAUNCH_FILE)
        if self.launch.get("version") != DETACHED_CONTROL_VERSION:
            raise ValueError("unexpected detached-control version")
        if Path(self.launch.get("control_dir", "")) != self.control_dir:
            raise ValueError("detached-control path disagrees with launch evidence")
        self.final_output = Path(self.launch["final_output_dir"])

    def record_worker_started(
        self,
        *,
        pid: int,
        process_start_token: str,
        started_at: str,
    ) -> dict:
        if (self.control_dir / WORKER_FILE).exists():
            raise FileExistsError("detached worker identity already exists")
        if (self.control_dir / EXIT_FILE).exists():
            raise RuntimeError("detached worker cannot start after exit")
        if not isinstance(pid, int) or pid <= 1:
            raise ValueError("detached worker PID must be greater than one")
        if not isinstance(process_start_token, str) or not process_start_token:
            raise ValueError("detached worker has no process start token")
        started_at = _parse_utc_timestamp(started_at, label="started_at")
        if _timestamp_value(started_at) < _timestamp_value(
            self.launch["prepared_at"]
        ):
            raise ValueError("detached worker started before launch preparation")
        worker = {
            "version": DETACHED_CONTROL_VERSION,
            "status": "started",
            "pid": pid,
            "process_start_token": process_start_token,
            "started_at": started_at,
            "expected_commit": self.launch["expected_commit"],
        }
        _write_exclusive_json(self.control_dir / WORKER_FILE, worker)
        return dict(worker)

    def record_progress(
        self,
        *,
        optimizer_step: int,
        rollout_records: int,
        scalar_logs: int,
        updated_at: str,
    ) -> dict:
        _read_json_object(self.control_dir / WORKER_FILE)
        if (self.control_dir / EXIT_FILE).exists():
            raise RuntimeError("detached progress cannot change after exit")
        if not isinstance(optimizer_step, int) or not 0 <= optimizer_step <= 300:
            raise ValueError("detached optimizer step is outside 0..300")
        if rollout_records != optimizer_step * GENERATIONS_PER_STEP:
            raise ValueError("detached rollout progress is not step times eight")
        if scalar_logs != optimizer_step:
            raise ValueError("detached scalar-log progress disagrees with step")
        updated_at = _parse_utc_timestamp(updated_at, label="updated_at")
        progress_path = self.control_dir / PROGRESS_FILE
        previous = (
            _read_json_object(progress_path) if progress_path.exists() else None
        )
        worker = _read_json_object(self.control_dir / WORKER_FILE)
        if _timestamp_value(updated_at) <= _timestamp_value(worker["started_at"]):
            raise ValueError("detached progress predates worker start")
        if previous is not None:
            if optimizer_step <= previous.get("optimizer_step", -1):
                raise ValueError("detached optimizer progress is not increasing")
            if _timestamp_value(updated_at) <= _timestamp_value(
                previous["updated_at"]
            ):
                raise ValueError("detached progress timestamp is not increasing")
        progress = {
            "version": DETACHED_CONTROL_VERSION,
            "status": "running" if optimizer_step < 300 else "training_complete",
            "optimizer_step": optimizer_step,
            "rollout_records": rollout_records,
            "scalar_logs": scalar_logs,
            "updated_at": updated_at,
        }
        _atomic_replace_json(progress_path, progress)
        return dict(progress)

    def record_worker_exit(
        self,
        *,
        exit_code: int,
        ended_at: str,
        bridge_report: dict | None = None,
    ) -> dict:
        worker = _read_json_object(self.control_dir / WORKER_FILE)
        if (self.control_dir / EXIT_FILE).exists():
            raise FileExistsError("detached worker exit already exists")
        if not isinstance(exit_code, int):
            raise ValueError("detached worker exit code is invalid")
        ended_at = _parse_utc_timestamp(ended_at, label="ended_at")
        if _timestamp_value(ended_at) <= _timestamp_value(worker["started_at"]):
            raise ValueError("detached worker ended before it started")
        progress_path = self.control_dir / PROGRESS_FILE
        if progress_path.exists():
            latest_progress = _read_json_object(progress_path)
            if _timestamp_value(ended_at) <= _timestamp_value(
                latest_progress["updated_at"]
            ):
                raise ValueError("detached worker exit predates latest progress")

        if exit_code == 0:
            progress = _read_json_object(self.control_dir / PROGRESS_FILE)
            if (
                progress.get("optimizer_step") != EXPECTED_STEPS
                or progress.get("rollout_records")
                != EXPECTED_STEPS * GENERATIONS_PER_STEP
                or progress.get("scalar_logs") != EXPECTED_STEPS
            ):
                raise RuntimeError("successful detached exit lacks complete progress")
            if not isinstance(bridge_report, dict):
                raise TypeError("successful detached exit requires a bridge report")
            if bridge_report.get("status") != "passed" or bridge_report.get(
                "published"
            ) is not True:
                raise ValueError("successful detached bridge report did not publish")
            if Path(bridge_report.get("final_output_dir", "")) != self.final_output:
                raise ValueError("detached bridge final-output path drifted")
            final_manifest = _read_json_object(self.final_output / "manifest.json")
            if final_manifest.get("status") != "completed":
                raise ValueError("detached final manifest is not completed")
            if final_manifest.get("run_summary_validation", {}).get(
                "status"
            ) != "passed":
                raise ValueError("detached final manifest lacks run-summary validation")
            _write_exclusive_json(
                self.control_dir / BRIDGE_REPORT_FILE, bridge_report
            )
            status = "completed"
        else:
            if bridge_report is not None:
                raise ValueError("failed detached exit may not claim a bridge report")
            if self.final_output.exists():
                raise RuntimeError("failed detached worker exposed a final output")
            status = "failed"

        exit_evidence = {
            "version": DETACHED_CONTROL_VERSION,
            "status": status,
            "pid": worker["pid"],
            "process_start_token": worker["process_start_token"],
            "exit_code": exit_code,
            "ended_at": ended_at,
            "final_output_exists": self.final_output.is_dir(),
            "bridge_report_written": exit_code == 0,
        }
        _write_exclusive_json(self.control_dir / EXIT_FILE, exit_evidence)
        return dict(exit_evidence)

    def snapshot(
        self,
        *,
        process_identity_fn: Callable[[int, str], bool] | None = None,
    ) -> dict:
        worker_path = self.control_dir / WORKER_FILE
        progress_path = self.control_dir / PROGRESS_FILE
        exit_path = self.control_dir / EXIT_FILE
        worker = _read_json_object(worker_path) if worker_path.exists() else None
        progress = (
            _read_json_object(progress_path) if progress_path.exists() else None
        )
        exit_evidence = _read_json_object(exit_path) if exit_path.exists() else None

        if worker is not None:
            if (
                worker.get("version") != DETACHED_CONTROL_VERSION
                or worker.get("status") != "started"
                or worker.get("expected_commit") != self.launch["expected_commit"]
                or not isinstance(worker.get("pid"), int)
                or worker["pid"] <= 1
                or not isinstance(worker.get("process_start_token"), str)
                or not worker["process_start_token"]
            ):
                raise ValueError("detached worker identity evidence drifted")
            started_at = _parse_utc_timestamp(
                worker.get("started_at"), label="started_at"
            )
            if _timestamp_value(started_at) < _timestamp_value(
                self.launch["prepared_at"]
            ):
                raise ValueError("detached worker timestamp drifted")
        if progress is not None:
            if worker is None:
                raise ValueError("detached progress exists without worker identity")
            step = progress.get("optimizer_step")
            expected_status = "training_complete" if step == 300 else "running"
            if (
                progress.get("version") != DETACHED_CONTROL_VERSION
                or progress.get("status") != expected_status
                or not isinstance(step, int)
                or not 0 <= step <= 300
                or progress.get("rollout_records") != step * GENERATIONS_PER_STEP
                or progress.get("scalar_logs") != step
            ):
                raise ValueError("detached progress evidence drifted")
            updated_at = _parse_utc_timestamp(
                progress.get("updated_at"), label="updated_at"
            )
            if _timestamp_value(updated_at) <= _timestamp_value(
                worker["started_at"]
            ):
                raise ValueError("detached progress timestamp drifted")
        if exit_evidence is not None:
            if worker is None:
                raise ValueError("detached exit exists without worker identity")
            code = exit_evidence.get("exit_code")
            expected_status = "completed" if code == 0 else "failed"
            if (
                exit_evidence.get("version") != DETACHED_CONTROL_VERSION
                or exit_evidence.get("status") != expected_status
                or exit_evidence.get("pid") != worker["pid"]
                or exit_evidence.get("process_start_token")
                != worker["process_start_token"]
                or not isinstance(code, int)
                or exit_evidence.get("final_output_exists") is not (code == 0)
                or exit_evidence.get("bridge_report_written") is not (code == 0)
            ):
                raise ValueError("detached exit evidence drifted")
            ended_at = _parse_utc_timestamp(
                exit_evidence.get("ended_at"), label="ended_at"
            )
            if _timestamp_value(ended_at) <= _timestamp_value(
                worker["started_at"]
            ):
                raise ValueError("detached exit timestamp drifted")
            if progress is not None and _timestamp_value(
                ended_at
            ) <= _timestamp_value(progress["updated_at"]):
                raise ValueError("detached exit timestamp drifted")
            bridge_path = self.control_dir / BRIDGE_REPORT_FILE
            if code == 0:
                if (
                    progress is None
                    or progress.get("optimizer_step") != EXPECTED_STEPS
                    or progress.get("rollout_records")
                    != EXPECTED_STEPS * GENERATIONS_PER_STEP
                    or progress.get("scalar_logs") != EXPECTED_STEPS
                ):
                    raise ValueError(
                        "completed exit lacks complete progress evidence"
                    )
                bridge = _read_json_object(bridge_path)
                if (
                    bridge.get("status") != "passed"
                    or bridge.get("published") is not True
                    or Path(bridge.get("final_output_dir", ""))
                    != self.final_output
                    or not self.final_output.is_dir()
                ):
                    raise ValueError("completed detached bridge evidence drifted")
                final_manifest = _read_json_object(
                    self.final_output / "manifest.json"
                )
                if (
                    final_manifest.get("status") != "completed"
                    or final_manifest.get("run_summary_validation", {}).get(
                        "status"
                    )
                    != "passed"
                ):
                    raise ValueError("completed detached final manifest drifted")
            elif bridge_path.exists() or self.final_output.exists():
                raise ValueError("failed detached run exposes success evidence")

        if exit_evidence is not None:
            status = exit_evidence["status"]
            process_identity_matches = False
        elif worker is None:
            status = "prepared"
            process_identity_matches = None
        else:
            if process_identity_fn is None:
                status = "started_unverified"
                process_identity_matches = None
            else:
                process_identity_matches = bool(
                    process_identity_fn(
                        worker["pid"], worker["process_start_token"]
                    )
                )
                status = (
                    "running"
                    if process_identity_matches
                    else "worker_missing_without_exit_record"
                )

        return {
            "version": DETACHED_CONTROL_VERSION,
            "status": status,
            "launch": dict(self.launch),
            "worker": worker,
            "progress": progress,
            "exit": exit_evidence,
            "process_identity_matches": process_identity_matches,
            "final_output_exists": self.final_output.is_dir(),
            "missing_exit_is_success": False,
        }
