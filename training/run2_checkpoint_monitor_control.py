"""Fail-closed supervisor and Trainer callback for Run 2 checkpoint monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_checkpoint_monitor_runtime import FINAL_FILES, _checkpoint_identity


VERSION = "grpo-run2-checkpoint-monitor-control-v1"
MAX_CAPTURE_BYTES = 256 * 1024
TERMINATION_GRACE_SECONDS = 5.0


class CheckpointMonitorError(RuntimeError):
    """Raised after durable failure evidence has been published."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _normalize_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("monitor command must be a nonempty argument list")
    normalized = list(command)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("monitor command contains an invalid argument")
    if any("\n" in value or "\x00" in value for value in normalized):
        raise ValueError("monitor command contains a control character")
    return normalized


def _safe_monitor_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if any("confirmation" in part for part in resolved.parts):
        raise ValueError(f"{label} crosses the confirmation boundary")
    return resolved


def _stream_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    retained = raw[-MAX_CAPTURE_BYTES:]
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tail": retained.decode("utf-8", errors="replace"),
        "tail_bytes": len(retained),
        "tail_truncated": len(retained) != len(raw),
    }


def _published_file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_success_bundle(
    *,
    output_dir: str | Path,
    expected_mode: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete atomic bundle before the supervisor accepts it."""
    output_dir = _safe_monitor_path(output_dir, label="monitor output")
    if expected_mode not in {"production", "smoke"}:
        raise ValueError("expected monitor mode is invalid")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise FileNotFoundError("monitor did not publish a regular output directory")
    observed = {path.name for path in output_dir.iterdir()}
    if observed != set(FINAL_FILES):
        raise RuntimeError(f"monitor output inventory drifted: {sorted(observed)}")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "checkpoint_monitor_complete":
        raise RuntimeError("monitor manifest does not declare success")
    if manifest.get("mode") != expected_mode:
        raise RuntimeError("monitor mode drifted")
    if expected_mode == "smoke" and manifest.get("quality_evidence") is not False:
        raise RuntimeError("smoke output was incorrectly marked as quality evidence")
    invariants = manifest.get("invariants", {})
    required = {
        "fixed_development_membership": True,
        "greedy_and_repeated_sampled_complete": True,
        "original_and_dense_rewards_scored": True,
        "confirmation_data_used": False,
        "quality_abort_threshold_applied": False,
        "published_exclusively_and_atomically": True,
    }
    if any(invariants.get(key) is not value for key, value in required.items()):
        raise RuntimeError("monitor success invariants are incomplete")
    checkpoint_sha = manifest.get("checkpoint", {}).get("adapter_model_sha256")
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError("monitor evaluated a different checkpoint")
    file_map = {
        "greedy_predictions": "greedy.jsonl",
        "sampled_predictions": "sampled.jsonl",
        "scored_report": "report.json",
        "resource_report": "resource.json",
    }
    identities = manifest.get("files", {})
    for key, filename in file_map.items():
        if identities.get(key) != {
            "path": filename,
            **_published_file_identity(output_dir / filename),
        }:
            raise RuntimeError(f"monitor file identity drifted: {filename}")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "checkpoint_outputs_scored":
        raise RuntimeError("monitor report was not completely scored")
    return manifest


def _stop_process_group(process: subprocess.Popen[Any]) -> dict[str, Any]:
    evidence = {"terminate_sent": False, "kill_sent": False}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        evidence["terminate_sent"] = True
    except ProcessLookupError:
        return evidence
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
        return evidence
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            evidence["kill_sent"] = True
        except ProcessLookupError:
            pass
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
        return evidence


def run_supervised_monitor(
    *,
    command: Sequence[str],
    repo_root: str | Path,
    output_dir: str | Path,
    failure_path: str | Path,
    receipt_path: str | Path,
    timeout_seconds: float,
    expected_mode: str,
    expected_checkpoint_sha256: str | None = None,
    popen_fn: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    clock_fn: Callable[[], str] = _utc_now,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one evaluator and publish exactly one success receipt or failure."""
    normalized = _normalize_command(command)
    repo_root = Path(repo_root).resolve()
    output_dir = _safe_monitor_path(output_dir, label="monitor output")
    failure_path = _safe_monitor_path(failure_path, label="monitor failure")
    receipt_path = _safe_monitor_path(receipt_path, label="monitor receipt")
    if not repo_root.is_dir():
        raise FileNotFoundError("monitor repository root does not exist")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("monitor timeout must be positive")
    if output_dir.exists() or failure_path.exists() or receipt_path.exists():
        raise FileExistsError("monitor output, failure, or receipt already exists")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = clock_fn()
    started = monotonic_fn()
    with tempfile.TemporaryDirectory(prefix="run2-monitor-control-") as temporary:
        stdout_path = Path(temporary) / "stdout.log"
        stderr_path = Path(temporary) / "stderr.log"
        termination: dict[str, Any] = {"terminate_sent": False, "kill_sent": False}
        timed_out = False
        return_code: int | None = None
        validation_error: str | None = None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = popen_fn(
                normalized,
                cwd=repo_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination = _stop_process_group(process)
                return_code = process.returncode
        streams = {
            "stdout": _stream_evidence(stdout_path),
            "stderr": _stream_evidence(stderr_path),
        }
        manifest = None
        if not timed_out and return_code == 0:
            try:
                manifest = validate_success_bundle(
                    output_dir=output_dir,
                    expected_mode=expected_mode,
                    expected_checkpoint_sha256=expected_checkpoint_sha256,
                )
            except Exception as exc:  # durable failure is more useful than traceback only
                validation_error = f"{type(exc).__name__}: {exc}"
        elapsed = monotonic_fn() - started
        common = {
            "version": VERSION,
            "command": normalized,
            "repo_root": str(repo_root),
            "output_dir": str(output_dir),
            "expected_mode": expected_mode,
            "expected_checkpoint_sha256": expected_checkpoint_sha256,
            "started_at": started_at,
            "ended_at": clock_fn(),
            "elapsed_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "return_code": return_code,
            "timed_out": timed_out,
            "termination": termination,
            "streams": streams,
        }
        if manifest is None:
            failure = {
                **common,
                "status": "checkpoint_monitor_failed",
                "reason": (
                    "timeout" if timed_out else "nonzero_exit" if return_code != 0 else "invalid_success_bundle"
                ),
                "validation_error": validation_error,
                "training_must_abort": True,
                "quality_threshold_involved": False,
            }
            write_exclusive_atomic_json(failure_path, failure)
            raise CheckpointMonitorError(
                f"checkpoint monitor failed ({failure['reason']}): {failure_path}"
            )
        receipt = {
            **common,
            "status": "checkpoint_monitor_accepted",
            "manifest": _published_file_identity(output_dir / "manifest.json"),
            "checkpoint_adapter_sha256": manifest["checkpoint"]["adapter_model_sha256"],
            "training_may_continue": True,
            "quality_abort_threshold_applied": False,
        }
        write_exclusive_atomic_json(receipt_path, receipt)
        return receipt


class CheckpointMonitorCoordinator:
    """CPU-only ordered bridge from Trainer checkpoint saves to the supervisor."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        contract_path: str | Path,
        monitor_root: str | Path,
        command_builder: Callable[[int, Path, Path], Sequence[str]],
        runner: Callable[..., Mapping[str, Any]] = run_supervised_monitor,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.contract_path = Path(contract_path).resolve()
        self.monitor_root = _safe_monitor_path(monitor_root, label="monitor root")
        self.command_builder = command_builder
        self.runner = runner
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.expected_steps = tuple(self.contract["checkpoints"]["required_steps"])
        if self.contract["abort_policy"].get("quality_abort_enabled") is not False:
            raise RuntimeError("Phase F monitor must not enable a quality threshold")
        self._begun = False
        self._ended = False
        self._completed: list[int] = []
        self._receipts: dict[int, dict[str, Any]] = {}

    def on_train_begin(self, args: object, state: object) -> dict[str, Any]:
        if self._begun:
            raise RuntimeError("checkpoint monitor began more than once")
        actual = {
            "max_steps": int(getattr(args, "max_steps", -1)),
            "save_steps": int(getattr(args, "save_steps", -1)),
        }
        if actual != {"max_steps": 300, "save_steps": 100}:
            raise RuntimeError(f"checkpoint monitor trainer arguments drifted: {actual}")
        if int(getattr(state, "global_step", -1)) != 0:
            raise RuntimeError("checkpoint monitor must begin at step zero")
        self.monitor_root.mkdir(parents=True, exist_ok=False)
        self._begun = True
        return {"status": "ready", "checkpoint_steps": list(self.expected_steps)}

    def on_save(self, args: object, state: object) -> dict[str, Any]:
        if not self._begun or self._ended:
            raise RuntimeError("checkpoint monitor save occurred outside training")
        step = int(getattr(state, "global_step", -1))
        expected = self.expected_steps[len(self._completed)] if len(self._completed) < len(self.expected_steps) else None
        if step != expected:
            raise RuntimeError(f"checkpoint monitor expected step {expected}, found {step}")
        checkpoint = Path(getattr(args, "output_dir", "")).resolve() / f"checkpoint-{step}"
        checkpoint_identity = _checkpoint_identity(checkpoint)
        output = self.monitor_root / f"checkpoint-{step}"
        command = self.command_builder(step, checkpoint, output)
        receipt = dict(
            self.runner(
                command=command,
                repo_root=self.repo_root,
                output_dir=output,
                failure_path=self.monitor_root / f"checkpoint-{step}.failure.json",
                receipt_path=self.monitor_root / f"checkpoint-{step}.receipt.json",
                timeout_seconds=self.contract["runtime"]["timeout_seconds_per_checkpoint"],
                expected_mode="production",
                expected_checkpoint_sha256=checkpoint_identity["adapter_model_sha256"],
            )
        )
        if receipt.get("quality_abort_threshold_applied") is not False:
            raise RuntimeError("monitor receipt unexpectedly applied a quality threshold")
        self._completed.append(step)
        self._receipts[step] = receipt
        return receipt

    def on_train_end(self, state: object) -> dict[str, Any]:
        if not self._begun or self._ended:
            raise RuntimeError("checkpoint monitor ended outside active training")
        if int(getattr(state, "global_step", -1)) != 300 or self._completed != list(self.expected_steps):
            raise RuntimeError("checkpoint monitor did not accept all retained checkpoints")
        self._ended = True
        return {"status": "complete", "steps": list(self._completed), "receipts": self._receipts}


def make_checkpoint_monitor_callback_class(base_callback_class: type) -> type:
    """Create a Transformers-compatible callback around the CPU coordinator."""
    if not isinstance(base_callback_class, type):
        raise TypeError("base callback must be a class")

    class CheckpointMonitorCallback(base_callback_class):
        def __init__(self, *, coordinator: CheckpointMonitorCoordinator):
            super().__init__()
            if not isinstance(coordinator, CheckpointMonitorCoordinator):
                raise TypeError("callback requires a CheckpointMonitorCoordinator")
            self.checkpoint_monitor = coordinator

        def on_train_begin(self, args, state, control, **kwargs):
            self.checkpoint_monitor.on_train_begin(args, state)
            return control

        def on_save(self, args, state, control, **kwargs):
            self.checkpoint_monitor.on_save(args, state)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.checkpoint_monitor.on_train_end(state)
            return control

    CheckpointMonitorCallback.__name__ = f"CheckpointMonitor{base_callback_class.__name__}"
    return CheckpointMonitorCallback


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="command_name", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", required=True)
    run.add_argument("--failure", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("--timeout-seconds", type=float, required=True)
    run.add_argument("--mode", choices=("production", "smoke"), required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    receipt = run_supervised_monitor(
        command=command,
        repo_root=args.repo_root,
        output_dir=args.output,
        failure_path=args.failure,
        receipt_path=args.receipt,
        timeout_seconds=args.timeout_seconds,
        expected_mode=args.mode,
    )
    print(json.dumps({"status": receipt["status"], "receipt": args.receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckpointMonitorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
