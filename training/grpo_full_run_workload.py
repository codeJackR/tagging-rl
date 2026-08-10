#!/usr/bin/env python3
"""Production workload boundary for one detached 300-step GRPO run.

The detached wrapper may execute this module, but the normal ``train_grpo`` CLI
does not dispatch it.  Heavy GPU imports remain inside ``run_full_run_300_gate``
and occur only after the locked read-only preflight passes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from training.grpo_detached_control import (
    EXIT_FILE,
    WORKLOAD_RESULT_FILE,
    DetachedRunControlWriter,
)
from training.grpo_detached_runtime import utc_now
from training.train_grpo import (
    DEFAULT_ADAPTER,
    DEFAULT_FULL_RUN_DATA,
    DEFAULT_FULL_RUN_MANIFEST,
    DEFAULT_FULL_RUN_OUTPUT_DIR,
    DEFAULT_MINIMUM_FREE_GIB,
    DEFAULT_SELECTION_MANIFEST,
    run_full_run_300_gate,
    run_full_run_300_preflight,
)

FULL_RUN_WORKLOAD_VERSION = "grpo-full-run-detached-workload-v1"


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_exclusive_atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_production_full_run_workload(
    *,
    control_dir: str | Path,
    repo_root: str | Path,
    expected_commit: str,
    environment: Mapping[str, str] | None = None,
    preflight_fn: Callable[..., dict] = run_full_run_300_preflight,
    bridge_fn: Callable[..., dict] = run_full_run_300_gate,
    clock_fn: Callable[[], str] = utc_now,
    result_writer_fn: Callable[[Path, dict], None] = (
        _write_exclusive_atomic_json
    ),
) -> dict:
    """Run locked preflight, bridge and detached result handoff in order."""
    if not _is_git_commit(expected_commit):
        raise ValueError("full-run workload expected commit is invalid")
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError("full-run workload repository root does not exist")
    writer = DetachedRunControlWriter(control_dir=control_dir)
    snapshot = writer.snapshot()
    if snapshot["worker"] is None or snapshot["exit"] is not None:
        raise RuntimeError("full-run workload requires one active detached worker")
    if writer.launch.get("expected_commit") != expected_commit:
        raise RuntimeError("full-run workload commit disagrees with launch evidence")
    launch_cwd = Path(
        writer.launch.get("subprocess_contract", {}).get("cwd", "")
    ).resolve()
    if launch_cwd != repo_root:
        raise RuntimeError("full-run workload repository root drifted from launch")

    final_output = (repo_root / DEFAULT_FULL_RUN_OUTPUT_DIR).resolve()
    if writer.final_output != final_output:
        raise RuntimeError("full-run workload output drifted from reserved path")
    result_path = writer.control_dir / WORKLOAD_RESULT_FILE
    if result_path.exists():
        raise FileExistsError("full-run workload result already exists")
    if (writer.control_dir / EXIT_FILE).exists():
        raise RuntimeError("full-run workload cannot start after worker exit")

    active_environment = os.environ if environment is None else environment
    control_value = active_environment.get("GRPO_DETACHED_CONTROL_DIR")
    result_value = active_environment.get("GRPO_DETACHED_WORKLOAD_RESULT")
    if not isinstance(control_value, str) or not control_value:
        raise RuntimeError("detached control environment path is missing")
    if not isinstance(result_value, str) or not result_value:
        raise RuntimeError("detached workload-result environment path is missing")
    environment_control = Path(control_value).resolve()
    environment_result = Path(result_value).resolve()
    if environment_control != writer.control_dir:
        raise RuntimeError("detached control environment path drifted")
    if environment_result != result_path:
        raise RuntimeError("detached workload-result environment path drifted")

    preflight = preflight_fn(
        repo_root=repo_root,
        training_data=DEFAULT_FULL_RUN_DATA,
        pool_manifest=DEFAULT_FULL_RUN_MANIFEST,
        selection_manifest=DEFAULT_SELECTION_MANIFEST,
        adapter=DEFAULT_ADAPTER,
        output_dir=DEFAULT_FULL_RUN_OUTPUT_DIR,
        minimum_free_bytes=int(DEFAULT_MINIMUM_FREE_GIB * 1024**3),
        expected_commit=expected_commit,
    )
    if preflight.get("status") != "passed":
        raise RuntimeError("full-run workload preflight did not pass")
    if preflight.get("cuda_imports_performed") is not False:
        raise RuntimeError("full-run workload preflight imported CUDA")
    if preflight.get("git", {}).get("commit") != expected_commit:
        raise RuntimeError("full-run workload preflight commit drifted")
    if Path(preflight.get("output", {}).get("path", "")).resolve() != final_output:
        raise RuntimeError("full-run workload preflight output drifted")

    def record_progress(
        *,
        optimizer_step: int,
        rollout_records: int,
        scalar_logs: int,
    ) -> dict:
        return writer.record_progress(
            optimizer_step=optimizer_step,
            rollout_records=rollout_records,
            scalar_logs=scalar_logs,
            updated_at=clock_fn(),
        )

    bridge_report = bridge_fn(
        preflight_report=preflight,
        training_data_path=preflight["pool"]["data_path"],
        adapter_path=preflight["sft_lock"]["adapter_path"],
        adapter_file=preflight["sft_lock"]["adapter_file"],
        final_output_dir=final_output,
        progress_callback=record_progress,
    )
    if (
        bridge_report.get("status") != "passed"
        or bridge_report.get("published") is not True
        or not final_output.is_dir()
    ):
        raise RuntimeError("full-run workload bridge did not publish success")
    completed_progress = writer.snapshot()["progress"]
    if (
        completed_progress is None
        or completed_progress.get("optimizer_step") != 300
        or completed_progress.get("rollout_records") != 2_400
        or completed_progress.get("scalar_logs") != 300
    ):
        raise RuntimeError("full-run workload lacks complete detached progress")

    result = {
        **bridge_report,
        "final_output_dir": str(final_output),
        "detached_workload": {
            "version": FULL_RUN_WORKLOAD_VERSION,
            "status": "passed",
            "expected_commit": expected_commit,
            "repo_root": str(repo_root),
            "control_dir": str(writer.control_dir),
            "optimizer_steps": 300,
            "rollout_records": 2_400,
            "scalar_logs": 300,
            "preflight_passed_before_bridge": True,
        },
    }
    result_writer_fn(result_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_production_full_run_workload(
        control_dir=args.control_dir,
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
