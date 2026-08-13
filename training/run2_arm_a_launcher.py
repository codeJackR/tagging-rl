#!/usr/bin/env python3
"""Read-only launch bridge for the Run 2 corrected-control Arm A.

This module proves that the locked causal inputs can be composed into one Arm A
runtime surface.  It deliberately exposes no training-dispatch command: model
loading, Trainer construction, output-directory creation and optimizer work are
all reserved for a later, explicit Phase H step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import read_jsonl
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.dataset import load_grpo_prompts
from training.run2_causal_experiment import (
    BASE_MODEL,
    MINIMUM_MONITOR_DRIVER_FREE_BYTES,
    VERSION as CAUSAL_CONTRACT_VERSION,
    CausalCheckpointMonitorCoordinator,
    _reward_callables,
    make_causal_monitor_callback_class,
    run_preflight as run_causal_preflight,
)
from training.run2_checkpoint_monitor_control import CheckpointMonitorCoordinator


VERSION = "grpo-run2-arm-a-launch-readiness-v1"
ARM = "A"
DEFAULT_CONTRACT = "runs/grpo-run2-causal-experiment-contract.json"
DEFAULT_CAUSAL_PREFLIGHT = "runs/grpo-run2-causal-preflight.json"
DEFAULT_CONSTRUCTION = "runs/grpo-run2-causal-construction.json"
DEFAULT_MONITOR_CONTRACT = "runs/grpo-run2-checkpoint-monitor-contract.json"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_OUTPUT = "runs/grpo-run2-arm-a-launch-readiness.json"
LAUNCHER_CODE_FILE = "training/run2_arm_a_launcher.py"
EXPECTED_OUTPUT_KEYS = (
    "output_dir",
    "monitor_root",
    "quality_root",
    "control_dir",
    "failure_dir",
)


class _ReadOnlyCallbackBase:
    """Minimal callback base used only to prove callback composition."""

    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _git_file_identity(root: Path, commit: str, relative: str) -> dict[str, Any]:
    if not _is_commit(commit):
        raise ValueError("launcher code commit must be a full lowercase Git SHA")
    try:
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"launcher file is absent from code commit: {relative}") from exc
    current = (root / relative).read_bytes()
    if current != committed:
        raise RuntimeError(f"launcher file differs from code commit: {relative}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("launcher code commit is not an ancestor of current HEAD")
    return {
        "path": relative,
        "bytes": len(committed),
        "sha256": hashlib.sha256(committed).hexdigest(),
        "git_commit": commit,
    }


def _driver_free_bytes() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError("Arm A readiness requires exactly one GPU free-memory value")
    return int(values[0]) * 1024**2


def build_monitor_command(
    *,
    root: Path,
    monitor_contract: Path,
    pack_path: Path,
    step: int,
    checkpoint: Path,
    output: Path,
    python_executable: str,
) -> list[str]:
    """Build the exact synchronous production evaluator command."""
    if step not in (100, 200, 300):
        raise ValueError("Arm A monitor step must be 100, 200 or 300")
    expected_checkpoint = (root / f"runs/grpo-run2-arm-a-original/checkpoint-{step}").resolve()
    if checkpoint.resolve() != expected_checkpoint:
        raise RuntimeError("Arm A monitor checkpoint path drifted")
    expected_output = (root / f"runs/grpo-run2-arm-a-original-monitor/checkpoint-{step}").resolve()
    if output.resolve() != expected_output:
        raise RuntimeError("Arm A monitor output path drifted")
    return [
        python_executable,
        "-m",
        "training.run2_checkpoint_monitor_runtime",
        "--repo-root",
        str(root),
        "--contract",
        str(monitor_contract.relative_to(root)),
        "--pack",
        str(pack_path.relative_to(root)),
        "--base-model",
        BASE_MODEL,
        "--local-files-only",
        "evaluate",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
        "--mode",
        "production",
    ]


def _validate_prior_artifacts(
    *,
    root: Path,
    contract_path: Path,
    preflight_path: Path,
    construction_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _load_json(contract_path)
    preflight = _load_json(preflight_path)
    construction = _load_json(construction_path)
    contract_identity = _identity(contract_path, root)
    preflight_identity = _identity(preflight_path, root)

    if (
        contract.get("version") != CAUSAL_CONTRACT_VERSION
        or contract.get("status") != "locked_no_gpu_training_dispatched"
    ):
        raise ValueError("Arm A launcher requires the locked causal contract")
    if contract.get("arm_order") != ["A", "B"]:
        raise RuntimeError("causal arm order drifted")
    if contract.get("boundaries", {}).get("gpu_training_dispatched_by_contract_build") is not False:
        raise RuntimeError("causal contract build crossed the training boundary")
    if contract.get("deferral_audit", {}).get("passed") is not True:
        raise RuntimeError("causal contract still has deferred decisions")

    if preflight.get("status") != "passed_read_only_no_training_dispatch":
        raise ValueError("Arm A launcher requires the accepted causal preflight")
    if preflight.get("contract") != contract_identity:
        raise RuntimeError("causal preflight names a different contract")
    if preflight.get("expected_execution_code_commit") != contract.get(
        "expected_execution_code_commit"
    ):
        raise RuntimeError("causal preflight code lineage drifted")
    if preflight.get("inputs_verified") != len(contract.get("inputs", {})):
        raise RuntimeError("causal preflight input count drifted")
    if preflight.get("execution_files_verified") != len(
        contract.get("execution_code", {})
    ):
        raise RuntimeError("causal preflight execution-file count drifted")

    if construction.get("status") != "both_arm_configs_constructed_no_trainer_no_dispatch":
        raise ValueError("Arm A launcher requires the accepted config construction")
    if construction.get("contract") != contract_identity:
        raise RuntimeError("config construction names a different contract")
    if construction.get("preflight") != preflight_identity:
        raise RuntimeError("config construction names a different causal preflight")
    if construction.get("causal_difference_audit") != contract.get(
        "causal_difference_audit"
    ):
        raise RuntimeError("config construction causal audit drifted")
    for evidence in (preflight, construction):
        for key in ("trainer_constructed", "training_dispatched"):
            if evidence.get(key) is not False:
                raise RuntimeError(f"prior artifact crossed boundary: {key}")
    if preflight.get("model_loaded") is not False or construction.get(
        "model_constructed"
    ) is not False:
        raise RuntimeError("prior artifact loaded or constructed a model")
    return contract, preflight, construction


def _validate_schedule(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    schedule_identity = contract["schedule"]["dataset"]
    schedule_path = root / schedule_identity["path"]
    if _identity(schedule_path, root) != schedule_identity:
        raise RuntimeError("Arm A schedule file drifted")
    rows = read_jsonl(schedule_path)
    sku_ids = [row.sku_id for row in rows]
    pass_rates = [row.difficulty.sft_pass_rate for row in rows]
    if len(rows) != 300 or len(set(sku_ids)) != 300:
        raise RuntimeError("Arm A schedule must contain 300 unique products")
    if _ordered_hash(sku_ids) != contract["schedule"]["ordered_sku_sha256"]:
        raise RuntimeError("Arm A schedule optimizer-step order drifted")
    if any(rate is None or not 0.0 < rate < 1.0 for rate in pass_rates):
        raise RuntimeError("Arm A schedule contains an ineligible pass rate")
    return {
        "dataset": schedule_identity,
        "rows": len(rows),
        "unique_skus": len(set(sku_ids)),
        "ordered_sku_sha256": _ordered_hash(sku_ids),
        "one_product_per_optimizer_step": True,
        "trainer_shuffle": contract["arms"][ARM]["config"]["shuffle_dataset"],
        "sft_pass_rate_minimum": min(pass_rates),
        "sft_pass_rate_maximum": max(pass_rates),
        "dataset_loader": {
            "module": load_grpo_prompts.__module__,
            "name": load_grpo_prompts.__name__,
            "arguments": {
                "path": schedule_identity["path"],
                "require_pass_rate_band": True,
            },
        },
        "dataset_materialized": False,
    }


def _validate_reward_binding(
    contract: Mapping[str, Any], construction: Mapping[str, Any]
) -> dict[str, Any]:
    spec = contract["arms"][ARM]
    callables = _reward_callables(ARM)
    names = [getattr(value, "__name__", type(value).__name__) for value in callables]
    modules = [value.__module__ for value in callables]
    if names != spec["reward"]["functions"]:
        raise RuntimeError("Arm A reward callable order drifted")
    if construction["arms"][ARM]["reward_callable_names"] != names:
        raise RuntimeError("Arm A reward construction evidence drifted")
    if construction["arms"][ARM]["reward_callable_modules"] != modules:
        raise RuntimeError("Arm A reward module binding drifted")
    if spec["reward"]["weights"] != [1.0, 1.0, 2.0]:
        raise RuntimeError("Arm A reward weights drifted")
    if any(not callable(value) for value in callables):
        raise TypeError("Arm A reward binding contains a non-callable")
    return {
        "policy": spec["reward"]["policy"],
        "callable_names": names,
        "callable_modules": modules,
        "weights": spec["reward"]["weights"],
        "same_binding_as_constructed_config": True,
        "reward_called": False,
    }


def _validate_callback_wiring(
    *,
    root: Path,
    contract: Mapping[str, Any],
    python_executable: str,
) -> dict[str, Any]:
    spec = contract["arms"][ARM]
    monitor_contract = (root / DEFAULT_MONITOR_CONTRACT).resolve()
    pack_path = (root / DEFAULT_PACK).resolve()
    monitor_root = (root / spec["monitor_root"]).resolve()
    quality_root = (root / spec["quality_root"]).resolve()

    def command_builder(step: int, checkpoint: Path, output: Path) -> list[str]:
        return build_monitor_command(
            root=root,
            monitor_contract=monitor_contract,
            pack_path=pack_path,
            step=step,
            checkpoint=checkpoint,
            output=output,
            python_executable=python_executable,
        )

    def forbidden_runner(**_kwargs: Any) -> Mapping[str, Any]:
        raise AssertionError("read-only Arm A readiness may not run a monitor")

    base = CheckpointMonitorCoordinator(
        repo_root=root,
        contract_path=monitor_contract,
        monitor_root=monitor_root,
        command_builder=command_builder,
        runner=forbidden_runner,
    )
    causal = CausalCheckpointMonitorCoordinator(
        base=base,
        quality_policy=contract["monitoring"]["quality_policy"],
        quality_output_root=quality_root,
    )
    callback_class = make_causal_monitor_callback_class(_ReadOnlyCallbackBase)
    callback = callback_class(coordinator=causal)
    templates = {}
    for step in (100, 200, 300):
        checkpoint = root / spec["output_dir"] / f"checkpoint-{step}"
        output = monitor_root / f"checkpoint-{step}"
        templates[str(step)] = command_builder(step, checkpoint, output)
    if callback.causal_monitor is not causal:
        raise RuntimeError("Arm A callback did not retain the causal coordinator")
    return {
        "base_coordinator_class": type(base).__name__,
        "causal_coordinator_class": type(causal).__name__,
        "callback_class": callback_class.__name__,
        "required_checkpoint_steps": list(base.expected_steps),
        "synchronous_after_checkpoint_save": True,
        "monitor_failure_aborts_immediately": True,
        "quality_policy": contract["monitoring"]["quality_policy"],
        "driver_free_gate_bytes": MINIMUM_MONITOR_DRIVER_FREE_BYTES,
        "driver_free_gate_rechecked_before_each_monitor": True,
        "monitor_commands": templates,
        "monitor_runner_invoked": False,
        "callback_lifecycle_invoked": False,
        "monitor_paths_created": False,
    }


def build_readiness_report(
    *,
    root: str | Path,
    expected_launcher_code_commit: str,
    contract_path: str | Path = DEFAULT_CONTRACT,
    causal_preflight_path: str | Path = DEFAULT_CAUSAL_PREFLIGHT,
    construction_path: str | Path = DEFAULT_CONSTRUCTION,
    python_executable: str = sys.executable,
    current_preflight_fn: Callable[..., dict[str, Any]] = run_causal_preflight,
    gpu_free_bytes_fn: Callable[[], int] = _driver_free_bytes,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
    launcher_identity_fn: Callable[[Path, str, str], dict[str, Any]] = _git_file_identity,
) -> dict[str, Any]:
    """Build one fail-closed Arm A readiness report without dispatching work."""
    root = Path(root).resolve()
    contract_path = (root / contract_path).resolve()
    causal_preflight_path = (root / causal_preflight_path).resolve()
    construction_path = (root / construction_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Arm A readiness repository root does not exist")
    launcher_code = launcher_identity_fn(
        root, expected_launcher_code_commit, LAUNCHER_CODE_FILE
    )
    contract, prior_preflight, construction = _validate_prior_artifacts(
        root=root,
        contract_path=contract_path,
        preflight_path=causal_preflight_path,
        construction_path=construction_path,
    )
    if contract["arms"][ARM].get("role") != "corrected_control":
        raise RuntimeError("Arm A no longer names the corrected control")

    current_preflight = current_preflight_fn(
        root=root,
        contract_path=contract_path,
    )
    if current_preflight.get("status") != "passed_read_only_no_training_dispatch":
        raise RuntimeError("fresh causal preflight did not pass")
    for key in ("model_loaded", "trainer_constructed", "training_dispatched"):
        if current_preflight.get(key) is not False:
            raise RuntimeError(f"fresh causal preflight crossed boundary: {key}")

    spec = contract["arms"][ARM]
    paths = {key: (root / spec[key]).resolve() for key in EXPECTED_OUTPUT_KEYS}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Arm A output/control/monitor path already exists")
    schedule = _validate_schedule(root, contract)
    reward = _validate_reward_binding(contract, construction)
    monitor = _validate_callback_wiring(
        root=root,
        contract=contract,
        python_executable=python_executable,
    )
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Arm A readiness unexpectedly created an output path")

    current_driver_free = int(gpu_free_bytes_fn())
    if current_driver_free < MINIMUM_MONITOR_DRIVER_FREE_BYTES:
        raise RuntimeError("Arm A readiness GPU has less than the 6 GiB monitor floor")
    disk_free = int(disk_usage_fn(root / "runs").free)
    suite_floor = int(contract["resources"]["minimum_suite_start_bytes"])
    if disk_free < suite_floor:
        raise RuntimeError("Arm A readiness disk floor is not met")

    config = spec["config"]
    constructed_settings = construction["arms"][ARM]["config"]["settings"]
    normalized_contract = dict(config)
    normalized_contract["report_to"] = []
    normalized_contract["reward_weights"] = spec["reward"]["weights"]
    if constructed_settings != normalized_contract:
        raise RuntimeError("Arm A constructed GRPOConfig drifted from contract")

    return {
        "version": VERSION,
        "status": "arm_a_launch_bridge_ready_no_dispatch",
        "mode": "read_only_preparation",
        "arm": ARM,
        "role": "corrected_control",
        "launcher_code": launcher_code,
        "artifacts": {
            "causal_contract": _identity(contract_path, root),
            "accepted_causal_preflight": _identity(causal_preflight_path, root),
            "accepted_config_construction": _identity(construction_path, root),
        },
        "causal_execution_code_commit": contract["expected_execution_code_commit"],
        "fresh_causal_preflight": {
            "status": current_preflight["status"],
            "current_git_commit": current_preflight["current_git_commit"],
            "inputs_verified": current_preflight["inputs_verified"],
            "execution_files_verified": current_preflight["execution_files_verified"],
            "gpu": current_preflight["gpu"],
            "disk_free_bytes": current_preflight["disk_free_bytes"],
            "deferred_decisions": current_preflight["deferred_decisions"],
            "model_loaded": False,
            "trainer_constructed": False,
            "training_dispatched": False,
        },
        "schedule_binding": schedule,
        "reward_binding": reward,
        "trainer_config": {
            "settings": config,
            "constructed_settings_match": True,
            "config_object_reused_from_proof": False,
            "future_runtime_must_reconstruct_and_revalidate": True,
        },
        "checkpoint_monitor_binding": monitor,
        "resource_readiness": {
            "driver_free_bytes_now": current_driver_free,
            "minimum_driver_free_bytes_before_each_monitor": (
                MINIMUM_MONITOR_DRIVER_FREE_BYTES
            ),
            "current_snapshot_passed": True,
            "current_snapshot_is_not_concurrent_training_proof": True,
            "disk_free_bytes": disk_free,
            "minimum_suite_start_bytes": suite_floor,
            "disk_floor_passed": True,
        },
        "reserved_paths": {key: str(path) for key, path in paths.items()},
        "boundaries": {
            "confirmation_data_used": False,
            "legacy_frozen_300_used": False,
            "dataset_materialized": False,
            "reward_called": False,
            "monitor_process_started": False,
            "callback_lifecycle_started": False,
            "arm_paths_created": False,
            "cuda_library_imported_by_launcher": False,
            "model_loaded": False,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_dispatched": False,
            "dispatch_cli_available": False,
        },
        "next_gate": (
            "implement and CPU-prove the Arm A runtime/trainer composition; "
            "keep actual detached GPU dispatch separate"
        ),
        "prior_preflight_was_read_only": prior_preflight["training_dispatched"] is False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--expected-launcher-code-commit", required=True)
    validate.add_argument("--contract", default=DEFAULT_CONTRACT)
    validate.add_argument("--causal-preflight", default=DEFAULT_CAUSAL_PREFLIGHT)
    validate.add_argument("--construction", default=DEFAULT_CONSTRUCTION)
    validate.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "validate":
        raise RuntimeError("Arm A dispatch is intentionally unavailable")
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Arm A readiness output already exists: {output}")
    report = build_readiness_report(
        root=root,
        expected_launcher_code_commit=args.expected_launcher_code_commit,
        contract_path=args.contract,
        causal_preflight_path=args.causal_preflight,
        construction_path=args.construction,
    )
    write_exclusive_atomic_json(output, report)
    print(json.dumps({"output": args.output, "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
