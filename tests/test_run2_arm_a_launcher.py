from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.run2_arm_a_launcher import (
    DEFAULT_CAUSAL_PREFLIGHT,
    DEFAULT_CONSTRUCTION,
    DEFAULT_CONTRACT,
    DEFAULT_MONITOR_CONTRACT,
    LAUNCHER_CODE_FILE,
    MINIMUM_MONITOR_DRIVER_FREE_BYTES,
    build_monitor_command,
    build_readiness_report,
    parse_args,
)


COMMIT = "a" * 40


def _copy(root: Path, target: Path, relative: str) -> None:
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / relative, destination)


@pytest.fixture
def readiness_root(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    for relative in (
        DEFAULT_CONTRACT,
        DEFAULT_CAUSAL_PREFLIGHT,
        DEFAULT_CONSTRUCTION,
        DEFAULT_MONITOR_CONTRACT,
        "data/grpo_run2_causal_schedule_v1.jsonl",
    ):
        _copy(source, tmp_path, relative)
    (tmp_path / "packs/vastraa_taste_v1").mkdir(parents=True)
    return tmp_path


def _fresh_preflight(root: Path) -> dict:
    contract = json.loads((root / DEFAULT_CONTRACT).read_text(encoding="utf-8"))
    return {
        "status": "passed_read_only_no_training_dispatch",
        "current_git_commit": COMMIT,
        "inputs_verified": len(contract["inputs"]),
        "execution_files_verified": len(contract["execution_code"]),
        "gpu": {
            "name": "NVIDIA GeForce RTX 3090",
            "driver": "590.48.01",
            "memory_mib": 24576,
            "memory_used_mib": 396,
            "utilization_percent": 0,
        },
        "disk_free_bytes": 4 * 1024**3,
        "deferred_decisions": 0,
        "model_loaded": False,
        "trainer_constructed": False,
        "training_dispatched": False,
    }


def _build(root: Path, **overrides):
    values = {
        "root": root,
        "expected_launcher_code_commit": COMMIT,
        "current_preflight_fn": lambda **_kwargs: _fresh_preflight(root),
        "gpu_free_bytes_fn": lambda: 20 * 1024**3,
        "disk_usage_fn": lambda _path: SimpleNamespace(free=4 * 1024**3),
        "launcher_identity_fn": lambda actual_root, commit, relative: {
            "path": relative,
            "bytes": 123,
            "sha256": "b" * 64,
            "git_commit": commit,
        },
    }
    values.update(overrides)
    return build_readiness_report(**values)


def test_readiness_composes_locked_arm_a_without_dispatch(readiness_root: Path):
    report = _build(readiness_root)

    assert report["status"] == "arm_a_launch_bridge_ready_no_dispatch"
    assert report["arm"] == "A"
    assert report["schedule_binding"]["rows"] == 300
    assert report["schedule_binding"]["unique_skus"] == 300
    assert report["schedule_binding"]["dataset_materialized"] is False
    assert report["reward_binding"]["callable_names"] == [
        "format_validity_reward",
        "vocab_rule_compliance_reward",
        "golden_agreement_reward",
    ]
    assert report["reward_binding"]["weights"] == [1.0, 1.0, 2.0]
    monitor = report["checkpoint_monitor_binding"]
    assert monitor["required_checkpoint_steps"] == [100, 200, 300]
    assert monitor["synchronous_after_checkpoint_save"] is True
    assert monitor["driver_free_gate_rechecked_before_each_monitor"] is True
    assert monitor["monitor_runner_invoked"] is False
    assert monitor["callback_lifecycle_invoked"] is False
    assert report["boundaries"]["dispatch_cli_available"] is False
    assert report["boundaries"]["optimizer_steps"] == 0
    assert all(not Path(path).exists() for path in report["reserved_paths"].values())


def test_monitor_commands_are_exact_and_reject_path_or_step_drift(
    readiness_root: Path,
):
    root = readiness_root.resolve()
    contract = root / DEFAULT_MONITOR_CONTRACT
    pack = root / "packs/vastraa_taste_v1"
    checkpoint = root / "runs/grpo-run2-arm-a-original/checkpoint-100"
    output = root / "runs/grpo-run2-arm-a-original-monitor/checkpoint-100"
    command = build_monitor_command(
        root=root,
        monitor_contract=contract,
        pack_path=pack,
        step=100,
        checkpoint=checkpoint,
        output=output,
        python_executable="/venv/rl/bin/python",
    )
    assert command[:3] == [
        "/venv/rl/bin/python",
        "-m",
        "training.run2_checkpoint_monitor_runtime",
    ]
    assert command[-7:] == [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--mode",
        "production",
    ]
    with pytest.raises(ValueError, match="100, 200 or 300"):
        build_monitor_command(
            root=root,
            monitor_contract=contract,
            pack_path=pack,
            step=99,
            checkpoint=checkpoint,
            output=output,
            python_executable="python",
        )
    with pytest.raises(RuntimeError, match="checkpoint path drifted"):
        build_monitor_command(
            root=root,
            monitor_contract=contract,
            pack_path=pack,
            step=100,
            checkpoint=root / "wrong/checkpoint-100",
            output=output,
            python_executable="python",
        )


def test_readiness_fails_closed_on_schedule_drift_or_output_collision(
    readiness_root: Path,
):
    schedule = readiness_root / "data/grpo_run2_causal_schedule_v1.jsonl"
    schedule.write_bytes(schedule.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="schedule file drifted"):
        _build(readiness_root)

    readiness_root = readiness_root.parent / "collision"
    source = Path(__file__).resolve().parents[1]
    for relative in (
        DEFAULT_CONTRACT,
        DEFAULT_CAUSAL_PREFLIGHT,
        DEFAULT_CONSTRUCTION,
        DEFAULT_MONITOR_CONTRACT,
        "data/grpo_run2_causal_schedule_v1.jsonl",
    ):
        _copy(source, readiness_root, relative)
    (readiness_root / "packs/vastraa_taste_v1").mkdir(parents=True)
    (readiness_root / "runs/grpo-run2-arm-a-original").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _build(readiness_root)


def test_readiness_rejects_prior_lineage_and_fresh_preflight_drift(
    readiness_root: Path,
):
    preflight_path = readiness_root / DEFAULT_CAUSAL_PREFLIGHT
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["contract"]["sha256"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different contract"):
        _build(readiness_root)

    source = Path(__file__).resolve().parents[1]
    _copy(source, readiness_root, DEFAULT_CAUSAL_PREFLIGHT)
    bad = _fresh_preflight(readiness_root)
    bad["trainer_constructed"] = True
    with pytest.raises(RuntimeError, match="crossed boundary"):
        _build(readiness_root, current_preflight_fn=lambda **_kwargs: bad)


@pytest.mark.parametrize(
    ("gpu_free", "disk_free", "message"),
    [
        (
            MINIMUM_MONITOR_DRIVER_FREE_BYTES - 1,
            4 * 1024**3,
            "less than the 6 GiB",
        ),
        (20 * 1024**3, 3 * 1024**3 - 1, "disk floor"),
    ],
)
def test_readiness_enforces_current_resource_floors(
    readiness_root: Path, gpu_free: int, disk_free: int, message: str
):
    with pytest.raises(RuntimeError, match=message):
        _build(
            readiness_root,
            gpu_free_bytes_fn=lambda: gpu_free,
            disk_usage_fn=lambda _path: SimpleNamespace(free=disk_free),
        )


def test_cli_has_validation_only_and_requires_code_commit():
    args = parse_args(
        ["validate", "--expected-launcher-code-commit", COMMIT]
    )
    assert args.command == "validate"
    assert args.expected_launcher_code_commit == COMMIT
    with pytest.raises(SystemExit):
        parse_args(["launch"])
    assert LAUNCHER_CODE_FILE == "training/run2_arm_a_launcher.py"


def test_published_readiness_receipt_preserves_no_dispatch_lineage_when_present():
    root = Path(__file__).resolve().parents[1]
    path = root / "runs/grpo-run2-arm-a-launch-readiness.json"
    if not path.exists():
        pytest.skip("Arm A readiness receipt has not been published")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "arm_a_launch_bridge_ready_no_dispatch"
    assert report["arm"] == "A"
    assert report["boundaries"]["training_dispatched"] is False
    assert report["boundaries"]["trainer_constructed"] is False
    assert report["boundaries"]["optimizer_steps"] == 0
    assert report["boundaries"]["arm_paths_created"] is False
    launcher = report["launcher_code"]
    committed = subprocess.run(
        ["git", "show", f"{launcher['git_commit']}:{launcher['path']}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    assert len(committed) == launcher["bytes"]
    assert hashlib.sha256(committed).hexdigest() == launcher["sha256"]
    for identity in report["artifacts"].values():
        artifact = root / identity["path"]
        assert artifact.stat().st_size == identity["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == identity["sha256"]
