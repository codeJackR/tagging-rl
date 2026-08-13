"""CPU-only tests for checkpoint-monitor supervision and Trainer handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.run2_checkpoint_monitor_control import (
    CheckpointMonitorCoordinator,
    CheckpointMonitorError,
    make_checkpoint_monitor_callback_class,
    run_supervised_monitor,
)


def _paths(tmp_path: Path):
    return {
        "output_dir": tmp_path / "monitor-step-100",
        "failure_path": tmp_path / "monitor-step-100.failure.json",
        "receipt_path": tmp_path / "monitor-step-100.receipt.json",
    }


def _success_command(output: Path, *, mode: str = "smoke") -> list[str]:
    code = r'''
import hashlib, json, os, sys, tempfile
from pathlib import Path
output = Path(sys.argv[1])
mode = sys.argv[2]
staging = Path(tempfile.mkdtemp(prefix=".monitor-staging-", dir=output.parent))
payloads = {
    "greedy.jsonl": "{\"raw\":\"{}\",\"sku_id\":\"one\"}\n",
    "sampled.jsonl": "{\"raw\":\"{}\",\"repeat\":0,\"seed\":1,\"sku_id\":\"one\"}\n",
    "report.json": json.dumps({"status": "checkpoint_outputs_scored"}) + "\n",
    "resource.json": json.dumps({"released": True}) + "\n",
}
for name, value in payloads.items():
    (staging / name).write_text(value, encoding="utf-8")
def identity(name):
    raw = (staging / name).read_bytes()
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
manifest = {
    "status": "checkpoint_monitor_complete",
    "mode": mode,
    "quality_evidence": mode == "production",
    "checkpoint": {"adapter_model_sha256": "abc123"},
    "files": {
        "greedy_predictions": identity("greedy.jsonl"),
        "sampled_predictions": identity("sampled.jsonl"),
        "scored_report": identity("report.json"),
        "resource_report": identity("resource.json"),
    },
    "invariants": {
        "fixed_development_membership": True,
        "greedy_and_repeated_sampled_complete": True,
        "original_and_dense_rewards_scored": True,
        "confirmation_data_used": False,
        "quality_abort_threshold_applied": False,
        "published_exclusively_and_atomically": True,
    },
}
(staging / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
os.rename(staging, output)
print("monitor child complete")
'''
    script = output.parent / "publish_valid_monitor_bundle.py"
    script.write_text(code, encoding="utf-8")
    return [sys.executable, str(script), str(output), mode]


def test_supervisor_accepts_only_complete_bound_success_bundle(tmp_path):
    paths = _paths(tmp_path)
    receipt = run_supervised_monitor(
        command=_success_command(paths["output_dir"]),
        repo_root=tmp_path,
        timeout_seconds=5,
        expected_mode="smoke",
        expected_checkpoint_sha256="abc123",
        **paths,
    )
    assert receipt["status"] == "checkpoint_monitor_accepted"
    assert receipt["streams"]["stdout"]["tail"] == "monitor child complete\n"
    assert receipt["quality_abort_threshold_applied"] is False
    assert paths["receipt_path"].is_file()
    assert not paths["failure_path"].exists()


def test_supervisor_times_out_process_group_and_publishes_failure(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(CheckpointMonitorError, match="timeout"):
        run_supervised_monitor(
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            repo_root=tmp_path,
            timeout_seconds=0.05,
            expected_mode="smoke",
            **paths,
        )
    failure = json.loads(paths["failure_path"].read_text(encoding="utf-8"))
    assert failure["reason"] == "timeout"
    assert failure["timed_out"] is True
    assert failure["termination"]["terminate_sent"] is True
    assert failure["training_must_abort"] is True
    assert not paths["receipt_path"].exists()


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"], "nonzero_exit"),
        ([sys.executable, "-c", "print('no bundle')"], "invalid_success_bundle"),
    ],
)
def test_supervisor_publishes_nonzero_and_invalid_success_failures(tmp_path, command, reason):
    paths = _paths(tmp_path)
    with pytest.raises(CheckpointMonitorError, match=reason):
        run_supervised_monitor(
            command=command,
            repo_root=tmp_path,
            timeout_seconds=5,
            expected_mode="smoke",
            **paths,
        )
    failure = json.loads(paths["failure_path"].read_text(encoding="utf-8"))
    assert failure["reason"] == reason
    assert failure["quality_threshold_involved"] is False


def test_supervisor_publishes_failure_when_child_cannot_be_spawned(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(CheckpointMonitorError, match="spawn_error"):
        run_supervised_monitor(
            command=[str(tmp_path / "missing-evaluator")],
            repo_root=tmp_path,
            timeout_seconds=5,
            expected_mode="smoke",
            **paths,
        )
    failure = json.loads(paths["failure_path"].read_text(encoding="utf-8"))
    assert failure["reason"] == "spawn_error"
    assert failure["spawn_error"].startswith("FileNotFoundError:")
    assert failure["return_code"] is None
    assert failure["training_must_abort"] is True


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "monitor-contract.json"
    path.write_text(
        json.dumps(
            {
                "checkpoints": {"required_steps": [100, 200, 300]},
                "runtime": {"timeout_seconds_per_checkpoint": 60},
                "abort_policy": {"quality_abort_enabled": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint(output_dir: Path, step: int) -> Path:
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
    return path


class FakeCallback:
    def __init__(self):
        self.initialized = True


def _coordinator(tmp_path: Path, calls: list[dict]):
    def runner(**kwargs):
        calls.append(kwargs)
        return {
            "status": "checkpoint_monitor_accepted",
            "quality_abort_threshold_applied": False,
        }

    return CheckpointMonitorCoordinator(
        repo_root=tmp_path,
        contract_path=_contract(tmp_path),
        monitor_root=tmp_path / "monitoring",
        command_builder=lambda step, checkpoint, output: [
            sys.executable,
            "-c",
            "pass",
            str(step),
            str(checkpoint),
            str(output),
        ],
        runner=runner,
    )


def test_callback_evaluates_exact_checkpoint_order_and_finishes(tmp_path):
    calls: list[dict] = []
    coordinator = _coordinator(tmp_path, calls)
    callback_class = make_checkpoint_monitor_callback_class(FakeCallback)
    callback = callback_class(coordinator=coordinator)
    args = SimpleNamespace(output_dir=tmp_path / "trainer", max_steps=300, save_steps=100)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(marker="same")
    assert callback.on_train_begin(args, state, control) is control
    for step in (100, 200, 300):
        _checkpoint(Path(args.output_dir), step)
        state.global_step = step
        assert callback.on_save(args, state, control) is control
    assert callback.on_train_end(args, state, control) is control
    assert [Path(call["output_dir"]).name for call in calls] == [
        "checkpoint-100",
        "checkpoint-200",
        "checkpoint-300",
    ]
    assert all(call["expected_checkpoint_sha256"] for call in calls)
    assert callback.initialized


def test_coordinator_rejects_out_of_order_save_before_dispatch(tmp_path):
    calls: list[dict] = []
    coordinator = _coordinator(tmp_path, calls)
    args = SimpleNamespace(output_dir=tmp_path / "trainer", max_steps=300, save_steps=100)
    coordinator.on_train_begin(args, SimpleNamespace(global_step=0))
    _checkpoint(Path(args.output_dir), 200)
    with pytest.raises(RuntimeError, match="expected step 100, found 200"):
        coordinator.on_save(args, SimpleNamespace(global_step=200))
    assert calls == []


def test_coordinator_propagates_monitor_failure_to_abort_training(tmp_path):
    coordinator = _coordinator(tmp_path, [])
    coordinator.runner = lambda **kwargs: (_ for _ in ()).throw(
        CheckpointMonitorError("durable failure")
    )
    args = SimpleNamespace(output_dir=tmp_path / "trainer", max_steps=300, save_steps=100)
    coordinator.on_train_begin(args, SimpleNamespace(global_step=0))
    _checkpoint(Path(args.output_dir), 100)
    with pytest.raises(CheckpointMonitorError, match="durable failure"):
        coordinator.on_save(args, SimpleNamespace(global_step=100))
    assert coordinator._completed == []


def test_quality_threshold_is_rejected_in_phase_f_contract(tmp_path):
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text(encoding="utf-8"))
    value["abort_policy"]["quality_abort_enabled"] = True
    contract.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not enable"):
        CheckpointMonitorCoordinator(
            repo_root=tmp_path,
            contract_path=contract,
            monitor_root=tmp_path / "monitoring",
            command_builder=lambda *_: ["true"],
        )
