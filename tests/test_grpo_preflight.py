from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.train_grpo import (
    LOCKED_BASE_MODEL,
    LOCKED_TARGET_MODULES,
    LOCKED_TRAINABLE_PARAMETERS,
    main,
    parse_args,
    run_preflight,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATA = ROOT / "data" / "train_weak_grpo_smoke_v1.jsonl"
FIXTURE_MANIFEST = ROOT / "data" / "splits" / "grpo-smoke-v1.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sft_lock(tmp_path: Path):
    adapter = tmp_path / "checkpoint-406"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"small fake adapter for CPU preflight")
    weights_sha = sha256_file(weights)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": LOCKED_BASE_MODEL,
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "bias": "none",
                "target_modules": sorted(LOCKED_TARGET_MODULES),
            }
        ),
        encoding="utf-8",
    )
    selection = {
        "status": "locked_before_frozen_eval",
        "selected_checkpoint": {
            "remote_path": str(adapter),
            "base_model": LOCKED_BASE_MODEL,
            "adapter_weights": {
                "file": weights.name,
                "bytes": weights.stat().st_size,
                "sha256": weights_sha,
            },
            "lora": {
                "rank": 16,
                "alpha": 16,
                "target_modules": sorted(LOCKED_TARGET_MODULES),
                "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
            },
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return adapter, selection_path, weights_sha


def passing_kwargs(tmp_path: Path) -> dict:
    adapter, selection, weights_sha = make_sft_lock(tmp_path)
    return {
        "repo_root": ROOT,
        "fixture_data": FIXTURE_DATA,
        "fixture_manifest": FIXTURE_MANIFEST,
        "selection_manifest": selection,
        "adapter": adapter,
        "output_dir": tmp_path / "new-output",
        "minimum_free_bytes": 3 * 1024**3,
        "expected_commit": "a" * 40,
        "expected_selection_manifest_sha256": sha256_file(selection),
        "expected_adapter_sha256": weights_sha,
        "git_state_fn": lambda _: {
            "commit": "a" * 40,
            "tracked_worktree_dirty": False,
            "index_dirty": False,
        },
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
    }


def test_entrypoint_is_cpu_only_and_training_is_unavailable():
    source_path = ROOT / "training" / "train_grpo.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"torch", "transformers", "trl", "peft", "unsloth", "vllm"}
    )
    assert parse_args(["--preflight-only"]).preflight_only
    with pytest.raises(SystemExit, match="training is intentionally unavailable"):
        main([])


def test_preflight_passes_without_creating_output_or_loading_cuda(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    report = run_preflight(**kwargs)

    assert report["status"] == "passed"
    assert report["fixture"]["rows"] == 5
    assert report["sft_lock"]["trainable_parameters_expected"] == 18_464_768
    assert report["sft_lock"]["runtime_trainable_parameter_assertion_required"]
    assert report["disk"]["passes"]
    assert not report["output"]["created"]
    assert not Path(report["output"]["path"]).exists()
    assert not report["cuda_imports_performed"]
    assert not report["model_loaded"]
    assert not report["trainer_constructed"]


def test_preflight_rejects_dirty_or_unexpected_git_state(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    kwargs["git_state_fn"] = lambda _: {
        "commit": "a" * 40,
        "tracked_worktree_dirty": True,
        "index_dirty": False,
    }
    with pytest.raises(RuntimeError, match="tracked Git state must be clean"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    kwargs["expected_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="commit disagrees"):
        run_preflight(**kwargs)


def test_preflight_rejects_fixture_or_adapter_drift(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    kwargs["expected_fixture_data_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fixture data checksum"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    Path(kwargs["adapter"], "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="adapter checksum"):
        run_preflight(**kwargs)


def test_preflight_rejects_adapter_config_drift(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    config_path = Path(kwargs["adapter"], "adapter_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected LoRA rank"):
        run_preflight(**kwargs)


def test_preflight_rejects_output_collision_and_low_disk(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    Path(kwargs["output_dir"]).mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    kwargs["disk_usage_fn"] = lambda _: SimpleNamespace(free=2 * 1024**3)
    with pytest.raises(RuntimeError, match="insufficient free disk"):
        run_preflight(**kwargs)
