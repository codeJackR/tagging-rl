#!/usr/bin/env python3
"""Guarded entry point for GRPO; currently implements preflight only.

This module deliberately has no Torch, Transformers, TRL, PEFT, Unsloth or vLLM
imports. The first CUDA-capable import belongs after ``run_preflight`` succeeds
in the future training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

PREFLIGHT_VERSION = "grpo-smoke-preflight-v1"
DEFAULT_FIXTURE_DATA = "data/train_weak_grpo_smoke_v1.jsonl"
DEFAULT_FIXTURE_MANIFEST = "data/splits/grpo-smoke-v1.json"
DEFAULT_SELECTION_MANIFEST = "runs/sft-selection.json"
DEFAULT_ADAPTER = "runs/sft-combined-2epoch/checkpoint-406"
DEFAULT_OUTPUT_DIR = "runs/grpo-first-smoke"
DEFAULT_MINIMUM_FREE_GIB = 3.0

LOCKED_FIXTURE_DATA_SHA256 = (
    "268373ceb08c53125976493340d972a47c90e10911e919002716590f75ca4084"
)
LOCKED_FIXTURE_MANIFEST_SHA256 = (
    "e898510534d967b9a35367e0aba5a564e6cb564e2c326d45804d0319d528dd05"
)
LOCKED_SELECTION_MANIFEST_SHA256 = (
    "e425635d323b3ffe9e7350fb61a2d9e1848345a95abab6b92032bf64d2718299"
)
LOCKED_ADAPTER_SHA256 = (
    "00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af"
)
LOCKED_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
LOCKED_TRAINABLE_PARAMETERS = 18_464_768
LOCKED_LORA_RANK = 16
LOCKED_LORA_ALPHA = 16
LOCKED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sku_sha256(sku_ids: Sequence[str]) -> str:
    payload = "\n".join(sku_ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def inspect_git_state(repo_root: Path) -> dict:
    """Resolve the exact tracked code state while allowing untracked run files."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_dirty = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            check=False,
        ).returncode != 0
        index_dirty = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
            check=False,
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not resolve Git state") from exc
    return {
        "commit": commit,
        "tracked_worktree_dirty": worktree_dirty,
        "index_dirty": index_dirty,
    }


def verify_fixture(
    *,
    fixture_data_path: Path,
    fixture_manifest_path: Path,
    expected_data_sha256: str,
    expected_manifest_sha256: str,
) -> dict:
    actual_manifest_sha = _sha256_file(fixture_manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise RuntimeError("locked smoke fixture manifest checksum mismatch")
    manifest = _read_json(fixture_manifest_path)
    if manifest.get("version") != "grpo-smoke-v1":
        raise RuntimeError("unexpected smoke fixture manifest version")
    if not all(manifest.get("invariants", {}).values()):
        raise RuntimeError("smoke fixture manifest contains a failed invariant")

    actual_data_sha = _sha256_file(fixture_data_path)
    if actual_data_sha != expected_data_sha256:
        raise RuntimeError("locked smoke fixture data checksum mismatch")
    if actual_data_sha != manifest.get("output", {}).get("smoke_dataset_sha256"):
        raise RuntimeError("smoke fixture data disagrees with its manifest")

    rows = _read_jsonl_objects(fixture_data_path)
    expected_rows = manifest.get("selection", {}).get("selected_rows")
    if len(rows) != expected_rows or len(rows) != 5:
        raise RuntimeError("smoke fixture must contain exactly five rows")
    sku_ids = [row.get("sku_id") for row in rows]
    if any(not isinstance(sku_id, str) or not sku_id for sku_id in sku_ids):
        raise RuntimeError("smoke fixture contains a missing SKU ID")
    if len(set(sku_ids)) != len(sku_ids):
        raise RuntimeError("smoke fixture contains duplicate SKU IDs")
    if sku_ids != manifest["selection"]["selected_skus_in_step_order"]:
        raise RuntimeError("smoke fixture SKU order disagrees with its manifest")
    if _ordered_sku_sha256(sku_ids) != manifest["selection"][
        "selected_sku_order_sha256"
    ]:
        raise RuntimeError("smoke fixture ordered-SKU checksum mismatch")
    if any(row.get("split") != "train" for row in rows):
        raise RuntimeError("smoke fixture contains a non-training row")
    if any(row.get("difficulty", {}).get("sft_pass_rate") != 0.5 for row in rows):
        raise RuntimeError("smoke fixture contains a row outside pass rate 0.5")

    return {
        "data_path": str(fixture_data_path),
        "data_sha256": actual_data_sha,
        "manifest_path": str(fixture_manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "rows": len(rows),
        "ordered_sku_sha256": manifest["selection"][
            "selected_sku_order_sha256"
        ],
        "sku_ids_in_step_order": sku_ids,
    }


def verify_sft_lock(
    *,
    repo_root: Path,
    selection_manifest_path: Path,
    adapter_path: Path,
    expected_selection_sha256: str,
    expected_adapter_sha256: str,
) -> dict:
    actual_selection_sha = _sha256_file(selection_manifest_path)
    if actual_selection_sha != expected_selection_sha256:
        raise RuntimeError("SFT selection manifest checksum mismatch")
    selection = _read_json(selection_manifest_path)
    if selection.get("status") != "locked_before_frozen_eval":
        raise RuntimeError("SFT selection manifest is not in the locked state")

    selected = selection.get("selected_checkpoint", {})
    locked_adapter_path = _resolve(repo_root, selected.get("remote_path", ""))
    if locked_adapter_path != adapter_path:
        raise RuntimeError("adapter path disagrees with the SFT selection lock")
    if selected.get("base_model") != LOCKED_BASE_MODEL:
        raise RuntimeError("base model disagrees with the SFT selection lock")

    lora = selected.get("lora", {})
    if lora.get("rank") != LOCKED_LORA_RANK:
        raise RuntimeError("LoRA rank disagrees with the SFT selection lock")
    if lora.get("alpha") != LOCKED_LORA_ALPHA:
        raise RuntimeError("LoRA alpha disagrees with the SFT selection lock")
    if set(lora.get("target_modules", [])) != LOCKED_TARGET_MODULES:
        raise RuntimeError("LoRA targets disagree with the SFT selection lock")
    if lora.get("trainable_parameters") != LOCKED_TRAINABLE_PARAMETERS:
        raise RuntimeError("trainable-parameter expectation disagrees with lock")

    weights = selected.get("adapter_weights", {})
    adapter_file = adapter_path / weights.get("file", "adapter_model.safetensors")
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    actual_adapter_sha = _sha256_file(adapter_file)
    if actual_adapter_sha != expected_adapter_sha256:
        raise RuntimeError("locked SFT adapter checksum mismatch")
    if actual_adapter_sha != weights.get("sha256"):
        raise RuntimeError("adapter checksum disagrees with SFT selection manifest")
    if adapter_file.stat().st_size != weights.get("bytes"):
        raise RuntimeError("adapter byte size disagrees with SFT selection manifest")

    adapter_config_path = adapter_path / "adapter_config.json"
    config = _read_json(adapter_config_path)
    if config.get("base_model_name_or_path") != LOCKED_BASE_MODEL:
        raise RuntimeError("adapter config names an unexpected base model")
    if config.get("r") != LOCKED_LORA_RANK:
        raise RuntimeError("adapter config has an unexpected LoRA rank")
    if config.get("lora_alpha") != LOCKED_LORA_ALPHA:
        raise RuntimeError("adapter config has an unexpected LoRA alpha")
    if config.get("lora_dropout") != 0 or config.get("bias") != "none":
        raise RuntimeError("adapter config has unexpected dropout or bias")
    if set(config.get("target_modules", [])) != LOCKED_TARGET_MODULES:
        raise RuntimeError("adapter config has unexpected target modules")

    return {
        "selection_manifest": str(selection_manifest_path),
        "selection_manifest_sha256": actual_selection_sha,
        "adapter_path": str(adapter_path),
        "adapter_file": str(adapter_file),
        "adapter_bytes": adapter_file.stat().st_size,
        "adapter_sha256": actual_adapter_sha,
        "adapter_config": str(adapter_config_path),
        "base_model": LOCKED_BASE_MODEL,
        "lora_rank": LOCKED_LORA_RANK,
        "lora_alpha": LOCKED_LORA_ALPHA,
        "target_modules": sorted(LOCKED_TARGET_MODULES),
        "trainable_parameters_expected": LOCKED_TRAINABLE_PARAMETERS,
        "runtime_trainable_parameter_assertion_required": True,
    }


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"no existing parent for disk check: {path}")
    return candidate


def run_preflight(
    *,
    repo_root: str | Path,
    fixture_data: str | Path,
    fixture_manifest: str | Path,
    selection_manifest: str | Path,
    adapter: str | Path,
    output_dir: str | Path,
    minimum_free_bytes: int,
    expected_commit: str | None = None,
    expected_fixture_data_sha256: str = LOCKED_FIXTURE_DATA_SHA256,
    expected_fixture_manifest_sha256: str = LOCKED_FIXTURE_MANIFEST_SHA256,
    expected_selection_manifest_sha256: str = LOCKED_SELECTION_MANIFEST_SHA256,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
    git_state_fn: Callable[[Path], dict] | None = None,
    disk_usage_fn: Callable[[Path], object] | None = None,
) -> dict:
    """Validate every CPU-visible smoke lock and return a read-only report."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    fixture_data_path = _resolve(repo_root, fixture_data)
    fixture_manifest_path = _resolve(repo_root, fixture_manifest)
    selection_manifest_path = _resolve(repo_root, selection_manifest)
    adapter_path = _resolve(repo_root, adapter)
    output_path = _resolve(repo_root, output_dir)

    git = (git_state_fn or inspect_git_state)(repo_root)
    if git.get("tracked_worktree_dirty") or git.get("index_dirty"):
        raise RuntimeError("tracked Git state must be clean before GRPO")
    if expected_commit is not None and git.get("commit") != expected_commit:
        raise RuntimeError("Git commit disagrees with expected smoke commit")
    if output_path.exists():
        raise FileExistsError(f"GRPO smoke output already exists: {output_path}")

    fixture = verify_fixture(
        fixture_data_path=fixture_data_path,
        fixture_manifest_path=fixture_manifest_path,
        expected_data_sha256=expected_fixture_data_sha256,
        expected_manifest_sha256=expected_fixture_manifest_sha256,
    )
    sft_lock = verify_sft_lock(
        repo_root=repo_root,
        selection_manifest_path=selection_manifest_path,
        adapter_path=adapter_path,
        expected_selection_sha256=expected_selection_manifest_sha256,
        expected_adapter_sha256=expected_adapter_sha256,
    )

    if minimum_free_bytes <= 0:
        raise ValueError("minimum free disk must be positive")
    disk_probe = _existing_parent(output_path.parent)
    usage = (disk_usage_fn or shutil.disk_usage)(disk_probe)
    free_bytes = int(usage.free)
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} < {minimum_free_bytes} bytes"
        )

    return {
        "version": PREFLIGHT_VERSION,
        "status": "passed",
        "git": git,
        "fixture": fixture,
        "sft_lock": sft_lock,
        "output": {
            "path": str(output_path),
            "collision_free": True,
            "created": False,
        },
        "disk": {
            "probe_path": str(disk_probe),
            "free_bytes": free_bytes,
            "minimum_free_bytes": minimum_free_bytes,
            "passes": True,
        },
        "cuda_imports_performed": False,
        "model_loaded": False,
        "trainer_constructed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fixture-data", default=DEFAULT_FIXTURE_DATA)
    parser.add_argument("--fixture-manifest", default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--expected-commit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.preflight_only:
        raise SystemExit(
            "training is intentionally unavailable; pass --preflight-only"
        )
    report = run_preflight(
        repo_root=args.repo_root,
        fixture_data=args.fixture_data,
        fixture_manifest=args.fixture_manifest,
        selection_manifest=args.selection_manifest,
        adapter=args.adapter,
        output_dir=args.output_dir,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
        expected_commit=args.expected_commit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
