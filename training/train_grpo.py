#!/usr/bin/env python3
"""Guarded entry point for GRPO preflight and locked-model loading.

Importing this module deliberately performs no Torch, Transformers, TRL, PEFT,
Unsloth or vLLM import. The model-load-only path imports Unsloth and Torch only
after ``run_preflight`` succeeds. GRPO trainer construction remains unavailable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

PREFLIGHT_VERSION = "grpo-smoke-preflight-v1"
MODEL_LOAD_VERSION = "grpo-smoke-model-load-v1"
DEFAULT_FIXTURE_DATA = "data/train_weak_grpo_smoke_v1.jsonl"
DEFAULT_FIXTURE_MANIFEST = "data/splits/grpo-smoke-v1.json"
DEFAULT_SELECTION_MANIFEST = "runs/sft-selection.json"
DEFAULT_ADAPTER = "runs/sft-combined-2epoch/checkpoint-406"
DEFAULT_OUTPUT_DIR = "runs/grpo-first-smoke"
DEFAULT_MINIMUM_FREE_GIB = 3.0
MODEL_MAX_SEQUENCE_LENGTH = 896

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


def inspect_model_trainability(
    model: object,
    *,
    expected_trainable_parameters: int = LOCKED_TRAINABLE_PARAMETERS,
    expected_target_modules: set[str] = LOCKED_TARGET_MODULES,
) -> dict:
    """Fail unless the loaded policy exposes exactly the locked LoRA for training."""
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("loaded model does not expose named_parameters()")

    total_parameters = 0
    trainable_parameters = 0
    trainable_tensors = 0
    trainable_names = []
    trainable_dtypes: set[str] = set()
    trainable_devices: set[str] = set()
    observed_target_modules: set[str] = set()

    for name, parameter in named_parameters():
        if not isinstance(name, str) or not name:
            raise RuntimeError("loaded model contains an invalid parameter name")
        numel_fn = getattr(parameter, "numel", None)
        if not callable(numel_fn):
            raise TypeError(f"parameter {name} does not expose numel()")
        numel = int(numel_fn())
        if numel < 0:
            raise RuntimeError(f"parameter {name} has a negative size")
        total_parameters += numel

        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        trainable_parameters += numel
        trainable_tensors += 1
        trainable_names.append(name)
        trainable_dtypes.add(str(getattr(parameter, "dtype", "unknown")))
        trainable_devices.add(str(getattr(parameter, "device", "unknown")))
        if "lora_" not in name.lower():
            raise RuntimeError(f"non-LoRA parameter is unexpectedly trainable: {name}")
        matched_targets = {
            module
            for module in expected_target_modules
            if f".{module}." in f".{name}."
        }
        if len(matched_targets) != 1:
            raise RuntimeError(
                f"trainable parameter does not map to one locked target module: {name}"
            )
        observed_target_modules.update(matched_targets)

    if trainable_parameters != expected_trainable_parameters:
        raise RuntimeError(
            "runtime trainable-parameter count mismatch: "
            f"{trainable_parameters} != {expected_trainable_parameters}"
        )
    if observed_target_modules != expected_target_modules:
        raise RuntimeError(
            "runtime LoRA target modules mismatch: "
            f"{sorted(observed_target_modules)} != {sorted(expected_target_modules)}"
        )
    if total_parameters <= trainable_parameters:
        raise RuntimeError("loaded model does not contain a frozen base model")

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": 100 * trainable_parameters / total_parameters,
        "trainable_tensors": trainable_tensors,
        "trainable_parameter_names": trainable_names,
        "trainable_dtypes": sorted(trainable_dtypes),
        "trainable_devices": sorted(trainable_devices),
        "target_modules_observed": sorted(observed_target_modules),
        "only_lora_parameters_trainable": True,
        "matches_locked_trainable_count": True,
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


def _cuda_memory_snapshot(torch_module: object) -> dict:
    """Capture one JSON-safe CUDA memory reading from the active device."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the GRPO model-load gate")
    device_index = int(cuda.current_device())
    properties = cuda.get_device_properties(device_index)
    free_bytes, total_bytes = cuda.mem_get_info(device_index)
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_bytes": int(properties.total_memory),
        "driver_free_bytes": int(free_bytes),
        "driver_used_bytes": int(total_bytes - free_bytes),
        "torch_allocated_bytes": int(cuda.memory_allocated(device_index)),
        "torch_reserved_bytes": int(cuda.memory_reserved(device_index)),
        "torch_peak_allocated_bytes": int(cuda.max_memory_allocated(device_index)),
        "torch_peak_reserved_bytes": int(cuda.max_memory_reserved(device_index)),
    }


def run_model_load_gate(
    *,
    adapter_path: str | Path,
    adapter_file: str | Path,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
) -> dict:
    """Load and inspect the locked policy, then release it without training."""
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()

    # Unsloth must patch the model stack before Torch/Transformers/TRL paths are
    # used. No heavyweight import occurs unless the CPU-only preflight passed.
    from unsloth import FastLanguageModel

    import torch

    model = None
    tokenizer = None
    report = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_memory_snapshot(torch)
    started = time.perf_counter()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),  # Locked PEFT checkpoint, not fresh Qwen.
            max_seq_length=MODEL_MAX_SEQUENCE_LENGTH,  # Measured SFT/GRPO ceiling.
            dtype=torch.bfloat16,  # Match the bf16 SFT policy on the RTX 3090.
            load_in_4bit=False,  # Continue the unquantized SFT adapter unchanged.
            local_files_only=True,  # Refuse downloads or remote revision drift.
            use_gradient_checkpointing="unsloth",  # Match planned GRPO memory mode.
            fast_inference=False,  # Do not start the colocated vLLM path yet.
        )
        torch.cuda.synchronize()
        trainability = inspect_model_trainability(model)
        after_load = _cuda_memory_snapshot(torch)
        adapter_sha_after_load = _sha256_file(adapter_file)
        if adapter_sha_after_load != expected_adapter_sha256:
            raise RuntimeError("source adapter changed during model loading")
        report = {
            "version": MODEL_LOAD_VERSION,
            "status": "passed",
            "adapter_path": str(adapter_path),
            "adapter_file": str(adapter_file),
            "adapter_sha256_after_load": adapter_sha_after_load,
            "source_adapter_unchanged": True,
            "base_model": LOCKED_BASE_MODEL,
            "max_sequence_length": MODEL_MAX_SEQUENCE_LENGTH,
            "dtype_requested": "bfloat16",
            "load_in_4bit": False,
            "local_files_only": True,
            "gradient_checkpointing": "unsloth",
            "fast_inference": False,
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "load_seconds": time.perf_counter() - started,
            "trainability": trainability,
            "cuda_before_load": before,
            "cuda_after_load": after_load,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "generation_performed": False,
            "training_steps": 0,
        }
    finally:
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("model-load gate ended without a report")
    report["cuda_after_release"] = _cuda_memory_snapshot(torch)
    report["model_retained"] = False
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--model-load-only", action="store_true")
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
    if not args.preflight_only and not args.model_load_only:
        raise SystemExit(
            "training is intentionally unavailable; pass --preflight-only or "
            "--model-load-only"
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
    if args.model_load_only:
        report["model_load"] = run_model_load_gate(
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
