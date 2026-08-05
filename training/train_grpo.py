#!/usr/bin/env python3
"""Guarded entry point for GRPO preflight, model loading and trainer construction.

Importing this module deliberately performs no Torch, Transformers, TRL, PEFT,
Unsloth or vLLM import. GPU-capable modes import their stack only after
``run_preflight`` succeeds. Generation and training remain unavailable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

PREFLIGHT_VERSION = "grpo-smoke-preflight-v1"
MODEL_LOAD_VERSION = "grpo-smoke-model-load-v1"
TRAINER_CONSTRUCTION_VERSION = "grpo-smoke-trainer-construction-v1"
ROLLOUT_GATE_VERSION = "grpo-smoke-rollout-gate-v1"
DEFAULT_FIXTURE_DATA = "data/train_weak_grpo_smoke_v1.jsonl"
DEFAULT_FIXTURE_MANIFEST = "data/splits/grpo-smoke-v1.json"
DEFAULT_SELECTION_MANIFEST = "runs/sft-selection.json"
DEFAULT_ADAPTER = "runs/sft-combined-2epoch/checkpoint-406"
DEFAULT_OUTPUT_DIR = "runs/grpo-first-smoke"
DEFAULT_MINIMUM_FREE_GIB = 3.0
MODEL_MAX_SEQUENCE_LENGTH = 896
LOCKED_REWARD_WEIGHTS = (1.0, 1.0, 2.0)

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


def grpo_smoke_config_kwargs(
    *,
    output_dir: str | Path,
    reward_weights: Sequence[float] = LOCKED_REWARD_WEIGHTS,
) -> dict:
    """Return the complete, auditable five-step smoke configuration contract."""
    normalized_weights = tuple(float(weight) for weight in reward_weights)
    if normalized_weights != LOCKED_REWARD_WEIGHTS:
        raise ValueError(
            f"reward weights must remain locked at {LOCKED_REWARD_WEIGHTS}"
        )
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "run_name": "grpo-first-smoke",
        "seed": 42,
        "data_seed": 42,
        "max_prompt_length": 600,
        "max_completion_length": 170,
        "num_generations": 8,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "steps_per_generation": 1,
        "max_steps": 5,
        "shuffle_dataset": False,
        "remove_unused_columns": False,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "use_vllm": False,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "beta": 0.0,
        "num_iterations": 1,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "scale_rewards": "group",
        "loss_type": "dapo",
        "mask_truncated_completions": True,
        "reward_weights": list(normalized_weights),
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "log_completions": True,
        "num_completions_to_print": 8,
        "report_to": "none",
        "save_strategy": "no",
        "save_only_model": True,
    }


def inspect_grpo_config(config: object) -> dict:
    """Assert TRL preserved the locked smoke settings after normalization."""
    expected = grpo_smoke_config_kwargs(output_dir=getattr(config, "output_dir"))
    normalized = {}
    for key, expected_value in expected.items():
        actual = getattr(config, key, None)
        if key == "report_to":
            # Transformers normalizes the user-facing "none" value to [].
            if actual not in ("none", [], ()):  # pragma: no branch - explicit forms
                raise RuntimeError(
                    f"GRPO config normalized {key} unexpectedly: {actual}"
                )
            normalized[key] = [] if actual != "none" else "none"
            continue
        if key == "reward_weights":
            actual = list(actual) if actual is not None else None
        if actual != expected_value:
            raise RuntimeError(
                f"GRPO config drift for {key}: {actual!r} != {expected_value!r}"
            )
        normalized[key] = actual

    generation_batch_size = getattr(config, "generation_batch_size", None)
    if generation_batch_size != 8:
        raise RuntimeError(
            f"generation batch size must be 8, found {generation_batch_size}"
        )
    return {
        "settings": normalized,
        "generation_batch_size": generation_batch_size,
        "prompts_per_generation_batch": generation_batch_size
        // expected["num_generations"],
        "settings_match_locked_contract": True,
    }


def build_rollout_evidence(
    *,
    sku_id: str,
    completions: Sequence[str],
    reward_names: Sequence[str],
    component_rewards: dict[str, Sequence[float]],
    reward_weights: Sequence[float],
    advantages: Sequence[float],
    effective_completion_tokens: Sequence[int],
    truncated_and_masked: Sequence[bool],
) -> dict:
    """Validate and assemble one auditable eight-completion rollout group."""
    expected = 8
    aligned = {
        "completions": len(completions),
        "reward_names": len(reward_names),
        "reward_weights": len(reward_weights),
        "advantages": len(advantages),
        "effective_completion_tokens": len(effective_completion_tokens),
        "truncated_and_masked": len(truncated_and_masked),
    }
    if aligned["completions"] != expected:
        raise RuntimeError(f"rollout must contain eight completions: {aligned}")
    if aligned["reward_names"] != aligned["reward_weights"]:
        raise RuntimeError(f"reward names and weights are misaligned: {aligned}")
    if any(
        aligned[key] != expected
        for key in (
            "advantages",
            "effective_completion_tokens",
            "truncated_and_masked",
        )
    ):
        raise RuntimeError(f"rollout evidence arrays are misaligned: {aligned}")
    if set(component_rewards) != set(reward_names):
        raise RuntimeError("component reward names disagree with trainer order")
    if any(len(component_rewards[name]) != expected for name in reward_names):
        raise RuntimeError("component reward arrays must each contain eight values")
    if any(not isinstance(completion, str) for completion in completions):
        raise TypeError("every logged completion must be text")

    weighted_totals = [
        sum(
            float(component_rewards[name][index]) * float(reward_weights[position])
            for position, name in enumerate(reward_names)
        )
        for index in range(expected)
    ]
    normalized_advantages = [float(value) for value in advantages]
    numeric_values = weighted_totals + normalized_advantages + [
        float(value)
        for name in reward_names
        for value in component_rewards[name]
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError("rollout produced a non-finite reward or advantage")

    records = []
    for index, completion in enumerate(completions):
        records.append(
            {
                "sku_id": sku_id,
                "rollout_index": index,
                "raw_output": completion,
                "component_rewards": {
                    name: float(component_rewards[name][index])
                    for name in reward_names
                },
                "weighted_total": weighted_totals[index],
                "advantage": normalized_advantages[index],
                "effective_completion_tokens": int(
                    effective_completion_tokens[index]
                ),
                "truncated_and_masked": bool(truncated_and_masked[index]),
            }
        )

    return {
        "component_reward_names": list(reward_names),
        "weighted_totals": weighted_totals,
        "weighted_total_unique_count": len(set(weighted_totals)),
        "weighted_total_has_variance": len(set(weighted_totals)) > 1,
        "advantages": normalized_advantages,
        "nonzero_advantage_count": sum(
            not math.isclose(value, 0.0, abs_tol=1e-8)
            for value in normalized_advantages
        ),
        "effective_completion_tokens": [
            int(value) for value in effective_completion_tokens
        ],
        "truncated_and_masked_count": sum(
            bool(value) for value in truncated_and_masked
        ),
        "records": records,
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


def _trainable_parameter_sha256(model: object) -> str:
    """Hash trainable tensor names, metadata and bytes without writing a checkpoint."""
    digest = hashlib.sha256()
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(parameter.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = parameter.detach().contiguous().cpu().numpy().tobytes()
        digest.update(raw)
        digest.update(b"\0")
    if trainable_tensors == 0:
        raise RuntimeError("cannot fingerprint a model with no trainable tensors")
    return digest.hexdigest()


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


def _load_locked_policy(
    FastLanguageModel: object,
    torch_module: object,
    adapter_path: Path,
):
    """Load the selected SFT adapter as trainable PEFT weights, never a fresh LoRA."""
    return FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),  # Locked PEFT checkpoint, not fresh Qwen.
        max_seq_length=MODEL_MAX_SEQUENCE_LENGTH,  # Measured SFT/GRPO ceiling.
        dtype=torch_module.bfloat16,  # Match the bf16 SFT policy on the RTX 3090.
        load_in_4bit=False,  # Continue the unquantized SFT adapter unchanged.
        local_files_only=True,  # Refuse downloads or remote revision drift.
        use_gradient_checkpointing="unsloth",  # Match planned GRPO memory mode.
        fast_inference=False,  # Do not start the colocated vLLM path yet.
    )


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
        model, tokenizer = _load_locked_policy(
            FastLanguageModel, torch, adapter_path
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


def _run_trainer_gate(
    *,
    fixture_data_path: str | Path,
    adapter_path: str | Path,
    adapter_file: str | Path,
    expected_sku_ids: Sequence[str],
    perform_rollout: bool,
    expected_adapter_sha256: str = LOCKED_ADAPTER_SHA256,
) -> dict:
    """Construct the exact trainer and optionally execute one no-loss rollout."""
    fixture_data_path = Path(fixture_data_path).resolve()
    adapter_path = Path(adapter_path).resolve()
    adapter_file = Path(adapter_file).resolve()

    # Import order is part of the remote dependency contract: Unsloth patches
    # the installed TRL/vLLM compatibility path before GRPOTrainer is imported.
    from unsloth import FastLanguageModel

    import torch
    from trl import GRPOConfig, GRPOTrainer

    from training.dataset import load_grpo_prompts
    from training.rewards import (
        FIRST_RUN_REWARD_FUNCTIONS,
        FIRST_RUN_REWARD_WEIGHTS,
    )
    from verifier import load_pack

    if tuple(FIRST_RUN_REWARD_WEIGHTS) != LOCKED_REWARD_WEIGHTS:
        raise RuntimeError("reward implementation weights drifted from GRPO lock")

    model = None
    tokenizer = None
    trainer = None
    dataset = None
    generation_batch = None
    prepared = None
    report = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_memory_snapshot(torch)
    started = time.perf_counter()
    temporary_output_path = None
    try:
        model, tokenizer = _load_locked_policy(
            FastLanguageModel, torch, adapter_path
        )
        trainability_before = inspect_model_trainability(model)

        pack = load_pack("packs/vastraa_taste_v1")
        dataset = load_grpo_prompts(
            pack,
            fixture_data_path,
            require_pass_rate_band=True,
        )
        actual_sku_ids = list(dataset["sku_id"])
        if actual_sku_ids != list(expected_sku_ids):
            raise RuntimeError("trainer dataset SKU order drifted from smoke manifest")
        if len(dataset) != 5:
            raise RuntimeError(
                f"trainer dataset must contain five rows, found {len(dataset)}"
            )
        required_columns = {"prompt", "gold", "sku_id"}
        if set(dataset.column_names) != required_columns:
            raise RuntimeError(
                f"trainer dataset columns drifted: {dataset.column_names}"
            )

        # Trainer construction may initialize logging internals. A temporary
        # output keeps this no-training gate isolated from the reserved smoke path.
        with tempfile.TemporaryDirectory(prefix="grpo-trainer-construction-") as temp:
            temporary_output_path = Path(temp).resolve()
            config = GRPOConfig(
                **grpo_smoke_config_kwargs(
                    output_dir=temporary_output_path,
                    reward_weights=FIRST_RUN_REWARD_WEIGHTS,
                )
            )
            config_report = inspect_grpo_config(config)
            trainer = GRPOTrainer(
                model=model,  # Locked SFT policy with its existing trainable LoRA.
                reward_funcs=list(FIRST_RUN_REWARD_FUNCTIONS),  # Three plain rewards.
                args=config,  # Fully asserted five-step smoke configuration.
                train_dataset=dataset,  # Five deterministic pass-rate-0.5 prompts.
                processing_class=tokenizer,  # Qwen chat template and tokenization.
            )
            torch.cuda.synchronize()

            trainer_reward_names = [
                getattr(reward, "__name__", type(reward).__name__)
                for reward in trainer.reward_funcs
            ]
            expected_reward_names = [
                reward.__name__ for reward in FIRST_RUN_REWARD_FUNCTIONS
            ]
            if trainer_reward_names != expected_reward_names:
                raise RuntimeError("trainer reward function order drifted")
            raw_reward_weights = trainer.reward_weights
            if hasattr(raw_reward_weights, "tolist"):
                raw_reward_weights = raw_reward_weights.tolist()
            runtime_reward_weights = [
                float(weight) for weight in raw_reward_weights
            ]
            if runtime_reward_weights != list(LOCKED_REWARD_WEIGHTS):
                raise RuntimeError("trainer reward weights drifted")
            if trainer.optimizer is not None:
                raise RuntimeError(
                    "trainer construction unexpectedly created an optimizer"
                )
            if trainer.lr_scheduler is not None:
                raise RuntimeError(
                    "trainer construction unexpectedly created an LR scheduler"
                )
            if trainer.ref_model is not None:
                raise RuntimeError(
                    "beta=0 trainer construction unexpectedly created a reference model"
                )
            if int(trainer.state.global_step) != 0:
                raise RuntimeError(
                    "trainer construction unexpectedly advanced global_step"
                )
            trainer_dataset_columns = list(trainer.train_dataset.column_names)
            trainer_sku_ids = list(trainer.train_dataset["sku_id"])
            if set(trainer_dataset_columns) != required_columns:
                raise RuntimeError("trainer dropped a hidden reward/audit column")
            if trainer_sku_ids != actual_sku_ids:
                raise RuntimeError("trainer changed the deterministic SKU order")

            rollout_report = None
            if perform_rollout:
                generation_batch = next(iter(trainer.get_train_dataloader()))
                if not isinstance(generation_batch, list) or len(generation_batch) != 8:
                    raise RuntimeError(
                        "rollout generation batch must be a list of eight rows"
                    )
                generation_sku_ids = [row.get("sku_id") for row in generation_batch]
                expected_first_sku = expected_sku_ids[0]
                if generation_sku_ids != [expected_first_sku] * 8:
                    raise RuntimeError(
                        "first rollout batch must repeat the first locked SKU eight times"
                    )
                if any("gold" not in row for row in generation_batch):
                    raise RuntimeError("rollout generation batch lost hidden gold")

                trainer.model.train()
                lora_sha_before_rollout = _trainable_parameter_sha256(trainer.model)
                cuda_before_rollout = _cuda_memory_snapshot(torch)
                rollout_started = time.perf_counter()
                prepared = trainer._prepare_inputs(generation_batch)
                torch.cuda.synchronize()
                rollout_seconds = time.perf_counter() - rollout_started
                cuda_after_rollout = _cuda_memory_snapshot(torch)
                lora_sha_after_rollout = _trainable_parameter_sha256(trainer.model)
                if lora_sha_after_rollout != lora_sha_before_rollout:
                    raise RuntimeError("trainable LoRA changed during rollout-only gate")
                if int(trainer.state.global_step) != 0:
                    raise RuntimeError("rollout-only gate advanced global_step")
                if trainer.optimizer is not None or trainer.lr_scheduler is not None:
                    raise RuntimeError("rollout-only gate created optimization state")

                completions = list(trainer._logs["completion"])
                advantages = [float(value) for value in trainer._logs["advantages"]]
                component_rewards = {
                    name: [float(value) for value in trainer._logs["rewards"][name]]
                    for name in trainer.reward_func_names
                }
                completion_mask = prepared["completion_mask"].detach().cpu()
                effective_completion_tokens = [
                    int(row.sum().item()) for row in completion_mask
                ]
                truncated = [length == 0 for length in effective_completion_tokens]
                if int(prepared["completion_ids"].shape[0]) != 8:
                    raise RuntimeError("prepared rollout tensor batch is not eight")

                rollout_report = {
                    "status": "passed",
                    "sku_id": expected_first_sku,
                    "generation_batch_rows": len(generation_batch),
                    "unique_generation_batch_skus": sorted(set(generation_sku_ids)),
                    "rollout_seconds": rollout_seconds,
                    "prepared_tensor_keys": sorted(prepared),
                    "lora_sha256_before_rollout": lora_sha_before_rollout,
                    "lora_sha256_after_rollout": lora_sha_after_rollout,
                    "trainable_lora_unchanged": True,
                    "cuda_before_rollout": cuda_before_rollout,
                    "cuda_after_rollout": cuda_after_rollout,
                    **build_rollout_evidence(
                        sku_id=expected_first_sku,
                        completions=completions,
                        reward_names=trainer.reward_func_names,
                        component_rewards=component_rewards,
                        reward_weights=runtime_reward_weights,
                        advantages=advantages,
                        effective_completion_tokens=effective_completion_tokens,
                        truncated_and_masked=truncated,
                    ),
                }

            trainability_after = inspect_model_trainability(trainer.model)
            adapter_sha_after = _sha256_file(adapter_file)
            if adapter_sha_after != expected_adapter_sha256:
                raise RuntimeError("source adapter changed during trainer construction")
            after_construction = _cuda_memory_snapshot(torch)
            report = {
                "version": ROLLOUT_GATE_VERSION
                if perform_rollout
                else TRAINER_CONSTRUCTION_VERSION,
                "status": "passed",
                "trainer_class": type(trainer).__name__,
                "config_class": type(config).__name__,
                "model_class": type(trainer.model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "construction_seconds_including_model_load": time.perf_counter()
                - started,
                "dataset": {
                    "rows": len(dataset),
                    "columns_retained_by_trainer": trainer_dataset_columns,
                    "sku_ids_in_step_order": trainer_sku_ids,
                    "order_matches_manifest": True,
                    "hidden_gold_retained": "gold" in trainer_dataset_columns,
                    "hidden_sku_id_retained": "sku_id" in trainer_dataset_columns,
                },
                "rewards": {
                    "names_in_trainer_order": trainer_reward_names,
                    "weights_in_trainer": runtime_reward_weights,
                    "order_matches_contract": True,
                },
                "config": config_report,
                "trainability_before_trainer": trainability_before,
                "trainability_after_trainer": trainability_after,
                "adapter_sha256_after_construction": adapter_sha_after,
                "source_adapter_unchanged": True,
                "cuda_before_load": before,
                "cuda_after_trainer_construction": after_construction,
                "optimizer_constructed": False,
                "lr_scheduler_constructed": False,
                "reference_model_constructed": False,
                "generation_performed": perform_rollout,
                "training_steps": 0,
                "global_step": int(trainer.state.global_step),
                "temporary_output_path": str(temporary_output_path),
            }
            if rollout_report is not None:
                report["rollout"] = rollout_report

            trainer = None
            config = None

        report["temporary_output_removed"] = not temporary_output_path.exists()
        if not report["temporary_output_removed"]:
            raise RuntimeError("temporary trainer-construction output was not removed")
    finally:
        trainer = None
        dataset = None
        generation_batch = None
        prepared = None
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if report is None:
        raise RuntimeError("trainer-construction gate ended without a report")
    report["cuda_after_release"] = _cuda_memory_snapshot(torch)
    report["trainer_retained"] = False
    report["model_retained"] = False
    return report


def run_trainer_construction_gate(**kwargs) -> dict:
    """Construct and inspect the exact GRPO trainer without generating."""
    return _run_trainer_gate(perform_rollout=False, **kwargs)


def run_rollout_gate(**kwargs) -> dict:
    """Generate and reward one eight-completion group without computing a loss."""
    return _run_trainer_gate(perform_rollout=True, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--model-load-only", action="store_true")
    mode.add_argument("--trainer-construction-only", action="store_true")
    mode.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fixture-data", default=DEFAULT_FIXTURE_DATA)
    parser.add_argument("--fixture-manifest", default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--report-file",
        help="new JSON evidence file; valid only with --rollout-only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (
        args.preflight_only
        or args.model_load_only
        or args.trainer_construction_only
        or args.rollout_only
    ):
        raise SystemExit(
            "training is intentionally unavailable; pass --preflight-only or "
            "--model-load-only or --trainer-construction-only or --rollout-only"
        )
    report_path = None
    if args.report_file is not None:
        if not args.rollout_only:
            raise SystemExit("--report-file is valid only with --rollout-only")
        report_path = _resolve(Path(args.repo_root).resolve(), args.report_file)
        if report_path.exists():
            raise FileExistsError(f"rollout report already exists: {report_path}")
        if not report_path.parent.is_dir():
            raise FileNotFoundError(
                f"rollout report parent does not exist: {report_path.parent}"
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
    if args.trainer_construction_only:
        report["trainer_construction"] = run_trainer_construction_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if args.rollout_only:
        report["rollout_gate"] = run_rollout_gate(
            fixture_data_path=report["fixture"]["data_path"],
            adapter_path=report["sft_lock"]["adapter_path"],
            adapter_file=report["sft_lock"]["adapter_file"],
            expected_sku_ids=report["fixture"]["sku_ids_in_step_order"],
        )
        report["cuda_imports_performed"] = True
        report["model_loaded"] = True
        report["trainer_constructed"] = True
        report["generation_performed"] = True
        report["optimizer_constructed"] = False
        report["training_steps"] = 0
        report["sft_lock"]["runtime_trainable_parameter_assertion_required"] = False
        report["sft_lock"]["runtime_trainable_parameter_assertion_passed"] = True
    if report_path is not None:
        report["report_artifact"] = {
            "path": str(report_path),
            "created": True,
            "overwrite_allowed": False,
        }
    serialized_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path is not None:
        with report_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized_report)
    print(serialized_report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
