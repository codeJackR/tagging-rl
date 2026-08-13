"""GPU evaluator and atomic publication for Run 2 checkpoint monitoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import sha256_file
from training.predict import prompt_messages
from training.run2_checkpoint_monitor import (
    load_development_rows,
    score_monitor_outputs,
    validate_contract_inputs,
)
from verifier import load_pack


VERSION = "grpo-run2-checkpoint-monitor-runtime-v1"
DEFAULT_CONTRACT = "runs/grpo-run2-checkpoint-monitor-contract.json"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
FINAL_FILES = frozenset(
    {"greedy.jsonl", "sampled.jsonl", "report.json", "resource.json", "manifest.json"}
)
CODE_FILES = (
    "training/run2_checkpoint_monitor_contract.py",
    "training/run2_checkpoint_monitor.py",
    "training/run2_checkpoint_monitor_runtime.py",
    "training/run2_checkpoint_monitor_control.py",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(_canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _identity(path: Path, *, display: str | None = None) -> dict[str, Any]:
    return {
        "path": display or path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular directory: {checkpoint}")
    required = {"adapter_config.json", "adapter_model.safetensors"}
    files = {
        path.name: _identity(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file() and not path.is_symlink()
    }
    if not required.issubset(files):
        raise FileNotFoundError("checkpoint lacks PEFT adapter config or weights")
    return {
        "path": str(checkpoint),
        "files": files,
        "adapter_model_sha256": files["adapter_model.safetensors"]["sha256"],
    }


def _git_context(root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for relative in CODE_FILES:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=root,
            check=True,
            capture_output=True,
        )
    if require_clean:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *CODE_FILES],
            cwd=root,
        )
        if result.returncode:
            raise RuntimeError("checkpoint-monitor code differs from Git HEAD")
    return {
        "git_commit": commit,
        "files": {
            relative: _identity(root / relative, display=relative)
            for relative in CODE_FILES
        },
    }


def _cuda_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint monitoring requires CUDA")
    free, total = torch.cuda.mem_get_info()
    return {
        "device_name": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _load_policy(base_model: str, checkpoint: Path, *, local_files_only: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return torch, model, tokenizer


def _generation_kwargs(
    *,
    tokenizer: Any,
    max_completion_length: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    generation = {
        "max_new_tokens": max_completion_length,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        if temperature is None or top_p is None:
            raise ValueError("sampled decoding requires temperature and top-p")
        generation.update({"temperature": temperature, "top_p": top_p})
    else:
        # Qwen's generation_config.json carries non-neutral sampling defaults.
        # They do not affect greedy decoding, but explicitly neutralizing them
        # prevents an ambiguous Transformers warning in the audit stream.
        generation.update({"temperature": 1.0, "top_p": 1.0, "top_k": 50})
    return generation


def _generate(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Any],
    batch_size: int,
    max_prompt_length: int,
    max_completion_length: int,
    do_sample: bool,
    seed: int | None,
    temperature: float | None = None,
    top_p: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(row),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(model.device)
        generation = _generation_kwargs(
            tokenizer=tokenizer,
            max_completion_length=max_completion_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation)
        prompt_width = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        outputs.extend(
            {"sku_id": row.sku_id, "raw": raw}
            for row, raw in zip(batch, completions, strict=True)
        )
        del generated, encoded
        if progress is not None:
            progress(min(start + len(batch), len(rows)), len(rows))
    return outputs


def publish_monitor_bundle(
    *,
    output_dir: str | Path,
    greedy: Sequence[Mapping[str, Any]],
    sampled: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    resource: Mapping[str, Any],
    manifest_context: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if any("confirmation" in part for part in output_dir.parts):
        raise ValueError("checkpoint-monitor output cannot use a confirmation path")
    if output_dir.exists():
        raise FileExistsError(f"checkpoint-monitor output exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"checkpoint-monitor output parent is absent: {output_dir.parent}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        greedy_path = staging / "greedy.jsonl"
        sampled_path = staging / "sampled.jsonl"
        report_path = staging / "report.json"
        resource_path = staging / "resource.json"
        manifest_path = staging / "manifest.json"
        _write_jsonl(greedy_path, greedy)
        _write_jsonl(sampled_path, sampled)
        _write_json(report_path, report)
        _write_json(resource_path, resource)
        manifest = {
            "version": VERSION,
            "status": "checkpoint_monitor_complete",
            **dict(manifest_context),
            "files": {
                "greedy_predictions": _identity(greedy_path),
                "sampled_predictions": _identity(sampled_path),
                "scored_report": _identity(report_path),
                "resource_report": _identity(resource_path),
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
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != FINAL_FILES:
            raise RuntimeError("checkpoint-monitor staging inventory drifted")
        os.rename(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_checkpoint_evaluation(
    *,
    root: str | Path,
    contract_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    mode: str,
    base_model: str = DEFAULT_BASE_MODEL,
    pack_path: str | Path = DEFAULT_PACK,
    local_files_only: bool = False,
    checkpoint_created_by_smoke: bool = False,
    checkpoint_save_resource: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    contract_path = (root / contract_path).resolve() if not Path(contract_path).is_absolute() else Path(contract_path)
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if any("confirmation" in part for part in checkpoint.parts):
        raise ValueError("checkpoint-monitor checkpoint path crosses confirmation boundary")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract_inputs(contract, root)
    rows, views = load_development_rows(root=root, contract=contract, mode=mode)
    pack = load_pack(root / pack_path)
    checkpoint_identity = _checkpoint_identity(checkpoint)
    code = _git_context(root)
    sampled = contract["decoding"]["sampled"]
    seeds = (
        contract["smoke"]["sampled_seeds"]
        if mode == "smoke"
        else sampled["seeds"]
    )
    batch_size = contract["decoding"]["batch_size"]
    max_prompt = contract["decoding"]["max_prompt_length"]
    max_completion = contract["decoding"]["max_completion_length"]

    torch = model = tokenizer = None
    greedy_outputs = []
    sampled_outputs = []
    timing: dict[str, float] = {}
    started = time.perf_counter()
    try:
        import torch as imported_torch

        torch = imported_torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_load = _cuda_snapshot(torch)
        phase = time.perf_counter()
        torch, model, tokenizer = _load_policy(
            base_model, checkpoint, local_files_only=local_files_only
        )
        torch.cuda.synchronize()
        timing["model_load_seconds"] = time.perf_counter() - phase
        after_load = _cuda_snapshot(torch)

        phase = time.perf_counter()
        greedy_outputs = _generate(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            batch_size=batch_size,
            max_prompt_length=max_prompt,
            max_completion_length=max_completion,
            do_sample=False,
            seed=None,
        )
        torch.cuda.synchronize()
        timing["greedy_generation_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        for repeat, seed in enumerate(seeds):
            generated = _generate(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                batch_size=batch_size,
                max_prompt_length=max_prompt,
                max_completion_length=max_completion,
                do_sample=True,
                seed=seed,
                temperature=sampled["temperature"],
                top_p=sampled["top_p"],
            )
            sampled_outputs.extend(
                {**item, "repeat": repeat, "seed": seed} for item in generated
            )
        torch.cuda.synchronize()
        timing["sampled_generation_seconds"] = time.perf_counter() - phase
        peak = _cuda_snapshot(torch)
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after_release = _cuda_snapshot(torch)
        else:
            after_release = None
    if torch is None or after_release is None:
        raise RuntimeError("checkpoint evaluator did not obtain CUDA cleanup evidence")
    if after_release["allocated_bytes"] > before_load["allocated_bytes"] + 64 * 1024**2:
        raise RuntimeError("checkpoint evaluator retained material CUDA allocations")

    phase = time.perf_counter()
    scored = score_monitor_outputs(
        rows=rows,
        views=views,
        greedy_predictions=greedy_outputs,
        sampled_predictions=sampled_outputs,
        sampled_seeds=seeds,
        pack=pack,
    )
    timing["cpu_scoring_seconds"] = time.perf_counter() - phase
    timing["total_seconds"] = time.perf_counter() - started
    resource = {
        "version": VERSION,
        "cuda_before_load": before_load,
        "cuda_after_load": after_load,
        "cuda_peak": peak,
        "cuda_after_release": after_release,
        "model_reference_released": True,
        "tokenizer_reference_released": True,
        "cuda_cleanup_within_64_mib_of_start": True,
        "timing": timing,
        "checkpoint_save_resource": dict(checkpoint_save_resource or {}),
    }
    context = {
        "mode": mode,
        "quality_evidence": mode == "production",
        "checkpoint": checkpoint_identity,
        "checkpoint_created_by_smoke": checkpoint_created_by_smoke,
        "contract": _identity(contract_path, display=str(contract_path.relative_to(root))),
        "code": code,
        "configuration": {
            "base_model": base_model,
            "rows": len(rows),
            "view_rows": {name: len(skus) for name, skus in views.items()},
            "batch_size": batch_size,
            "max_prompt_length": max_prompt,
            "max_completion_length": max_completion,
            "greedy_repetitions": 1,
            "sampled_repetitions": len(seeds),
            "sampled_seeds": list(seeds),
            "temperature": sampled["temperature"],
            "top_p": sampled["top_p"],
        },
        "execution_boundary": {
            "confirmation_data_used": False,
            "full_grpo_training_dispatched": False,
            "quality_abort_threshold_applied": False,
        },
    }
    return publish_monitor_bundle(
        output_dir=output_dir,
        greedy=greedy_outputs,
        sampled=sampled_outputs,
        report=scored,
        resource=resource,
        manifest_context=context,
    )


def create_smoke_checkpoint(
    *,
    source_adapter: Path,
    checkpoint: Path,
    base_model: str,
    local_files_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint.exists():
        raise FileExistsError(f"smoke checkpoint exists: {checkpoint}")
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_load = _cuda_snapshot(torch)
    started = time.perf_counter()
    model = tokenizer = None
    try:
        torch, model, tokenizer = _load_policy(
            base_model, source_adapter, local_files_only=local_files_only
        )
        torch.cuda.synchronize()
        after_load = _cuda_snapshot(torch)
        model.save_pretrained(checkpoint, safe_serialization=True)
        torch.cuda.synchronize()
        saved = _checkpoint_identity(checkpoint)
        after_save = _cuda_snapshot(torch)
    finally:
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_release = _cuda_snapshot(torch)
    if after_release["allocated_bytes"] > before_load["allocated_bytes"] + 64 * 1024**2:
        raise RuntimeError("smoke checkpoint save retained material CUDA allocations")
    return saved, {
        "cuda_before_save_load": before_load,
        "cuda_after_save_load": after_load,
        "cuda_after_save": after_save,
        "cuda_after_save_release": after_release,
        "cuda_cleanup_within_64_mib_of_start": True,
        "save_seconds": time.perf_counter() - started,
        "checkpoint_saved": True,
        "save_model_reference_released": True,
    }


def run_gpu_smoke(
    *,
    root: Path,
    contract_path: Path,
    source_adapter: Path,
    output_dir: Path,
    base_model: str,
    pack_path: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"smoke output exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix="run2-monitor-smoke-checkpoint-", dir=output_dir.parent) as temporary:
        checkpoint = Path(temporary) / "checkpoint-smoke"
        saved, save_resource = create_smoke_checkpoint(
            source_adapter=source_adapter,
            checkpoint=checkpoint,
            base_model=base_model,
            local_files_only=local_files_only,
        )
        manifest = run_checkpoint_evaluation(
            root=root,
            contract_path=contract_path,
            checkpoint=checkpoint,
            output_dir=output_dir,
            mode="smoke",
            base_model=base_model,
            pack_path=pack_path,
            local_files_only=local_files_only,
            checkpoint_created_by_smoke=True,
            checkpoint_save_resource={**save_resource, "saved_checkpoint": saved},
        )
    if checkpoint.exists():
        raise RuntimeError("temporary smoke checkpoint survived cleanup")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--mode", choices=("production", "smoke"), default="production")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--source-adapter", required=True)
    smoke.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    if args.command == "evaluate":
        result = run_checkpoint_evaluation(
            root=root,
            contract_path=root / args.contract,
            checkpoint=args.checkpoint,
            output_dir=args.output,
            mode=args.mode,
            base_model=args.base_model,
            pack_path=args.pack,
            local_files_only=args.local_files_only,
        )
    else:
        result = run_gpu_smoke(
            root=root,
            contract_path=root / args.contract,
            source_adapter=Path(args.source_adapter).resolve(),
            output_dir=Path(args.output).resolve(),
            base_model=args.base_model,
            pack_path=Path(args.pack),
            local_files_only=args.local_files_only,
        )
    print(json.dumps({"status": result["status"], "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
