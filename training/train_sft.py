#!/usr/bin/env python3
"""Train one controlled LoRA-SFT arm, or run its five-step smoke test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def target_modules(arm: str) -> list[str]:
    if arm == "attention":
        return ATTENTION_MODULES
    if arm == "combined":
        return ATTENTION_MODULES + MLP_MODULES
    raise ValueError(f"unknown SFT arm: {arm}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["attention", "combined"], required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--manifest", default="data/splits/sft-v1.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Unsloth must patch the stack before transformers, PEFT, or TRL is imported.
    from unsloth import FastLanguageModel

    from trl import SFTConfig, SFTTrainer

    from training.dataset import MAX_SFT_TOKENS, load_sft_splits
    from verifier import load_pack

    output = Path(
        args.output_dir
        or f"runs/sft-{args.arm}{'-smoke' if args.smoke else ''}"
    )
    output.mkdir(parents=True, exist_ok=True)

    pack = load_pack("packs/vastraa_taste_v1")
    train_dataset, eval_dataset = load_sft_splits(pack, args.manifest)
    if args.smoke:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(32, len(eval_dataset))))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # Cached Qwen base model shared by both SFT arms.
        max_seq_length=MAX_SFT_TOKENS,  # 896: covers every rendered train row.
        load_in_4bit=False,  # Keep bf16 base weights; 1.5B fits on the 24 GB 3090.
    )
    model = FastLanguageModel.get_peft_model(
        model,  # Freeze this base model and attach trainable LoRA adapters to it.
        r=16,  # LoRA's narrow middle dimension; shared by both experiment arms.
        target_modules=target_modules(args.arm),  # Attention-only or attention+MLP.
        lora_alpha=16,  # Scale the LoRA update; alpha/rank = 1 for this baseline.
        lora_dropout=0,  # Unsloth's optimized path expects zero; data is already varied.
        bias="none",  # Do not train original layer biases—only LoRA parameters.
        use_gradient_checkpointing="unsloth",  # Recompute activations to save VRAM.
        random_state=args.seed,  # Make adapter initialization reproducible.
    )
    model.print_trainable_parameters()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "tagging-rl")

    config = SFTConfig(
        output_dir=str(output),  # Checkpoints, trainer state, and local run artifacts.
        run_name=f"sft-{args.arm}{'-smoke' if args.smoke else ''}",  # W&B label.
        seed=args.seed,  # Reproduce model-side randomness such as data sampling.
        data_seed=args.seed,  # Reproduce the trainer's ordering of training examples.
        max_length=MAX_SFT_TOKENS,  # Truncate only beyond the measured 896 ceiling.
        completion_only_loss=True,  # Learn from assistant JSON, not prompt tokens.
        packing=False,  # Keep each product as its own sequence for a clear baseline.
        per_device_train_batch_size=4,  # Four examples fit on the GPU simultaneously.
        per_device_eval_batch_size=4,  # Match the memory-safe validation batch size.
        gradient_accumulation_steps=4,  # Four micro-batches make one model update.
        learning_rate=1e-4,  # Starting step size for the trainable LoRA parameters.
        warmup_ratio=0.05,  # Ramp up LR during the first 5% of optimizer updates.
        lr_scheduler_type="cosine",  # Smoothly reduce LR after the warmup period.
        optim="adamw_torch_fused",  # Fast standard AdamW implementation from PyTorch.
        num_train_epochs=args.epochs,  # At most two full passes in the real run.
        max_steps=5 if args.smoke else -1,  # Smoke stops at five; -1 uses epochs.
        logging_steps=1 if args.smoke else 10,  # Dense smoke logs, lighter real logs.
        logging_first_step=True,  # Capture initial loss and gradients immediately.
        eval_strategy="steps" if args.smoke else "epoch",  # When validation runs.
        eval_steps=5 if args.smoke else None,  # Validate at smoke completion only.
        save_strategy="no" if args.smoke else "epoch",  # Real run saves each epoch.
        save_total_limit=2,  # Bound disk use to the two most recent checkpoints.
        save_only_model=True,  # Save adapter weights, not resumable optimizer state.
        bf16=True,  # Ampere-native 16-bit math with better range than fp16.
        fp16=False,  # Avoid enabling a second, conflicting 16-bit mode.
        report_to=args.report_to,  # Either local-only smoke logging or W&B.
    )
    trainer = SFTTrainer(
        model=model,  # Qwen with the selected trainable LoRA adapters attached.
        processing_class=tokenizer,  # Apply Qwen's chat template and tokenization.
        args=config,  # All optimization, evaluation, logging, and saving behavior.
        train_dataset=train_dataset,  # Frozen manifest's 3,240 training rows.
        eval_dataset=eval_dataset,  # Frozen manifest's 360 validation rows.
    )

    import torch

    # Measure this training call rather than model-loading and tokenization peaks.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    wall_seconds = time.perf_counter() - started
    gib = 1024**3
    peak_allocated_gib = torch.cuda.max_memory_allocated() / gib
    peak_reserved_gib = torch.cuda.max_memory_reserved() / gib

    adapter_dir = output / "final-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    disk = shutil.disk_usage(output)
    summary = {
        "arm": args.arm,
        "smoke": args.smoke,
        "epochs_requested": args.epochs,
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset),
        "target_modules": target_modules(args.arm),
        "trainable_parameters": 4_358_144 if args.arm == "attention" else 18_464_768,
        "training_loss": result.training_loss,
        "trainer_metrics": result.metrics,
        "wall_seconds": wall_seconds,
        "peak_gpu_allocated_gib": peak_allocated_gib,
        "peak_gpu_reserved_gib": peak_reserved_gib,
        "disk_free_gib_after": disk.free / gib,
        "adapter_dir": str(adapter_dir),
    }
    (output / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    trainer.log(
        {
            "peak_gpu_allocated_gib": peak_allocated_gib,
            "peak_gpu_reserved_gib": peak_reserved_gib,
            "wall_seconds_measured": wall_seconds,
        }
    )
    print(
        f"SFT {args.arm} {'smoke' if args.smoke else 'run'} complete: "
        f"loss={result.training_loss:.6f}, "
        f"peak_allocated={peak_allocated_gib:.2f} GiB, "
        f"peak_reserved={peak_reserved_gib:.2f} GiB, "
        f"adapter={adapter_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
