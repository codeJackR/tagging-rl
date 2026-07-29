#!/usr/bin/env python3
"""Train one controlled LoRA-SFT arm, or run its five-step smoke test."""

from __future__ import annotations

import argparse
import os
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
        model_name=args.model,
        max_seq_length=MAX_SFT_TOKENS,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=target_modules(args.arm),
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "tagging-rl")

    config = SFTConfig(
        output_dir=str(output),
        run_name=f"sft-{args.arm}{'-smoke' if args.smoke else ''}",
        seed=args.seed,
        data_seed=args.seed,
        max_length=MAX_SFT_TOKENS,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        num_train_epochs=args.epochs,
        max_steps=5 if args.smoke else -1,
        logging_steps=1 if args.smoke else 10,
        logging_first_step=True,
        eval_strategy="steps" if args.smoke else "epoch",
        eval_steps=5 if args.smoke else None,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=2,
        save_only_model=True,
        bf16=True,
        fp16=False,
        report_to=args.report_to,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    result = trainer.train()

    adapter_dir = output / "final-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(
        f"SFT {args.arm} {'smoke' if args.smoke else 'run'} complete: "
        f"loss={result.training_loss:.6f}, adapter={adapter_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
