#!/usr/bin/env python3
"""Generate raw, unconstrained predictions for the tagging eval harness.

Example on the GPU box:

    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/rl/bin/python \
      -m training.predict \
      --model unsloth/Qwen2.5-1.5B-Instruct \
      --input data/eval_300/eval.jsonl \
      --output runs/qwen2.5-1.5b-zero-shot.jsonl \
      --local-files-only

The output deliberately stores the model's literal text rather than parsed JSON.
Parsing before evaluation would erase schema failures and make the validity
measurement meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from labeling.lengths import render_prompt
from labeling.records import read_jsonl
from training.dataset import SYSTEM


def prompt_messages(row) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render_prompt(row)},
    ]


def filter_rows(rows, manifest_path: str | Path | None, split: str = "validation"):
    """Select a frozen SFT split and refuse to use it if its source has drifted."""
    if manifest_path is None:
        return rows

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    if not source.is_absolute():
        source = Path.cwd() / source
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = manifest["source_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"SFT source drifted: expected {expected}, found {actual}"
        )

    wanted = manifest[split]
    by_sku = {row.sku_id: row for row in rows}
    missing = [sku for sku in wanted if sku not in by_sku]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest SKUs missing from input")
    return [by_sku[sku] for sku in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", help="optional PEFT/LoRA adapter directory")
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", help="optional frozen SFT split manifest")
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="manifest split to predict (default: validation)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-input-length", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=170)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    rows = read_jsonl(args.input)
    rows = filter_rows(rows, args.manifest, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]

    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
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
                max_length=args.max_input_length,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_width = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
            )
            for row, raw in zip(batch, completions, strict=True):
                handle.write(
                    json.dumps(
                        {"sku_id": row.sku_id, "raw": raw},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            print(f"generated {min(start + len(batch), len(rows))}/{len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
