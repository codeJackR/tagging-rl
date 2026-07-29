"""Row -> chat-format training example. One renderer for SFT and GRPO alike.

Two rules keep this file boring on purpose:

1. **Rendering is imported, not re-implemented.** `render_prompt` / `render_target`
   come from `labeling.lengths` — the same functions the token budgets were
   measured with. A trainer that renders even slightly differently is training
   against budgets measured on text it never sees.

2. **The system prompt is a compressed cousin of the labeling one, and diverging
   is deliberate.** The frontier labeler got the full vocabulary (1,908 tokens);
   a 0.5B student gets the field names and the three-state rule (~150 tokens).
   The vocabulary itself is what SFT is supposed to teach — spelling out every
   value in the prompt would let the model read the answer's shape instead of
   learning it, and would triple the cost of every training step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling.lengths import render_prompt, render_target  # noqa: E402
from labeling.records import Row, read_jsonl  # noqa: E402

# Budgets measured with the Qwen2.5 tokenizer on the real corpus (W2 step 0):
#   prompt max 585 -> 600 clips nothing; completion max 118 -> 170 clips nothing.
# 324/p95 was rejected: it cuts 55 rows, and precisely the information-rich ones.
MAX_PROMPT_TOKENS = 600
MAX_COMPLETION_TOKENS = 170

SYSTEM = """You label clothing products from their listing text.

Answer with a single JSON object, one key per field:
closure, collar_type, colour_primary, details, fit, garment_category,
garment_length, material, neckline, occasion, pattern, silhouette,
sleeve_length, sleeve_style, waistline.

Rules:
- "details" is a list; every other field is a single value.
- Use "unknown" when the text does not say. Never guess.
- Use null when the field cannot apply to this kind of product.
- Output only the JSON object, nothing else."""


def to_messages(row: Row, pack) -> dict:
    """One row -> chat messages + the assistant's gold reply."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_prompt(row)},
        ],
        "completion": [
            {"role": "assistant", "content": render_target(row, pack)}
        ],
        "sku_id": row.sku_id,
    }


def load_sft_dataset(pack, path: str | Path = ROOT / "data" / "train_weak.jsonl"):
    """The 3,600 weak rows as a HF Dataset in prompt-completion chat format.

    TRL's SFTTrainer detects this shape and applies the tokenizer's own chat
    template, masking loss to the completion. Rolling our own template instead
    would train the model on a format its tokenizer special-cases differently.
    """
    from datasets import Dataset

    rows = read_jsonl(path)
    return Dataset.from_list([to_messages(r, pack) for r in rows])


def load_grpo_prompts(
    pack,
    path: str | Path = ROOT / "data" / "train_weak.jsonl",
    *,
    require_pass_rate_band: bool = False,
):
    """Prompts only — GRPO generates its own completions and scores them.

    Gold labels ride along as a JSON column (`gold`), because reward functions
    receive dataset columns as kwargs and the golden-agreement reward needs the
    answer key at scoring time.

    `require_pass_rate_band` filters to rows where the SFT baseline sometimes
    succeeds and sometimes fails (0 < sft_pass_rate < 1) — the only rows that
    produce within-group variance, hence gradient. Off by default because the
    column is null until the baseline exists; turning it on with nulls everywhere
    would silently yield an empty dataset, so it raises instead.
    """
    from datasets import Dataset

    rows = read_jsonl(path)
    if require_pass_rate_band:
        scored = [r for r in rows if r.difficulty.sft_pass_rate is not None]
        if not scored:
            raise RuntimeError(
                "no row has sft_pass_rate set — run the baseline scoring pass "
                "(training/score_difficulty.py) before filtering on it"
            )
        rows = [r for r in scored if 0.0 < r.difficulty.sft_pass_rate < 1.0]

    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": render_prompt(r)},
                ],
                "gold": json.dumps(r.to_verifier_record(pack), sort_keys=True),
                "sku_id": r.sku_id,
            }
            for r in rows
        ]
    )
