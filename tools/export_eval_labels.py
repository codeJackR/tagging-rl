#!/usr/bin/env python3
"""Export the frozen eval set's labels without the merchant text.

`data/eval_300/eval.jsonl` mixes two things with very different publication
status. The **labels** are this project's own work product: consensus
annotations, provenance, the human review trail. The **input** is verbatim
marketing copy belonging to 50 retailers, collected from public Shopify
endpoints that the Run 2 terms audit found mostly prohibit that access.

Publishing the first and withholding the second needs them separated, which is
all this does. The output carries every field the evaluation harness and the
blog post's claim tests read, and none of the prose.

What deliberately survives the split:

- `sku_id`, which embeds the store domain. It is the join key for every
  prediction file in `runs/`, so removing it would sever the labels from the
  evidence they are meant to verify. A domain is not the copy.
- `provenance.frontier_labels`, the pre-review model answer, because the
  reliability numbers are computed by comparing it against the final label.

What does not survive: `title`, `description`, `raw_tags`, `image_url`.

This is a derivative, not a replacement. It is not hash-pinned by anything and
must never be substituted for `eval.jsonl` in a provenance check, because the
frozen manifest checksums the original file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "eval_300" / "eval.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval_300" / "eval-labels.jsonl"

# Prose fields. Everything here is the retailer's, not ours.
WITHHELD = ("title", "description", "raw_tags", "image_url")


def strip_row(row: dict) -> dict:
    """Blank the merchant prose, keep everything the harness scores from.

    `input` is emptied rather than removed. `labeling.records.Row` requires it
    and forbids extra keys, so a row without it will not load, and the point of
    this export is that the published labels stay usable by the harness a reader
    already has. Blanked prose scores identically; only prompt rendering, which
    needs the text, becomes impossible.
    """
    out = {key: value for key, value in row.items() if key != "input"}
    source = row.get("input") or {}
    # `brand` and `category` are one-word retailer taxonomy terms and several
    # cross-field rules key off category. They are kept; the sentences are not.
    out["input"] = {
        "title": "",
        "description": "",
        "raw_tags": [],
        "brand": source.get("brand"),
        "category": source.get("category"),
    }
    return out


def export(source: Path, output: Path) -> dict:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body = "\n".join(json.dumps(strip_row(row), sort_keys=True) for row in rows) + "\n"
    output.write_text(body, encoding="utf-8")

    # Verify against the emitted bytes rather than the intent: any string long
    # enough to be a sentence means prose survived the split.
    longest = 0
    for line in body.splitlines():
        for value in (json.loads(line).get("input") or {}).values():
            if isinstance(value, str):
                longest = max(longest, len(value))
            elif isinstance(value, list):
                longest = max([longest] + [len(v) for v in value if isinstance(v, str)])
    if longest > 40:
        raise SystemExit(f"prose survived the export: a kept field is {longest} chars")

    return {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(rows),
        "withheld_fields": list(WITHHELD),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
