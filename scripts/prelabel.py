#!/usr/bin/env python3
"""Frontier pre-labeling via the Batch API, with k-sample self-consistency.

    python scripts/prelabel.py submit  --feed data/raw/feed.jsonl --k 5
    python scripts/prelabel.py collect --batch-id msgbatch_... --feed data/raw/feed.jsonl

Two API facts shaped this script
--------------------------------
1. `temperature` is REMOVED on Claude Opus 5 and Sonnet 5 — sending it returns a
   400. The plan says "sample k=5 at temperature"; that is not available on the
   frontier models. Diversity here comes from **prompt perturbation**: each of the
   k samples asks for the attributes in a different order, under a slightly
   different framing. This is arguably a better probe anyway — it surfaces answers
   that are sensitive to how the question was asked, which is exactly the kind of
   answer a human should look at. The same constraint applies to W3 Step 3, so the
   perturbation set is defined once, here, and reused there.

2. Prompt caching is a **prefix** match. The perturbation therefore lives entirely
   in the user turn, after the cache breakpoint; the system block (vocabulary,
   rules, instructions) stays byte-identical across all N x k requests so it is
   written to cache once and read at ~0.1x thereafter. Perturbing the system
   prompt instead would silently cost roughly full price on every request.

Structured outputs are ON here, deliberately — this is production-style labeling,
not RL training. W2 trains unconstrained so the model can emit garbage and be seen
learning not to; making valid JSON free there would delete the reward signal.

Batch API: 50% cheaper, results within hours. Results arrive in ANY order — every
lookup below is keyed by `custom_id`, never by position.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeling.consensus import consensus_labels  # noqa: E402
from labeling.records import (  # noqa: E402
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    from_verifier_record,
    write_jsonl,
)
from verifier import load_pack  # noqa: E402

MODEL = "claude-opus-5"
PROMPT_VERSION = "prelabel-v1"

# Unsupported by structured outputs: numeric/string/array constraints. Pydantic
# emits maxItems for the capped multi-value field; leaving it in risks a 400.
_STRIP = {
    "maxItems", "minItems", "uniqueItems",
    "maxLength", "minLength", "pattern",
    "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum", "multipleOf",
}


def sanitize_schema(node):
    if isinstance(node, dict):
        return {k: sanitize_schema(v) for k, v in node.items() if k not in _STRIP}
    if isinstance(node, list):
        return [sanitize_schema(v) for v in node]
    return node


def build_system(pack) -> str:
    """Frozen across every request in the run. Cached; never perturbed."""
    lines = [
        "You are labeling apparel products for a controlled-vocabulary catalog.",
        "Read the listing text and fill in every field.",
        "",
        "Three possible answers per field, and the distinction matters:",
        "  a value from the list  — the listing supports this answer",
        f'  "{pack.unknown_token}"  — the listing does not say, and you will not guess',
        "  null                   — the field cannot apply to this kind of product",
        "                           (a sleeveless dress has no sleeve length at all)",
        "",
        "Never guess to fill a blank. An honest "
        f'"{pack.unknown_token}" is worth more than a plausible invention: these '
        "labels become a reward function, and an invented value teaches the model "
        "to invent.",
        "",
        "Controlled vocabulary:",
    ]
    for name, spec in pack.specs.items():
        scope = "" if spec.applies_to is None else f"  [only: {', '.join(sorted(spec.applies_to))}]"
        kind = " (choose up to 4)" if spec.kind == "multi" else ""
        lines.append(f"  {name}{kind}: {', '.join(spec.values)}{scope}")
    return "\n".join(lines)


# Perturbations: user-turn only, so the cached system prefix is untouched.
PERTURBATIONS = [
    "Answer every field.",
    "Work through the fields in reverse order, then answer every field.",
    "Before answering, note which fields the text is actually silent about.",
    "Consider whether each field can apply to this product type at all, then answer.",
    "Quote nothing; answer only what the listing text supports.",
]


def build_user(inp: RowInput, variant: int) -> str:
    body = [f"Title: {inp.title}"]
    if inp.brand:
        body.append(f"Brand: {inp.brand}")
    if inp.category:
        body.append(f"Category: {inp.category}")
    if inp.description:
        body.append(f"Description: {inp.description}")
    if inp.raw_tags:
        body.append(f"Tags: {', '.join(inp.raw_tags)}")
    return "\n".join(body) + "\n\n" + PERTURBATIONS[variant % len(PERTURBATIONS)]


def load_feed(path) -> list[tuple[str, RowInput]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        inp = rec["input"] if "input" in rec else rec
        out.append((rec["sku_id"], RowInput.model_validate(inp)))
    return out


def cmd_submit(args) -> int:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pack = load_pack(args.pack)
    feed = load_feed(args.feed)
    system = build_system(pack)
    schema = sanitize_schema(pack.json_schema())

    requests = [
        Request(
            custom_id=f"{sku}::{v}",
            params=MessageCreateParamsNonStreaming(
                model=args.model,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": build_user(inp, v)}],
            ),
        )
        for sku, inp in feed
        for v in range(args.k)
    ]

    print(f"{len(feed)} products x k={args.k} = {len(requests)} requests")
    if len(requests) > 100_000:
        sys.exit("batch limit is 100,000 requests — split the feed or lower k")

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    Path(args.state).write_text(
        json.dumps({"batch_id": batch.id, "k": args.k, "model": args.model}, indent=2)
    )
    print(f"batch {batch.id} — status {batch.processing_status}")
    print(f"state written to {args.state}; collect with:")
    print(f"  python scripts/prelabel.py collect --batch-id {batch.id} --feed {args.feed}")
    return 0


def cmd_collect(args) -> int:
    import anthropic

    pack = load_pack(args.pack)
    feed = dict(load_feed(args.feed))
    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(args.batch_id)
        if batch.processing_status == "ended":
            break
        print(f"  {batch.processing_status}: {batch.request_counts}", flush=True)
        time.sleep(args.poll)

    samples: dict[str, list] = {}
    failures: list[str] = []
    for result in client.messages.batches.results(args.batch_id):
        sku, _, _variant = result.custom_id.partition("::")
        if result.result.type != "succeeded":
            failures.append(f"{result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            failures.append(f"{result.custom_id}: refusal")
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            failures.append(f"{result.custom_id}: no text block")
            continue
        try:
            samples.setdefault(sku, []).append(
                from_verifier_record(json.loads(text), pack.unknown_token)
            )
        except json.JSONDecodeError as exc:
            failures.append(f"{result.custom_id}: bad JSON ({exc})")

    rows = []
    for sku, inp in feed.items():
        got = samples.get(sku)
        if not got:
            failures.append(f"{sku}: no usable samples")
            continue
        labels, agreement = consensus_labels(got)
        rows.append(
            Row(
                sku_id=sku,
                source=args.source,
                split="train",
                input=inp,
                labels=labels,
                provenance=Provenance(
                    labeler=f"{args.model}@{PROMPT_VERSION}",
                    prompt_version=PROMPT_VERSION,
                    self_consistency=SelfConsistency(k=len(got), agreement=agreement),
                ),
            )
        )

    write_jsonl(rows, args.out)
    print(f"\nwrote {len(rows)} labeled rows -> {args.out}")
    if failures:
        print(f"{len(failures)} failures (first 10):")
        for f in failures[:10]:
            print(f"  {f}")
        print("\nThese rows are absent, not silently defaulted. Re-submit them or")
        print("accept the smaller corpus — do not let a gap look like a label.")
    return 0


def cmd_estimate(args) -> int:
    """Rough cost, before spending anything."""
    pack = load_pack(args.pack)
    feed = load_feed(args.feed)
    system_tokens = len(build_system(pack)) // 4
    user_tokens = sum(len(build_user(i, 0)) for _, i in feed) // 4 // max(1, len(feed))
    n = len(feed) * args.k
    # cached system reads at ~0.1x; one write at 1.25x
    in_cost = (system_tokens * 0.1 * n + user_tokens * n) / 1e6 * 5.0
    out_cost = (220 * n) / 1e6 * 25.0
    total = (in_cost + out_cost) * 0.5  # Batch API
    print(f"{len(feed)} products x k={args.k} = {n} requests")
    print(f"  system {system_tokens} tok (cached)   user ~{user_tokens} tok")
    print(f"  estimated Batch API cost: ${total:.2f}  (${(in_cost + out_cost):.2f} unbatched)")
    print("  rough — output length is the biggest unknown until the first run")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--feed", required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--model", default=MODEL)
    s.add_argument("--state", default=str(ROOT / "data" / "raw" / "batch_state.json"))
    s.set_defaults(fn=cmd_submit)

    c = sub.add_parser("collect")
    c.add_argument("--batch-id", required=True)
    c.add_argument("--feed", required=True)
    c.add_argument("--model", default=MODEL)
    c.add_argument("--source", default="sovrn")
    c.add_argument("--poll", type=int, default=60)
    c.add_argument("--out", default=str(ROOT / "data" / "raw" / "labeled.jsonl"))
    c.set_defaults(fn=cmd_collect)

    e = sub.add_parser("estimate")
    e.add_argument("--feed", required=True)
    e.add_argument("--k", type=int, default=5)
    e.set_defaults(fn=cmd_estimate)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
