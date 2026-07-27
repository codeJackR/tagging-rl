#!/usr/bin/env python3
"""Frontier pre-labeling via a Batch API, with k-sample self-consistency.

    python scripts/prelabel.py estimate --feed data/raw/feed.jsonl --k 5
    python scripts/prelabel.py submit   --feed data/raw/feed.jsonl --k 5
    python scripts/prelabel.py collect  --batch-id <id> --feed data/raw/feed.jsonl

Provider defaults to OpenAI `gpt-5.6-luna`; `--provider anthropic` runs the same
prompts through Claude. Backends live in `labeling/providers.py` — the system
prompt, perturbation set, consensus and provenance below are shared, so the two
runs are comparable and the Step 3 reliability table can be diffed across them.

Diversity without a temperature dial
------------------------------------
The plan says "sample k=5 at temperature". Modern frontier models increasingly
reject sampling parameters outright (Claude Opus 5 and Sonnet 5 return a 400), so
the k samples come from **prompt perturbation** instead: each asks for the same
record under a different framing. This is arguably the better probe — an answer
that changes with the phrasing is exactly the answer a human should check — and it
works identically on every provider.

Perturbation lives in the **user turn only**. Both vendors cache on a byte-identical
prefix (Anthropic via an explicit breakpoint, OpenAI automatically), so varying the
system block would silently forfeit the discount on every one of the N x k requests.

Structured outputs are ON here, deliberately. This is production-style labeling,
not RL training — W2 trains unconstrained so the model can emit garbage and be seen
learning not to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # gitignored; see .env.example

from labeling.consensus import consensus_labels  # noqa: E402
from labeling.providers import (  # noqa: E402
    DEFAULT_MODELS,
    PRICING,
    get_provider,
)
from labeling.records import (  # noqa: E402
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    from_verifier_record,
    write_jsonl,
)
from verifier import load_pack  # noqa: E402

PROMPT_VERSION = "prelabel-v1"
MAX_TOKENS = 2048
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

def require_key(provider: str) -> None:
    var = KEY_ENV[provider]
    if not os.environ.get(var):
        sys.exit(
            f"{var} is not set.\n"
            f"  Put it in {ROOT / '.env'} (gitignored) as {var}=...\n"
            f"  See .env.example."
        )


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
        scope = (
            "" if spec.applies_to is None
            else f"  [only for: {', '.join(sorted(spec.applies_to))}]"
        )
        kind = " (choose up to 4)" if spec.kind == "multi" else ""
        lines.append(f"  {name}{kind}: {', '.join(spec.values)}{scope}")
    return "\n".join(lines)


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


def build_items(feed, pack, provider, k: int) -> list[tuple[str, dict]]:
    system = build_system(pack)
    schema = provider.adapt_schema(pack.json_schema())
    return [
        (f"{sku}::{v}", provider.build_body(system, build_user(inp, v), schema, MAX_TOKENS))
        for sku, inp in feed
        for v in range(k)
    ]


def cmd_submit(args) -> int:
    require_key(args.provider)
    pack = load_pack(args.pack)
    provider = get_provider(args.provider, args.model)
    feed = load_feed(args.feed)
    items = build_items(feed, pack, provider, args.k)

    print(f"{len(feed)} products x k={args.k} = {len(items)} requests")
    print(f"  provider {provider.name}  model {provider.model}")
    if len(items) > 50_000:
        print("  NOTE: large batch — check your tier's queue limit before submitting.")

    batch_id = provider.submit(items)
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    Path(args.state).write_text(
        json.dumps(
            {"batch_id": batch_id, "k": args.k, "provider": provider.name,
             "model": provider.model, "feed": str(args.feed)},
            indent=2,
        )
    )
    print(f"\nbatch {batch_id}\nstate -> {args.state}\ncollect with:")
    print(
        f"  python scripts/prelabel.py collect --batch-id {batch_id} "
        f"--feed {args.feed} --provider {provider.name}"
    )
    return 0


def cmd_status(args) -> int:
    """Non-blocking progress check. `collect` polls until done; this returns now."""
    import datetime as dt

    require_key(args.provider)
    provider = get_provider(args.provider, args.model)
    state = Path(args.state)
    batch_id = args.batch_id
    if not batch_id and state.exists():
        batch_id = json.loads(state.read_text())["batch_id"]
    if not batch_id:
        sys.exit(f"no --batch-id and no state file at {state}")

    info = provider.status(batch_id)
    print(f"batch    {batch_id}")
    print(f"status   {info['status']}")
    done, total, failed = info["completed"], info["total"], info["failed"]
    if total:
        pct = done / total
        bar = "#" * int(pct * 30)
        print(f"progress {done}/{total} ({pct:.1%}) {failed} failed")
        print(f"         [{bar:<30}]")
        elapsed = info.get("elapsed_s") or 0
        if done and elapsed and done < total:
            eta = elapsed / done * (total - done)
            print(f"         ~{dt.timedelta(seconds=int(eta))} remaining at current rate")
    if info["ready"]:
        print("\nREADY. Collect with:")
        print(
            f"  uv run python scripts/prelabel.py collect --batch-id {batch_id} "
            f"--feed {args.feed or 'data/raw/feed.jsonl'} --provider {provider.name}"
        )
    return 0


def cmd_collect(args) -> int:
    require_key(args.provider)
    pack = load_pack(args.pack)
    provider = get_provider(args.provider, args.model)
    feed = dict(load_feed(args.feed))

    provider.wait(args.batch_id, args.poll)

    samples: dict[str, list] = {}
    failures: list[str] = []
    for result in provider.results(args.batch_id):
        sku, _, _variant = result.custom_id.partition("::")
        if result.error or not result.text:
            failures.append(f"{result.custom_id}: {result.error or 'no text'}")
            continue
        # Every per-row failure is caught. A run of 20,000 requests died on ONE
        # unexpected shape once; the cost of that is not one lost row, it is the
        # other 19,999 and the wait for the batch. Anything unparseable is recorded
        # and skipped, never guessed at.
        try:
            samples.setdefault(sku, []).append(
                from_verifier_record(json.loads(result.text), pack)
            )
        except Exception as exc:  # noqa: BLE001 — model output is untrusted input
            failures.append(f"{result.custom_id}: {type(exc).__name__}: {str(exc)[:120]}")

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
                    labeler=f"{provider.model}@{PROMPT_VERSION}",
                    prompt_version=PROMPT_VERSION,
                    self_consistency=SelfConsistency(k=len(got), agreement=agreement),
                ),
            )
        )

    write_jsonl(rows, args.out)
    print(f"\nwrote {len(rows)} labeled rows -> {args.out}")
    print(f"  labeler recorded as {provider.model}@{PROMPT_VERSION}")
    if failures:
        print(f"\n{len(failures)} failures (first 10):")
        for f in failures[:10]:
            print(f"  {f}")
        print("\nThese rows are absent, not silently defaulted. Re-submit them or")
        print("accept the smaller corpus — do not let a gap look like a label.")
    return 0


def cmd_estimate(args) -> int:
    """Cost, calibrated against one real request rather than guessed.

    Offline token estimation for this model does not work. Measured on the actual
    system prompt: chars/4 said 746 tokens, tiktoken's o200k_base said 805, the API
    said 1908. Two heuristics, two confidently wrong answers in opposite directions
    — and the cheap one would also have produced a false "caching will not engage"
    warning, since 746 sits below the 1024-token threshold and 1908 sits well above.

    So `estimate` sends exactly one request, reads `usage` back, and multiplies.
    It costs a fraction of a cent and is exact. `--offline` skips it and says
    plainly that the resulting figure is unreliable.
    """
    pack = load_pack(args.pack)
    model = args.model or DEFAULT_MODELS[args.provider]
    feed = load_feed(args.feed)
    if not feed:
        sys.exit("empty feed")
    n = len(feed) * args.k

    price = PRICING.get(model)
    if price is None:
        print(f"no pricing on file for {model!r} — request count only: {n}")
        return 0
    in_rate, cached_rate, out_rate = price

    print(f"{len(feed)} products x k={args.k} = {n} requests")
    print(f"  provider {args.provider}  model {model}")

    if args.offline:
        system = build_system(pack)
        in_tok = len(system) // 4 + sum(len(build_user(i, 0)) for _, i in feed) // 4 // len(feed)
        out_tok, cached_tok, measured = 400, 0, False
    else:
        require_key(args.provider)
        in_tok, out_tok, cached_tok = probe_usage(pack, args, model)
        measured = True

    uncached = max(0, in_tok - cached_tok)
    full = (uncached * in_rate + cached_tok * cached_rate + out_tok * out_rate) * n / 1e6

    print(f"  per request: {in_tok} in ({cached_tok} cacheable) · {out_tok} out")
    print(f"\n  estimated Batch API cost: ${full * 0.5:.2f}   (${full:.2f} unbatched)")
    if measured:
        if cached_tok:
            saved = cached_tok * (in_rate - cached_rate) * n / 1e6 * 0.5
            print(f"  caching engaged on {cached_tok} tokens — saves ~${saved:.2f}")
        else:
            print(
                "  NO CACHING on this request. If the shared prefix is below the\n"
                "  provider's threshold, a LONGER system prompt (few-shot examples,\n"
                "  fuller vocabulary notes) would cost LESS and likely label better."
            )
    else:
        print(
            "\n  OFFLINE ESTIMATE — treat as unreliable. Measured against this pack,\n"
            "  chars/4 was 2.6x low and tiktoken 2.4x low. Re-run without --offline."
        )
    return 0


def probe_usage(pack, args, model) -> tuple[int, int, int]:
    """One live request; return (input, output, cacheable-input) token counts."""
    provider = get_provider(args.provider, args.model)
    feed = load_feed(args.feed)
    schema = provider.adapt_schema(pack.json_schema())
    body = provider.build_body(
        build_system(pack), build_user(feed[0][1], 0), schema, MAX_TOKENS
    )
    if args.provider == "openai":
        import openai

        r = openai.OpenAI().chat.completions.create(**body)
        details = getattr(r.usage, "prompt_tokens_details", None)
        cacheable = max(
            getattr(details, "cached_tokens", 0) or 0,
            getattr(details, "cache_write_tokens", 0) or 0,
        )
        return r.usage.prompt_tokens, r.usage.completion_tokens, cacheable

    import anthropic

    body.pop("output_config", None)  # count against a plain call
    r = anthropic.Anthropic().messages.create(**body)
    u = r.usage
    cacheable = (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
    return u.input_tokens + cacheable, u.output_tokens, cacheable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    common.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    common.add_argument("--model", default=None, help="override the provider default")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", parents=[common])
    e.add_argument("--feed", required=True)
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--offline", action="store_true",
                   help="skip the calibration request; result is unreliable")
    e.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("submit", parents=[common])
    s.add_argument("--feed", required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--state", default=str(ROOT / "data" / "raw" / "batch_state.json"))
    s.set_defaults(fn=cmd_submit)

    st = sub.add_parser("status", parents=[common],
                        help="progress without blocking (collect polls until done)")
    st.add_argument("--batch-id", default=None, help="defaults to the saved state file")
    st.add_argument("--feed", default=None)
    st.add_argument("--state", default=str(ROOT / "data" / "raw" / "batch_state.json"))
    st.set_defaults(fn=cmd_status)

    c = sub.add_parser("collect", parents=[common])
    c.add_argument("--batch-id", required=True)
    c.add_argument("--feed", required=True)
    c.add_argument("--source", default="shopify")
    c.add_argument("--poll", type=int, default=60)
    c.add_argument("--out", default=str(ROOT / "data" / "raw" / "labeled.jsonl"))
    c.set_defaults(fn=cmd_collect)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
