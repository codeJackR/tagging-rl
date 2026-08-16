#!/usr/bin/env python3
"""Run the production path over real products and report five numbers.

The five the W3 plan asks for: accuracy, escalation rate, cost per SKU,
throughput and p95 latency. Each carries the caveat that belongs to it, at the
point it is stated rather than once in a footer, because a number quoted without
its caveat travels further than the caveat does.

The run is deliberately gated on a projected cost before it spends anything. A
pipeline that discovers its own bill afterwards is not a production pipeline.

    uv run python -m production.run_report estimate --products 500
    uv run python -m production.run_report run --products 500 --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling.lengths import render_prompt  # noqa: E402
from labeling.records import read_jsonl  # noqa: E402
from production.escalation import (  # noqa: E402
    DEFAULT_K,
    escalate,
    threshold_curve,
    write_queue,
)
from production.tagger import (  # noqa: E402
    DEFAULT_PRICE_PER_MTOK,
    TagResult,
    tag_many,
    write_results,
)
from verifier import load_pack  # noqa: E402

VERSION = "production-run-report-v1"
DEFAULT_SOURCE = "data/train_weak.jsonl"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_OUT = "runs/production-demo"

# Refuse to spend more than this without an explicit acknowledgement. The W3
# plan sets the figure; the point is that the number is checked before the run,
# not discovered from an invoice afterwards.
COST_CEILING_USD = 10.0

# Tokens per request, measured on a two-product smoke against the real API.
#
# The first version of this estimate reused W2's training budget of 600 prompt
# tokens and was low by a factor of two. W2 measured the prompt the *model*
# sees; the production request also carries the JSON schema, because that is how
# constrained decoding is specified, and the schema enumerates all 156
# vocabulary values. Constrained decoding is not free: the grammar is billed as
# prompt tokens on every request, including every self-consistency repeat.
#
# Measured: prompt 1,214, completion 220 across the smoke.
ESTIMATED_PROMPT_TOKENS = 1250
ESTIMATED_COMPLETION_TOKENS = 230

SYSTEM = """You label clothing products from their listing text.

Answer with a single JSON object, one key per field. Use "unknown" when the
text does not say; never guess. Use null when the field cannot apply to this
kind of product. Output only the JSON object."""


def select_products(source: Path, limit: int) -> list[tuple[str, str]]:
    """Take the first `limit` products in file order.

    Deterministic and unshuffled on purpose: a sampled demo whose selection
    cannot be reproduced is an anecdote. The file order was itself fixed by a
    seeded process in W1.
    """
    rows = read_jsonl(source)
    return [(row.sku_id, render_prompt(row)) for row in rows[:limit]]


def estimate_cost(products: int, k: int, price_table: dict[str, float]) -> dict[str, Any]:
    requests = products * k
    prompt_tokens = requests * ESTIMATED_PROMPT_TOKENS
    completion_tokens = requests * ESTIMATED_COMPLETION_TOKENS
    projected = (
        prompt_tokens * price_table["prompt"]
        + completion_tokens * price_table["completion"]
    ) / 1_000_000
    return {
        "products": products,
        "k": k,
        "requests": requests,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "projected_cost_usd": round(projected, 4),
        "ceiling_usd": COST_CEILING_USD,
        "within_ceiling": projected <= COST_CEILING_USD,
        "note": (
            "Projected from measured token budgets, not from a rate card alone. "
            "The run reports actual cost from API-reported usage; expect these to "
            "differ and trust the measured one."
        ),
    }


def load_pass(path: Path, *, expected: int) -> list[TagResult] | None:
    """Reload a previously completed pass, or None if it is absent or partial.

    Completeness is checked by row count. A pass truncated mid-write would
    otherwise be resumed as if it were whole, and its missing products would
    silently reduce the sample the report is computed from.
    """
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        print(f"  {path.name}: {len(rows)}/{expected} rows, rerunning", flush=True)
        return None
    return [TagResult(**row) for row in rows]


def build_provider(model: str | None):
    """Construct the provider, loading .env the way the labelling scripts do.

    The two-product smoke found this: the key lives in .env, which the shell
    does not export, so the provider raised on construction. Cheap to discover
    for half a cent; expensive to discover part-way through a paid run.
    """
    from dotenv import load_dotenv
    from labeling.providers import OpenAIProvider

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set and was not found in .env")
    return OpenAIProvider(model=model)


def run(
    *,
    products: int,
    k: int,
    threshold: float,
    source: Path,
    pack_path: Path,
    out_dir: Path,
    model: str | None,
    price_table: dict[str, float],
) -> dict[str, Any]:
    pack = load_pack(pack_path)
    items = select_products(source, products)
    if not items:
        raise SystemExit("no products selected")

    provider = build_provider(model)
    started = time.time()

    # k independent passes over the same products. Sampling per product would
    # interleave differently, but the totals are identical and pass-major order
    # makes a partial run interpretable: it is complete for some passes rather
    # than partial for all products.
    passes: list[list] = []
    resumed_passes: list[int] = []
    for index in range(k):
        path = out_dir / f"pass-{index + 1}.jsonl"
        # Resume a complete pass rather than paying for it twice. An
        # interrupted run had already spent real money on two passes; a
        # pipeline that cannot resume turns every interruption into a refund
        # request nobody will honour. A pass counts only if it is complete,
        # so a half-written file is redone rather than silently trusted.
        existing = load_pass(path, expected=len(items))
        if existing is not None:
            passes.append(existing)
            resumed_passes.append(index + 1)
            print(f"  pass {index + 1}/{k}: resumed from disk", flush=True)
            continue

        results, totals = tag_many(
            items, system=SYSTEM, provider=provider, pack=pack,
            price_table=price_table,
        )
        passes.append(results)
        write_results(path, results)
        print(
            f"  pass {index + 1}/{k}: "
            f"gate {totals.summary()['gate_pass_rate']:.3f} "
            f"cost ${totals.summary()['cost_usd']:.4f}",
            flush=True,
        )
    wall_seconds = time.time() - started

    # Fold the passes back together per product for the confidence measurement.
    by_sku: dict[str, list[dict]] = {sku: [] for sku, _ in items}
    for results in passes:
        for result in results:
            if result.parsed is not None:
                by_sku[result.sku_id].append(result.parsed)
    per_product = [(sku, samples) for sku, samples in by_sku.items()]

    cells, _confidences, escalation = escalate(
        per_product, pack=pack, threshold=threshold, k=k
    )
    write_queue(out_dir / "review-queue.csv", cells)
    curve = threshold_curve(per_product, pack=pack, k=k)

    # Quality is measured on the first pass only. Averaging gate rates across
    # passes would describe a pipeline nobody runs; production emits one record
    # per product, and the first pass is that record.
    first = passes[0]
    scored = [r for r in first if r.error is None]
    gate_passed = sum(1 for r in scored if r.gate_passed)
    schema_valid = sum(1 for r in scored if r.schema_valid)
    vocab_valid = sum(1 for r in scored if r.vocab_valid)
    violations = sum(len(r.rule_violations) for r in scored)

    all_results = [r for results in passes for r in results]
    total_cost = sum(r.cost_usd for r in all_results)
    latencies = sorted(r.latency_seconds for r in all_results)
    request_seconds = sum(r.latency_seconds for r in all_results)
    p95 = latencies[max(0, min(len(latencies) - 1, int(round(0.95 * len(latencies))) - 1))]

    report = {
        "version": VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "products": len(items),
            "k": k,
            "threshold": threshold,
            "pack": pack.name,
            "model": model or "provider default",
            "price_per_mtok": price_table,
        },
        "quality_first_pass": {
            "scored": len(scored),
            "failed": len(first) - len(scored),
            "schema_validity": round(schema_valid / len(scored), 4) if scored else 0.0,
            "vocab_validity": round(vocab_valid / len(scored), 4) if scored else 0.0,
            "gate_pass_rate": round(gate_passed / len(scored), 4) if scored else 0.0,
            "rule_violations": violations,
            "caveat": (
                "Gate pass rate is not accuracy against ground truth. It is the "
                "share of records that are structurally valid, in-vocabulary and "
                "rule-consistent. A confidently wrong record passes the gate."
            ),
        },
        "escalation": escalation.summary(),
        "escalation_threshold_curve": curve,
        "cost": {
            "total_usd": round(total_cost, 4),
            "cost_per_sku_usd": round(total_cost / len(items), 6),
            "cost_per_sku_single_pass_usd": round(total_cost / len(items) / k, 6),
            "requests": len(all_results),
            "caveat": (
                "From API-reported token usage, including retries. Cost per SKU "
                f"covers all {k} self-consistency passes; the single-pass figure "
                "is what tagging alone would cost without confidence measurement."
            ),
        },
        "performance": {
            "wall_seconds": round(wall_seconds, 1),
            # Throughput comes from summed request latencies, not from wall
            # clock. A resumed pass costs no time to reload, so wall clock on a
            # resumed run measures file reads rather than tagging and reports a
            # throughput several orders of magnitude too high. The latencies
            # were recorded when the requests actually ran and stay true across
            # a resume; wall_seconds is kept alongside so the two are
            # comparable on a run that did no resuming.
            "products_per_minute": round(len(items) / (request_seconds / 60), 2)
            if request_seconds
            else 0.0,
            "requests_per_minute": round(len(all_results) / (request_seconds / 60), 2)
            if request_seconds
            else 0.0,
            "request_seconds": round(request_seconds, 1),
            "resumed_passes": resumed_passes,
            "latency_p50_seconds": round(latencies[len(latencies) // 2], 3),
            "latency_p95_seconds": round(p95, 3),
            "latency_max_seconds": round(latencies[-1], 3),
            "caveat": (
                "Sequential, single process, no concurrency. Throughput is a "
                "floor rather than a capability claim, and is derived from "
                "summed request latencies rather than wall clock so that it "
                "survives a resume."
            ),
        },
        "limitations": [
            "Products come from the corpus already collected. The planned "
            "Shopify sources were unavailable: 16 of 20 candidate stores "
            "prohibit the access, none approve it.",
            "These products were seen during SFT training, so any comparison "
            "with their stored labels measures agreement with weak labels, not "
            "production accuracy.",
            "Agreement is not correctness: five samples can agree on the same "
            "wrong answer, so the escalation rate bounds review labour rather "
            "than error.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--products", type=int, default=500)
        p.add_argument("--k", type=int, default=DEFAULT_K)
        p.add_argument("--threshold", type=float, default=1.0)
        p.add_argument("--source", default=DEFAULT_SOURCE)
        p.add_argument("--pack", default=DEFAULT_PACK)
        p.add_argument("--out", default=DEFAULT_OUT)
        p.add_argument("--model", default=os.environ.get("PRODUCTION_MODEL"))
        if name == "run":
            p.add_argument(
                "--confirm-cost",
                action="store_true",
                help="acknowledge the projected cost; required to spend anything",
            )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    estimate = estimate_cost(args.products, args.k, DEFAULT_PRICE_PER_MTOK)

    if args.command == "estimate":
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0

    if not estimate["within_ceiling"]:
        print(json.dumps({"refused": "projected cost exceeds ceiling", **estimate}, indent=2))
        return 2
    if not args.confirm_cost:
        print(json.dumps({"refused": "pass --confirm-cost to spend", **estimate}, indent=2))
        return 2

    out_dir = ROOT / args.out
    if (out_dir / "report.json").exists():
        raise SystemExit(f"refusing to overwrite a finished report: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run(
        products=args.products, k=args.k, threshold=args.threshold,
        source=ROOT / args.source, pack_path=ROOT / args.pack, out_dir=out_dir,
        model=args.model, price_table=DEFAULT_PRICE_PER_MTOK,
    )
    print(json.dumps({
        "gate_pass_rate": report["quality_first_pass"]["gate_pass_rate"],
        "escalation_rate": report["escalation"]["escalation_rate"],
        "cost_per_sku_usd": report["cost"]["cost_per_sku_usd"],
        "products_per_minute": report["performance"]["products_per_minute"],
        "latency_p95_seconds": report["performance"]["latency_p95_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
