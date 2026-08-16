#!/usr/bin/env python3
"""The production tagging path: constrained decoding, then the gate.

This is deliberately the mirror image of the RL path. W2 trained and evaluated
with decoding **unconstrained**, because whether the model learns to emit valid
JSON was the behaviour under study, and a grammar would have made that reward
free and unmeasurable. In production the reasoning inverts: nobody benefits from
a malformed record reaching a catalog, so the schema is enforced during decoding
and validity stops being a question.

What remains a question, and is what this module measures:

- **vocabulary and rule compliance.** A grammar can guarantee the shape of a
  record and the membership of each enum, but it cannot guarantee that a `solid`
  pattern is not also `multicolour`. Cross-field rules are checked after
  generation, by the same verifier the reward function uses.
- **cost per SKU**, from the token counts the API reports.
- **latency**, per product, so a p95 can be quoted rather than estimated.

Deliberately absent, per the W3 plan's stop condition: caching, concurrency
beyond simple sequential batching, persistence, and any queue. The output is a
JSONL file.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from verifier import Pack, verify

ROOT = Path(__file__).resolve().parent.parent
VERSION = "production-tagger-v1"

# Price per million tokens. This is an *input*, not a measurement: token counts
# come from the API, dollars come from this table, and the two are kept separate
# so a reader can substitute their own rates without re-running anything.
DEFAULT_PRICE_PER_MTOK = {"prompt": 1.25, "completion": 10.0}

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 4.0)


class SupportsCompletion(Protocol):
    """The slice of a provider this module needs, so tests can supply a fake."""

    def adapt_schema(self, schema: dict) -> dict: ...

    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict: ...

    def complete_with_usage(
        self, body: dict
    ) -> tuple[str | None, str | None, dict[str, int]]: ...


def idempotency_key(sku_id: str, pack_name: str, prompt: str) -> str:
    """A stable identity for one tagging request.

    Keyed on the product, the vocabulary judging it and the exact prompt text.
    A retry of the same request reuses the key, so a caller can tell a retry
    from a genuinely new request; changing the pack or the prompt changes the
    key, because the previous answer is no longer the answer to this question.
    """
    material = f"{pack_name}\x00{sku_id}\x00{prompt}".encode()
    return hashlib.sha256(material).hexdigest()[:32]


@dataclass
class TagResult:
    sku_id: str
    idempotency_key: str
    raw: str | None
    parsed: dict[str, Any] | None
    # Constrained decoding is expected to make this always true. It is recorded
    # rather than assumed, because "guaranteed by construction" is a claim about
    # a vendor's implementation and this is the number that would falsify it.
    schema_valid: bool
    vocab_valid: bool
    rule_violations: list[str]
    gate_passed: bool
    errors: list[str]
    attempts: int
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str | None = None


@dataclass
class RunTotals:
    products: int = 0
    gate_passed: int = 0
    schema_valid: int = 0
    vocab_valid: int = 0
    rule_violations: int = 0
    failed: int = 0
    retried: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.latencies)
        n = len(ordered)

        def pct(p: float) -> float:
            if not ordered:
                return 0.0
            # Nearest-rank: with tens of samples, interpolation invents
            # precision the sample size does not support.
            index = max(0, min(n - 1, int(round(p * n)) - 1))
            return round(ordered[index], 3)

        scored = self.products - self.failed
        return {
            "products": self.products,
            "scored": scored,
            "failed": self.failed,
            "retried": self.retried,
            "schema_validity": round(self.schema_valid / scored, 4) if scored else 0.0,
            "vocab_validity": round(self.vocab_valid / scored, 4) if scored else 0.0,
            "gate_pass_rate": round(self.gate_passed / scored, 4) if scored else 0.0,
            "rule_violations": self.rule_violations,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_per_sku_usd": (
                round(self.cost_usd / self.products, 6) if self.products else 0.0
            ),
            "latency_p50_seconds": pct(0.50),
            "latency_p95_seconds": pct(0.95),
            "latency_max_seconds": round(ordered[-1], 3) if ordered else 0.0,
        }


def price(usage: dict[str, int], table: dict[str, float]) -> float:
    return (
        usage.get("prompt_tokens", 0) * table["prompt"]
        + usage.get("completion_tokens", 0) * table["completion"]
    ) / 1_000_000


def tag_one(
    *,
    sku_id: str,
    prompt: str,
    system: str,
    provider: SupportsCompletion,
    pack: Pack,
    schema: dict,
    max_tokens: int,
    price_table: dict[str, float],
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> TagResult:
    """Tag one product under a schema, then put the answer through the gate."""
    key = idempotency_key(sku_id, pack.name, prompt)
    body = provider.build_body(system, prompt, schema, max_tokens)

    started = clock_fn()
    prompt_tokens = completion_tokens = 0
    cost = 0.0
    text: str | None = None
    error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        text, error, usage = provider.complete_with_usage(body)
        # Charge every attempt. A retry that consumed tokens cost money whether
        # or not it produced an answer, and a cost figure that hides retries
        # understates what the pipeline actually costs to operate.
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        cost += price(usage, price_table)
        if text is not None:
            break
        if attempt < MAX_ATTEMPTS:
            sleep_fn(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
    latency = clock_fn() - started

    if text is None:
        return TagResult(
            sku_id=sku_id, idempotency_key=key, raw=None, parsed=None,
            schema_valid=False, vocab_valid=False, rule_violations=[],
            gate_passed=False, errors=[], attempts=attempt,
            latency_seconds=latency, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cost_usd=cost, error=error,
        )

    # The gate. Same verifier as the reward function, no repair, no leniency.
    result = verify(text, pack)
    return TagResult(
        sku_id=sku_id, idempotency_key=key, raw=text, parsed=result.parsed,
        schema_valid=result.schema_valid, vocab_valid=result.vocab_valid,
        rule_violations=list(result.rule_violations), gate_passed=result.ok,
        errors=list(result.errors), attempts=attempt, latency_seconds=latency,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost,
    )


def tag_many(
    items: Iterable[tuple[str, str]],
    *,
    system: str,
    provider: SupportsCompletion,
    pack: Pack,
    max_tokens: int = 400,
    price_table: dict[str, float] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> tuple[list[TagResult], RunTotals]:
    """Tag a sequence of (sku_id, prompt) pairs, accumulating run totals."""
    table = price_table or DEFAULT_PRICE_PER_MTOK
    schema = provider.adapt_schema(pack.json_schema())
    results: list[TagResult] = []
    totals = RunTotals()

    for sku_id, prompt in items:
        result = tag_one(
            sku_id=sku_id, prompt=prompt, system=system, provider=provider,
            pack=pack, schema=schema, max_tokens=max_tokens,
            price_table=table, sleep_fn=sleep_fn, clock_fn=clock_fn,
        )
        results.append(result)
        totals.products += 1
        totals.prompt_tokens += result.prompt_tokens
        totals.completion_tokens += result.completion_tokens
        totals.cost_usd += result.cost_usd
        totals.latencies.append(result.latency_seconds)
        if result.attempts > 1:
            totals.retried += 1
        if result.error is not None:
            totals.failed += 1
            continue
        totals.schema_valid += int(result.schema_valid)
        totals.vocab_valid += int(result.vocab_valid)
        totals.gate_passed += int(result.gate_passed)
        totals.rule_violations += len(result.rule_violations)

    return results, totals


def write_results(path: Path, results: list[TagResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
