"""The production tagging path, exercised without spending anything.

The provider is a fake, so every failure path — a transient error, a permanent
error, a retry that succeeds — is reachable on CPU. What the fake cannot prove
is that constrained decoding actually yields valid JSON at the vendor; that is
measured on the real run, which is why the tagger records schema validity
rather than assuming it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.tagger import (
    DEFAULT_PRICE_PER_MTOK,
    MAX_ATTEMPTS,
    RunTotals,
    idempotency_key,
    price,
    tag_many,
    tag_one,
    write_results,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent

CLEAN = json.dumps({
    "closure": None, "collar_type": None, "colour_primary": "black",
    "details": ["unknown"], "fit": None, "garment_category": "shoe",
    "garment_length": None, "material": "leather", "neckline": None,
    "occasion": "casual", "pattern": "unknown", "silhouette": None,
    "sleeve_length": None, "sleeve_style": None, "waistline": None,
})
RULE_BREAKING = json.dumps({
    "closure": "unknown", "collar_type": None, "colour_primary": "multicolour",
    "details": ["unknown"], "fit": "unknown", "garment_category": "top",
    "garment_length": "unknown", "material": "cotton", "neckline": "crew",
    "occasion": "casual", "pattern": "solid", "silhouette": "unknown",
    "sleeve_length": "short", "sleeve_style": "unknown", "waistline": None,
})


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


class FakeProvider:
    """Returns queued responses and reports usage, like the real client."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def adapt_schema(self, schema):
        return schema

    def build_body(self, system, user, schema, max_tokens):
        body = {"system": system, "user": user, "schema": schema, "max_tokens": max_tokens}
        self.bodies.append(body)
        return body

    def complete_with_usage(self, body):
        text, error = self.responses.pop(0)
        usage = {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620}
        return text, error, usage


def run_one(pack, responses, **kw):
    provider = FakeProvider(responses)
    return tag_one(
        sku_id="sku-1", prompt="Title: A Shoe", system="tag it",
        provider=provider, pack=pack, schema={}, max_tokens=400,
        price_table=DEFAULT_PRICE_PER_MTOK, sleep_fn=lambda _s: None,
        clock_fn=iter([0.0, 1.5]).__next__, **kw,
    )


# --- the gate is applied, not assumed ----------------------------------------


def test_a_clean_record_passes_the_gate(pack):
    result = run_one(pack, [(CLEAN, None)])
    assert result.schema_valid and result.vocab_valid and result.gate_passed
    assert result.rule_violations == []
    assert result.attempts == 1


def test_a_rule_violation_fails_the_gate_despite_valid_shape(pack):
    """A grammar can guarantee shape and enum membership. It cannot guarantee
    that a solid pattern is not also multicolour, which is why the gate runs
    after generation rather than being replaced by the schema."""
    result = run_one(pack, [(RULE_BREAKING, None)])
    assert result.schema_valid is True
    assert result.vocab_valid is True
    assert result.rule_violations, "cross-field rule must fire"
    assert result.gate_passed is False


def test_prose_is_recorded_as_invalid_not_repaired(pack):
    result = run_one(pack, [("Sure! Here you go:", None)])
    assert result.schema_valid is False and result.gate_passed is False
    assert result.errors


# --- retries, idempotency, cost ----------------------------------------------


def test_a_transient_failure_is_retried_and_can_succeed(pack):
    result = run_one(pack, [(None, "RateLimitError"), (CLEAN, None)])
    assert result.attempts == 2 and result.gate_passed is True
    assert result.error is None


def test_retries_are_bounded_and_the_failure_is_reported(pack):
    result = run_one(pack, [(None, "ServerError")] * MAX_ATTEMPTS)
    assert result.attempts == MAX_ATTEMPTS
    assert result.error == "ServerError"
    assert result.gate_passed is False and result.raw is None


def test_every_attempt_is_charged(pack):
    """A retry that consumed tokens cost money whether or not it answered. A
    cost figure that hides retries understates what the pipeline costs to run."""
    one = run_one(pack, [(CLEAN, None)])
    three = run_one(pack, [(None, "e"), (None, "e"), (CLEAN, None)])
    assert three.prompt_tokens == 3 * one.prompt_tokens
    assert three.cost_usd == pytest.approx(3 * one.cost_usd)


def test_cost_comes_from_reported_tokens_not_an_estimate(pack):
    result = run_one(pack, [(CLEAN, None)])
    expected = (500 * DEFAULT_PRICE_PER_MTOK["prompt"]
                + 120 * DEFAULT_PRICE_PER_MTOK["completion"]) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert result.prompt_tokens == 500 and result.completion_tokens == 120


def test_the_idempotency_key_is_stable_and_scoped(pack):
    """Same product, pack and prompt gives the same key, so a retry is
    identifiable. Change the pack or the prompt and the previous answer is no
    longer the answer to this question, so the key must change."""
    base = idempotency_key("sku-1", "pack_a", "Title: A Shoe")
    assert base == idempotency_key("sku-1", "pack_a", "Title: A Shoe")
    assert base != idempotency_key("sku-2", "pack_a", "Title: A Shoe")
    assert base != idempotency_key("sku-1", "pack_b", "Title: A Shoe")
    assert base != idempotency_key("sku-1", "pack_a", "Title: A Boot")


# --- run totals ---------------------------------------------------------------


def test_totals_report_the_five_numbers(pack):
    provider = FakeProvider([(CLEAN, None), (RULE_BREAKING, None), (None, "boom"),
                             (None, "boom"), (None, "boom")])
    clock = iter([0.0, 0.5, 0.0, 2.0, 0.0, 9.0]).__next__
    results, totals = tag_many(
        [("a", "p"), ("b", "p"), ("c", "p")], system="s", provider=provider,
        pack=pack, sleep_fn=lambda _s: None, clock_fn=clock,
    )
    summary = totals.summary()
    assert summary["products"] == 3 and summary["failed"] == 1
    assert summary["scored"] == 2
    assert summary["gate_pass_rate"] == 0.5      # one clean, one rule-breaking
    assert summary["schema_validity"] == 1.0     # both parsed
    assert summary["cost_per_sku_usd"] > 0
    assert summary["latency_p95_seconds"] > 0
    assert len(results) == 3


def test_percentiles_use_nearest_rank():
    """With tens of samples, interpolation invents precision the sample size
    does not support."""
    totals = RunTotals(products=4, latencies=[1.0, 2.0, 3.0, 100.0])
    summary = totals.summary()
    assert summary["latency_p95_seconds"] == 100.0
    assert summary["latency_max_seconds"] == 100.0


def test_an_empty_run_does_not_divide_by_zero():
    assert RunTotals().summary()["cost_per_sku_usd"] == 0.0


def test_results_round_trip_to_jsonl(tmp_path, pack):
    results, _ = tag_many(
        [("a", "p")], system="s", provider=FakeProvider([(CLEAN, None)]),
        pack=pack, sleep_fn=lambda _s: None, clock_fn=iter([0.0, 1.0]).__next__,
    )
    path = tmp_path / "out" / "tags.jsonl"
    write_results(path, results)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["sku_id"] == "a"
    assert rows[0]["gate_passed"] is True


def test_price_is_a_pure_function_of_reported_usage():
    assert price({"prompt_tokens": 1_000_000, "completion_tokens": 0},
                 {"prompt": 2.0, "completion": 9.0}) == pytest.approx(2.0)
    assert price({}, DEFAULT_PRICE_PER_MTOK) == 0.0
