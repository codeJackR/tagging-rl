"""The run report: five numbers, each with its caveat attached.

The cost gate is tested hardest because it is the only thing standing between a
typo in --products and a surprise invoice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.run_report import (
    COST_CEILING_USD,
    estimate_cost,
    main,
    select_products,
)
from production.tagger import DEFAULT_PRICE_PER_MTOK

ROOT = Path(__file__).resolve().parent.parent


# --- the cost gate ------------------------------------------------------------


def test_estimate_scales_with_products_and_k():
    one = estimate_cost(100, 1, DEFAULT_PRICE_PER_MTOK)
    five = estimate_cost(100, 5, DEFAULT_PRICE_PER_MTOK)
    assert five["requests"] == 5 * one["requests"]
    # Compare on the token counts, which are exact. The dollar figures are
    # rounded for display and 5 x round(x) need not equal round(5x).
    assert five["estimated_prompt_tokens"] == 5 * one["estimated_prompt_tokens"]
    assert five["projected_cost_usd"] == pytest.approx(
        5 * one["projected_cost_usd"], rel=1e-3
    )


def test_the_planned_run_is_inside_the_ceiling():
    estimate = estimate_cost(500, 5, DEFAULT_PRICE_PER_MTOK)
    assert estimate["within_ceiling"] is True
    assert estimate["projected_cost_usd"] < COST_CEILING_USD


def test_a_run_that_would_exceed_the_ceiling_is_refused(capsys):
    """The only thing between a typo in --products and a surprise invoice."""
    code = main(["run", "--products", "500000", "--confirm-cost"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["refused"].startswith("projected cost")


def test_spending_requires_an_explicit_acknowledgement(capsys):
    code = main(["run", "--products", "10"])
    assert code == 2
    assert "confirm-cost" in json.loads(capsys.readouterr().out)["refused"]


def test_estimate_never_spends(capsys):
    assert main(["estimate", "--products", "500"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["requests"] == 2500
    assert "projected_cost_usd" in body


def test_the_estimate_says_to_trust_the_measured_cost_instead():
    """A projection is an input; the run reports what was actually spent."""
    assert "trust the measured one" in estimate_cost(10, 5, DEFAULT_PRICE_PER_MTOK)["note"]


# --- selection ----------------------------------------------------------------


def test_selection_is_deterministic_and_in_file_order():
    """A sampled demo whose selection cannot be reproduced is an anecdote."""
    source = ROOT / "data" / "train_weak.jsonl"
    first = select_products(source, 25)
    second = select_products(source, 25)
    assert first == second
    assert len(first) == 25
    assert all(isinstance(sku, str) and prompt for sku, prompt in first)


def test_selection_prompts_match_the_training_renderer():
    """Production must send the same text the model was trained against, or the
    demo measures a prompt nobody trained or evaluated on."""
    from labeling.lengths import render_prompt
    from labeling.records import read_jsonl

    source = ROOT / "data" / "train_weak.jsonl"
    rows = read_jsonl(source)[:5]
    selected = select_products(source, 5)
    assert [p for _s, p in selected] == [render_prompt(r) for r in rows]


# --- the report contract ------------------------------------------------------


def test_a_finished_report_is_never_overwritten(tmp_path, monkeypatch):
    """A report is evidence. Overwriting one silently replaces a measurement."""
    target = tmp_path / "finished"
    target.mkdir()
    (target / "report.json").write_text("{}")
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["run", "--products", "2", "--confirm-cost", "--out", str(target)])


# --- resume -------------------------------------------------------------------


def test_a_complete_pass_is_resumed_not_repurchased(tmp_path):
    """An interrupted run had already spent real money on two passes. A
    pipeline that cannot resume turns every interruption into a refund request
    nobody will honour."""
    from production.run_report import load_pass
    from production.tagger import TagResult, write_results

    results = [
        TagResult(
            sku_id=f"s{i}", idempotency_key="k", raw="{}", parsed={},
            schema_valid=True, vocab_valid=True, rule_violations=[],
            gate_passed=True, errors=[], attempts=1, latency_seconds=1.0,
            prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
        )
        for i in range(3)
    ]
    path = tmp_path / "pass-1.jsonl"
    write_results(path, results)

    loaded = load_pass(path, expected=3)
    assert loaded is not None and len(loaded) == 3
    assert loaded[0].sku_id == "s0" and loaded[0].gate_passed is True


def test_a_partial_pass_is_rerun_rather_than_trusted(tmp_path, capsys):
    """A pass truncated mid-write would otherwise be resumed as whole, and its
    missing products would silently shrink the sample the report is built on."""
    from production.run_report import load_pass
    from production.tagger import TagResult, write_results

    write_results(tmp_path / "pass-1.jsonl", [
        TagResult(
            sku_id="s0", idempotency_key="k", raw="{}", parsed={},
            schema_valid=True, vocab_valid=True, rule_violations=[],
            gate_passed=True, errors=[], attempts=1, latency_seconds=1.0,
            prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
        )
    ])
    assert load_pass(tmp_path / "pass-1.jsonl", expected=5) is None
    assert "1/5 rows" in capsys.readouterr().out


def test_an_absent_pass_is_simply_run(tmp_path):
    from production.run_report import load_pass

    assert load_pass(tmp_path / "nope.jsonl", expected=3) is None


# --- throughput must survive a resume ----------------------------------------


def test_throughput_is_derived_from_request_time_not_wall_clock():
    """A resumed pass reloads instantly, so wall clock measures file reads
    rather than tagging. Computing throughput from it reported 1,416,888
    products per minute on a run whose real rate was under 25."""
    report_path = ROOT / "runs" / "production-demo" / "report.json"
    if not report_path.exists():
        pytest.skip("no production report committed yet")
    performance = json.loads(report_path.read_text())["performance"]

    products = json.loads(report_path.read_text())["escalation"]["products"]
    expected = round(products / (performance["request_seconds"] / 60), 2)
    assert performance["products_per_minute"] == expected

    # The bound that would have caught the original defect. One sequential
    # process against a hosted API cannot exceed a request per second per
    # product, and p50 latency is seconds.
    assert performance["products_per_minute"] < 600, "throughput is not physical"


def test_the_report_records_which_passes_were_resumed():
    """Without this the reader cannot tell whether wall_seconds means anything."""
    report_path = ROOT / "runs" / "production-demo" / "report.json"
    if not report_path.exists():
        pytest.skip("no production report committed yet")
    performance = json.loads(report_path.read_text())["performance"]
    assert "resumed_passes" in performance
    assert isinstance(performance["resumed_passes"], list)
    if performance["resumed_passes"]:
        assert performance["request_seconds"] > performance["wall_seconds"], (
            "a resumed run must show more request time than wall time"
        )


def test_the_throughput_caveat_says_where_the_number_came_from():
    report_path = ROOT / "runs" / "production-demo" / "report.json"
    if not report_path.exists():
        pytest.skip("no production report committed yet")
    caveat = json.loads(report_path.read_text())["performance"]["caveat"]
    assert "wall clock" in caveat and "resume" in caveat
