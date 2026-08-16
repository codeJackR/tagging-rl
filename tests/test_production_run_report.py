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


def test_an_existing_report_directory_is_never_overwritten(tmp_path, monkeypatch):
    """A report is evidence. Overwriting one silently replaces a measurement."""
    target = tmp_path / "existing"
    target.mkdir()
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["run", "--products", "2", "--confirm-cost", "--out", str(target)])
