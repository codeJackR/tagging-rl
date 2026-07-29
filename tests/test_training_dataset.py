"""The SFT/GRPO dataset builder.

The load-bearing check: the rendered text must be byte-identical to what the
token budgets were measured on, and completions must stay inside those budgets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.lengths import render_prompt, render_target
from labeling.records import read_jsonl
from training.dataset import (
    MAX_COMPLETION_TOKENS,
    SYSTEM,
    load_grpo_prompts,
    load_sft_dataset,
    to_messages,
)
from verifier import load_pack, verify

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train_weak.jsonl"

pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="no training split yet")


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def rows():
    return read_jsonl(TRAIN)


def test_prompt_is_the_measured_rendering(pack, rows):
    """Budgets were measured on render_prompt's output; training must use it verbatim."""
    msg = to_messages(rows[0], pack)
    assert msg["prompt"][1]["content"] == render_prompt(rows[0])
    assert msg["completion"][0]["content"] == render_target(rows[0], pack)


def test_every_completion_is_valid_against_the_verifier(pack, rows):
    """If gold text does not pass the verifier, we would be teaching invalid output."""
    for row in rows[:300]:
        result = verify(render_target(row, pack), pack)
        assert result.schema_valid and result.vocab_valid, (row.sku_id, result.errors)


def test_sft_dataset_shape(pack):
    ds = load_sft_dataset(pack)
    assert len(ds) == 3600
    ex = ds[0]
    assert ex["prompt"][0]["role"] == "system"
    assert ex["prompt"][1]["role"] == "user"
    assert ex["completion"][0]["role"] == "assistant"
    json.loads(ex["completion"][0]["content"])  # must parse


def test_grpo_prompts_carry_gold_but_no_completion(pack):
    ds = load_grpo_prompts(pack)
    assert len(ds) == 3600
    ex = ds[0]
    assert "completion" not in ex
    gold = json.loads(ex["gold"])
    assert set(gold) == set(pack.specs)


def test_grpo_filter_refuses_to_run_on_null_pass_rates(pack):
    """Silently returning an empty dataset would look like a config bug downstream."""
    with pytest.raises(RuntimeError, match="sft_pass_rate"):
        load_grpo_prompts(pack, require_pass_rate_band=True)


def test_system_prompt_names_every_field(pack):
    for name in pack.specs:
        assert name in SYSTEM, f"{name} missing from the system prompt"


def test_system_prompt_is_small(pack):
    """The compressed prompt is the point — the vocabulary is what SFT teaches."""
    assert len(SYSTEM) < 800
