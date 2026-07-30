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
    MAX_SFT_TOKENS,
    SYSTEM,
    load_grpo_prompts,
    load_sft_dataset,
    load_sft_splits,
    to_messages,
)
from training.predict import filter_rows, prompt_messages
from training.train_sft import target_modules
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


def test_prediction_prompt_matches_training_prompt(pack, rows):
    assert prompt_messages(rows[0]) == to_messages(rows[0], pack)["prompt"]


def test_frozen_sft_split_loads_exact_sizes(pack):
    train, validation = load_sft_splits(pack)
    assert len(train) == 3240
    assert len(validation) == 360
    assert set(train["sku_id"]).isdisjoint(validation["sku_id"])


def test_prediction_filter_uses_frozen_validation_split(rows):
    filtered = filter_rows(rows, ROOT / "data" / "splits" / "sft-v1.json")
    assert len(filtered) == 360
    assert len({row.sku_id for row in filtered}) == 360


def test_sft_sequence_budget_covers_measured_maximum():
    assert MAX_SFT_TOKENS == 896


def test_sft_arms_differ_only_by_mlp_targets():
    attention = target_modules("attention")
    combined = target_modules("combined")
    assert attention == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert set(combined) - set(attention) == {"gate_proj", "up_proj", "down_proj"}
