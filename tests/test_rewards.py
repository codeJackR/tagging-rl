"""CPU-only contract tests for the first unconstrained GRPO rewards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.records import read_jsonl
from training.rewards import (
    FIRST_RUN_REWARD_FUNCTIONS,
    FIRST_RUN_REWARD_WEIGHTS,
    format_validity_reward,
    golden_agreement_reward,
    vocab_rule_compliance_reward,
)
from verifier import load_pack, verify

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train_weak_grpo_cap4.jsonl"


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def row():
    return read_jsonl(TRAIN)[0]


def chat_completions(*texts: str) -> list[list[dict[str, str]]]:
    """Match the conversational completion container passed by TRL 0.24."""
    return [[{"role": "assistant", "content": text}] for text in texts]


def find_clean_wrong_prediction(gold: dict, pack) -> dict:
    """Find a legal one-field error so this test does not depend on one rule."""
    for name, spec in pack.specs.items():
        original = gold[name]
        if isinstance(original, list):
            continue
        for candidate in spec.values:
            if candidate in {original, pack.unknown_token}:
                continue
            prediction = dict(gold)
            prediction[name] = candidate
            if verify(json.dumps(prediction), pack).ok:
                return prediction
    raise AssertionError("fixture has no verifier-clean wrong prediction")


def test_first_run_function_order_and_weights_are_locked():
    assert [function.__name__ for function in FIRST_RUN_REWARD_FUNCTIONS] == [
        "format_validity_reward",
        "vocab_rule_compliance_reward",
        "golden_agreement_reward",
    ]
    assert FIRST_RUN_REWARD_WEIGHTS == (1.0, 1.0, 2.0)


def test_format_reward_uses_literal_schema_validity(pack):
    first_field = next(iter(pack.specs))
    out_of_vocab = json.dumps({first_field: "__not_in_vocab__"})

    rewards = format_validity_reward(
        chat_completions("not json", out_of_vocab, "{}"),
        pack=pack,
        prompts=["ignored"] * 3,
        sku_id=["ignored"] * 3,
    )

    # OOV text is structurally valid JSON; vocabulary is a separate component.
    assert rewards == [0.0, 1.0, 1.0]


def test_compliance_reward_requires_schema_vocab_and_rules(pack, row):
    gold = row.to_verifier_record(pack)
    first_field = next(iter(pack.specs))
    out_of_vocab = json.dumps({first_field: "__not_in_vocab__"})

    rewards = vocab_rule_compliance_reward(
        chat_completions("not json", out_of_vocab, "{}", json.dumps(gold)),
        pack=pack,
    )

    # The optional-field schema makes {} verifier-clean. This is the deliberate
    # empty-but-valid loophole that the first unconstrained run is meant to expose.
    assert rewards == [0.0, 0.0, 1.0, 1.0]


def test_golden_agreement_rejects_empty_and_clean_but_wrong(pack, row):
    gold = row.to_verifier_record(pack)
    wrong = find_clean_wrong_prediction(gold, pack)
    encoded_gold = json.dumps(gold, sort_keys=True)

    rewards = golden_agreement_reward(
        chat_completions("{}", json.dumps(wrong), json.dumps(gold)),
        gold=[encoded_gold, encoded_gold, encoded_gold],
        pack=pack,
    )

    assert rewards == [0.0, 0.0, 1.0]


def test_golden_agreement_excludes_unknown_gold_fields(pack, row):
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    unknown_field = next(
        name for name, label in row.labels.items() if label.status.value == "unknown"
    )
    prediction[unknown_field] = next(
        value
        for value in pack.specs[unknown_field].values
        if value != pack.unknown_token
    )
    assert verify(json.dumps(prediction), pack).ok

    rewards = golden_agreement_reward(
        chat_completions(json.dumps(prediction)),
        gold=[json.dumps(gold)],
        pack=pack,
    )

    assert rewards == [1.0]


def test_rewards_also_accept_plain_text_completions(pack, row):
    gold = row.to_verifier_record(pack)

    assert format_validity_reward([json.dumps(gold)], pack=pack) == [1.0]
    assert vocab_rule_compliance_reward([json.dumps(gold)], pack=pack) == [1.0]
    assert golden_agreement_reward(
        [json.dumps(gold)], gold=[gold], pack=pack
    ) == [1.0]


def test_default_pack_and_trl_keyword_interface_work_without_trainer_import(row):
    gold = row.to_verifier_record(load_pack(ROOT / "packs" / "vastraa_taste_v1"))
    completion = chat_completions(json.dumps(gold))
    shared_kwargs = {
        "prompts": [[{"role": "user", "content": "ignored"}]],
        "completion_ids": [[1, 2, 3]],
        "sku_id": [row.sku_id],
        "trainer_state": object(),
    }

    assert format_validity_reward(
        completions=completion, **shared_kwargs
    ) == [1.0]
    assert vocab_rule_compliance_reward(
        completions=completion, **shared_kwargs
    ) == [1.0]
    assert golden_agreement_reward(
        completions=completion,
        gold=[json.dumps(gold)],
        **shared_kwargs,
    ) == [1.0]


def test_integration_shape_errors_raise_instead_of_becoming_model_rewards(pack, row):
    with pytest.raises(TypeError, match="one assistant message"):
        format_validity_reward([[{"role": "user", "content": "{}"}]], pack=pack)

    with pytest.raises(ValueError, match="same length"):
        golden_agreement_reward(["{}", "{}"], gold=[{}], pack=pack)

    with pytest.raises(ValueError, match=r"gold\[0\] is not valid JSON"):
        golden_agreement_reward(["{}"], gold=["not json"], pack=pack)
