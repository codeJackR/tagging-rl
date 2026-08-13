"""Adversarial CPU tests for the GRPO Run 2 candidate reward contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from labeling.records import LabelStatus, read_jsonl
import training.run2_rewards as run2_rewards
from training.reward_scale_contract import (
    KNOWN_MIX_WEIGHT,
    MALFORMED_FLOOR,
    UNKNOWN_MIX_WEIGHT,
)
from training.run2_rewards import (
    CBClassWeightLookup,
    candidate_ua_reward,
    candidate_u_reward,
    load_cb_class_weight_lookup,
    make_candidate_cb_reward,
    prepare_cb_class_weight_lookup,
    score_class_balanced_known_fields,
    score_class_balanced_unknown_aware_semantics,
    score_candidate_cb,
    score_candidate_ua,
    score_candidate_u,
    score_uniform_known_fields,
    score_uniform_unknown_aware_semantics,
    score_uniform_unknown_fields,
    strict_semantic_gate,
)
from verifier import load_pack, verify


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def cb_artifact() -> dict:
    return json.loads(
        (ROOT / "runs" / "grpo-run2-cb-class-weights.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def cb_lookup(pack) -> CBClassWeightLookup:
    return load_cb_class_weight_lookup(pack)


@pytest.fixture(scope="module")
def active_rows_by_sku() -> dict:
    rows = read_jsonl(ROOT / "data" / "train_weak_grpo_cap4_sft_train_v1.jsonl")
    return {row.sku_id: row for row in rows}


@pytest.fixture()
def complete_record() -> dict:
    return {
        "garment_category": "dress",
        "silhouette": "a_line",
        "fit": "regular",
        "garment_length": "midi",
        "sleeve_length": "three_quarter",
        "sleeve_style": "set_in",
        "neckline": "v_neck",
        "collar_type": "none",
        "waistline": "normal",
        "closure": "zip",
        "pattern": "floral",
        "details": ["lined"],
        "material": "silk",
        "colour_primary": "navy",
        "occasion": "work",
    }


def test_gate_accepts_complete_clean_and_rule_violating_records(pack, complete_record):
    clean = strict_semantic_gate(json.dumps(complete_record), pack)
    assert clean.eligible
    assert clean.parsed == complete_record
    assert clean.errors == ()
    assert clean.rule_violations == ()

    incoherent = dict(complete_record)
    incoherent["sleeve_length"] = "sleeveless"
    incoherent["sleeve_style"] = "puff"
    result = strict_semantic_gate(json.dumps(incoherent), pack)
    assert result.eligible
    assert "sleeveless_has_no_sleeve_style" in result.rule_violations


@pytest.mark.parametrize(
    ("raw_output", "message"),
    [
        ("not JSON", "not literal JSON"),
        ("[1,2,3]", "top level must be an object"),
        ("```json\n{}\n```", "not literal JSON"),
        ("{}", "missing fields"),
        ('{"details":[]}', "missing fields"),
        ('{"details":["unknown","none"]}', "missing fields"),
    ],
)
def test_gate_rejects_literal_and_completeness_failures(pack, raw_output, message):
    result = strict_semantic_gate(raw_output, pack)
    assert not result.eligible
    assert any(message in error for error in result.errors)


def test_gate_closes_the_three_base_verifier_loopholes(pack):
    loopholes = ("{}", '{"details":[]}', '{"details":["unknown","none"]}')
    for raw_output in loopholes:
        assert verify(raw_output, pack).ok
        assert not strict_semantic_gate(raw_output, pack).eligible


def test_gate_rejects_duplicate_json_keys_before_last_value_wins(pack, complete_record):
    pairs = [f"{json.dumps(key)}:{json.dumps(value)}" for key, value in complete_record.items()]
    pairs.append('"garment_category":"coat"')
    raw_output = "{" + ",".join(pairs) + "}"

    # Python's normal decoder silently keeps "coat"; the Run 2 gate must not.
    assert json.loads(raw_output)["garment_category"] == "coat"
    result = strict_semantic_gate(raw_output, pack)
    assert not result.eligible
    assert result.errors == ("gate: duplicate JSON key: garment_category",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"details": []}, "details must not be empty"),
        ({"details": ["unknown", "none"]}, "cannot mix unknown"),
        ({"details": ["lined", "lined"]}, "contains duplicate values"),
        ({"garment_category": "not_in_vocab"}, "vocab:"),
        ({"garment_category": ["dress"]}, "schema:"),
    ],
)
def test_gate_rejects_full_but_ambiguous_or_invalid_records(
    pack, complete_record, mutation, message
):
    candidate = {**complete_record, **mutation}
    result = strict_semantic_gate(json.dumps(candidate), pack)
    assert not result.eligible
    assert any(message in error for error in result.errors)


def test_gate_requires_exact_field_set(pack, complete_record):
    missing = dict(complete_record)
    missing.pop("occasion")
    missing_result = strict_semantic_gate(json.dumps(missing), pack)
    assert not missing_result.eligible
    assert missing_result.errors == ("gate: missing fields: occasion",)

    extra = {**complete_record, "explanation": "looks like a dress"}
    extra_result = strict_semantic_gate(json.dumps(extra), pack)
    assert not extra_result.eligible
    assert extra_result.errors == ("gate: extra fields: explanation",)


def test_gate_accepts_complete_all_abstain_and_null_multi_value(pack):
    all_abstain = {
        field_name: ([pack.unknown_token] if spec.kind == "multi" else pack.unknown_token)
        for field_name, spec in pack.specs.items()
    }
    assert strict_semantic_gate(json.dumps(all_abstain), pack).eligible

    null_details = dict(all_abstain)
    null_details["details"] = None
    assert strict_semantic_gate(json.dumps(null_details), pack).eligible


def test_non_string_input_is_an_integration_error(pack):
    with pytest.raises(TypeError, match="raw_output must be a string"):
        strict_semantic_gate({}, pack)  # type: ignore[arg-type]


def _all_unknown(pack) -> dict:
    return {
        field_name: ([pack.unknown_token] if spec.kind == "multi" else pack.unknown_token)
        for field_name, spec in pack.specs.items()
    }


@pytest.mark.parametrize(
    ("prediction_value", "expected_score", "expected_outcome"),
    [
        ("dress", 1.0, "correct"),
        ("unknown", 0.0, "abstain"),
        ("coat", -1.0, "wrong"),
        (None, -1.0, "wrong"),
    ],
)
def test_candidate_u_scalar_correct_abstain_and_wrong(
    pack, prediction_value, expected_score, expected_outcome
):
    gold = _all_unknown(pack)
    gold["garment_category"] = "dress"
    prediction = _all_unknown(pack)
    prediction["garment_category"] = prediction_value

    result = score_uniform_known_fields(prediction, gold, pack)
    assert result.semantic_score == expected_score
    assert result.scorable_fields == 1
    assert result.field_outcomes[0].outcome == expected_outcome
    assert len(result.excluded_gold_unknown_fields) == 14


@pytest.mark.parametrize(
    ("prediction_value", "expected_score", "expected_outcome"),
    [
        (None, 1.0, "correct"),
        ("unknown", 0.0, "abstain"),
        ("crew", -1.0, "wrong"),
    ],
)
def test_candidate_u_treats_null_as_not_applicable_not_abstention(
    pack, prediction_value, expected_score, expected_outcome
):
    gold = _all_unknown(pack)
    gold["neckline"] = None
    prediction = _all_unknown(pack)
    prediction["neckline"] = prediction_value

    result = score_uniform_known_fields(prediction, gold, pack)
    assert result.semantic_score == expected_score
    assert result.field_outcomes[0].outcome == expected_outcome


@pytest.mark.parametrize(
    ("predicted", "expected_f1", "expected_utility", "expected_outcome"),
    [
        (["lined", "washed"], 1.0, 1.0, "correct"),
        (["washed", "lined"], 1.0, 1.0, "correct"),
        (["lined"], 2 / 3, 1 / 3, "partial"),
        (["lined", "embroidered"], 0.5, 0.0, "partial"),
        (["embroidered"], 0.0, -1.0, "wrong"),
        (["unknown"], None, 0.0, "abstain"),
        (None, None, -1.0, "wrong"),
    ],
)
def test_candidate_u_details_uses_monotonic_set_f1(
    pack, predicted, expected_f1, expected_utility, expected_outcome
):
    gold = _all_unknown(pack)
    gold["details"] = ["lined", "washed"]
    prediction = _all_unknown(pack)
    prediction["details"] = predicted

    result = score_uniform_known_fields(prediction, gold, pack)
    outcome = result.field_outcomes[0]
    assert outcome.outcome == expected_outcome
    assert outcome.utility == pytest.approx(expected_utility)
    if expected_f1 is None:
        assert outcome.set_f1 is None
    else:
        assert outcome.set_f1 == pytest.approx(expected_f1)
    assert result.semantic_score == pytest.approx(expected_utility)


def test_candidate_u_normalizes_by_known_field_count(pack):
    gold = _all_unknown(pack)
    gold["garment_category"] = "dress"
    gold["occasion"] = "work"
    prediction = _all_unknown(pack)
    prediction["garment_category"] = "dress"
    prediction["occasion"] = "casual"

    result = score_uniform_known_fields(prediction, gold, pack)
    assert result.scorable_fields == 2
    assert result.semantic_score == 0.0
    assert [outcome.utility for outcome in result.field_outcomes] == [1.0, -1.0]


def test_candidate_u_excludes_gold_unknown_prediction_changes(pack):
    gold = _all_unknown(pack)
    gold["garment_category"] = "dress"
    abstaining = _all_unknown(pack)
    abstaining["garment_category"] = "dress"
    committing = dict(abstaining)
    committing["occasion"] = "formal"

    abstain_score = score_uniform_known_fields(abstaining, gold, pack)
    commit_score = score_uniform_known_fields(committing, gold, pack)
    assert abstain_score.semantic_score == commit_score.semantic_score == 1.0
    assert "occasion" in commit_score.excluded_gold_unknown_fields


def test_candidate_u_fails_closed_on_all_unknown_or_invalid_records(pack):
    all_unknown = _all_unknown(pack)
    with pytest.raises(ValueError, match="at least one gold-known field"):
        score_uniform_known_fields(all_unknown, all_unknown, pack)

    incomplete = dict(all_unknown)
    incomplete.pop("occasion")
    gold = dict(all_unknown)
    gold["garment_category"] = "dress"
    with pytest.raises(ValueError, match="strict semantic gate"):
        score_uniform_known_fields(incomplete, gold, pack)

    invalid_gold = dict(gold)
    invalid_gold["details"] = []
    with pytest.raises(ValueError, match="gold must satisfy"):
        score_uniform_known_fields(all_unknown, invalid_gold, pack)


def test_candidate_cb_loads_hash_locked_read_only_lookup(pack, cb_lookup):
    assert cb_lookup.artifact_version == "grpo-run2-cb-class-weights-v1"
    assert cb_lookup.field_names == pack.field_names
    assert sum(len(classes) for classes in cb_lookup.weights.values()) == 116
    with pytest.raises(TypeError):
        cb_lookup.weights["garment_category"]["vest"] = 1.0  # type: ignore[index]


def test_candidate_cb_changes_only_weighted_aggregation(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    prediction["sleeve_length"] = "long"  # wrong versus gold N/A
    prediction["sleeve_style"] = pack.unknown_token  # abstain versus gold N/A

    uniform = score_uniform_known_fields(prediction, gold, pack)
    balanced = score_class_balanced_known_fields(
        prediction, gold, pack, cb_lookup
    )

    assert [outcome.utility for outcome in uniform.field_outcomes] == [
        1.0,
        -1.0,
        0.0,
        1.0,
    ]
    assert [outcome.utility for outcome in balanced.field_outcomes] == [
        outcome.utility for outcome in uniform.field_outcomes
    ]
    expected_weights = [
        cb_lookup.weights["garment_category"]["vest"],
        cb_lookup.weights["sleeve_length"]["__not_applicable__"],
        cb_lookup.weights["sleeve_style"]["__not_applicable__"],
        cb_lookup.weights["waistline"]["__not_applicable__"],
    ]
    assert [outcome.field_weight for outcome in balanced.field_outcomes] == (
        expected_weights
    )
    expected_score = sum(
        utility * weight
        for utility, weight in zip(
            [1.0, -1.0, 0.0, 1.0], expected_weights, strict=True
        )
    ) / sum(expected_weights)
    assert uniform.semantic_score == 0.25
    assert balanced.semantic_score == pytest.approx(expected_score)
    assert balanced.total_field_weight == pytest.approx(sum(expected_weights))
    assert len(balanced.excluded_gold_unknown_fields) == 11


def test_candidate_cb_details_uses_mean_gold_label_weight_once(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.thursdayboots.com:6546548424794"]
    gold = row.to_verifier_record(pack)
    result = score_class_balanced_known_fields(gold, gold, pack, cb_lookup)
    details = next(
        outcome for outcome in result.field_outcomes if outcome.field_name == "details"
    )

    assert details.gold_class_keys == ("studded", "lined")
    assert details.gold_class_weights == pytest.approx(
        (
            cb_lookup.weights["details"]["studded"],
            cb_lookup.weights["details"]["lined"],
        )
    )
    assert details.field_weight == pytest.approx(
        sum(details.gold_class_weights) / 2
    )
    assert sum(outcome.field_name == "details" for outcome in result.field_outcomes) == 1
    assert result.scorable_fields == sum(
        label.status is not LabelStatus.UNKNOWN for label in row.labels.values()
    )
    assert result.semantic_score == 1.0


def test_candidate_cb_fails_closed_when_gold_class_is_missing_from_lookup(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    copied_weights = {
        field_name: dict(classes)
        for field_name, classes in cb_lookup.weights.items()
    }
    copied_weights["garment_category"].pop("vest")
    incomplete_lookup = CBClassWeightLookup(
        artifact_version=cb_lookup.artifact_version,
        field_names=cb_lookup.field_names,
        unknown_token=cb_lookup.unknown_token,
        weights=copied_weights,
    )

    with pytest.raises(ValueError, match="garment_category/vest"):
        score_class_balanced_known_fields(gold, gold, pack, incomplete_lookup)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda artifact: artifact["invariants"].__setitem__(
                "all_weights_within_locked_bounds", False
            ),
            "failed invariant",
        ),
        (
            lambda artifact: artifact["weight_map"]["attributes"].pop("occasion"),
            "differ from pack fields",
        ),
        (
            lambda artifact: artifact["weight_map"]["attributes"][
                "garment_category"
            ]["classes"]["vest"].__setitem__("weight", 1.5),
            "clipped weight mismatch",
        ),
    ],
)
def test_candidate_cb_preparation_rejects_inconsistent_artifacts(
    pack, cb_artifact, mutation, message
):
    altered = deepcopy(cb_artifact)
    mutation(altered)
    with pytest.raises(ValueError, match=message):
        prepare_cb_class_weight_lookup(altered, pack)


def test_candidate_cb_loader_rejects_any_file_hash_change(pack, cb_artifact, tmp_path):
    altered = deepcopy(cb_artifact)
    altered["role"] = "changed after lock"
    path = tmp_path / "altered-cb-map.json"
    path.write_text(json.dumps(altered), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_cb_class_weight_lookup(pack, path)


def test_candidate_u_composes_clean_semantics_and_zero_rule_cost(pack, complete_record):
    result = score_candidate_u(json.dumps(complete_record), complete_record, pack)

    assert result.candidate == "U"
    assert result.eligible
    assert result.reward == 1.0
    assert result.gate_errors == ()
    assert result.rule_violations == ()
    assert result.rule_adjustment == 0.0
    assert result.known_semantics is not None
    assert result.known_semantics.semantic_score == 1.0
    assert result.known_semantics.scorable_fields == 15


@pytest.mark.parametrize("raw_output", ["not JSON", "{}", "[1,2,3]"])
def test_candidate_u_assigns_only_the_fixed_floor_to_ineligible_output(
    pack, complete_record, raw_output
):
    result = score_candidate_u(raw_output, complete_record, pack)

    assert not result.eligible
    assert result.reward == MALFORMED_FLOOR == -1.25
    assert result.gate_errors
    assert result.rule_violations == ()
    assert result.rule_adjustment is None
    assert result.known_semantics is None


def test_candidate_u_adds_one_bounded_cost_per_rule_violation(pack, complete_record):
    incoherent = dict(complete_record)
    incoherent["sleeve_length"] = "sleeveless"
    incoherent["sleeve_style"] = "puff"
    incoherent["pattern"] = "solid"
    incoherent["colour_primary"] = "multicolour"

    # Exact agreement isolates rule composition from field-level correctness.
    result = score_candidate_u(json.dumps(incoherent), incoherent, pack)
    assert result.eligible
    assert set(result.rule_violations) == {
        "sleeveless_has_no_sleeve_style",
        "solid_is_not_multicolour",
    }
    assert result.known_semantics is not None
    assert result.known_semantics.semantic_score == 1.0
    assert result.rule_adjustment == pytest.approx(-0.10)
    assert result.reward == pytest.approx(0.90)


def test_candidate_u_composes_nontrivial_semantics_before_rule_cost(
    pack, complete_record
):
    prediction = dict(complete_record)
    prediction["occasion"] = "casual"  # wrong: -1 instead of +1
    prediction["closure"] = "unknown"  # abstain: 0 instead of +1

    result = score_candidate_u(json.dumps(prediction), complete_record, pack)
    assert result.eligible
    assert result.rule_violations == ()
    assert result.known_semantics is not None
    assert result.known_semantics.semantic_score == pytest.approx(12 / 15)
    assert result.reward == pytest.approx(12 / 15)


def test_candidate_u_validates_gold_even_when_model_output_is_malformed(
    pack, complete_record
):
    invalid_gold = dict(complete_record)
    invalid_gold.pop("occasion")

    with pytest.raises(ValueError, match="gold must satisfy"):
        score_candidate_u("not JSON", invalid_gold, pack)

    all_unknown = _all_unknown(pack)
    with pytest.raises(ValueError, match="at least one gold-known field"):
        score_candidate_u("not JSON", all_unknown, pack)


def _chat_completions(*texts: str) -> list[list[dict[str, str]]]:
    return [[{"role": "assistant", "content": text}] for text in texts]


def test_candidate_u_batch_preserves_plain_text_order_and_accepts_both_gold_shapes(
    pack, complete_record
):
    wrong = dict(complete_record)
    wrong["occasion"] = "casual"
    completions = [
        json.dumps(complete_record),
        "{}",
        json.dumps(wrong),
    ]
    gold = [complete_record, json.dumps(complete_record), complete_record]

    rewards = candidate_u_reward(
        completions,
        gold,
        pack=pack,
        prompts=["ignored"] * 3,
        sku_id=["a", "b", "c"],
        trainer_state=object(),
    )
    assert rewards == pytest.approx([1.0, MALFORMED_FLOOR, 13 / 15])


def test_candidate_u_batch_accepts_conversational_completion_shape(
    pack, complete_record
):
    rewards = candidate_u_reward(
        _chat_completions(json.dumps(complete_record), "not JSON"),
        [json.dumps(complete_record), complete_record],
        pack=pack,
    )
    assert rewards == [1.0, MALFORMED_FLOOR]


def test_candidate_u_batch_default_pack_works_without_trl_import(complete_record):
    rewards = candidate_u_reward(
        completions=_chat_completions(json.dumps(complete_record)),
        gold=[json.dumps(complete_record)],
        completion_ids=[[1, 2, 3]],
    )
    assert rewards == [1.0]


def test_candidate_u_batch_rejects_misalignment_and_scalar_sequences(
    pack, complete_record
):
    encoded = json.dumps(complete_record)
    with pytest.raises(ValueError, match="same length: got 1 and 2"):
        candidate_u_reward([encoded, encoded], [complete_record], pack=pack)
    with pytest.raises(TypeError, match="gold must be a sequence"):
        candidate_u_reward([encoded], encoded, pack=pack)
    with pytest.raises(TypeError, match="completions must be a sequence"):
        candidate_u_reward(encoded, [complete_record], pack=pack)


def test_candidate_u_batch_rejects_bad_containers_and_indexed_gold_json(
    pack, complete_record
):
    with pytest.raises(TypeError, match="one assistant message"):
        candidate_u_reward(
            [[{"role": "user", "content": "{}"}]],
            [complete_record],
            pack=pack,
        )
    with pytest.raises(ValueError, match=r"gold\[1\] is not valid JSON"):
        candidate_u_reward(
            [json.dumps(complete_record), "{}"],
            [complete_record, "not JSON"],
            pack=pack,
        )
    with pytest.raises(TypeError, match=r"gold\[0\] must decode to a JSON object"):
        candidate_u_reward(["{}"], [["not", "an", "object"]], pack=pack)


def test_candidate_u_batch_validates_all_gold_before_scoring_model_text(
    pack, complete_record
):
    invalid_second_gold = dict(complete_record)
    invalid_second_gold.pop("occasion")

    with pytest.raises(ValueError, match="gold must satisfy"):
        candidate_u_reward(
            ["not JSON", json.dumps(complete_record)],
            [complete_record, invalid_second_gold],
            pack=pack,
        )


def test_candidate_u_batch_empty_alignment_is_an_empty_map(pack):
    assert candidate_u_reward([], [], pack=pack) == []


@pytest.mark.parametrize(
    ("prediction_value", "expected_score", "expected_outcome"),
    [
        ("unknown", 1.0, "abstain"),
        ("dress", -1.0, "commit"),
        (None, -1.0, "commit"),
    ],
)
def test_candidate_ua_unknown_scalar_abstention_vs_commitment(
    pack, complete_record, prediction_value, expected_score, expected_outcome
):
    gold = dict(complete_record)
    gold["garment_category"] = "unknown"
    prediction = dict(complete_record)
    prediction["garment_category"] = prediction_value

    result = score_uniform_unknown_fields(prediction, gold, pack)
    assert result.semantic_score == expected_score
    assert result.scorable_fields == 1
    target = result.field_outcomes[0]
    assert target.field_name == "garment_category"
    assert target.outcome == expected_outcome
    assert target.utility == expected_score


@pytest.mark.parametrize(
    ("prediction_value", "expected_score", "expected_outcome"),
    [
        (["unknown"], 1.0, "abstain"),
        (["lined"], -1.0, "commit"),
        (None, -1.0, "commit"),
    ],
)
def test_candidate_ua_unknown_multi_value_abstention_vs_commitment(
    pack, complete_record, prediction_value, expected_score, expected_outcome
):
    gold = dict(complete_record)
    gold["details"] = ["unknown"]
    prediction = dict(complete_record)
    prediction["details"] = prediction_value

    result = score_uniform_unknown_fields(prediction, gold, pack)
    assert result.scorable_fields == 1
    assert result.semantic_score == expected_score
    target = result.field_outcomes[0]
    assert target.field_name == "details"
    assert target.outcome == expected_outcome
    assert target.utility == expected_score


def test_candidate_ua_normalizes_unknowns_separately_from_known_fields(
    pack, complete_record
):
    # Make exactly two fields unknown and every other field known using a valid
    # clean record, then balance one abstention and one unsupported commitment.
    gold = dict(complete_record)
    unknown_fields = ("material", "occasion")
    for field_name in unknown_fields:
        gold[field_name] = pack.unknown_token
    prediction = dict(gold)
    prediction[unknown_fields[0]] = pack.unknown_token
    prediction[unknown_fields[1]] = next(iter(pack.specs[unknown_fields[1]].values))

    result = score_uniform_unknown_fields(prediction, gold, pack)
    assert result.scorable_fields == 2
    assert result.semantic_score == 0.0
    assert [outcome.utility for outcome in result.field_outcomes] == [1.0, -1.0]
    assert len(result.excluded_gold_known_fields) == 13


def test_candidate_ua_marks_component_absent_when_gold_has_no_unknowns(
    pack, complete_record
):
    result = score_uniform_unknown_fields(complete_record, complete_record, pack)

    assert result.semantic_score is None
    assert result.scorable_fields == 0
    assert result.field_outcomes == ()
    assert result.excluded_gold_known_fields == tuple(pack.field_names)


def test_candidate_ua_unknown_component_rejects_invalid_records(pack, complete_record):
    invalid_prediction = dict(complete_record)
    invalid_prediction["details"] = []
    with pytest.raises(ValueError, match="strict semantic gate"):
        score_uniform_unknown_fields(invalid_prediction, complete_record, pack)

    invalid_gold = dict(complete_record)
    invalid_gold.pop("occasion")
    with pytest.raises(ValueError, match="gold must satisfy"):
        score_uniform_unknown_fields(complete_record, invalid_gold, pack)

    with pytest.raises(TypeError, match="prediction must be a mapping"):
        score_uniform_unknown_fields("{}", complete_record, pack)  # type: ignore[arg-type]


def test_candidate_ua_combines_locked_known_and_unknown_weights(
    pack, complete_record
):
    gold = dict(complete_record)
    gold["occasion"] = pack.unknown_token
    prediction = dict(complete_record)
    prediction["occasion"] = "formal"

    result = score_uniform_unknown_aware_semantics(prediction, gold, pack)
    assert result.known_semantics.semantic_score == 1.0
    assert result.unknown_semantics.semantic_score == -1.0
    assert result.semantic_score == pytest.approx(
        KNOWN_MIX_WEIGHT - UNKNOWN_MIX_WEIGHT
    )


def test_candidate_ua_no_unknown_component_preserves_known_score_exactly(
    pack, complete_record
):
    prediction = dict(complete_record)
    prediction["occasion"] = "casual"

    result = score_uniform_unknown_aware_semantics(
        prediction, complete_record, pack
    )
    assert result.unknown_semantics.semantic_score is None
    assert result.known_semantics.semantic_score == pytest.approx(13 / 15)
    assert result.semantic_score == result.known_semantics.semantic_score


def test_candidate_ua_combination_depends_on_component_means_not_unknown_count(
    pack, complete_record
):
    scores = []
    for unknown_fields in (("occasion",), ("occasion", "material")):
        gold = dict(complete_record)
        prediction = dict(complete_record)
        for field_name in unknown_fields:
            gold[field_name] = pack.unknown_token
            prediction[field_name] = None
        result = score_uniform_unknown_aware_semantics(prediction, gold, pack)
        assert result.known_semantics.semantic_score == 1.0
        assert result.unknown_semantics.semantic_score == -1.0
        scores.append(result.semantic_score)

    assert scores[0] == scores[1]


def test_candidate_ua_combined_semantics_reaches_both_locked_bounds(pack):
    gold = _all_unknown(pack)
    gold["garment_category"] = "dress"

    maximum = score_uniform_unknown_aware_semantics(gold, gold, pack)
    assert maximum.known_semantics.semantic_score == 1.0
    assert maximum.unknown_semantics.semantic_score == 1.0
    assert maximum.semantic_score == 1.0

    all_null = {field_name: None for field_name in pack.field_names}
    minimum = score_uniform_unknown_aware_semantics(all_null, gold, pack)
    assert minimum.known_semantics.semantic_score == -1.0
    assert minimum.unknown_semantics.semantic_score == -1.0
    assert minimum.semantic_score == -1.0


def test_candidate_ua_combined_result_retains_both_audit_ledgers(
    pack, complete_record
):
    gold = dict(complete_record)
    gold["material"] = pack.unknown_token
    prediction = dict(gold)

    result = score_uniform_unknown_aware_semantics(prediction, gold, pack)
    assert result.known_semantics.scorable_fields == 14
    assert result.unknown_semantics.scorable_fields == 1
    assert len(result.known_semantics.field_outcomes) == 14
    assert result.unknown_semantics.field_outcomes[0].field_name == "material"


def test_candidate_cb_combines_locked_known_and_unknown_population_weights(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    for field_name in (
        field_name
        for field_name, value in gold.items()
        if value == pack.unknown_token or value == [pack.unknown_token]
    ):
        prediction[field_name] = None

    result = score_class_balanced_unknown_aware_semantics(
        prediction, gold, pack, cb_lookup
    )
    assert result.known_semantics.semantic_score == 1.0
    assert result.unknown_semantics.semantic_score == -1.0
    assert result.semantic_score == pytest.approx(
        KNOWN_MIX_WEIGHT - UNKNOWN_MIX_WEIGHT
    )


def test_candidate_cb_no_unknown_component_preserves_weighted_known_score_exactly(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.allbirds.com:7257961955408"]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    prediction["occasion"] = "formal"

    known = score_class_balanced_known_fields(prediction, gold, pack, cb_lookup)
    combined = score_class_balanced_unknown_aware_semantics(
        prediction, gold, pack, cb_lookup
    )
    assert combined.unknown_semantics.semantic_score is None
    assert combined.known_semantics == known
    assert combined.semantic_score == known.semantic_score


def test_candidate_cb_combination_depends_on_component_means_not_unknown_count(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.allbirds.com:7257961955408"]
    original = row.to_verifier_record(pack)
    scores = []
    for unknown_fields in (("occasion",), ("occasion", "material")):
        gold = dict(original)
        prediction = dict(original)
        for field_name in unknown_fields:
            gold[field_name] = pack.unknown_token
            prediction[field_name] = None
        result = score_class_balanced_unknown_aware_semantics(
            prediction, gold, pack, cb_lookup
        )
        assert result.known_semantics.semantic_score == 1.0
        assert result.unknown_semantics.semantic_score == -1.0
        scores.append(result.semantic_score)

    assert scores[0] == scores[1]


def test_candidate_cb_differs_from_ua_only_through_known_field_weighting(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    prediction["sleeve_length"] = "long"
    prediction["sleeve_style"] = pack.unknown_token

    ua = score_uniform_unknown_aware_semantics(prediction, gold, pack)
    cb = score_class_balanced_unknown_aware_semantics(
        prediction, gold, pack, cb_lookup
    )
    assert cb.unknown_semantics == ua.unknown_semantics
    assert ua.known_semantics.semantic_score == 0.25
    assert cb.known_semantics.semantic_score == pytest.approx(0.48676357208115734)
    assert cb.semantic_score - ua.semantic_score == pytest.approx(
        KNOWN_MIX_WEIGHT
        * (cb.known_semantics.semantic_score - ua.known_semantics.semantic_score)
    )


def test_candidate_cb_composes_clean_exact_record(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.allbirds.com:7257961955408"]
    gold = row.to_verifier_record(pack)
    result = score_candidate_cb(json.dumps(gold), gold, pack, cb_lookup)

    assert result.candidate == "CB"
    assert result.eligible
    assert result.reward == 1.0
    assert result.gate_errors == ()
    assert result.rule_violations == ()
    assert result.rule_adjustment == 0.0
    assert result.unknown_aware_semantics is not None
    assert result.unknown_aware_semantics.semantic_score == 1.0
    assert result.unknown_aware_semantics.unknown_semantics.semantic_score is None


@pytest.mark.parametrize("raw_output", ["not JSON", "{}", "[1,2,3]"])
def test_candidate_cb_assigns_only_fixed_floor_to_ineligible_output(
    pack, cb_lookup, active_rows_by_sku, raw_output
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    result = score_candidate_cb(raw_output, gold, pack, cb_lookup)

    assert not result.eligible
    assert result.reward == MALFORMED_FLOOR
    assert result.gate_errors
    assert result.rule_violations == ()
    assert result.rule_adjustment is None
    assert result.unknown_aware_semantics is None


def test_candidate_cb_composes_nontrivial_semantics_without_rule_cost(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    prediction = dict(gold)
    prediction["garment_category"] = "top"

    result = score_candidate_cb(
        json.dumps(prediction), gold, pack, cb_lookup
    )
    assert result.eligible
    assert result.rule_violations == ()
    assert result.rule_adjustment == 0.0
    assert result.unknown_aware_semantics is not None
    assert result.unknown_aware_semantics.semantic_score == pytest.approx(
        0.3717332035607318
    )
    assert result.reward == result.unknown_aware_semantics.semantic_score


def test_candidate_cb_applies_shared_rule_cost_once_after_semantics(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    exact_but_incoherent = row.to_verifier_record(pack)
    exact_but_incoherent.update(
        garment_category="top",
        sleeve_length="sleeveless",
        sleeve_style="puff",
        pattern="solid",
        colour_primary="multicolour",
    )

    result = score_candidate_cb(
        json.dumps(exact_but_incoherent),
        exact_but_incoherent,
        pack,
        cb_lookup,
    )
    assert result.eligible
    assert set(result.rule_violations) == {
        "sleeveless_has_no_sleeve_style",
        "solid_is_not_multicolour",
    }
    assert result.unknown_aware_semantics is not None
    assert result.unknown_aware_semantics.semantic_score == 1.0
    assert result.rule_adjustment == pytest.approx(-0.10)
    assert result.reward == pytest.approx(0.90)


def test_candidate_cb_validates_gold_and_lookup_before_malformed_model_text(
    pack, cb_lookup, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    invalid_gold = dict(gold)
    invalid_gold.pop("occasion")
    with pytest.raises(ValueError, match="gold must satisfy"):
        score_candidate_cb("not JSON", invalid_gold, pack, cb_lookup)

    with pytest.raises(ValueError, match="at least one gold-known field"):
        score_candidate_cb("not JSON", _all_unknown(pack), pack, cb_lookup)

    copied_weights = {
        field_name: dict(classes)
        for field_name, classes in cb_lookup.weights.items()
    }
    copied_weights["garment_category"].pop("vest")
    incomplete_lookup = CBClassWeightLookup(
        artifact_version=cb_lookup.artifact_version,
        field_names=cb_lookup.field_names,
        unknown_token=cb_lookup.unknown_token,
        weights=copied_weights,
    )
    with pytest.raises(ValueError, match="garment_category/vest"):
        score_candidate_cb("not JSON", gold, pack, incomplete_lookup)


def test_candidate_ua_composes_clean_no_unknown_semantics(pack, complete_record):
    result = score_candidate_ua(json.dumps(complete_record), complete_record, pack)

    assert result.candidate == "UA"
    assert result.eligible
    assert result.reward == 1.0
    assert result.gate_errors == ()
    assert result.rule_violations == ()
    assert result.rule_adjustment == 0.0
    assert result.unknown_aware_semantics is not None
    assert result.unknown_aware_semantics.semantic_score == 1.0
    assert result.unknown_aware_semantics.unknown_semantics.semantic_score is None


@pytest.mark.parametrize("raw_output", ["not JSON", "{}", "[1,2,3]"])
def test_candidate_ua_assigns_only_the_fixed_floor_to_ineligible_output(
    pack, complete_record, raw_output
):
    result = score_candidate_ua(raw_output, complete_record, pack)

    assert not result.eligible
    assert result.reward == MALFORMED_FLOOR
    assert result.gate_errors
    assert result.rule_violations == ()
    assert result.rule_adjustment is None
    assert result.unknown_aware_semantics is None


def test_candidate_ua_penalizes_commitment_that_candidate_u_excludes(
    pack, complete_record
):
    gold = dict(complete_record)
    gold["occasion"] = pack.unknown_token
    prediction = dict(complete_record)
    prediction["occasion"] = "formal"
    raw_output = json.dumps(prediction)

    candidate_u = score_candidate_u(raw_output, gold, pack)
    candidate_ua = score_candidate_ua(raw_output, gold, pack)
    assert candidate_u.reward == 1.0
    assert candidate_ua.unknown_aware_semantics is not None
    assert candidate_ua.unknown_aware_semantics.known_semantics.semantic_score == 1.0
    assert candidate_ua.unknown_aware_semantics.unknown_semantics.semantic_score == -1.0
    assert candidate_ua.reward == pytest.approx(
        KNOWN_MIX_WEIGHT - UNKNOWN_MIX_WEIGHT
    )


def test_candidate_ua_adds_rule_cost_once_after_combined_semantics(
    pack, complete_record
):
    exact_but_incoherent = dict(complete_record)
    exact_but_incoherent["sleeve_length"] = "sleeveless"
    exact_but_incoherent["sleeve_style"] = "puff"
    exact_but_incoherent["pattern"] = "solid"
    exact_but_incoherent["colour_primary"] = "multicolour"
    exact_but_incoherent["material"] = pack.unknown_token

    result = score_candidate_ua(
        json.dumps(exact_but_incoherent), exact_but_incoherent, pack
    )
    assert result.eligible
    assert set(result.rule_violations) == {
        "sleeveless_has_no_sleeve_style",
        "solid_is_not_multicolour",
    }
    assert result.unknown_aware_semantics is not None
    assert result.unknown_aware_semantics.semantic_score == 1.0
    assert result.rule_adjustment == pytest.approx(-0.10)
    assert result.reward == pytest.approx(0.90)


def test_candidate_ua_validates_gold_before_malformed_model_output(
    pack, complete_record
):
    invalid_gold = dict(complete_record)
    invalid_gold.pop("occasion")
    with pytest.raises(ValueError, match="gold must satisfy"):
        score_candidate_ua("not JSON", invalid_gold, pack)

    with pytest.raises(ValueError, match="at least one gold-known field"):
        score_candidate_ua("not JSON", _all_unknown(pack), pack)


def test_candidate_ua_batch_preserves_order_and_accepts_mixed_gold_shapes(
    pack, complete_record
):
    gold_with_unknown = dict(complete_record)
    gold_with_unknown["occasion"] = pack.unknown_token
    unsupported_commitment = dict(complete_record)
    unsupported_commitment["occasion"] = "formal"

    rewards = candidate_ua_reward(
        [
            json.dumps(gold_with_unknown),
            "{}",
            json.dumps(unsupported_commitment),
        ],
        [
            gold_with_unknown,
            json.dumps(gold_with_unknown),
            gold_with_unknown,
        ],
        pack=pack,
        prompts=["ignored"] * 3,
        sku_id=["a", "b", "c"],
    )
    assert rewards == pytest.approx(
        [1.0, MALFORMED_FLOOR, KNOWN_MIX_WEIGHT - UNKNOWN_MIX_WEIGHT]
    )


def test_candidate_ua_batch_accepts_conversational_shape_and_default_pack(
    complete_record,
):
    rewards = candidate_ua_reward(
        _chat_completions(json.dumps(complete_record), "not JSON"),
        [json.dumps(complete_record), complete_record],
        completion_ids=[[1], [2]],
    )
    assert rewards == [1.0, MALFORMED_FLOOR]


def test_candidate_ua_batch_reuses_alignment_and_gold_first_failures(
    pack, complete_record
):
    encoded = json.dumps(complete_record)
    with pytest.raises(ValueError, match="same length: got 1 and 2"):
        candidate_ua_reward([encoded, encoded], [complete_record], pack=pack)

    invalid_second_gold = dict(complete_record)
    invalid_second_gold.pop("occasion")
    with pytest.raises(ValueError, match="gold must satisfy"):
        candidate_ua_reward(
            ["not JSON", encoded],
            [complete_record, invalid_second_gold],
            pack=pack,
        )


def test_candidate_ua_batch_empty_alignment_is_an_empty_map(pack):
    assert candidate_ua_reward([], [], pack=pack) == []


def test_candidate_cb_factory_loads_lookup_once_per_adapter(
    pack, monkeypatch
):
    calls = []
    original = run2_rewards.load_cb_class_weight_lookup

    def counting_loader(active_pack, artifact_path=None):
        calls.append((active_pack, artifact_path))
        return original(active_pack, artifact_path)

    monkeypatch.setattr(
        run2_rewards,
        "load_cb_class_weight_lookup",
        counting_loader,
    )
    reward = make_candidate_cb_reward(pack=pack)
    assert reward.__name__ == "candidate_cb_reward"
    assert len(calls) == 1

    assert reward([], [], ignored_trainer_column=True) == []
    assert reward([], []) == []
    assert len(calls) == 1

    make_candidate_cb_reward(pack=pack)
    assert len(calls) == 2


def test_candidate_cb_batch_preserves_order_and_accepts_mixed_gold_shapes(
    pack, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    wrong = dict(gold)
    wrong["garment_category"] = "top"
    reward = make_candidate_cb_reward(pack=pack)

    rewards = reward(
        [json.dumps(gold), "{}", json.dumps(wrong)],
        [gold, json.dumps(gold), gold],
        prompts=["ignored"] * 3,
        sku_id=["a", "b", "c"],
        trainer_state=object(),
    )
    assert rewards == pytest.approx([1.0, MALFORMED_FLOOR, 0.3717332035607318])


def test_candidate_cb_batch_accepts_conversational_shape_and_default_pack(
    pack, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.allbirds.com:7257961955408"]
    gold = row.to_verifier_record(pack)
    reward = make_candidate_cb_reward()

    rewards = reward(
        _chat_completions(json.dumps(gold), "not JSON"),
        [json.dumps(gold), gold],
        completion_ids=[[1], [2]],
    )
    assert rewards == [1.0, MALFORMED_FLOOR]


def test_candidate_cb_batch_validates_all_gold_weights_before_mapping(
    pack, active_rows_by_sku, complete_record, monkeypatch
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    valid_gold = row.to_verifier_record(pack)
    scorer_calls = []

    def recording_scorer(*args, **kwargs):
        scorer_calls.append((args, kwargs))
        return SimpleNamespace(reward=123.0)

    reward = make_candidate_cb_reward(pack=pack)
    monkeypatch.setattr(run2_rewards, "score_candidate_cb", recording_scorer)
    with pytest.raises(ValueError, match="sleeve_length/three_quarter"):
        reward(
            ["not JSON", json.dumps(complete_record)],
            [valid_gold, complete_record],
        )
    assert scorer_calls == []


def test_candidate_cb_batch_delegates_all_reward_math_to_single_record_scorer(
    pack, active_rows_by_sku, monkeypatch
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    calls = []

    def sentinel_scorer(raw_output, gold_record, active_pack, class_weights):
        calls.append((raw_output, gold_record, active_pack, class_weights))
        return SimpleNamespace(reward=7.25)

    reward = make_candidate_cb_reward(pack=pack)
    monkeypatch.setattr(run2_rewards, "score_candidate_cb", sentinel_scorer)
    assert reward(["first", "second"], [gold, json.dumps(gold)]) == [7.25, 7.25]
    assert [call[0] for call in calls] == ["first", "second"]
    assert all(call[1] == gold for call in calls)
    assert all(call[2] is pack for call in calls)
    assert calls[0][3] is calls[1][3]


def test_candidate_cb_batch_reuses_shared_alignment_failures(
    pack, active_rows_by_sku
):
    row = active_rows_by_sku["shopify:www.american-giant.com:10297735479477"]
    gold = row.to_verifier_record(pack)
    encoded = json.dumps(gold)
    reward = make_candidate_cb_reward(pack=pack)

    with pytest.raises(ValueError, match="same length: got 1 and 2"):
        reward([encoded, encoded], [gold])
    with pytest.raises(TypeError, match="gold must be a sequence"):
        reward([encoded], encoded)
    with pytest.raises(TypeError, match="completions must be a sequence"):
        reward(encoded, [gold])
