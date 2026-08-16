"""The architecture claim, made executable.

The project has said since W1 that one verifier serves two consumers: an RL
reward during training and a quality gate in production. Until now that was
supported by an import graph. An import graph proves the service *can* call the
verifier; it does not prove the two consumers *agree*.

`test_the_two_consumers_never_disagree` is the load-bearing test in this file
and arguably in W3. It pushes the same records through both paths and requires
identical verdicts. Everything else here is ordinary endpoint coverage.

The records are drawn from real committed artifacts rather than invented, so the
comparison runs over output a model actually produced, including its failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.verifier_service import DEFAULT_PACK, app, get_pack
from training.rewards import format_validity_reward, vocab_rule_compliance_reward
from verifier import load_pack, verify

ROOT = Path(__file__).resolve().parent.parent
SFT_PREDICTIONS = ROOT / "runs" / "sft-combined-2epoch" / "frozen-eval-300-predictions.jsonl"
GRPO_PREDICTIONS = ROOT / "runs" / "grpo-first-300-frozen-eval-300-predictions.jsonl"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / DEFAULT_PACK)


def real_raw_outputs(limit: int = 120) -> list[str]:
    """Literal model outputs from committed runs, including the malformed ones.

    Invented fixtures would agree trivially: the interesting cases are the ones
    a model actually produced, such as the record that opened with prose or the
    one that invented `supraprise` as a neckline.
    """
    outputs: list[str] = []
    for path in (SFT_PREDICTIONS, GRPO_PREDICTIONS):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("raw")
            if isinstance(raw, str):
                outputs.append(raw)
            if len(outputs) >= limit:
                return outputs
    return outputs


# --- the architecture claim ---------------------------------------------------


def test_real_predictions_are_available_to_compare():
    """If this skips, the equivalence test below is vacuous. Fail loudly."""
    outputs = real_raw_outputs()
    assert len(outputs) >= 100, "committed prediction artifacts are missing"


def test_the_two_consumers_never_disagree(client, pack):
    """One verifier, two consumers, identical verdicts on real model output.

    The reward path calls `verify` in-process. The gate path calls it across
    HTTP with its own request parsing, pack loading and response projection. If
    those can diverge on any record, the project has two verifiers and the
    architecture claim is false.
    """
    disagreements = []
    for raw in real_raw_outputs():
        local = verify(raw, pack)
        response = client.post("/verify", json={"raw": raw})
        assert response.status_code == 200
        remote = response.json()

        if (
            remote["schema_valid"] != local.schema_valid
            or remote["vocab_valid"] != local.vocab_valid
            or remote["ok"] != local.ok
            or remote["rule_violations"] != list(local.rule_violations)
            or remote["parsed"] != local.parsed
        ):
            disagreements.append(raw[:120])

    assert not disagreements, f"{len(disagreements)} records judged differently"


def test_the_gate_agrees_with_the_reward_functions_themselves(client):
    """Not just with `verify`, but with the callables GRPO actually scored.

    The reward functions wrap the verifier. Comparing against `verify` alone
    would miss a wrapper that reinterprets the result.
    """
    outputs = real_raw_outputs(limit=60)
    completions = [[{"role": "assistant", "content": raw}] for raw in outputs]
    format_scores = format_validity_reward(completions=completions)
    compliance_scores = vocab_rule_compliance_reward(completions=completions)

    for raw, format_score, compliance_score in zip(
        outputs, format_scores, compliance_scores
    ):
        remote = client.post("/verify", json={"raw": raw}).json()
        assert remote["schema_valid"] is (format_score == 1.0)
        assert remote["ok"] is (compliance_score == 1.0)


# --- the gate's own behaviour -------------------------------------------------


def test_health_identifies_the_vocabulary_being_enforced(client):
    """A gate whose pack is unidentifiable is not auditable: a record rejected
    today and accepted tomorrow would look the same as a vocabulary change."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["fields"] == 15
    assert body["vocabulary_values"] == 156
    # 25 written plus 9 derived from applies_to metadata, reported separately
    # so an auditor can see how much of the rule set was hand-authored.
    assert body["rules_written"] == 25
    assert body["rules_total"] == 34
    assert body["unknown_token"] == "unknown"


def test_a_clean_record_passes(client, pack):
    record = {name: None for name in pack.field_names}
    record.update(
        {
            "garment_category": "shoe",
            "material": "leather",
            "colour_primary": "black",
            "details": ["unknown"],
            "occasion": "unknown",
            "pattern": "unknown",
        }
    )
    body = client.post("/verify", json={"record": record}).json()
    assert body["schema_valid"] is True and body["vocab_valid"] is True


def test_prose_instead_of_json_is_rejected_not_repaired(client):
    """The verifier does no markdown-stripping or brace-hunting, and the gate
    must not add any: leniency would let a malformed record into a catalog."""
    body = client.post("/verify", json={"raw": "Sure! Here are the tags: {...}"}).json()
    assert body["schema_valid"] is False and body["ok"] is False
    assert body["errors"]


def test_an_invented_value_fails_vocabulary_not_schema(client):
    """`supraprise` is a real failure from the frozen evaluation: structurally
    perfect JSON carrying a word that is not in the vocabulary."""
    record = {
        "closure": "unknown", "collar_type": "none", "colour_primary": "black",
        "details": ["gathered"], "fit": "unknown", "garment_category": "dress",
        "garment_length": "maxi", "material": "silk", "neckline": "supraprise",
        "occasion": "formal", "pattern": "floral", "silhouette": "unknown",
        "sleeve_length": None, "sleeve_style": None, "waistline": "normal",
    }
    body = client.post("/verify", json={"record": record}).json()
    assert body["schema_valid"] is True
    assert body["vocab_valid"] is False
    assert body["ok"] is False


def test_a_rule_violation_is_reported_by_id(client):
    """Every value legal, the combination impossible."""
    record = {
        "closure": "unknown", "collar_type": None, "colour_primary": "multicolour",
        "details": ["unknown"], "fit": "unknown", "garment_category": "top",
        "garment_length": "unknown", "material": "cotton", "neckline": "crew",
        "occasion": "casual", "pattern": "solid", "silhouette": "unknown",
        "sleeve_length": "short", "sleeve_style": "unknown", "waistline": None,
    }
    body = client.post("/verify", json={"record": record}).json()
    assert body["rule_violations"], "solid plus multicolour must violate a rule"
    assert body["ok"] is False


def test_supplying_both_or_neither_input_is_rejected(client):
    for payload in ({}, {"raw": "{}", "record": {}}):
        assert client.post("/verify", json=payload).status_code == 422


def test_an_unknown_pack_is_a_404_not_a_crash(client):
    response = client.post("/verify", json={"raw": "{}", "pack": "packs/does_not_exist"})
    assert response.status_code == 404


def test_the_pack_is_loaded_once_per_process():
    """Reloading per request would make the gate's cost depend on traffic."""
    assert get_pack(DEFAULT_PACK) is get_pack(DEFAULT_PACK)


def test_the_service_adds_no_verification_logic():
    """The gate must project the verifier's verdict, not re-derive it.

    A service that recomputed acceptability would be the second verifier this
    whole step exists to avoid, and the equivalence test above could still pass
    while the two drifted apart later.
    """
    source = (ROOT / "service" / "verifier_service.py").read_text(encoding="utf-8")
    for forbidden in ("rules.yaml", "vocab.yaml", "ValidationError", "json.loads("):
        assert forbidden not in source, f"service reimplements verification: {forbidden}"
