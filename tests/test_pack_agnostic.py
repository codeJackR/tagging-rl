"""The pack-agnosticism claim, executable.

Every test here is parametrized over BOTH packs and runs the identical assertions.
Nothing in this file names a garment, a sleeve, or a spacecraft — the records are
built by reading each pack's own vocabulary. A fashion assumption leaking into
verifier/ shows up as demo_pack failing a test vastraa_taste_v1 passes.

W1 Step 2's done-when: "a second, fake pack loads without touching library code."
That is this file. demo_pack contains two YAML files and zero Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier import load_pack, verify, verify_record
from verifier.rules import RuleError, validate_rules

ROOT = Path(__file__).resolve().parent.parent
PACKS = [ROOT / "packs" / "vastraa_taste_v1", ROOT / "packs" / "demo_pack"]


@pytest.fixture(params=PACKS, ids=lambda p: p.name)
def pack(request):
    return load_pack(request.param)


# --- helpers that read the pack rather than assuming a domain ----------------


def first_value(pack, fname):
    return pack.specs[fname].values[0]


def minimal_record(pack):
    """One valid value per field, shaped correctly for single vs multi."""
    rec = {}
    for fname, spec in pack.specs.items():
        v = spec.values[0]
        rec[fname] = [v] if spec.kind == "multi" else v
    return rec


def all_unknown_record(pack):
    tok = pack.unknown_token
    return {
        f: ([tok] if s.kind == "multi" else tok) for f, s in pack.specs.items()
    }


# --- loading -----------------------------------------------------------------


def test_pack_loads(pack):
    assert pack.specs, "pack has no fields"
    assert pack.name
    assert pack.category_field in pack.specs


def test_pack_dir_contains_no_python(pack):
    """A pack is data. The moment it ships code, it is no longer pack-agnostic."""
    py = list(pack.path.glob("**/*.py"))
    assert not py, f"{pack.name} ships Python: {[p.name for p in py]}"


def test_rules_reference_only_real_vocabulary(pack):
    validate_rules(pack.rules, pack.specs)  # raises RuleError if not


def test_every_rule_has_a_unique_id(pack):
    ids = [r["id"] for r in pack.rules]
    assert len(ids) == len(set(ids))


# --- schema validation -------------------------------------------------------


def test_minimal_record_is_schema_and_vocab_valid(pack):
    result = verify(json.dumps(minimal_record(pack)), pack)
    assert result.schema_valid, result.errors
    assert result.vocab_valid, result.errors


def test_unknown_key_fails_schema(pack):
    rec = minimal_record(pack)
    rec["definitely_not_a_field_in_any_pack"] = "x"
    result = verify(json.dumps(rec), pack)
    assert not result.schema_valid
    assert not result.ok


def test_out_of_vocab_value_fails_vocab_but_not_schema(pack):
    """The two signals must move independently — W2 rewards them separately."""
    fname = next(f for f, s in pack.specs.items() if s.kind == "single")
    rec = minimal_record(pack)
    rec[fname] = "an_invented_value_no_pack_contains"
    result = verify(json.dumps(rec), pack)
    assert result.schema_valid, "a string in a string slot is structurally fine"
    assert not result.vocab_valid


def test_wrong_arity_fails_schema(pack):
    """A list where a scalar belongs is a structural error, not a vocab error."""
    fname = next(f for f, s in pack.specs.items() if s.kind == "single")
    rec = minimal_record(pack)
    rec[fname] = [rec[fname]]
    result = verify(json.dumps(rec), pack)
    assert not result.schema_valid


@pytest.mark.parametrize(
    "junk",
    [
        {},
        {"": ""},
        {"a": {"nested": "object"}},
        {"a": 1},
        {"a": [[]]},
        {"a": None},
        {"a": True},
        {"a": [{"b": 1}]},
    ],
)
def test_malformed_records_never_raise(pack, junk):
    """The verifier grades adversarial output for a living. It must never crash.

    Regression: a list in the category slot used to raise TypeError out of the
    applicability check, taking down the whole reward computation for that rollout.
    """
    cat = pack.category_field
    for candidate in ({**junk}, {**junk, cat: junk.get("a", [])}):
        result = verify_record(candidate, pack)
        assert isinstance(result.rule_violations, list)


def test_non_json_fails_cleanly(pack):
    result = verify("Sure! Here are the tags you asked for:", pack)
    assert not result.schema_valid
    assert not result.vocab_valid
    assert result.parsed is None
    assert result.errors


def test_json_array_is_not_a_record(pack):
    result = verify("[1, 2, 3]", pack)
    assert not result.schema_valid
    assert result.parsed is None


def test_fenced_json_is_not_silently_repaired(pack):
    """Leniency here would delete W2's format-validity reward before measuring it."""
    fenced = "```json\n" + json.dumps(minimal_record(pack)) + "\n```"
    assert not verify(fenced, pack).schema_valid


# --- abstention --------------------------------------------------------------


def test_abstention_is_valid_and_counted(pack):
    result = verify(json.dumps(all_unknown_record(pack)), pack)
    assert result.schema_valid and result.vocab_valid, result.errors
    assert set(result.abstentions) == set(pack.specs)


def test_abstention_differs_from_null(pack):
    """null = not applicable; unknown = the model declined. W4 prices these apart."""
    tok = pack.unknown_token
    fname = next(f for f, s in pack.specs.items() if s.kind == "single")
    abstained = verify_record({fname: tok}, pack)
    omitted = verify_record({fname: None}, pack)
    assert abstained.abstentions == [fname]
    assert omitted.abstentions == []


# --- rules -------------------------------------------------------------------


def test_derived_applies_to_rule_fires(pack):
    """Asserting a value for a field that cannot apply is a violation, in any pack."""
    cat_field = pack.category_field
    target = next(
        (f for f, s in pack.specs.items() if s.applies_to and f != cat_field), None
    )
    assert target, f"{pack.name} has no applies_to field to exercise"
    spec = pack.specs[target]
    bad_cat = next(
        c for c in pack.specs[cat_field].values if c not in spec.applies_to
    )
    rec = {cat_field: bad_cat, target: spec.values[0]}
    result = verify_record(rec, pack)
    assert f"auto:applies_to:{target}" in result.rule_violations


def test_abstaining_on_inapplicable_field_is_not_a_violation(pack):
    cat_field = pack.category_field
    target = next(f for f, s in pack.specs.items() if s.applies_to and f != cat_field)
    spec = pack.specs[target]
    bad_cat = next(c for c in pack.specs[cat_field].values if c not in spec.applies_to)
    rec = {cat_field: bad_cat, target: pack.unknown_token}
    assert f"auto:applies_to:{target}" not in verify_record(rec, pack).rule_violations


def test_rule_inventory_counts_written_and_derived(pack):
    inv = pack.rule_inventory()
    assert inv["total"] == len(inv["written"]) + len(inv["derived"])
    assert inv["written"] == [r["id"] for r in pack.rules]


def test_bad_rule_raises_instead_of_never_firing(pack):
    """A rule naming a field that does not exist must fail loudly at load time."""
    bogus = [{"id": "x", "type": "requires", "if": {"nope": "nope"}, "then_present": []}]
    with pytest.raises(RuleError):
        validate_rules(bogus, pack.specs)


# --- normalisation -----------------------------------------------------------


def test_aliases_normalise_only_when_asked(pack):
    fname, spec = next(
        (f, s)
        for f, s in pack.specs.items()
        if s.kind == "single"
        and any(a != v.casefold() for a, v in s.aliases.items())
    )
    alias = next(a for a, v in spec.aliases.items() if a != v.casefold())
    canonical = spec.aliases[alias]

    strict = verify_record({fname: alias}, pack)
    assert not strict.vocab_valid, "grading must not silently accept aliases"

    lenient = verify_record({fname: alias}, pack, normalize=True)
    assert lenient.vocab_valid
    assert lenient.parsed[fname] == canonical
    assert lenient.normalized


# --- W3 handoff --------------------------------------------------------------


def test_json_schema_covers_every_field(pack):
    schema = pack.json_schema()
    assert set(schema["properties"]) == set(pack.specs)
