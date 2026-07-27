"""The single entry point. One import path, two callers.

    W2 / W4-5   reward function      verify(rollout_output, pack).rule_violations
    W3          production QA gate   verify(frontier_output, pack).schema_valid

Nothing in this package knows what a garment is. Everything domain-specific lives
in packs/<name>/{vocab.yaml,rules.yaml}, which are pure data — no Python. Adding a
pack is adding two YAML files.

    from verifier import load_pack, verify

    pack = load_pack("packs/vastraa_taste_v1")
    result = verify(model_output, pack)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from . import rules as rules_mod
from .schema import FieldSpec, build_models, json_schema, parse_fields

__all__ = [
    "Pack",
    "VerifierResult",
    "load_pack",
    "verify",
    "verify_record",
    "FieldSpec",
]

DEFAULT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pack:
    """A loaded schema pack: vocabulary + rules + the models generated from them."""

    name: str
    path: Path
    vocab: dict[str, Any]
    rules: list[dict[str, Any]]
    specs: dict[str, FieldSpec]
    loose_model: type[BaseModel]
    strict_model: type[BaseModel]
    unknown_token: str
    category_field: str | None

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.specs)

    def json_schema(self) -> dict[str, Any]:
        """For vLLM `structured_outputs` in W3. Not used during W2 training."""
        return json_schema(self.strict_model)

    def rule_inventory(self) -> dict[str, Any]:
        return rules_mod.describe(self.rules, self.specs, self.category_field)


@dataclass
class VerifierResult:
    schema_valid: bool  # parsed as JSON + structurally valid (keys, shapes, arity)
    vocab_valid: bool  # every value is in the controlled vocabulary
    rule_violations: list[str]  # violated rule ids from rules.yaml (+ derived)
    parsed: dict | None

    # --- additions to the sketched interface, each earning its place ---------
    # errors:      why schema_valid/vocab_valid failed. Without it a failed run is
    #              undebuggable, and W2's reward shaping needs to distinguish
    #              "emitted prose instead of JSON" from "one bad enum value".
    # abstentions: which fields chose the abstain token. W4's reward is
    #              +1 correct / 0 abstain / -lambda wrong, so abstention must be
    #              countable; W3's escalation queue keys off the same list.
    # normalized:  near-miss telemetry when normalize=True — "3/4 sleeve" ->
    #              "three_quarter". How often the model lands one alias away from
    #              correct is a reward-shaping signal in its own right.
    errors: list[str] = dc_field(default_factory=list)
    abstentions: list[str] = dc_field(default_factory=list)
    normalized: dict[str, str] = dc_field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.schema_valid and self.vocab_valid and not self.rule_violations


def load_pack(path: str | Path) -> Pack:
    """Load a pack directory. Raises on a rules file that contradicts its vocab."""
    path = Path(path)
    vocab_path = path / "vocab.yaml"
    if not vocab_path.exists():
        raise FileNotFoundError(f"no vocab.yaml in {path}")

    vocab = yaml.safe_load(vocab_path.read_text())
    specs = parse_fields(vocab)

    conventions = vocab.get("conventions") or {}
    unknown_token = conventions.get("unknown_token", DEFAULT_UNKNOWN)
    category_field = conventions.get("category_field")
    if category_field and category_field not in specs:
        raise ValueError(
            f"{path.name}: conventions.category_field {category_field!r} is not a field"
        )

    rules_path = path / "rules.yaml"
    rules: list[dict[str, Any]] = []
    if rules_path.exists():
        rules = yaml.safe_load(rules_path.read_text()) or []
        if not isinstance(rules, list):
            raise ValueError(f"{path.name}: rules.yaml must be a list of rules")
    rules_mod.validate_rules(rules, specs)

    loose_model, strict_model = build_models(specs, unknown_token)

    return Pack(
        name=vocab.get("pack", path.name),
        path=path,
        vocab=vocab,
        rules=rules,
        specs=specs,
        loose_model=loose_model,
        strict_model=strict_model,
        unknown_token=unknown_token,
        category_field=category_field,
    )


def _flatten_errors(exc: ValidationError, prefix: str) -> list[str]:
    out = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        out.append(f"{prefix}: {loc}: {e['msg']}")
    return out


def _normalize(parsed: dict, pack: Pack) -> tuple[dict, dict[str, str]]:
    """Map aliases to canonical values. Ingest path only — never used for grading."""
    fixed: dict[str, Any] = {}
    changes: dict[str, str] = {}
    for fname, value in parsed.items():
        spec = pack.specs.get(fname)
        if spec is None or value is None:
            fixed[fname] = value
            continue
        if isinstance(value, list):
            new_list = []
            for v in value:
                canon = spec.aliases.get(str(v).strip().casefold(), v)
                if canon != v:
                    changes[f"{fname}[{v}]"] = canon
                new_list.append(canon)
            fixed[fname] = new_list
        else:
            canon = spec.aliases.get(str(value).strip().casefold(), value)
            if canon != value:
                changes[f"{fname}[{value}]"] = canon
            fixed[fname] = canon
    return fixed, changes


def verify_record(record: dict, pack: Pack, *, normalize: bool = False) -> VerifierResult:
    """Verify an already-parsed dict (eval harness path)."""
    errors: list[str] = []
    normalized: dict[str, str] = {}

    try:
        pack.loose_model(**record)
        schema_valid = True
    except ValidationError as exc:
        schema_valid = False
        errors += _flatten_errors(exc, "schema")

    if normalize:
        record, normalized = _normalize(record, pack)

    try:
        pack.strict_model(**record)
        vocab_valid = True
    except ValidationError as exc:
        vocab_valid = False
        errors += _flatten_errors(exc, "vocab")

    violations = rules_mod.evaluate(
        record,
        specs=pack.specs,
        rules=pack.rules,
        unknown_token=pack.unknown_token,
        category_field=pack.category_field,
    )

    abstentions = [
        f
        for f, v in record.items()
        if f in pack.specs
        and (
            v == pack.unknown_token
            or (isinstance(v, list) and pack.unknown_token in v)
        )
    ]

    return VerifierResult(
        schema_valid=schema_valid,
        vocab_valid=vocab_valid,
        rule_violations=violations,
        parsed=record,
        errors=errors,
        abstentions=sorted(abstentions),
        normalized=normalized,
    )


def verify(raw_output: str, pack: Pack, *, normalize: bool = False) -> VerifierResult:
    """Verify a raw model output string.

    Deliberately strict: no markdown-fence stripping, no trailing-comma repair, no
    "find the first {...}". In W2 the model trains UNCONSTRAINED so you can watch it
    learn to emit clean JSON — silently repairing its output here would delete the
    format-validity reward signal before it was ever measured. Leniency, if wanted,
    belongs in the caller and should be logged when it fires.
    """
    try:
        obj = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        return VerifierResult(
            schema_valid=False,
            vocab_valid=False,
            rule_violations=[],
            parsed=None,
            errors=[f"schema: not valid JSON: {exc}"],
        )

    if not isinstance(obj, dict):
        return VerifierResult(
            schema_valid=False,
            vocab_valid=False,
            rule_violations=[],
            parsed=None,
            errors=[f"schema: top level must be an object, got {type(obj).__name__}"],
        )

    return verify_record(obj, pack, normalize=normalize)
