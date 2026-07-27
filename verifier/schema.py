"""Build Pydantic models from a pack's vocab.yaml, so the two cannot drift.

Nothing here knows what a garment is. It reads `fields:` out of a vocab file and
produces two models:

  loose  — every value typed as `str` (or `list[str]`). Validates STRUCTURE only:
           is it JSON, are the keys real field names, is a single-valued field a
           string rather than a list, is a multi-valued field within max_values.
  strict — every value typed as an Enum of that field's canonical values.
           Validates VOCABULARY.

Two models rather than one on purpose. W2 trains against three separate reward
components, two of which are "format validity" and "vocab/rule compliance". If a
single enum-typed model did both jobs, those two rewards would collapse into one
number and you would lose the ability to see the model learn them at different
rates — which is exactly the thing worth watching.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

_IDENT = re.compile(r"[^0-9a-zA-Z_]")


@dataclass(frozen=True)
class FieldSpec:
    """One field from vocab.yaml, flattened to what the verifier needs."""

    name: str
    kind: str  # "single" | "multi"
    values: tuple[str, ...]  # canonical values, excluding the abstain token
    aliases: dict[str, str] = dc_field(default_factory=dict)  # casefolded -> canonical
    applies_to: frozenset[str] | None = None  # None == always applicable
    max_values: int | None = None
    required: bool = False

    def accepts(self, category: object) -> bool:
        """Is this field applicable given the pack's category field value?

        `category` arrives straight from model output and may be anything at all —
        a list, a dict, a number. A malformed category has already failed schema
        validation, so treat the field as applicable rather than stacking a second
        violation on one error (and rather than raising: the verifier grades
        garbage for a living and must never crash on it).
        """
        if self.applies_to is None or category is None:
            return True
        if not isinstance(category, str):
            return True
        return category in self.applies_to


def _member_name(value: str) -> str:
    """Enum member names must be identifiers and must not collide with keywords."""
    name = _IDENT.sub("_", value)
    if not name or name[0].isdigit():
        name = f"v_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    return name


def parse_fields(vocab: dict[str, Any]) -> dict[str, FieldSpec]:
    """Flatten vocab.yaml's `fields:` block into FieldSpecs.

    Only `name` is required per value. Provenance (`from:`) is deliberately NOT
    required here — that is enforced separately by tools/check_vocab_provenance.py.
    Loading and provenance are different concerns; a pack missing provenance should
    fail the audit, not fail to load.
    """
    fields = vocab.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("vocab.yaml has no `fields:` block")

    specs: dict[str, FieldSpec] = {}
    for fname, fdef in fields.items():
        raw_values = fdef.get("values") or []
        values: list[str] = []
        aliases: dict[str, str] = {}
        for v in raw_values:
            vname = v.get("name") if isinstance(v, dict) else v
            if not vname:
                raise ValueError(f"{fname}: a value has no `name`")
            values.append(vname)
            if isinstance(v, dict):
                for alias in v.get("aliases") or []:
                    aliases[alias.strip().casefold()] = vname
            aliases.setdefault(vname.casefold(), vname)

        if not values:
            raise ValueError(f"{fname}: no values")

        applies = fdef.get("applies_to")
        applies_to = None
        if isinstance(applies, dict) and applies.get("categories"):
            applies_to = frozenset(applies["categories"])

        specs[fname] = FieldSpec(
            name=fname,
            kind=fdef.get("kind", "single"),
            values=tuple(values),
            aliases=aliases,
            applies_to=applies_to,
            max_values=fdef.get("max_values"),
            required=bool(fdef.get("required", False)),
        )
    return specs


def build_models(
    specs: dict[str, FieldSpec], unknown_token: str
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Return (loose_model, strict_model) for a set of FieldSpecs."""
    loose: dict[str, Any] = {}
    strict: dict[str, Any] = {}

    for name, spec in specs.items():
        members = {_member_name(v): v for v in spec.values}
        # The abstain token is a real, emittable value in every field. See
        # verifier.__init__ for why abstention is not modelled as null.
        members[_member_name(unknown_token)] = unknown_token
        enum_cls = Enum(f"{name}__v", members, type=str)

        if spec.kind == "multi":
            constraint = (
                {"max_length": spec.max_values} if spec.max_values else {}
            )
            loose[name] = (
                Annotated[list[str], Field(**constraint)] | None,
                None,
            )
            strict[name] = (
                Annotated[list[enum_cls], Field(**constraint)] | None,
                None,
            )
        else:
            loose[name] = (str | None, None)
            strict[name] = (enum_cls | None, None)

    config = ConfigDict(extra="forbid")  # hallucinated keys must fail, not be ignored
    loose_model = create_model("LooseRecord", __config__=config, **loose)
    strict_model = create_model("StrictRecord", __config__=config, **strict)
    return loose_model, strict_model


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for the strict model — feeds vLLM `structured_outputs` in W3.

    Note W3 only. Training in W2 runs UNCONSTRAINED on purpose: constrained
    decoding would make schema-valid JSON free and delete the reward signal.
    """
    return model.model_json_schema()
