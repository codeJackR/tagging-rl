"""Declarative cross-field rules. Three types, plus one derived automatically.

    requires           if A=X, then fields B,C must carry a value
    excludes           if A=X, then B may not be Y  (Y may be "*" = any value)
    conditional_vocab  if A=X, then B is restricted to a subset of its vocabulary

Plus `auto:applies_to:<field>`, derived from vocab.yaml's `applies_to:` blocks
rather than written by hand. Same reasoning as generating the Pydantic model from
the vocab: a hand-copied applicability list is a list that drifts. Deriving it
means the vocabulary is the single source of truth for what applies where.

Rules are validated against the vocabulary at load time. A rule naming a field or
value that does not exist raises instead of silently never firing — a rule that
never fires looks identical to a rule that always passes.
"""

from __future__ import annotations

from typing import Any

from .schema import FieldSpec

RULE_TYPES = {"requires", "excludes", "conditional_vocab"}
ANY = "*"


class RuleError(ValueError):
    """A rules.yaml that does not match its vocab.yaml."""


def _as_list(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def validate_rules(rules: list[dict], specs: dict[str, FieldSpec]) -> None:
    """Fail loudly on a rules file that references vocabulary that isn't there."""
    seen_ids: set[str] = set()

    def check_field(fname: str, where: str) -> FieldSpec:
        if fname not in specs:
            raise RuleError(f"{where}: unknown field {fname!r}")
        return specs[fname]

    def check_values(fname: str, values: list[Any], where: str) -> None:
        spec = check_field(fname, where)
        for v in values:
            if v == ANY:
                continue
            if v not in spec.values:
                raise RuleError(
                    f"{where}: {v!r} is not a value of field {fname!r} "
                    f"(have: {', '.join(spec.values)})"
                )

    for i, rule in enumerate(rules):
        rid = rule.get("id")
        where = f"rules[{i}]" + (f" ({rid})" if rid else "")
        if not rid:
            raise RuleError(f"{where}: every rule needs an `id:` — it is what gets reported")
        if rid in seen_ids:
            raise RuleError(f"{where}: duplicate rule id {rid!r}")
        seen_ids.add(rid)

        rtype = rule.get("type")
        if rtype not in RULE_TYPES:
            raise RuleError(f"{where}: type {rtype!r} not in {sorted(RULE_TYPES)}")

        cond = rule.get("if")
        if not isinstance(cond, dict) or not cond:
            raise RuleError(f"{where}: needs a non-empty `if:` condition")
        for fname, expected in cond.items():
            check_values(fname, _as_list(expected), f"{where}.if")

        if rtype == "requires":
            targets = rule.get("then_present")
            if not targets:
                raise RuleError(f"{where}: `requires` needs `then_present:`")
            for fname in _as_list(targets):
                check_field(fname, f"{where}.then_present")
        elif rtype == "excludes":
            forbid = rule.get("forbid")
            if not isinstance(forbid, dict) or not forbid:
                raise RuleError(f"{where}: `excludes` needs a `forbid:` mapping")
            for fname, banned in forbid.items():
                check_values(fname, _as_list(banned), f"{where}.forbid")
        else:  # conditional_vocab
            restrict = rule.get("restrict")
            if not isinstance(restrict, dict) or not restrict:
                raise RuleError(f"{where}: `conditional_vocab` needs a `restrict:` mapping")
            for fname, allowed in restrict.items():
                if not allowed:
                    raise RuleError(f"{where}.restrict.{fname}: empty allow-list")
                check_values(fname, _as_list(allowed), f"{where}.restrict")


def _present(value: Any, unknown_token: str, allow_abstain: bool) -> bool:
    """Does this slot carry a real claim?"""
    if value is None:
        return False
    if isinstance(value, list):
        vals = [v for v in value if v != unknown_token or allow_abstain]
        return bool(vals)
    if value == unknown_token:
        return allow_abstain
    return True


def _matches(parsed: dict, cond: dict) -> bool:
    for fname, expected in cond.items():
        actual = parsed.get(fname)
        if actual is None:
            return False
        options = _as_list(expected)
        if isinstance(actual, list):
            if not any(a in options for a in actual):
                return False
        elif actual not in options:
            return False
    return True


def evaluate(
    parsed: dict,
    *,
    specs: dict[str, FieldSpec],
    rules: list[dict],
    unknown_token: str,
    category_field: str | None,
) -> list[str]:
    """Return the ids of every violated rule. Empty list == clean."""
    violations: list[str] = []

    # --- hand-written rules -------------------------------------------------
    for rule in rules:
        if not _matches(parsed, rule["if"]):
            continue
        rid, rtype = rule["id"], rule["type"]
        allow_abstain = bool(rule.get("allow_abstain", True))

        if rtype == "requires":
            for fname in _as_list(rule["then_present"]):
                if not _present(parsed.get(fname), unknown_token, allow_abstain):
                    violations.append(rid)
                    break

        elif rtype == "excludes":
            for fname, banned in rule["forbid"].items():
                actual = parsed.get(fname)
                if actual is None:
                    continue
                banned_list = _as_list(banned)
                actuals = actual if isinstance(actual, list) else [actual]
                actuals = [a for a in actuals if a != unknown_token]
                if any(ANY in banned_list or a in banned_list for a in actuals):
                    violations.append(rid)
                    break

        else:  # conditional_vocab
            for fname, allowed in rule["restrict"].items():
                actual = parsed.get(fname)
                if actual is None:
                    continue
                allowed_list = _as_list(allowed) + [unknown_token]
                actuals = actual if isinstance(actual, list) else [actual]
                if any(a not in allowed_list for a in actuals):
                    violations.append(rid)
                    break

    # --- derived applicability rules ----------------------------------------
    # Abstaining on an inapplicable field is not a violation: `unknown` makes no
    # claim. Asserting a concrete value for a field that cannot apply is.
    if category_field:
        category = parsed.get(category_field)
        if category is not None:
            for fname, spec in specs.items():
                if fname == category_field or spec.accepts(category):
                    continue
                if _present(parsed.get(fname), unknown_token, allow_abstain=False):
                    violations.append(f"auto:applies_to:{fname}")

    return violations


def describe(rules: list[dict], specs: dict[str, FieldSpec], category_field: str | None) -> dict:
    """Rule inventory — hand-written vs derived. Used by the CLI and tests."""
    derived = (
        [f"auto:applies_to:{n}" for n, s in specs.items() if s.applies_to and n != category_field]
        if category_field
        else []
    )
    return {
        "written": [r["id"] for r in rules],
        "derived": derived,
        "total": len(rules) + len(derived),
    }
