"""Provider backends — schema adaptation and result parsing, offline.

The strict-schema rules differ between vendors in a way that is invisible until
the API rejects the request, so they are pinned here rather than discovered in a
20,000-request batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.providers import (
    OpenAIProvider,
    parse_openai_line,
    require_all_properties,
    strip_unsupported,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def schema():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1").json_schema()


def test_unsupported_constraints_are_stripped(schema):
    """maxItems on the multi-valued field is rejected by both vendors' strict mode."""
    assert "maxItems" in json.dumps(schema)
    assert "maxItems" not in json.dumps(strip_unsupported(schema))


def test_openai_requires_every_property(schema):
    """Pydantic omits optional fields from `required`; OpenAI strict mode 400s.

    Optionality is carried by the anyOf-with-null union, which we already emit —
    but the key must still be listed.
    """
    assert "required" not in schema
    adapted = require_all_properties(strip_unsupported(schema))
    assert set(adapted["required"]) == set(adapted["properties"])
    assert len(adapted["required"]) == 15


def test_openai_adaptation_keeps_nullability(schema):
    adapted = require_all_properties(strip_unsupported(schema))
    neckline = adapted["properties"]["neckline"]
    assert {"type": "null"} in neckline["anyOf"], "still optional, via the type union"


def test_additional_properties_false_everywhere(schema):
    adapted = require_all_properties(strip_unsupported(schema))

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(adapted)


def test_luna_model_id_is_not_the_bare_alias():
    """`gpt-5.6` routes to Sol. Getting this wrong silently labels with another model."""
    assert OpenAIProvider.default_model == "gpt-5.6-luna"


# --- batch output parsing ----------------------------------------------------


def test_parse_success():
    line = {
        "custom_id": "SKU1::0",
        "response": {"status_code": 200,
                     "body": {"choices": [{"message": {"content": '{"a":1}'}}]}},
    }
    r = parse_openai_line(line)
    assert r.text == '{"a":1}' and r.error is None


def test_parse_refusal_is_not_silently_empty():
    line = {
        "custom_id": "SKU1::0",
        "response": {"status_code": 200,
                     "body": {"choices": [{"message": {"refusal": "I can't help"}}]}},
    }
    r = parse_openai_line(line)
    assert r.text is None and "refusal" in r.error


def test_parse_transport_error():
    r = parse_openai_line({"custom_id": "X::0", "error": {"message": "rate limited"}})
    assert r.text is None and "rate limited" in r.error


def test_parse_non_200():
    r = parse_openai_line({"custom_id": "X::0", "response": {"status_code": 500, "body": {}}})
    assert "HTTP 500" in r.error


def test_parse_empty_content_is_an_error_not_a_label():
    line = {"custom_id": "X::0",
            "response": {"status_code": 200, "body": {"choices": [{"message": {"content": ""}}]}}}
    assert parse_openai_line(line).error == "empty content"


def test_field_named_like_a_schema_keyword_survives(schema):
    """Regression: this pack has a field called `pattern`, which is also a JSON
    Schema keyword. A blanket key filter deleted it from the request schema — the
    model would never be asked for it, and every row would come back missing it."""
    assert "pattern" in schema["properties"]
    assert "pattern" in strip_unsupported(schema)["properties"]
    assert "pattern" in require_all_properties(strip_unsupported(schema))["required"]


def test_other_keyword_named_fields_would_survive_too():
    fake = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "default": {"type": "string"},
            "items": {"type": "string"},
            "minimum": {"type": "string"},
            "title": {"type": "string"},
        },
    }
    kept = strip_unsupported(fake)["properties"]
    assert set(kept) == {"default", "items", "minimum", "title"}
