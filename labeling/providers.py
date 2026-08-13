"""Batch-labeling backends. Same prompts, same consensus, different vendor.

Two providers, one interface. Not gold-plating: the W1 Step 3 reliability table
measures a *specific labeler's* per-attribute accuracy, so "which labeler" is a
variable the pipeline is built to compare, not a constant to hardcode. Running
both against the same 300 rows and diffing the tables is a real experiment.

What actually differs between them
----------------------------------
1. **Strict-schema rules.** OpenAI's strict mode requires every property to appear
   in `required` — optionality is expressed as a nullable type union, never by
   omission. Pydantic emits optional fields as absent-from-required, so the schema
   needs rewriting or the API rejects it. Anthropic accepts the schema as-is.
2. **Batch mechanics.** Anthropic takes requests inline. OpenAI wants a JSONL file
   uploaded first, then a batch created against the file id.
3. **Prompt caching.** Anthropic uses explicit `cache_control` breakpoints; OpenAI
   caches long prefixes automatically. Either way the shared vocabulary block must
   stay byte-identical, which is why per-sample perturbation lives in the user turn.
4. **Refusals.** Anthropic signals `stop_reason == "refusal"`; OpenAI puts a
   `refusal` string on the message.

Everything else — the system prompt, the perturbation set, consensus, provenance —
is shared and lives in `scripts/prelabel.py`.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

# Constraint keywords neither provider's strict mode accepts.
_STRIP = {
    "maxItems", "minItems", "uniqueItems",
    "maxLength", "minLength", "pattern",
    "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum", "multipleOf",
    "default", "title",
}


# Inside these, the keys are user-chosen field names — never schema keywords.
_NAME_MAPS = ("properties", "$defs", "definitions", "patternProperties")


def strip_unsupported(node: Any) -> Any:
    """Drop constraint keywords, without touching user-named fields.

    The distinction is load-bearing. This pack has a field called `pattern`
    (solid / stripe / floral), and `pattern` is also a JSON Schema keyword. A
    blanket key filter deleted the field from the request schema entirely: the
    model would never have been asked for it, and `pattern` would have come back
    missing on all 4,000 rows with nothing in the output to explain why.

    Any vocabulary is free to contain a field named `default`, `items`, `format`
    or `minimum`. Only positions that hold schema keywords get filtered.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _NAME_MAPS and isinstance(value, dict):
                out[key] = {name: strip_unsupported(sub) for name, sub in value.items()}
            elif key in _STRIP:
                continue
            else:
                out[key] = strip_unsupported(value)
        return out
    if isinstance(node, list):
        return [strip_unsupported(v) for v in node]
    return node


def require_all_properties(node: Any) -> Any:
    """OpenAI strict mode: every property must be in `required`, at every level.

    Optionality is carried by the type union (`anyOf: [..., {"type":"null"}]`),
    which our schema already produces. Omitting a key from `required` is a 400,
    not a permissive default — so a schema that works fine on Anthropic is
    rejected outright here.
    """
    if isinstance(node, dict):
        out = {k: require_all_properties(v) for k, v in node.items()}
        if out.get("type") == "object" or "properties" in out:
            props = out.get("properties") or {}
            out["required"] = list(props)
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(node, list):
        return [require_all_properties(v) for v in node]
    return node


@dataclass
class BatchResult:
    custom_id: str
    text: str | None
    error: str | None = None
    provider_result_id: str | None = None
    provider_request_id: str | None = None
    usage: dict[str, Any] | None = None


class Provider(Protocol):
    name: str
    model: str

    def adapt_schema(self, schema: dict) -> dict: ...
    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict: ...
    def submit(self, items: list[tuple[str, dict]]) -> str: ...
    def wait(self, batch_id: str, poll: int) -> str: ...
    def status(self, batch_id: str) -> dict: ...
    def complete(self, body: dict) -> tuple[str | None, str | None]: ...
    def results(self, batch_id: str) -> Iterator[BatchResult]: ...


# --- Anthropic ---------------------------------------------------------------


class AnthropicProvider:
    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or self.default_model
        self._client = anthropic.Anthropic()

    def adapt_schema(self, schema: dict) -> dict:
        return strip_unsupported(schema)

    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            # Explicit breakpoint: the vocabulary block is identical across every
            # request in the run, so it is written once and read at ~0.1x after.
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": user}],
        }

    def submit(self, items: list[tuple[str, dict]]) -> str:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        batch = self._client.messages.batches.create(
            requests=[
                Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**body))
                for cid, body in items
            ]
        )
        return batch.id

    def wait(self, batch_id: str, poll: int) -> str:
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return "ended"
            print(f"  {batch.processing_status}: {batch.request_counts}", flush=True)
            time.sleep(poll)

    def complete(self, body: dict) -> tuple[str | None, str | None]:
        """One synchronous request. Returns (text, error) — never raises."""
        try:
            msg = self._client.messages.create(**body)
        except Exception as exc:  # noqa: BLE001 — a network blip must not kill the run
            return None, f"{type(exc).__name__}: {str(exc)[:100]}"
        if msg.stop_reason == "refusal":
            return None, "refusal"
        text = next((b.text for b in msg.content if b.type == "text"), None)
        return text, None if text else "no text block"

    def status(self, batch_id: str) -> dict:
        import time as _t

        b = self._client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        created = getattr(b, "created_at", None)
        return {
            "status": b.processing_status,
            "completed": done,
            "failed": c.errored,
            "total": done + c.processing,
            "ready": b.processing_status == "ended",
            "elapsed_s": (_t.time() - created.timestamp()) if created else 0,
        }

    def results(self, batch_id: str) -> Iterator[BatchResult]:
        for result in self._client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                yield BatchResult(result.custom_id, None, result.result.type)
                continue
            msg = result.result.message
            if msg.stop_reason == "refusal":
                yield BatchResult(result.custom_id, None, "refusal")
                continue
            text = next((b.text for b in msg.content if b.type == "text"), None)
            yield BatchResult(result.custom_id, text, None if text else "no text block")


# --- OpenAI ------------------------------------------------------------------


class OpenAIProvider:
    name = "openai"
    # The full ID matters: the `gpt-5.6` alias routes to Sol, not Luna.
    default_model = "gpt-5.6-luna"

    def __init__(self, model: str | None = None):
        import openai

        self.model = model or self.default_model
        self._client = openai.OpenAI()

    def adapt_schema(self, schema: dict) -> dict:
        return require_all_properties(strip_unsupported(schema))

    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        return {
            "model": self.model,
            # Caching is automatic on long shared prefixes — no breakpoint to set,
            # but the prefix must still be byte-identical, hence perturbation in the
            # user turn only.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "product_tags", "strict": True, "schema": schema},
            },
            "max_completion_tokens": max_tokens,
        }

    def submit(self, items: list[tuple[str, dict]]) -> str:
        payload = "\n".join(
            json.dumps(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            for cid, body in items
        )
        upload = self._client.files.create(
            file=("prelabel.jsonl", io.BytesIO(payload.encode())), purpose="batch"
        )
        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def wait(self, batch_id: str, poll: int) -> str:
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self._client.batches.retrieve(batch_id)
            if batch.status in terminal:
                if batch.status != "completed":
                    raise RuntimeError(f"batch {batch_id} ended as {batch.status}")
                return batch.status
            print(f"  {batch.status}: {batch.request_counts}", flush=True)
            time.sleep(poll)

    def complete(self, body: dict) -> tuple[str | None, str | None]:
        """One synchronous request. Returns (text, error) — never raises."""
        try:
            r = self._client.chat.completions.create(**body)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {str(exc)[:100]}"
        m = r.choices[0].message
        if m.refusal:
            return None, f"refusal: {str(m.refusal)[:80]}"
        return m.content, None if m.content else "empty content"

    def status(self, batch_id: str) -> dict:
        import time as _t

        b = self._client.batches.retrieve(batch_id)
        c = b.request_counts
        return {
            "status": b.status,
            "completed": c.completed,
            "failed": c.failed,
            "total": c.total,
            "ready": b.status == "completed",
            "elapsed_s": (_t.time() - b.created_at) if b.created_at else 0,
        }

    def results(self, batch_id: str) -> Iterator[BatchResult]:
        batch = self._client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            raise RuntimeError(f"batch {batch_id} has no output file (status {batch.status})")
        content = self._client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            yield parse_openai_line(json.loads(line))


def parse_openai_line(obj: dict) -> BatchResult:
    """Pull one batch output line apart. Split out so it is testable offline."""
    cid = obj.get("custom_id", "")
    result_id = obj.get("id")
    response = obj.get("response") or {}
    request_id = response.get("request_id")
    body = response.get("body") or {}
    lineage = {
        "provider_result_id": result_id,
        "provider_request_id": request_id,
        "usage": body.get("usage"),
    }
    if obj.get("error"):
        return BatchResult(cid, None, str(obj["error"])[:120], **lineage)
    if response.get("status_code") != 200:
        return BatchResult(cid, None, f"HTTP {response.get('status_code')}", **lineage)
    choices = body.get("choices") or []
    if not choices:
        return BatchResult(cid, None, "no choices", **lineage)
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        return BatchResult(
            cid,
            None,
            f"refusal: {str(message['refusal'])[:80]}",
            **lineage,
        )
    content = message.get("content")
    if not content:
        return BatchResult(cid, None, "empty content", **lineage)
    return BatchResult(cid, content, None, **lineage)


PROVIDERS = {"openai": OpenAIProvider, "anthropic": AnthropicProvider}
DEFAULT_MODELS = {
    "openai": OpenAIProvider.default_model,
    "anthropic": AnthropicProvider.default_model,
}
# input $/1M, cached input $/1M, output $/1M — for the estimate command only.
PRICING = {
    "gpt-5.6-luna": (1.00, 0.10, 6.00),
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-sonnet-5": (3.00, 0.30, 15.00),
}


def get_provider(name: str, model: str | None = None) -> Provider:
    if name not in PROVIDERS:
        raise SystemExit(f"unknown provider {name!r} — one of {sorted(PROVIDERS)}")
    return PROVIDERS[name](model)


# --- Gemini ------------------------------------------------------------------


def inline_defs(schema: dict) -> dict:
    """Resolve every `$ref` against `$defs` and drop the definitions block.

    Gemini's responseSchema is an OpenAPI subset with no `$ref` support, while
    Pydantic emits one `$def` per enum. A third provider, a third schema dialect —
    which is the argument for the provider layer rather than one hardcoded vendor.
    """
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**walk(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def nullable_from_anyof(node):
    """`anyOf: [X, {type: null}]` -> X with `nullable: true`.

    Gemini expresses optionality with a nullable flag, not a union type. Left as a
    union it rejects the schema outright.
    """
    if isinstance(node, dict):
        options = node.get("anyOf")
        if isinstance(options, list):
            non_null = [o for o in options if o.get("type") != "null"]
            had_null = len(non_null) < len(options)
            if len(non_null) == 1:
                merged = {**nullable_from_anyof(non_null[0]),
                          **{k: v for k, v in node.items() if k != "anyOf"}}
                if had_null:
                    merged["nullable"] = True
                return merged
        return {k: nullable_from_anyof(v) for k, v in node.items()}
    if isinstance(node, list):
        return [nullable_from_anyof(v) for v in node]
    return node


class GeminiProvider:
    name = "gemini"
    default_model = "gemini-3.6-flash"

    def __init__(self, model: str | None = None):
        import os

        from google import genai

        self.model = model or self.default_model
        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def adapt_schema(self, schema: dict) -> dict:
        s = nullable_from_anyof(inline_defs(strip_unsupported(schema)))
        s.pop("additionalProperties", None)  # not part of Gemini's subset
        return _drop_additional_properties(s)

    # Gemini 3.x thinks before answering and thinking tokens are charged against
    # max_output_tokens. Measured on this pack: ~850-1050 thinking tokens for ~150
    # tokens of JSON. At 2048 that left too little headroom and 20% of rows came
    # back as truncated JSON — which surfaces as a JSONDecodeError and looks like a
    # parsing bug rather than a budget one. thinking_budget=0 is rejected by this
    # model, so headroom is the only lever.
    THINKING_HEADROOM = 8192

    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "contents": user,
            "config": {
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": max(max_tokens, self.THINKING_HEADROOM),
            },
        }

    def complete(self, body: dict) -> tuple[str | None, str | None]:
        try:
            r = self._client.models.generate_content(**body)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {str(exc)[:100]}"
        # Name truncation for what it is. Returning the partial JSON makes a budget
        # problem present as a parse error, which is how 61 rows were misdiagnosed.
        cands = getattr(r, "candidates", None) or []
        if cands and str(getattr(cands[0], "finish_reason", "")).endswith("MAX_TOKENS"):
            thoughts = getattr(getattr(r, "usage_metadata", None), "thoughts_token_count", "?")
            return None, f"truncated at max_output_tokens (thinking used {thoughts})"
        text = getattr(r, "text", None)
        return text, None if text else "empty response"

    def _unsupported(self, *_a, **_k):
        raise NotImplementedError(
            "GeminiProvider is synchronous-only here — it exists for cross-family "
            "adjudication (a few hundred requests), not for the 20,000-request "
            "pre-labeling batch."
        )

    submit = wait = status = results = _unsupported


def _drop_additional_properties(node):
    if isinstance(node, dict):
        return {k: _drop_additional_properties(v)
                for k, v in node.items() if k != "additionalProperties"}
    if isinstance(node, list):
        return [_drop_additional_properties(v) for v in node]
    return node


PROVIDERS["gemini"] = GeminiProvider
DEFAULT_MODELS["gemini"] = GeminiProvider.default_model
