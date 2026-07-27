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


class Provider(Protocol):
    name: str
    model: str

    def adapt_schema(self, schema: dict) -> dict: ...
    def build_body(self, system: str, user: str, schema: dict, max_tokens: int) -> dict: ...
    def submit(self, items: list[tuple[str, dict]]) -> str: ...
    def wait(self, batch_id: str, poll: int) -> str: ...
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
    if obj.get("error"):
        return BatchResult(cid, None, str(obj["error"])[:120])
    response = obj.get("response") or {}
    if response.get("status_code") != 200:
        return BatchResult(cid, None, f"HTTP {response.get('status_code')}")
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return BatchResult(cid, None, "no choices")
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        return BatchResult(cid, None, f"refusal: {str(message['refusal'])[:80]}")
    content = message.get("content")
    if not content:
        return BatchResult(cid, None, "empty content")
    return BatchResult(cid, content, None)


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
