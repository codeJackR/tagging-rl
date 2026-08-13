from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.providers import BatchResult
from training.run2_confirmation_labeling_workflow import (
    collect_completed_batch,
    initialize_workflow_state,
    submit_pending_batch,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


class FakeProvider:
    name = "openai"
    model = "gpt-5.6-luna"

    def __init__(self, record, *, submit_error=None, ready=True):
        self.record = record
        self.submit_error = submit_error
        self.ready = ready
        self.submitted = []
        self.intent_to_check = None

    def adapt_schema(self, schema):
        return schema

    def build_body(self, system, user, schema, max_tokens):
        return {
            "model": self.model,
            "system": system,
            "user": user,
            "schema": schema,
            "max_tokens": max_tokens,
        }

    def submit(self, items):
        assert self.intent_to_check is None or self.intent_to_check.exists()
        self.submitted = list(items)
        if self.submit_error:
            raise self.submit_error
        return "batch-001"

    def status(self, batch_id):
        assert batch_id == "batch-001"
        return {
            "status": "completed" if self.ready else "in_progress",
            "completed": len(self.submitted) if self.ready else 0,
            "failed": 0,
            "total": len(self.submitted),
            "ready": self.ready,
        }

    def results(self, batch_id):
        assert batch_id == "batch-001"
        text = json.dumps(self.record)
        for index, (custom_id, _body) in enumerate(self.submitted):
            yield BatchResult(
                custom_id=custom_id,
                text=text,
                provider_result_id=f"result-{index}",
                provider_request_id=f"request-{index}",
                usage={"prompt_tokens": 10, "completion_tokens": 3},
            )


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def _valid_record(pack):
    return {
        name: [pack.unknown_token] if spec.kind == "multi" else pack.unknown_token
        for name, spec in pack.specs.items()
    }


def _selected(tmp_path: Path):
    selected = tmp_path / "selected.jsonl"
    row = {
        "sku_id": "shopify:store.example:1",
        "source": "shopify:store.example",
        "input": {
            "title": "Cotton Dress",
            "description": "Cotton midi dress.",
            "raw_tags": ["dress", "cotton"],
            "brand": "Example",
            "category": "Dress",
            "image_url": None,
        },
    }
    selected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    import hashlib

    raw = selected.read_bytes()
    prelabel = tmp_path / "manifest.json"
    prelabel.write_text(
        json.dumps(
            {
                "status": "confirmation_membership_frozen_before_labeling",
                "files": {
                    "selected_snapshot": {
                        "path": "selected.jsonl",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return selected, prelabel


def _initial(tmp_path, pack, provider):
    selected, prelabel = _selected(tmp_path)
    state = tmp_path / "state-0000.json"
    initialize_workflow_state(
        selected_path=selected,
        prelabel_manifest_path=prelabel,
        output_state_path=state,
        pack=pack,
        provider=provider,
        expected_products=1,
    )
    return selected, state


def test_init_submit_collect_preserves_external_lineage(tmp_path, pack):
    provider = FakeProvider(_valid_record(pack))
    selected, state0 = _initial(tmp_path, pack, provider)
    intent = tmp_path / "intent.json"
    receipt = tmp_path / "receipt.json"
    provider.intent_to_check = intent

    submit_pending_batch(
        state_path=state0,
        selected_path=selected,
        intent_path=intent,
        receipt_path=receipt,
        pack=pack,
        provider=provider,
        submitted_at_utc="2026-08-13T00:00:00Z",
    )
    state1 = tmp_path / "state-0001.json"
    updated = collect_completed_batch(
        state_path=state0,
        selected_path=selected,
        intent_path=intent,
        receipt_path=receipt,
        output_state_path=state1,
        pack=pack,
        provider=provider,
        collected_at_utc="2026-08-13T01:00:00Z",
    )

    assert updated["status"] == "all_slots_usable_ready_to_finalize"
    assert updated["workflow"]["generation"] == 1
    assert updated["counts"]["usable_slots"] == 5
    attempts = [attempt for slot in updated["slots"] for attempt in slot["attempts"]]
    assert {attempt["provider_result_id"] for attempt in attempts} == {
        f"result-{index}" for index in range(5)
    }
    assert updated["batches"][0]["submission_receipt"]["sha256"]


def test_submit_failure_leaves_intent_and_no_receipt(tmp_path, pack):
    provider = FakeProvider(_valid_record(pack), submit_error=RuntimeError("network uncertain"))
    selected, state0 = _initial(tmp_path, pack, provider)
    intent = tmp_path / "intent.json"
    receipt = tmp_path / "receipt.json"
    provider.intent_to_check = intent

    with pytest.raises(RuntimeError, match="network uncertain"):
        submit_pending_batch(
            state_path=state0,
            selected_path=selected,
            intent_path=intent,
            receipt_path=receipt,
            pack=pack,
            provider=provider,
            submitted_at_utc="2026-08-13T00:00:00Z",
        )

    assert intent.exists()
    assert json.loads(intent.read_text())["safety"]["do_not_resubmit_if_receipt_missing"] is True
    assert not receipt.exists()


def test_collect_refuses_a_batch_that_is_not_ready(tmp_path, pack):
    provider = FakeProvider(_valid_record(pack), ready=False)
    selected, state0 = _initial(tmp_path, pack, provider)
    intent = tmp_path / "intent.json"
    receipt = tmp_path / "receipt.json"
    submit_pending_batch(
        state_path=state0,
        selected_path=selected,
        intent_path=intent,
        receipt_path=receipt,
        pack=pack,
        provider=provider,
        submitted_at_utc="2026-08-13T00:00:00Z",
    )

    with pytest.raises(RuntimeError, match="not ready"):
        collect_completed_batch(
            state_path=state0,
            selected_path=selected,
            intent_path=intent,
            receipt_path=receipt,
            output_state_path=tmp_path / "state-0001.json",
            pack=pack,
            provider=provider,
            collected_at_utc="2026-08-13T01:00:00Z",
        )


def test_collect_refuses_receipt_state_mismatch(tmp_path, pack):
    provider = FakeProvider(_valid_record(pack))
    selected, state0 = _initial(tmp_path, pack, provider)
    intent = tmp_path / "intent.json"
    receipt = tmp_path / "receipt.json"
    submit_pending_batch(
        state_path=state0,
        selected_path=selected,
        intent_path=intent,
        receipt_path=receipt,
        pack=pack,
        provider=provider,
        submitted_at_utc="2026-08-13T00:00:00Z",
    )
    state0.write_text(state0.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="supplied state"):
        collect_completed_batch(
            state_path=state0,
            selected_path=selected,
            intent_path=intent,
            receipt_path=receipt,
            output_state_path=tmp_path / "state-0001.json",
            pack=pack,
            provider=provider,
            collected_at_utc="2026-08-13T01:00:00Z",
        )
