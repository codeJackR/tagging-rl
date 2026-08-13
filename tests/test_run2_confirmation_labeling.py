from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_confirmation_labeling import (
    apply_batch_results,
    build_initial_labeling_state,
    build_pending_submission_items,
    finalize_frontier_bundle,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


class FakeProvider:
    name = "openai"
    model = "gpt-5.6-luna"

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


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture(scope="module")
def provider():
    return FakeProvider()


def _rows(n=400):
    return [
        {
            "sku_id": f"shopify:store-{index % 10:02d}.example:{index}",
            "source": f"shopify:store-{index % 10:02d}.example",
            "input": {
                "title": f"Dress {index}",
                "description": "Cotton midi dress.",
                "raw_tags": ["dress", "cotton"],
                "brand": f"Brand {index}",
                "category": "Dress",
                "image_url": None,
            },
        }
        for index in range(n)
    ]


def _valid_record(pack):
    return {
        name: [pack.unknown_token] if spec.kind == "multi" else pack.unknown_token
        for name, spec in pack.specs.items()
    }


def _initial(pack, provider):
    rows = _rows()
    state = build_initial_labeling_state(
        selected_rows=rows,
        membership_identity={"sha256": "a" * 64, "rows": 400},
        pack=pack,
        provider=provider,
    )
    return rows, state


def _results(items, pack):
    text = json.dumps(_valid_record(pack))
    return [
        {
            "custom_id": custom_id,
            "text": text,
            "error": None,
            "provider_result_id": f"result-{index}",
            "provider_request_id": f"request-{index}",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        for index, (custom_id, _body) in enumerate(items)
    ]


def test_initial_state_locks_exactly_400_by_five_slots(pack, provider) -> None:
    rows, state = _initial(pack, provider)

    assert state["counts"] == {
        "products": 400,
        "logical_slots": 2000,
        "usable_slots": 0,
        "pending_slots": 2000,
        "attempts": 0,
        "failed_attempts": 0,
        "batches": 0,
    }
    assert len(state["slots"]) == 2000
    assert {slot["variant"] for slot in state["slots"]} == {0, 1, 2, 3, 4}
    assert len({slot["pending"]["custom_id"] for slot in state["slots"]}) == 2000
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )
    assert len(items) == 2000


def test_wrong_membership_size_or_provider_fails_before_submission(pack, provider) -> None:
    with pytest.raises(ValueError, match="399 rows"):
        build_initial_labeling_state(
            selected_rows=_rows(399),
            membership_identity={"sha256": "a" * 64},
            pack=pack,
            provider=provider,
        )
    wrong = FakeProvider()
    wrong.model = "wrong-model"
    with pytest.raises(ValueError, match="provider/model"):
        build_initial_labeling_state(
            selected_rows=_rows(),
            membership_identity={"sha256": "a" * 64},
            pack=pack,
            provider=wrong,
        )


def test_all_usable_batch_becomes_ready_to_finalize(pack, provider) -> None:
    rows, state = _initial(pack, provider)
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )

    updated = apply_batch_results(
        state=state,
        batch_id="batch-initial",
        results=_results(items, pack),
        pack=pack,
    )

    assert updated["status"] == "all_slots_usable_ready_to_finalize"
    assert updated["counts"]["usable_slots"] == 2000
    assert updated["counts"]["pending_slots"] == 0
    assert updated["counts"]["attempts"] == 2000
    assert updated["counts"]["failed_attempts"] == 0


def test_malformed_attempt_retries_same_sku_and_variant(pack, provider) -> None:
    rows, state = _initial(pack, provider)
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )
    results = _results(items, pack)
    failed_custom_id = results[123]["custom_id"]
    results[123]["text"] = "not json"

    retry_state = apply_batch_results(
        state=state, batch_id="batch-initial", results=results, pack=pack
    )
    pending = [slot for slot in retry_state["slots"] if slot["pending"] is not None]
    assert len(pending) == 1
    slot = pending[0]
    assert slot["attempts"][0]["custom_id"] == failed_custom_id
    assert slot["pending"]["custom_id"].endswith("-a2")
    original_sku = slot["sku_id"]
    original_variant = slot["variant"]

    retry_items = build_pending_submission_items(
        state=retry_state, selected_rows=rows, pack=pack, provider=provider
    )
    final_state = apply_batch_results(
        state=retry_state,
        batch_id="batch-retry-1",
        results=_results(retry_items, pack),
        pack=pack,
    )
    final_slot = next(item for item in final_state["slots"] if item["slot_order"] == slot["slot_order"])
    assert final_slot["sku_id"] == original_sku
    assert final_slot["variant"] == original_variant
    assert len(final_slot["attempts"]) == 2
    assert final_state["counts"]["attempts"] == 2001
    assert final_state["counts"]["failed_attempts"] == 1
    assert final_state["status"] == "all_slots_usable_ready_to_finalize"


def test_missing_result_becomes_a_same_slot_retry(pack, provider) -> None:
    rows, state = _initial(pack, provider)
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )
    results = _results(items, pack)[:-1]

    updated = apply_batch_results(
        state=state, batch_id="batch-missing-one", results=results, pack=pack
    )

    assert updated["counts"]["pending_slots"] == 1
    missing = next(slot for slot in updated["slots"] if slot["pending"] is not None)
    assert missing["attempts"][0]["provider_error"] == "missing batch result"
    assert missing["pending"]["attempt"] == 2


def test_unexpected_duplicate_or_reapplied_batch_fails_closed(pack, provider) -> None:
    rows, state = _initial(pack, provider)
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )
    one = _results(items[:1], pack)[0]
    with pytest.raises(ValueError, match="duplicate result"):
        apply_batch_results(
            state=state, batch_id="duplicate", results=[one, one], pack=pack
        )
    with pytest.raises(ValueError, match="unexpected result"):
        apply_batch_results(
            state=state,
            batch_id="unexpected",
            results=[{"custom_id": "not-planned", "text": "{}", "error": None}],
            pack=pack,
        )

    complete = apply_batch_results(
        state=state,
        batch_id="once",
        results=_results(items, pack),
        pack=pack,
    )
    with pytest.raises(ValueError, match="already applied"):
        apply_batch_results(state=complete, batch_id="once", results=[], pack=pack)


def test_frontier_bundle_retains_raw_attempts_and_400_consensus_rows(
    pack, provider, tmp_path
) -> None:
    rows, state = _initial(pack, provider)
    items = build_pending_submission_items(
        state=state, selected_rows=rows, pack=pack, provider=provider
    )
    complete = apply_batch_results(
        state=state,
        batch_id="batch-initial",
        results=_results(items, pack),
        pack=pack,
    )
    output = tmp_path / "frontier-v1"

    manifest = finalize_frontier_bundle(
        output_dir=output,
        state=complete,
        selected_rows=rows,
        pack=pack,
    )

    assert manifest["status"] == "frontier_labels_complete_awaiting_human_review"
    assert manifest["counts"]["products"] == 400
    assert manifest["counts"]["logical_slots"] == 2000
    assert manifest["counts"]["frontier_cells"] == 6000
    assert manifest["invariants"]["all_attempts_and_raw_text_retained"] is True
    assert len((output / "attempts.jsonl").read_text().splitlines()) == 2000
    frontier = [json.loads(line) for line in (output / "frontier.jsonl").read_text().splitlines()]
    assert len(frontier) == 400
    assert all(row["provenance"]["self_consistency"]["k"] == 5 for row in frontier)
    assert all(len(row["labels"]) == 15 for row in frontier)

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_frontier_bundle(
            output_dir=output,
            state=complete,
            selected_rows=rows,
            pack=pack,
        )


def test_frontier_cannot_finalize_with_pending_slot(pack, provider, tmp_path) -> None:
    rows, state = _initial(pack, provider)

    with pytest.raises(ValueError, match="pending slots"):
        finalize_frontier_bundle(
            output_dir=tmp_path / "incomplete",
            state=state,
            selected_rows=rows,
            pack=pack,
        )
    assert list(tmp_path.iterdir()) == []
