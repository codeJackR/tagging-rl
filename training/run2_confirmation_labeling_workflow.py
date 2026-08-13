"""Immutable production workflow for Run 2 confirmation frontier labeling.

Every external Batch API action is surrounded by exclusive local evidence:
an intent is published before submission, a receipt is published after the
provider returns a batch ID, and collection creates a new state generation.
An unresolved intent must be investigated at the provider before resubmission;
it is never silently retried.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from labeling.providers import OpenAIProvider
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_confirmation_labeling import (
    EXPECTED_PRODUCTS,
    EXPECTED_SAMPLES_PER_PRODUCT,
    apply_batch_results,
    build_initial_labeling_state,
    build_pending_submission_items,
    finalize_frontier_bundle,
)
from verifier import Pack, load_pack


VERSION = "grpo-run2-confirmation-labeling-workflow-v1"
DEFAULT_SELECTED = "data/confirmation_run2_v1_prelabel/selected.jsonl"
DEFAULT_PRELABEL_MANIFEST = "data/confirmation_run2_v1_prelabel/manifest.json"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_INITIAL_STATE = "runs/grpo-run2-confirmation-labeling-state-0000.json"
DEFAULT_FRONTIER_OUTPUT = "data/confirmation_run2_v1_frontier"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _same_bytes(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _request_identity(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = _canonical_json(body).encode("utf-8")
    return {
        "custom_id": custom_id,
        "body_bytes": len(payload),
        "body_sha256": hashlib.sha256(payload).hexdigest(),
    }


def initialize_workflow_state(
    *,
    selected_path: str | Path,
    prelabel_manifest_path: str | Path,
    output_state_path: str | Path,
    pack: Pack,
    provider: Any,
    expected_products: int = EXPECTED_PRODUCTS,
    samples_per_product: int = EXPECTED_SAMPLES_PER_PRODUCT,
) -> dict[str, Any]:
    """Publish generation zero only after proving frozen membership identity."""

    selected_identity = _identity(selected_path)
    prelabel_identity = _identity(prelabel_manifest_path)
    prelabel = _read_json(prelabel_manifest_path)
    if prelabel.get("status") != "confirmation_membership_frozen_before_labeling":
        raise ValueError("prelabel bundle has not frozen membership before labeling")
    declared_selected = prelabel.get("files", {}).get("selected_snapshot")
    if not isinstance(declared_selected, dict) or not _same_bytes(
        selected_identity, declared_selected
    ):
        raise ValueError("selected.jsonl identity does not match prelabel manifest")
    rows = _read_jsonl(selected_path)
    state = build_initial_labeling_state(
        selected_rows=rows,
        membership_identity={
            "prelabel_manifest": prelabel_identity,
            "selected_snapshot": selected_identity,
        },
        pack=pack,
        provider=provider,
        expected_products=expected_products,
        samples_per_product=samples_per_product,
    )
    state["workflow"] = {
        "version": VERSION,
        "generation": 0,
        "parent_state": None,
        "submission_receipt": None,
    }
    write_exclusive_atomic_json(output_state_path, state)
    return state


def submit_pending_batch(
    *,
    state_path: str | Path,
    selected_path: str | Path,
    intent_path: str | Path,
    receipt_path: str | Path,
    pack: Pack,
    provider: Any,
    submitted_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish intent, submit once, then publish the returned provider batch ID."""

    intent_path = Path(intent_path)
    receipt_path = Path(receipt_path)
    if intent_path.exists() or receipt_path.exists():
        raise FileExistsError("submission intent or receipt already exists")
    state = _read_json(state_path)
    selected_identity = _identity(selected_path)
    if not _same_bytes(
        selected_identity,
        state.get("membership_identity", {}).get("selected_snapshot", {}),
    ):
        raise ValueError("selected membership drifted from labeling state")
    rows = _read_jsonl(selected_path)
    items = build_pending_submission_items(
        state=state,
        selected_rows=rows,
        pack=pack,
        provider=provider,
    )
    if not items:
        raise ValueError("labeling state has no pending requests")
    intent = {
        "version": VERSION,
        "status": "submission_intent_published_provider_state_unknown",
        "state": _identity(state_path),
        "selected_snapshot": selected_identity,
        "provider": provider.name,
        "model": provider.model,
        "request_count": len(items),
        "requests": [_request_identity(custom_id, body) for custom_id, body in items],
        "safety": {
            "do_not_resubmit_if_receipt_missing": True,
            "investigate_provider_account_for_orphan_batch": True,
        },
    }
    write_exclusive_atomic_json(intent_path, intent)
    batch_id = provider.submit(items)
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise RuntimeError("provider returned no batch ID; intent remains unresolved")
    receipt = {
        "version": VERSION,
        "status": "provider_batch_submitted_awaiting_completion",
        "submitted_at_utc": submitted_at_utc,
        "batch_id": batch_id.strip(),
        "provider": provider.name,
        "model": provider.model,
        "request_count": len(items),
        "intent": _identity(intent_path),
        "state": _identity(state_path),
        "execution_boundary": {
            "membership_changed": False,
            "human_review_performed": False,
            "sft_or_grpo_predictions_generated": False,
        },
    }
    write_exclusive_atomic_json(receipt_path, receipt)
    return intent, receipt


def collect_completed_batch(
    *,
    state_path: str | Path,
    selected_path: str | Path,
    intent_path: str | Path,
    receipt_path: str | Path,
    output_state_path: str | Path,
    pack: Pack,
    provider: Any,
    collected_at_utc: str,
) -> dict[str, Any]:
    """Collect one completed receipt into a new immutable state generation."""

    if Path(output_state_path).exists():
        raise FileExistsError(f"output state already exists: {output_state_path}")
    state = _read_json(state_path)
    intent = _read_json(intent_path)
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "provider_batch_submitted_awaiting_completion":
        raise ValueError("submission receipt is not awaiting completion")
    if not _same_bytes(receipt.get("intent", {}), _identity(intent_path)):
        raise ValueError("submission receipt does not identify the supplied intent")
    if not _same_bytes(receipt.get("state", {}), _identity(state_path)):
        raise ValueError("submission receipt does not identify the supplied state")
    if provider.name != receipt.get("provider") or provider.model != receipt.get("model"):
        raise ValueError("provider/model drifted from submission receipt")

    rows = _read_jsonl(selected_path)
    items = build_pending_submission_items(
        state=state,
        selected_rows=rows,
        pack=pack,
        provider=provider,
    )
    observed_requests = [_request_identity(custom_id, body) for custom_id, body in items]
    if observed_requests != intent.get("requests"):
        raise ValueError("pending request bodies drifted from submission intent")
    status = provider.status(receipt["batch_id"])
    if status.get("ready") is not True:
        raise RuntimeError(
            f"provider batch {receipt['batch_id']} is not ready: {status.get('status')}"
        )
    results = [dataclasses.asdict(result) for result in provider.results(receipt["batch_id"])]
    updated = apply_batch_results(
        state=state,
        batch_id=receipt["batch_id"],
        results=results,
        pack=pack,
    )
    generation = state.get("workflow", {}).get("generation")
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("input state has invalid workflow generation")
    updated["workflow"] = {
        "version": VERSION,
        "generation": generation + 1,
        "parent_state": _identity(state_path),
        "submission_receipt": _identity(receipt_path),
        "provider_status_at_collection": status,
        "collected_at_utc": collected_at_utc,
    }
    updated["batches"][-1]["submission_intent"] = _identity(intent_path)
    updated["batches"][-1]["submission_receipt"] = _identity(receipt_path)
    updated["batches"][-1]["provider_status_at_collection"] = status
    write_exclusive_atomic_json(output_state_path, updated)
    return updated


def finalize_workflow(
    *,
    state_path: str | Path,
    selected_path: str | Path,
    output_dir: str | Path,
    pack: Pack,
) -> dict[str, Any]:
    state = _read_json(state_path)
    selected = _read_jsonl(selected_path)
    return finalize_frontier_bundle(
        output_dir=output_dir,
        state=state,
        selected_rows=selected,
        pack=pack,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _provider() -> OpenAIProvider:
    return OpenAIProvider("gpt-5.6-luna")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--selected", default=DEFAULT_SELECTED)
    init.add_argument("--prelabel-manifest", default=DEFAULT_PRELABEL_MANIFEST)
    init.add_argument("--output-state", default=DEFAULT_INITIAL_STATE)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--state", required=True)
    submit.add_argument("--selected", default=DEFAULT_SELECTED)
    submit.add_argument("--intent", required=True)
    submit.add_argument("--receipt", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--receipt", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--state", required=True)
    collect.add_argument("--selected", default=DEFAULT_SELECTED)
    collect.add_argument("--intent", required=True)
    collect.add_argument("--receipt", required=True)
    collect.add_argument("--output-state", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--state", required=True)
    finalize.add_argument("--selected", default=DEFAULT_SELECTED)
    finalize.add_argument("--output", default=DEFAULT_FRONTIER_OUTPUT)

    args = parser.parse_args(argv)
    pack = load_pack(args.pack)
    provider = _provider()
    if args.command == "init":
        state_value = initialize_workflow_state(
            selected_path=args.selected,
            prelabel_manifest_path=args.prelabel_manifest,
            output_state_path=args.output_state,
            pack=pack,
            provider=provider,
        )
        result = {"status": state_value["status"], "output": args.output_state}
    elif args.command == "submit":
        _intent, receipt = submit_pending_batch(
            state_path=args.state,
            selected_path=args.selected,
            intent_path=args.intent,
            receipt_path=args.receipt,
            pack=pack,
            provider=provider,
            submitted_at_utc=_utc_now(),
        )
        result = {"status": receipt["status"], "batch_id": receipt["batch_id"]}
    elif args.command == "status":
        receipt = _read_json(args.receipt)
        result = {"batch_id": receipt["batch_id"], **provider.status(receipt["batch_id"])}
    elif args.command == "collect":
        state_value = collect_completed_batch(
            state_path=args.state,
            selected_path=args.selected,
            intent_path=args.intent,
            receipt_path=args.receipt,
            output_state_path=args.output_state,
            pack=pack,
            provider=provider,
            collected_at_utc=_utc_now(),
        )
        result = {"status": state_value["status"], "output": args.output_state}
    else:
        manifest = finalize_workflow(
            state_path=args.state,
            selected_path=args.selected,
            output_dir=args.output,
            pack=pack,
        )
        result = {"status": manifest["status"], "output": args.output}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
