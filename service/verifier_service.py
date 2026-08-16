#!/usr/bin/env python3
"""The verifier's second consumer: a production quality gate over HTTP.

W2 used the verifier as an RL reward. This is the other half of the claim the
project has been making since W1, and it is deliberately thin. It imports
`verifier.verify` and `verifier.verify_record` and adds **no verification logic
of its own**. If a record is judged differently here than it would be by the
reward function, the project has two verifiers regardless of what the import
graph says, and the architecture claim is false.

`tests/test_verifier_service.py` asserts exactly that equivalence. It is the
load-bearing artifact of this module; everything else here is plumbing.

Deliberately absent: authentication, persistence, rate limiting, deployment
configuration, async workers. This is a gate that answers one question about one
record. The W3 plan sets that stop condition explicitly, because backend work is
the most familiar thing in this project and therefore the most able to consume
the week without producing evidence.

Run it with:

    uv run uvicorn service.verifier_service:app --port 8100
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from verifier import Pack, VerifierResult, load_pack, verify, verify_record

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = os.environ.get("VERIFIER_PACK", "packs/vastraa_taste_v1")
SERVICE_VERSION = "verifier-service-v1"


@lru_cache(maxsize=4)
def get_pack(pack_path: str = DEFAULT_PACK) -> Pack:
    """Load a pack once per process.

    Cached because a pack is immutable data and reloading it per request would
    make the gate's cost depend on traffic rather than on the record.
    """
    path = Path(pack_path)
    if not path.is_absolute():
        path = ROOT / path
    return load_pack(path)


class VerifyRequest(BaseModel):
    """One record to judge, in exactly one of two forms.

    `raw` is what a model actually emitted, including any malformation, and is
    the form the reward function sees. `record` is an already-parsed object, the
    form the eval harness sees. Both are offered because a production caller may
    hold either, and both must reach the same verifier.
    """

    raw: str | None = Field(
        default=None, description="literal model output, judged without repair"
    )
    record: dict[str, Any] | None = Field(
        default=None, description="an already-parsed record"
    )
    normalize: bool = Field(
        default=False,
        description="record near-miss aliases; reports them, never silently repairs",
    )
    pack: str | None = Field(default=None, description="pack path; defaults to the apparel pack")


class VerifyResponse(BaseModel):
    schema_valid: bool
    vocab_valid: bool
    ok: bool
    rule_violations: list[str]
    errors: list[str]
    abstentions: list[str]
    normalized: dict[str, str]
    parsed: dict[str, Any] | None


def to_response(result: VerifierResult) -> VerifyResponse:
    """Project a VerifierResult onto the wire, adding nothing.

    `ok` is the verifier's own property rather than a re-derivation here. A
    service that recomputed "is this acceptable" would be the second verifier
    this module exists to avoid.
    """
    return VerifyResponse(
        schema_valid=result.schema_valid,
        vocab_valid=result.vocab_valid,
        ok=result.ok,
        rule_violations=list(result.rule_violations),
        errors=list(result.errors),
        abstentions=list(result.abstentions),
        normalized=dict(result.normalized),
        parsed=result.parsed,
    )


app = FastAPI(
    title="Catalog verifier gate",
    version=SERVICE_VERSION,
    description=(
        "The same verifier that scores RL rollouts, exposed as a production "
        "quality gate. It adds no verification logic of its own."
    ),
)


@app.get("/health")
def health(pack: str | None = None) -> dict[str, Any]:
    """Report which vocabulary a caller is being judged against.

    A gate whose pack is unidentifiable is not auditable: a record rejected
    today and accepted tomorrow would be indistinguishable from a vocabulary
    change. The field and value counts are the cheap version of that identity.
    """
    loaded = get_pack(pack or DEFAULT_PACK)
    inventory = loaded.rule_inventory()
    return {
        "status": "ok",
        "service_version": SERVICE_VERSION,
        "pack": loaded.name,
        "fields": len(loaded.field_names),
        "vocabulary_values": sum(len(spec.values) for spec in loaded.specs.values()),
        # Written rules and those derived from applies_to metadata are counted
        # separately: a caller needs to know the enforced total, and a reader
        # of an audit needs to know how much of it was hand-authored.
        "rules_written": len(inventory.get("written", [])),
        "rules_derived": len(inventory.get("derived", [])),
        # `total` is the pack's own key, not a sum over the others: summing
        # would count it twice and report 68 rules where there are 34.
        "rules_total": inventory["total"],
        "unknown_token": loaded.unknown_token,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify_endpoint(request: VerifyRequest) -> VerifyResponse:
    """Judge one record. Exactly one of `raw` or `record` must be supplied."""
    if (request.raw is None) == (request.record is None):
        raise HTTPException(
            status_code=422,
            detail="supply exactly one of 'raw' or 'record'",
        )
    try:
        loaded = get_pack(request.pack or DEFAULT_PACK)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown pack: {exc}") from exc

    if request.raw is not None:
        result = verify(request.raw, loaded, normalize=request.normalize)
    else:
        result = verify_record(request.record, loaded, normalize=request.normalize)
    return to_response(result)
