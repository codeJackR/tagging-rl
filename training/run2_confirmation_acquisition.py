"""Permission-gated acquisition and pre-label membership publication.

This module is intentionally separate from the general W1 Shopify fetcher.  It
implements the stronger Run 2 confirmation contract: source permission first,
global stop on HTTP 403/429, complete page lineage, no label/model fields, and
one collision-protected bundle containing both the raw candidate snapshot and
the deterministic 400-product membership selected from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.fetch_shopify import (
    _get,
    build_apparel_matcher,
    prune_ubiquitous_tags,
    to_row,
)
from labeling.records import read_jsonl
from training.audit_data_boundaries import (
    sha256_file,
    sku_set_sha256,
    write_exclusive_atomic_json,
)
from training.run2_confirmation_selector import (
    SelectionPolicy,
    select_confirmation_candidates,
)
from training.run2_confirmation_source_gate import evaluate_source_audit
from training.split_sft import group_key
from verifier import Pack, load_pack


VERSION = "grpo-run2-confirmation-acquisition-v1"
BUNDLE_VERSION = "grpo-run2-confirmation-prelabel-bundle-v1"
MINIMUM_DELAY_SECONDS = 1.0
HARD_STOP_STATUSES = frozenset({403, 429})
DEFAULT_MAX_PAGES = 20
PAGE_LIMIT = 250
DEFAULT_SOURCE_AUDIT = "runs/grpo-run2-confirmation-terms-audit.json"
DEFAULT_PRIOR_LABELED = "data/raw/labeled.jsonl"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_OUTPUT = "data/confirmation_run2_v1_prelabel"
DEFAULT_FAILURE_OUTPUT = "runs/grpo-run2-confirmation-acquisition-failure.json"
FINAL_FILES = frozenset(
    {
        "acquisition-manifest.json",
        "candidates.jsonl",
        "manifest.json",
        "selected.jsonl",
        "selection-manifest.json",
    }
)

PageRequester = Callable[[str, int], tuple[int, bytes]]
Clock = Callable[[], str]
Sleeper = Callable[[float], None]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _body_identity(body: bytes) -> dict[str, Any]:
    return {
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def _approved_domains_in_audit_order(audit: Mapping[str, Any]) -> list[str]:
    return [
        str(candidate["endpoint_domain"])
        for candidate in audit["candidates"]
        if candidate["decision"] == "approved"
    ]


def _new_store_stats() -> dict[str, Any]:
    return {
        "requests": 0,
        "successful_pages": 0,
        "rows_seen": 0,
        "rows_retained": 0,
        "non_apparel": 0,
        "empty_title": 0,
        "duplicate_sku": 0,
        "errors": [],
    }


def _parse_products(body: bytes) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "response was not JSON"
    if not isinstance(value, Mapping):
        return None, "JSON response was not an object"
    products = value.get("products")
    if not isinstance(products, list):
        return None, "JSON response had no products array"
    if any(not isinstance(product, Mapping) for product in products):
        return None, "products array contained a non-object"
    return [dict(product) for product in products], None


def acquire_confirmation_candidates(
    *,
    source_audit: Mapping[str, Any],
    pack: Pack,
    request_page: PageRequester,
    now_utc: Clock,
    sleep: Sleeper = time.sleep,
    delay_seconds: float = MINIMUM_DELAY_SECONDS,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch all available approved pages in deterministic round-robin order.

    A 403 or 429 stops the entire acquisition immediately. Other source errors
    retire only that source and remain visible in the request ledger.
    """

    gate = evaluate_source_audit(source_audit)
    if not gate["passed"]:
        raise PermissionError(gate["failure_reason"])
    if delay_seconds < MINIMUM_DELAY_SECONDS:
        raise ValueError(
            f"delay_seconds must be at least {MINIMUM_DELAY_SECONDS}"
        )
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if pack.category_field is None:
        raise ValueError("confirmation pack has no category field")

    domains = _approved_domains_in_audit_order(source_audit)
    if domains != gate["approved_domains"]:
        # The gate returns a sorted set. Acquisition order must also be stable,
        # so the audit itself must list approvals lexicographically.
        raise ValueError("approved domains must be sorted in the source audit")

    matcher = build_apparel_matcher(pack)
    started_at = now_utc()
    store_stats = {domain: _new_store_stats() for domain in domains}
    request_ledger: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_skus: set[str] = set()
    live = list(domains)
    first_request = True
    hard_stop: dict[str, Any] | None = None

    for page in range(1, max_pages + 1):
        if not live or hard_stop is not None:
            break
        for domain in list(live):
            if not first_request:
                sleep(delay_seconds)
            first_request = False
            requested_at = now_utc()
            status, body = request_page(domain, page)
            completed_at = now_utc()
            if not isinstance(status, int) or not isinstance(body, bytes):
                raise TypeError("request_page must return (int, bytes)")

            identity = _body_identity(body)
            request_record = {
                "request_order": len(request_ledger) + 1,
                "domain": domain,
                "page": page,
                "url": (
                    f"https://{domain}/products.json?limit={PAGE_LIMIT}&page={page}"
                ),
                "requested_at_utc": requested_at,
                "completed_at_utc": completed_at,
                "http_status": status,
                "response_bytes": identity["bytes"],
                "response_sha256": identity["sha256"],
            }
            request_ledger.append(request_record)
            stats = store_stats[domain]
            stats["requests"] += 1

            if status in HARD_STOP_STATUSES:
                message = f"global hard stop on HTTP {status} from {domain} page {page}"
                stats["errors"].append(message)
                hard_stop = {**request_record, "reason": message}
                break
            if status != 200:
                stats["errors"].append(f"HTTP {status} on page {page}")
                live.remove(domain)
                continue

            products, parse_error = _parse_products(body)
            if parse_error is not None:
                stats["errors"].append(f"page {page}: {parse_error}")
                live.remove(domain)
                continue
            assert products is not None
            stats["successful_pages"] += 1
            if not products:
                live.remove(domain)
                continue

            for product in products:
                stats["rows_seen"] += 1
                title = str(product.get("title") or "").strip()
                if not title:
                    stats["empty_title"] += 1
                    continue
                if not matcher(product):
                    stats["non_apparel"] += 1
                    continue
                row = to_row(product, domain)
                sku_id = row["sku_id"]
                if sku_id in seen_skus:
                    stats["duplicate_sku"] += 1
                    continue
                seen_skus.add(sku_id)
                rows.append(row)
                stats["rows_retained"] += 1

    tags_pruned = prune_ubiquitous_tags(rows, pack)
    finished_at = now_utc()
    report = {
        "version": VERSION,
        "status": "stopped_on_blocking_http_status"
        if hard_stop is not None
        else "candidate_snapshot_acquired",
        "source_gate": gate,
        "configuration": {
            "approved_domains": domains,
            "minimum_delay_seconds": MINIMUM_DELAY_SECONDS,
            "configured_delay_seconds": delay_seconds,
            "maximum_pages_per_store": max_pages,
            "page_limit": PAGE_LIMIT,
            "traversal": "domain-sorted round-robin by page",
            "hard_stop_http_statuses": sorted(HARD_STOP_STATUSES),
            "apparel_filter": "locked pack garment-category aliases",
        },
        "timing": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        },
        "counts": {
            "requests": len(request_ledger),
            "approved_domains": len(domains),
            "candidate_rows": len(rows),
            "tags_pruned": tags_pruned,
        },
        "per_store": store_stats,
        "requests": request_ledger,
        "hard_stop": hard_stop,
        "execution_boundary": {
            "frontier_labeling_performed": False,
            "human_review_performed": False,
            "sft_or_grpo_model_inference_performed": False,
            "gpu_training_performed": False,
        },
    }
    return rows, report


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_prelabel_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    forbidden = {
        "labels",
        "provenance",
        "difficulty",
        "prediction",
        "predictions",
        "reward",
        "rewards",
        "model_output",
    }
    for index, row in enumerate(rows, start=1):
        overlap = forbidden.intersection(row)
        if overlap:
            raise ValueError(f"candidate row {index} contains post-treatment keys: {sorted(overlap)}")


def publish_prelabel_bundle(
    *,
    output_dir: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    acquisition_report: Mapping[str, Any],
    prior_sku_ids: set[str] | frozenset[str],
    prior_family_keys: set[str] | frozenset[str],
    pack: Pack,
    source_audit_identity: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    expected_prior_sku_count: int = 4_000,
    policy: SelectionPolicy = SelectionPolicy(),
) -> dict[str, Any]:
    """Select membership and atomically publish one immutable pre-label bundle."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"pre-label output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"pre-label output parent does not exist: {output_dir.parent}")
    if acquisition_report.get("status") != "candidate_snapshot_acquired":
        raise ValueError("only a completed candidate snapshot may be published")
    if acquisition_report.get("hard_stop") is not None:
        raise ValueError("hard-stopped acquisition cannot become a pre-label bundle")
    if acquisition_report.get("counts", {}).get("candidate_rows") != len(candidates):
        raise ValueError("acquisition candidate count does not match supplied rows")
    if len(prior_sku_ids) != expected_prior_sku_count:
        raise ValueError(
            f"prior exclusion universe has {len(prior_sku_ids)} SKUs; "
            f"expected {expected_prior_sku_count}"
        )
    _validate_prelabel_rows(candidates)
    if pack.category_field is None:
        raise ValueError("confirmation pack has no category field")
    category_spec = pack.specs[pack.category_field]

    selection = select_confirmation_candidates(
        candidates,
        prior_sku_ids=prior_sku_ids,
        prior_family_keys=prior_family_keys,
        category_aliases=category_spec.aliases,
        category_order=category_spec.values,
        policy=policy,
    )
    candidate_by_sku = {str(row["sku_id"]): row for row in candidates}
    if len(candidate_by_sku) != len(candidates):
        raise ValueError("candidate snapshot contains duplicate SKU IDs")
    selected_rows = [candidate_by_sku[item["sku_id"]] for item in selection["selected"]]

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        candidates_path = staging / "candidates.jsonl"
        selected_path = staging / "selected.jsonl"
        acquisition_path = staging / "acquisition-manifest.json"
        selection_path = staging / "selection-manifest.json"
        manifest_path = staging / "manifest.json"

        _write_jsonl(candidates_path, candidates)
        _write_jsonl(selected_path, selected_rows)
        acquisition_manifest = {
            **dict(acquisition_report),
            "source_audit_identity": dict(source_audit_identity),
            "candidate_snapshot": _file_identity(candidates_path),
        }
        _write_json(acquisition_path, acquisition_manifest)

        selection_manifest = {
            **selection,
            "inputs": {
                "acquisition_manifest": _file_identity(acquisition_path),
                "candidate_snapshot": _file_identity(candidates_path),
                "prior_exclusion_universe": {
                    "sku_count": len(prior_sku_ids),
                    "family_count": len(prior_family_keys),
                    "sorted_sku_set_sha256": sku_set_sha256(prior_sku_ids),
                    "sorted_family_set_sha256": sku_set_sha256(prior_family_keys),
                },
                "pack": {
                    "name": pack.name,
                    "vocab_sha256": sha256_file(pack.path / "vocab.yaml"),
                    "rules_sha256": sha256_file(pack.path / "rules.yaml"),
                },
            },
            "code": dict(code_identity),
            "selected_snapshot": _file_identity(selected_path),
        }
        _write_json(selection_path, selection_manifest)

        manifest = {
            "version": BUNDLE_VERSION,
            "status": "confirmation_membership_frozen_before_labeling",
            "role": "immutable_prelabel_confirmation_membership",
            "files": {
                "acquisition_manifest": _file_identity(acquisition_path),
                "candidate_snapshot": _file_identity(candidates_path),
                "selected_snapshot": _file_identity(selected_path),
                "selection_manifest": _file_identity(selection_path),
            },
            "counts": {
                "candidate_rows": len(candidates),
                "selected_rows": len(selected_rows),
                "selected_stores": selection["counts"]["selected_stores"],
            },
            "invariants": {
                "membership_selected_before_labels": True,
                "exact_prior_sku_overlap": 0,
                "normalized_prior_family_overlap": 0,
                "label_or_model_fields_present": False,
                "published_exclusively_and_atomically": True,
            },
            "execution_boundary": {
                "frontier_labeling_performed": False,
                "human_review_performed": False,
                "sft_or_grpo_model_inference_performed": False,
                "gpu_training_performed": False,
            },
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != FINAL_FILES:
            raise RuntimeError("pre-label staging bundle has unexpected files")
        os.rename(staging, output_dir)
        if staging.exists() or not output_dir.is_dir():
            raise RuntimeError("atomic pre-label publication did not complete")
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _network_request_page(domain: str, page: int, *, timeout: int = 20) -> tuple[int, bytes]:
    url = f"https://{domain}/products.json?limit={PAGE_LIMIT}&page={page}"
    return _get(url, timeout)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_context(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("confirmation acquisition requires a clean committed worktree")
    paths = [
        "training/run2_confirmation_acquisition.py",
        "training/run2_confirmation_selector.py",
        "training/run2_confirmation_source_gate.py",
        "tools/fetch_shopify.py",
    ]
    return {
        "git_commit": commit,
        "files": {
            path: {"bytes": (repo_root / path).stat().st_size, "sha256": sha256_file(repo_root / path)}
            for path in paths
        },
    }


def _identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-audit", default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--prior-labeled", default=DEFAULT_PRIOR_LABELED)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=MINIMUM_DELAY_SECONDS)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    source_audit_path = (root / args.source_audit).resolve()
    prior_path = (root / args.prior_labeled).resolve()
    pack_path = (root / args.pack).resolve()
    output = (root / args.output).resolve()
    failure_output = (root / args.failure_output).resolve()
    if output.exists():
        raise FileExistsError(f"pre-label output already exists: {output}")
    if failure_output.exists():
        raise FileExistsError(f"acquisition failure output already exists: {failure_output}")

    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    gate = evaluate_source_audit(source_audit)
    if not gate["passed"]:
        print(json.dumps(gate, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    code_identity = _git_context(root)
    pack = load_pack(pack_path)
    prior_rows = read_jsonl(prior_path)
    prior_sku_ids = frozenset(row.sku_id for row in prior_rows)
    prior_family_keys = frozenset(group_key(row) for row in prior_rows)
    if len(prior_sku_ids) != 4_000:
        raise ValueError(f"prior labeled universe has {len(prior_sku_ids)} unique SKUs, expected 4000")

    candidates, acquisition_report = acquire_confirmation_candidates(
        source_audit=source_audit,
        pack=pack,
        request_page=_network_request_page,
        now_utc=_utc_now,
        delay_seconds=args.delay_seconds,
        max_pages=args.max_pages,
    )
    if acquisition_report["status"] != "candidate_snapshot_acquired":
        failure = {
            **acquisition_report,
            "source_audit_identity": _identity(source_audit_path, relative_to=root),
            "code": code_identity,
            "partial_candidate_rows_retained_in_memory_only": len(candidates),
            "prelabel_bundle_published": False,
        }
        write_exclusive_atomic_json(failure_output, failure)
        print(json.dumps({"status": failure["status"], "output": str(failure_output)}))
        return 3

    manifest = publish_prelabel_bundle(
        output_dir=output,
        candidates=candidates,
        acquisition_report=acquisition_report,
        prior_sku_ids=prior_sku_ids,
        prior_family_keys=prior_family_keys,
        pack=pack,
        source_audit_identity=_identity(source_audit_path, relative_to=root),
        code_identity=code_identity,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_rows": manifest["counts"]["candidate_rows"],
                "selected_rows": manifest["counts"]["selected_rows"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
