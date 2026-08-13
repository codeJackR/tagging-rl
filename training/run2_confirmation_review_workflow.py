"""Production commands for blinded Run 2 confirmation human review.

This wrapper only transforms immutable frontier evidence and completed human
CSV packets. It never calls a model and never computes confirmation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_confirmation_review import (
    EXPECTED_PRODUCTS,
    SECOND_REVIEW_PRODUCTS,
    build_adjudication_packet,
    build_review_packets,
    compare_independent_reviews,
    finalize_reviewed_bundle,
    import_adjudication,
    import_completed_review,
    publish_review_packets,
)
from verifier import Pack, load_pack


VERSION = "grpo-run2-confirmation-review-workflow-v1"
DEFAULT_FRONTIER_DIR = "data/confirmation_run2_v1_frontier"
DEFAULT_PACKET_DIR = "data/confirmation_run2_v1_review_packets"
DEFAULT_COMPARISON_DIR = "data/confirmation_run2_v1_review_comparison"
DEFAULT_REVIEWED_DIR = "data/confirmation_run2_v1_reviewed"
DEFAULT_PACK = "packs/vastraa_taste_v1"
ADJUDICATION_COLUMNS = (
    "cell_id",
    "membership_order",
    "sku_id",
    "attribute",
    "primary_reviewer_id",
    "second_reviewer_id",
    "primary_label_json",
    "second_label_json",
    "adjudicator_id",
    "decision",
    "custom_status",
    "custom_value_json",
    "rationale",
    "adjudicated_at_utc",
)
COMPARISON_FILES = frozenset({"adjudication.csv", "comparison.json", "manifest.json"})


def _identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _same_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("bytes") == right.get("bytes") and left.get("sha256") == right.get("sha256")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object: {path}")
        values.append(value)
    return values


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def prepare_review_workflow(
    *,
    frontier_dir: str | Path,
    output_dir: str | Path,
    pack: Pack,
    expected_products: int = EXPECTED_PRODUCTS,
    audit_products: int = SECOND_REVIEW_PRODUCTS,
) -> dict[str, Any]:
    frontier_dir = Path(frontier_dir).resolve()
    frontier_path = frontier_dir / "frontier.jsonl"
    attempts_path = frontier_dir / "attempts.jsonl"
    manifest_path = frontier_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "frontier_labels_complete_awaiting_human_review":
        raise ValueError("frontier bundle is not ready for human review")
    for key, path in (("frontier", frontier_path), ("attempts", attempts_path)):
        declared = manifest.get("files", {}).get(key)
        if not isinstance(declared, Mapping) or not _same_bytes(declared, _identity(path)):
            raise ValueError(f"frontier manifest {key} identity drifted")
    primary, secondary, plan = build_review_packets(
        frontier_rows=_read_jsonl(frontier_path),
        attempts=_read_jsonl(attempts_path),
        pack=pack,
        expected_products=expected_products,
        audit_products=audit_products,
    )
    return publish_review_packets(
        output_dir=output_dir,
        primary=primary,
        secondary=secondary,
        plan_manifest=plan,
        frontier_identity={
            "bundle_manifest": _identity(manifest_path),
            "frontier": _identity(frontier_path),
            "attempts": _identity(attempts_path),
        },
    )


def import_review_workflow(
    *,
    expected_packet_path: str | Path,
    completed_packet_path: str | Path,
    output_path: str | Path,
    pack: Pack,
    review_role: str,
) -> dict[str, Any]:
    if review_role not in {"primary_review", "second_review"}:
        raise ValueError("review_role must be primary_review or second_review")
    result = import_completed_review(
        expected_packet=_read_csv(expected_packet_path),
        completed_packet=_read_csv(completed_packet_path),
        pack=pack,
        review_role=review_role,
    )
    result["workflow"] = {
        "version": VERSION,
        "expected_packet": _identity(expected_packet_path),
        "completed_packet": _identity(completed_packet_path),
    }
    write_exclusive_atomic_json(output_path, result)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def compare_review_workflow(
    *,
    primary_review_path: str | Path,
    second_review_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"comparison parent does not exist: {output_dir.parent}")
    primary = _read_json(primary_review_path)
    second = _read_json(second_review_path)
    comparison = compare_independent_reviews(
        primary_review=primary,
        second_review=second,
    )
    adjudication_rows = build_adjudication_packet(comparison)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        comparison_path = staging / "comparison.json"
        adjudication_path = staging / "adjudication.csv"
        manifest_path = staging / "manifest.json"
        _write_json(comparison_path, comparison)
        _write_csv(adjudication_path, adjudication_rows)
        manifest = {
            "version": VERSION,
            "status": "independent_reviews_compared_adjudication_packet_published",
            "inputs": {
                "primary_review": _identity(primary_review_path),
                "second_review": _identity(second_review_path),
            },
            "counts": dict(comparison["counts"]),
            "files": {
                "comparison": _identity(comparison_path),
                "adjudication_packet": _identity(adjudication_path),
            },
            "publication": {"exclusive": True, "atomic": True},
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != COMPARISON_FILES:
            raise RuntimeError("review comparison staging has unexpected files")
        os.rename(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def import_adjudication_workflow(
    *,
    comparison_path: str | Path,
    completed_packet_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    result = import_adjudication(
        comparison=_read_json(comparison_path),
        completed_packet=_read_csv(completed_packet_path),
    )
    result["workflow"] = {
        "version": VERSION,
        "comparison": _identity(comparison_path),
        "completed_adjudication_packet": _identity(completed_packet_path),
    }
    write_exclusive_atomic_json(output_path, result)
    return result


def finalize_review_workflow(
    *,
    frontier_path: str | Path,
    primary_review_path: str | Path,
    second_review_path: str | Path,
    comparison_path: str | Path,
    adjudication_path: str | Path,
    output_dir: str | Path,
    pack: Pack,
) -> dict[str, Any]:
    return finalize_reviewed_bundle(
        output_dir=output_dir,
        frontier_rows=_read_jsonl(frontier_path),
        primary_review=_read_json(primary_review_path),
        second_review=_read_json(second_review_path),
        comparison=_read_json(comparison_path),
        adjudication=_read_json(adjudication_path),
        pack=pack,
        frontier_identity=_identity(frontier_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--frontier-dir", default=DEFAULT_FRONTIER_DIR)
    prepare.add_argument("--output", default=DEFAULT_PACKET_DIR)
    import_review = subs.add_parser("import-review")
    import_review.add_argument("--role", required=True, choices=("primary_review", "second_review"))
    import_review.add_argument("--expected", required=True)
    import_review.add_argument("--completed", required=True)
    import_review.add_argument("--output", required=True)
    compare = subs.add_parser("compare")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--second", required=True)
    compare.add_argument("--output", default=DEFAULT_COMPARISON_DIR)
    import_adj = subs.add_parser("import-adjudication")
    import_adj.add_argument("--comparison", required=True)
    import_adj.add_argument("--completed", required=True)
    import_adj.add_argument("--output", required=True)
    finalize = subs.add_parser("finalize")
    finalize.add_argument("--frontier", default=f"{DEFAULT_FRONTIER_DIR}/frontier.jsonl")
    finalize.add_argument("--primary", required=True)
    finalize.add_argument("--second", required=True)
    finalize.add_argument("--comparison", default=f"{DEFAULT_COMPARISON_DIR}/comparison.json")
    finalize.add_argument("--adjudication", required=True)
    finalize.add_argument("--output", default=DEFAULT_REVIEWED_DIR)
    args = parser.parse_args(argv)
    pack = load_pack(args.pack)
    if args.command == "prepare":
        result = prepare_review_workflow(frontier_dir=args.frontier_dir, output_dir=args.output, pack=pack)
    elif args.command == "import-review":
        result = import_review_workflow(
            expected_packet_path=args.expected,
            completed_packet_path=args.completed,
            output_path=args.output,
            pack=pack,
            review_role=args.role,
        )
    elif args.command == "compare":
        result = compare_review_workflow(
            primary_review_path=args.primary,
            second_review_path=args.second,
            output_dir=args.output,
        )
    elif args.command == "import-adjudication":
        result = import_adjudication_workflow(
            comparison_path=args.comparison,
            completed_packet_path=args.completed,
            output_path=args.output,
        )
    else:
        result = finalize_review_workflow(
            frontier_path=args.frontier,
            primary_review_path=args.primary,
            second_review_path=args.second,
            comparison_path=args.comparison,
            adjudication_path=args.adjudication,
            output_dir=args.output,
            pack=pack,
        )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
