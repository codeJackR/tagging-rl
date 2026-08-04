#!/usr/bin/env python3
"""Build the deterministic five-row fixture for the first GRPO smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from labeling.records import LabelStatus, read_jsonl, write_jsonl
from training.build_grpo_pool import POOL_VERSION
from training.split_sft import family_title, group_key, row_category

SMOKE_VERSION = "grpo-smoke-v1"
DEFAULT_ACTIVE_DATA = "data/train_weak_grpo_cap4.jsonl"
DEFAULT_POOL_MANIFEST = "runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json"
DEFAULT_OUTPUT_DATA = "data/train_weak_grpo_smoke_v1.jsonl"
DEFAULT_OUTPUT_MANIFEST = "data/splits/grpo-smoke-v1.json"
DEFAULT_SEED = 42
DEFAULT_TARGET_PASS_RATE = 0.5
DEFAULT_ROWS = 5


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sku_sha256(rows) -> str:
    payload = "\n".join(row.sku_id for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sku_set_sha256(rows) -> str:
    payload = "\n".join(sorted(row.sku_id for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_key(row, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{row.sku_id}".encode("utf-8")).hexdigest()


def _store(row) -> str:
    parts = row.sku_id.split(":", 2)
    return parts[1] if len(parts) == 3 else "<unknown>"


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], check=False
        ).returncode != 0
        return {"git_commit": commit, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "tracked_worktree_dirty": None}


def verify_active_handoff(rows, manifest: dict, active_path: str | Path) -> None:
    """Refuse to select from anything other than the locked cap-four handoff."""
    if manifest.get("version") != POOL_VERSION:
        raise ValueError("unexpected cap-four pool manifest version")
    if len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("active GRPO dataset contains duplicate SKU IDs")

    output = manifest.get("output", {})
    selection = manifest.get("selection", {})
    actual_hash = _sha256_file(active_path)
    if actual_hash != output.get("active_dataset_sha256"):
        raise ValueError("active GRPO dataset hash disagrees with pool manifest")
    if len(rows) != output.get("active_dataset_rows"):
        raise ValueError("active GRPO row count disagrees with pool manifest")
    if len(rows) != selection.get("active_rows"):
        raise ValueError("active selection count disagrees with pool manifest")
    if _sku_set_sha256(rows) != selection.get("active_sku_set_sha256"):
        raise ValueError("active SKU set disagrees with pool manifest")


def select_smoke_rows(
    rows,
    *,
    seed: int = DEFAULT_SEED,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    count: int = DEFAULT_ROWS,
):
    """Select hash-ordered target-rate rows, at most one per product family."""
    if count <= 0:
        raise ValueError("smoke row count must be positive")
    if not 0 < target_pass_rate < 1:
        raise ValueError("target pass rate must be strictly between zero and one")
    if len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("active GRPO dataset contains duplicate SKU IDs")

    candidates = [
        row
        for row in rows
        if row.difficulty.sft_pass_rate == target_pass_rate
    ]
    candidates.sort(key=lambda row: _selection_key(row, seed=seed))

    selected = []
    seen_families = set()
    for row in candidates:
        family = group_key(row)
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(row)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} distinct families available at pass rate "
            f"{target_pass_rate}; need {count}"
        )
    return candidates, selected


def _gold_density(rows) -> dict:
    scorable = []
    substantive = []
    for row in rows:
        counts = Counter(label.status for label in row.labels.values())
        scorable.append(
            counts[LabelStatus.LABELED] + counts[LabelStatus.NOT_APPLICABLE]
        )
        substantive.append(counts[LabelStatus.LABELED])
    return {
        "mean_scorable": sum(scorable) / len(rows),
        "mean_substantive_labeled": sum(substantive) / len(rows),
        "minimum_scorable": min(scorable),
        "maximum_scorable": max(scorable),
    }


def build_smoke_manifest(
    *,
    rows,
    candidates,
    selected,
    active_path: str | Path,
    pool_manifest_path: str | Path,
    output_data_path: str | Path,
    seed: int,
    target_pass_rate: float,
    created_at_utc: str,
) -> dict:
    output_rows = read_jsonl(output_data_path)
    if output_rows != selected:
        raise ValueError("written smoke data changed selection order or contents")

    source_positions = {row.sku_id: index for index, row in enumerate(rows)}
    selected_families = [group_key(row) for row in selected]
    implementation_path = Path(__file__)
    try:
        implementation_display = str(
            implementation_path.resolve().relative_to(Path.cwd())
        )
    except ValueError:
        implementation_display = str(implementation_path)

    return {
        "version": SMOKE_VERSION,
        "created_at_utc": created_at_utc,
        "inputs": {
            "active_dataset": str(active_path),
            "active_dataset_rows": len(rows),
            "active_dataset_sha256": _sha256_file(active_path),
            "active_sku_set_sha256": _sku_set_sha256(rows),
            "cap_four_pool_manifest": str(pool_manifest_path),
            "cap_four_pool_manifest_sha256": _sha256_file(pool_manifest_path),
            "cap_four_pool_manifest_version": POOL_VERSION,
        },
        "policy": {
            "target_sft_pass_rate": target_pass_rate,
            "selection_seed": seed,
            "candidate_order": f"SHA-256 of '{seed}\\0<sku_id>', ascending",
            "family_grouping": (
                "training.split_sft.group_key: normalized brand + title before "
                "spaced variant separator"
            ),
            "maximum_rows_per_family": 1,
            "requested_rows": len(selected),
            "output_order": "selection order; one product group per smoke step",
        },
        "selection": {
            "candidate_rows": len(candidates),
            "candidate_families": len({group_key(row) for row in candidates}),
            "selected_rows": len(selected),
            "selected_families": len(set(selected_families)),
            "selected_skus_in_step_order": [row.sku_id for row in selected],
            "selected_sku_order_sha256": _ordered_sku_sha256(selected),
            "selected_sku_set_sha256": _sku_set_sha256(selected),
            "records_in_step_order": [
                {
                    "step": index,
                    "sku_id": row.sku_id,
                    "active_source_position_zero_based": source_positions[row.sku_id],
                    "selection_sha256": _selection_key(row, seed=seed),
                    "family_key": group_key(row).replace("\0", "::"),
                    "family_title": family_title(row.input.title),
                    "store": _store(row),
                    "brand": row.input.brand,
                    "title": row.input.title,
                    "garment_category": row_category(row),
                    "sft_pass_rate": row.difficulty.sft_pass_rate,
                }
                for index, row in enumerate(selected, start=1)
            ],
        },
        "output": {
            "smoke_dataset": str(output_data_path),
            "smoke_dataset_rows": len(output_rows),
            "smoke_dataset_bytes": Path(output_data_path).stat().st_size,
            "smoke_dataset_sha256": _sha256_file(output_data_path),
        },
        "composition": {
            "store_counts": dict(
                sorted(Counter(_store(row) for row in selected).items())
            ),
            "category_counts": dict(
                sorted(Counter(row_category(row) for row in selected).items())
            ),
            "gold_density": _gold_density(selected),
        },
        "invariants": {
            "all_rows_come_from_active_dataset": all(
                row.sku_id in source_positions for row in selected
            ),
            "all_pass_rates_equal_target": all(
                row.difficulty.sft_pass_rate == target_pass_rate
                for row in selected
            ),
            "all_skus_distinct": len({row.sku_id for row in selected})
            == len(selected),
            "all_families_distinct": len(set(selected_families))
            == len(selected),
            "all_rows_are_training_rows": all(row.split == "train" for row in selected),
            "output_matches_selection_order_and_contents": output_rows == selected,
        },
        "code": {
            **_git_state(),
            "implementation_file": implementation_display,
            "implementation_file_sha256": _sha256_file(implementation_path),
        },
    }


def write_smoke_artifacts(
    *,
    active_path: str | Path,
    pool_manifest_path: str | Path,
    output_data_path: str | Path,
    output_manifest_path: str | Path,
    seed: int,
    target_pass_rate: float,
    count: int,
    created_at_utc: str,
    overwrite: bool = False,
) -> dict:
    output_data_path = Path(output_data_path)
    output_manifest_path = Path(output_manifest_path)
    collisions = [
        path for path in (output_data_path, output_manifest_path) if path.exists()
    ]
    if collisions and not overwrite:
        raise FileExistsError(
            "output already exists: " + ", ".join(str(path) for path in collisions)
        )

    rows = read_jsonl(active_path)
    pool_manifest = json.loads(
        Path(pool_manifest_path).read_text(encoding="utf-8")
    )
    verify_active_handoff(rows, pool_manifest, active_path)
    candidates, selected = select_smoke_rows(
        rows,
        seed=seed,
        target_pass_rate=target_pass_rate,
        count=count,
    )
    write_jsonl(selected, output_data_path)
    manifest = build_smoke_manifest(
        rows=rows,
        candidates=candidates,
        selected=selected,
        active_path=active_path,
        pool_manifest_path=pool_manifest_path,
        output_data_path=output_data_path,
        seed=seed,
        target_pass_rate=target_pass_rate,
        created_at_utc=created_at_utc,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-data", default=DEFAULT_ACTIVE_DATA)
    parser.add_argument("--pool-manifest", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--output-data", default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-pass-rate", type=float, default=DEFAULT_TARGET_PASS_RATE)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_smoke_artifacts(
        active_path=args.active_data,
        pool_manifest_path=args.pool_manifest,
        output_data_path=args.output_data,
        output_manifest_path=args.output_manifest,
        seed=args.seed,
        target_pass_rate=args.target_pass_rate,
        count=args.rows,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        overwrite=args.overwrite,
    )
    print(
        "GRPO smoke fixture complete: "
        f"candidates={manifest['selection']['candidate_rows']}, "
        f"selected={manifest['selection']['selected_rows']}, "
        f"manifest={args.output_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
