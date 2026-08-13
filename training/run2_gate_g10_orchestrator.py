#!/usr/bin/env python3
"""Synthetic manifest-verified streaming orchestration for Gate G10.

This module deliberately exposes only a synthetic entry point. It verifies the
active/full fixture identities before opening the full gzip once, reproduces
its declared lineage, and then invokes the in-memory Gate G10 collector. It
does not publish an artifact.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from training.run2_analysis_orchestrator import run_preflight
from training.run2_gate_g10_collector import collect_and_calculate_gate_g10


VERSION = "grpo-run2-gate-g10-synthetic-orchestrator-v1"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def stream_full_groups_once(records_path: Path) -> list[Mapping[str, Any]]:
    """Read one verified full-training gzip into ordered in-memory groups."""
    groups: list[Mapping[str, Any]] = []
    try:
        with gzip.open(records_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"blank full-training replay record at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid full-training replay JSON at line "
                        f"{line_number}: {exc}"
                    ) from exc
                groups.append(
                    _mapping(value, f"full-training replay line {line_number}")
                )
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"full-training replay gzip could not be decoded: {exc}"
        ) from exc
    return groups


def run_synthetic_gate_g10_stream(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    contract_path: str | Path,
    class_weights_path: str | Path,
    full_manifest_path: str | Path,
    full_records_path: str | Path,
) -> dict[str, Any]:
    """Verify, stream and calculate one synthetic dual-scope Gate G10 fixture."""
    root = Path(repo_root).resolve()
    preflight = run_preflight(
        repo_root=root,
        manifest_path=manifest_path,
        records_path=records_path,
        contract_path=contract_path,
        class_weights_path=class_weights_path,
        full_manifest_path=full_manifest_path,
        full_records_path=full_records_path,
        test_mode=True,
    )
    if preflight.get("status") != "synthetic_preflight_passed":
        raise RuntimeError("synthetic Gate G10 preflight did not pass")
    if preflight.get("mode") != "synthetic_dual_replay":
        raise RuntimeError("synthetic Gate G10 preflight mode drifted")
    full_scope = _mapping(
        preflight.get("full_training_replay"),
        "synthetic preflight full-training replay",
    )
    boundary = _mapping(
        preflight.get("selection_boundary"),
        "synthetic preflight selection boundary",
    )
    for key in (
        "replay_gzip_decompressed",
        "replay_records_parsed",
        "full_replay_gzip_decompressed",
        "full_replay_records_parsed",
        "candidate_aggregate_metrics_calculated",
        "gate_g10_calculated",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"synthetic preflight boundary {key} drifted")

    full_records = _resolve(root, full_records_path)
    groups = stream_full_groups_once(full_records)
    expected_groups = full_scope.get("groups")
    if not isinstance(expected_groups, int) or expected_groups <= 0:
        raise RuntimeError("synthetic preflight full-training denominator is invalid")
    collected = collect_and_calculate_gate_g10(
        groups,
        expected_groups=expected_groups,
    )
    lineage = _mapping(collected.get("lineage"), "Gate G10 collector lineage")
    expected_lineage = {
        "groups": full_scope.get("groups"),
        "completions": full_scope.get("completions"),
        "ordered_sku_sha256": full_scope.get("ordered_sku_sha256"),
        "ordered_rollout_key_sha256": full_scope.get(
            "ordered_rollout_key_sha256"
        ),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(
                "streamed full-training lineage differs from verified manifest: "
                f"{key}"
            )

    return {
        "version": VERSION,
        "status": "synthetic_manifest_verified_gate_g10_completed",
        "mode": "synthetic_fixture_only",
        "preflight": preflight,
        "lineage": {
            **expected_lineage,
            "physical_records_identity_verified_before_gzip_open": True,
            "full_gzip_open_count": 1,
            "groups_streamed_once": True,
            "manifest_lineage_matches_stream": True,
            "all_candidates_share_ordered_denominator": lineage[
                "all_candidates_share_ordered_denominator"
            ],
        },
        "candidate_results": collected["candidate_results"],
        "boundary": {
            "synthetic_fixture_only": True,
            "active_replay_gzip_opened": False,
            "full_replay_gzip_opened_once": True,
            "real_full_training_replay_opened": False,
            "real_gate_g10_calculated": False,
            "active_candidate_aggregates_calculated": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
        },
    }
