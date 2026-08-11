from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLOSEOUT = ROOT / "runs" / "grpo-first-300-closeout.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_closeout_all_tracked_artifact_and_code_hashes_match():
    closeout = _load(CLOSEOUT)
    entries = closeout["tracked_artifacts"] + closeout["analysis_code"]
    paths = [entry["path"] for entry in entries]

    assert len(paths) == len(set(paths))
    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]


def test_closeout_verdict_recomputes_from_locked_evidence():
    closeout = _load(CLOSEOUT)
    report = _load(ROOT / "runs/grpo-first-300-frozen-eval-300-report.json")
    bootstrap = _load(ROOT / "runs/grpo-first-300-frozen-eval-300-bootstrap.json")
    replay = _load(ROOT / "runs/grpo-first-300-frozen-eval-300-reward-replay.json")

    macro = bootstrap["bootstrap"]["metrics"]["macro_f1"]
    assert closeout["result"]["grpo"]["macro_f1"] == report["macro_f1"]
    assert closeout["result"]["delta_grpo_minus_sft"]["macro_f1"] == macro[
        "delta_candidate_minus_baseline"
    ]["point"]
    assert closeout["result"]["delta_grpo_minus_sft"][
        "macro_f1_paired_ci_95"
    ] == macro["delta_candidate_minus_baseline"]["ci"]
    assert macro["delta_candidate_minus_baseline"]["ci"][1] < 0.0
    assert closeout["reward_replay"]["delta_weighted_total"] == replay["replay"][
        "delta_candidate_minus_baseline"
    ]["weighted_total_mean"]
    assert closeout["verdict"]["scientific_result"] == "negative_regression"
    assert closeout["verdict"]["selected_model"] == "sft-combined-checkpoint-406"


def test_closeout_external_adapter_and_future_blindness_policy():
    closeout = _load(CLOSEOUT)
    archive = ROOT / closeout["model_lineage"]["external_archive"]["path"]
    adapter = archive / closeout["model_lineage"]["external_archive"][
        "adapter_relative_path"
    ]
    manifest = archive / "manifest.json"

    if archive.exists():
        assert _sha256(manifest) == closeout["model_lineage"]["external_archive"][
            "manifest_sha256"
        ]
        assert _sha256(adapter) == closeout["model_lineage"][
            "grpo_final_adapter_sha256"
        ]

    assert closeout["frozen_evaluation"]["blind_status"].startswith("consumed")
    assert closeout["future_experiment_gate"]["status"] == "new_design_required"
    assert closeout["validation"]["expected_full_cpu_tests"] == 487
