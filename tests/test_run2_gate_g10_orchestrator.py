from __future__ import annotations

import gzip
import json
from copy import deepcopy

import pytest

import training.run2_gate_g10_collector as collector
import training.run2_gate_g10_orchestrator as orchestrator
from tests.test_run2_analysis_orchestrator import (
    _add_full_scope_fixture,
    _fixture,
    _write_json,
)
from training.audit_data_boundaries import sha256_file
from training.run2_gate_g10_orchestrator import (
    VERSION,
    run_synthetic_gate_g10_stream,
)


def _files(tmp_path):
    return _add_full_scope_fixture(tmp_path, _fixture(tmp_path))


def _run(tmp_path, files):
    return run_synthetic_gate_g10_stream(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        full_manifest_path=files["full_manifest"],
        full_records_path=files["full_records"],
    )


def _read_groups(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_groups(path, groups):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for group in groups:
                handle.write(
                    (json.dumps(group, sort_keys=True) + "\n").encode("utf-8")
                )


def _refresh_physical_identity(files):
    manifest = json.loads(files["full_manifest"].read_text(encoding="utf-8"))
    manifest["output"]["bytes"] = files["full_records"].stat().st_size
    manifest["output"]["sha256"] = sha256_file(files["full_records"])
    _write_json(files["full_manifest"], manifest)


def test_synthetic_orchestrator_preflights_then_opens_full_gzip_once(
    tmp_path, monkeypatch
):
    files = _files(tmp_path)
    real_gzip_open = orchestrator.gzip.open
    calls = []

    def counting_open(*args, **kwargs):
        calls.append(args[0])
        return real_gzip_open(*args, **kwargs)

    monkeypatch.setattr(orchestrator.gzip, "open", counting_open)
    result = _run(tmp_path, files)

    assert result["version"] == VERSION
    assert result["status"] == "synthetic_manifest_verified_gate_g10_completed"
    assert result["mode"] == "synthetic_fixture_only"
    assert calls == [files["full_records"].resolve()]
    assert result["lineage"]["groups"] == 3
    assert result["lineage"]["completions"] == 24
    assert result["lineage"]["full_gzip_open_count"] == 1
    assert result["lineage"]["manifest_lineage_matches_stream"] is True
    assert result["lineage"]["all_candidates_share_ordered_denominator"] is True
    assert set(result["candidate_results"]) == {"U", "UA", "CB"}
    assert result["boundary"] == {
        "synthetic_fixture_only": True,
        "active_replay_gzip_opened": False,
        "full_replay_gzip_opened_once": True,
        "real_full_training_replay_opened": False,
        "real_gate_g10_calculated": False,
        "active_candidate_aggregates_calculated": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
        "artifact_published": False,
    }


def test_identity_failure_happens_before_gzip_open_or_collection(
    tmp_path, monkeypatch
):
    files = _files(tmp_path)
    files["full_records"].write_bytes(files["full_records"].read_bytes() + b"tamper")

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight failure must stop before gzip or collection")

    monkeypatch.setattr(orchestrator.gzip, "open", forbidden)
    monkeypatch.setattr(orchestrator, "collect_and_calculate_gate_g10", forbidden)
    with pytest.raises(ValueError, match="byte count mismatch"):
        _run(tmp_path, files)


def test_validly_hashed_position_drift_fails_before_calculation(tmp_path, monkeypatch):
    files = _files(tmp_path)
    groups = _read_groups(files["full_records"])
    groups[1]["group_position"] = 2
    _write_groups(files["full_records"], groups)
    _refresh_physical_identity(files)

    def forbidden(*args, **kwargs):
        raise AssertionError("calculator must not run after stream-order failure")

    monkeypatch.setattr(collector, "calculate_gate_g10", forbidden)
    with pytest.raises(ValueError, match="contiguous from zero"):
        _run(tmp_path, files)


def test_streamed_order_hash_drift_rejected_after_single_collection(tmp_path):
    files = _files(tmp_path)
    groups = _read_groups(files["full_records"])
    reordered = [deepcopy(groups[index]) for index in (1, 0, 2)]
    for position, group in enumerate(reordered):
        group["group_position"] = position
    _write_groups(files["full_records"], reordered)
    _refresh_physical_identity(files)

    with pytest.raises(ValueError, match="ordered_sku_sha256"):
        _run(tmp_path, files)


def test_invalid_json_fails_without_result_or_artifact(tmp_path):
    files = _files(tmp_path)
    with files["full_records"].open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(b'{"incomplete":\n')
    _refresh_physical_identity(files)
    before = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match="invalid full-training replay JSON"):
        _run(tmp_path, files)
    assert set(tmp_path.iterdir()) == before
