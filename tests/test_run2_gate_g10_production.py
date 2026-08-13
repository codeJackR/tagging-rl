from __future__ import annotations

import json

import pytest

import training.run2_gate_g10_production as launcher
from tests.test_run2_gate_g10_result_contract import _collection, _preflight
from training.run2_gate_g10_production import (
    DEFAULT_ACTIVE_MANIFEST,
    DEFAULT_ACTIVE_RECORDS,
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_COMPARISON_CONTRACT,
    DEFAULT_FULL_MANIFEST,
    DEFAULT_FULL_RECORDS,
    VERSION,
    main,
    parse_args,
    run_production_gate_g10_execution,
    run_production_gate_g10_preflight_only,
)
from training.run2_gate_g10_result_contract import DEFAULT_OUTPUT


def test_preflight_only_uses_locked_production_inputs_and_stops(tmp_path, monkeypatch):
    calls = []

    def fake_preflight(**kwargs):
        calls.append(kwargs)
        return _preflight()

    monkeypatch.setattr(launcher, "run_preflight", fake_preflight)
    report = run_production_gate_g10_preflight_only(repo_root=tmp_path)

    assert calls == [
        {
            "repo_root": tmp_path.resolve(),
            "manifest_path": DEFAULT_ACTIVE_MANIFEST,
            "records_path": DEFAULT_ACTIVE_RECORDS,
            "contract_path": DEFAULT_COMPARISON_CONTRACT,
            "class_weights_path": DEFAULT_CLASS_WEIGHTS,
            "full_manifest_path": DEFAULT_FULL_MANIFEST,
            "full_records_path": DEFAULT_FULL_RECORDS,
            "test_mode": False,
        }
    ]
    assert report["version"] == VERSION
    assert report["status"] == "production_gate_g10_preflight_only_passed"
    assert report["mode"] == "preflight_only_no_execution_path"
    assert report["locked_output"] == {
        "path": DEFAULT_OUTPUT,
        "exists_before_preflight": False,
        "exists_after_preflight": False,
        "exclusive_atomic_publication_required": True,
    }
    assert report["locked_lineage"]["groups"] == 3_240
    assert report["locked_lineage"]["completions"] == 25_920
    assert report["locked_lineage"]["ordered_sku_hash_matches_contract"] is True
    assert (
        report["locked_lineage"]["ordered_rollout_hash_matches_contract"] is True
    )
    assert all(value is False for value in report["selection_boundary"].values())
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_existing_result_stops_before_source_preflight(tmp_path, monkeypatch):
    output = tmp_path / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True)
    output.write_text("user evidence\n", encoding="utf-8")

    def forbidden(**kwargs):
        raise AssertionError("source preflight must not run after output collision")

    monkeypatch.setattr(launcher, "run_preflight", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        run_production_gate_g10_preflight_only(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "user evidence\n"


def test_result_appearing_during_preflight_fails_closed(tmp_path, monkeypatch):
    output = tmp_path / DEFAULT_OUTPUT

    def racing_preflight(**kwargs):
        output.parent.mkdir(parents=True)
        output.write_text("concurrent evidence\n", encoding="utf-8")
        return _preflight()

    monkeypatch.setattr(launcher, "run_preflight", racing_preflight)
    with pytest.raises(FileExistsError, match="appeared during preflight"):
        run_production_gate_g10_preflight_only(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "concurrent evidence\n"


def test_drifted_preflight_cannot_authorize_launcher(tmp_path, monkeypatch):
    bad = _preflight()
    bad["selection_boundary"]["gate_g10_calculated"] = True
    monkeypatch.setattr(launcher, "run_preflight", lambda **kwargs: bad)

    with pytest.raises(ValueError, match="gate_g10_calculated"):
        run_production_gate_g10_preflight_only(repo_root=tmp_path)
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_cli_requires_exactly_one_explicit_mode():
    args = parse_args(["--repo-root", "/tmp/repo", "--preflight-only"])
    assert args.repo_root == "/tmp/repo"
    assert args.preflight_only is True
    assert args.execute is False
    execute = parse_args(["--execute"])
    assert execute.execute is True
    assert execute.preflight_only is False
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--preflight-only", "--execute"])


def test_main_prints_only_preflight_report(monkeypatch, capsys):
    expected = {"status": "fixture preflight only"}
    monkeypatch.setattr(
        launcher,
        "run_production_gate_g10_preflight_only",
        lambda **kwargs: expected,
    )
    assert main(["--repo-root", "/tmp/repo", "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_execute_composes_synthetic_substitutes_and_publishes(tmp_path, monkeypatch):
    calls = []
    sentinel_groups = [object()]

    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())

    def fake_stream(path):
        calls.append(("stream", path))
        return sentinel_groups

    def fake_collect(groups, *, expected_groups):
        calls.append(("collect", groups, expected_groups))
        return _collection()

    monkeypatch.setattr(launcher, "stream_full_groups_once", fake_stream)
    monkeypatch.setattr(launcher, "collect_and_calculate_gate_g10", fake_collect)

    report = run_production_gate_g10_execution(repo_root=tmp_path)
    output = tmp_path / DEFAULT_OUTPUT

    assert calls == [
        ("stream", (tmp_path / DEFAULT_FULL_RECORDS).resolve()),
        ("collect", sentinel_groups, 3_240),
    ]
    assert report["status"] == "production_gate_g10_executed_and_published"
    assert report["mode"] == "explicit_execute_full_training_gate_g10_only"
    assert report["groups"] == 3_240
    assert report["completions"] == 25_920
    assert report["candidate_order"] == ["U", "UA", "CB"]
    assert report["publication"]["path"] == DEFAULT_OUTPUT
    assert report["publication"]["published_exclusively"] is True
    assert output.is_file()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["gate_summary"] == report["gate_summary"]
    assert artifact["selection_boundary"]["winner_selected"] is False


def test_execute_rejects_streamed_lineage_before_build_or_publish(
    tmp_path, monkeypatch
):
    bad = _collection()
    bad["lineage"]["ordered_sku_sha256"] = "0" * 64
    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())
    monkeypatch.setattr(launcher, "stream_full_groups_once", lambda path: [])
    monkeypatch.setattr(
        launcher,
        "collect_and_calculate_gate_g10",
        lambda groups, *, expected_groups: bad,
    )

    with pytest.raises(ValueError, match="ordered_sku_sha256"):
        run_production_gate_g10_execution(repo_root=tmp_path)
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_execute_preserves_result_that_appears_during_stream(tmp_path, monkeypatch):
    output = tmp_path / DEFAULT_OUTPUT
    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())

    def racing_stream(path):
        output.parent.mkdir(parents=True)
        output.write_text("concurrent evidence\n", encoding="utf-8")
        return []

    monkeypatch.setattr(launcher, "stream_full_groups_once", racing_stream)
    monkeypatch.setattr(
        launcher,
        "collect_and_calculate_gate_g10",
        lambda groups, *, expected_groups: _collection(),
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        run_production_gate_g10_execution(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "concurrent evidence\n"


def test_execute_cli_dispatches_only_after_explicit_flag(monkeypatch, capsys):
    expected = {"status": "synthetic execution path"}
    monkeypatch.setattr(
        launcher,
        "run_production_gate_g10_execution",
        lambda **kwargs: expected,
    )
    assert main(["--repo-root", "/tmp/repo", "--execute"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
