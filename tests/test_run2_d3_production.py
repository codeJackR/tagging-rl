from __future__ import annotations

import json

import pytest

import training.run2_d3_production as launcher
from tests.test_run2_d3_result_contract import _artifact, _preflight
from training.run2_d3_production import (
    DEFAULT_ACTIVE_MANIFEST,
    DEFAULT_ACTIVE_RECORDS,
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_COMPARISON_CONTRACT,
    VERSION,
    main,
    parse_args,
    run_production_d3_execution,
    run_production_d3_preflight_only,
)
from training.run2_d3_result_contract import DEFAULT_OUTPUT


def test_preflight_uses_only_locked_active_inputs_and_stops(tmp_path, monkeypatch):
    calls = []

    def fake_preflight(**kwargs):
        calls.append(kwargs)
        return _preflight()

    monkeypatch.setattr(launcher, "run_active_preflight", fake_preflight)
    report = run_production_d3_preflight_only(repo_root=tmp_path)

    assert calls == [
        {
            "repo_root": tmp_path.resolve(),
            "manifest_path": DEFAULT_ACTIVE_MANIFEST,
            "records_path": DEFAULT_ACTIVE_RECORDS,
            "contract_path": DEFAULT_COMPARISON_CONTRACT,
            "class_weights_path": DEFAULT_CLASS_WEIGHTS,
            "test_mode": False,
        }
    ]
    assert report["version"] == VERSION
    assert report["status"] == "production_d3_preflight_only_passed"
    assert report["locked_lineage"]["groups"] == 1_438
    assert report["locked_lineage"]["completions"] == 11_504
    assert report["locked_lineage"]["ordered_sku_hash_matches_contract"] is True
    assert report["locked_lineage"]["ordered_rollout_hash_matches_contract"] is True
    assert report["locked_output"] == {
        "path": DEFAULT_OUTPUT,
        "exists_before_preflight": False,
        "exists_after_preflight": False,
        "complete_contract_validation_before_publication": True,
        "exclusive_atomic_publication_required": True,
    }
    assert all(value is False for value in report["selection_boundary"].values())
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_existing_or_racing_result_stops_preflight_without_overwrite(
    tmp_path, monkeypatch
):
    output = tmp_path / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True)
    output.write_text("existing evidence\n", encoding="utf-8")

    def forbidden(**kwargs):
        raise AssertionError("source preflight must not run after collision")

    monkeypatch.setattr(launcher, "run_active_preflight", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        run_production_d3_preflight_only(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "existing evidence\n"

    output.unlink()

    def racing_preflight(**kwargs):
        output.write_text("concurrent evidence\n", encoding="utf-8")
        return _preflight()

    monkeypatch.setattr(launcher, "run_active_preflight", racing_preflight)
    with pytest.raises(FileExistsError, match="appeared during preflight"):
        run_production_d3_preflight_only(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "concurrent evidence\n"


def test_drifted_preflight_cannot_authorize_launcher(tmp_path, monkeypatch):
    bad = _preflight()
    bad["lineage"]["groups"] -= 1
    monkeypatch.setattr(launcher, "run_active_preflight", lambda **kwargs: bad)

    with pytest.raises(ValueError, match="lineage drifted"):
        run_production_d3_preflight_only(repo_root=tmp_path)
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

    def forbidden(**kwargs):
        raise AssertionError("preflight mode must not dispatch execution")

    monkeypatch.setattr(
        launcher,
        "run_production_d3_preflight_only",
        lambda **kwargs: expected,
    )
    monkeypatch.setattr(launcher, "run_production_d3_execution", forbidden)
    assert main(["--repo-root", "/tmp/repo", "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_execute_cli_dispatches_only_after_explicit_flag(monkeypatch, capsys):
    expected = {"status": "synthetic execution dispatch"}

    def forbidden(**kwargs):
        raise AssertionError("execute must not dispatch preflight-only reporting")

    monkeypatch.setattr(launcher, "run_production_d3_preflight_only", forbidden)
    monkeypatch.setattr(
        launcher,
        "run_production_d3_execution",
        lambda **kwargs: expected,
    )
    assert main(["--repo-root", "/tmp/repo", "--execute"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_future_execution_validates_complete_artifact_then_publishes(
    tmp_path, monkeypatch
):
    events = []
    artifact = _artifact()
    real_validate = launcher.validate_production_d3_result
    real_publish = launcher.publish_production_d3_result

    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())

    def fake_build(**kwargs):
        events.append("build")
        assert kwargs == {
            "repo_root": tmp_path.resolve(),
            "manifest_path": DEFAULT_ACTIVE_MANIFEST,
            "records_path": DEFAULT_ACTIVE_RECORDS,
            "contract_path": DEFAULT_COMPARISON_CONTRACT,
            "class_weights_path": DEFAULT_CLASS_WEIGHTS,
            "test_mode": False,
        }
        return artifact

    def traced_validate(value):
        events.append("validate")
        return real_validate(value)

    def traced_publish(**kwargs):
        events.append("publish")
        return real_publish(**kwargs)

    monkeypatch.setattr(launcher, "build_analysis_artifact", fake_build)
    monkeypatch.setattr(launcher, "validate_production_d3_result", traced_validate)
    monkeypatch.setattr(launcher, "publish_production_d3_result", traced_publish)

    report = run_production_d3_execution(repo_root=tmp_path)

    assert events == ["build", "validate", "publish"]
    assert report["mode"] == "explicit_execute_active_replay_d3_only"
    assert report["groups"] == 1_438
    assert report["completions"] == 11_504
    assert report["selection_boundary"]["winner_selected"] is False
    assert (tmp_path / DEFAULT_OUTPUT).is_file()


def test_invalid_built_artifact_never_reaches_publication(tmp_path, monkeypatch):
    invalid = _artifact()
    invalid["selection_boundary"]["winner_selected"] = True
    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())
    monkeypatch.setattr(launcher, "build_analysis_artifact", lambda **kwargs: invalid)

    def forbidden(**kwargs):
        raise AssertionError("invalid artifact must never reach publication")

    monkeypatch.setattr(launcher, "publish_production_d3_result", forbidden)
    with pytest.raises(ValueError, match="selection boundary drifted"):
        run_production_d3_execution(repo_root=tmp_path)
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_execute_preserves_result_that_appears_during_build(tmp_path, monkeypatch):
    output = tmp_path / DEFAULT_OUTPUT
    monkeypatch.setattr(launcher, "_run_locked_preflight", lambda root: _preflight())

    def racing_build(**kwargs):
        output.parent.mkdir(parents=True)
        output.write_text("concurrent evidence\n", encoding="utf-8")
        return _artifact()

    monkeypatch.setattr(launcher, "build_analysis_artifact", racing_build)
    with pytest.raises(FileExistsError, match="output already exists"):
        run_production_d3_execution(repo_root=tmp_path)
    assert output.read_text(encoding="utf-8") == "concurrent evidence\n"
