"""CPU-only tests for bounded GRPO checkpoint retention and publication."""

from __future__ import annotations

import copy

import pytest

from training.grpo_full_run_artifacts import (
    EXPECTED_ADAPTER_FILES,
    MAX_FINAL_BUNDLE_BYTES,
    STEP_100_EXPORT_FILES,
    build_full_run_lifecycle_plan,
    validate_full_run_lifecycle_events,
)

STARTING_SHA = "a" * 64
STEP_100_SHA = "b" * 64
FINAL_SHA = "c" * 64


def make_plan(tmp_path):
    final = tmp_path / "grpo-first-300"
    staging = tmp_path / ".grpo-first-300.staging-test"
    return build_full_run_lifecycle_plan(
        final_output_dir=final,
        staging_dir=staging,
    )


def make_events(plan):
    checkpoints = plan["checkpoint_paths"]
    return [
        {
            "event": "checkpoint_saved_100",
            "step": 100,
            "path": checkpoints["100"],
            "save_only_model": True,
        },
        {
            "event": "milestone_exported_100",
            "step": 100,
            "path": plan["step_100_export"]["directory"],
            "files": sorted(STEP_100_EXPORT_FILES),
            "rollout_records": 800,
            "trainer_step_logs": 100,
            "adapter_model_sha256": STEP_100_SHA,
            "adapter_config_sha256": "d" * 64,
        },
        {
            "event": "checkpoint_saved_200",
            "step": 200,
            "path": checkpoints["200"],
            "save_only_model": True,
        },
        {
            "event": "checkpoint_saved_300",
            "step": 300,
            "path": checkpoints["300"],
            "save_only_model": True,
        },
        {
            "event": "checkpoint_evicted_100",
            "step": 100,
            "path": checkpoints["100"],
            "milestone_export_verified": True,
        },
        {
            "event": "retention_verified",
            "retained_steps": [200, 300],
            "absent_steps": [100],
        },
        {
            "event": "final_adapter_saved",
            "step": 300,
            "path": plan["final_adapter_dir"],
        },
        {
            "event": "final_adapter_validated",
            "step": 300,
            "path": plan["final_adapter_dir"],
            "files": sorted(EXPECTED_ADAPTER_FILES),
            "adapter_model_sha256": FINAL_SHA,
            "contains_optimizer_state": False,
            "contains_full_model": False,
        },
        {
            "event": "bundle_validated",
            "step": 300,
            "path": plan["staging_dir"],
            "rollout_records": 2_400,
            "trainer_step_logs": 300,
            "total_bytes": 400 * 1024**2,
            "disk_free_after_bytes": 4 * 1024**3,
        },
        {
            "event": "bundle_published",
            "step": 300,
            "source": plan["staging_dir"],
            "path": plan["final_output_dir"],
            "atomic": True,
        },
    ]


def test_lifecycle_plan_binds_all_outputs_to_atomic_staging_root(tmp_path):
    plan = make_plan(tmp_path)

    assert plan["status"] == "planned_not_executed"
    assert plan["checkpoint_policy"]["retained_steps"] == [200, 300]
    assert plan["checkpoint_policy"]["evicted_steps"] == [100]
    assert plan["step_100_export"]["rollout_records"] == 800
    assert plan["root_evidence"]["rollout_records"] == 2_400
    assert plan["publication"]["maximum_bundle_bytes"] == 512 * 1024**2

    with pytest.raises(ValueError, match="not bound"):
        build_full_run_lifecycle_plan(
            final_output_dir=tmp_path / "grpo-first-300",
            staging_dir=tmp_path / "unrelated-staging",
        )


def test_complete_lifecycle_preserves_step_100_and_publishes_atomically(tmp_path):
    plan = make_plan(tmp_path)
    report = validate_full_run_lifecycle_events(
        make_events(plan),
        plan=plan,
        starting_adapter_sha256=STARTING_SHA,
    )

    assert report["status"] == "passed"
    assert report["events"] == 10
    assert report["step_100_exported_before_eviction"]
    assert report["retained_checkpoint_steps"] == [200, 300]
    assert report["final_adapter_sha256"] == FINAL_SHA
    assert report["published_atomically"]


def test_lifecycle_rejects_missing_or_late_step_100_export(tmp_path):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    del events[1]
    with pytest.raises(ValueError, match="event sequence drifted"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )

    events = make_events(plan)
    events[1], events[4] = events[4], events[1]
    with pytest.raises(ValueError, match="event sequence drifted"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rollout_records", 799, "rollout count"),
        ("trainer_step_logs", 99, "trainer-log count"),
        ("adapter_model_sha256", STARTING_SHA, "invalid or unchanged"),
    ],
)
def test_lifecycle_rejects_incomplete_step_100_evidence(
    tmp_path, field, value, message
):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    events[1][field] = value
    with pytest.raises(ValueError, match=message):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )


def test_lifecycle_rejects_checkpoint_retention_drift(tmp_path):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    events[5]["retained_steps"] = [100, 300]
    with pytest.raises(ValueError, match="retained checkpoint set"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )


def test_lifecycle_rejects_unchanged_or_unsafe_final_adapter(tmp_path):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    events[7]["adapter_model_sha256"] = STEP_100_SHA
    with pytest.raises(ValueError, match="invalid or unchanged"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )

    events = make_events(plan)
    events[7]["contains_optimizer_state"] = True
    with pytest.raises(ValueError, match="optimizer state"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_bytes", MAX_FINAL_BUNDLE_BYTES + 1, "size bound"),
        ("disk_free_after_bytes", 3 * 1024**3 - 1, "insufficient free disk"),
        ("rollout_records", 2_399, "rollout count"),
        ("trainer_step_logs", 299, "trainer-log count"),
    ],
)
def test_lifecycle_rejects_incomplete_or_oversized_bundle(
    tmp_path, field, value, message
):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    events[8][field] = value
    with pytest.raises(ValueError, match=message):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )


def test_lifecycle_rejects_nonatomic_publication(tmp_path):
    plan = make_plan(tmp_path)
    events = make_events(plan)
    events[9]["atomic"] = False
    with pytest.raises(ValueError, match="not published atomically"):
        validate_full_run_lifecycle_events(
            events, plan=plan, starting_adapter_sha256=STARTING_SHA
        )
