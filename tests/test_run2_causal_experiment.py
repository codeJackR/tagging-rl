from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.run2_causal_experiment import (
    CausalCheckpointMonitorCoordinator,
    CausalQualityAbort,
    CHECKPOINT_STEPS,
    MINIMUM_MONITOR_DRIVER_FREE_BYTES,
    PRACTICAL_MARGINS,
    QualityBreachTracker,
    _arm_diff,
    arm_spec,
    build_contract,
    build_quality_policy,
    common_training_config,
    quality_observations,
    run_construction,
)
from training.run2_checkpoint_monitor_control import CheckpointMonitorCoordinator


def _baseline_report(mean: float = 0.8, std: float = 0.01):
    views = {}
    for view, metric, _direction, _margin in PRACTICAL_MARGINS:
        views.setdefault(
            view,
            {"greedy": {"scalars": {}}, "sampled": {"aggregate": {}}},
        )
        views[view]["greedy"]["scalars"][metric] = mean + 0.01
        views[view]["sampled"]["aggregate"][metric] = {
            "mean": mean,
            "population_stddev": std,
            "minimum": mean - std,
            "maximum": mean + std,
            "values": [mean] * 8,
        }
    return {
        "status": "checkpoint_outputs_scored",
        "rows": 360,
        "sampled_repetitions": 8,
        "views": views,
    }


def _checkpoint_report(policy, *, breached_keys=()):
    views = {}
    for guardrail in policy["guardrails"]:
        view = views.setdefault(
            guardrail["view"],
            {"greedy": {"scalars": {}}, "sampled": {"aggregate": {}}},
        )
        greedy_key = f"{guardrail['key']}:greedy"
        sampled_key = f"{guardrail['key']}:sampled_mean"
        greedy_threshold = guardrail["thresholds"]["greedy"]
        sampled_threshold = guardrail["thresholds"]["sampled_mean"]
        greedy = (
            greedy_threshold + 0.01
            if guardrail["direction"] == "lower"
            else greedy_threshold - 0.01
        )
        sampled = (
            sampled_threshold + 0.01
            if guardrail["direction"] == "lower"
            else sampled_threshold - 0.01
        )
        if greedy_key in breached_keys:
            greedy = (
                greedy_threshold - 0.01
                if guardrail["direction"] == "lower"
                else greedy_threshold + 0.01
            )
        if sampled_key in breached_keys:
            sampled = (
                sampled_threshold - 0.01
                if guardrail["direction"] == "lower"
                else sampled_threshold + 0.01
            )
        view["greedy"]["scalars"][guardrail["metric"]] = greedy
        view["sampled"]["aggregate"][guardrail["metric"]] = {"mean": sampled}
    return {"status": "checkpoint_outputs_scored", "views": views}


def test_arms_differ_only_in_reward_and_bookkeeping_paths():
    arm_a = arm_spec("A")
    arm_b = arm_spec("B")
    difference = _arm_diff(arm_a, arm_b)
    assert set(difference["trainer_config_differences"]) == {"output_dir", "run_name"}
    assert difference["all_other_trainer_settings_equal"] is True
    assert difference["beta_equal_and_explicitly_zero"] is True
    assert arm_a["reward"]["weights"] == [1.0, 1.0, 2.0]
    assert arm_b["reward"]["functions"] == ["candidate_ua_reward"]
    assert arm_a["quality_root"] != arm_b["quality_root"]


def test_common_config_locks_duration_optimizer_generation_and_disk_policy():
    config = common_training_config(output_dir="runs/test", run_name="test")
    assert config["max_steps"] == 300
    assert config["num_generations"] == 8
    assert config["shuffle_dataset"] is False
    assert config["seed"] == config["data_seed"] == 42
    assert config["beta"] == 0.0
    assert config["learning_rate"] == 5e-6
    assert config["warmup_ratio"] == 0.1
    assert config["save_steps"] == 100
    assert config["save_total_limit"] == 2
    assert config["save_only_model"] is True


def test_quality_threshold_uses_larger_of_variability_or_practical_margin():
    policy = build_quality_policy(_baseline_report(mean=0.8, std=0.01))
    assert policy["checkpoints"] == list(CHECKPOINT_STEPS)
    macro = next(
        item
        for item in policy["guardrails"]
        if item["key"] == "representative_all:macro_f1"
    )
    assert macro["variability_allowance"] == 0.05
    assert macro["thresholds"] == {"greedy": 0.76, "sampled_mean": 0.75}
    rules = next(
        item
        for item in policy["guardrails"]
        if item["key"] == "representative_all:rule_violation_rate"
    )
    assert rules["thresholds"] == {"greedy": 0.83, "sampled_mean": 0.82}
    variable = build_quality_policy(_baseline_report(mean=0.8, std=0.04))
    macro_variable = next(
        item
        for item in variable["guardrails"]
        if item["key"] == "representative_all:macro_f1"
    )
    assert macro_variable["variability_allowance"] == 0.08
    assert macro_variable["thresholds"] == {
        "greedy": 0.73,
        "sampled_mean": 0.72,
    }


def test_same_mode_must_breach_at_two_consecutive_checkpoints_to_abort():
    policy = build_quality_policy(_baseline_report())
    key = "representative_all:macro_f1:greedy"
    tracker = QualityBreachTracker(policy)
    first = tracker.observe(step=100, report=_checkpoint_report(policy, breached_keys={key}))
    assert first["status"] == "warn"
    assert first["abort_training"] is False
    second = tracker.observe(step=200, report=_checkpoint_report(policy, breached_keys={key}))
    assert second["status"] == "abort"
    assert second["repeated_consecutive_breach_keys"] == [key]


def test_clean_checkpoint_resets_quality_breach_sequence():
    policy = build_quality_policy(_baseline_report())
    key = "representative_all:macro_f1:sampled_mean"
    tracker = QualityBreachTracker(policy)
    assert tracker.observe(
        step=100, report=_checkpoint_report(policy, breached_keys={key})
    )["status"] == "warn"
    assert tracker.observe(step=200, report=_checkpoint_report(policy))["status"] == "pass"
    final = tracker.observe(
        step=300, report=_checkpoint_report(policy, breached_keys={key})
    )
    assert final["status"] == "warn"
    assert final["abort_training"] is False


def test_greedy_and_sampled_modes_are_tracked_separately():
    policy = build_quality_policy(_baseline_report())
    greedy = "representative_all:macro_f1:greedy"
    sampled = "representative_all:macro_f1:sampled_mean"
    tracker = QualityBreachTracker(policy)
    tracker.observe(step=100, report=_checkpoint_report(policy, breached_keys={greedy}))
    result = tracker.observe(
        step=200, report=_checkpoint_report(policy, breached_keys={sampled})
    )
    assert result["status"] == "warn"
    assert result["abort_training"] is False


def test_report_must_be_complete_and_checkpoint_order_is_fixed():
    policy = build_quality_policy(_baseline_report())
    with pytest.raises(ValueError, match="completely scored"):
        quality_observations({"status": "partial"}, policy)
    tracker = QualityBreachTracker(policy)
    with pytest.raises(RuntimeError, match="expected step 100, found 200"):
        tracker.observe(step=200, report=_checkpoint_report(policy))


def test_driver_headroom_constant_is_six_gib():
    assert MINIMUM_MONITOR_DRIVER_FREE_BYTES == 6 * 1024**3


def test_published_causal_contract_rebuilds_from_its_pinned_code_commit():
    """The active contract must rebuild byte-for-byte from the commit it pins.

    v3 is active, after the quality-policy amendment (v2) and the rollout
    collector parameterisation (v3). Superseded contracts deliberately no
    longer rebuild: the execution files they pinned have since changed, which
    is exactly the drift this check exists to detect. See
    W2_GRPO_RUN2_POLICY_AMENDMENT.md.
    """
    root = Path(__file__).resolve().parents[1]
    # Newest first. Each amendment issues a new contract and retires the last,
    # so only the active one is expected to rebuild from the commit it pins.
    for candidate in (
        "runs/grpo-run2-causal-experiment-contract-v3.json",
        "runs/grpo-run2-causal-experiment-contract-v2.json",
        "runs/grpo-run2-causal-experiment-contract.json",
    ):
        path = root / candidate
        if path.exists():
            break
    if not path.exists():
        pytest.skip("causal experiment contract has not been published")
    published = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build_contract(
        root=path.parents[1],
        expected_code_commit=published["expected_execution_code_commit"],
    )
    assert published == rebuilt
    assert published["resources"]["monitor_peak_allocated_bytes"] == 3_918_550_016
    assert published["resources"]["conservative_additive_peak_bytes"] == 8_833_412_096


def _base_monitor(tmp_path: Path):
    contract = tmp_path / "phase-f-contract.json"
    contract.write_text(
        json.dumps(
            {
                "checkpoints": {"required_steps": [100, 200, 300]},
                "runtime": {"timeout_seconds_per_checkpoint": 60},
                "abort_policy": {"quality_abort_enabled": False},
            }
        ),
        encoding="utf-8",
    )
    base = CheckpointMonitorCoordinator(
        repo_root=tmp_path,
        contract_path=contract,
        monitor_root=tmp_path / "monitor",
        command_builder=lambda *_: ["true"],
        runner=lambda **_: {},
    )
    base.monitor_root.mkdir()
    return base


def test_causal_wrapper_checks_gpu_headroom_before_monitor_dispatch(tmp_path):
    base = _base_monitor(tmp_path)
    called = []
    base.on_save = lambda *args: called.append(True)
    wrapper = CausalCheckpointMonitorCoordinator(
        base=base,
        quality_policy=build_quality_policy(_baseline_report()),
        quality_output_root=tmp_path / "quality",
        gpu_free_bytes_fn=lambda: MINIMUM_MONITOR_DRIVER_FREE_BYTES - 1,
    )
    wrapper.quality_output_root.mkdir()
    with pytest.raises(CausalQualityAbort, match="insufficient GPU headroom"):
        wrapper.on_save(SimpleNamespace(), SimpleNamespace(global_step=100))
    assert called == []
    evidence = json.loads(
        (tmp_path / "quality/checkpoint-100.resource-block.json").read_text()
    )
    assert evidence["training_must_abort"] is True


def test_causal_wrapper_publishes_decision_and_aborts_repeated_breach(tmp_path):
    base = _base_monitor(tmp_path)
    policy = build_quality_policy(_baseline_report())
    key = "representative_all:macro_f1:greedy"
    step = {"value": 100}

    def accepted(*_args):
        output = base.monitor_root / f"checkpoint-{step['value']}"
        output.mkdir()
        (output / "report.json").write_text(
            json.dumps(_checkpoint_report(policy, breached_keys={key})),
            encoding="utf-8",
        )
        return {"status": "checkpoint_monitor_accepted"}

    base.on_save = accepted
    wrapper = CausalCheckpointMonitorCoordinator(
        base=base,
        quality_policy=policy,
        quality_output_root=tmp_path / "quality",
        gpu_free_bytes_fn=lambda: MINIMUM_MONITOR_DRIVER_FREE_BYTES,
    )
    wrapper.quality_output_root.mkdir()
    first = wrapper.on_save(SimpleNamespace(), SimpleNamespace(global_step=100))
    assert first["quality_decision"]["status"] == "warn"
    step["value"] = 200
    with pytest.raises(CausalQualityAbort, match="repeated checkpoint quality breach"):
        wrapper.on_save(SimpleNamespace(), SimpleNamespace(global_step=200))
    decision = json.loads(
        (tmp_path / "quality/checkpoint-200.quality.json").read_text()
    )
    assert decision["abort_training"] is True
    assert decision["repeated_consecutive_breach_keys"] == [key]


def test_read_only_construction_binds_both_configs_and_rewards(tmp_path):
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "runs/grpo-run2-causal-experiment-contract.json"
    if not contract_path.exists():
        pytest.skip("causal contract has not been published")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    from training.run2_causal_experiment import _identity

    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed_read_only_no_training_dispatch",
                "contract": _identity(contract_path, root),
                "model_loaded": False,
                "trainer_constructed": False,
                "training_dispatched": False,
            }
        ),
        encoding="utf-8",
    )

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.report_to = [] if kwargs.get("report_to") == "none" else kwargs.get("report_to")
            self.generation_batch_size = 8

    report = run_construction(
        root=root,
        contract_path=contract_path,
        preflight_path=preflight,
        config_class=FakeConfig,
    )
    assert report["status"] == "both_arm_configs_constructed_no_trainer_no_dispatch"
    assert report["arms"]["A"]["reward_callable_names"] == [
        "format_validity_reward",
        "vocab_rule_compliance_reward",
        "golden_agreement_reward",
    ]
    assert report["arms"]["B"]["reward_callable_names"] == ["candidate_ua_reward"]
    assert report["trainer_constructed"] is False
    assert report["cuda_context"]["material_cuda_allocation"] is False


# --- the guardrail split, added after Arm A's first production attempt --------


def _policy_for_tracker():
    """A minimal policy carrying one quality and one compliance guardrail."""
    return {
        "checkpoints": [100, 200, 300],
        "guardrails": [
            {
                "key": "representative_all:macro_f1",
                "view": "representative_all",
                "metric": "macro_f1",
                "direction": "lower",
                "thresholds": {"greedy": 0.80, "sampled_mean": 0.80},
            },
            {
                "key": "representative_all:rule_violation_rate",
                "view": "representative_all",
                "metric": "rule_violation_rate",
                "direction": "upper",
                "thresholds": {"greedy": 0.05, "sampled_mean": 0.05},
            },
        ],
    }


def _report(*, macro_f1, rule_rate):
    scalars = {"macro_f1": macro_f1, "rule_violation_rate": rule_rate}
    return {
        "status": "checkpoint_outputs_scored",
        "views": {
            "representative_all": {
                "greedy": {"scalars": dict(scalars)},
                "sampled": {"aggregate": {k: {"mean": v} for k, v in scalars.items()}},
            }
        }
    }


def test_repeated_compliance_breach_warns_but_does_not_abort():
    """Arm A's control run was killed for exhibiting the degradation it exists
    to demonstrate. Compliance metrics are now recorded, never terminal."""
    from training.run2_causal_experiment import QualityBreachTracker

    tracker = QualityBreachTracker(_policy_for_tracker())
    bad_rules = _report(macro_f1=0.86, rule_rate=0.09)
    first = tracker.observe(step=100, report=bad_rules)
    second = tracker.observe(step=200, report=bad_rules)

    assert first["status"] == "warn" and first["abort_training"] is False
    assert second["abort_training"] is False, "compliance must not end a run"
    assert "representative_all:rule_violation_rate:greedy" in second["breached_keys"]
    assert second["recorded_only_breached_keys"], "the breach must still be recorded"
    assert not second["abortable_breached_keys"]


def test_repeated_quality_breach_still_aborts():
    """The protection that matters is unchanged: a run whose macro-F1 collapses
    at two consecutive checkpoints is still stopped."""
    from training.run2_causal_experiment import QualityBreachTracker

    tracker = QualityBreachTracker(_policy_for_tracker())
    collapsed = _report(macro_f1=0.40, rule_rate=0.01)
    tracker.observe(step=100, report=collapsed)
    second = tracker.observe(step=200, report=collapsed)

    assert second["abort_training"] is True
    assert second["repeated_consecutive_breach_keys"]
    assert "representative_all:macro_f1:greedy" in second["abortable_breached_keys"]


def test_a_single_quality_breach_warns_without_aborting():
    from training.run2_causal_experiment import QualityBreachTracker

    tracker = QualityBreachTracker(_policy_for_tracker())
    decision = tracker.observe(step=100, report=_report(macro_f1=0.40, rule_rate=0.01))
    assert decision["status"] == "warn" and decision["abort_training"] is False


def test_compliance_breaches_do_not_mask_a_later_quality_abort():
    """A run breaching compliance throughout must still be stoppable on quality."""
    from training.run2_causal_experiment import QualityBreachTracker

    tracker = QualityBreachTracker(_policy_for_tracker())
    tracker.observe(step=100, report=_report(macro_f1=0.86, rule_rate=0.09))
    tracker.observe(step=200, report=_report(macro_f1=0.40, rule_rate=0.09))
    third = tracker.observe(step=300, report=_report(macro_f1=0.40, rule_rate=0.09))
    assert third["abort_training"] is True
