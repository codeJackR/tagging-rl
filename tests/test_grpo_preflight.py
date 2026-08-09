from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.train_grpo as train_grpo_module

from training.train_grpo import (
    LOCKED_ADAM_BETAS,
    LOCKED_ADAM_EPSILON,
    LOCKED_BASE_MODEL,
    LOCKED_LEARNING_RATE,
    LOCKED_MAX_GRAD_NORM,
    LOCKED_REWARD_WEIGHTS,
    LOCKED_TARGET_MODULES,
    LOCKED_TRAINABLE_PARAMETERS,
    LOCKED_TRAINABLE_TENSORS,
    LOCKED_WARMUP_RATIO,
    LOCKED_WEIGHT_DECAY,
    build_rollout_evidence,
    build_logged_reward_evidence,
    grpo_smoke_config_kwargs,
    inspect_grpo_config,
    inspect_model_trainability,
    inspect_trainable_parameter_values,
    main,
    parse_args,
    run_preflight,
    validate_optimizer_evidence,
    validate_gradient_evidence,
    validate_initialized_optimizer_evidence,
    validate_parameter_update_evidence,
    validate_five_step_launch_args,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DATA = ROOT / "data" / "train_weak_grpo_smoke_v1.jsonl"
FIXTURE_MANIFEST = ROOT / "data" / "splits" / "grpo-smoke-v1.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sft_lock(tmp_path: Path):
    adapter = tmp_path / "checkpoint-406"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"small fake adapter for CPU preflight")
    weights_sha = sha256_file(weights)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": LOCKED_BASE_MODEL,
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "bias": "none",
                "target_modules": sorted(LOCKED_TARGET_MODULES),
            }
        ),
        encoding="utf-8",
    )
    selection = {
        "status": "locked_before_frozen_eval",
        "selected_checkpoint": {
            "remote_path": str(adapter),
            "base_model": LOCKED_BASE_MODEL,
            "adapter_weights": {
                "file": weights.name,
                "bytes": weights.stat().st_size,
                "sha256": weights_sha,
            },
            "lora": {
                "rank": 16,
                "alpha": 16,
                "target_modules": sorted(LOCKED_TARGET_MODULES),
                "trainable_parameters": LOCKED_TRAINABLE_PARAMETERS,
            },
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return adapter, selection_path, weights_sha


def passing_kwargs(tmp_path: Path) -> dict:
    adapter, selection, weights_sha = make_sft_lock(tmp_path)
    return {
        "repo_root": ROOT,
        "fixture_data": FIXTURE_DATA,
        "fixture_manifest": FIXTURE_MANIFEST,
        "selection_manifest": selection,
        "adapter": adapter,
        "output_dir": tmp_path / "new-output",
        "minimum_free_bytes": 3 * 1024**3,
        "expected_commit": "a" * 40,
        "expected_selection_manifest_sha256": sha256_file(selection),
        "expected_adapter_sha256": weights_sha,
        "git_state_fn": lambda _: {
            "commit": "a" * 40,
            "tracked_worktree_dirty": False,
            "index_dirty": False,
        },
        "disk_usage_fn": lambda _: SimpleNamespace(free=4 * 1024**3),
    }


def test_importing_entrypoint_is_cpu_only_and_training_is_unavailable():
    source_path = ROOT / "training" / "train_grpo.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    # Runtime-only imports may exist inside guarded functions. This assertion
    # covers imports executed merely by importing training.train_grpo.
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"torch", "transformers", "trl", "peft", "unsloth", "vllm"}
    )
    assert parse_args(["--preflight-only"]).preflight_only
    assert parse_args(["--model-load-only"]).model_load_only
    assert parse_args(["--trainer-construction-only"]).trainer_construction_only
    assert parse_args(["--rollout-only"]).rollout_only
    assert parse_args(["--gradient-only"]).gradient_only
    assert parse_args(
        ["--optimizer-construction-only"]
    ).optimizer_construction_only
    assert parse_args(["--one-update-only"]).one_update_only
    assert parse_args(["--five-step-smoke"]).five_step_smoke
    with pytest.raises(SystemExit):
        parse_args(["--preflight-only", "--model-load-only"])
    with pytest.raises(SystemExit):
        parse_args(["--one-update-only", "--five-step-smoke"])
    with pytest.raises(SystemExit, match="training is intentionally unavailable"):
        main([])


def test_rollout_report_file_is_mode_locked_and_collision_safe(tmp_path):
    report_path = tmp_path / "rollout.json"
    with pytest.raises(SystemExit, match="valid only with --rollout-only"):
        main(["--preflight-only", "--report-file", str(report_path)])

    report_path.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="evidence report already exists"):
        main(["--rollout-only", "--report-file", str(report_path)])


def test_five_step_launch_control_rejects_unlocked_arguments(tmp_path):
    valid = [
        "--five-step-smoke",
        "--repo-root",
        str(tmp_path),
        "--expected-commit",
        "a" * 40,
    ]
    report = validate_five_step_launch_args(parse_args(valid))
    assert report["passed"]
    assert report["standalone_report_forbidden"]
    assert report["reserved_output"] == str(
        (tmp_path / "runs" / "grpo-first-smoke").resolve()
    )

    with pytest.raises(SystemExit, match="full lowercase"):
        main(["--five-step-smoke", "--repo-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="full lowercase"):
        main(
            [
                "--five-step-smoke",
                "--repo-root",
                str(tmp_path),
                "--expected-commit",
                "A" * 40,
            ]
        )
    with pytest.raises(SystemExit, match="reserved"):
        main([*valid, "--output-dir", "runs/not-the-smoke"])
    with pytest.raises(SystemExit, match="at least 3 GiB"):
        main([*valid, "--minimum-free-gib", "2.99"])
    with pytest.raises(SystemExit, match="--report-file is forbidden"):
        main([*valid, "--report-file", str(tmp_path / "extra.json")])


def test_five_step_cli_dispatches_only_after_preflight(monkeypatch, tmp_path, capsys):
    observed = {}
    final_output = tmp_path / "runs" / "grpo-first-smoke"

    def fake_preflight(**kwargs):
        observed["preflight"] = kwargs
        return {
            "version": "fake-preflight",
            "status": "passed",
            "fixture": {
                "data_path": str(tmp_path / "fixture.jsonl"),
                "sku_ids_in_step_order": [f"sku-{index}" for index in range(5)],
            },
            "sft_lock": {
                "adapter_path": str(tmp_path / "checkpoint-406"),
                "adapter_file": str(tmp_path / "adapter_model.safetensors"),
                "runtime_trainable_parameter_assertion_required": True,
            },
            "output": {
                "path": str(final_output),
                "collision_free": True,
                "created": False,
            },
            "cuda_imports_performed": False,
            "model_loaded": False,
            "trainer_constructed": False,
        }

    def fake_five_step_gate(**kwargs):
        observed["gate"] = kwargs
        return {
            "version": "grpo-five-step-smoke-gate-v1",
            "status": "passed",
        }

    monkeypatch.setattr(train_grpo_module, "run_preflight", fake_preflight)
    monkeypatch.setattr(
        train_grpo_module,
        "run_five_step_smoke_gate",
        fake_five_step_gate,
    )
    commit = "a" * 40
    exit_code = main(
        [
            "--five-step-smoke",
            "--repo-root",
            str(tmp_path),
            "--expected-commit",
            commit,
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert observed["preflight"]["expected_commit"] == commit
    assert observed["preflight"]["minimum_free_bytes"] == 3 * 1024**3
    assert observed["gate"]["preflight_report"]["status"] == "passed"
    assert observed["gate"]["final_output_dir"] == str(final_output)
    assert printed["training_steps"] == 5
    assert printed["rollout_records"] == 40
    assert printed["output"]["created"] is True
    assert printed["atomic_bundle_published"] is True

    observed.clear()
    monkeypatch.setattr(
        train_grpo_module,
        "run_preflight",
        lambda **_kwargs: {"status": "failed"},
    )
    with pytest.raises(RuntimeError, match="failed preflight"):
        main(
            [
                "--five-step-smoke",
                "--repo-root",
                str(tmp_path),
                "--expected-commit",
                commit,
            ]
        )
    assert "gate" not in observed


def test_rollout_evidence_preserves_components_weights_and_raw_outputs():
    reward_names = ["format", "compliance", "agreement"]
    component_rewards = {
        "format": [1.0] * 8,
        "compliance": [1.0] * 4 + [0.0] * 4,
        "agreement": [1.0, 0.0] * 4,
    }
    report = build_rollout_evidence(
        sku_id="sku-1",
        completions=[f"completion-{index}" for index in range(8)],
        reward_names=reward_names,
        component_rewards=component_rewards,
        reward_weights=LOCKED_REWARD_WEIGHTS,
        advantages=[1.0, -1.0] * 4,
        effective_completion_tokens=list(range(10, 18)),
        truncated_and_masked=[False] * 8,
    )

    assert report["weighted_totals"] == [4.0, 2.0, 4.0, 2.0, 3.0, 1.0, 3.0, 1.0]
    assert report["weighted_total_has_variance"]
    assert report["nonzero_advantage_count"] == 8
    assert report["truncated_and_masked_count"] == 0
    assert report["records"][0]["raw_output"] == "completion-0"
    assert report["records"][0]["component_rewards"]["agreement"] == 1.0


def test_rollout_evidence_rejects_misalignment_and_nonfinite_values():
    kwargs = {
        "sku_id": "sku-1",
        "completions": ["{}"] * 8,
        "reward_names": ["format"],
        "component_rewards": {"format": [1.0] * 8},
        "reward_weights": [1.0],
        "advantages": [0.0] * 8,
        "effective_completion_tokens": [10] * 8,
        "truncated_and_masked": [False] * 8,
    }
    bad_alignment = dict(kwargs, completions=["{}"] * 7)
    with pytest.raises(RuntimeError, match="eight completions"):
        build_rollout_evidence(**bad_alignment)

    bad_rewards = dict(kwargs)
    bad_rewards["component_rewards"] = {"format": [float("nan")] * 8}
    with pytest.raises(RuntimeError, match="non-finite"):
        build_rollout_evidence(**bad_rewards)


def test_logged_update_reward_evidence_preserves_raw_outputs_and_variance():
    report = build_logged_reward_evidence(
        sku_id="sku-1",
        completions=[f"completion-{index}" for index in range(8)],
        reward_names=["format", "compliance", "agreement"],
        component_rewards={
            "format": [1.0] * 8,
            "compliance": [1.0] * 8,
            "agreement": [0.0, 1.0] * 4,
        },
        reward_weights=LOCKED_REWARD_WEIGHTS,
        advantages=[-1.0, 1.0] * 4,
    )

    assert report["weighted_totals"] == [2.0, 4.0] * 4
    assert report["weighted_total_has_variance"]
    assert report["nonzero_advantage_count"] == 8
    assert report["records"][1]["raw_output"] == "completion-1"

    with pytest.raises(RuntimeError, match="exactly eight completions"):
        build_logged_reward_evidence(
            sku_id="sku-1",
            completions=["{}"] * 7,
            reward_names=["format"],
            component_rewards={"format": [1.0] * 8},
            reward_weights=[1.0],
            advantages=[0.0] * 8,
        )


def test_gradient_evidence_requires_complete_finite_nonzero_lora_gradients():
    stats = {
        "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
        "tensors_with_gradient": LOCKED_TRAINABLE_TENSORS,
        "gradient_elements": LOCKED_TRAINABLE_PARAMETERS,
        "nonfinite_gradient_elements": 0,
        "nonzero_gradient_tensors": 300,
        "nonzero_gradient_elements": 1_000_000,
        "global_l2_norm": 0.75,
    }
    report = validate_gradient_evidence(stats)
    assert report["all_trainable_tensors_have_gradients"]
    assert report["all_gradients_finite"]
    assert report["has_nonzero_gradient"]
    assert report["matches_locked_gradient_footprint"]

    with pytest.raises(RuntimeError, match="has no gradient"):
        validate_gradient_evidence(
            {**stats, "tensors_with_gradient": LOCKED_TRAINABLE_TENSORS - 1}
        )
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        validate_gradient_evidence({**stats, "nonfinite_gradient_elements": 1})
    with pytest.raises(RuntimeError, match="all LoRA gradient tensors are zero"):
        validate_gradient_evidence({**stats, "nonzero_gradient_tensors": 0})


def test_optimizer_evidence_requires_exact_lora_scope_and_lazy_state():
    stats = {
        "optimizer_class": "AdamW",
        "optimizer_module": "bitsandbytes.optim.adamw",
        "optimizer_bits": 8,
        "is_paged": False,
        "optimizer_initialized_flag": False,
        "parameter_groups": [
            {
                "parameter_tensors": LOCKED_TRAINABLE_TENSORS,
                "lr": LOCKED_LEARNING_RATE,
                "weight_decay": LOCKED_WEIGHT_DECAY,
                "betas": list(LOCKED_ADAM_BETAS),
                "eps": LOCKED_ADAM_EPSILON,
            },
            {
                "parameter_tensors": 0,
                "lr": LOCKED_LEARNING_RATE,
                "weight_decay": 0.0,
                "betas": list(LOCKED_ADAM_BETAS),
                "eps": LOCKED_ADAM_EPSILON,
            },
        ],
        "trainable_model_tensors": LOCKED_TRAINABLE_TENSORS,
        "trainable_model_elements": LOCKED_TRAINABLE_PARAMETERS,
        "unique_optimizer_parameter_tensors": LOCKED_TRAINABLE_TENSORS,
        "unique_optimizer_parameter_elements": LOCKED_TRAINABLE_PARAMETERS,
        "missing_trainable_tensors": 0,
        "frozen_optimizer_tensors": 0,
        "duplicate_optimizer_references": 0,
        "optimizer_state_entries": 0,
        "optimizer_state_tensor_count": 0,
        "optimizer_state_tensor_bytes": 0,
        "gradients_attached": 0,
        "trainable_lora_unchanged": True,
        "global_step": 0,
    }
    report = validate_optimizer_evidence(stats)
    assert report["controls_exact_locked_lora"]
    assert report["optimizer_state_is_lazy"]
    assert report["no_parameter_update"]

    with pytest.raises(RuntimeError, match="does not control every"):
        validate_optimizer_evidence(
            {
                **stats,
                "unique_optimizer_parameter_tensors": LOCKED_TRAINABLE_TENSORS
                - 1,
            }
        )
    with pytest.raises(RuntimeError, match="frozen tensor"):
        validate_optimizer_evidence({**stats, "frozen_optimizer_tensors": 1})
    with pytest.raises(RuntimeError, match="initialized parameter state"):
        validate_optimizer_evidence({**stats, "optimizer_state_entries": 1})
    bad_groups = [dict(group) for group in stats["parameter_groups"]]
    bad_groups[0]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="learning rate drifted"):
        validate_optimizer_evidence({**stats, "parameter_groups": bad_groups})
    with pytest.raises(RuntimeError, match="LoRA weights changed"):
        validate_optimizer_evidence({**stats, "trainable_lora_unchanged": False})


def test_parameter_update_evidence_requires_complete_finite_lora_change():
    stats = {
        "trainable_tensors": LOCKED_TRAINABLE_TENSORS,
        "trainable_elements": LOCKED_TRAINABLE_PARAMETERS,
        "changed_tensors": LOCKED_TRAINABLE_TENSORS,
        "changed_elements": 18_000_000,
        "nonfinite_after_elements": 0,
        "nonfinite_delta_elements": 0,
        "global_delta_l2_norm": 0.01,
    }
    report = validate_parameter_update_evidence(stats)
    assert report["all_lora_tensors_changed"]
    assert report["all_updated_weights_finite"]
    assert report["has_finite_nonzero_update"]

    with pytest.raises(RuntimeError, match="did not change every"):
        validate_parameter_update_evidence(
            {**stats, "changed_tensors": LOCKED_TRAINABLE_TENSORS - 1}
        )
    with pytest.raises(RuntimeError, match="non-finite LoRA value"):
        validate_parameter_update_evidence(
            {**stats, "nonfinite_after_elements": 1}
        )
    with pytest.raises(RuntimeError, match="not finite and positive"):
        validate_parameter_update_evidence(
            {**stats, "global_delta_l2_norm": 0.0}
        )


def test_initialized_optimizer_evidence_requires_complete_step_one_state():
    stats = {
        "optimizer_class": "AdamW",
        "optimizer_bits": 8,
        "is_paged": False,
        "optimizer_initialized_flag": True,
        "state_parameter_entries": LOCKED_TRAINABLE_TENSORS,
        "missing_trainable_state_entries": 0,
        "foreign_state_entries": 0,
        "state_step_values": [1],
        "state1_elements": LOCKED_TRAINABLE_PARAMETERS,
        "state2_elements": LOCKED_TRAINABLE_PARAMETERS,
        "state1_dtypes": ["torch.uint8"],
        "state2_dtypes": ["torch.uint8"],
        "absmax1_elements": 72_128,
        "absmax2_elements": 72_128,
        "expected_quantization_blocks": 72_128,
        "nonfinite_state_elements": 0,
        "unique_state_tensor_bytes": 37_508_608,
    }
    report = validate_initialized_optimizer_evidence(stats)
    assert report["state_covers_exact_locked_lora"]
    assert report["state_initialized_at_step_one"]
    assert report["state_is_finite"]

    step_five = validate_initialized_optimizer_evidence(
        {**stats, "state_step_values": [5]},
        expected_step=5,
    )
    assert step_five["state_initialized_at_expected_step"]
    assert not step_five["state_initialized_at_step_one"]

    with pytest.raises(RuntimeError, match="exactly at step one"):
        validate_initialized_optimizer_evidence({**stats, "state_step_values": [2]})
    with pytest.raises(RuntimeError, match="not fully 8-bit"):
        validate_initialized_optimizer_evidence(
            {**stats, "state1_dtypes": ["torch.float32"]}
        )
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        validate_initialized_optimizer_evidence(
            {**stats, "nonfinite_state_elements": 1}
        )


def test_grpo_smoke_config_is_complete_and_one_prompt_group_per_step(tmp_path):
    kwargs = grpo_smoke_config_kwargs(output_dir=tmp_path)

    assert kwargs["num_generations"] == 8
    assert kwargs["per_device_train_batch_size"] == 8
    assert kwargs["gradient_accumulation_steps"] == 1
    assert kwargs["steps_per_generation"] == 1
    assert kwargs["max_steps"] == 5
    assert not kwargs["shuffle_dataset"]
    assert kwargs["reward_weights"] == list(LOCKED_REWARD_WEIGHTS)
    assert not kwargs["use_vllm"]
    assert kwargs["beta"] == 0.0
    assert kwargs["save_strategy"] == "no"
    assert kwargs["learning_rate"] == LOCKED_LEARNING_RATE
    assert kwargs["weight_decay"] == LOCKED_WEIGHT_DECAY
    assert (kwargs["adam_beta1"], kwargs["adam_beta2"]) == LOCKED_ADAM_BETAS
    assert kwargs["adam_epsilon"] == LOCKED_ADAM_EPSILON
    assert kwargs["warmup_ratio"] == LOCKED_WARMUP_RATIO == 0.0
    assert kwargs["max_grad_norm"] == LOCKED_MAX_GRAD_NORM == 1.0

    with pytest.raises(ValueError, match="reward weights must remain locked"):
        grpo_smoke_config_kwargs(output_dir=tmp_path, reward_weights=(1, 1, 1))


def test_grpo_config_inspector_accepts_normalization_and_rejects_drift(tmp_path):
    kwargs = grpo_smoke_config_kwargs(output_dir=tmp_path)
    config = SimpleNamespace(**kwargs, generation_batch_size=8)
    config.report_to = []

    report = inspect_grpo_config(config)
    assert report["generation_batch_size"] == 8
    assert report["prompts_per_generation_batch"] == 1
    assert report["settings_match_locked_contract"]

    config.temperature = 1.0
    with pytest.raises(RuntimeError, match="config drift for temperature"):
        inspect_grpo_config(config)


class FakeParameter:
    def __init__(
        self,
        size: int,
        *,
        requires_grad: bool,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
        finite_count: int | None = None,
    ):
        self.size = size
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.device = device
        self.finite_count = size if finite_count is None else finite_count

    def numel(self):
        return self.size

    def detach(self):
        return self

    def isfinite(self):
        return SimpleNamespace(
            sum=lambda: SimpleNamespace(item=lambda: self.finite_count)
        )


class FakeModel:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


def locked_fake_model(*, bad_name: str | None = None, missing_target: bool = False):
    targets = sorted(LOCKED_TARGET_MODULES)
    quotient, remainder = divmod(LOCKED_TRAINABLE_PARAMETERS, len(targets))
    parameters = [
        (
            "model.embed_tokens.weight",
            FakeParameter(1_500_000_000, requires_grad=False),
        )
    ]
    for index, target in enumerate(targets):
        if missing_target and target == targets[-1]:
            target = targets[0]
        name = f"base_model.model.layers.0.{target}.lora_A.default.weight"
        if bad_name is not None and index == 0:
            name = bad_name
        size = quotient + (1 if index < remainder else 0)
        parameters.append((name, FakeParameter(size, requires_grad=True)))
    return FakeModel(parameters)


def test_model_trainability_matches_locked_lora_contract():
    report = inspect_model_trainability(locked_fake_model())

    assert report["trainable_parameters"] == LOCKED_TRAINABLE_PARAMETERS
    assert report["trainable_tensors"] == len(LOCKED_TARGET_MODULES)
    assert report["target_modules_observed"] == sorted(LOCKED_TARGET_MODULES)
    assert report["only_lora_parameters_trainable"]
    assert report["matches_locked_trainable_count"]
    assert report["total_parameters"] > report["trainable_parameters"]


def test_model_trainability_rejects_count_or_parameter_scope_drift():
    model = locked_fake_model()
    model.parameters[-1][1].size -= 1
    with pytest.raises(RuntimeError, match="trainable-parameter count mismatch"):
        inspect_model_trainability(model)

    with pytest.raises(RuntimeError, match="non-LoRA parameter"):
        inspect_model_trainability(
            locked_fake_model(bad_name="base_model.model.layers.0.q_proj.weight")
        )

    with pytest.raises(RuntimeError, match="target modules mismatch"):
        inspect_model_trainability(locked_fake_model(missing_target=True))


def test_trainable_parameter_value_audit_rejects_nonfinite_values():
    model = locked_fake_model()
    report = inspect_trainable_parameter_values(
        model,
        expected_trainable_tensors=len(LOCKED_TARGET_MODULES),
    )
    assert report["all_trainable_values_finite"]

    model.parameters[-1][1].finite_count -= 1
    with pytest.raises(RuntimeError, match="contains NaN or infinity"):
        inspect_trainable_parameter_values(
            model,
            expected_trainable_tensors=len(LOCKED_TARGET_MODULES),
        )


def test_preflight_passes_without_creating_output_or_loading_cuda(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    report = run_preflight(**kwargs)

    assert report["status"] == "passed"
    assert report["fixture"]["rows"] == 5
    assert report["sft_lock"]["trainable_parameters_expected"] == 18_464_768
    assert report["sft_lock"]["runtime_trainable_parameter_assertion_required"]
    assert report["disk"]["passes"]
    assert not report["output"]["created"]
    assert not Path(report["output"]["path"]).exists()
    assert not report["cuda_imports_performed"]
    assert not report["model_loaded"]
    assert not report["trainer_constructed"]


def test_preflight_rejects_dirty_or_unexpected_git_state(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    kwargs["git_state_fn"] = lambda _: {
        "commit": "a" * 40,
        "tracked_worktree_dirty": True,
        "index_dirty": False,
    }
    with pytest.raises(RuntimeError, match="tracked Git state must be clean"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    kwargs["expected_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="commit disagrees"):
        run_preflight(**kwargs)


def test_preflight_rejects_fixture_or_adapter_drift(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    kwargs["expected_fixture_data_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fixture data checksum"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    Path(kwargs["adapter"], "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="adapter checksum"):
        run_preflight(**kwargs)


def test_preflight_rejects_adapter_config_drift(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    config_path = Path(kwargs["adapter"], "adapter_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected LoRA rank"):
        run_preflight(**kwargs)


def test_preflight_rejects_output_collision_and_low_disk(tmp_path):
    kwargs = passing_kwargs(tmp_path)
    Path(kwargs["output_dir"]).mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        run_preflight(**kwargs)

    kwargs = passing_kwargs(tmp_path / "second")
    kwargs["disk_usage_fn"] = lambda _: SimpleNamespace(free=2 * 1024**3)
    with pytest.raises(RuntimeError, match="insufficient free disk"):
        run_preflight(**kwargs)
