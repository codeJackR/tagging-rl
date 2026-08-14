"""CPU-only composition proof for a Run 2 arm's runtime surface.

Every GPU-bearing collaborator is a fake that records what it received. The
tests assert both directions: a correct composition passes and reports the
truth, and each way the assembly could silently lose an invariant fails closed.

The load-bearing tests are `test_dropped_*_callback_fails_closed`. A trainer
that quietly discards a callback still trains to completion and still publishes
its artifacts; the only visible symptom is that the instrument which was
supposed to catch a regression at step 100 never ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run2_arm_runtime_composition import (
    CompositionError,
    REQUIRED_DATASET_COLUMNS,
    VERSION,
    compose_arm_runtime,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "runs" / "grpo-run2-causal-experiment-contract.json"
MONITOR_CONTRACT = ROOT / "runs" / "grpo-run2-checkpoint-monitor-contract.json"


# --- fakes -------------------------------------------------------------------


class FakeCallbackBase:
    """Stands in for `transformers.TrainerCallback`."""

    def __init__(self, *args, **kwargs):
        pass


class FakeConfig:
    """Mirrors the real GRPOConfig's attribute surface.

    The real class exposes every setting as a plain attribute and has no
    `.settings` mapping, so the fake must too: a fake that stored settings in a
    dict would let the production drift check pass here and no-op on the box.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # TRL normalizes these two the same way; the inspector expects it.
        self.report_to = [] if kwargs.get("report_to") == "none" else kwargs.get("report_to")
        self.generation_batch_size = (
            kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"]
        )


class FakeAccelerator:
    def backward(self, *a, **k):
        return None


class FakeCallbackHandler:
    def __init__(self):
        self.callbacks: list[object] = []


class FakeTrainer:
    """Mirrors the surface `compose_arm_runtime` inspects, and nothing else."""

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.reward_funcs = list(kwargs["reward_funcs"])
        self.args = kwargs["args"]
        self.train_dataset = kwargs["train_dataset"]
        self.processing_class = kwargs["processing_class"]
        self.reward_weights = list(kwargs["args"].reward_weights)
        self.optimizer = None
        self.lr_scheduler = None
        self.ref_model = None
        self.state = type("State", (), {"global_step": 0})()
        self.callback_handler = FakeCallbackHandler()
        self.accelerator = FakeAccelerator()

    # Defined on the class, exactly like the real GRPOTrainer, so that
    # instance-dict checking is the only way to detect wrapping.
    def _generate(self, *a, **k):
        return None

    def _calculate_rewards(self, *a, **k):
        return None

    def compute_loss(self, *a, **k):
        return None

    def add_callback(self, callback):
        self.callback_handler.callbacks.append(callback)


def forbidden_runner(**_kwargs):
    raise AssertionError("composition proof may not run a monitor")


def fake_command_builder(step, checkpoint, output):
    return ["python", "-m", "fake.monitor", str(step), str(checkpoint), str(output)]


def plenty_of_gpu() -> int:
    return 24 * 1024**3


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


@pytest.fixture
def compose(contract, pack, tmp_path):
    """Compose an arm with every collaborator faked and outputs redirected."""

    def _compose(arm="A", *, trainer_class=None, **overrides):
        kwargs = dict(
            root=ROOT,
            arm=arm,
            contract=contract,
            pack=pack,
            model_loader=lambda: (object(), object()),
            config_class=FakeConfig,
            trainer_class=trainer_class or FakeTrainer,
            callback_base_class=FakeCallbackBase,
            synchronize_fn=lambda: None,
            monitor_contract_path=MONITOR_CONTRACT,
            monitor_command_builder=fake_command_builder,
            monitor_runner=forbidden_runner,
            gpu_free_bytes_fn=plenty_of_gpu,
            trainer_scratch_dir=tmp_path / "trainer-scratch",
        )
        kwargs.update(overrides)
        return compose_arm_runtime(**kwargs)

    return _compose


# --- the happy path ----------------------------------------------------------


@pytest.mark.parametrize("arm,policy,n_rewards", [("A", "original_1_1_2", 3), ("B", "candidate_ua", 1)])
def test_both_arms_compose(compose, arm, policy, n_rewards):
    report, trainer = compose(arm)
    assert report["version"] == VERSION
    assert report["status"] == "arm_runtime_composed_no_training_dispatched"
    assert report["arm"] == arm
    assert report["reward_binding"]["policy"] == policy
    assert len(report["reward_binding"]["callable_names"]) == n_rewards
    assert len(trainer.reward_funcs) == n_rewards


def test_dataset_is_the_ordered_300_product_schedule(compose, contract):
    report, trainer = compose("A")
    dataset = report["dataset_binding"]
    assert dataset["rows"] == 300
    assert dataset["unique_skus"] == 300
    assert set(dataset["columns"]) == REQUIRED_DATASET_COLUMNS
    assert dataset["ordered_sku_sha256"] == contract["schedule"]["ordered_sku_sha256"]
    assert dataset["order_survived_materialization"] is True
    assert dataset["trainer_shuffle"] is False
    assert len(trainer.train_dataset) == 300


def test_both_callbacks_are_registered(compose):
    report, trainer = compose("A")
    registered = trainer.callback_handler.callbacks
    assert len(registered) == 2
    assert report["callback_binding"]["profiler_present"] is True
    assert report["callback_binding"]["causal_monitor_present"] is True
    assert report["callback_binding"]["required_callbacks_retained"] is True
    assert report["callback_binding"]["monitor_checkpoint_steps"] == [100, 200, 300]
    assert any(hasattr(value, "phase_profiler") for value in registered)
    assert any(hasattr(value, "causal_monitor") for value in registered)


def test_trainer_is_at_step_zero_with_no_optimizer_state(compose):
    report, _ = compose("A")
    binding = report["trainer_binding"]
    assert binding["global_step"] == 0
    assert binding["optimizer_constructed"] is False
    assert binding["lr_scheduler_constructed"] is False
    assert binding["reference_model_constructed"] is False
    assert report["boundaries"]["training_dispatched"] is False
    assert report["boundaries"]["optimizer_steps"] == 0


def test_composition_does_not_create_the_reserved_arm_path(compose, contract):
    report, _ = compose("A")
    assert set(report["reserved_paths"]) == {
        "output_dir", "monitor_root", "quality_root", "control_dir", "failure_dir"
    }
    for path in report["reserved_paths"].values():
        assert not Path(path).exists()
    assert report["reserved_paths_created"] is False
    assert Path(report["reserved_paths"]["output_dir"]) == (
        ROOT / contract["arms"]["A"]["output_dir"]
    ).resolve()


def test_nothing_is_executed_during_composition(compose):
    """The runner raises if called and the GPU probe is never needed to pass."""
    report, _ = compose("A", monitor_runner=forbidden_runner)
    assert report["callback_binding"]["monitor_runner_invoked_by_contract"] is False
    assert report["boundaries"]["rewards_observed_called"] == 0
    assert set(report["reward_binding"]["observed_reward_calls"].values()) == {0}


# --- the failures that would otherwise be silent ------------------------------


def test_dropped_monitor_callback_fails_closed(compose):
    """Losing the quality monitor is the failure this experiment cannot afford."""

    class DropsMonitorTrainer(FakeTrainer):
        def add_callback(self, callback):
            if hasattr(callback, "causal_monitor"):
                return
            self.callback_handler.callbacks.append(callback)

    with pytest.raises(CompositionError, match="dropped the"):
        compose("A", trainer_class=DropsMonitorTrainer)


def test_dropped_profiler_callback_fails_closed(compose):
    class DropsProfilerTrainer(FakeTrainer):
        def add_callback(self, callback):
            if hasattr(callback, "phase_profiler"):
                return
            self.callback_handler.callbacks.append(callback)

    with pytest.raises(CompositionError, match="dropped the"):
        compose("A", trainer_class=DropsProfilerTrainer)


def test_trainer_registering_no_callbacks_fails_closed(compose):
    class RegistersNothingTrainer(FakeTrainer):
        def add_callback(self, callback):
            return

    with pytest.raises(CompositionError, match="no registered callbacks"):
        compose("A", trainer_class=RegistersNothingTrainer)


def test_reordered_reward_binding_fails_closed(compose):
    class ReordersRewardsTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reward_funcs = list(reversed(self.reward_funcs))

    with pytest.raises(CompositionError, match="mangled the reward binding"):
        compose("A", trainer_class=ReordersRewardsTrainer)


def test_trainer_dropping_a_reward_fails_closed(compose):
    """Arm A must train on all three rewards, including the weight-2.0 term.

    An earlier version restored the raw callables unconditionally, so a trainer
    that kept only two composed cleanly and the report listed all three.
    """

    class DropsRewardTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reward_funcs = self.reward_funcs[:2]

    with pytest.raises(CompositionError, match="mangled the reward binding"):
        compose("A", trainer_class=DropsRewardTrainer)


def test_trainer_substituting_a_reward_fails_closed(compose):
    class SubstitutesRewardTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            foreign = lambda *a, **k: [0.0]
            foreign.__name__ = "format_validity_reward"  # even the name matches
            self.reward_funcs = [foreign, *self.reward_funcs[1:]]

    with pytest.raises(CompositionError, match="mangled the reward binding"):
        compose("A", trainer_class=SubstitutesRewardTrainer)


def test_drifted_reward_weights_fail_closed(compose):
    class DriftsWeightsTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reward_weights = [1.0, 1.0, 1.0]

    with pytest.raises(CompositionError, match="reward weights drifted"):
        compose("A", trainer_class=DriftsWeightsTrainer)


def test_prebuilt_optimizer_fails_closed(compose):
    class HasOptimizerTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.optimizer = object()

    with pytest.raises(CompositionError, match="created an optimizer"):
        compose("A", trainer_class=HasOptimizerTrainer)


def test_reference_model_under_beta_zero_fails_closed(compose):
    """beta=0 means no reference policy; one appearing is a config regression."""

    class HasRefModelTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ref_model = object()

    with pytest.raises(CompositionError, match="reference model"):
        compose("A", trainer_class=HasRefModelTrainer)


def test_nonzero_global_step_fails_closed(compose):
    class WarmTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state = type("State", (), {"global_step": 7})()

    with pytest.raises(CompositionError, match="not at step zero"):
        compose("A", trainer_class=WarmTrainer)


@pytest.mark.parametrize(
    "attribute,message",
    [
        ("args", "retain the constructed config"),
        ("processing_class", "retain the tokenizer"),
        ("model", "neither the loaded policy nor a wrapper"),
    ],
)
def test_trainer_dropping_a_handed_object_fails_closed(compose, attribute, message):
    """The previous version of this test asserted on the fake's own bookkeeping.

    It inspected a `received_kwargs` attribute that the fake planted and the
    production default made vacuous. These attributes are the ones Run 1's real
    construction gate reads off a real GRPOTrainer.
    """

    class SubstitutingTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            setattr(self, attribute, object())

    with pytest.raises(CompositionError, match=message):
        compose("A", trainer_class=SubstitutingTrainer)


@pytest.mark.parametrize("attribute", ["model", "args", "processing_class"])
def test_trainer_missing_an_attribute_entirely_fails_closed(compose, attribute):
    """`getattr(trainer, name, default)` where default IS the compared value
    passes for a trainer that lacks the attribute. That idiom made the old
    required-kwargs check a no-op and it must not come back."""

    class AmnesiacTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            object.__delattr__(self, attribute)

    with pytest.raises(CompositionError, match=f"exposes no {attribute}"):
        compose("A", trainer_class=AmnesiacTrainer)


def test_substituted_dataset_fails_closed(compose):
    class SwapsDatasetTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.train_dataset = {"sku_id": ["not-the-schedule"]}

    with pytest.raises(CompositionError, match="dataset content drifted"):
        compose("A", trainer_class=SwapsDatasetTrainer)


def test_peft_style_model_rewrap_is_allowed(compose):
    """TRL and Unsloth may wrap the policy; a wrapper is not a substitution."""

    class WrapsModelTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = type("PeftWrapper", (), {"base_model": kwargs["model"]})()

    report, _ = compose("A", trainer_class=WrapsModelTrainer)
    assert report["trainer_binding"]["model_present"] is True
    assert report["trainer_binding"]["model_retained_by_identity"] is False


def test_remapped_dataset_with_identical_content_is_allowed(compose):
    """TRL may hand back a mapped copy; Run 1's real gate checks content, not id."""

    class RemapsDatasetTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.train_dataset = {"sku_id": list(kwargs["train_dataset"]["sku_id"])}

    report, _ = compose("A", trainer_class=RemapsDatasetTrainer)
    assert report["dataset_binding"]["retained_by_identity"] is False


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("beta", 0.1),            # run 1 shipped beta=0 unnoticed; never again
        ("learning_rate", 1e-4),  # locked at 5e-6
        ("temperature", 1.0),     # sampling must match the difficulty run
        ("max_steps", 500),       # the step budget is part of the comparison
        ("shuffle_dataset", True),  # would destroy the fixed product schedule
        ("save_steps", 50),       # checkpoints must land at 100/200/300
        ("seed", 7),              # arms must share randomness
    ],
)
def test_config_drift_fails_closed(compose, field, bad_value):
    """Any locked setting drifting must stop composition, not just one.

    This is checked through the same inspector Phase G used against the real
    GRPOConfig, which reads plain attributes. An earlier draft compared a
    `.settings` mapping that only the test fake had, so this check passed here
    and silently did nothing in production.
    """

    class DriftingConfig(FakeConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            setattr(self, field, bad_value)

    with pytest.raises(CompositionError, match="config drifted"):
        compose("A", config_class=DriftingConfig)


def test_model_loader_returning_nothing_fails_closed(compose):
    with pytest.raises(CompositionError, match="no model or tokenizer"):
        compose("A", model_loader=lambda: (None, None))


def test_unknown_arm_fails_closed(compose):
    with pytest.raises(CompositionError, match="unknown arm"):
        compose("C")


@pytest.mark.parametrize(
    "reserved_key",
    ["output_dir", "monitor_root", "quality_root", "control_dir", "failure_dir"],
)
def test_any_existing_reserved_path_blocks_composition(
    contract, pack, tmp_path, reserved_key
):
    """All five arm paths are checked, not only the training output.

    A stale monitor_root used to pass composition and then explode inside
    `on_train_begin`, which does `mkdir(exist_ok=False)` with the model already
    resident on the GPU. Built against a temporary repo root holding real
    directories rather than by monkeypatching `Path.exists`.
    """
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / contract["arms"]["A"][reserved_key]).mkdir(parents=True)

    with pytest.raises(CompositionError, match="reserved arm paths already exist"):
        compose_arm_runtime(
            root=fake_root,
            arm="A",
            contract=contract,
            pack=pack,
            model_loader=lambda: (object(), object()),
            config_class=FakeConfig,
            trainer_class=FakeTrainer,
            callback_base_class=FakeCallbackBase,
            synchronize_fn=lambda: None,
            monitor_contract_path=MONITOR_CONTRACT,
            monitor_command_builder=fake_command_builder,
            monitor_runner=forbidden_runner,
            gpu_free_bytes_fn=plenty_of_gpu,
            trainer_scratch_dir=tmp_path / "scratch",
        )


# --- the confound the proof previously would have signed off on ---------------


def _mutated_contract(contract, arm, **config_overrides):
    """Deep-copy the contract and drift one arm's trainer config."""
    clone = json.loads(json.dumps(contract))
    clone["arms"][arm]["config"].update(config_overrides)
    return clone


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 5e-5),
        ("seed", 999),
        ("temperature", 1.2),
        ("beta", 0.05),
        ("max_steps", 500),
    ],
)
def test_arms_differing_outside_the_reward_fail_closed(
    contract, pack, tmp_path, field, value
):
    """The causal claim is enforced, not merely asserted in a document.

    Checking each arm against its own spec cannot detect a confound: a contract
    whose arm B carries a different learning rate satisfies both per-arm checks
    while destroying the experiment. Before this gate existed, composing such a
    contract succeeded and every assertion in `test_arms_differ_only_in_reward`
    still passed, because that test compares values that are structurally
    guaranteed to match.
    """
    drifted = _mutated_contract(contract, "B", **{field: value})
    with pytest.raises(CompositionError, match="arms differ outside the reward"):
        compose_arm_runtime(
            root=ROOT,
            arm="A",
            contract=drifted,
            pack=pack,
            model_loader=lambda: (object(), object()),
            config_class=FakeConfig,
            trainer_class=FakeTrainer,
            callback_base_class=FakeCallbackBase,
            synchronize_fn=lambda: None,
            monitor_contract_path=MONITOR_CONTRACT,
            monitor_command_builder=fake_command_builder,
            monitor_runner=forbidden_runner,
            gpu_free_bytes_fn=plenty_of_gpu,
            trainer_scratch_dir=tmp_path / "scratch",
        )


def test_causal_audit_is_recomputed_and_reported(compose, contract):
    report, _ = compose("A")
    audit = report["causal_difference_audit"]
    assert audit == contract["causal_difference_audit"]
    assert audit["all_other_trainer_settings_equal"] is True
    assert audit["beta_equal_and_explicitly_zero"] is True
    assert set(audit["trainer_config_differences"]) == {"output_dir", "run_name"}


# --- lineage, quality policy, collaborator retention, instrumentation ---------


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("version", "made-up-v0", "locked causal contract"),
        ("status", "scratch_not_locked", "not locked"),
        ("arm_order", ["B", "A"], "arm order drifted"),
    ],
)
def test_unlocked_contract_fails_closed(contract, pack, tmp_path, key, value, message):
    """Composition used to accept any dictionary, including a scratch contract."""
    clone = json.loads(json.dumps(contract))
    clone[key] = value
    with pytest.raises(CompositionError, match=message):
        compose_arm_runtime(
            root=ROOT, arm="A", contract=clone, pack=pack,
            model_loader=lambda: (object(), object()),
            config_class=FakeConfig, trainer_class=FakeTrainer,
            callback_base_class=FakeCallbackBase, synchronize_fn=lambda: None,
            monitor_contract_path=MONITOR_CONTRACT,
            monitor_command_builder=fake_command_builder,
            monitor_runner=forbidden_runner, gpu_free_bytes_fn=plenty_of_gpu,
            trainer_scratch_dir=tmp_path / "scratch",
        )


def test_quality_policy_checkpoint_drift_fails_closed(contract, pack, tmp_path):
    """The monitor and the breach tracker read their steps from different places.

    When they disagreed, composition passed and reported [100, 200, 300] while
    the tracker demanded (100, 200). Training then ran to the final checkpoint
    before raising, six GPU-hours in.
    """
    clone = json.loads(json.dumps(contract))
    clone["monitoring"]["quality_policy"]["checkpoints"] = [100, 200]
    with pytest.raises(CompositionError, match="quality policy expects steps"):
        compose_arm_runtime(
            root=ROOT, arm="A", contract=clone, pack=pack,
            model_loader=lambda: (object(), object()),
            config_class=FakeConfig, trainer_class=FakeTrainer,
            callback_base_class=FakeCallbackBase, synchronize_fn=lambda: None,
            monitor_contract_path=MONITOR_CONTRACT,
            monitor_command_builder=fake_command_builder,
            monitor_runner=forbidden_runner, gpu_free_bytes_fn=plenty_of_gpu,
            trainer_scratch_dir=tmp_path / "scratch",
        )


def test_profiler_actually_instruments_the_trainer(compose):
    """Registering the callback without instrumenting yields empty phase records."""
    report, trainer = compose("A")
    assert report["profiler_binding"]["instrumented_phases"] == [
        "forward_loss",
        "generation",
        "reward",
    ]
    # `_wrap_callable` writes a function named `measured` into the instance
    # dict. Asserting "not a lambda" only discriminated because the old fake
    # used lambdas; an unwrapped real `compute_loss` would have satisfied it.
    for name in ("_generate", "_calculate_rewards", "compute_loss"):
        assert vars(trainer)[name].__qualname__.split(".")[-1] == "measured"
    assert vars(trainer.accelerator)["backward"].__qualname__.split(".")[-1] == "measured"


def test_rewards_invoked_during_construction_are_detected(compose):
    """This was previously a hardcoded False in the report."""

    class ProbingTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for fn in kwargs["reward_funcs"]:
                fn(completions=[], prompts=[], gold=[])

    with pytest.raises(CompositionError, match="rewards were invoked during construction"):
        compose("A", trainer_class=ProbingTrainer)


def test_scratch_directory_may_not_be_a_reserved_path(compose, contract):
    with pytest.raises(CompositionError, match="scratch directory may not be"):
        compose("A", trainer_scratch_dir=ROOT / contract["arms"]["A"]["output_dir"])


def test_dispatched_trainer_holds_the_raw_reward_callables(compose, contract):
    """The measurement apparatus must not survive into the artifact under test.

    Counting proxies make "no reward ran during construction" a measurement
    rather than a claim, but an earlier version left them bound to the trainer.
    Run 2 would then have trained against closures whose signature is
    `(*args, **kwargs)` while TRL introspects reward signatures to decide which
    dataset columns to forward, and the run would no longer be comparable with
    the offline G10 selection that scored these exact functions.
    """
    from training.run2_causal_experiment import _reward_callables

    report, trainer = compose("A")
    expected = list(_reward_callables("A"))
    # Composition assigns this list, so on its own the assertion below only
    # catches an assignment that silently no-ops. The checks that matter are
    # test_trainer_{reordering,dropping,substituting}_* above, which prove the
    # binding is verified BEFORE it is restored.
    assert list(trainer.reward_funcs) == expected
    for bound, raw in zip(trainer.reward_funcs, expected):
        assert bound is raw
        assert bound.__module__ == "training.rewards"
    assert report["reward_binding"]["callable_modules"] == [
        "training.rewards"
    ] * len(expected)
    assert report["boundaries"]["rewards_observed_called"] == 0


# --- the smoke escape hatch ---------------------------------------------------


def test_smoke_mode_relaxes_the_step_budget_and_is_stamped(compose):
    """A short GPU smoke cannot otherwise reach this path.

    The production gate requires the locked 300-step schedule, so proving that
    the instrumentation actually FIRES needs a deliberate escape hatch. It must
    be impossible to mistake the result for a production composition.
    """
    report, _ = compose("A", smoke_max_steps=1)
    assert report["status"] == "arm_runtime_smoke_composed"
    assert report["smoke_mode"] is True
    assert report["step_schedule"]["max_steps"] == 1
    assert report["step_schedule"]["checkpointing"] == "disabled"


def test_production_composition_is_never_stamped_as_smoke(compose):
    report, _ = compose("A")
    assert report["status"] == "arm_runtime_composed_no_training_dispatched"
    assert report["smoke_mode"] is False
    assert report["step_schedule"]["max_steps"] == 300


def test_smoke_mode_disables_checkpointing_so_the_monitor_cannot_fire(compose):
    """With saving off, `on_save` never runs and the quality tracker is never
    asked for a step the smoke will not reach."""
    _, trainer = compose("A", smoke_max_steps=2)
    assert trainer.args.save_strategy == "no"
    assert trainer.args.max_steps == 2


@pytest.mark.parametrize("bad", [0, -1, 300, 500])
def test_smoke_budget_outside_the_permitted_range_fails_closed(compose, bad):
    """It must not be possible to run a full-length job through the smoke path."""
    with pytest.raises(CompositionError, match="smoke step budget"):
        compose("A", smoke_max_steps=bad)
