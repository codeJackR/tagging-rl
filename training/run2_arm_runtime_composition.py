#!/usr/bin/env python3
"""CPU-only proof that one Run 2 arm's runtime surface composes correctly.

Phase G proved the *configuration* for both arms; the Arm A launcher proved the
*inputs* can be validated. This module proves the remaining assembly step: that
an ordered 300-product dataset, the arm's reward binding, its trainer
configuration, the phase profiler and the causal checkpoint monitor can be put
together into one trainer without any of them being silently dropped.

Every GPU-bearing collaborator is injected. Nothing here imports Torch, Unsloth
or TRL, constructs an optimizer, starts a monitor process or dispatches
training. The production runtime supplies the real classes; the tests supply
fakes that record what they were handed.

**Why a proof over fakes is not a tautology.** Every attribute this module
reads was checked against the real classes on the GPU host before being relied
on, because a fake that exposes a surface the real trainer lacks would let this
proof pass while production silently skipped the same check:

- `trainer.reward_funcs`, `reward_weights`, `optimizer`, `lr_scheduler`,
  `ref_model` and `state.global_step` are read by Run 1's own construction gate
  in `training/train_grpo.py` against a real `GRPOTrainer`, so they are
  precedent-verified.
- `callback_handler.callbacks` exists on the real `CallbackHandler`, and
  `add_callback` appends the instance it is given, so comparing callbacks by
  identity is meaningful in production and not just against the fake.
- The real `GRPOConfig` has **no** `.settings` mapping. Config drift is
  therefore checked with `_inspect_constructed_config`, the attribute-reading
  inspector Phase G already ran against the real class.
- Under the installed TRL 0.24.0, `GRPOTrainer.__init__` performs no dataset
  remapping and the base `Trainer.__init__` does a plain
  `self.train_dataset = train_dataset`, so object identity is expected to
  survive construction. Identity is nevertheless *reported* rather than
  required, and content is what is enforced, matching Run 1's real construction
  gate: a future TRL that returns a mapped copy should not turn this proof into
  a false alarm on the first dispatch.

Two properties are enforced that a naive assembly would lose:

1. **Order is identity.** The 300 products define the optimizer-step schedule,
   so dataset order is verified against the contract's ordered SKU hash *after*
   materialization, not assumed from the file.
2. **Both callbacks must survive.** Run 1 carried a profiler; Run 2 adds the
   quality monitor that would have caught the regression at step 100. A
   composition that keeps one and loses the other still trains, still publishes,
   and silently removes the instrument this whole experiment exists to add.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import read_jsonl
from training.dataset import load_grpo_prompts
from training.grpo_phase_profiler import (
    PHASE_METHODS,
    FullRunPhaseProfiler,
    make_phase_profiler_callback_class,
)
from training.run2_causal_experiment import (
    CHECKPOINT_STEPS,
    SCHEDULE_ROWS,
    TRAINING_STEPS,
    VERSION as CAUSAL_CONTRACT_VERSION,
    CausalCheckpointMonitorCoordinator,
    _arm_diff,
    _inspect_constructed_config,
    _reward_callables,
    make_causal_monitor_callback_class,
)
from training.run2_checkpoint_monitor_control import CheckpointMonitorCoordinator

VERSION = "grpo-run2-arm-runtime-composition-v1"
REQUIRED_DATASET_COLUMNS = frozenset({"prompt", "gold", "sku_id"})
# TRL accepts a bare callable or a list; the contract fixes the ordered list
# form so a single-reward arm and a three-reward arm are shaped identically.
# Every reserved path an arm owns. Run 1's preflight checks all five; an
# earlier draft of this module checked only output_dir, so a stale monitor_root
# survived composition and detonated inside on_train_begin with the model
# already resident on the GPU.
RESERVED_PATH_KEYS = ("output_dir", "monitor_root", "quality_root", "control_dir", "failure_dir")


class CompositionError(RuntimeError):
    """Raised when an arm's runtime surface fails any composition invariant."""


def _validate_contract_lineage(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse to compose against anything but the accepted, locked contract.

    Composition previously accepted any dictionary, so a scratch contract with
    an invented version composed cleanly. These are the same gates the Arm A
    launcher applies before it validates inputs.
    """
    if contract.get("version") != CAUSAL_CONTRACT_VERSION:
        raise CompositionError(
            f"composition requires the locked causal contract, got "
            f"{contract.get('version')!r}"
        )
    if contract.get("status") != "locked_no_gpu_training_dispatched":
        raise CompositionError(f"causal contract is not locked: {contract.get('status')!r}")
    if contract.get("arm_order") != ["A", "B"]:
        raise CompositionError("causal arm order drifted")
    if contract.get("boundaries", {}).get("gpu_training_dispatched_by_contract_build") is not False:
        raise CompositionError("causal contract build crossed the training boundary")
    if contract.get("deferral_audit", {}).get("passed") is not True:
        raise CompositionError("causal contract still has deferred decisions")
    return {
        "version": contract["version"],
        "status": contract["status"],
        # Copied from the contract, not verified here. The Arm A launcher is
        # what chains contract -> preflight -> construction receipts.
        "expected_execution_code_commit_from_contract": contract.get(
            "expected_execution_code_commit"
        ),
    }


def _validate_causal_isolation(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Prove both arms differ only in reward, not merely that each matches itself.

    Checking each arm against its own spec cannot detect a confound: a contract
    whose arm B carries a different learning rate satisfies both per-arm checks
    while destroying the experiment. `_arm_diff` raises unless the entire
    trainer-config difference is exactly {output_dir, run_name}.
    """
    try:
        audit = _arm_diff(contract["arms"]["A"], contract["arms"]["B"])
    except RuntimeError as exc:
        raise CompositionError(f"arms differ outside the reward: {exc}") from exc
    if contract.get("causal_difference_audit") != audit:
        raise CompositionError("recomputed causal difference disagrees with the contract")
    if not audit["beta_equal_and_explicitly_zero"]:
        raise CompositionError("arms disagree on beta or beta is not explicitly zero")
    return audit


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _callable_name(value: object) -> str:
    return getattr(value, "__name__", type(value).__name__)


def _normalized_expected_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Contract settings in the shape a constructed config reports them.

    `report_to="none"` normalizes to an empty reporter list and the weights ride
    on the config rather than the arm spec, exactly as the Phase G construction
    proof recorded them.
    """
    expected = dict(spec["config"])
    expected["report_to"] = []
    expected["reward_weights"] = spec["reward"]["weights"]
    return expected


def build_arm_dataset(
    *,
    root: Path,
    arm: str,
    contract: Mapping[str, Any],
    pack: Any,
) -> tuple[Any, dict[str, Any]]:
    """Materialize the ordered schedule and prove it is the contracted one."""
    schedule_identity = contract["schedule"]["dataset"]
    schedule_path = root / schedule_identity["path"]
    if not schedule_path.is_file():
        raise CompositionError(f"schedule file is absent: {schedule_identity['path']}")

    rows = read_jsonl(schedule_path)
    file_order = [row.sku_id for row in rows]
    dataset = load_grpo_prompts(pack, schedule_path, require_pass_rate_band=True)

    materialized_order = list(dataset["sku_id"])
    if len(dataset) != SCHEDULE_ROWS:
        raise CompositionError(
            f"arm dataset must contain {SCHEDULE_ROWS} rows, found {len(dataset)}"
        )
    if len(set(materialized_order)) != SCHEDULE_ROWS:
        raise CompositionError("arm dataset contains duplicate products")
    if materialized_order != file_order:
        raise CompositionError("dataset loading reordered the optimizer-step schedule")
    if _ordered_hash(materialized_order) != contract["schedule"]["ordered_sku_sha256"]:
        raise CompositionError("materialized dataset order drifted from the contract")
    if set(dataset.column_names) != REQUIRED_DATASET_COLUMNS:
        raise CompositionError(f"arm dataset columns drifted: {dataset.column_names}")

    shuffle = contract["arms"][arm]["config"]["shuffle_dataset"]
    if shuffle is not False or contract["schedule"]["trainer_shuffle"] is not False:
        raise CompositionError(
            f"arm {arm} would shuffle the fixed optimizer-step schedule"
        )

    return dataset, {
        "path": schedule_identity["path"],
        "rows": len(dataset),
        "unique_skus": len(set(materialized_order)),
        "columns": sorted(dataset.column_names),
        "ordered_sku_sha256": _ordered_hash(materialized_order),
        "order_survived_materialization": True,
        # Read this arm's own value. An earlier draft indexed arm_order[0], so
        # an arm B report asserted arm A's shuffle setting: a published artifact
        # stating the opposite of the truth for the one flag that would destroy
        # the fixed 300-product schedule.
        "trainer_shuffle": shuffle,
    }


def build_reward_binding(arm: str, contract: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Bind the arm's reward callables in contract order without calling them."""
    spec = contract["arms"][arm]
    callables = list(_reward_callables(arm))
    names = [_callable_name(value) for value in callables]
    modules = [value.__module__ for value in callables]

    if names != spec["reward"]["functions"]:
        raise CompositionError(f"arm {arm} reward callable order drifted: {names}")
    if len(spec["reward"]["weights"]) != len(callables):
        raise CompositionError(
            f"arm {arm} has {len(callables)} rewards but "
            f"{len(spec['reward']['weights'])} weights"
        )
    return callables, {
        "policy": spec["reward"]["policy"],
        "callable_names": names,
        "callable_modules": modules,
        "weights": spec["reward"]["weights"],
        "reward_calls_measured": True,
    }


def build_callbacks(
    *,
    root: Path,
    arm: str,
    contract: Mapping[str, Any],
    callback_base_class: type,
    synchronize_fn: Callable[[], object],
    expected_steps: int,
    monitor_contract_path: Path,
    monitor_command_builder: Callable[[int, Path, Path], list[str]],
    monitor_runner: Callable[..., Mapping[str, Any]],
    gpu_free_bytes_fn: Callable[[], int],
) -> tuple[list[Any], FullRunPhaseProfiler, dict[str, Any]]:
    """Build exactly the profiler and causal-monitor callbacks, in that order."""
    spec = contract["arms"][arm]
    base = CheckpointMonitorCoordinator(
        repo_root=root,
        contract_path=monitor_contract_path,
        monitor_root=(root / spec["monitor_root"]).resolve(),
        command_builder=monitor_command_builder,
        runner=monitor_runner,
    )
    causal = CausalCheckpointMonitorCoordinator(
        base=base,
        quality_policy=contract["monitoring"]["quality_policy"],
        quality_output_root=(root / spec["quality_root"]).resolve(),
        gpu_free_bytes_fn=gpu_free_bytes_fn,
    )

    # The profiler is pure stdlib, so the real class is used rather than a fake:
    # the real factory rejects a non-FullRunPhaseProfiler, and injecting a fake
    # factory would strictly weaken the proof for no benefit. Only the CUDA
    # synchronize call is injected.
    phase_profiler = FullRunPhaseProfiler(
        expected_steps=expected_steps, synchronize_fn=synchronize_fn
    )
    profiler_callback_class = make_phase_profiler_callback_class(callback_base_class)
    profiler_callback = profiler_callback_class(phase_profiler=phase_profiler)
    monitor_callback_class = make_causal_monitor_callback_class(callback_base_class)
    monitor_callback = monitor_callback_class(coordinator=causal)

    if profiler_callback.phase_profiler is not phase_profiler:
        raise CompositionError("profiler callback did not retain its profiler")
    if monitor_callback.causal_monitor is not causal:
        raise CompositionError("monitor callback did not retain its coordinator")
    if tuple(base.expected_steps) != tuple(CHECKPOINT_STEPS):
        raise CompositionError(
            f"monitor expects steps {tuple(base.expected_steps)}, "
            f"contract requires {tuple(CHECKPOINT_STEPS)}"
        )
    # The Phase F coordinator and the Phase G breach tracker read their step
    # lists from different places. If they disagree, training runs to the final
    # checkpoint and only then raises "expected step None".
    if tuple(causal.tracker.expected_steps) != tuple(CHECKPOINT_STEPS):
        raise CompositionError(
            f"quality policy expects steps {tuple(causal.tracker.expected_steps)}, "
            f"contract requires {tuple(CHECKPOINT_STEPS)}"
        )
    if tuple(contract["monitoring"]["checkpoint_steps"]) != tuple(CHECKPOINT_STEPS):
        raise CompositionError("contract monitoring checkpoint steps drifted")
    # A coordinator that dropped an injected collaborator would fall back to the
    # real GPU probe or the real supervised runner without ever being called
    # during composition, so the substitution would be invisible.
    if base.runner is not monitor_runner:
        raise CompositionError("monitor coordinator did not retain the injected runner")
    if base.command_builder is not monitor_command_builder:
        raise CompositionError("monitor coordinator did not retain the command builder")
    if causal.gpu_free_bytes_fn is not gpu_free_bytes_fn:
        raise CompositionError("causal monitor did not retain the injected GPU probe")

    callbacks = [profiler_callback, monitor_callback]
    return callbacks, phase_profiler, {
        "callback_classes": [type(value).__name__ for value in callbacks],
        "profiler_present": True,
        "causal_monitor_present": True,
        "monitor_checkpoint_steps": list(base.expected_steps),
        "monitor_runner_invoked_by_contract": False,
        "callback_lifecycle_invoked_by_contract": False,
    }


def _instrument_and_verify(phase_profiler: FullRunPhaseProfiler, trainer: Any) -> dict[str, Any]:
    """Wrap the trainer's phase methods and prove the wrapping actually happened.

    Run 1's real sequence is construct profiler, `instrument_trainer`, build the
    callback, register it. A composition that registers the callback but skips
    instrumentation produces a profiler whose phase records stay empty for the
    whole run, and every count in the published timing artifact is zero.
    """
    try:
        phase_profiler.instrument_trainer(trainer)
    except (TypeError, RuntimeError) as exc:
        raise CompositionError(f"phase profiler could not instrument the trainer: {exc}") from exc

    # `_wrap_callable` does `setattr(owner, name, measured)`, so a wrapped
    # method lands in the instance dict under the name `measured`. Comparing
    # `getattr` before and after cannot work on a real trainer: those methods
    # live on the class, so every attribute access mints a fresh bound method
    # and the comparison is unequal whether or not wrapping occurred.
    expected_qualname = "FullRunPhaseProfiler._wrap_callable.<locals>.measured"

    def _is_wrapped(owner: Any, name: str) -> bool:
        try:
            attrs = vars(owner)
        except TypeError as exc:  # __slots__ owners have no __dict__
            raise CompositionError(
                f"cannot inspect {type(owner).__name__} for instrumentation: {exc}"
            ) from exc
        wrapper = attrs.get(name)
        # Match the full qualname and require a closure: a name-only match is
        # satisfied by any function called `measured`.
        return (
            wrapper is not None
            and getattr(wrapper, "__qualname__", None) == expected_qualname
            and bool(getattr(wrapper, "__closure__", None))
        )

    wrapped = []
    for phase, method_name in PHASE_METHODS.items():
        if not _is_wrapped(trainer, method_name):
            raise CompositionError(f"phase profiler did not wrap {method_name}")
        wrapped.append(phase)
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None or not _is_wrapped(accelerator, "backward"):
        raise CompositionError("phase profiler did not wrap accelerator.backward")
    return {
        "instrumented_phases": sorted(wrapped),
        "accelerator_backward_wrapped": True,
        "optimizer_instrumented": False,
    }


def _assert_callbacks_survived(trainer: Any, callbacks: Sequence[Any]) -> dict[str, Any]:
    """A trainer that quietly drops a callback still trains. Catch that here.

    Transformers exposes registered callbacks through `callback_handler`; the
    injected fakes mirror that surface. Identity comparison is deliberate: an
    equal-but-different callback would be a different instrument.
    """
    handler = getattr(trainer, "callback_handler", None)
    registered = list(getattr(handler, "callbacks", []) or [])
    if not registered:
        raise CompositionError("trainer exposes no registered callbacks")
    for callback in callbacks:
        if not any(candidate is callback for candidate in registered):
            raise CompositionError(
                f"trainer dropped the {type(callback).__name__} callback"
            )
    return {
        "registered_callback_classes": [type(value).__name__ for value in registered],
        "required_callbacks_retained": True,
        "retained_by_identity": True,
    }


def _unwrap_to(candidate: Any, target: Any, *, depth: int = 8) -> Any | None:
    """Follow the usual wrapper attributes looking for `target`.

    PEFT and Accelerate wrap a policy rather than replacing it, so identity can
    legitimately fail while the loaded model is still what trains. Anything not
    reachable this way is a different object, not a wrapper.
    """
    node = candidate
    for _ in range(depth):
        if node is target:
            return node
        if node is None:
            return None
        # Chain on None, not on truthiness. An empty `nn.ModuleList` or any
        # module defining `__len__`/`__bool__` is falsy, so an `or` chain walks
        # straight past the layer that would have found the target and rejects
        # a legitimate wrapper.
        getter = getattr(node, "get_base_model", None)
        nxt = getter() if callable(getter) else None
        for attribute in ("base_model", "module", "model"):
            if nxt is not None:
                break
            nxt = getattr(node, attribute, None)
        node = nxt
    return None


def _assert_trainer_retained(
    trainer: Any, *, model: Any, config: Any, tokenizer: Any, reward_funcs: Sequence[Any]
) -> dict[str, Any]:
    """Prove the trainer kept every object it was handed.

    An earlier draft inspected a `received_kwargs` attribute that only the test
    fake set, defaulting to the very set being checked, so in production the
    check reduced to `issubset(itself)` and was always true. These attributes
    are all read by Run 1's real construction gate instead.
    """
    # Presence is required first. Defaulting `getattr` to the object being
    # compared is the same idiom that made the old `received_kwargs` check a
    # no-op: a trainer missing the attribute entirely would have passed.
    for attribute in ("model", "args", "processing_class", "reward_funcs"):
        if not hasattr(trainer, attribute):
            raise CompositionError(f"trainer exposes no {attribute}")

    # `Trainer.__init__` assigns `self.args = args` plainly, so identity is
    # required there. The model is different: TRL and Unsloth may rewrap it for
    # PEFT before delegating to the base initializer, so identity is reported
    # and presence is enforced, the same treatment the dataset gets. Discovering
    # that distinction at dispatch time is what this asymmetry avoids.
    if trainer.args is not config:
        raise CompositionError("trainer did not retain the constructed config")
    if trainer.processing_class is not tokenizer:
        raise CompositionError("trainer did not retain the tokenizer")
    # `is not None` is not a floor: a trainer holding an unrelated object passed.
    # TRL and Unsloth may legitimately rewrap the policy for PEFT, so walk the
    # standard unwrap chain and require the loaded model to be reachable.
    if trainer.model is not model and _unwrap_to(trainer.model, model) is None:
        raise CompositionError(
            "trainer model is neither the loaded policy nor a wrapper around it"
        )
    if list(trainer.reward_funcs) != list(reward_funcs):
        raise CompositionError("trainer did not retain the bound reward callables")
    return {
        "model_present": True,
        "model_retained_by_identity": trainer.model is model,
        "config_retained": True,
        "tokenizer_retained": True,
        "reward_callables_retained": True,
    }


def _assert_trainer_boundaries(trainer: Any, reward_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the constructed trainer is at step zero with no optimizer state."""
    reward_names = [_callable_name(value) for value in trainer.reward_funcs]
    if reward_names != reward_binding["callable_names"]:
        raise CompositionError(f"trainer reward order drifted: {reward_names}")

    weights = trainer.reward_weights
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    if [float(value) for value in weights] != [
        float(value) for value in reward_binding["weights"]
    ]:
        raise CompositionError("trainer reward weights drifted")

    if getattr(trainer, "optimizer", None) is not None:
        raise CompositionError("composition unexpectedly created an optimizer")
    if getattr(trainer, "lr_scheduler", None) is not None:
        raise CompositionError("composition unexpectedly created an LR scheduler")
    if getattr(trainer, "ref_model", None) is not None:
        raise CompositionError("beta=0 composition unexpectedly created a reference model")
    global_step = int(getattr(getattr(trainer, "state", None), "global_step", -1))
    if global_step != 0:
        raise CompositionError(f"composed trainer is not at step zero: {global_step}")

    return {
        "reward_order_preserved": True,
        "reward_weights_preserved": True,
        "optimizer_constructed": False,
        "lr_scheduler_constructed": False,
        "reference_model_constructed": False,
        "global_step": global_step,
    }


def compose_arm_runtime(
    *,
    root: str | Path,
    arm: str,
    contract: Mapping[str, Any],
    pack: Any,
    model_loader: Callable[[], tuple[Any, Any]],
    config_class: type,
    trainer_class: type,
    callback_base_class: type,
    synchronize_fn: Callable[[], object],
    monitor_contract_path: str | Path,
    monitor_command_builder: Callable[[int, Path, Path], list[str]],
    monitor_runner: Callable[..., Mapping[str, Any]],
    gpu_free_bytes_fn: Callable[[], int],
    trainer_scratch_dir: str | Path,
    smoke_max_steps: int | None = None,
) -> tuple[dict[str, Any], Any]:
    """Compose one arm's trainer surface; return its report and the trainer.

    `trainer_scratch_dir` is required, not optional. A real `Trainer.__init__`
    calls `os.makedirs(args.output_dir)`, so pointing it at the reserved arm
    path would create that directory during a proof and then permanently fail
    the launch preflight, which requires the path to be absent. Composition
    therefore always builds into scratch space, and the reserved paths are
    checked before and after.
    """
    root = Path(root).resolve()
    if arm not in contract.get("arms", {}):
        raise CompositionError(f"unknown arm: {arm}")

    lineage = _validate_contract_lineage(contract)
    causal_audit = _validate_causal_isolation(contract)
    spec = contract["arms"][arm]

    reserved = {key: (root / spec[key]).resolve() for key in RESERVED_PATH_KEYS}
    existing = sorted(key for key, path in reserved.items() if path.exists())
    if existing:
        raise CompositionError(f"reserved arm paths already exist: {existing}")

    dataset, dataset_report = build_arm_dataset(
        root=root, arm=arm, contract=contract, pack=pack
    )
    # `_arm_diff` only catches settings that differ BETWEEN arms. A both-arms
    # drift in the step schedule would place checkpoints where the monitor is
    # not looking, and surface at train begin on the GPU.
    max_steps = spec["config"]["max_steps"]
    save_steps = spec["config"]["save_steps"]
    if smoke_max_steps is not None:
        # A short GPU smoke cannot otherwise reach this path: the production
        # gate below requires the locked 300-step schedule. Smoke mode relaxes
        # the step budget and disables checkpointing outright, so the quality
        # monitor never fires and cannot demand a step the smoke will not
        # reach. The report is stamped so a smoke artifact can never be read as
        # a production composition.
        max_steps = int(smoke_max_steps)
        if not 1 <= max_steps < TRAINING_STEPS:
            raise CompositionError(
                f"smoke step budget {max_steps} must be between 1 and "
                f"{TRAINING_STEPS - 1}"
            )
    elif max_steps != TRAINING_STEPS:
        raise CompositionError(
            f"arm max_steps {max_steps} does not match the locked {TRAINING_STEPS}"
        )
    if (
        smoke_max_steps is None
        and tuple(range(save_steps, max_steps + 1, save_steps)) != tuple(CHECKPOINT_STEPS)
    ):
        raise CompositionError(
            f"save_steps {save_steps} over {max_steps} steps would not produce "
            f"checkpoints {tuple(CHECKPOINT_STEPS)}"
        )

    reward_funcs, reward_report = build_reward_binding(arm, contract)
    callbacks, phase_profiler, callback_report = build_callbacks(
        root=root,
        arm=arm,
        contract=contract,
        callback_base_class=callback_base_class,
        synchronize_fn=synchronize_fn,
        expected_steps=max_steps,
        monitor_contract_path=Path(monitor_contract_path).resolve(),
        monitor_command_builder=monitor_command_builder,
        monitor_runner=monitor_runner,
        gpu_free_bytes_fn=gpu_free_bytes_fn,
    )

    model, tokenizer = model_loader()
    if model is None or tokenizer is None:
        raise CompositionError("model loader returned no model or tokenizer")

    # Phase G's accepted construction proof builds the config from the arm's
    # settings plus the reward weights; composition must use identical kwargs or
    # it validates a different object than the one that was accepted.
    scratch = Path(trainer_scratch_dir).resolve()
    # Exact equality is not enough: a real `Trainer.__init__` calls makedirs
    # with parents, so a scratch dir nested under a reserved path creates that
    # reserved path and permanently fails the launch preflight.
    if any(
        scratch == path or scratch.is_relative_to(path) or path.is_relative_to(scratch)
        for path in reserved.values()
    ):
        raise CompositionError(
            "trainer scratch directory may not be, contain or sit inside a reserved arm path"
        )
    config_kwargs = {**spec["config"], "reward_weights": spec["reward"]["weights"]}
    config_kwargs["output_dir"] = str(scratch)
    config_kwargs["max_steps"] = max_steps
    if smoke_max_steps is not None:
        config_kwargs["save_strategy"] = "no"
    config = config_class(**config_kwargs)
    expected_config = _normalized_expected_config(spec)
    expected_config["output_dir"] = str(scratch)
    expected_config["max_steps"] = max_steps
    if smoke_max_steps is not None:
        expected_config["save_strategy"] = "no"
    try:
        config_inspection = _inspect_constructed_config(config, expected_config)
    except RuntimeError as exc:
        raise CompositionError(f"composed config drifted from contract: {exc}") from exc

    # Measure rather than assert that no reward ran during construction: a
    # trainer that probed its reward functions in __init__ would otherwise be
    # reported as having called none.
    call_counts = {name: 0 for name in reward_report["callable_names"]}

    def _counting(fn: Any, name: str) -> Any:
        @functools.wraps(fn)
        def proxy(*args: Any, **kwargs: Any) -> Any:
            call_counts[name] += 1
            return fn(*args, **kwargs)

        return proxy

    probed_rewards = [
        _counting(fn, name)
        for fn, name in zip(reward_funcs, reward_report["callable_names"])
    ]

    trainer = trainer_class(
        model=model,
        reward_funcs=probed_rewards,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if any(call_counts.values()):
        raise CompositionError(f"rewards were invoked during construction: {call_counts}")

    # Restore the raw callables immediately. The proxies exist only to make
    # "no reward ran during construction" a measurement rather than a claim;
    # leaving them bound would dispatch training against closures whose
    # signature is (*args, **kwargs), and TRL introspects reward signatures to
    # decide which columns to forward. The measurement apparatus must not
    # survive into the artifact under test, and it would also break
    # comparability with the offline G10 selection, which scored these exact
    # functions.
    # Verify before overwriting. An unconditional assignment erases whatever the
    # trainer did to the binding during __init__, and every check downstream then
    # reads back what this line just wrote. A trainer that reordered, dropped or
    # substituted a reward would have been certified as correct: the same silent
    # instrument loss the callback checks exist to catch, applied to the one
    # thing the two arms are supposed to differ in.
    observed = list(trainer.reward_funcs)
    if observed != probed_rewards:
        raise CompositionError(
            "trainer mangled the reward binding during construction: "
            f"{[_callable_name(value) for value in observed]}"
        )
    try:
        trainer.reward_funcs = list(reward_funcs)
    except AttributeError as exc:
        # A trainer that will not accept the raw callables back cannot be
        # dispatched: it would train against the measurement proxies.
        raise CompositionError(f"trainer refused reward restoration: {exc}") from exc
    restored = [_callable_name(value) for value in trainer.reward_funcs]
    if restored != reward_report["callable_names"]:
        raise CompositionError(f"reward restoration left the wrong callables: {restored}")
    if any(value is proxy for value in trainer.reward_funcs for proxy in probed_rewards):
        raise CompositionError("a measurement proxy survived into the dispatched trainer")

    retention = _assert_trainer_retained(
        trainer,
        model=model,
        config=config,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
    )
    dataset_retained = getattr(trainer, "train_dataset", dataset) is dataset
    if not dataset_retained:
        # Run 1's real gate checks dataset CONTENT, not identity, because TRL
        # may return a mapped copy. Identity is reported, content is enforced.
        materialized = list(getattr(trainer, "train_dataset")["sku_id"])
        if materialized != list(dataset["sku_id"]):
            raise CompositionError("trainer dataset content drifted from the schedule")

    instrumentation = _instrument_and_verify(phase_profiler, trainer)
    for callback in callbacks:
        trainer.add_callback(callback)
    callback_report |= _assert_callbacks_survived(trainer, callbacks)
    trainer_report = _assert_trainer_boundaries(trainer, reward_report) | retention

    created = sorted(key for key, path in reserved.items() if path.exists())
    if created:
        raise CompositionError(f"composition created reserved arm paths: {created}")
    callback_report["monitor_paths_created"] = False

    return {
        "version": VERSION,
        "status": (
            "arm_runtime_smoke_composed"
            if smoke_max_steps is not None
            else "arm_runtime_composed_no_training_dispatched"
        ),
        "smoke_mode": smoke_max_steps is not None,
        "step_schedule": {
            "max_steps": max_steps,
            "checkpointing": "disabled" if smoke_max_steps is not None else "epoch/steps",
        },
        "arm": arm,
        "role": spec["role"],
        "contract_lineage": lineage,
        "causal_difference_audit": causal_audit,
        "dataset_binding": dataset_report | {"retained_by_identity": dataset_retained},
        "reward_binding": reward_report | {"observed_reward_calls": call_counts},
        "config_binding": config_inspection,
        "callback_binding": callback_report,
        "profiler_binding": instrumentation,
        "trainer_binding": trainer_report,
        "reserved_paths": {key: str(path) for key, path in reserved.items()},
        "reserved_paths_created": False,
        "trainer_scratch_dir": str(scratch),
        "boundaries": {
            "rewards_observed_called": sum(call_counts.values()),
            "monitor_process_started": False,
            "callback_lifecycle_started": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_dispatched": False,
        },
    }, trainer
