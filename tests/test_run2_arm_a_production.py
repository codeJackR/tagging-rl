"""The production entry point must not be able to change the experiment.

There is no GPU here, so these tests cover exactly what CPU can decide: that
the CLI exposes no experimental knob, that smoke mode is unreachable, and that
the module imports nothing GPU-bearing until it is actually run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training.run2_arm_a_production import ARM, VERSION, parse_args

MODULE = Path(__file__).resolve().parent.parent / "training" / "run2_arm_a_production.py"


def test_only_paths_are_configurable():
    """No arm, step budget, reward or output-path option may exist."""
    args = parse_args(["--repo-root", ".", "--scratch-root", "/tmp/x"])
    assert set(vars(args)) == {"repo_root", "scratch_root"}
    assert ARM == "A"
    assert VERSION == "grpo-run2-arm-a-production-v1"


def test_scratch_root_is_required():
    with pytest.raises(SystemExit):
        parse_args(["--repo-root", "."])


def test_smoke_mode_is_unreachable_from_production():
    """A production launcher that can run one step is a launcher that can
    publish a one-step bundle as if it were the control arm."""
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    keywords = [
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg
    ]
    assert "smoke_max_steps" not in keywords
    assert "smoke" not in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }


def test_no_gpu_import_at_module_scope():
    """Importing the module must not pull in torch, TRL or Unsloth, so the CLI
    contract stays testable and an import error cannot masquerade as a run."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    names = {
        alias.name.split(".")[0]
        for node in top_level
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in top_level
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not names & {"torch", "trl", "unsloth", "transformers", "training"}


def test_the_real_monitor_runner_is_bound():
    """A production run with a stubbed monitor would train blind."""
    source = MODULE.read_text(encoding="utf-8")
    assert "monitor_runner=run_supervised_monitor" in source
    assert "FullRunRolloutCollector()" in source
