"""Both production entry points must differ only in which arm they name.

There is no GPU here, so these cover what CPU can decide: that neither entry
point exposes an experimental knob, that smoke mode is unreachable from either,
and that the two files are byte-identical apart from the arm and its prose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training.run2_arm_a_production import ARM as ARM_A, main as main_a
from training.run2_arm_b_production import ARM as ARM_B, main as main_b
from training.run2_arm_production import VERSION, run_arm_cli

ROOT = Path(__file__).resolve().parent.parent
ENTRY_A = ROOT / "training" / "run2_arm_a_production.py"
ENTRY_B = ROOT / "training" / "run2_arm_b_production.py"
CORE = ROOT / "training" / "run2_arm_production.py"


def test_each_entry_point_names_exactly_one_arm():
    assert ARM_A == "A" and ARM_B == "B"
    assert VERSION == "grpo-run2-arm-production-v1"


def _code_without_docstring(path: Path) -> str:
    """Executable source only, so prose differences do not count as drift."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_the_two_entry_points_are_identical_except_for_the_arm():
    """Two near-copies would be free to drift, and drift between arms is the
    exact confound the causal contract exists to prevent."""
    a = _code_without_docstring(ENTRY_A).replace("'A'", "<ARM>")
    b = _code_without_docstring(ENTRY_B).replace("'B'", "<ARM>")
    assert a == b


def test_neither_entry_point_lets_an_operator_choose_the_arm(capsys):
    for main in (main_a, main_b):
        with pytest.raises(SystemExit):
            main(["--arm", "B", "--scratch-root", "/tmp/x"])


def test_scratch_root_is_required():
    with pytest.raises(SystemExit):
        run_arm_cli("A", ["--repo-root", "."])


def test_smoke_mode_is_unreachable_from_production():
    """A production launcher able to run one step is a launcher able to publish
    a one-step bundle as if it were a real arm."""
    for path in (ENTRY_A, ENTRY_B, CORE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keywords = [n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg]
        assert "smoke_max_steps" not in keywords, path.name


def test_no_gpu_import_at_module_scope():
    """Importing must not pull in torch, TRL or Unsloth, so the CLI contract
    stays testable and an import error cannot masquerade as a run."""
    for path in (ENTRY_A, ENTRY_B, CORE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
        assert not names & {"torch", "trl", "unsloth", "transformers"}, path.name


def test_the_real_monitor_runner_and_collector_are_bound():
    """A production run with a stubbed monitor would train blind."""
    source = CORE.read_text(encoding="utf-8")
    assert "monitor_runner=run_supervised_monitor" in source
    assert "FullRunRolloutCollector(" in source
    # and it must be told this arm's reward identity, not left on the defaults
    assert "expected_reward_names=" in source
    # `smoke` may appear in prose explaining what the smoke test found. What
    # must not appear is an identifier, keyword or attribute reaching it, so
    # inspect names rather than the raw text.
    tree = ast.parse(source)
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    assert not any("smoke" in name.lower() for name in identifiers)
