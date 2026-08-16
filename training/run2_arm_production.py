#!/usr/bin/env python3
"""Shared production runtime for one Run 2 causal arm.

Both arms run this code. The arm is the only input, and each arm's entry point
supplies it as a constant, so an operator cannot select one. Everything else --
model, config class, trainer class, callbacks, monitor runner, rollout collector
-- is fixed here.

One implementation rather than two near-identical ones is a deliberate choice
for this experiment specifically. Two copies would be free to drift, and drift
between the arms is the precise confound the causal contract exists to prevent.
If the arms are meant to differ only in their reward, the code that runs them
should differ only in an argument.

Smoke mode is deliberately absent. The one-step smoke lives in its own path and
must never be reachable from a production entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

VERSION = "grpo-run2-arm-production-v1"
DEFAULT_PACK = "packs/vastraa_taste_v1"
# v2 supersedes v1 after the quality-policy amendment; v1 is retained
# unchanged as the original predeclaration. See W2_GRPO_RUN2_POLICY_AMENDMENT.md.
DEFAULT_CONTRACT = "runs/grpo-run2-causal-experiment-contract-v2.json"
DEFAULT_MONITOR_CONTRACT = "runs/grpo-run2-checkpoint-monitor-contract.json"
MAX_SEQUENCE_LENGTH = 896


def build_report(*, repo_root: Path, scratch_root: Path, arm: str) -> dict[str, Any]:
    """Load the real stack, run one arm to completion and publish its bundle."""
    # Unsloth must patch the stack before transformers or TRL is imported.
    from unsloth import FastLanguageModel

    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from training.run2_arm_workload import run_arm
    from training.run2_causal_experiment import (
        BASE_MODEL,
        CHECKPOINT_STEPS,
        STARTING_ADAPTER,
    )
    from training.run2_checkpoint_monitor_control import run_supervised_monitor
    from training.train_grpo import FullRunRolloutCollector
    from verifier import load_pack

    contract = json.loads(
        (repo_root / DEFAULT_CONTRACT).read_text(encoding="utf-8")
    )
    pack = load_pack(repo_root / DEFAULT_PACK)
    monitor_contract = (repo_root / DEFAULT_MONITOR_CONTRACT).resolve()

    def load_model() -> tuple[Any, Any]:
        # Load the adapter path directly. Attaching it to a fresh base with
        # `load_adapter` yields an inference-mode model with zero trainable
        # parameters: it completes a run, reports a loss, and learns nothing.
        # The one-step smoke caught exactly that.
        return FastLanguageModel.from_pretrained(
            model_name=str(repo_root / STARTING_ADAPTER),
            max_seq_length=MAX_SEQUENCE_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing="unsloth",
            fast_inference=False,
        )

    trainer_root = (scratch_root / "trainer").resolve()
    monitor_root = (repo_root / contract["arms"][arm]["monitor_root"]).resolve()

    def monitor_command(step: int, checkpoint: Path, output: Path) -> list[str]:
        """Build the evaluator command for one checkpoint.

        The launcher's `build_monitor_command` cannot be used here: it requires
        the checkpoint to sit at the reserved production path, while the
        trainer writes into scratch precisely so that a real `Trainer.__init__`
        cannot create that reserved path and permanently fail the preflight.
        The two constraints are incompatible, so the path assertions are made
        against where the trainer actually writes. Which checkpoint is being
        evaluated is not left to the path: the coordinator independently
        verifies the adapter's SHA-256.
        """
        if step not in CHECKPOINT_STEPS:
            raise RuntimeError(f"monitor step {step} is not a contract checkpoint")
        if checkpoint.resolve() != trainer_root / f"checkpoint-{step}":
            raise RuntimeError(f"monitor checkpoint path drifted: {checkpoint}")
        if output.resolve() != monitor_root / f"checkpoint-{step}":
            raise RuntimeError(f"monitor output path drifted: {output}")
        return [
            sys.executable,
            "-m",
            "training.run2_checkpoint_monitor_runtime",
            "--repo-root",
            str(repo_root),
            "--contract",
            str(monitor_contract.relative_to(repo_root)),
            "--pack",
            DEFAULT_PACK,
            "--base-model",
            BASE_MODEL,
            "--local-files-only",
            "evaluate",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--output",
            str(output.resolve()),
            "--mode",
            "production",
        ]

    return run_arm(
        root=repo_root,
        arm=arm,
        contract=contract,
        pack=pack,
        model_loader=load_model,
        config_class=GRPOConfig,
        trainer_class=GRPOTrainer,
        callback_base_class=TrainerCallback,
        synchronize_fn=torch.cuda.synchronize,
        monitor_contract_path=monitor_contract,
        monitor_command_builder=monitor_command,
        monitor_runner=run_supervised_monitor,
        gpu_free_bytes_fn=lambda: torch.cuda.mem_get_info()[0],
        scratch_root=scratch_root,
        rollout_collector=FullRunRolloutCollector(),
    )


def run_arm_cli(arm: str, argv: Sequence[str] | None = None) -> int:
    """One arm's command line. The arm is supplied by the caller, never parsed."""
    parser = argparse.ArgumentParser(description=f"Run Run 2 causal arm {arm}.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--scratch-root",
        required=True,
        help="working directory for trainer output; never a reserved arm path",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    scratch_root = Path(args.scratch_root).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    report = build_report(repo_root=repo_root, scratch_root=scratch_root, arm=arm)
    print(
        json.dumps(
            {
                "version": VERSION,
                "arm": arm,
                "status": report["status"],
                "evidence": report["evidence"],
                "published_to": report["published_to"],
                "wall_seconds": round(report["wall_seconds"], 1),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
