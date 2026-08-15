#!/usr/bin/env python3
"""Production entry point for the Run 2 Arm A corrected control.

This is the executable that a detached launcher runs. It fixes every real
collaborator that `training.run2_arm_workload.run_arm` takes as a parameter and
exposes no options that could change the experiment: no arm selection, no step
budget, no reward choice, no output path. Anything worth deciding was decided in
the causal contract and is asserted there.

It is committed rather than kept as a scratch script because its artifacts are
the control arm's evidence. A run whose launcher exists only in `/tmp` cannot be
reproduced from the repository, which is the standard every other Run 2 artifact
is held to.

Smoke mode is deliberately absent. The one-step smoke lives in its own path and
must never be reachable from the production entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ARM = "A"
VERSION = "grpo-run2-arm-a-production-v1"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_CONTRACT = "runs/grpo-run2-causal-experiment-contract.json"
DEFAULT_MONITOR_CONTRACT = "runs/grpo-run2-checkpoint-monitor-contract.json"
MAX_SEQUENCE_LENGTH = 896


def build_report(*, repo_root: Path, scratch_root: Path) -> dict[str, Any]:
    """Load the real stack, run Arm A to completion and publish its bundle."""
    # Unsloth must patch the stack before transformers or TRL is imported.
    from unsloth import FastLanguageModel

    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from training.run2_arm_a_launcher import build_monitor_command
    from training.run2_arm_workload import run_arm
    from training.run2_causal_experiment import BASE_MODEL, STARTING_ADAPTER
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

    def monitor_command(step: int, checkpoint: Path, output: Path) -> list[str]:
        return build_monitor_command(
            root=repo_root,
            monitor_contract=monitor_contract,
            pack_path=(repo_root / DEFAULT_PACK).resolve(),
            step=step,
            checkpoint=checkpoint,
            output=output,
            python_executable=sys.executable,
        )

    return run_arm(
        root=repo_root,
        arm=ARM,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--scratch-root",
        required=True,
        help="working directory for trainer output; never a reserved arm path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    scratch_root = Path(args.scratch_root).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    report = build_report(repo_root=repo_root, scratch_root=scratch_root)
    print(
        json.dumps(
            {
                "version": VERSION,
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


if __name__ == "__main__":
    raise SystemExit(main())
