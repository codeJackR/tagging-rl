#!/usr/bin/env python3
"""Production entry point for Run 2 Arm B: the dense-reward treatment.

The arm is a constant here and is not exposed on the command line. Running the
wrong arm should require editing a committed file, not passing a flag.

Arm B changes exactly one thing against Arm A: the reward becomes Candidate UA,
which scores each of the fifteen fields separately instead of grading the record
as a whole. Every other setting, the product schedule and its order, the seed and
the quality policy are identical.
"""

from __future__ import annotations

from collections.abc import Sequence

from training.run2_arm_production import run_arm_cli

ARM = "B"


def main(argv: Sequence[str] | None = None) -> int:
    return run_arm_cli(ARM, argv)


if __name__ == "__main__":
    raise SystemExit(main())
