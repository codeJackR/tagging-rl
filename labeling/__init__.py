"""W1 Step 3 — the answer key and the practice pile.

    from labeling import Row, AttributeLabel, LabelStatus
    from labeling import consensus, reliability, splits, lengths, review, freeze

The one-line version of why this package is shaped the way it is: in RLVR the
labels *are* the reward function, so label noise is not an accuracy problem but a
reward-specification problem. Everything here exists to find and quantify that
noise before it is baked into a training run.
"""

from . import consensus, freeze, lengths, reliability, review, splits
from .records import (
    AttributeLabel,
    Difficulty,
    LabelStatus,
    LengthStats,
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    canonical_line,
    from_verifier_record,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "AttributeLabel",
    "Difficulty",
    "LabelStatus",
    "LengthStats",
    "Provenance",
    "Row",
    "RowInput",
    "SelfConsistency",
    "canonical_line",
    "consensus",
    "freeze",
    "from_verifier_record",
    "lengths",
    "read_jsonl",
    "reliability",
    "review",
    "splits",
    "write_jsonl",
]
