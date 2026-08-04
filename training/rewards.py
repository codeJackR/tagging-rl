"""CPU-only reward functions for the first unconstrained GRPO run.

The functions in this module intentionally know nothing about a model or trainer.
TRL supplies generated completions plus the dataset's hidden ``gold`` column;
the existing verifier remains the single source of truth for grading.

The first run keeps three binary components separate so TRL/W&B can report each
one independently. ``FIRST_RUN_REWARD_WEIGHTS`` supplies the proposed 1:1:2
combination without baking those weights into the component outputs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from training.score_difficulty import score_completion
from verifier import Pack, load_pack, verify

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK_PATH = ROOT / "packs" / "vastraa_taste_v1"


@lru_cache(maxsize=1)
def _default_pack() -> Pack:
    """Load the W2 schema pack once per reward-worker process."""
    return load_pack(DEFAULT_PACK_PATH)


def _resolve_pack(pack: Pack | None) -> Pack:
    return pack if pack is not None else _default_pack()


def _completion_text(completion: Any) -> str:
    """Extract raw assistant text from TRL's standard or conversational shape.

    A malformed container is a trainer integration bug, not a low-quality model
    answer, so it raises instead of silently assigning a zero reward.
    """
    if isinstance(completion, str):
        return completion

    if (
        isinstance(completion, list)
        and len(completion) == 1
        and isinstance(completion[0], dict)
        and completion[0].get("role") == "assistant"
        and isinstance(completion[0].get("content"), str)
    ):
        return completion[0]["content"]

    raise TypeError(
        "each completion must be raw text or one assistant message with string content"
    )


def _completion_texts(completions: Sequence[Any]) -> list[str]:
    if isinstance(completions, (str, bytes)):
        raise TypeError("completions must be a sequence, not one string")
    return [_completion_text(completion) for completion in completions]


def _gold_record(value: Any, *, index: int) -> dict:
    """Parse one trusted hidden-gold value and fail loudly if it is corrupted."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gold[{index}] is not valid JSON") from exc

    if not isinstance(value, dict):
        raise TypeError(f"gold[{index}] must decode to a JSON object")
    return value


def format_validity_reward(
    completions: Sequence[Any],
    *,
    pack: Pack | None = None,
    **_: Any,
) -> list[float]:
    """Return 1 when literal output satisfies the structural schema, else 0."""
    resolved_pack = _resolve_pack(pack)
    return [
        float(verify(text, resolved_pack).schema_valid)
        for text in _completion_texts(completions)
    ]


def vocab_rule_compliance_reward(
    completions: Sequence[Any],
    *,
    pack: Pack | None = None,
    **_: Any,
) -> list[float]:
    """Return 1 for schema-, vocabulary- and rule-clean output, else 0."""
    resolved_pack = _resolve_pack(pack)
    return [
        float(verify(text, resolved_pack).ok)
        for text in _completion_texts(completions)
    ]


def golden_agreement_reward(
    completions: Sequence[Any],
    gold: Sequence[Any],
    *,
    pack: Pack | None = None,
    **_: Any,
) -> list[float]:
    """Return 1 for exact verifier-clean agreement on every known-gold field.

    ``score_completion`` deliberately excludes gold ``unknown`` fields from the
    answer key. Abstention behavior remains available through separate metrics;
    it is not silently converted into fabricated ground truth here.
    """
    texts = _completion_texts(completions)
    if isinstance(gold, (str, bytes)):
        raise TypeError("gold must be a sequence aligned with completions")
    if len(gold) != len(texts):
        raise ValueError(
            "gold and completions must have the same length: "
            f"got {len(gold)} and {len(texts)}"
        )

    resolved_pack = _resolve_pack(pack)
    return [
        float(score_completion(text, _gold_record(answer, index=index), resolved_pack).passed)
        for index, (text, answer) in enumerate(zip(texts, gold))
    ]


# Order is part of the contract: GRPOConfig.reward_weights must align with the
# reward function list supplied to GRPOTrainer.
FIRST_RUN_REWARD_FUNCTIONS = (
    format_validity_reward,
    vocab_rule_compliance_reward,
    golden_agreement_reward,
)
FIRST_RUN_REWARD_WEIGHTS = (1.0, 1.0, 2.0)
