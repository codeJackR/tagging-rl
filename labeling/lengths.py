"""Token budgets — measured now, so W2 does not lose a day to a length bug.

The failure this prevents
------------------------
A smoke-test run set `max_completion_length=256` and clipped 50-87% of
completions. The format reward became structurally unreachable: the model could
not emit a closing brace inside the budget, so validity never fired. The symptom
looked exactly like a broken reward function. It was a length budget.

Measure the distribution while the data is being built — it costs one pass — and
set the budgets from p95/p99 rather than from a round number that felt safe.

Which tokenizer
---------------
These numbers exist to size TRL's `max_prompt_length` / `max_completion_length`,
which are counted by the **policy model's** tokenizer (Qwen2.5 for W2), not by the
labeler's. Counting with the wrong tokenizer produces a plausible number that is
quietly wrong by 10-30%.

`HeuristicTokenizer` exists so the pipeline runs before `transformers` is
installed, but it stamps its identity into every `LengthStats` it writes, and
`recommend_budgets` refuses to hand back a budget derived from it without saying
so. An estimate must never be mistaken for a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol

from .records import Row

DEFAULT_POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """chars/4. Fine for a smoke test, not for setting a real budget."""

    def __init__(self, divisor: float = 4.0):
        self.divisor = divisor
        self.name = f"heuristic:chars-over-{divisor:g}"

    def count(self, text: str) -> int:
        return max(1, round(len(text) / self.divisor))


class HFTokenizer:
    """The real thing. Requires `transformers` and a one-time model download."""

    def __init__(self, model: str = DEFAULT_POLICY_MODEL):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is not installed — `uv add --optional tokenizer "
                "transformers`, or pass HeuristicTokenizer() and accept that the "
                "budget is an estimate."
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(model)
        self.name = model

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


def get_tokenizer(spec: str | None) -> Tokenizer:
    if spec in (None, "", "heuristic"):
        return HeuristicTokenizer()
    return HFTokenizer(spec)


# --- canonical renderers ------------------------------------------------------
# These define what the model reads and what it must produce. W2's dataset builder
# must import these rather than re-implement them: if training renders a prompt
# differently from how the budget was measured, the budget is measuring nothing.


def render_prompt(row: Row) -> str:
    parts = [f"Title: {row.input.title}"]
    if row.input.brand:
        parts.append(f"Brand: {row.input.brand}")
    if row.input.category:
        parts.append(f"Category: {row.input.category}")
    if row.input.description:
        parts.append(f"Description: {row.input.description}")
    if row.input.raw_tags:
        parts.append(f"Tags: {', '.join(row.input.raw_tags)}")
    return "\n".join(parts)


def render_target(row: Row, pack=None) -> str:
    return json.dumps(
        row.to_verifier_record(pack), sort_keys=True, ensure_ascii=False
    )


def _pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


@dataclass
class LengthReport:
    tokenizer: str
    n: int
    prompt: dict[str, int]
    target: dict[str, int]
    measured: bool

    def recommend(self, *, reasoning_block: bool = False) -> dict:
        """Budgets, with headroom, and a loud caveat if they came from a guess."""
        completion = self.target["p99"]
        if reasoning_block:
            # A <think> block roughly doubles output; the smoke-test run saw
            # 400-800 tokens for a 20-attribute record with reasoning.
            completion = int(completion * 2.2)
        rec = {
            "max_prompt_length": int(self.prompt["p95"] * 1.15) + 16,
            "max_completion_length": int(completion * 1.20) + 32,
            "basis": {
                "prompt_p95": self.prompt["p95"],
                "target_p99": self.target["p99"],
                "reasoning_block": reasoning_block,
            },
        }
        if not self.measured:
            rec["WARNING"] = (
                f"derived from {self.tokenizer!r}, not a real tokenizer — re-run with "
                f"--tokenizer {DEFAULT_POLICY_MODEL} before setting the GRPO config"
            )
        return rec


def length_report(
    rows: list[Row],
    tokenizer: Tokenizer,
    *,
    prompt_fn: Callable[[Row], str] = render_prompt,
    target_fn: Callable[[Row], str] = render_target,
) -> LengthReport:
    p = [tokenizer.count(prompt_fn(r)) for r in rows]
    t = [tokenizer.count(target_fn(r)) for r in rows]
    dist = lambda v: {  # noqa: E731
        "p50": _pct(v, 0.50),
        "p95": _pct(v, 0.95),
        "p99": _pct(v, 0.99),
        "max": max(v) if v else 0,
        "mean": round(sum(v) / len(v)) if v else 0,
    }
    return LengthReport(
        tokenizer=tokenizer.name,
        n=len(rows),
        prompt=dist(p),
        target=dist(t),
        measured=not tokenizer.name.startswith("heuristic:"),
    )


def stamp_rows(rows: list[Row], tokenizer: Tokenizer) -> None:
    """Write per-row token counts into `length_stats`, tokenizer identity included."""
    for row in rows:
        row.length_stats.prompt_tokens = tokenizer.count(render_prompt(row))
        row.length_stats.target_tokens = tokenizer.count(render_target(row))
        row.length_stats.tokenizer = tokenizer.name


def format_report(report: LengthReport, *, reasoning_block: bool = False) -> str:
    rec = report.recommend(reasoning_block=reasoning_block)
    lines = [
        f"token lengths — {report.n} rows, tokenizer {report.tokenizer!r}",
        f"  {'':<10} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} {'mean':>6}",
        "  " + "-" * 48,
    ]
    for label, d in (("prompt", report.prompt), ("target", report.target)):
        lines.append(
            f"  {label:<10} {d['p50']:>6} {d['p95']:>6} {d['p99']:>6} "
            f"{d['max']:>6} {d['mean']:>6}"
        )
    lines.append(
        f"\n  max_prompt_length     = {rec['max_prompt_length']}   (p95 + 15%)"
    )
    lines.append(
        f"  max_completion_length = {rec['max_completion_length']}   (p99 + 20%"
        + (", x2.2 for a reasoning block)" if reasoning_block else ")")
    )
    if "WARNING" in rec:
        lines.append(f"\n  WARNING: {rec['WARNING']}")
    return "\n".join(lines) + "\n"
