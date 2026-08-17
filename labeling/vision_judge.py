#!/usr/bin/env python3
"""Judge vision-produced attribute answers with models that did not produce them.

The project has no accuracy number because `gpt-5.6-luna` labelled every dataset
and `gpt-5.6-luna` is the production model, so every score is a model agreeing
with itself. This module exists to not repeat that.

**The rule is that a judge may never be the answerer.** Gemini answered from the
photographs, so the judges are OpenAI models. Same-lab judges would be a weaker
panel than three labs, and the roster here is what the available keys allow;
adding an Anthropic key would improve it and nothing else needs to change.

Two design choices are load-bearing.

**Judges are shown the vocabulary definition, aliases included.** A full-length
pant was labelled `maxi` and looked wrong to a human reviewer who knew trousers
better than the schema. The vocabulary defines `maxi` as covering "full length"
and "ankle length", so the answer was right and the review was wrong. A judge
without the aliases reproduces that error confidently.

**A judge may answer `unsure`.** Forcing a verdict on a cell that is genuinely
ambiguous from one photograph manufactures disagreement and hides the real
signal, which is *where* the panel splits.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

VERSION = "vision-judge-v1"

VERDICTS = ("correct", "incorrect", "unsure")


@dataclass
class Verdict:
    sku_id: str
    attribute: str
    proposed: str
    judge: str
    verdict: str
    correction: str | None = None
    note: str = ""


@dataclass
class PanelResult:
    judges: list[str]
    cells: int = 0
    unanimous_correct: int = 0
    unanimous_incorrect: int = 0
    split: int = 0
    any_unsure: int = 0
    per_attribute: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        decided = self.unanimous_correct + self.unanimous_incorrect
        return {
            "version": VERSION,
            "judges": self.judges,
            "answerer_excluded_from_panel": True,
            "cells": self.cells,
            "unanimous_correct": self.unanimous_correct,
            "unanimous_incorrect": self.unanimous_incorrect,
            "split": self.split,
            "any_unsure": self.any_unsure,
            # The headline. Deliberately over decided cells only: a cell the
            # panel could not agree on is unmeasured, and counting it either way
            # would be an invention.
            "accuracy_on_decided": (
                round(self.unanimous_correct / decided, 4) if decided else 0.0
            ),
            "decided_share": (
                round(decided / self.cells, 4) if self.cells else 0.0
            ),
            "per_attribute": {
                name: {
                    **stats,
                    "accuracy_on_decided": (
                        round(stats["correct"] / (stats["correct"] + stats["incorrect"]), 4)
                        if (stats["correct"] + stats["incorrect"]) else 0.0
                    ),
                }
                for name, stats in sorted(self.per_attribute.items())
            },
            "caveat": (
                "Judges are OpenAI models; the answerer was Gemini, so no judge "
                "graded its own output. Both judges share a lab, which is a "
                "weaker panel than three independent labs."
            ),
        }


def vocabulary_brief(pack, attribute: str) -> str:
    """The permitted values and their aliases, so a judge grades the schema.

    Without the aliases a judge grades its own intuition about what a word
    means, which is exactly how the `maxi` pant became a false error.
    """
    spec = pack.specs[attribute]
    lines = []
    for value in spec.values:
        aliases = [a for a in getattr(spec, "aliases", {}) or {} if
                   (getattr(spec, "aliases", {}) or {}).get(a) == value]
        lines.append(f"- {value}" + (f"  (also means: {', '.join(aliases)})" if aliases else ""))
    applies = sorted(spec.applies_to) if spec.applies_to else None
    tail = f"\nApplies only to: {', '.join(applies)}" if applies else ""
    return "\n".join(lines) + tail


def build_prompt(pack, attribute: str, proposed: str, title: str) -> str:
    return (
        f"You are checking one attribute on one apparel product, against a fixed "
        f"vocabulary. Judge the ANSWER against the PHOTOGRAPH and the vocabulary "
        f"below. Do not judge it against your own sense of what the word usually "
        f"means; the vocabulary's definition governs.\n\n"
        f"Product title: {title}\n"
        f"Attribute: {attribute}\n"
        f"Proposed answer: {proposed}\n\n"
        f"Permitted values for {attribute}:\n{vocabulary_brief(pack, attribute)}\n\n"
        f"Reply with JSON only:\n"
        f'{{"verdict": "correct" | "incorrect" | "unsure", '
        f'"correction": "<a permitted value, only if incorrect>", '
        f'"note": "<up to 12 words>"}}\n\n'
        f'Use "unsure" when the photograph genuinely cannot settle it. Guessing '
        f"on an ambiguous cell hides where the real uncertainty is."
    )


def parse_verdict(raw: str) -> tuple[str, str | None, str]:
    """Pull the verdict out of a reply, tolerating fences and stray prose."""
    text = (raw or "").strip()
    if "```" in text:
        text = text.split("```")[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return "unsure", None, "unparseable reply"
    try:
        body = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return "unsure", None, "unparseable reply"
    verdict = str(body.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return "unsure", None, f"unknown verdict {verdict!r}"
    correction = body.get("correction") or None
    return verdict, (str(correction) if correction else None), str(body.get("note", ""))[:80]


def tally(verdicts: Sequence[Verdict], judges: Sequence[str]) -> PanelResult:
    """Fold per-judge verdicts into a panel result, keyed on (sku, attribute)."""
    result = PanelResult(judges=list(judges))
    by_cell: dict[tuple[str, str], list[Verdict]] = {}
    for v in verdicts:
        by_cell.setdefault((v.sku_id, v.attribute), []).append(v)

    for (_sku, attribute), cell in by_cell.items():
        if len(cell) < len(judges):
            continue  # a cell only one judge saw is not a panel verdict
        result.cells += 1
        stats = result.per_attribute.setdefault(
            attribute, {"cells": 0, "correct": 0, "incorrect": 0, "split": 0, "unsure": 0}
        )
        stats["cells"] += 1
        calls = {v.verdict for v in cell}
        if "unsure" in calls:
            result.any_unsure += 1
            stats["unsure"] += 1
        elif calls == {"correct"}:
            result.unanimous_correct += 1
            stats["correct"] += 1
        elif calls == {"incorrect"}:
            result.unanimous_incorrect += 1
            stats["incorrect"] += 1
        else:
            result.split += 1
            stats["split"] += 1
    return result


def judge_cells(
    cells: Iterable[dict],
    pack,
    judges: Sequence[str],
    ask: Callable[[str, str, bytes], str],
) -> list[Verdict]:
    """Ask every judge about every cell, independently.

    `ask(judge, prompt, image_bytes) -> raw reply`. Judges never see each
    other's verdicts; a panel whose members can read each other is one judge
    with extra steps.
    """
    out: list[Verdict] = []
    for cell in cells:
        image = Path(cell["image_path"]).read_bytes()
        for judge in judges:
            prompt = build_prompt(pack, cell["attribute"], cell["proposed"], cell["title"])
            verdict, correction, note = parse_verdict(ask(judge, prompt, image))
            out.append(
                Verdict(
                    sku_id=cell["sku_id"], attribute=cell["attribute"],
                    proposed=cell["proposed"], judge=judge,
                    verdict=verdict, correction=correction, note=note,
                )
            )
    return out


def encode_image(data: bytes) -> str:
    return base64.b64encode(data).decode()
