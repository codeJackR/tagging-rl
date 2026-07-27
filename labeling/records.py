"""The dataset row schema — one product, one set of labels, full provenance.

Three-state labels, not two
---------------------------
Every attribute carries a `status`, not just a value:

    labeled          a real claim          {"value": "crew", "status": "labeled"}
    not_applicable   the question does not apply to this garment
    unknown          the listing does not say

Collapsing the last two is the failure this schema exists to prevent. A sleeveless
dress genuinely has no sleeve_length; a listing with no fabric line genuinely has
no fabric. If both become `null`, a correct not-applicable prediction scores as
wrong — and it scores as wrong precisely on the rare rows macro-F1 weights most.

This maps 1:1 onto the W1 Step 2 verifier conventions, so no translation layer is
needed anywhere else:

    labeled        -> the value                     (a claim)
    not_applicable -> null                          (field does not apply)
    unknown        -> the pack's abstain token      (declined)

Which means the labeling target format and the W2 training target format are the
same format. `to_verifier_record()` / `from_verifier_record()` are the bridge.
"""

from __future__ import annotations

import json
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LabelStatus(str, Enum):
    LABELED = "labeled"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class AttributeLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | list[str] | None = None
    status: LabelStatus

    def model_post_init(self, _ctx) -> None:
        if self.status is LabelStatus.LABELED and self.value in (None, [], ""):
            raise ValueError("status 'labeled' requires a value")
        if self.status is not LabelStatus.LABELED and self.value not in (None, []):
            raise ValueError(f"status {self.status.value!r} must not carry a value")

    def key(self) -> tuple:
        """Comparable identity — order-insensitive for multi-valued fields."""
        if isinstance(self.value, list):
            return (self.status.value, tuple(sorted(self.value)))
        return (self.status.value, self.value)


class RowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str = ""
    raw_tags: list[str] = Field(default_factory=list)
    brand: str | None = None
    category: str | None = None
    image_url: str | None = None  # stored, unused — vision is the optional W6 track


class SelfConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    agreement: dict[str, float] = Field(default_factory=dict)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labeler: str  # "<model>@<version>"
    prompt_version: str
    human_corrected: bool = False
    corrected_at: str | None = None
    self_consistency: SelfConsistency | None = None

    # The untouched frontier output, snapshotted before any human edit.
    # Without this the per-attribute reliability table (the highest-value artifact
    # of this step) cannot be computed — once a cell is corrected in place, the
    # evidence that it was ever wrong is gone.
    frontier_labels: dict[str, AttributeLabel] | None = None


class LengthStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    target_tokens: int | None = None
    # Which tokenizer produced these. A heuristic estimate must never be mistaken
    # for a measured count — the W2 length budget depends on it.
    tokenizer: str | None = None


class Difficulty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Fraction of k rollouts from the W2 SFT baseline that get this row right.
    # Deliberately null until that baseline exists. GRPO's gradient comes from
    # within-group disagreement, so rows at 0.0 or 1.0 contribute nothing — but
    # that can only be measured against a model, never guessed from the data.
    sft_pass_rate: float | None = None


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str
    source: str  # "sovrn" | "shopify:store.com" | "synthetic"
    split: Literal["eval", "probe", "train"]
    input: RowInput
    labels: dict[str, AttributeLabel] = Field(default_factory=dict)
    provenance: Provenance
    length_stats: LengthStats = Field(default_factory=LengthStats)
    difficulty: Difficulty = Field(default_factory=Difficulty)

    # ---- bridge to the W1 Step 2 verifier ----------------------------------

    def to_verifier_record(self, pack=None, unknown_token: str = "unknown") -> dict[str, Any]:
        """Flatten to the shape `verifier.verify_record()` grades.

        `pack` is optional but strongly preferred: abstention has to respect the
        field's arity. A multi-valued field declines with `["unknown"]`, not
        `"unknown"` — a bare string in a list slot is a structural error, and
        without the pack there is no way to know which fields are multi-valued.
        """
        if pack is not None:
            unknown_token = pack.unknown_token
        out: dict[str, Any] = {}
        for name, lab in self.labels.items():
            if lab.status is LabelStatus.LABELED:
                out[name] = lab.value
            elif lab.status is LabelStatus.NOT_APPLICABLE:
                out[name] = None
            else:
                spec = pack.specs.get(name) if pack is not None else None
                multi = spec is not None and spec.kind == "multi"
                out[name] = [unknown_token] if multi else unknown_token
        return out

    def mark_corrected(self, when: date | str | None = None) -> None:
        self.provenance.human_corrected = True
        self.provenance.corrected_at = str(when or date.today())


def from_verifier_record(
    record: dict[str, Any], unknown_token: str = "unknown"
) -> dict[str, AttributeLabel]:
    """Inverse of `Row.to_verifier_record` — model output to three-state labels."""
    labels: dict[str, AttributeLabel] = {}
    for name, value in record.items():
        if value is None:
            labels[name] = AttributeLabel(status=LabelStatus.NOT_APPLICABLE)
        elif value == unknown_token or (
            isinstance(value, list) and value == [unknown_token]
        ):
            labels[name] = AttributeLabel(status=LabelStatus.UNKNOWN)
        else:
            labels[name] = AttributeLabel(value=value, status=LabelStatus.LABELED)
    return labels


# ---- JSONL i/o, canonical so freezing is meaningful -------------------------


def canonical_line(row: Row) -> str:
    """Deterministic one-line serialization. Byte-stable across runs and machines."""
    return json.dumps(
        row.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def write_jsonl(rows, path) -> int:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    path.write_text("\n".join(canonical_line(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def read_jsonl(path) -> list[Row]:
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return [Row.model_validate_json(line) for line in text.splitlines() if line.strip()]
