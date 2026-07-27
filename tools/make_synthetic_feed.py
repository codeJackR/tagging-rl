#!/usr/bin/env python3
"""Synthetic product feed — so the Step 3 pipeline is provable before Sovrn data.

Not training data. A fixture. It exists for two reasons:

1. The whole pipeline (consensus -> review -> corrections -> reliability -> splits
   -> freeze) can be run end-to-end and tested today, with no feed and no API key.
2. It can simulate a labeler with a *known* per-attribute bias, which is the only
   way to check that `reliability.py` actually detects bias rather than merely
   producing a plausible-looking table. A reliability table that cannot be shown
   to catch a planted error is not evidence of anything.

Generation respects the pack: values are drawn Zipf-ish (as real catalogs are),
`applies_to` decides not_applicable, and a configurable share of attributes is
simply left out of the text so `unknown` occurs for the right reason.

    python tools/make_synthetic_feed.py --n 1200 --out data/raw/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeling.records import (  # noqa: E402
    AttributeLabel,
    LabelStatus,
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    write_jsonl,
)
from verifier import load_pack  # noqa: E402


def zipf_pick(rng: random.Random, values: tuple[str, ...], s: float = 1.1) -> str:
    weights = [1.0 / ((i + 1) ** s) for i in range(len(values))]
    return rng.choices(list(values), weights=weights, k=1)[0]


def make_truth(pack, rng: random.Random, *, unknown_rate: float = 0.18) -> dict:
    """Ground truth for one product, honouring applicability and missing info."""
    cat_field = pack.category_field
    category = zipf_pick(rng, pack.specs[cat_field].values)
    labels = {cat_field: AttributeLabel(value=category, status=LabelStatus.LABELED)}

    for name, spec in pack.specs.items():
        if name == cat_field:
            continue
        if not spec.accepts(category):
            labels[name] = AttributeLabel(status=LabelStatus.NOT_APPLICABLE)
            continue
        if rng.random() < unknown_rate:
            labels[name] = AttributeLabel(status=LabelStatus.UNKNOWN)
            continue
        if spec.kind == "multi":
            k = rng.choice([1, 1, 1, 2, 2, 3])
            vals = rng.sample(list(spec.values), k=min(k, len(spec.values)))
            labels[name] = AttributeLabel(value=sorted(vals), status=LabelStatus.LABELED)
        else:
            labels[name] = AttributeLabel(
                value=zipf_pick(rng, spec.values), status=LabelStatus.LABELED
            )
    return labels


def render_listing(pack, labels: dict, rng: random.Random) -> RowInput:
    """Write the product copy a shop would write, using the vocab's own aliases."""

    def surface(attr: str, value: str) -> str:
        aliases = [a for a, v in pack.specs[attr].aliases.items() if v == value]
        return rng.choice(aliases) if aliases and rng.random() < 0.65 else value

    def val(attr):
        lab = labels.get(attr)
        if lab is None or lab.status is not LabelStatus.LABELED:
            return None
        v = lab.value[0] if isinstance(lab.value, list) else lab.value
        return surface(attr, v)

    bits = [val("colour_primary"), val("pattern"), val("fit"), val("material")]
    head = " ".join(b for b in bits if b)
    cat = val(pack.category_field) or "item"
    title = f"{head} {cat}".strip().title()

    sentences = []
    for attr, phrase in (
        ("sleeve_length", "{} sleeves"),
        ("neckline", "{} neckline"),
        ("garment_length", "{} length"),
        ("closure", "{} fastening"),
        ("silhouette", "a {} shape"),
        ("waistline", "{} waist"),
        ("collar_type", "{} collar"),
        ("occasion", "made for {}"),
    ):
        v = val(attr)
        if v:
            sentences.append(phrase.format(v))
    rng.shuffle(sentences)
    description = ". ".join(s.capitalize() for s in sentences[:6]) + "."

    details = labels.get("details")
    tags = []
    if details and details.status is LabelStatus.LABELED and isinstance(details.value, list):
        tags = [surface("details", d) for d in details.value]

    return RowInput(
        title=title,
        description=description,
        raw_tags=tags,
        brand=rng.choice(["Aria", "Northbank", "Corda", "Ply & Co", "Semaine"]),
        category=cat,
        image_url=f"https://example.invalid/img/{rng.randrange(10**8):08d}.jpg",
    )


def simulate_labeler(
    pack, truth: dict, rng: random.Random, *, k: int, bias: dict[str, float]
) -> list[dict]:
    """k noisy samples of `truth`. `bias[attr]` is that attribute's error rate.

    Errors are *systematic* where it matters: a biased attribute tends to flip the
    same way (toward the most common value, and toward `labeled` over
    `not_applicable`) — which is how a real frontier labeler fails, and what the
    reliability table has to be able to see.
    """
    samples = []
    for _ in range(k):
        s = {}
        for name, lab in truth.items():
            rate = bias.get(name, 0.02)
            if rng.random() >= rate:
                s[name] = lab.model_copy(deep=True)
                continue
            spec = pack.specs[name]
            # Arity follows the field, not the branch. Emitting a bare string for a
            # multi-valued field produced structurally invalid rows the first time
            # round — the Step 2 verifier caught it, which is what it is for.
            wrong = [spec.values[0]] if spec.kind == "multi" else spec.values[0]
            s[name] = AttributeLabel(value=wrong, status=LabelStatus.LABELED)
        samples.append(s)
    return samples


def build(
    pack,
    n: int,
    *,
    seed: int = 0,
    k: int = 5,
    bias: dict[str, float] | None = None,
    labeler: str = "synthetic-labeler@v1",
) -> tuple[list[Row], dict[str, dict]]:
    """Return (rows, truth_by_sku). Truth is the fixture's secret, never in the Row."""
    from labeling.consensus import consensus_labels

    rng = random.Random(seed)
    bias = bias or {}
    rows: list[Row] = []
    truth_by_sku: dict[str, dict] = {}

    for i in range(n):
        sku = f"SYN{i:06d}"
        truth = make_truth(pack, rng)
        listing = render_listing(pack, truth, rng)
        samples = simulate_labeler(pack, truth, rng, k=k, bias=bias)
        labels, agreement = consensus_labels(samples)

        rows.append(
            Row(
                sku_id=sku,
                source="synthetic",
                split="train",
                input=listing,
                labels=labels,
                provenance=Provenance(
                    labeler=labeler,
                    prompt_version="synthetic-v1",
                    self_consistency=SelfConsistency(k=k, agreement=agreement),
                ),
            )
        )
        truth_by_sku[sku] = truth
    return rows, truth_by_sku


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "synthetic.jsonl"))
    ap.add_argument(
        "--bias",
        default="material=0.35,occasion=0.30",
        help="attr=rate,... — planted systematic errors so reliability.py can be checked",
    )
    args = ap.parse_args()

    pack = load_pack(args.pack)
    bias = {}
    for part in filter(None, args.bias.split(",")):
        attr, _, rate = part.partition("=")
        bias[attr.strip()] = float(rate)

    rows, _ = build(pack, args.n, seed=args.seed, k=args.k, bias=bias)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} synthetic rows -> {args.out}")
    print(f"planted bias: {bias or 'none'}")
    print("\nNOT training data. A fixture, so the pipeline can be proven before the")
    print("real Sovrn feed exists. Delete it before any run whose numbers you quote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
