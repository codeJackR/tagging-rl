#!/usr/bin/env python3
"""Cross-model adjudication — shrink the review queue without faking the check.

    uv run python scripts/adjudicate.py run          --provider anthropic
    uv run python scripts/adjudicate.py audit-report --corrections data/review/corrections.csv

What this is, and what it is not
--------------------------------
The 300 hand-corrections exist to break one circle: eval labels and train labels
both come from `gpt-5.6-luna`, so a systematic error is invisible — the training
data teaches it and the scoreboard confirms it. Re-running the *same* model does
not break that circle, it adds a link to it.

An *independent* model is different. Two families, two training sets, errors that
are far less correlated. Where both agree the cell is very likely right; where they
disagree is where the information is.

Self-consistency (5 samples of one model) found where Luna was **unsure**.
Cross-model finds where it is **wrong** — a confidently biased model agrees with
itself perfectly, which is precisely the blind spot the reliability table exists
to expose.

The blind audit is what keeps it honest
---------------------------------------
Skipping the agreed cells is only defensible if you measure what it costs. So a
random sample of *agreed* cells goes into the same CSV, shuffled in, with nothing
marking them. The reviewer cannot tell them apart — which is the point; an audit
you can see is an audit you unconsciously pass.

`audit-report` then computes how often the human overruled a cell both models
agreed on. That converts "skipping is probably fine" into a number that belongs in
the README beside the macro-F1. Without it the eval set has an error rate that is
not merely unknown but unknowable.

What it does not fix: correlated error. Both models read similar text about
fashion. If the industry labels "cropped" a certain way, both agree and both may
differ from what you mean. Adjudication reduces independent error, not shared bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from labeling.records import LabelStatus, from_verifier_record, read_jsonl  # noqa: E402
from labeling.providers import PROVIDERS, get_provider  # noqa: E402
from labeling.review import COLUMNS  # noqa: E402
from prelabel import MAX_TOKENS, build_system, build_user  # noqa: E402
from verifier import load_pack  # noqa: E402

SIDECAR = "adjudication.json"


def adjudicate_rows(rows, pack, provider, workers: int):
    """One independent opinion per row, in parallel. Failures are recorded, not raised."""
    system = build_system(pack)
    schema = provider.adapt_schema(pack.json_schema())

    def one(row):
        body = provider.build_body(system, build_user(row.input, 0), schema, MAX_TOKENS)
        text, err = provider.complete(body)
        if err or not text:
            return row.sku_id, None, err or "no text"
        try:
            return row.sku_id, from_verifier_record(json.loads(text), pack), None
        except Exception as exc:  # noqa: BLE001 — model output is untrusted
            return row.sku_id, None, f"{type(exc).__name__}: {str(exc)[:80]}"

    out, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (sku, labels, err) in enumerate(pool.map(one, rows), 1):
            if err:
                errors.append(f"{sku}: {err}")
            else:
                out[sku] = labels
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    return out, errors


def cmd_run(args) -> int:
    pack = load_pack(args.pack)
    provider = get_provider(args.provider, args.model)
    rows = read_jsonl(args.eval_candidates)
    rng = random.Random(args.seed)

    print(f"adjudicating {len(rows)} rows with {provider.model}")
    print(f"  primary labeler was {rows[0].provenance.labeler}\n")
    second, errors = adjudicate_rows(rows, pack, provider, args.workers)
    print(f"\n  {len(second)} adjudicated, {len(errors)} failed")

    disagree, agree = [], []
    for row in rows:
        other = second.get(row.sku_id)
        if other is None:
            # No second opinion: fall back to reviewing it rather than assuming.
            disagree += [(row, name, None) for name in row.labels]
            continue
        for name, primary in row.labels.items():
            alt = other.get(name)
            if alt is None or primary.key() != alt.key():
                disagree.append((row, name, alt))
            else:
                agree.append((row, name))

    total = len(disagree) + len(agree)
    audit = rng.sample(agree, min(args.audit, len(agree)))
    queue = [(r, n, a, False) for r, n, a in disagree] + [(r, n, None, True) for r, n in audit]
    rng.shuffle(queue)  # blind: an audit cell must be indistinguishable in the CSV

    out_dir = Path(args.out_dir)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "review" / "review_queue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row, name, alt, _is_audit in queue:
            primary = row.labels[name]
            w.writerow({
                "sku_id": row.sku_id,
                "attribute": name,
                "title": row.input.title,
                "evidence": (row.input.description or "")[:240],
                "proposed_value": _flat(primary),
                "proposed_status": primary.status.value,
                "agreement": f"{(row.provenance.self_consistency.agreement.get(name, 0.0) if row.provenance.self_consistency else 0.0):.2f}",
                "alternatives": _flat(alt) if alt is not None else "",
                "corrected_value": "", "corrected_status": "", "note": "",
            })

    sidecar = out_dir / "review" / SIDECAR
    sidecar.write_text(json.dumps({
        "adjudicator": provider.model,
        "primary": rows[0].provenance.labeler,
        "cells_total": total,
        "cells_disagreed": len(disagree),
        "cells_agreed": len(agree),
        "audit_cells": [[r.sku_id, n] for r, n, _a, is_a in queue if is_a],
        "failed_rows": errors[:50],
    }, indent=2) + "\n")

    print(f"\n  cells: {total} total · {len(disagree)} disagreed · {len(agree)} agreed")
    print(f"  queue: {len(queue)} ({len(disagree)} disagreements + {len(audit)} blind audit)")
    print(f"  reduction vs reviewing every flagged cell: "
          f"{1 - len(queue) / max(1, total):.0%}")
    print(f"\n  -> {csv_path}")
    print(f"  -> {sidecar}")
    print(f"\n  ~{len(queue) * 3 / 60:.0f} minutes at 3s/cell.")
    print("  The audit cells are shuffled in and unmarked — that is deliberate.")
    print("  Review them exactly as you would any other cell.")
    return 0


def _flat(label) -> str:
    if label is None or label.status is not LabelStatus.LABELED:
        return "" if label is None else f"<{label.status.value}>"
    return "|".join(label.value) if isinstance(label.value, list) else str(label.value)


def cmd_audit_report(args) -> int:
    """How often did the human overrule a cell both models agreed on?"""
    sidecar = json.loads((Path(args.out_dir) / "review" / SIDECAR).read_text())
    audit = {(s, a) for s, a in sidecar["audit_cells"]}
    if not audit:
        sys.exit("no audit cells recorded")

    changed = checked = 0
    with Path(args.corrections).open(newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            key = (rec.get("sku_id", ""), rec.get("attribute", ""))
            if key not in audit:
                continue
            checked += 1
            if rec.get("corrected_value", "").strip() or rec.get("corrected_status", "").strip():
                changed += 1

    from labeling.reliability import wilson

    lo, hi = wilson(changed, checked) if checked else (0.0, 1.0)
    agreed = sidecar["cells_agreed"]
    print(f"blind audit — cells where {sidecar['primary']} and {sidecar['adjudicator']} agreed")
    print(f"  sampled   {checked}")
    print(f"  overruled {changed}  ({changed / checked:.1%})" if checked else "  none checked")
    print(f"  95% CI    [{lo:.1%}, {hi:.1%}]")
    print(f"\n  {agreed} cells were skipped on the strength of that agreement.")
    print(f"  Implied errors among them: ~{int(agreed * changed / checked)} "
          f"(range {int(agreed * lo)}-{int(agreed * hi)})" if checked else "")
    print("\n  This is the eval set's noise floor. Quote it beside macro-F1 — a model")
    print("  cannot be shown to beat a baseline by less than the label noise.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    common.add_argument("--out-dir", default=str(ROOT / "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common])
    r.add_argument("--provider", default="gemini", choices=sorted(PROVIDERS))
    r.add_argument("--model", default=None)
    r.add_argument("--eval-candidates", default=str(ROOT / "data" / "eval_candidates.jsonl"))
    r.add_argument("--audit", type=int, default=50, help="blind audit sample size")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("audit-report", parents=[common])
    a.add_argument("--corrections", required=True)
    a.set_defaults(fn=cmd_audit_report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
