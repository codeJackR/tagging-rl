#!/usr/bin/env python3
"""W1 Step 3 orchestration — split, review, correct, freeze.

The human sits in the middle, so this is two commands, not one:

    plan      labeled.jsonl -> eval/probe/train + review queue CSV
              (then you open review_queue.csv and fill in the wrong cells)
    finalize  corrections   -> reliability table + frontier baseline + frozen eval

Between them, `data/review/review_queue.csv` is the only artifact that needs a
person. Everything else is mechanical and reproducible from a seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeling import consensus, freeze, lengths, reliability, review, splits  # noqa: E402
from labeling.records import read_jsonl, write_jsonl  # noqa: E402
from verifier import load_pack, verify_record  # noqa: E402


def _sanity(rows, pack) -> None:
    """Grade the labels with the Step 2 verifier before anyone hand-corrects them.

    If the labeler's own output does not satisfy the pack, correcting 300 rows on
    top of it is wasted work. And if a rule fires on labels a human later confirms,
    the rule is wrong — cheaper to learn that here than at row 300.
    """
    bad_shape = rule_hits = 0
    fired: dict[str, int] = {}
    for r in rows:
        res = verify_record(r.to_verifier_record(pack), pack)
        if not (res.schema_valid and res.vocab_valid):
            bad_shape += 1
        for rid in res.rule_violations:
            fired[rid] = fired.get(rid, 0) + 1
            rule_hits += 1
    print(f"  verifier: {bad_shape}/{len(rows)} rows structurally invalid")
    print(f"  verifier: {rule_hits} rule violations across {len(fired)} distinct rules")
    for rid, n in sorted(fired.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      {rid:<40} x{n}")
    if fired:
        print("  A rule that fires on labels you go on to confirm is a wrong rule.")
        print("  Check the top offenders before starting the review pass.")


def cmd_plan(args) -> int:
    pack = load_pack(args.pack)
    rows = read_jsonl(args.labeled)
    out = Path(args.out_dir)

    print(f"loaded {len(rows)} labeled rows from {args.labeled}\n")

    print("== verifier sanity ==")
    _sanity(rows, pack)

    print("\n== label distribution ==")
    for attr, d in splits.distribution(rows).items():
        flag = "  <-- flatters the model" if d["flatters_model"] else ""
        print(
            f"  {attr:<20} n={d['n']:<6} distinct={d['distinct']:<4} "
            f"top={d['top_value']!r} {d['top_share']:.0%}{flag}"
        )

    print("\n== three-state split ==")
    for attr, d in consensus.unlabeled_share(rows).items():
        note = "  <-- rarely extractable from this feed" if d["labeled_rate"] < 0.3 else ""
        print(
            f"  {attr:<20} labeled {d['labeled']:<5} n/a {d['not_applicable']:<5} "
            f"unknown {d['unknown']:<5} ({d['labeled_rate']:.0%} labeled){note}"
        )

    print("\n== labeler instability (pre-correction preview) ==")
    for attr, d in sorted(
        consensus.attribute_disagreement(rows).items(), key=lambda kv: kv[1]["mean_agreement"]
    )[:8]:
        print(
            f"  {attr:<20} mean agreement {d['mean_agreement']:.3f}  "
            f"unstable on {d['unstable_rate']:.0%} of rows"
        )
    print("  Low agreement = definitely a problem. High agreement = no evidence either")
    print("  way; a confidently biased labeler agrees with itself perfectly.")

    print("\n== splits ==")
    plan = splits.stratify(
        rows,
        eval_size=args.eval_size,
        probe_size=args.probe_size,
        min_per_value=args.min_per_value,
        seed=args.seed,
    )
    print(splits.format_coverage(plan, min_per_value=args.min_per_value))

    print("== token lengths ==")
    tok = lengths.get_tokenizer(args.tokenizer)
    lengths.stamp_rows(rows, tok)
    print(lengths.format_report(lengths.length_report(rows, tok), reasoning_block=args.reasoning))

    write_jsonl(plan.eval_rows, out / "eval_candidates.jsonl")
    write_jsonl(plan.probe_rows, out / "probe_100.jsonl")
    write_jsonl(plan.train_rows, out / "train_weak.jsonl")

    print("== review queue ==")
    summary = review.export_review_csv(
        plan.eval_rows,
        out / "review" / "review_queue.csv",
        threshold=args.threshold,
        always_review=tuple(a for a in args.always_review.split(",") if a),
    )
    review.export_vocab_reference(pack, out / "review" / "vocab_reference.csv")
    print(
        f"  {summary['cells_to_review']} of {summary['cells_total']} cells need a human "
        f"({summary['reduction']:.0%} skipped by consensus)"
    )
    print(f"  -> {summary['path']}")
    print("  -> vocab_reference.csv (bind as dropdown validation on the value column)")

    est = summary["cells_to_review"] * 3 / 3600
    print(f"\n  at ~3s/cell that is ~{est:.1f}h of review. Timebox is 6h — if this is")
    print("  over budget, raise --threshold or cut an attribute; do not extend the clock.")
    return 0


def cmd_finalize(args) -> int:
    pack = load_pack(args.pack)
    out = Path(args.out_dir)
    rows = read_jsonl(out / "eval_candidates.jsonl")

    print(f"applying corrections from {args.corrections}")
    result = review.import_review_csv(rows, args.corrections, pack)
    print(
        f"  {result['rows_touched']} rows touched, {result['cells_changed']} cells "
        f"changed, {result['cells_accepted']} proposals accepted"
    )
    for err in result["errors"][:10]:
        print(f"  ERROR {err}")
    if result["errors"]:
        print(f"  {len(result['errors'])} errors — fix the CSV and re-run, nothing frozen")
        return 1

    print("\n" + reliability.format_report(rows))

    base = reliability.frontier_baseline(rows)
    weights = reliability.reward_weights(rows)
    (out / "reliability.json").write_text(
        json.dumps(
            {
                "frontier_baseline": base,
                "reward_weights": weights,
                "table": {
                    k: vars(v) for k, v in reliability.reliability_table(rows).items()
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"reward weights -> {out / 'reliability.json'}  (W2 reward reads this)")

    meta = freeze.freeze(
        rows, out / "eval_300" / "eval.jsonl", note=f"frontier macro {base['macro_accuracy']}"
    )
    print(f"\nfrozen: {meta['n_rows']} rows  sha256 {meta['sha256'][:16]}...")
    print(f"  {meta['corrected_rows']} corrected, {meta['with_frontier_snapshot']} snapshotted")
    print("\n  git add data/eval_300 && git commit && git tag eval-v1")
    print("  The tag records intent; the checksum catches the edit nobody reviews.")
    return 0


def cmd_verify(args) -> int:
    result = freeze.verify(Path(args.out_dir) / "eval_300" / "eval.jsonl")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--labeled", required=True)
    p.add_argument("--eval-size", type=int, default=300)
    p.add_argument("--probe-size", type=int, default=100)
    p.add_argument("--min-per-value", type=int, default=8)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--always-review", default="")
    p.add_argument("--tokenizer", default="heuristic")
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_plan)

    f = sub.add_parser("finalize")
    f.add_argument("--corrections", required=True)
    f.set_defaults(fn=cmd_finalize)

    v = sub.add_parser("verify")
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
