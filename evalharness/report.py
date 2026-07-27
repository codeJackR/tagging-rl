#!/usr/bin/env python3
"""One command, one report.

    python -m evalharness.report --pack packs/vastraa_taste_v1 \\
        --pred runs/frontier_baseline.jsonl --gold data/eval_300

    python -m evalharness.report --gold data/eval_300 --from-frontier \\
        --markdown-row "frontier (ceiling)"

The frozen eval set is checksum-verified before any number is printed. A metric
computed against a silently edited answer key is worse than no metric — it looks
authoritative and is not comparable to anything measured before the edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evalharness import predictions as preds_mod  # noqa: E402
from evalharness.metrics import LOW_SUPPORT, Report, evaluate  # noqa: E402
from labeling import freeze  # noqa: E402
from labeling.records import read_jsonl  # noqa: E402
from verifier import load_pack  # noqa: E402


def resolve_gold(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        candidate = p / "eval.jsonl"
        if not candidate.exists():
            sys.exit(
                f"no eval.jsonl in {p}\n"
                "  Run W1 Step 3 first: scripts/build_dataset.py finalize"
            )
        return candidate
    if not p.exists():
        sys.exit(f"no gold file at {p}")
    return p


def format_report(rep: Report, *, pack_name: str, gold: Path, source: str) -> str:
    L: list[str] = []
    L.append(f"eval harness v0 — {pack_name}  vs  {gold}")
    L.append(f"  predictions: {source}")

    fz = rep.freeze
    if fz.get("ok"):
        L.append(f"  freeze:      OK  sha256 {fz['sha256'][:16]}...")
    else:
        L.append(f"  freeze:      *** {fz.get('reason', 'unverified')} ***")

    L.append("")
    miss = f" · {len(rep.n_missing)} MISSING" if rep.n_missing else ""
    L.append(f"  rows              {rep.n_gold} gold · {rep.n_predicted} predicted{miss}")
    if rep.schema_validity is None:
        L.append("  schema validity   n/a — predictions arrived pre-parsed")
        L.append("                    (only raw model output can show a format failure)")
    else:
        L.append(
            f"  schema validity   {rep.schema_validity:.1%}"
            f"   ({rep.schema_valid}/{rep.n_attempted} attempts; "
            f"{rep.n_unparseable} unparseable, dropped not zero-filled)"
        )
    L.append(
        f"  vocab validity    {rep.vocab_valid / rep.n_predicted:.1%}"
        if rep.n_predicted
        else "  vocab validity    n/a"
    )
    L.append(
        f"  rule violations   {rep.rule_violations} across "
        f"{len(rep.rule_histogram)} distinct rules"
    )
    for rid, n in rep.rule_histogram.most_common(5):
        L.append(f"                      {rid:<42} x{n}")

    L.append("")
    L.append(
        f"  {'attribute':<20} {'n':>4} {'unk':>4} {'cov':>6} {'exact':>7} "
        f"{'macroF1':>8} {'selF1':>7}  w"
    )
    L.append(f"  {'-' * 20} {'-' * 4} {'-' * 4} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 7}  -")
    for name, a in sorted(rep.attributes.items(), key=lambda kv: kv[1].macro_f1):
        w = rep.reward_weights.get(name, 1.0)
        flag = "" if w > 0 else "  <-- gold not trustworthy"
        L.append(
            f"  {name:<20} {a.n_scorable:>4} {a.n_gold_unknown:>4} "
            f"{a.coverage:>6.2f} {a.exact_match:>7.3f} {a.macro_f1:>8.3f} "
            f"{a.selective_macro_f1:>7.3f}  {w:.1f}{flag}"
        )

    L.append("")
    L.append(f"  HEADLINE macro-F1    {rep.macro_f1:.4f}    <- the number RL has to beat")
    L.append(
        f"  selective macro-F1   {rep.selective_macro_f1:.4f}"
        f"    at coverage {rep.coverage:.4f}"
    )
    L.append(
        "                       (accuracy when it commits, and how often it does —\n"
        "                        quote the pair, never the first without the second)"
    )
    if rep.reward_weights:
        L.append(
            f"  trusted macro-F1     {rep.trusted_macro_f1:.4f}"
            "    excluding attributes the frontier was unreliable on"
        )

    thin = {n: a.low_support_classes for n, a in rep.attributes.items() if a.low_support_classes}
    if thin:
        total = sum(len(v) for v in thin.values())
        L.append(
            f"\n  {total} classes have <{LOW_SUPPORT} gold instances — their F1 terms are"
        )
        L.append("  high-variance; a real gain there is not distinguishable from noise.")
        for name, classes in sorted(thin.items())[:6]:
            L.append(f"    {name}: {', '.join(classes[:8])}")

    halluc = {n: a.hallucinated for n, a in rep.attributes.items() if a.hallucinated}
    if halluc:
        L.append("\n  predicted but never correct (in-vocab, wrong every time):")
        for name, counter in sorted(halluc.items())[:6]:
            top = ", ".join(f"{k} x{v}" for k, v in counter.most_common(3))
            L.append(f"    {name}: {top}")

    if rep.n_missing:
        L.append(
            f"\n  {len(rep.n_missing)} gold rows had no prediction and were skipped, not"
            "\n  scored as wrong. Fix the run rather than reading the number as-is."
        )
    return "\n".join(L) + "\n"


def markdown_row(rep: Report, label: str, cost: str = "—") -> str:
    validity = "—" if rep.schema_validity is None else f"{rep.schema_validity:.1%}"
    return (
        f"| {label} | {rep.macro_f1:.4f} | {validity} | {rep.rule_violations} | {cost} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    ap.add_argument("--gold", default=str(ROOT / "data" / "eval_300"))
    ap.add_argument("--pred", help="prediction JSONL")
    ap.add_argument(
        "--from-frontier",
        action="store_true",
        help="score the frontier snapshots stored in the gold file — the ceiling run",
    )
    ap.add_argument("--reliability", default=str(ROOT / "data" / "reliability.json"))
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--markdown-row", metavar="LABEL")
    ap.add_argument("--cost-per-sku", default="—")
    ap.add_argument(
        "--allow-drift",
        action="store_true",
        help="report anyway when the eval set fails its checksum (it will say so)",
    )
    args = ap.parse_args()

    if not args.pred and not args.from_frontier:
        ap.error("give --pred FILE or --from-frontier")

    pack = load_pack(args.pack)
    gold_path = resolve_gold(args.gold)
    gold = read_jsonl(gold_path)

    fz = freeze.verify(gold_path)
    if not fz.get("ok") and not args.allow_drift:
        print(json.dumps(fz, indent=2), file=sys.stderr)
        print(
            "\nRefusing to report against an unverified eval set. Restore from the\n"
            "tagged commit, or pass --allow-drift if you changed it deliberately.",
            file=sys.stderr,
        )
        return 2

    if args.from_frontier:
        loaded = preds_mod.from_frontier(gold, pack)
        source = "frontier snapshots from the gold file (ceiling run)"
    else:
        loaded = preds_mod.load(args.pred, pack)
        source = f"{args.pred}" + (" (raw output)" if loaded.raw_mode else " (pre-parsed)")

    weights: dict[str, float] = {}
    rel = Path(args.reliability)
    if rel.exists():
        weights = json.loads(rel.read_text()).get("reward_weights", {})

    rep = evaluate(
        gold,
        loaded.records,
        pack,
        reward_weights=weights,
        schema_valid=loaded.schema_valid,
        vocab_valid=loaded.vocab_valid,
        rule_histogram=loaded.rule_histogram,
        unparseable=loaded.unparseable,
        n_attempted=loaded.n_attempted,
    )
    rep.freeze = fz

    if args.as_json:
        print(
            json.dumps(
                {
                    "macro_f1": rep.macro_f1,
                    "selective_macro_f1": rep.selective_macro_f1,
                    "trusted_macro_f1": rep.trusted_macro_f1,
                    "coverage": rep.coverage,
                    "schema_validity": rep.schema_validity,
                    "vocab_validity": rep.vocab_valid / rep.n_predicted
                    if rep.n_predicted
                    else None,
                    "rule_violations": rep.rule_violations,
                    "rule_histogram": dict(rep.rule_histogram),
                    "n_gold": rep.n_gold,
                    "n_missing": len(rep.n_missing),
                    "freeze_ok": bool(fz.get("ok")),
                    "attributes": {
                        n: {
                            "n_scorable": a.n_scorable,
                            "n_gold_unknown": a.n_gold_unknown,
                            "coverage": a.coverage,
                            "exact_match": a.exact_match,
                            "macro_f1": a.macro_f1,
                            "selective_macro_f1": a.selective_macro_f1,
                        }
                        for n, a in rep.attributes.items()
                    },
                },
                indent=2,
            )
        )
    else:
        print(format_report(rep, pack_name=pack.name, gold=gold_path, source=source))

    if args.markdown_row:
        print("| model | macro-F1 | validity | rule viol. | cost/SKU |")
        print("|---|---|---|---|---|")
        print(markdown_row(rep, args.markdown_row, args.cost_per_sku))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
