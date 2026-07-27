#!/usr/bin/env python3
"""Interactive cell tagger. One keypress per decision.

    uv run python scripts/tag.py

Reads `data/review/review_queue.csv`, writes corrections straight back into it
after every keystroke, and skips anything already answered — so quitting and
resuming costs nothing, and a crash loses at most one cell.

Ordered by information, not by file order:

    1. both models committed to DIFFERENT values   <- genuine factual conflict
    2. blind audit cells                           <- measures the skipped majority
    3. one model committed, the other abstained    <- "was it too cautious?"
    4. not_applicable disagreements                <- schema reading, not fact

That ordering means stopping early is a real strategy rather than an
abandonment: the first 76 cells carry most of the signal, and everything after
degrades gracefully. `audit-report` still works on a partial pass — it just has
fewer audit cells and a wider confidence interval, which it will tell you.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import termios
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeling.records import read_jsonl  # noqa: E402
from labeling.review import COLUMNS  # noqa: E402
from verifier import load_pack  # noqa: E402

RESERVED = set("\r\n gunsbq?\x03\x04")
VALUE_KEYS = "123456789acdefhijklmoprtvwxyz"

C = {
    "dim": "\033[2m", "b": "\033[1m", "r": "\033[0m",
    "cy": "\033[36m", "yl": "\033[33m", "gn": "\033[32m",
    "mg": "\033[35m", "rd": "\033[31m",
}


def getkey() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def priority(rec: dict, audit: set) -> int:
    key = (rec.get("sku_id", ""), rec.get("attribute", ""))
    alt, prop = rec.get("alternatives", ""), rec.get("proposed_status", "")
    alt_status = alt[1:-1] if alt.startswith("<") and alt.endswith(">") else ("labeled" if alt else "")
    if key in audit:
        return 1
    if prop == "labeled" and alt_status == "labeled":
        return 0  # both committed, different values
    if "not_applicable" in (prop, alt_status):
        return 3
    return 2  # one abstained


LABELS = {0: ("conflict", "rd"), 1: ("check", "mg"), 2: ("abstain?", "yl"), 3: ("n/a?", "dim")}


def render(rec, spec, idx, total, done, kmap, ctx):
    width = min(shutil.get_terminal_size((90, 30)).columns, 92)
    prio, _ = priority(rec, ctx["audit"]), None
    tag, colour = LABELS[prio]
    print("\033[2J\033[H", end="")

    left = f"{C['b']}[{done}/{total}]{C['r']} {C[colour]}{tag}{C['r']}"
    remain = f"{C['dim']}~{max(0, total - done) * 4 // 60}m left{C['r']}"
    print(f"{left}   {remain}")
    print(C["dim"] + "─" * width + C["r"])

    row = ctx["rows"].get(rec["sku_id"])
    if row:
        meta = " / ".join(x for x in (row.input.brand, row.input.category) if x)
        print(f"{C['b']}{row.input.title[:width - 2]}{C['r']}")
        if meta:
            print(f"{C['dim']}{meta[:width - 2]}{C['r']}")
        desc = (row.input.description or "")[:300]
        for i in range(0, min(len(desc), 300), width - 2):
            print(f"{C['dim']}{desc[i:i + width - 2]}{C['r']}")
        if row.input.raw_tags:
            print(f"{C['dim']}tags: {', '.join(row.input.raw_tags[:10])[:width - 8]}{C['r']}")
    print(C["dim"] + "─" * width + C["r"])

    print(f"  {C['cy']}{C['b']}{rec['attribute']}{C['r']}   {C['dim']}agreement {rec.get('agreement','')}{C['r']}\n")
    prop = rec["proposed_value"] or f"<{rec['proposed_status']}>"
    alt = rec["alternatives"] or "(agrees)"
    print(f"    {C['gn']}A{C['r']}  {prop:<26} {C['dim']}[enter] accept{C['r']}")
    print(f"    {C['yl']}B{C['r']}  {alt:<26} {C['dim']}[g] take this{C['r']}\n")

    cols, line = 4, []
    for key, val in kmap.items():
        line.append(f"{C['cy']}{key}{C['r']} {val[:16]:<17}")
        if len(line) == cols:
            print("  " + "".join(line)); line = []
    if line:
        print("  " + "".join(line))
    print(f"\n  {C['dim']}[u] unknown  [n] n/a  [s] skip  [b] back  [q] save+quit  [?] help{C['r']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    ap.add_argument("--queue", default=str(ROOT / "data" / "review" / "review_queue.csv"))
    ap.add_argument("--sidecar", default=str(ROOT / "data" / "review" / "adjudication.json"))
    ap.add_argument("--candidates", default=str(ROOT / "data" / "eval_candidates.jsonl"))
    args = ap.parse_args()

    pack = load_pack(args.pack)
    queue = Path(args.queue)
    recs = list(csv.DictReader(queue.open(newline="", encoding="utf-8")))
    if not recs:
        sys.exit(f"empty queue: {queue}")

    audit = set()
    side = Path(args.sidecar)
    if side.exists():
        audit = {tuple(x) for x in json.loads(side.read_text()).get("audit_cells", [])}

    rows = {r.sku_id: r for r in read_jsonl(args.candidates)} if Path(args.candidates).exists() else {}
    ctx = {"audit": audit, "rows": rows}

    order = sorted(range(len(recs)), key=lambda i: (priority(recs[i], audit), i))

    def answered(r):
        return bool(r.get("corrected_value", "").strip() or r.get("corrected_status", "").strip())

    def save():
        with queue.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(recs)

    pos = 0
    while pos < len(order):
        i = order[pos]
        rec = recs[i]
        if answered(rec):
            pos += 1
            continue

        spec = pack.specs.get(rec["attribute"])
        values = list(spec.values) if spec else []
        keys = [k for k in VALUE_KEYS if k not in RESERVED][: len(values)]
        kmap = dict(zip(keys, values))
        done = sum(1 for r in recs if answered(r))
        render(rec, spec, i, len(recs), done, kmap, ctx)

        ch = getkey().lower()
        if ch in ("\x03", "\x04", "q"):
            save()
            n = sum(1 for r in recs if answered(r))
            print(f"\n\nsaved {n}/{len(recs)} answered -> {queue}")
            print("resume any time: uv run python scripts/tag.py")
            if n:
                print("\nwhen finished:")
                print(f"  uv run python scripts/adjudicate.py audit-report --corrections {queue}")
                print("  uv run python scripts/build_dataset.py finalize --corrections "
                      f"{queue}")
            return 0
        if ch == "?":
            print(f"\n  {C['dim']}A=primary labeler  B=adjudicator. Judge the text, not the"
                  f" models.{C['r']}\n  press any key")
            getkey(); continue
        if ch == "b":
            pos = max(0, pos - 1)
            recs[order[pos]]["corrected_value"] = ""
            recs[order[pos]]["corrected_status"] = ""
            continue
        if ch == "s":
            pos += 1
            continue

        if ch in ("\r", "\n"):
            rec["corrected_value"] = rec["proposed_value"]
            rec["corrected_status"] = rec["proposed_status"]
        elif ch == "g":
            alt = rec["alternatives"]
            if alt.startswith("<") and alt.endswith(">"):
                rec["corrected_value"], rec["corrected_status"] = "", alt[1:-1]
            elif alt:
                rec["corrected_value"], rec["corrected_status"] = alt, "labeled"
            else:  # audit cell: B agrees with A
                rec["corrected_value"] = rec["proposed_value"]
                rec["corrected_status"] = rec["proposed_status"]
        elif ch == "u":
            rec["corrected_value"], rec["corrected_status"] = "", "unknown"
        elif ch == "n":
            rec["corrected_value"], rec["corrected_status"] = "", "not_applicable"
        elif ch in kmap:
            rec["corrected_value"], rec["corrected_status"] = kmap[ch], "labeled"
        else:
            continue

        save()  # after every keystroke: a crash costs one cell, not a session
        pos += 1

    save()
    print(f"\n\nall {len(recs)} cells answered -> {queue}")
    print("\nnext:")
    print(f"  uv run python scripts/adjudicate.py audit-report --corrections {queue}")
    print(f"  uv run python scripts/build_dataset.py finalize --corrections {queue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
