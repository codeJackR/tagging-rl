#!/usr/bin/env python3
"""Verify a schema pack's vocab.yaml against the frozen Fashionpedia ontology.

W1 / Step 1 done-when: "vocab.yaml exists, every value traceable to the ontology".
This turns that from a claim into a check.

Checks
  1. every `fp:` reference points at a real Fashionpedia category/attribute id
  2. every `fp:` reference's recorded name matches the ontology exactly
     (catches transcription slips, which are the realistic failure mode)
  3. every value declares provenance: direct | derived | custom
  4. every `custom` value carries a written justification
  5. no duplicate value names within a field
  6. no alias collisions within a field (an alias resolving to two values would
     make the normaliser non-deterministic)
  7. no alias that shadows a *different* value's canonical name in the same field
  8. `applies_to.categories` only names real garment_category values

Then prints a coverage report: which of the 294 attributes this pack consumes,
and the provenance split.

Usage:
    python tools/check_vocab_provenance.py [pack_dir ...]
Exit code 0 = clean, 1 = failures.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "data" / "raw" / "fashionpedia" / "ontology.json"
DEFAULT_PACKS = [ROOT / "packs" / "vastraa_taste_v1"]

VALID_TYPES = {"direct", "derived", "custom"}


def load_ontology(path: Path) -> tuple[dict, dict, dict]:
    if not path.exists():
        sys.exit(
            f"ontology snapshot missing: {path}\n"
            "regenerate it from instances_attributes_val2020.json (see README)"
        )
    raw = json.loads(path.read_text())
    cats = {c["id"]: c for c in raw["categories"]}
    atts = {a["id"]: a for a in raw["attributes"]}
    return raw, cats, atts


def walk_fp_refs(node, trail=""):
    """Yield (path, ref) for every entry under any `fp:` key, anywhere in the tree."""
    if isinstance(node, dict):
        for key, val in node.items():
            here = f"{trail}.{key}" if trail else key
            if key == "fp" and isinstance(val, list):
                for i, ref in enumerate(val):
                    yield f"{here}[{i}]", ref
            else:
                yield from walk_fp_refs(val, here)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from walk_fp_refs(item, f"{trail}[{i}]")


def check_pack(pack_dir: Path, cats: dict, atts: dict) -> tuple[list[str], dict]:
    errors: list[str] = []
    vocab_path = pack_dir / "vocab.yaml"
    if not vocab_path.exists():
        return [f"{pack_dir.name}: no vocab.yaml"], {}

    vocab = yaml.safe_load(vocab_path.read_text())
    fields = vocab.get("fields") or {}
    if not fields:
        return [f"{pack_dir.name}: vocab.yaml has no `fields`"], {}

    # -- 1 & 2: every fp reference resolves and its name matches -----------------
    used_attrs: set[int] = set()
    used_cats: set[int] = set()
    for path, ref in walk_fp_refs(vocab):
        kind, rid, rname = ref.get("k"), ref.get("id"), ref.get("name")
        table = {"category": cats, "attribute": atts}.get(kind)
        if table is None:
            errors.append(f"{path}: unknown reference kind {kind!r} (expected category|attribute)")
            continue
        if rid not in table:
            errors.append(f"{path}: {kind} id {rid} does not exist in the ontology")
            continue
        actual = table[rid]["name"]
        if rname != actual:
            errors.append(
                f"{path}: {kind} id {rid} name mismatch\n"
                f"        vocab.yaml says : {rname!r}\n"
                f"        ontology says   : {actual!r}"
            )
        (used_cats if kind == "category" else used_attrs).add(rid)

    # -- 3..7: per-field value checks -------------------------------------------
    prov_counts: Counter[str] = Counter()
    garment_values: set[str] = set()

    for fname, fdef in fields.items():
        values = fdef.get("values") or []
        if not values:
            errors.append(f"fields.{fname}: no values")
            continue

        seen_names: set[str] = set()
        alias_owner: dict[str, str] = {}

        for v in values:
            vname = v.get("name")
            where = f"fields.{fname}.{vname}"
            if not vname:
                errors.append(f"fields.{fname}: a value has no `name`")
                continue
            if vname in seen_names:
                errors.append(f"{where}: duplicate value name")
            seen_names.add(vname)

            frm = v.get("from")
            if not isinstance(frm, dict):
                errors.append(f"{where}: missing `from:` provenance block")
                continue
            ptype = frm.get("type")
            if ptype not in VALID_TYPES:
                errors.append(f"{where}: provenance type {ptype!r} not in {sorted(VALID_TYPES)}")
                continue
            prov_counts[ptype] += 1

            if ptype in ("direct", "derived"):
                if not frm.get("fp"):
                    errors.append(f"{where}: type {ptype!r} must carry at least one `fp:` reference")
                if ptype == "direct" and len(frm.get("fp") or []) != 1:
                    errors.append(f"{where}: type 'direct' must reference exactly one ontology id")
                if ptype == "derived" and not frm.get("op"):
                    errors.append(f"{where}: type 'derived' must declare an `op:` (merge|split|rename|...)")
            else:  # custom
                if frm.get("fp"):
                    errors.append(f"{where}: type 'custom' must not carry `fp:` references")
                if not (frm.get("why") or "").strip():
                    errors.append(f"{where}: type 'custom' requires a written `why:` justification")

            for alias in v.get("aliases") or []:
                key = alias.strip().casefold()
                if key in alias_owner and alias_owner[key] != vname:
                    errors.append(
                        f"fields.{fname}: alias {alias!r} claimed by both "
                        f"{alias_owner[key]!r} and {vname!r} — normaliser would be non-deterministic"
                    )
                alias_owner[key] = vname

        for alias, owner in alias_owner.items():
            if alias in seen_names and alias != owner:
                errors.append(
                    f"fields.{fname}: alias {alias!r} (owned by {owner!r}) shadows the "
                    f"canonical value name {alias!r}"
                )

        if fname == "garment_category":
            garment_values = seen_names

    # -- 8: applies_to references real garment categories -----------------------
    for fname, fdef in fields.items():
        applies = fdef.get("applies_to")
        if isinstance(applies, dict):
            for cat in applies.get("categories") or []:
                if cat not in garment_values:
                    errors.append(
                        f"fields.{fname}.applies_to: {cat!r} is not a garment_category value"
                    )

    stats = {
        "pack": vocab.get("pack", pack_dir.name),
        "fields": len(fields),
        "values": sum(len(f.get("values") or []) for f in fields.values()),
        "provenance": dict(prov_counts),
        "used_attrs": used_attrs,
        "used_cats": used_cats,
    }
    return errors, stats


def coverage_report(stats: dict, raw: dict, atts: dict) -> None:
    used = stats["used_attrs"]
    by_super: dict[str, list[int]] = defaultdict(list)
    for a in raw["attributes"]:
        by_super[a["supercategory"]].append(a["id"])

    print(f"\n  fields                 {stats['fields']}")
    print(f"  values                 {stats['values']}")
    prov = stats["provenance"]
    total = sum(prov.values()) or 1
    for t in ("direct", "derived", "custom"):
        n = prov.get(t, 0)
        print(f"    {t:<20} {n:>4}   {100 * n / total:4.1f}%")

    print(f"\n  Fashionpedia attributes consumed: {len(used)} / {len(atts)}")
    print(f"  Fashionpedia categories consumed: {len(stats['used_cats'])} / {len(raw['categories'])}")
    print("\n  by attribute supercategory:")
    for sc in sorted(by_super, key=lambda s: -len(by_super[s])):
        ids = by_super[sc]
        hit = len([i for i in ids if i in used])
        bar = "#" * round(20 * hit / len(ids))
        flag = "" if hit else "   (unused)"
        print(f"    {sc:<48} {hit:>3}/{len(ids):<4} {bar:<20}{flag}")


def main(argv: list[str]) -> int:
    packs = [Path(p).resolve() for p in argv[1:]] or DEFAULT_PACKS
    raw, cats, atts = load_ontology(ONTOLOGY)

    print(f"ontology: {ONTOLOGY.relative_to(ROOT)}  "
          f"({len(cats)} categories, {len(atts)} attributes)")

    failed = False
    for pack in packs:
        print(f"\n=== {pack.name} " + "=" * (62 - len(pack.name)))
        errors, stats = check_pack(pack, cats, atts)
        if errors:
            failed = True
            print(f"\n  FAIL — {len(errors)} problem(s):\n")
            for e in errors:
                print(f"    - {e}")
        else:
            print("\n  PASS — every value traceable, every custom value justified")
        if stats:
            coverage_report(stats, raw, atts)

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
