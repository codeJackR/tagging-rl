#!/usr/bin/env python3
"""Inspect a loaded schema pack: fields, rule inventory, generated JSON Schema.

    python tools/pack_info.py                        # all packs
    python tools/pack_info.py packs/demo_pack        # one pack
    python tools/pack_info.py packs/demo_pack --json-schema

Loading a pack validates its rules against its vocabulary, so this doubles as a
syntax check: if it prints, the pack is coherent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier import load_pack  # noqa: E402

DEFAULT_PACKS = [ROOT / "packs" / "vastraa_taste_v1", ROOT / "packs" / "demo_pack"]


def show(path: Path, want_schema: bool) -> None:
    pack = load_pack(path)
    inv = pack.rule_inventory()

    print(f"\n=== {pack.name} " + "=" * max(0, 62 - len(pack.name)))
    print(f"  path            {path}")
    print(f"  category field  {pack.category_field}")
    print(f"  abstain token   {pack.unknown_token!r}")
    print(f"  fields          {len(pack.specs)}")
    print(f"  values          {sum(len(s.values) for s in pack.specs.values())}")
    print(f"  aliases         {sum(len(s.aliases) for s in pack.specs.values())}")
    print(f"  rules           {inv['total']}  "
          f"({len(inv['written'])} written + {len(inv['derived'])} derived)")

    print("\n  fields:")
    for name, spec in pack.specs.items():
        kind = "multi" if spec.kind == "multi" else "single"
        scope = "always" if spec.applies_to is None else f"{len(spec.applies_to)} categories"
        req = " required" if spec.required else ""
        print(f"    {name:<18} {kind:<7} {len(spec.values):>3} values   {scope}{req}")

    print("\n  written rules:")
    for rid in inv["written"]:
        print(f"    {rid}")
    if inv["derived"]:
        print("\n  derived rules (from vocab applies_to, never hand-written):")
        for rid in inv["derived"]:
            print(f"    {rid}")

    if want_schema:
        print("\n  JSON Schema (feeds vLLM structured_outputs in W3):")
        print(json.dumps(pack.json_schema(), indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="*", type=Path, default=DEFAULT_PACKS)
    ap.add_argument("--json-schema", action="store_true")
    args = ap.parse_args()
    for p in args.packs:
        show(p, args.json_schema)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
