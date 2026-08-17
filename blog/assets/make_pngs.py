#!/usr/bin/env python3
"""Rasterise the post's figures and tables to PNG for platforms that take neither.

X Articles accepts no SVG and has no table support; Medium accepts no inline SVG
and no tables either. Between them that is five figures and five tables, roughly
a third of this post's evidence, which would otherwise simply not travel.

Rasterising goes through `npx sharp-cli`. Two alternatives were tried and
rejected: cairosvg needs a system cairo matching the interpreter's architecture,
which is x86_64 here against an arm64 Python, and macOS Quick Look forces a
square canvas that silently crops wide figures, cutting a whole panel off the
frozen-eval chart.

Tables are drawn as SVG from the post's own markdown rather than screenshotted
from the rendered page, so their contents cannot drift from the article. They
use a monospace stack because that is how the article sets them, and because
monospace makes column widths computable rather than guessed.

    uv run --with markdown-it-py python blog/assets/make_pngs.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS = ROOT / "blog" / "assets"
POST = ROOT / "blog" / "01-an-sft-baseline-an-rl-result-can-stand-on.md"

WIDTH = 1760  # 2x the article's 880px figure width

# The page's palette, so exported images look like they came from the article.
PAPER = "#fcfcfb"
INK = "#1b1e1c"
MUTED = "#5d635e"
RULE = "#dedbd2"

# Menlo ships with macOS and resolves through fontconfig, which a bare
# `ui-monospace` does not. Its advance width is very close to 0.6em, which is
# what makes the column arithmetic below honest rather than approximate.
MONO = "Menlo, 'DejaVu Sans Mono', monospace"
CHAR_W = 0.6
FONT_PX = 15
ROW_H = 34
PAD = 26


def sharp(source: Path, out_dir: Path, width: int) -> Path:
    subprocess.run(
        ["npx", "--yes", "sharp-cli", "-i", str(source), "-o", str(out_dir),
         "--format", "png", "resize", str(width)],
        check=True, capture_output=True, timeout=300,
    )
    return out_dir / f"{source.stem}.png"


def parse_table(block: str) -> tuple[list[str], list[list[str]], list[bool]]:
    """Rows, and which columns the markdown aligns right."""
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
    header, spec, body = cells[0], cells[1], cells[2:]
    right = [s.endswith(":") for s in spec]
    strip_md = lambda v: re.sub(r"\*\*|`", "", v)
    return (
        [strip_md(h) for h in header],
        [[strip_md(c) for c in row] for row in body],
        right,
    )


def table_svg(header: list[str], body: list[list[str]], right: list[bool]) -> str:
    widths = [
        max(len(row[i]) for row in [header, *body]) * FONT_PX * CHAR_W + 34
        for i in range(len(header))
    ]
    w = int(sum(widths)) + PAD * 2
    h = PAD * 2 + ROW_H * (len(body) + 1) + 6

    def cell(text: str, col: int, x: float, y: float, bold: bool) -> str:
        anchor, tx = ("end", x + widths[col] - 17) if right[col] else ("start", x + 17)
        weight = ' font-weight="600"' if bold else ""
        safe = text.replace("&", "&amp;").replace("<", "&lt;")
        return (
            f'<text x="{tx:.0f}" y="{y:.0f}" text-anchor="{anchor}" '
            f'font-family="{MONO}" font-size="{FONT_PX}" fill="{INK}"{weight}>{safe}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>',
    ]

    y = PAD + ROW_H
    x = PAD
    for i, text in enumerate(header):
        parts.append(cell(text, i, x, y - 10, bold=True))
        x += widths[i]
    parts.append(
        f'<line x1="{PAD}" y1="{y}" x2="{w - PAD}" y2="{y}" '
        f'stroke="{INK}" stroke-width="2"/>'
    )

    for row in body:
        y += ROW_H
        x = PAD
        for i, text in enumerate(row):
            parts.append(cell(text, i, x, y - 10, bold=False))
            x += widths[i]
        if row is not body[-1]:
            parts.append(
                f'<line x1="{PAD}" y1="{y}" x2="{w - PAD}" y2="{y}" '
                f'stroke="{RULE}" stroke-width="1"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS / "png")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.tables_only:
        print("figures:")
        for svg in sorted(ASSETS.glob("fig*.svg")):
            out = sharp(svg, args.output, WIDTH)
            print(f"  {out.name:<32} {out.stat().st_size // 1024} KB")

    print("tables:")
    text = POST.read_text(encoding="utf-8")
    blocks = re.findall(r"(?:^\|.*\|\s*$\n?){2,}", text, re.MULTILINE)
    scratch = args.output / "_svg"
    scratch.mkdir(exist_ok=True)
    for index, block in enumerate(blocks, start=1):
        header, body, right = parse_table(block)
        svg_path = scratch / f"table-{index}.svg"
        svg_path.write_text(table_svg(header, body, right), encoding="utf-8")
        out = sharp(svg_path, args.output, WIDTH)
        print(f"  {out.name:<32} {len(body)} rows x {len(header)} cols  "
              f"{out.stat().st_size // 1024} KB  [{header[0]}]")

    print(f"\n-> {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
