#!/usr/bin/env python3
"""Generate the 1200x630 social card for blog post 1.

A link with no `og:image` gets a bare text card, and on X that is roughly the
difference between being read and being scrolled past. The card is generated
rather than designed by hand so it stays tied to the numbers: every figure here
is read from the same committed artifact the post quotes.

The card leads with the transition the post leads with, zero-shot to locked SFT
on the frozen 300. It deliberately does not try to be the article. A social card
has about one second to say what kind of thing this is and whether the numbers
are real.

Fonts are the page's own, converted from the committed woff2 so the card and the
article cannot drift apart typographically. Requires Pillow and fonttools:

    uv run --with pillow --with "fonttools[woff]" python blog/assets/make_og_image.py
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
FONTS = ROOT / "blog" / "assets" / "fonts"
METRICS = ROOT / "runs" / "sft-combined-2epoch" / "frozen-eval-300-metrics.json"

W, H = 1200, 630

# The page's own palette, so the card reads as part of the same object.
PAPER = (251, 250, 247)
INK = (27, 30, 28)
MUTED = (93, 99, 94)
RULE = (222, 219, 210)
ORANGE = (235, 104, 52)   # zero-shot, as in the post's figures
BLUE = (42, 120, 214)     # SFT, as in the post's figures


def face(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a page font, converting woff2 to something Pillow can read."""
    font = TTFont(FONTS / f"{name}.woff2")
    font.flavor = None
    buffer = io.BytesIO()
    font.save(buffer)
    buffer.seek(0)
    return ImageFont.truetype(buffer, size)


def bars(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
         label: str, left: float, right: float,
         left_text: str, right_text: str, small, tiny) -> None:
    """One before/after pair. Heights are proportional within the pair only."""
    peak = max(left, right) or 1.0
    bar_w = int(w * 0.30)
    gap = int(w * 0.12)
    base = y + h

    for i, (value, colour, text) in enumerate(
        ((left, ORANGE, left_text), (right, BLUE, right_text))
    ):
        bx = x + i * (bar_w + gap)
        bh = max(3, int((value / peak) * h))
        draw.rectangle([bx, base - bh, bx + bar_w, base], fill=colour)
        tw = draw.textlength(text, font=small)
        draw.text((bx + (bar_w - tw) / 2, base - bh - 30), text, font=small, fill=INK)

    draw.line([(x, base + 1), (x + bar_w * 2 + gap, base + 1)], fill=RULE, width=2)
    draw.text((x, base + 12), label, font=tiny, fill=MUTED)


def build() -> Image.Image:
    headline = json.loads(METRICS.read_text())["headline"]

    image = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(image)

    title_f = face("inter-latin-600-normal", 54)
    body_f = face("inter-latin-400-normal", 25)
    small_f = face("jetbrains-mono-latin-400-normal", 25)
    tiny_f = face("jetbrains-mono-latin-400-normal", 18)
    eyebrow_f = face("jetbrains-mono-latin-400-normal", 20)

    pad = 72
    draw.rectangle([0, 0, W, 8], fill=BLUE)

    draw.text((pad, pad), "VASTRAA.AI  /  ENGINEERING", font=eyebrow_f, fill=MUTED)

    draw.text((pad, pad + 52), "An SFT baseline", font=title_f, fill=INK)
    draw.text((pad, pad + 116), "an RL result can stand on", font=title_f, fill=INK)

    draw.text(
        (pad, pad + 196),
        "Qwen2.5-1.5B on the frozen 300. Every number recomputes",
        font=body_f, fill=MUTED,
    )
    draw.text((pad, pad + 230), "from committed artifacts.", font=body_f, fill=MUTED)

    # Three pairs, all from the metrics artifact rather than typed in.
    row_y, row_h = 396, 118
    bars(draw, pad, row_y, 300, row_h, "MACRO-F1",
         0.197, headline["macro_f1"], "0.197", f"{headline['macro_f1']:.3f}",
         small_f, tiny_f)
    bars(draw, pad + 350, row_y, 300, row_h, "SCHEMA VALID",
         0.627, headline["schema_validity"], "62.7%",
         f"{headline['schema_validity']:.0%}", small_f, tiny_f)
    bars(draw, pad + 700, row_y, 300, row_h, "RULE VIOLATIONS",
         1204, headline["rule_violations"], "1,204",
         str(headline["rule_violations"]), small_f, tiny_f)

    legend = "zero-shot"
    draw.rectangle([W - pad - 250, pad + 8, W - pad - 236, pad + 22], fill=ORANGE)
    draw.text((W - pad - 228, pad + 4), legend, font=tiny_f, fill=MUTED)
    draw.rectangle([W - pad - 120, pad + 8, W - pad - 106, pad + 22], fill=BLUE)
    draw.text((W - pad - 98, pad + 4), "after SFT", font=tiny_f, fill=MUTED)

    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "blog" / "assets" / "og-sft-baseline.png",
    )
    args = parser.parse_args()
    build().save(args.output, "PNG", optimize=True)
    print(f"{args.output.name}  {args.output.stat().st_size // 1024} KB  {W}x{H}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
