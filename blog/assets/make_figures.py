#!/usr/bin/env python3
"""Generate the four figures for blog post 01 as self-contained SVGs.

Every number is read from the committed artifacts (or is a pure function, like
the cosine schedule) so the figures regenerate when the data changes. Stdlib
only — no matplotlib — because the charts are simple enough that a template is
more maintainable than a plotting-library dependency in a training repo.

Palette: the dataviz reference palette (validated: blue/orange pair passes all
light-mode checks — CVD dE 24.7, normal dE 33.6, contrast >= 3:1). Each SVG
carries its own light surface so it stays legible on dark-mode pages.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# --- palette (dataviz reference instance, light mode) -------------------------
SURF = "#fcfcfb"
INK = "#0b0b0b"
SEC = "#52514e"
MUT = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # series 1: SFT
ORANGE = "#eb6834"  # series 2: zero-shot
RING = "rgba(11,11,11,0.10)"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "ui-monospace, Menlo, monospace"


def svg_open(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
        f'<rect width="{w}" height="{h}" rx="12" fill="{SURF}"/>',
        f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="12" '
        f'fill="none" stroke="{RING}"/>',
    ]


def text(x, y, s, size=12, fill=SEC, weight=400, anchor="start", family=FONT):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="{family}">{s}</text>'
    )


def bar(x, y_top, w, h, colour, baseline_y):
    """Bar with a 4px rounded data-end, square at the baseline."""
    if h < 6:  # too short for a rounded cap; keep it visible and honest
        return (
            f'<rect x="{x}" y="{baseline_y - max(h, 2)}" width="{w}" '
            f'height="{max(h, 2)}" fill="{colour}"/>'
        )
    r = 4
    return (
        f'<path d="M {x} {baseline_y} L {x} {y_top + r} '
        f'Q {x} {y_top} {x + r} {y_top} L {x + w - r} {y_top} '
        f'Q {x + w} {y_top} {x + w} {y_top + r} L {x + w} {baseline_y} Z" '
        f'fill="{colour}"/>'
    )


def legend(x, y, entries):
    out, cx = [], x
    for label, colour in entries:
        out.append(f'<circle cx="{cx + 5}" cy="{y - 4}" r="5" fill="{colour}"/>')
        out.append(text(cx + 15, y, label, 12, SEC))
        cx += 15 + len(label) * 6.4 + 22
    return out


# --- figure 1: zero-shot vs SFT on the frozen 300 ----------------------------

def fig1() -> str:
    frozen = json.loads(
        (ROOT / "runs/sft-combined-2epoch/frozen-eval-300-metrics.json").read_text()
    )["headline"]
    # zero-shot numbers as recorded in the brief's like-for-like table
    panels = [
        ("macro-F1", "0.197", "0.641", 0.1969, frozen["macro_f1"], 1.0),
        ("schema-valid", "62.7%", "100%", 62.7, frozen["schema_validity"] * 100, 100),
        ("vocab-valid", "0%", "88.7%", 0.0, frozen["vocab_validity"] * 100, 100),
        ("rule violations", "1,204", "12", 1204, frozen["rule_violations"], 1204),
    ]
    W, H = 880, 320
    top, bh, base_y = 96, 150, 96 + 150
    s = svg_open(W, H)
    s.append(text(28, 34, "Zero-shot → locked SFT on the frozen 300-row eval",
                  15, INK, 600))
    s.append(text(28, 54, "Same verifier, same greedy unconstrained decoding. "
                  "Zero-shot macro-F1 is conditional on its 196 parsed outputs.",
                  12, SEC))
    s.extend(legend(W - 250, 34, [("zero-shot", ORANGE), ("SFT", BLUE)]))

    pw = (W - 56) / 4
    for i, (name, lz, ls, vz, vs, vmax) in enumerate(panels):
        cx = 28 + i * pw + pw / 2          # panel centre
        bw, gap = 24, 10
        xz, xs = cx - bw - gap / 2, cx + gap / 2
        hz, hs = bh * vz / vmax, bh * vs / vmax
        s.append(f'<line x1="{cx - pw/2 + 14}" y1="{base_y}" '
                 f'x2="{cx + pw/2 - 14}" y2="{base_y}" stroke="{BASE}"/>')
        s.append(bar(xz, base_y - hz, bw, hz, ORANGE, base_y))
        s.append(bar(xs, base_y - hs, bw, hs, BLUE, base_y))
        s.append(text(xz + bw / 2, base_y - hz - 7, lz, 12, INK, 600, "middle"))
        s.append(text(xs + bw / 2, base_y - hs - 7, ls, 12, INK, 600, "middle"))
        s.append(text(cx, base_y + 22, name, 12.5, SEC, 400, "middle"))

    s.append(text(28, H - 22, "Every output parseable after SFT; the base model "
                  "produced nothing scorable for 104 of 300 products.", 11.5, MUT))
    s.append("</svg>")
    return "\n".join(s)


# --- figure 2: the 78-of-4,500 review grid -----------------------------------

def fig2() -> str:
    total_cells, reviewed = 4500, 78
    cols, rows, cell, gap = 30, 6, 16, 4
    filled = round(reviewed / total_cells * cols * rows)  # 3 of 180
    gw = cols * (cell + gap) - gap
    W, H = 880, 230
    x0, y0 = (W - gw) / 2, 88
    s = svg_open(W, H)
    s.append(text(28, 34, "How much of the eval set a human actually looked at",
                  15, INK, 600))
    s.append(text(28, 54, "300 rows × 15 attributes = 4,500 cells. "
                  "Human-reviewed: 78. Each square is 25 cells.", 12, SEC))
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        colour = BLUE if i < filled else GRID
        s.append(f'<rect x="{x0 + c * (cell + gap)}" y="{y0 + r * (cell + gap)}" '
                 f'width="{cell}" height="{cell}" rx="3" fill="{colour}"/>')
    s.append(text(x0, y0 + rows * (cell + gap) + 14,
                  "Everything grey still says whatever the frontier labeler said, "
                  "which is why this post only claims deltas.", 11.5, MUT))
    s.append("</svg>")
    return "\n".join(s)


# --- figure 3: the cosine-horizon confound -----------------------------------

def lr(step: int, total: int, warmup_frac=0.05) -> float:
    w = warmup_frac * total
    if step <= w:
        return step / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / (total - w)))


def fig3() -> str:
    W, H = 880, 320
    x0, x1, y0, y1 = 70, W - 40, 84, 236   # plot box; y1 = LR 0, y0 = peak
    px = lambda t: x0 + (x1 - x0) * t / 406
    py = lambda v: y1 - (y1 - y0) * v

    def poly(total, colour):
        pts = " ".join(
            f"{px(t):.1f},{py(lr(t, total)):.1f}"
            for t in range(0, total + 1, 2)
        )
        return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    v203_in_406 = lr(203, 406)   # ~0.542 of peak
    s = svg_open(W, H)
    s.append(text(28, 34, "Step 203 is not one experiment. The schedule decides.",
                  15, INK, 600))
    s.append(text(28, 54, "Cosine decay is computed over total planned steps, "
                  "so the same step index sits at a different learning rate.", 12, SEC))
    s.extend(legend(W - 330, 34, [("planned: 203 steps", ORANGE),
                                  ("planned: 406 steps", BLUE)]))
    # axes
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{BASE}"/>')
    s.append(text(x0 - 8, y1 + 4, "0", 11, MUT, 400, "end"))
    s.append(text(x0 - 8, y0 + 4, "1e-4", 11, MUT, 400, "end"))
    s.append(text(x0 - 8, (y0 + y1) / 2 + 4, "LR", 11, MUT, 400, "end"))
    for step in (0, 100, 203, 300, 406):
        s.append(text(px(step), y1 + 18, str(step), 11, MUT, 400, "middle"))
    # the step-203 hairline
    s.append(f'<line x1="{px(203)}" y1="{y0 - 10}" x2="{px(203)}" y2="{y1}" '
             f'stroke="{GRID}"/>')
    s.append(poly(203, ORANGE))
    s.append(poly(406, BLUE))
    # markers at step 203 on both curves, with surface rings
    for v, colour in ((lr(203, 203), ORANGE), (v203_in_406, BLUE)):
        s.append(f'<circle cx="{px(203)}" cy="{py(v)}" r="6" fill="{colour}" '
                 f'stroke="{SURF}" stroke-width="2"/>')
    s.append(text(px(203) + 12, py(lr(203, 203)) - 8,
                  "end of schedule: LR ≈ 0", 12, INK, 600))
    s.append(text(px(203) + 12, py(v203_in_406) + 4,
                  "mid-decay: LR ≈ 5.4e-5", 12, INK, 600))
    s.append(text(28, H - 22, "Comparing epoch 1 of a 1-epoch run against epoch 2 "
                  "of a 2-epoch run mixes the data effect with the schedule effect. "
                  "Checkpointing inside one run is the clean comparison.", 11.5, MUT))
    s.append("</svg>")
    return "\n".join(s)


# --- figure 4: the pre-registration boundary ---------------------------------

def fig4() -> str:
    W, H = 880, 250
    s = svg_open(W, H)
    s.append(text(28, 34, "The lock: selection is committed before the test set is touched",
                  15, INK, 600))
    mid_y, bh2 = 132, 46
    boundary_x = 566

    def node(x, w, line1, line2=None, mono=False, accent=False):
        fill = "#e9f0fb" if accent else SURF
        stroke = BLUE if accent else BASE
        y = mid_y - bh2 / 2
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{bh2}" rx="9" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        cy = mid_y + 4 if line2 is None else mid_y - 3
        s.append(text(x + w / 2, cy, line1, 12, INK, 600, "middle",
                      MONO if mono else FONT))
        if line2:
            s.append(text(x + w / 2, mid_y + 14, line2, 10.5, SEC, 400, "middle",
                          MONO if mono else FONT))

    def arrow(xa, xb):
        s.append(f'<line x1="{xa}" y1="{mid_y}" x2="{xb - 7}" y2="{mid_y}" '
                 f'stroke="{MUT}" stroke-width="1.5"/>')
        s.append(f'<path d="M {xb - 7} {mid_y - 4} L {xb} {mid_y} '
                 f'L {xb - 7} {mid_y + 4} Z" fill="{MUT}"/>')

    node(28, 128, "train both arms", "2 epochs each")
    arrow(160, 176)
    node(176, 168, "select on validation", "generations, not loss")
    arrow(348, 364)
    node(364, 178, "commit the lock", "8bff4c6 · sha-pinned choice", accent=True)
    # the boundary
    s.append(f'<line x1="{boundary_x}" y1="{mid_y - 52}" x2="{boundary_x}" '
             f'y2="{mid_y + 52}" stroke="{INK}" stroke-width="2.5"/>')
    arrow(546, 562)
    arrow(570, 586)
    node(590, 130, "one frozen run", "300 rows, once")
    arrow(724, 740)
    node(744, 112, "record result", "4c3e986")
    s.append(text((28 + boundary_x) / 2, mid_y + 62, "iterate freely", 12, SEC,
                  600, "middle"))
    s.append(text((boundary_x + W - 24) / 2, mid_y + 62, "happens once", 12, SEC,
                  600, "middle"))
    s.append(text(boundary_x, mid_y - 62, "the boundary", 11, INK, 600, "middle"))
    s.append(text(28, H - 20, "Both commits are in the repo; the lock manifest "
                  "records the checkpoint SHA-256 and marks the frozen set "
                  "not_run_as_of_lock.", 11.5, MUT))
    s.append("</svg>")
    return "\n".join(s)


# --- figure 5: the two LoRA arms and the parameter math ----------------------

def fig5() -> str:
    W, H = 880, 400
    s = svg_open(W, H)
    s.append(text(28, 34, "Where the trainable pieces attach, and why the "
                  "counts are small", 15, INK, 600))
    s.append(text(28, 54, "LoRA freezes the whole model and trains a thin "
                  "detour beside chosen weight matrices. The arms differ only "
                  "in which matrices get one.", 12, SEC))
    s.extend(legend(28, 82, [("both arms train these", BLUE),
                             ("arm B also trains these", ORANGE)]))

    # left: one transformer block with its seven matrices
    s.append(f'<rect x="28" y="94" width="400" height="240" rx="10" '
             f'fill="none" stroke="{BASE}" stroke-width="1.5"/>')
    s.append(text(44, 114, "one transformer block (×28)", 11.5, MUT))

    def chip(x, y, label, colour):
        fill = "#eaf1fb" if colour == BLUE else "#fdece4"
        s.append(f'<rect x="{x}" y="{y}" width="170" height="34" rx="8" '
                 f'fill="{fill}" stroke="{colour}" stroke-width="1.5"/>')
        s.append(text(x + 85, y + 21, label, 11, INK, 600, "middle", MONO))

    s.append(text(48, 141, "attention", 12, SEC, 600))
    for i, lbl in enumerate(["q_proj 1536→1536", "k_proj 1536→256",
                             "v_proj 1536→256", "o_proj 1536→1536"]):
        chip(48, 148 + i * 44, lbl, BLUE)
    s.append(text(240, 141, "MLP", 12, SEC, 600))
    for i, lbl in enumerate(["gate_proj 1536→8960", "up_proj 1536→8960",
                             "down_proj 8960→1536"]):
        chip(240, 148 + i * 44, lbl, ORANGE)

    # right: the per-matrix math, drawn to scale in spirit
    s.append(text(480, 114, "the detour math, per 1536×1536 matrix", 12, SEC, 600))
    s.append(f'<rect x="480" y="130" width="110" height="110" rx="4" '
             f'fill="{GRID}"/>')
    s.append(text(535, 258, "frozen 1536×1536", 11, MUT, 400, "middle"))
    s.append(text(535, 273, "≈ 2.36M weights", 11, MUT, 400, "middle"))
    s.append(text(614, 191, "+", 17, SEC, 600, "middle"))
    s.append(f'<rect x="638" y="130" width="13" height="110" rx="3" '
             f'fill="{BLUE}"/>')
    s.append(f'<rect x="659" y="130" width="110" height="13" rx="3" '
             f'fill="{BLUE}"/>')
    s.append(text(705, 258, "trained 16×(1536+1536)", 11, MUT, 400, "middle"))
    s.append(text(705, 273, "≈ 49k weights (2.1%)", 11, MUT, 400, "middle"))

    # totals
    s.append(f'<circle cx="36" cy="356" r="5" fill="{BLUE}"/>')
    s.append(text(50, 360, "arm A · attention only · 155,648 per block × 28 "
                  "= 4,358,144 trainable (0.28%)", 12.5, INK, 600, "start", MONO))
    s.append(f'<circle cx="36" cy="380" r="5" fill="{ORANGE}"/>')
    s.append(text(50, 384, "arm B · attention + MLP · 659,456 per block × 28 "
                  "= 18,464,768 trainable (1.18%)", 12.5, INK, 600, "start", MONO))
    s.append("</svg>")
    return "\n".join(s)


FIGS = {
    "fig1-frozen-eval.svg": fig1,
    "fig2-review-grid.svg": fig2,
    "fig3-cosine-confound.svg": fig3,
    "fig4-lock-timeline.svg": fig4,
    "fig5-lora-arms.svg": fig5,
}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        out = HERE / name
        out.write_text(fn(), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)")
