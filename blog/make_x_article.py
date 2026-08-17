#!/usr/bin/env python3
"""Produce a paste-ready X Article from the post.

X Articles is a rich-text editor with no import and no API, so the only way in
is to paste. Pasting markdown yields literal `##` and backticks, so this emits
**formatted HTML**: open it in a browser, select all, copy, paste into the X
composer, and the headings, bold, lists and quotes arrive already applied.

What the platform cannot take, and what happens to it here:

- **SVG.** Not accepted at all. The five figures become absolute PNG URLs.
- **Tables.** No support. The five tables become PNG images, rendered from the
  post's own markdown so they cannot drift from the article.
- **Code blocks.** Degraded to blockquotes by X on paste, so they are emitted as
  blockquotes already, rather than letting the platform decide.

The long reproduction commands are dropped and replaced with a link. They are
reference material, and a reader on X is not going to run them out of a
blockquote; keeping them would cost a screen and a half of scrolling for nothing.

    uv run --with markdown-it-py python blog/make_x_article.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "blog" / "01-an-sft-baseline-an-rl-result-can-stand-on.md"
CANONICAL = "https://www.vastraa.ai/engineering/sft-baseline.html"
IMG_BASE = "https://www.vastraa.ai/engineering/img"
REPO = "https://github.com/codeJackR/tagging-rl/blob/master"

# Plain enough that X's editor keeps the structure and discards the rest.
STYLE = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     font-size:17px;line-height:1.6;max-width:44rem;margin:0 auto;padding:40px 24px;color:#111}
h1{font-size:2em;line-height:1.15}h2{font-size:1.35em;margin-top:2em}h3{font-size:1.1em}
img{max-width:100%;height:auto;display:block;margin:1.4em 0 .4em}
blockquote{border-left:3px solid #ccc;margin:1.2em 0;padding:.4em 0 .4em 1em;color:#333}
blockquote pre{margin:0;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:.85em}
figcaption{font-size:.85em;color:#666;margin-bottom:1.4em}
.note{background:#fff8e1;border:1px solid #f0e0a0;padding:14px 16px;border-radius:6px;
      font-size:.9em;margin-bottom:2em}
"""


def transform(text: str) -> str:
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.DOTALL)

    # Reproduction commands are reference material, not reading material.
    text = re.sub(
        r"## Reproduce this\n.*?(?=\n## The bill)",
        "## Reproduce this\n\nEvery command, config and artifact is in the repo, "
        f"and every table above recomputes from committed files: [the full "
        f"walkthrough]({CANONICAL}) and [the repository]({REPO.rsplit('/blob', 1)[0]}).\n",
        text, flags=re.DOTALL,
    )

    # Figures: SVG is not accepted, so point at the exported PNGs.
    text = re.sub(
        r"!\[([^\]]*)\]\(assets/(fig[^.]+)\.svg\)",
        rf"<<IMG:{IMG_BASE}/\2.png|\1>>",
        text,
    )

    # Tables become images, numbered in document order to match the exports.
    counter = {"n": 0}

    def swap_table(match: re.Match) -> str:
        counter["n"] += 1
        return f"<<IMG:{IMG_BASE}/table-{counter['n']}.png|Table {counter['n']}>>\n"

    text = re.sub(r"(?:^\|.*\|\s*$\n?){2,}", swap_table, text, flags=re.MULTILINE)

    text = re.sub(r"\]\(\.\./([^)]+)\)", rf"]({REPO}/\1)", text)
    return text


def render(text: str) -> str:
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    html = md.render(text)

    # Code blocks are converted by X into blockquotes anyway; do it here so the
    # result is predictable rather than whatever the paste handler decides.
    html = re.sub(r"<pre><code[^>]*>", "<blockquote><pre>", html)
    html = html.replace("</code></pre>", "</pre></blockquote>")

    def image(match: re.Match) -> str:
        url, alt = match.group(1), match.group(2)
        return (
            f'<img src="{url}" alt="{alt}">'
            f"<figcaption>{alt}</figcaption>"
        )

    html = re.sub(r"<p>&lt;&lt;IMG:([^|]+)\|([^&]*)&gt;&gt;</p>", image, html)
    html = re.sub(r"&lt;&lt;IMG:([^|]+)\|([^&]*)&gt;&gt;", image, html)
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "blog" / "x-article.html")
    args = parser.parse_args()

    html = render(transform(POST.read_text(encoding="utf-8")))
    note = (
        '<div class="note"><strong>Paste instructions.</strong> Select all on this '
        "page, copy, and paste into the X Articles composer. Headings, bold, lists "
        "and quotes carry over. If the images do not come across, upload them from "
        "<code>blog/assets/png/</code> at the points marked by their captions. "
        "X Articles requires Premium+ or a Verified Organization."
        "</div>"
    )
    page = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<title>X Article draft</title>"
        f"<style>{STYLE}</style></head><body>{note}{html}</body></html>"
    )
    args.output.write_text(page, encoding="utf-8")

    words = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", html)))
    print(f"{args.output.relative_to(ROOT)}  {words:,} words  "
          f"{len(re.findall(r'<img ', html))} images  "
          f"{args.output.stat().st_size // 1024} KB")
    print(f"  X Articles limit is about 100,000 characters; this is "
          f"{len(re.sub(r'<[^>]+>', '', html)):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
