#!/usr/bin/env python3
"""Render blog post 1 to a single self-contained HTML file.

Three things have to happen that a plain markdown-to-HTML pass will not do.

**Figures are inlined.** The post references five SVGs by relative path. A
hosted page can serve them alongside, but a single file cannot, and inlining
also removes five requests and any chance of a broken image. The SVGs carry no
external references, which is checked here rather than assumed.

**Repo-relative links are rewritten to absolute GitHub URLs.** The post links
its own evidence with paths like `../runs/sft-selection.json`, which resolve
when GitHub renders the file from `blog/` and resolve nowhere else. Since the
repository is public they can point at it directly, so the evidence stays one
click away from wherever the post is hosted.

**The draft status comment is dropped.** It is a note to the author.

The stylesheet is inline and the page is theme-aware, because a reader's system
preference is not the author's business.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
FONTS = Path(__file__).resolve().parent / "assets" / "fonts"
POST = ROOT / "blog" / "01-an-sft-baseline-an-rl-result-can-stand-on.md"
ASSETS = ROOT / "blog" / "assets"
REPO = "https://github.com/codeJackR/tagging-rl/blob/master"

# Inter and JetBrains Mono, both SIL Open Font License, embedded rather than
# linked because the artifact CSP blocks font CDNs and a blocked link fails
# silently to a system fallback. Latin subsets: four faces, about 90 KB before
# base64.
FACES = (
    ("Inter", 400, "normal", "inter-latin-400-normal.woff2"),
    ("Inter", 600, "normal", "inter-latin-600-normal.woff2"),
    ("Inter", 400, "italic", "inter-latin-400-italic.woff2"),
    ("JetBrains Mono", 400, "normal", "jetbrains-mono-latin-400-normal.woff2"),
)


def font_faces() -> str:
    out = []
    for family, weight, style, filename in FACES:
        data = (FONTS / filename).read_bytes()
        if data[:4] != b"wOF2":
            raise SystemExit(f"{filename} is not a woff2 file")
        b64 = base64.b64encode(data).decode()
        out.append(
            f"@font-face{{font-family:'{family}';font-style:{style};"
            f"font-weight:{weight};font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "\n".join(out)


STYLE = """
:root {
  --paper:  #fbfaf7;
  --raised: #f3f1eb;
  --ink:    #1b1e1c;
  --muted:  #5d635e;
  --rule:   #dedbd2;
  --accent: #24506b;
  --accent-soft: #e6eef3;
  --pos:    #2f6b52;
  --neg:    #9c3f36;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#14171a; --raised:#1c2126; --ink:#e2e5e3; --muted:#98a09c;
    --rule:#2c3339; --accent:#8fc0dd; --accent-soft:#1d2b34;
    --pos:#63b18c; --neg:#dd8078;
  }
}
:root[data-theme="dark"] {
  --paper:#14171a; --raised:#1c2126; --ink:#e2e5e3; --muted:#98a09c;
  --rule:#2c3339; --accent:#8fc0dd; --accent-soft:#1d2b34;
  --pos:#63b18c; --neg:#dd8078;
}
:root[data-theme="light"] {
  --paper:#fbfaf7; --raised:#f3f1eb; --ink:#1b1e1c; --muted:#5d635e;
  --rule:#dedbd2; --accent:#24506b; --accent-soft:#e6eef3;
  --pos:#2f6b52; --neg:#9c3f36;
}

* { box-sizing: border-box; }

.page {
  background: var(--paper);
  color: var(--ink);
  /* Inter, embedded above. A neo-grotesque drawn for screen UI: tall x-height,
     tight even colour, and a large set of alternates. `cv05` gives the l a
     tail and `cv08` gives the 1 a base serif, which disambiguates 1/l/I on a
     page carrying hashes like 8bff4c6 and run ids like s0ar902g. */
  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI',
               Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-feature-settings: 'cv05' 1, 'cv08' 1, 'ss01' 1;
  font-size: 15.5px;
  line-height: 1.6;
  padding: clamp(24px, 5vw, 72px) clamp(18px, 5vw, 32px) 96px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
/* One width for everything. An earlier version let tables, code and figures
   break out wider than the prose, which put every element on a different left
   edge and read as ragged rather than as a deliberate rhythm. A single column
   is worth more here than the extra width was. */
.col { width: 100%; max-width: 48rem; }

.mono, code, kbd, pre, table {
  font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', SFMono-Regular, Menlo,
               Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

h1 {
  font-size: clamp(27px, 5.4vw, 40px);
  line-height: 1.08;
  letter-spacing: -0.032em;
  font-weight: 650;
  margin: 0 0 .6rem;
  text-wrap: balance;
}
h2 {
  font-size: clamp(19px, 3.2vw, 24px);
  line-height: 1.22;
  letter-spacing: -0.019em;
  font-weight: 640;
  margin: 3.2rem 0 .9rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--rule);
  text-wrap: balance;
}
h3 { font-size: 1.04em; font-weight: 640; margin: 2rem 0 .6rem; letter-spacing: -0.008em; }

p, ul, ol { margin: 0 0 1.15rem; }
li { margin-bottom: .38rem; }
strong { font-weight: 700; }

/* The italic standfirst under the title. */
.col > p:first-of-type em {
  display: block;
  font-style: normal;
  font-size: .95em;
  color: var(--muted);
  line-height: 1.5;
  border-left: 2px solid var(--accent);
  padding-left: 1rem;
  margin-bottom: 2rem;
}

a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:hover { background: var(--accent-soft); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

code {
  font-size: .84em;
  background: var(--raised);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: .08em .32em;
}
pre {
  background: var(--raised);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 1rem 1.1rem;
  overflow-x: auto;
  font-size: .8em;
  line-height: 1.5;
  margin: 0 0 1.3rem;
}
pre code { background: none; border: 0; padding: 0; font-size: 1em; }

.tablewrap { overflow-x: auto; margin: 0 0 1.4rem; }
table { border-collapse: collapse; width: 100%; font-size: .8em; }
th, td { padding: .48rem .7rem; border-bottom: 1px solid var(--rule); text-align: left; }
th { font-weight: 700; border-bottom: 2px solid var(--ink); white-space: nowrap; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
tbody tr:last-child td { border-bottom: 0; }

blockquote {
  margin: 0 0 1.3rem;
  padding: .85rem 1.1rem;
  background: var(--raised);
  border-left: 3px solid var(--accent);
  border-radius: 0 3px 3px 0;
}
blockquote p:last-child { margin-bottom: 0; }

/* The figures are generated by blog/assets/make_figures.py with a fixed
   palette: near-black ink on a near-white ground. Recolouring them per theme
   would break the byte-identical regeneration the post claims, so instead they
   keep their own light plate in both themes, the way a printed plate sits on a
   page. The border and radius make that read as deliberate rather than as a
   transparency bug. */
figure {
  margin: 1.8rem 0 2rem;
  background: #fcfcfb;
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 1rem 1.1rem .9rem;
}
figure svg { width: 100%; height: auto; display: block; }
figure figcaption { color: #52514e; }
figcaption {
  font-size: .78em;
  color: var(--muted);
  margin-top: .6rem;
  line-height: 1.45;
  font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', SFMono-Regular, Menlo,
               Consolas, monospace;
}

hr { border: 0; border-top: 1px solid var(--rule); margin: 3rem 0 1.6rem; }

/* Closing teaser: the post ends with an italic forward pointer. */
.col > p:last-of-type em { color: var(--muted); }

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


def inline_figures(html: str) -> str:
    """Replace <img src="assets/x.svg"> with the SVG itself, inside a figure."""

    def swap(match: re.Match) -> str:
        name = match.group("src")
        alt = match.group("alt")
        svg_path = ASSETS / Path(name).name
        svg = svg_path.read_text(encoding="utf-8")
        if re.search(r'(?:xlink:)?href="http|<image|url\(http', svg):
            raise SystemExit(f"{svg_path.name} has an external reference")
        svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg).strip()
        # An inline SVG needs its own accessible name; the alt text is it.
        svg = svg.replace("<svg", f'<svg role="img" aria-label="{alt}"', 1)
        return f"<figure>{svg}<figcaption>{alt}</figcaption></figure>"

    # markdown-it emits src before alt; matching on attribute order is how this
    # silently produced zero figures the first time, so both orders are handled
    # and the count is asserted by the caller.
    patterns = (
        r'<p><img src="(?P<src>assets/[^"]+)" alt="(?P<alt>[^"]*)"\s*/?></p>',
        r'<p><img alt="(?P<alt>[^"]*)" src="(?P<src>assets/[^"]+)"\s*/?></p>',
    )
    for pattern in patterns:
        html = re.sub(pattern, swap, html)
    return html


def render(source: Path, *, standalone: bool = False) -> str:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.DOTALL)
    # ../runs/x -> the public repo, so evidence links work off GitHub too.
    text = re.sub(r"\]\(\.\./([^)]+)\)", rf"]({REPO}/\1)", text)

    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    html = md.render(text)

    expected = len(re.findall(r"^!\[", text, re.MULTILINE))
    html = inline_figures(html)
    inlined = html.count("<figure>")
    if inlined != expected or "<img" in html:
        raise SystemExit(
            f"inlined {inlined} of {expected} figures, "
            f"{html.count('<img')} img tags left; a hosted page would 404"
        )
    html = re.sub(r"(<table>)", r'<div class="tablewrap">\1', html)
    html = re.sub(r"(</table>)", r"\1</div>", html)

    title = re.search(r"^# (.+)$", text, re.MULTILINE).group(1)
    # The standfirst doubles as the meta description; it is the one sentence
    # already written to describe the piece.
    lede = re.search(r"^\*(.+?)\*$", text, re.MULTILINE | re.DOTALL)
    desc = re.sub(r"\s+", " ", lede.group(1)).strip() if lede else ""

    head = (
        f"<title>{title}</title>\n<style>{font_faces()}\n{STYLE}</style>\n"
    )
    body = f'<div class="page"><div class="col">\n{html}\n</div></div>\n'

    if not standalone:
        # Artifact hosting supplies doctype, head and body itself.
        return head + body

    esc = lambda v: v.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{esc(desc)}">\n'
        f'<meta property="og:title" content="{esc(title)}">\n'
        f'<meta property="og:description" content="{esc(desc)}">\n'
        '<meta property="og:type" content="article">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"{head}"
        "</head>\n<body>\n"
        f"{body}"
        "</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=POST)
    parser.add_argument("--output", type=Path, default=ROOT / "blog" / "01-post.html")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="emit a complete HTML document, for serving as a file rather than "
             "through artifact hosting, which supplies its own head and body",
    )
    args = parser.parse_args()
    args.output.write_text(render(args.source, standalone=args.standalone), encoding="utf-8")
    try:
        shown = args.output.relative_to(ROOT)
    except ValueError:
        shown = args.output
    print(f"{shown}  {args.output.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
