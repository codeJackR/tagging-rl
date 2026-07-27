#!/usr/bin/env python3
"""Pull apparel listings from public Shopify `/products.json` endpoints.

    python tools/fetch_shopify.py probe --stores-file tools/shopify_candidates.txt
    python tools/fetch_shopify.py fetch --stores-file tools/shopify_stores.txt \\
        --target 4000 --out data/raw/feed.jsonl

Why `probe` exists
------------------
The claim that every Shopify store ships a usable public products.json is
optimistic. Measured on a sample of real apparel stores: most returned **HTTP 200
with an HTML page**, not JSON — the endpoint is disabled, or a bot-protection
layer answers first. A fetcher that assumes JSON produces zero rows and blames the
parser, so `probe` tests a candidate list first and tells you which domains
actually work. Expect to discard a good share of any list you start from.

Two normalizations that are not optional
----------------------------------------
- **`tags` is a list on some stores and a comma-joined string on others.** Both
  shapes appear in the wild; the plan flagged this and it is real.
- **Tags carry internal junk.** A real store returned
  `['allbirds::cfId => color-8d8c…', 'DNAM BRANDS', 'EC STOCK']` — merchandising
  codes, not product attributes. Passed through untouched they become noise in
  every labeling prompt and cost tokens on every one of N x k requests.

Politeness: one request per second by default, a descriptive User-Agent, and a
hard stop on 403/429. This reads public product data that stores publish for
programmatic use; it is not a crawler. Check a store's terms before pulling at
volume, and keep `--delay` sane.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeling.records import RowInput  # noqa: E402
from verifier import load_pack  # noqa: E402

UA = "tagging-rl-dataset-research/0.1 (+public products.json; contact via repo)"
PAGE_LIMIT = 250  # Shopify's documented maximum for this endpoint

# Case-sensitive by design. An `re.I` on the whole pattern makes the ALL-CAPS
# alternative match "Summer" (S + "ummer") and silently eat every ordinary tag —
# which is exactly what it did on the first run. Only the ops-prefix alternative
# is case-insensitive, scoped inline.
_TAG_BLOCK = re.compile(
    r"""^(
        .*::.*                    # namespaced internals: "allbirds::cfId => ..."
      | .*=>.*                    # key/value fragments
      | [A-Z0-9][A-Z0-9 _\-]{2,}  # ALL-CAPS merchandising codes: "EC STOCK"
      | \d+                       # bare numbers
      | .*_.*                     # underscores mean machine, not merchandising
      | .*\S:\S.*                # namespaced: "vendor:Faherty", "sizing:tops"
      | (?i:(spo|yg|bfcm|hide|no-?index|exclude)[-_].*)   # common ops prefixes
    )$""",
    re.X,
)
# The underscore rule replaced a growing prefix blocklist. Real stores emit
# `YCRF_mens-move-shoes-half-sizes` and `YGroup_ygroup_mens-cruiser-terralux`, and
# chasing each new prefix is unwinnable. Human-facing apparel tags use spaces and
# hyphens; underscores are how internal systems namespace things. It costs the
# occasional legitimate `made_in_canada`, which is worth it — junk tags ride into
# every labeling prompt and are paid for on all N x k requests.
_BREAK = re.compile(r"</(p|div|li|br|h[1-6]|tr)\s*>|<br\s*/?>", re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def html_to_text(raw: str | None) -> str:
    """Strip markup without a parser dependency, keeping block boundaries."""
    if not raw:
        return ""
    text = _BREAK.sub("\n", raw)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln).strip()


def normalize_tags(raw: Any) -> list[str]:
    """Handle both wire shapes, then drop merchandising codes."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else str(raw).split(",")
    out: list[str] = []
    for item in items:
        tag = str(item).strip()
        if not tag or len(tag) > 60 or _TAG_BLOCK.match(tag):
            continue
        if tag.lower() not in {t.lower() for t in out}:
            out.append(tag)
    return out


@dataclass
class StoreResult:
    domain: str
    ok: bool
    reason: str = ""
    n_products: int = 0
    tags_shape: str = ""
    sample_type: str = ""


def _get(url: str, timeout: int) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception as exc:  # noqa: BLE001 — network is allowed to fail any way it likes
        return 0, str(exc).encode()


def fetch_page(domain: str, page: int, *, timeout: int = 20) -> tuple[list[dict], str]:
    """Return (products, error). A 200 carrying HTML is an error, not an empty page."""
    url = f"https://{domain}/products.json?limit={PAGE_LIMIT}&page={page}"
    status, body = _get(url, timeout)
    if status in (403, 429):
        return [], f"blocked (HTTP {status}) — backing off, not retrying"
    if status == 0:
        return [], f"network error: {body.decode(errors='replace')[:80]}"
    if status != 200:
        return [], f"HTTP {status}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        head = body[:60].decode(errors="replace").strip().replace("\n", " ")
        return [], f"HTTP 200 but not JSON (endpoint disabled or bot wall): {head!r}"
    products = data.get("products")
    if products is None:
        return [], "JSON without a 'products' key"
    return products, ""


def probe(domains: list[str], *, delay: float) -> list[StoreResult]:
    results: list[StoreResult] = []
    for i, domain in enumerate(domains):
        if i:
            time.sleep(delay)
        products, err = fetch_page(domain, 1)
        if err:
            results.append(StoreResult(domain, False, err))
            continue
        if not products:
            results.append(StoreResult(domain, False, "0 products returned"))
            continue
        first = products[0]
        results.append(
            StoreResult(
                domain,
                True,
                n_products=len(products),
                tags_shape=type(first.get("tags")).__name__,
                sample_type=str(first.get("product_type") or "")[:24],
            )
        )
    return results


# --- apparel filter, driven by the pack rather than a hardcoded list ---------


def build_apparel_matcher(pack):
    """Accept a product if its type or title mentions any garment_category alias.

    Pack-driven on purpose: a retailer pack with a different category vocabulary
    gets the right filter for free, and nothing here needs to know about clothes.
    """
    spec = pack.specs[pack.category_field]
    terms = {a for a in spec.aliases if len(a) > 2} | {v for v in spec.values}
    terms = {t.replace("_", " ") for t in terms}
    pattern = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, terms), key=len, reverse=True)) + r")\b",
        re.I,
    )

    def matches(product: dict) -> bool:
        haystack = " ".join(
            str(product.get(k) or "") for k in ("product_type", "title", "handle")
        )
        return bool(pattern.search(haystack.replace("-", " ")))

    return matches


def to_row(product: dict, domain: str) -> dict:
    images = product.get("images") or []
    return {
        "sku_id": f"shopify:{domain}:{product.get('id')}",
        "source": f"shopify:{domain}",
        "input": RowInput(
            title=str(product.get("title") or "").strip(),
            description=html_to_text(product.get("body_html")),
            raw_tags=normalize_tags(product.get("tags")),
            brand=(str(product.get("vendor")).strip() or None) if product.get("vendor") else None,
            category=(str(product.get("product_type")).strip() or None)
            if product.get("product_type")
            else None,
            image_url=(images[0].get("src") if images else None),
        ).model_dump(mode="json"),
    }


def _singular(word: str) -> str:
    """Crude de-pluralizer, adequate for apparel nouns.

    "tops" -> "top", "dresses" -> "dress", and "dress" is left alone because a
    naive rstrip("s") would turn it into "dre".
    """
    if len(word) > 4 and word.endswith("es") and not word.endswith("ees"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def prune_ubiquitous_tags(rows: list[dict], pack, *, threshold: float = 0.9) -> int:
    """Drop tags carried by ~every product of a store. Returns tags removed.

    Pattern-matching ops tags is unwinnable — after blocking `::`, ALL-CAPS and
    underscores, a real pull still yielded `vendor:Faherty`, `retail-sync`,
    `talkablequalify`, `Inventory GT PP Min`. There is no lexical rule separating
    those from a genuine tag.

    There is an information-theoretic one. A tag on 52 of a store's 52 products
    says nothing about any individual product: zero discriminative content, pure
    token cost on every one of the N x k labeling requests.

    The exception matters as much as the rule: a tag that appears in the pack's own
    vocabulary is never dropped. A store selling only tops tags everything "tops",
    which is ubiquitous *and* exactly the evidence the labeler wants.
    """
    # Stores tag in the plural ("tops", "dresses"); the vocabulary is singular
    # ("top", "dress"). Without matching across that, every plural category tag
    # loses its protection and gets pruned as ops noise — losing exactly the
    # strongest evidence in the feed.
    protected: set[str] = set()
    for spec in pack.specs.values():
        for term in list(spec.values) + list(spec.aliases):
            protected.add(_singular(term.lower()))

    by_store: dict[str, list[dict]] = {}
    for row in rows:
        by_store.setdefault(row["source"], []).append(row)

    removed = 0
    for store_rows in by_store.values():
        n = len(store_rows)
        if n < 5:  # too few to judge ubiquity
            continue
        df: dict[str, int] = {}
        for row in store_rows:
            for tag in set(row["input"]["raw_tags"]):
                df[tag] = df.get(tag, 0) + 1
        drop = {
            tag
            for tag, count in df.items()
            if count / n >= threshold and _singular(tag.lower()) not in protected
        }
        if not drop:
            continue
        for row in store_rows:
            before = len(row["input"]["raw_tags"])
            row["input"]["raw_tags"] = [
                t for t in row["input"]["raw_tags"] if t not in drop
            ]
            removed += before - len(row["input"]["raw_tags"])
    return removed


@dataclass
class FetchStats:
    seen: int = 0
    kept: int = 0
    non_apparel: int = 0
    duplicate: int = 0
    empty_title: int = 0
    tags_pruned: int = 0
    per_store: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def fetch(
    domains: list[str],
    *,
    pack,
    target: int,
    max_pages: int,
    delay: float,
    apparel_only: bool,
) -> tuple[list[dict], FetchStats]:
    matcher = build_apparel_matcher(pack) if apparel_only else (lambda _p: True)
    rows: list[dict] = []
    seen_keys: set[str] = set()
    stats = FetchStats()
    stats.per_store = {d: 0 for d in domains}

    # Round-robin with a per-pass quota, rather than draining store 1 before
    # touching store 2. Filling sequentially let the first store supply a third of
    # a 700-row corpus, and the whole point of pulling from many retailers is
    # distribution breadth — sequential fill quietly destroys the thing you came for.
    #
    # A page holds up to 250 products, so page-level interleaving alone is not
    # enough: one page can exceed a modest target on its own. Each store therefore
    # contributes at most `quota` rows per pass, and whatever is left of its page is
    # buffered rather than discarded — dropping it would silently lose real rows.
    page = {d: 1 for d in domains}
    buffer: dict[str, list[dict]] = {d: [] for d in domains}
    live = list(domains)
    first_request = True

    while live and len(rows) < target:
        quota = max(1, (target - len(rows)) // len(live))
        for domain in list(live):
            if len(rows) >= target:
                break

            if not buffer[domain]:
                if page[domain] > max_pages:
                    live.remove(domain)
                    continue
                if not first_request:
                    time.sleep(delay)
                first_request = False
                products, err = fetch_page(domain, page[domain])
                if err:
                    stats.errors.append(f"{domain} p{page[domain]}: {err}")
                    live.remove(domain)
                    continue
                if not products:
                    live.remove(domain)
                    continue
                buffer[domain] = products
                page[domain] += 1

            taken = 0
            while buffer[domain] and taken < quota and len(rows) < target:
                product = buffer[domain].pop(0)
                stats.seen += 1
                title = str(product.get("title") or "").strip()
                if not title:
                    stats.empty_title += 1
                    continue
                if not matcher(product):
                    stats.non_apparel += 1
                    continue
                key = f"{title.lower()}|{str(product.get('vendor') or '').lower()}"
                if key in seen_keys:
                    stats.duplicate += 1
                    continue
                seen_keys.add(key)
                rows.append(to_row(product, domain))
                stats.per_store[domain] += 1
                stats.kept += 1
                taken += 1

    stats.tags_pruned = prune_ubiquitous_tags(rows, pack)
    return rows, stats


def read_domains(args) -> list[str]:
    raw: list[str] = []
    if args.stores:
        raw += [s for s in args.stores.split(",") if s.strip()]
    if args.stores_file:
        for line in Path(args.stores_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)
    out, seen = [], set()
    for d in raw:
        d = d.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    if not out:
        sys.exit("no stores — pass --stores or --stores-file")
    return out


def cmd_probe(args) -> int:
    results = probe(read_domains(args), delay=args.delay)
    ok = [r for r in results if r.ok]
    print(f"probed {len(results)} domains — {len(ok)} serve JSON\n")
    for r in results:
        if r.ok:
            print(
                f"  OK    {r.domain:<32} {r.n_products:>3} products  "
                f"tags={r.tags_shape:<5} type={r.sample_type!r}"
            )
        else:
            print(f"  FAIL  {r.domain:<32} {r.reason}")
    if ok and args.write_working:
        Path(args.write_working).write_text("\n".join(r.domain for r in ok) + "\n")
        print(f"\nworking domains -> {args.write_working}")
    if not ok:
        print(
            "\nNone usable. HTTP 200 with HTML means the endpoint is disabled or a bot\n"
            "wall answered — not something a retry fixes. Try other stores."
        )
        return 1
    return 0


def cmd_fetch(args) -> int:
    pack = load_pack(args.pack)
    rows, stats = fetch(
        read_domains(args),
        pack=pack,
        target=args.target,
        max_pages=args.max_pages,
        delay=args.delay,
        apparel_only=not args.no_apparel_filter,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    print(f"wrote {len(rows)} rows -> {out}\n")
    print(f"  seen {stats.seen} · kept {stats.kept} · non-apparel {stats.non_apparel} "
          f"· duplicate {stats.duplicate} · untitled {stats.empty_title}")
    print(f"  pruned {stats.tags_pruned} store-ubiquitous tags (zero information, pure token cost)")
    for domain, n in stats.per_store.items():
        print(f"    {domain:<32} {n}")
    if stats.errors:
        print(f"\n  {len(stats.errors)} store/page errors:")
        for e in stats.errors[:10]:
            print(f"    {e}")
    if len(rows) < args.target:
        print(
            f"\n  Short of --target {args.target}. products.json caps out per store, so\n"
            "  breadth comes from more domains, not more pages. Add stores and re-run."
        )
    print("\n  next:  python scripts/prelabel.py estimate --feed " + str(out) + " --k 5")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shared flags live on a parent parser so they work *after* the subcommand,
    # which is the order everyone types and the order the docstring shows.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pack", default=str(ROOT / "packs" / "vastraa_taste_v1"))
    common.add_argument("--stores", help="comma-separated domains")
    common.add_argument("--stores-file", help="one domain per line, # comments allowed")
    common.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", parents=[common],
                       help="which domains actually serve products.json")
    p.add_argument("--write-working", help="save the usable domains to this file")
    p.set_defaults(fn=cmd_probe)

    f = sub.add_parser("fetch", parents=[common], help="pull listings into a feed JSONL")
    f.add_argument("--target", type=int, default=4000)
    f.add_argument("--max-pages", type=int, default=20)
    f.add_argument("--no-apparel-filter", action="store_true")
    f.add_argument("--out", default=str(ROOT / "data" / "raw" / "feed.jsonl"))
    f.set_defaults(fn=cmd_fetch)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
