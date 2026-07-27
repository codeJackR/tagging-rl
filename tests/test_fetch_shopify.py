"""Shopify feed fetcher — parsing, not networking.

Every fixture here is a shape observed on a real store, not an invented one. The
two that matter: `tags` arrives as both a list and a comma-joined string depending
on the store, and real tag lists are full of merchandising codes that would
otherwise ride into every labeling prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import fetch_shopify as fx
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


# --- tags: both wire shapes, and the junk inside them ------------------------


def test_tags_as_list():
    assert fx.normalize_tags(["Summer", "Linen"]) == ["Summer", "Linen"]


def test_tags_as_comma_string():
    """The same field, the other shape. Both are live in the wild."""
    assert fx.normalize_tags("Summer, Linen ,Dress") == ["Summer", "Linen", "Dress"]


def test_internal_merchandising_codes_are_dropped():
    """Verbatim from a real store's first product."""
    raw = [
        "allbirds::cfId => color-8d8c461eb0f65396c57fcd7467def939",
        "DNAM BRANDS",
        "EC STOCK",
        "linen",
        "summer dress",
    ]
    assert fx.normalize_tags(raw) == ["linen", "summer dress"]


def test_tag_dedup_is_case_insensitive():
    assert fx.normalize_tags(["Linen", "linen", "LINEN "]) == ["Linen"]


def test_absurdly_long_tags_are_dropped():
    assert fx.normalize_tags(["x" * 200, "ok"]) == ["ok"]


def test_no_tags_is_empty_not_none():
    assert fx.normalize_tags(None) == []


# --- description -------------------------------------------------------------


def test_html_is_stripped_to_readable_text():
    raw = "<p>Inspired by a classic court style.</p>"
    assert fx.html_to_text(raw) == "Inspired by a classic court style."


def test_block_tags_become_separators_not_run_ons():
    assert fx.html_to_text("<li>Cotton</li><li>Machine wash</li>") == "Cotton Machine wash"


def test_entities_are_unescaped():
    assert fx.html_to_text("<p>Rib &amp; cuff &mdash; 100% cotton</p>") == (
        "Rib & cuff — 100% cotton"
    )


def test_empty_description_is_empty_string():
    assert fx.html_to_text(None) == ""


# --- row shape ---------------------------------------------------------------


def test_row_matches_the_pipeline_input_shape(pack):
    product = {
        "id": 7292464955472,
        "title": "Women's Linen Wrap Dress",
        "body_html": "<p>A relaxed <b>midi</b> dress.</p>",
        "vendor": "Kotn",
        "product_type": "Dresses",
        "tags": ["linen", "EC STOCK"],
        "images": [{"src": "https://cdn.example/1.jpg"}, {"src": "https://cdn.example/2.jpg"}],
    }
    row = fx.to_row(product, "kotn.com")
    assert row["sku_id"] == "shopify:kotn.com:7292464955472"
    assert row["source"] == "shopify:kotn.com"
    assert row["input"]["brand"] == "Kotn"
    assert row["input"]["category"] == "Dresses"
    assert row["input"]["raw_tags"] == ["linen"]
    assert row["input"]["description"] == "A relaxed midi dress."
    assert row["input"]["image_url"] == "https://cdn.example/1.jpg"


def test_row_survives_a_product_missing_everything_optional():
    row = fx.to_row({"id": 1, "title": "Thing"}, "x.com")
    assert row["input"]["brand"] is None
    assert row["input"]["image_url"] is None
    assert row["input"]["raw_tags"] == []


def test_row_is_json_serializable():
    json.dumps(fx.to_row({"id": 1, "title": "T"}, "x.com"))


# --- apparel filter, driven by the pack --------------------------------------


def test_apparel_filter_keeps_clothes(pack):
    matches = fx.build_apparel_matcher(pack)
    assert matches({"product_type": "Dresses", "title": "Linen Wrap Dress"})
    assert matches({"product_type": "", "title": "Merino Wool Sweater", "handle": ""})
    assert matches({"product_type": "", "title": "", "handle": "mens-oxford-shirt"})


def test_apparel_filter_rejects_non_clothes(pack):
    matches = fx.build_apparel_matcher(pack)
    assert not matches({"product_type": "Home", "title": "Soy Candle", "handle": "candle"})
    assert not matches({"product_type": "Gift Card", "title": "Gift Card", "handle": "gc"})


def test_apparel_filter_is_pack_driven_not_hardcoded(pack):
    """Swap the pack, get the right filter for free — same claim as W1 Step 2."""
    demo = load_pack(ROOT / "packs" / "demo_pack")
    matches = fx.build_apparel_matcher(demo)
    assert matches({"product_type": "Lander", "title": "Mars lander", "handle": ""})
    assert not matches({"product_type": "Dresses", "title": "Linen Dress", "handle": ""})


# --- the failure mode that motivated `probe` ---------------------------------


def test_html_served_as_200_is_an_error_not_an_empty_page(monkeypatch):
    """Most real stores do this. Treating it as 'no products' hides the cause."""
    monkeypatch.setattr(fx, "_get", lambda url, t: (200, b"<!DOCTYPE html><html>..."))
    products, err = fx.fetch_page("blocked.example", 1)
    assert products == []
    assert "not JSON" in err


def test_blocked_status_is_reported_and_not_retried(monkeypatch):
    monkeypatch.setattr(fx, "_get", lambda url, t: (429, b""))
    _, err = fx.fetch_page("x.example", 1)
    assert "blocked" in err and "429" in err


def test_valid_json_parses(monkeypatch):
    payload = json.dumps({"products": [{"id": 1, "title": "Dress"}]}).encode()
    monkeypatch.setattr(fx, "_get", lambda url, t: (200, payload))
    products, err = fx.fetch_page("ok.example", 1)
    assert err == ""
    assert products[0]["title"] == "Dress"


def test_json_without_products_key_is_an_error(monkeypatch):
    monkeypatch.setattr(fx, "_get", lambda url, t: (200, b'{"errors":"nope"}'))
    _, err = fx.fetch_page("x.example", 1)
    assert "products" in err


# --- fetch loop --------------------------------------------------------------


def test_fetch_dedups_and_filters(pack, monkeypatch):
    page = {
        "products": [
            {"id": 1, "title": "Linen Dress", "vendor": "A", "product_type": "Dresses"},
            {"id": 2, "title": "Linen Dress", "vendor": "A", "product_type": "Dresses"},
            {"id": 3, "title": "Soy Candle", "vendor": "A", "product_type": "Home"},
            {"id": 4, "title": "", "vendor": "A", "product_type": "Dresses"},
        ]
    }
    calls = {"n": 0}

    def fake(url, t):
        calls["n"] += 1
        return (200, json.dumps(page if calls["n"] == 1 else {"products": []}).encode())

    monkeypatch.setattr(fx, "_get", fake)
    rows, stats = fx.fetch(
        ["x.example"], pack=pack, target=100, max_pages=2, delay=0, apparel_only=True
    )
    assert len(rows) == 1
    assert stats.duplicate == 1
    assert stats.non_apparel == 1
    assert stats.empty_title == 1


def test_fetch_stops_at_target(pack, monkeypatch):
    products = [
        {"id": i, "title": f"Dress {i}", "vendor": "A", "product_type": "Dresses"}
        for i in range(50)
    ]
    monkeypatch.setattr(
        fx, "_get", lambda url, t: (200, json.dumps({"products": products}).encode())
    )
    rows, _ = fx.fetch(
        ["x.example"], pack=pack, target=10, max_pages=5, delay=0, apparel_only=True
    )
    assert len(rows) == 10


def test_store_errors_are_collected_not_raised(pack, monkeypatch):
    monkeypatch.setattr(fx, "_get", lambda url, t: (200, b"<html>"))
    rows, stats = fx.fetch(
        ["a.example", "b.example"], pack=pack, target=10, max_pages=2, delay=0,
        apparel_only=True,
    )
    assert rows == []
    assert len(stats.errors) == 2


def test_machine_tags_with_underscores_are_dropped():
    """Real store output. Chasing prefixes one at a time is unwinnable."""
    raw = ["YCRF_mens-move-shoes-half-sizes", "YGroup_ygroup_mens-cruiser-terralux", "linen"]
    assert fx.normalize_tags(raw) == ["linen"]


def test_fetch_round_robins_across_stores(pack, monkeypatch):
    """Sequential fill let one store supply a third of the corpus — which defeats
    the reason for pulling from many retailers at all."""
    def fake(url, t):
        domain = url.split("//")[1].split("/")[0]
        return (200, json.dumps({"products": [
            {"id": f"{domain}-{i}", "title": f"{domain} Dress {i}",
             "vendor": domain, "product_type": "Dresses"}
            for i in range(50)
        ]}).encode())

    monkeypatch.setattr(fx, "_get", fake)
    rows, stats = fx.fetch(
        ["a.example", "b.example", "c.example"],
        pack=pack, target=30, max_pages=3, delay=0, apparel_only=True,
    )
    assert len(rows) == 30
    per = [n for n in stats.per_store.values() if n]
    assert len(per) == 3, "every store should contribute, not just the first"
    assert max(per) - min(per) <= 1, f"contributions should be even, got {per}"


def test_ubiquitous_tags_are_pruned(pack):
    """A tag on every product of a store says nothing about any of them."""
    rows = [
        {"source": "shopify:x.com", "input": {"raw_tags": ["retail-sync", f"style-{i}"]}}
        for i in range(10)
    ]
    removed = fx.prune_ubiquitous_tags(rows, pack)
    assert removed == 10
    assert all(r["input"]["raw_tags"] == [f"style-{i}"] for i, r in enumerate(rows))


def test_vocabulary_tags_are_never_pruned(pack):
    """A store selling only tops tags everything 'tops' — ubiquitous AND the
    exact evidence the labeler wants."""
    rows = [
        {"source": "shopify:x.com", "input": {"raw_tags": ["tops", "retail-sync"]}}
        for _ in range(10)
    ]
    fx.prune_ubiquitous_tags(rows, pack)
    assert all(r["input"]["raw_tags"] == ["tops"] for r in rows)


def test_pruning_is_per_store_not_global(pack):
    rows = [{"source": "shopify:a.com", "input": {"raw_tags": ["acme-only"]}} for _ in range(10)]
    rows += [{"source": "shopify:b.com", "input": {"raw_tags": ["acme-only", "x"]}}] * 1
    fx.prune_ubiquitous_tags(rows, pack)
    assert rows[0]["input"]["raw_tags"] == []
    assert "acme-only" in rows[-1]["input"]["raw_tags"], "b.com had too few rows to judge"


def test_colon_namespaced_tags_are_dropped():
    assert fx.normalize_tags(["vendor:Faherty", "sizing:tops", "linen"]) == ["linen"]


def test_singularizer_handles_apparel_nouns():
    assert fx._singular("tops") == "top"
    assert fx._singular("dresses") == "dress"
    assert fx._singular("dress") == "dress", "naive rstrip('s') would give 'dre'"
    assert fx._singular("jackets") == "jacket"
