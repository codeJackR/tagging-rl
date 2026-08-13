from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from training.run2_confirmation_selector import (
    SelectionPolicy,
    provisional_category,
    select_confirmation_candidates,
)
from training.split_sft import group_key


ALIASES = {
    "dress": "dress",
    "gown": "dress",
    "top": "top",
    "tee": "top",
    "shirt": "shirt_blouse",
    "button down": "shirt_blouse",
    "sweater": "sweater",
    "coat": "coat",
}
CATEGORY_ORDER = ("shirt_blouse", "top", "sweater", "coat", "dress")


def _row(
    store: str,
    number: int,
    *,
    brand: str | None = None,
    title: str | None = None,
    category: str = "dress",
) -> dict:
    return {
        "sku_id": f"shopify:{store}:{number}",
        "source": f"shopify:{store}",
        "input": {
            "title": title or f"Dress Product {store} {number}",
            "description": "Pre-label merchant copy only.",
            "raw_tags": ["apparel"],
            "brand": brand or f"Brand {store} {number}",
            "category": category,
            "image_url": None,
        },
    }


def _family(brand: str, title: str) -> str:
    return group_key(
        SimpleNamespace(input=SimpleNamespace(brand=brand, title=title))
    )


def _default_candidates() -> list[dict]:
    return [
        _row(f"store-{store}.example", number)
        for store in range(10)
        for number in range(90)
    ]


def _select(candidates: list[dict], **kwargs) -> dict:
    return select_confirmation_candidates(
        candidates,
        prior_sku_ids=kwargs.pop("prior_sku_ids", set()),
        prior_family_keys=kwargs.pop("prior_family_keys", set()),
        category_aliases=ALIASES,
        category_order=CATEGORY_ORDER,
        **kwargs,
    )


def test_default_contract_selects_400_from_800_plus_clean_rows() -> None:
    result = _select(_default_candidates())

    assert result["counts"]["family_clean_candidates"] == 900
    assert result["counts"]["selected_rows"] == 400
    assert result["counts"]["selected_stores"] == 10
    assert result["invariants"]["maximum_observed_rows_per_store"] <= 60
    assert result["invariants"]["maximum_observed_rows_per_family"] <= 4


def test_selection_is_identical_when_candidate_input_order_changes() -> None:
    candidates = _default_candidates()

    first = _select(candidates)
    second = _select(list(reversed(candidates)))

    assert first["selected"] == second["selected"]
    assert first["selected_store_counts"] == second["selected_store_counts"]


def test_exact_sku_and_normalized_family_overlap_are_excluded() -> None:
    exact = _row("store-0.example", 0)
    family_variant = _row(
        "new.example",
        1,
        brand="Acme",
        title="Classic Tee - Red",
        category="top",
    )
    filler = _default_candidates()
    policy = SelectionPolicy(minimum_clean_candidates=800)

    result = _select(
        [exact, family_variant, *filler[1:]],
        prior_sku_ids={exact["sku_id"]},
        prior_family_keys={_family("Acme", "Classic Tee - Navy")},
        policy=policy,
    )

    reasons = {
        item["sku_id"]: item["reason"]
        for item in result["excluded_before_selection"]
    }
    assert reasons[exact["sku_id"]] == "prior_exact_sku_overlap"
    assert reasons[family_variant["sku_id"]] == "prior_family_overlap"
    assert result["counts"]["family_clean_candidates"] == 899


def test_family_cap_is_active_not_just_reported() -> None:
    unique = [
        _row(f"store-{index % 8}.example", index)
        for index in range(76)
    ]
    hot_family = [
        _row(
            f"store-{index % 8}.example",
            10_000 + index,
            brand="One Brand",
            title=f"One Jacket - Variant {index}",
            category="coat",
        )
        for index in range(12)
    ]
    policy = SelectionPolicy(
        target_rows=80,
        minimum_clean_candidates=80,
        maximum_rows_per_store=20,
        minimum_stores=8,
    )

    result = _select(unique + hot_family, policy=policy)

    assert result["invariants"]["maximum_observed_rows_per_family"] == 4
    assert sum(
        item["reason"] == "family_cap_reached"
        for item in result["unselected_clean_candidates"]
    ) == 8


def test_store_cap_forces_eight_store_coverage() -> None:
    candidates = [
        *[_row("large.example", number) for number in range(100)],
        *[
            _row(f"store-{store}.example", number)
            for store in range(1, 8)
            for number in range(10)
        ],
    ]
    policy = SelectionPolicy(
        target_rows=80,
        minimum_clean_candidates=160,
        maximum_rows_per_store=10,
        minimum_stores=8,
    )

    result = _select(candidates, policy=policy)

    assert result["counts"]["selected_stores"] == 8
    assert set(result["selected_store_counts"].values()) == {10}
    assert any(
        item["reason"] == "store_cap_reached"
        for item in result["unselected_clean_candidates"]
    )


def test_post_treatment_fields_fail_closed_instead_of_influencing_membership() -> None:
    candidates = _default_candidates()
    candidates[0]["labels"] = {"garment_category": "dress"}

    with pytest.raises(ValueError, match="post-treatment field.*labels"):
        _select(candidates)


def test_unused_merchant_metadata_cannot_change_membership() -> None:
    candidates = _default_candidates()
    changed = deepcopy(candidates)
    for row in changed:
        row["input"]["description"] = "Completely changed description"
        row["input"]["raw_tags"] = ["different", "tags"]
        row["input"]["image_url"] = "https://example.test/image.jpg"

    assert _select(candidates)["selected"] == _select(changed)["selected"]


def test_provisional_category_uses_product_type_before_title() -> None:
    assert provisional_category(
        product_category="dress",
        title="Merino Sweater",
        aliases=ALIASES,
        category_order=CATEGORY_ORDER,
    ) == "dress"
    assert provisional_category(
        product_category=None,
        title="Merino Sweater",
        aliases=ALIASES,
        category_order=CATEGORY_ORDER,
    ) == "sweater"


def test_insufficient_clean_buffer_fails_before_selection() -> None:
    candidates = _default_candidates()[:800]
    prior = {row["sku_id"] for row in candidates[:1]}

    with pytest.raises(RuntimeError, match="799 < 800"):
        _select(candidates, prior_sku_ids=prior)


def test_duplicate_candidate_sku_fails_closed() -> None:
    candidates = _default_candidates()
    candidates[-1] = deepcopy(candidates[0])

    with pytest.raises(ValueError, match="duplicate candidate SKU"):
        _select(candidates)
