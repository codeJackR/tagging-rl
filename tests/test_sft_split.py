from __future__ import annotations

from types import SimpleNamespace

from labeling.records import read_jsonl
from training.split_sft import build_assignments, family_title, group_key


def fake(sku: str, brand: str, title: str):
    return SimpleNamespace(
        sku_id=sku,
        input=SimpleNamespace(brand=brand, title=title),
    )


def test_family_title_removes_only_conservative_variant_suffix():
    assert family_title("Everyday Tote | Red (XL)") == "everyday tote"
    assert family_title("Classic Tee - Navy") == "classic tee"
    assert family_title("Short-Sleeve Shirt") == "short sleeve shirt"


def test_same_title_different_brand_is_not_one_group():
    assert group_key(fake("1", "A", "Classic Tee - Navy")) != group_key(
        fake("2", "B", "Classic Tee - Navy")
    )


def test_variant_family_never_crosses_split():
    variants = [
        fake("red", "A", "Classic Tee - Red"),
        fake("blue", "A", "Classic Tee - Blue"),
        fake("black", "A", "Classic Tee - Black"),
    ]
    singles = [fake(str(i), "B", f"Product {i}") for i in range(20)]
    manifest = build_assignments(variants + singles, validation_size=5, seed=42)
    val = set(manifest["validation"])
    assert len({row.sku_id in val for row in variants}) == 1


def test_real_split_is_complete_exact_and_deterministic():
    rows = read_jsonl("data/train_weak.jsonl")
    first = build_assignments(rows, validation_size=360, seed=42)
    second = build_assignments(rows, validation_size=360, seed=42)
    train = set(first["train"])
    val = set(first["validation"])

    assert first == second
    assert len(train) == 3240
    assert len(val) == 360
    assert not train & val
    assert train | val == {row.sku_id for row in rows}

    assignments = {sku: "validation" for sku in val} | {
        sku: "train" for sku in train
    }
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), set()).add(assignments[row.sku_id])
    assert all(len(splits) == 1 for splits in grouped.values())
