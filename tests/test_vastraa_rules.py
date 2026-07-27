"""Concrete rule behaviour for vastraa_taste_v1.

test_pack_agnostic.py proves the machinery is domain-blind. This file proves the
rules encode what they claim to. Each case is one rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verifier import load_pack, verify_record

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def violations(pack, **record):
    return verify_record(record, pack).rule_violations


# --- a record that should be entirely clean ----------------------------------


def test_coherent_record_has_no_violations(pack):
    result = verify_record(
        {
            "garment_category": "dress",
            "silhouette": "a_line",
            "fit": "regular",
            "garment_length": "midi",
            "sleeve_length": "three_quarter",
            "sleeve_style": "set_in",
            "neckline": "v_neck",
            "collar_type": "none",
            "waistline": "normal",
            "closure": "zip",
            "pattern": "floral",
            "details": ["lined"],
            "material": "silk",
            "colour_primary": "navy",
            "occasion": "work",
        },
        pack,
    )
    assert result.schema_valid and result.vocab_valid, result.errors
    assert result.rule_violations == []
    assert result.ok


# --- requires ----------------------------------------------------------------


def test_dress_without_neckline_violates(pack):
    assert "dress_needs_length_and_neckline" in violations(
        pack, garment_category="dress", garment_length="midi"
    )


def test_dress_may_abstain_on_a_neckline_the_text_never_mentions(pack):
    """Deliberately relaxed after the first real labeling run.

    This rule had allow_abstain: false. It then fired 249 times on real listings,
    232 of them because the copy simply never mentions a neckline — punishing the
    model for being honest about what the text does not say, which is the opposite
    of what the three-state design exists for.
    """
    assert "dress_needs_length_and_neckline" not in violations(
        pack, garment_category="dress", garment_length="midi", neckline="unknown"
    )


def test_allow_abstain_false_still_works_where_it_is_right(pack):
    """The mechanism is sound; only its use on the dress rule was wrong.

    A peplum IS a waistline feature, so abstaining on waistline while claiming a
    peplum silhouette is incoherent regardless of what the listing says.
    """
    assert "peplum_needs_waistline" in violations(
        pack, garment_category="dress", silhouette="peplum", waistline="unknown"
    )


def test_outerwear_may_abstain_on_closure(pack):
    """This rule leaves allow_abstain at its default, so `unknown` satisfies it."""
    assert "outerwear_needs_closure" not in violations(
        pack, garment_category="coat", closure="unknown"
    )


# --- excludes ----------------------------------------------------------------


def test_sleeveless_with_sleeve_style_violates(pack):
    assert "sleeveless_has_no_sleeve_style" in violations(
        pack, garment_category="dress", sleeve_length="sleeveless", sleeve_style="puff"
    )


def test_solid_cannot_be_multicolour(pack):
    assert "solid_is_not_multicolour" in violations(
        pack, garment_category="top", pattern="solid", colour_primary="multicolour"
    )


def test_lapels_rejected_on_knitwear(pack):
    assert "lapels_are_tailored_only" in violations(
        pack, garment_category="sweater", collar_type="notched_lapel"
    )


# --- conditional_vocab -------------------------------------------------------


def test_pants_cannot_be_mini_length(pack):
    assert "pants_length_subset" in violations(
        pack, garment_category="pants", garment_length="mini"
    )


def test_pants_can_be_cropped(pack):
    assert "pants_length_subset" not in violations(
        pack, garment_category="pants", garment_length="cropped"
    )


def test_bodycon_must_be_tight(pack):
    assert "bodycon_is_tight" in violations(
        pack, garment_category="dress", silhouette="bodycon", fit="loose"
    )


def test_quilting_restricted_to_outerwear(pack):
    """`details` is multi-valued; the condition matches if any member matches."""
    assert "quilting_is_outerwear" in violations(
        pack, garment_category="dress", details=["quilted", "lined"]
    )


def test_quilted_jacket_is_fine(pack):
    assert "quilting_is_outerwear" not in violations(
        pack, garment_category="jacket", details=["quilted"]
    )


# --- abstention never counts as a wrong answer -------------------------------


def test_abstention_does_not_trigger_restrictions(pack):
    assert "pants_length_subset" not in violations(
        pack, garment_category="pants", garment_length="unknown"
    )


# --- derived applicability ---------------------------------------------------


def test_shoe_cannot_have_a_sleeve_length(pack):
    assert "auto:applies_to:sleeve_length" in violations(
        pack, garment_category="shoe", sleeve_length="long"
    )


# --- anti-hack ---------------------------------------------------------------


def test_other_category_forfeits_structural_fields(pack):
    """Routing to `other` must not also let the model keep the easy fields."""
    v = violations(pack, garment_category="other", neckline="crew", silhouette="a_line")
    assert "other_category_claims_nothing" in v


def test_other_category_alone_is_clean(pack):
    assert violations(pack, garment_category="other", colour_primary="black") == []


# --- rule inventory ----------------------------------------------------------


def test_rule_count_is_in_the_planned_band(pack):
    inv = pack.rule_inventory()
    assert len(inv["derived"]) == 9, inv["derived"]
    assert 20 <= inv["total"] <= 40, inv["total"]
