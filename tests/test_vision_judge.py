"""The judge panel, and the two things that would quietly break it.

A panel is only worth running if no judge grades its own output and if judges
cannot see each other. Both are easy to lose in a refactor and neither fails
loudly: the numbers keep coming, they just stop meaning anything.

`test_the_brief_carries_the_aliases_that_caused_a_false_error` is the specific
regression. A full-length pant labelled `maxi` was recorded as wrong by a human
who knew trousers better than the schema; the vocabulary defines `maxi` as
covering "full length". A judge shown values without aliases reproduces that
error, confidently, at scale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labeling.vision_judge import (
    VERDICTS,
    PanelResult,
    Verdict,
    build_prompt,
    parse_verdict,
    tally,
    vocabulary_brief,
)
from verifier import load_pack

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


# --- the regression this module exists for -----------------------------------


def test_the_brief_carries_the_aliases_that_caused_a_false_error(pack):
    brief = vocabulary_brief(pack, "garment_length")
    assert "full length" in brief and "ankle length" in brief
    assert "maxi" in brief


def test_the_prompt_tells_the_judge_the_vocabulary_governs(pack):
    prompt = build_prompt(pack, "garment_length", "maxi", "High Stride Pant")
    lowered = prompt.lower()
    assert "vocabulary's definition governs" in lowered
    assert "full length" in lowered, "the alias must reach the judge"
    assert "unsure" in lowered, "a judge must be allowed to decline"


def test_the_brief_states_which_categories_an_attribute_applies_to(pack):
    assert "Applies only to:" in vocabulary_brief(pack, "waistline")


# --- parsing replies ----------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ('{"verdict":"correct"}', "correct"),
    ('```json\n{"verdict":"incorrect","correction":"midi"}\n```', "incorrect"),
    ('Sure! {"verdict":"unsure","note":"blurry"}', "unsure"),
])
def test_verdicts_survive_fences_and_preamble(raw, expected):
    assert parse_verdict(raw)[0] == expected


@pytest.mark.parametrize("raw", ["", "not json at all", '{"verdict":"maybe"}'])
def test_an_unreadable_reply_becomes_unsure_not_correct(raw):
    """Defaulting a broken reply to `correct` would inflate the headline with
    parse failures, which is the worst possible direction to fail in."""
    verdict, _correction, note = parse_verdict(raw)
    assert verdict == "unsure"
    assert note


def test_a_correction_is_captured_when_the_judge_disagrees():
    verdict, correction, _ = parse_verdict('{"verdict":"incorrect","correction":"knee"}')
    assert verdict == "incorrect" and correction == "knee"


# --- the panel arithmetic -----------------------------------------------------


def v(judge, verdict, attribute="fit", sku="a"):
    return Verdict(sku_id=sku, attribute=attribute, proposed="loose",
                   judge=judge, verdict=verdict)


JUDGES = ("luna", "sol")


def test_unanimous_agreement_counts_as_decided():
    r = tally([v("luna", "correct"), v("sol", "correct")], JUDGES)
    assert (r.unanimous_correct, r.split, r.any_unsure) == (1, 0, 0)
    assert r.summary()["accuracy_on_decided"] == 1.0


def test_a_split_panel_is_not_counted_as_either():
    """A cell the judges disagree on is unmeasured. Counting it as correct
    flatters the number; counting it as wrong invents a defect."""
    r = tally([v("luna", "correct"), v("sol", "incorrect")], JUDGES)
    s = r.summary()
    assert r.split == 1
    assert s["accuracy_on_decided"] == 0.0 and s["decided_share"] == 0.0


def test_one_unsure_removes_the_cell_from_the_denominator():
    r = tally([v("luna", "correct"), v("sol", "unsure")], JUDGES)
    assert r.any_unsure == 1
    assert r.summary()["decided_share"] == 0.0


def test_a_cell_only_one_judge_saw_is_not_a_panel_verdict():
    """Otherwise a judge that errored on half the batch silently becomes the
    sole authority on those cells."""
    r = tally([v("luna", "correct")], JUDGES)
    assert r.cells == 0


def test_accuracy_is_reported_per_attribute_as_well_as_overall():
    verdicts = [
        v("luna", "correct", "fit", "a"), v("sol", "correct", "fit", "a"),
        v("luna", "incorrect", "pattern", "b"), v("sol", "incorrect", "pattern", "b"),
    ]
    s = tally(verdicts, JUDGES).summary()
    assert s["per_attribute"]["fit"]["accuracy_on_decided"] == 1.0
    assert s["per_attribute"]["pattern"]["accuracy_on_decided"] == 0.0
    assert s["accuracy_on_decided"] == 0.5


def test_the_summary_states_the_panel_did_not_grade_its_own_output():
    """The whole reason this module exists. If the claim disappears, a reader
    cannot tell this panel from the circular arrangement it replaced."""
    s = tally([], JUDGES).summary()
    assert s["answerer_excluded_from_panel"] is True
    assert "gemini" in s["caveat"].lower()
    assert "same lab" in s["caveat"].lower() or "share a lab" in s["caveat"].lower()


def test_empty_panel_does_not_divide_by_zero():
    s = tally([], JUDGES).summary()
    assert s["accuracy_on_decided"] == 0.0 and s["decided_share"] == 0.0
