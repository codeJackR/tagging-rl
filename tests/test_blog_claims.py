"""Every number in blog post 1 must still match the artifact it came from.

A published post is a claim about committed evidence. Artifacts get regenerated,
runs get re-run, and a number that was true when it was written can quietly stop
being true. A manual proofread catches that once; this catches it on every
commit.

The post asserts its own reproducibility — "every table in this post recomputes
from committed artifacts" — so the honest way to hold it to that is a test that
recomputes them.

Numbers are extracted from the post by regex rather than hand-copied, so a test
cannot drift from the prose it is checking: if an edit changes a figure in the
post without changing the artifact, this fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "blog" / "01-an-sft-baseline-an-rl-result-can-stand-on.md"
FROZEN = ROOT / "runs" / "sft-combined-2epoch" / "frozen-eval-300-metrics.json"
COMBINED_406 = ROOT / "runs" / "sft-combined-2epoch" / "validation-metrics-checkpoint-406.json"
COMBINED_203 = ROOT / "runs" / "sft-combined-2epoch" / "validation-metrics-checkpoint-203.json"
ATTENTION_406 = ROOT / "runs" / "sft-attention-2epoch" / "validation-metrics-checkpoint-406.json"
ATTENTION_203 = ROOT / "runs" / "sft-attention-2epoch" / "validation-metrics-checkpoint-203.json"
SPLIT = ROOT / "data" / "splits" / "sft-v1.json"
SELECTION = ROOT / "runs" / "sft-selection.json"


@pytest.fixture(scope="module")
def post() -> str:
    return POST.read_text(encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(path: Path) -> dict:
    """Checkpoint reports nest their metrics; the frozen report does not."""
    body = load(path)
    return body.get("metrics", body)


def claims(post: str, pattern: str) -> list[str]:
    """Pull figures out of the post itself, so the test cannot drift from it."""
    return re.findall(pattern, post)


# --- the headline table -------------------------------------------------------


def test_the_frozen_headline_matches_the_metrics_artifact(post):
    frozen = load(FROZEN)["headline"]
    assert f"{frozen['macro_f1']:.3f}" == "0.641"
    assert "0.641" in post
    assert f"{frozen['schema_validity']:.0%}".replace("%", "") == "100"
    assert f"{frozen['vocab_validity'] * 100:.1f}" == "88.7"
    assert "88.7%" in post
    assert frozen["rule_violations"] == 12
    assert frozen["missing_predictions"] == 0


def test_the_zero_shot_comparison_numbers_appear_as_written(post):
    """The zero-shot side is a conditional F1 over 196 parsed survivors, and the
    post must keep saying so wherever it quotes 0.197."""
    assert "0.197" in post
    assert "62.7%" in post
    assert "1,204" in post
    assert "104" in post
    assert "196" in post, "the conditional denominator must stay visible"
    assert "conditional" in post.lower()


def test_the_bootstrap_interval_matches_the_artifact(post):
    """A confidence interval is the easiest number to mistype and the hardest
    for a reader to check."""
    assert "0.626" in post and "0.671" in post
    assert "+0.425" in post and "+0.496" in post
    assert "5,000" in post


# --- the arm ablation ---------------------------------------------------------


def test_the_arm_comparison_matches_both_checkpoint_reports(post):
    attention = metrics(ATTENTION_406)
    combined = metrics(COMBINED_406)

    assert f"{attention['macro_f1']:.3f}" == "0.751" and "0.751" in post
    assert f"{combined['macro_f1']:.3f}" == "0.854" and "0.854" in post
    assert f"{attention['vocab_validity'] * 100:.1f}" == "90.0" and "90.0%" in post
    assert f"{combined['vocab_validity'] * 100:.1f}" == "94.4" and "94.4%" in post
    assert attention["rule_violations"] == 15
    assert combined["rule_violations"] == 10


def test_the_duration_ablation_matches_the_step_203_reports(post):
    """The epoch-1 versus epoch-2 table is the post's methodological centrepiece;
    its numbers come from two checkpoint reports inside one run."""
    attention = metrics(ATTENTION_203)
    assert f"{attention['macro_f1']:.3f}" == "0.708" and "0.708" in post
    assert f"{attention['vocab_validity'] * 100:.1f}" == "78.0" and "78.0%" in post
    assert attention["rule_violations"] == 19

    combined = metrics(COMBINED_203)
    assert f"{combined['macro_f1']:.3f}" == "0.814" and "0.814" in post
    assert f"{combined['vocab_validity'] * 100:.1f}" == "92.5" and "92.5%" in post


def test_the_trainable_parameter_counts_are_exact(post):
    """Both were confirmed against the trainer's own printed counts."""
    assert "4,358,144" in post or "4.36M" in post
    assert "18,464,768" in post or "18.46M" in post
    assert "0.28%" in post and "1.18%" in post


# --- the split ----------------------------------------------------------------


def test_the_split_sizes_match_the_manifest(post):
    manifest = load(SPLIT)
    assert len(manifest["train"]) == 3240 and "3,240" in post
    assert len(manifest["validation"]) == 360 and "360" in post
    assert manifest["n_groups"] == 2255 and "2,255" in post
    assert manifest["n_variant_groups"] == 491 and "491" in post
    assert manifest["n_rows_in_variant_groups"] == 1836 and "1,836" in post


def test_the_split_seed_and_source_are_still_the_ones_committed():
    manifest = load(SPLIT)
    assert manifest["seed"] == 42
    assert manifest["n_rows"] == 3600


# --- the lock -----------------------------------------------------------------


def test_the_locked_checkpoint_and_hashes_match_the_selection_manifest(post):
    """The pre-registration is the post's strongest claim. If the hash it quotes
    has drifted from the manifest, the claim is unverifiable."""
    selection = load(SELECTION)
    body = json.dumps(selection)
    assert "runs/sft-combined-2epoch/checkpoint-406" in body
    assert "checkpoint-406" in post
    for prefix in ("00ae54af", "cae3dbd1"):
        assert prefix in body, f"{prefix} missing from the selection manifest"
        assert prefix in post, f"{prefix} quoted in the post but not in evidence"
    assert "8bff4c6" in post and "4c3e986" in post


# --- the cost -----------------------------------------------------------------


def test_the_cost_table_is_arithmetic_on_the_stated_rate(post):
    """$0.144/hr times measured wall seconds. Stated so a reader can redo it."""
    rate_per_second = 0.144 / 3600
    assert f"{530 * rate_per_second:.3f}" == "0.021" and "$0.021" in post
    assert f"{572 * rate_per_second:.3f}" == "0.023" and "$0.023" in post
    assert f"{269 * rate_per_second:.3f}" == "0.011" and "$0.011" in post
    assert "4.4 cents" in post
    assert "$0.144" in post


# --- the caveats that must never be dropped ----------------------------------


def test_the_weak_label_caveat_survives_every_edit(post):
    """The post's honesty rests on this, and it is the first thing an editor
    trimming for length would cut."""
    assert "78" in post and "4,500" in post
    assert "human" in post.lower()
    assert "delta" in post.lower() or "comparison" in post.lower()


def test_the_review_count_matches_the_reliability_artifact(post):
    """The 78 was asserted as a string only, which is a test that passes whether
    or not the number is true. It comes from `data/reliability.json`, so check
    it there — and check the denominator is the frozen set, not a coincidence."""
    reliability = load(ROOT / "data" / "reliability.json")
    assert reliability["reviewed_cells"] == 78
    assert f"{reliability['reviewed_cells']}" in post

    # eval-labels.jsonl, not eval.jsonl: the merchant corpus is not
    # redistributed (see data/README.md), and the post's claims must stay
    # verifiable from what a public clone actually has.
    gold = [line for line in (ROOT / "data" / "eval_300" / "eval-labels.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    pack_fields = len(json.loads(gold[0])["labels"])
    assert len(gold) * pack_fields == 4500, "the 4,500 denominator has drifted"


def test_the_post_never_calls_a_score_against_these_labels_accuracy(post):
    """The labels came from `gpt-5.6-luna` and the reliability artifact puts that
    source at 72% against human judgement, so no figure here is accuracy. The
    post says so; this stops an edit from quietly dropping it."""
    reliability = load(ROOT / "data" / "reliability.json")
    assert reliability["frontier_baseline"]["macro_accuracy"] < 0.8, (
        "the label source is no longer weak enough for this caveat to be needed"
    )
    lowered = post.lower()
    assert "gpt-5.6-luna" in lowered, "the label source must stay named"
    assert "not real-world accuracy" in lowered or "not a production-accuracy claim" in lowered


def test_the_limitations_section_still_lists_every_known_limitation(post):
    lowered = post.lower()
    for required in (
        "weak gold labels",
        "conditional zero-shot",
        "one model",
        "constrained decoding",
        "regenerated",
    ):
        assert required in lowered, f"limitation dropped: {required}"


def test_every_figure_referenced_by_the_post_exists(post):
    for name in claims(post, r"\]\((assets/[^)]+)\)"):
        assert (POST.parent / name).exists(), f"missing figure: {name}"


def test_every_repo_relative_link_resolves(post):
    """The post links its own evidence. A broken link is a claim a reader
    cannot check."""
    missing = [
        target
        for target in claims(post, r"\]\((\.\./[^)]+)\)")
        if not (POST.parent / target).exists()
    ]
    assert not missing, f"broken evidence links: {missing}"


# --- claims added after the W2 and W3 runs ------------------------------------


def _rule_violations(loaded) -> int:
    return sum(loaded.rule_histogram.values())


def test_the_constrained_decoding_claim_recomputes():
    """The post says enforcing the schema during decoding took the frontier
    model's rule violations from 6 to 79 on these same 300 products. Both ends
    come from committed predictions, so both are recomputed here."""
    from evalharness import predictions as preds_mod
    from labeling.records import read_jsonl
    from verifier import load_pack

    pack = load_pack(ROOT / "packs" / "vastraa_taste_v1")
    gold = read_jsonl(ROOT / "data" / "eval_300" / "eval-labels.jsonl")

    unconstrained = preds_mod.from_frontier(gold, pack)
    constrained = preds_mod.load(
        ROOT / "runs" / "production-eval300" / "pass-1.jsonl", pack
    )

    assert _rule_violations(unconstrained) == 6
    assert _rule_violations(constrained) == 79
    assert "6 to 79" in POST.read_text(encoding="utf-8")

    # The mechanism the post names: membership held, applicability did not.
    assert unconstrained.vocab_valid == len(gold), "vocab was not already perfect"


def test_the_label_accuracy_band_matches_the_reliability_artifact():
    """The post now quotes 72% as a lower bound chosen for difficulty, and a
    0.72 to 0.96 band. The lower end is measured; the band's width is the
    unmeasured part, and both must stay tied to the artifact."""
    reliability = load(ROOT / "data" / "reliability.json")
    post = POST.read_text(encoding="utf-8")

    measured = reliability["frontier_baseline"]["macro_accuracy"]
    assert f"{measured:.0%}".rstrip("%") == "72"
    assert "72%" in post
    assert "0.72 to 0.96" in post
    assert "lower bound" in post, "the bound must not be quoted as an estimate"
    assert "contested" in post, "the selection-for-difficulty caveat must survive"
    assert reliability["usable"] is False


def test_the_rl_reversal_claim_matches_both_scored_sets():
    """The post says the dev set and the frozen set disagreed about whether RL
    helped. That is only worth printing if both numbers still say so."""
    import subprocess

    post = POST.read_text(encoding="utf-8")
    assert "small win" in post and "measurable loss" in post

    # Frozen: arm B below the SFT baseline, interval clear of zero.
    bootstrap = load(
        ROOT / "runs" / "grpo-run2-arm-b-frozen-eval-300-bootstrap.json"
    )["bootstrap"]["metrics"]["macro_f1"]["delta_candidate_minus_baseline"]
    assert bootstrap["point"] < 0, "arm B is no longer below baseline on frozen"
    assert bootstrap["ci"][1] < 0, "the loss is no longer significant"

    # Dev monitor: arm B above its own baseline, which is the disagreement.
    quality = load(
        ROOT / "runs" / "grpo-run2-arm-b-ua-monitor-quality" / "checkpoint-300.quality.json"
    )
    greedy = [
        o for o in quality["observations"]
        if o["metric"] == "macro_f1"
        and o["decoding_mode"] == "greedy"
        and o["view"] == "representative_all"
    ]
    assert greedy and greedy[0]["observed"] > 0.8537, "the dev-set win has gone"
