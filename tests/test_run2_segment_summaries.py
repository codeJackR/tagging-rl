from __future__ import annotations

from dataclasses import replace

import pytest

from training.analyze_run2_candidates import GroupObservation
from training.run2_replay_adapter import (
    AdaptedReplayGroup,
    ClassContribution,
    FieldContribution,
    GroupSegments,
)
from training.run2_segment_summaries import (
    summarize_contribution_dominance,
    summarize_in_memory_diagnostics,
    summarize_product_segments,
)


def _adapted_group(
    position: int,
    *,
    category: str,
    difficulty: str,
    known_count: int,
    reverse_candidate: bool = False,
) -> AdaptedReplayGroup:
    group_id = f"synthetic-{position}"
    target = tuple(float(value) for value in range(8))
    coverage = tuple(float(8 - value) for value in range(8))
    increasing = tuple(value / 10 for value in range(8))
    decreasing = tuple(reversed(increasing))
    rewards = {
        "original": (0, 0, 0, 0, 1, 1, 1, 1),
        "U": decreasing if reverse_candidate else increasing,
        "UA": increasing,
        "CB": increasing,
    }
    targets = {
        "canonical_known_utility": target,
        "known_exact_rate": target,
        "known_coverage": coverage,
        "selective_correctness": target,
        "unknown_abstention_rate": tuple(reversed(target)),
        "rule_quality": target,
        "class_balanced_known_utility": target,
    }
    field_rows = []
    class_rows = []
    for rollout_index in range(8):
        for candidate in ("U", "UA", "CB"):
            # Across the whole group, neckline contributes 80% and fit 20%.
            for field_name, contribution in (("neckline", 0.8), ("fit", 0.2)):
                field_rows.append(
                    FieldContribution(
                        group_id=group_id,
                        rollout_index=rollout_index,
                        candidate=candidate,
                        field_name=field_name,
                        component="known",
                        utility=1.0,
                        signed_contribution=contribution,
                        absolute_contribution=contribution,
                    )
                )
                if candidate == "CB":
                    support, band, class_key, weight = (
                        (3, "rare_1_4", "v_neck", 2.0)
                        if field_name == "neckline"
                        else (80, "common_50_plus", "regular", 0.5)
                    )
                    class_rows.append(
                        ClassContribution(
                            group_id=group_id,
                            rollout_index=rollout_index,
                            candidate="CB",
                            field_name=field_name,
                            class_key=class_key,
                            class_support=support,
                            class_support_band=band,
                            class_weight=weight,
                            absolute_contribution=contribution,
                        )
                    )
    return AdaptedReplayGroup(
        version="synthetic",
        group_position=position,
        observation=GroupObservation(
            group_id=group_id,
            rewards=rewards,
            targets=targets,
        ),
        segments=GroupSegments(
            product_category=category,
            difficulty_band=difficulty,
            gold_known_field_count=known_count,
        ),
        field_contributions=tuple(field_rows),
        class_contributions=tuple(class_rows),
        boundary={"file_io_performed": False},
    )


def _groups():
    return [
        _adapted_group(
            0,
            category="dress",
            difficulty="low_mixed_1_2_of_8",
            known_count=2,
        ),
        _adapted_group(
            1,
            category="dress",
            difficulty="middle_mixed_3_5_of_8",
            known_count=5,
            reverse_candidate=True,
        ),
        _adapted_group(
            2,
            category="top",
            difficulty="middle_mixed_3_5_of_8",
            known_count=5,
        ),
    ]


def test_field_and_class_dominance_preserve_separate_child_grains():
    result = summarize_contribution_dominance(_groups())

    assert result["groups"] == 3
    for candidate in ("U", "UA", "CB"):
        summary = result["field_contributions"][candidate]
        assert summary["child_rows"] == 3 * 8 * 2
        assert summary["largest_field"] == "neckline"
        assert summary["largest_field_share"] == pytest.approx(0.8)
        assert summary["dominance_gate_applied"] is False
        assert summary["per_group_largest_field_share_distribution"]["median"] == pytest.approx(0.8)

    classes = result["cb_class_contributions"]
    assert classes["child_rows"] == 3 * 8 * 2
    assert classes["largest_class"] == "neckline::v_neck"
    assert classes["largest_class_share"] == pytest.approx(0.8)
    assert classes["support_bands"]["rare_1_4"]["share"] == pytest.approx(0.8)
    assert classes["support_bands"]["common_50_plus"]["share"] == pytest.approx(0.2)
    assert result["grain_guardrails"]["class_child_rows_used_as_product_denominator"] is False


def test_product_segments_cover_each_product_once_per_dimension():
    result = summarize_product_segments(_groups())

    assert result["groups"] == 3
    for dimension in result["dimensions"].values():
        assert dimension["group_memberships"] == 3
        assert dimension["every_group_appears_exactly_once"] is True
    categories = result["dimensions"]["product_category"]["segments"]
    assert categories["dress"]["groups"] == 2
    assert categories["dress"]["completions"] == 16
    assert categories["top"]["groups"] == 1
    assert categories["dress"]["interpretation_allowed"] is False
    assert categories["dress"]["minimum_groups_for_interpretation"] == 30
    assert categories["dress"]["reward_summaries"]["original"][
        "unique_reward_levels_per_group_histogram"
    ] == {"2": 2}


def test_segment_alignment_is_group_first_and_direction_can_differ_by_product():
    result = summarize_product_segments(_groups())
    dress = result["dimensions"]["product_category"]["segments"]["dress"]
    alignment = dress["directional_alignments"]["U"][
        "canonical_known_utility"
    ]
    assert alignment["groups_contributing"] == 2
    # One perfectly aligned product and one perfectly reversed product.
    assert alignment["group_net_alignment_distribution"]["mean"] == 0.0
    assert alignment["group_net_alignment_distribution"]["median"] == 0.0


def test_integrated_diagnostics_are_in_memory_and_do_not_apply_gates():
    result = summarize_in_memory_diagnostics(_groups())
    assert result["status"] == "in_memory_diagnostics_completed"
    assert result["boundary"] == {
        "file_io_performed": False,
        "real_candidate_replay_opened_by_this_module": False,
        "acceptance_gates_applied": False,
        "winner_selected": False,
    }
    assert result["dominance"]["groups"] == 3
    assert result["product_segments"]["groups"] == 3


def test_summaries_fail_closed_on_duplicate_groups_and_child_join_errors():
    groups = _groups()
    with pytest.raises(ValueError, match="duplicate group IDs"):
        summarize_in_memory_diagnostics([groups[0], groups[0]])

    bad_position = replace(groups[1], group_position=3)
    with pytest.raises(ValueError, match="canonical and contiguous"):
        summarize_product_segments([groups[0], bad_position, groups[2]])

    bad_field_rows = list(groups[0].field_contributions)
    bad_field_rows.append(bad_field_rows[0])
    duplicate_field = replace(groups[0], field_contributions=tuple(bad_field_rows))
    with pytest.raises(ValueError, match="duplicate field-contribution key"):
        summarize_contribution_dominance([duplicate_field])

    bad_classes = list(groups[0].class_contributions)
    bad_classes[0] = replace(bad_classes[0], absolute_contribution=0.7)
    bad_allocation = replace(groups[0], class_contributions=tuple(bad_classes))
    with pytest.raises(ValueError, match="do not reconstruct CB field"):
        summarize_contribution_dominance([bad_allocation])

    bad_band_rows = list(groups[0].class_contributions)
    bad_band_rows[0] = replace(bad_band_rows[0], class_support_band="common_50_plus")
    bad_band = replace(groups[0], class_contributions=tuple(bad_band_rows))
    with pytest.raises(ValueError, match="support band does not match"):
        summarize_contribution_dominance([bad_band])
