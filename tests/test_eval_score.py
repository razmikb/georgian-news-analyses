"""Tests for the threshold maths.

No network, no API key: every test hands the metric functions synthetic scores whose
right answer is known by construction. This is the arithmetic that decides the
clustering threshold, so a quiet mistake here would be inherited by coverage bars and
blindspots without anything looking broken.
"""

import math

import pytest

from pipeline import extract
from pipeline.eval_score import (
    Scored,
    ambiguous_band,
    auc,
    best_threshold,
    common_subset,
    score_at,
    share_in_band,
    variant_text,
    worst_misses,
)


def scored(*pairs) -> list[Scored]:
    """Build eval rows from (label, score) tuples."""
    return [
        Scored(pair_id=str(i), label=label, score=score, headline_a="ა", headline_b="ბ")
        for i, (label, score) in enumerate(pairs, start=1)
    ]


# --- AUC: the threshold-free way to compare text variants ------------------------


def test_auc_is_one_when_every_same_outranks_every_different():
    assert auc(scored(("same", 0.9), ("same", 0.8), ("different", 0.3))) == 1.0


def test_auc_is_zero_when_the_ordering_is_exactly_backwards():
    assert auc(scored(("same", 0.1), ("different", 0.9))) == 0.0


def test_auc_is_half_for_a_coin_flip():
    """One `same` scores top, the other bottom — right as often as it is wrong."""
    assert auc(scored(("same", 0.9), ("different", 0.8), ("different", 0.2), ("same", 0.1))) == 0.5


def test_auc_counts_a_tie_as_half_a_win():
    assert auc(scored(("same", 0.5), ("different", 0.5))) == 0.5


def test_auc_ignores_unsure_rows():
    """`unsure` has no right answer, so it cannot be scored against one."""
    with_unsure = scored(("same", 0.9), ("different", 0.2), ("unsure", 0.5))
    without = scored(("same", 0.9), ("different", 0.2))
    assert auc(with_unsure) == auc(without)


def test_auc_of_a_one_sided_set_is_not_a_number():
    """No positives means nothing to measure — better an obvious NaN than a fake 0.0."""
    assert math.isnan(auc(scored(("different", 0.2), ("different", 0.3))))


# --- Precision / recall / F1 -----------------------------------------------------


def test_score_at_a_perfect_threshold():
    precision, recall, f1 = score_at(scored(("same", 0.9), ("different", 0.2)), 0.5)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_threshold_is_inclusive():
    """A pair scoring exactly at the threshold counts as a match."""
    precision, recall, _ = score_at(scored(("same", 0.5)), 0.5)
    assert (precision, recall) == (1.0, 1.0)


def test_a_too_loose_threshold_costs_precision():
    result = scored(("same", 0.9), ("different", 0.8), ("different", 0.7))
    precision, recall, _ = score_at(result, 0.5)
    assert recall == 1.0
    assert precision == pytest.approx(1 / 3)


def test_a_too_tight_threshold_costs_recall():
    result = scored(("same", 0.9), ("same", 0.6), ("different", 0.1))
    precision, recall, _ = score_at(result, 0.8)
    assert precision == 1.0
    assert recall == 0.5


def test_a_threshold_nothing_reaches_scores_zero_rather_than_dividing_by_zero():
    assert score_at(scored(("same", 0.4), ("different", 0.1)), 0.99) == (0.0, 0.0, 0.0)


def test_best_threshold_finds_the_clean_split():
    result = scored(("same", 0.9), ("same", 0.85), ("different", 0.4), ("different", 0.2))
    threshold, precision, recall, f1 = best_threshold(result)
    assert f1 == 1.0
    assert threshold == 0.85


# --- The ambiguous band: the actual deliverable ----------------------------------


def test_band_reports_clean_separation():
    """The strongest possible result: any threshold in the gap is perfect."""
    low, high, separated = ambiguous_band(
        scored(("same", 0.9), ("same", 0.8), ("different", 0.4), ("different", 0.3))
    )
    assert separated
    assert (low, high) == (0.4, 0.8)


def test_band_spans_the_overlap_when_classes_mix():
    """A `different` pair scoring 0.85 and a `same` pair scoring 0.6 means anything in
    between is genuinely undecidable from the number alone — that is the verifier's job."""
    low, high, separated = ambiguous_band(
        scored(("same", 0.95), ("same", 0.6), ("different", 0.85), ("different", 0.2))
    )
    assert not separated
    assert (low, high) == (0.6, 0.85)


def test_band_ignores_unsure_rows():
    low, high, _ = ambiguous_band(scored(("same", 0.9), ("different", 0.3), ("unsure", 0.99)))
    assert (low, high) == (0.3, 0.9)


def test_band_of_a_one_sided_set_is_not_a_number():
    low, high, separated = ambiguous_band(scored(("different", 0.2)))
    assert math.isnan(low) and math.isnan(high)
    assert not separated


def test_share_in_band_counts_the_verifier_workload():
    result = scored(("same", 0.9), ("different", 0.7), ("different", 0.5), ("same", 0.1))
    assert share_in_band(result, 0.4, 0.8) == 0.5


def test_share_in_band_of_nothing_is_zero():
    assert share_in_band([], 0.4, 0.8) == 0.0


# --- Worst misses: the rows a human should actually read -------------------------


def test_worst_misses_surfaces_the_closest_calls():
    result = scored(("different", 0.88), ("different", 0.1), ("same", 0.95), ("same", 0.32))
    false_alarms, misses = worst_misses(result, count=1)
    assert [s.score for s in false_alarms] == [0.88]
    assert [s.score for s in misses] == [0.32]


# --- Variant construction --------------------------------------------------------


def test_headline_variant_needs_no_article_text():
    """Every pair is scorable this way, which is why it is the baseline."""
    assert variant_text("headline", "სათაური", None) == "სათაური"


def test_lead_variant_is_impossible_without_text():
    assert variant_text("headline+lead", "სათაური", None) is None


# Long enough to clear the "is this even an article?" floor (extract.MIN_TEXT_CHARS).
ARTICLE = "\n".join(
    [
        "პირველი აბზაცი, რომელიც საკმარისად გრძელია იმისთვის, რომ ნამდვილ სტატიად ჩაითვალოს.",
        "მეორე აბზაცი, ასევე გრძელი და შინაარსიანი, რომელიც ლიდის ნაწილია.",
        "მესამე აბზაცი, რომელიც მხოლოდ სრულ ტექსტში უნდა მოხვდეს და არა ლიდში.",
    ]
)


def test_lead_variant_appends_the_opening_paragraphs():
    result = variant_text("headline+lead", "სათაური", ARTICLE)
    assert result.startswith("სათაური\n")
    assert "პირველი აბზაცი" in result
    assert "მესამე აბზაცი" not in result


def test_body_variant_includes_later_paragraphs():
    assert "მესამე აბზაცი" in variant_text("headline+body", "სათაური", ARTICLE)


def test_an_article_that_is_all_boilerplate_is_unusable(monkeypatch):
    """Imedi video stubs are mostly sidebar. Once the furniture is stripped there can be
    nothing left to embed — and no lead or body is better than a lead made of scraps."""
    text = "სათაური\nფოტო: IMEDI\n25 ივლისი 2026, 23:42\nსხვა სტატია რომელიც აქ არ ეკუთვნის."
    assert variant_text("headline+lead", "სათაური", text) is None


def test_body_variant_is_capped(monkeypatch):
    monkeypatch.setattr(extract, "BODY_MAX_CHARS", 30)
    result = variant_text("headline+body", "სათაური", "ა" * 500)
    assert len(result) < 100


# --- Comparing variants fairly ---------------------------------------------------


def test_common_subset_is_what_every_variant_could_score():
    """An article whose page would not load has no lead and no body, so it drops out of
    two variants. Comparing on different subsets would flatter whichever got the easier one."""
    results = {
        "headline": scored(("same", 0.1), ("same", 0.2), ("different", 0.3)),
        "headline+lead": scored(("same", 0.1), ("same", 0.2)),
        "headline+body": scored(("same", 0.1)),
    }
    assert common_subset(results) == {"1"}


def test_common_subset_of_nothing_is_empty():
    assert common_subset({}) == set()
