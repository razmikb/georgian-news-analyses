"""Tests for the eval-set pair sampler.

Offline: the pure scoring/selection functions are fed made-up Georgian headlines, so no
database and no network. `load_items` (the only part that talks to Supabase) is not
covered here — the sampling logic is what can silently go wrong.
"""

from datetime import UTC, datetime, timedelta

import pytest

from pipeline.eval_pairs import (
    BAND_SIZES,
    FIELDS,
    MAX_PAIRS_PER_ARTICLE,
    Item,
    candidate_pairs,
    normalize,
    select,
    similarity,
    trigrams,
    write_csv,
)

NOON = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def item(id_: int, headline: str, source: str = "Netgazeti", hours: float = 0) -> Item:
    return Item(
        id=id_,
        headline=headline,
        url=f"https://example.ge/{id_}",
        source=source,
        leaning="independent",
        when=NOON + timedelta(hours=hours),
        grams=trigrams(headline),
    )


# --- scoring --------------------------------------------------------------------


def test_normalize_strips_punctuation_and_odd_spacing():
    # \xa0 is the non-breaking space feeds emit; em dashes and quotes vary by outlet.
    assert normalize("„პროტესტი“ —\xa0თბილისში!") == "პროტესტი თბილისში"


def test_identical_headlines_score_one():
    grams = trigrams("პროკურატურა გირაოს მოითხოვს")
    assert similarity(grams, grams) == 1.0


def test_punctuation_differences_do_not_split_a_pair():
    """Outlets write the same sentence with different dashes and quotes."""
    a = trigrams("თიბისის და FMO-ს პარტნიორობა – 210 მლნ დოლარი")
    b = trigrams("თიბისის და FMO-ს პარტნიორობა — 210 მლნ დოლარი")
    assert similarity(a, b) > 0.9


def test_unrelated_headlines_score_near_zero():
    a = trigrams("ლეჩხუმში არქეოლოგიური აღმოჩენა")
    b = trigrams("ბრიტანეთის პრემიერმინისტრი გადადგა")
    assert similarity(a, b) < 0.1


def test_shared_stem_survives_georgian_case_endings():
    """The reason for character trigrams: Georgian glues endings onto nouns."""
    a = trigrams("აქცია თბილისში გაიმართა")
    b = trigrams("აქციაზე თბილისში დააკავეს")
    assert similarity(a, b) > 0.2


def test_empty_headline_scores_zero_instead_of_dividing_by_zero():
    assert similarity(trigrams(""), trigrams("აქცია თბილისში")) == 0.0


# --- pairing --------------------------------------------------------------------


def test_pairs_are_every_combination_once():
    items = [item(i, f"სათაური {i}") for i in range(4)]
    assert len(candidate_pairs(items)) == 6


def test_pairs_outside_the_48h_window_are_dropped():
    items = [item(1, "სათაური ერთი"), item(2, "სათაური ორი", hours=60)]
    assert candidate_pairs(items) == []


def test_pairs_come_back_most_similar_first():
    items = [
        item(1, "პროკურატურა გირაოს მოითხოვს"),
        item(2, "პროკურატურა გირაოს მოითხოვს დღეს"),
        item(3, "ამინდი გაუარესდება კვირას"),
    ]
    scores = [score for score, _, _ in candidate_pairs(items)]
    assert scores == sorted(scores, reverse=True)


# --- selection ------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    """A corpus with enough variety to fill every band."""
    items = []
    for n in range(40):
        # Near-duplicates across two outlets → the "similar" band.
        items.append(item(n * 4, f"პროკურატურა გირაოს მოითხოვს საქმეზე ნომერი {n}", "Imedi News"))
        items.append(item(n * 4 + 1, f"პროკურატურა გირაოს მოითხოვს საქმეზე ნომერი {n}", "Formula"))
        # Same outlet twice → the "same-outlet" band.
        items.append(item(n * 4 + 2, f"პროკურატურა გირაოს მოითხოვს ნომერი {n}", "Imedi News"))
        # Unrelated wording → the "unrelated" band.
        items.append(item(n * 4 + 3, f"ამინდი {n} გაუარესდება მთიან რეგიონებში ხვალ", "Publika"))
    return select(candidate_pairs(items))


def test_no_headline_is_overused(sample):
    """Otherwise one busy news day fills the sheet, seen from twelve angles."""
    # Counted by URL, not headline: two outlets can publish a byte-identical headline
    # (re-published press releases do exactly this) and those are still two articles.
    seen: dict[str, int] = {}
    for row in sample:
        for url in (row["url_a"], row["url_b"]):
            seen[url] = seen.get(url, 0) + 1
    assert max(seen.values()) <= MAX_PAIRS_PER_ARTICLE


def test_no_pair_is_a_headline_against_itself(sample):
    assert all(row["url_a"] != row["url_b"] for row in sample)


def test_the_same_pair_never_appears_twice(sample):
    keys = {frozenset((row["url_a"], row["url_b"])) for row in sample}
    assert len(keys) == len(sample)


def test_cross_source_bands_really_are_cross_source(sample):
    cross = [r for r in sample if r["band"] != "D-same-outlet"]
    assert cross and all(r["source_a"] != r["source_b"] for r in cross)


def test_same_outlet_band_really_is_one_outlet(sample):
    same = [r for r in sample if r["band"] == "D-same-outlet"]
    assert same and all(r["source_a"] == r["source_b"] for r in same)


def test_label_column_is_left_empty_for_the_human(sample):
    assert all(row["label"] == "" and row["notes"] == "" for row in sample)


def test_selection_is_repeatable():
    """Same corpus, same sheet — so a regenerate doesn't reshuffle a half-labelled file."""
    items = [
        item(i, f"სათაური ნომერი {i} თბილისში", "Imedi News" if i % 2 else "Formula")
        for i in range(30)
    ]
    pairs = candidate_pairs(items)
    assert [r["url_a"] for r in select(pairs)] == [r["url_a"] for r in select(pairs)]


def test_bands_are_never_over_filled(sample):
    for row in sample:
        assert row["band"] in BAND_SIZES
    for band, wanted in BAND_SIZES.items():
        assert len([r for r in sample if r["band"] == band]) <= wanted


# --- output ---------------------------------------------------------------------


def test_csv_opens_as_utf8_with_bom_so_excel_shows_georgian(tmp_path, sample):
    path = tmp_path / "pairs.csv"
    write_csv(sample, str(path))
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert text.splitlines()[0] == ",".join(FIELDS)
    assert "პროკურატურა" in text


def test_csv_keeps_urls_for_every_pair(tmp_path, sample):
    import csv

    path = tmp_path / "pairs.csv"
    write_csv(sample, str(path))
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(sample)
    assert all(r["url_a"].startswith("https://") for r in rows)
    assert all(r["url_b"].startswith("https://") for r in rows)
