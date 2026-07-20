"""Tests for the HTML scrapers (Imedi, Formula).

Run entirely offline against trimmed page snapshots (fixtures/imedi_archive.html,
fixtures/formula_list.html) — fast, deterministic, no secrets, no network. Each fixture
holds the first 4 real cards from the live listing; article lead text is stripped from
the Imedi snapshot so no article prose is committed (copyright, PLAN.md §9). Regenerate
by re-fetching the listing pages and keeping the first few cards.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.scrape import parse_formula, parse_imedi

FIXTURES = Path(__file__).parent / "fixtures"
IMEDI_ID = 5
FORMULA_ID = 6


# --- Imedi ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def imedi():
    return parse_imedi((FIXTURES / "imedi_archive.html").read_bytes(), IMEDI_ID)


def test_imedi_finds_every_card(imedi):
    assert len(imedi) == 4


def test_imedi_georgian_headlines_survive_intact(imedi):
    """Encoding check: Georgian script must round-trip, not turn into mojibake."""
    assert imedi[0].headline == (
        "უკრაინის გენერალური შტაბი სირსკის თანამდებობიდან "
        "გადაყენების შესახებ გავრცელებულ ცნობებს უარყოფს"
    )
    assert all(a.headline.strip() == a.headline and a.headline for a in imedi)


def test_imedi_urls_are_absolute_and_unique(imedi):
    urls = [a.url for a in imedi]
    assert all(u.startswith("https://imedinews.ge/ge/") for u in urls)
    assert len(set(urls)) == len(urls)


def test_imedi_local_time_is_converted_to_utc(imedi):
    """Imedi prints Tbilisi time (UTC+4); we store UTC. 00:23 local → 20:23 prior day."""
    assert imedi[0].published_at == datetime(2026, 7, 20, 20, 23, tzinfo=UTC)
    assert all(a.published_at.tzinfo is UTC for a in imedi)


def test_imedi_source_id_is_attached(imedi):
    assert all(a.source_id == IMEDI_ID for a in imedi)


def test_imedi_row_shape_matches_articles_table(imedi):
    row = imedi[0].to_row()
    assert set(row) == {"source_id", "headline", "url", "published_at"}
    assert row["published_at"] == "2026-07-20T20:23:00+00:00"


def test_imedi_skips_a_card_with_no_title():
    """A card missing its headline cannot be stored — dropped, not inserted half-formed."""
    html = b'<a href="/ge/politika/1/x" class="single-item"><div class="info-wrap"></div></a>'
    assert parse_imedi(html, IMEDI_ID) == []


def test_imedi_unreadable_date_keeps_the_article():
    """A headline is worth storing even when the date is missing or unparseable."""
    html = (
        '<a href="/ge/politika/1/x" class="single-item">'
        '<p class="date">not a date</p><h3 class="title">სათაური</h3></a>'
    ).encode()
    (article,) = parse_imedi(html, IMEDI_ID)
    assert article.published_at is None
    assert article.headline == "სათაური"


def test_imedi_empty_page_returns_nothing():
    """0 articles is the signal the scraper broke — it must not crash."""
    assert parse_imedi(b"<html><body></body></html>", IMEDI_ID) == []


# --- Formula --------------------------------------------------------------------


@pytest.fixture(scope="module")
def formula():
    return parse_formula((FIXTURES / "formula_list.html").read_bytes(), FORMULA_ID)


def test_formula_finds_every_card(formula):
    """4 cards, not 8: each card links twice (thumbnail + headline) and we dedupe."""
    assert len(formula) == 4


def test_formula_georgian_headlines_survive_intact(formula):
    assert formula[0].headline == (
        "სააგენტო: 21 ივლისის დილამდე თბილისში და შემოგარენში მოსალოდნელია წვიმა"
    )
    assert all(a.headline.strip() == a.headline and a.headline for a in formula)


def test_formula_urls_are_absolute_and_unique(formula):
    urls = [a.url for a in formula]
    assert all(u.startswith("https://formulanews.ge/News/") for u in urls)
    assert len(set(urls)) == len(urls)


def test_formula_has_no_publish_time(formula):
    """The listing shows only relative times, so published_at is intentionally None."""
    assert all(a.published_at is None for a in formula)


def test_formula_source_id_is_attached(formula):
    assert all(a.source_id == FORMULA_ID for a in formula)


def test_formula_ignores_the_image_only_link():
    """The thumbnail anchor carries no text and must not become a headline-less row."""
    html = (
        '<div class="card"><a href="/News/1"><img src="x.png"/></a>'
        '<a href="/News/1">რეალური სათაური</a></div>'
    ).encode()
    (article,) = parse_formula(html, FORMULA_ID)
    assert article.headline == "რეალური სათაური"
    assert article.url == "https://formulanews.ge/News/1"


def test_formula_empty_page_returns_nothing():
    assert parse_formula(b"<html><body></body></html>", FORMULA_ID) == []
