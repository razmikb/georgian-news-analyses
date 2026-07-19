"""Tests for RSS parsing.

Run entirely offline against a saved Netgazeti snapshot (fixtures/netgazeti_feed.xml),
so they are fast, deterministic, and need no secrets in CI. The snapshot holds 8 real
entries plus 2 deliberately broken ones (one missing a link, one missing a headline).
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.rss import parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "netgazeti_feed.xml"
SOURCE_ID = 1


@pytest.fixture(scope="module")
def articles():
    return parse_feed(FIXTURE.read_bytes(), SOURCE_ID)


def test_skips_entries_missing_a_headline_or_link(articles):
    """The 2 unusable entries must be dropped — they cannot be stored or deduped."""
    assert len(articles) == 8


def test_georgian_headlines_survive_intact(articles):
    """Encoding check: Georgian script must round-trip, not turn into mojibake."""
    first = articles[0]
    assert first.headline == (
        "კახეთის ქორწილის მონაწილე მომღერლის მიმართ ადმინისტრაციული წარმოება დაიწყო"
    )
    assert all(a.headline.strip() == a.headline and a.headline for a in articles)


def test_urls_are_absolute_and_unique(articles):
    urls = [a.url for a in articles]
    assert all(u.startswith("https://netgazeti.ge/") for u in urls)
    assert len(set(urls)) == len(urls)


def test_dates_are_timezone_aware_utc(articles):
    """Postgres must never have to guess a timezone."""
    first = articles[0]
    assert first.published_at == datetime(2026, 7, 19, 20, 3, 21, tzinfo=UTC)
    assert all(a.published_at.tzinfo is UTC for a in articles)


def test_source_id_is_attached(articles):
    assert all(a.source_id == SOURCE_ID for a in articles)


def test_row_shape_matches_articles_table(articles):
    """to_row() must produce exactly the columns the `articles` table expects."""
    row = articles[0].to_row()
    assert set(row) == {"source_id", "headline", "url", "published_at"}
    assert row["published_at"] == "2026-07-19T20:03:21+00:00"


def test_missing_dates_are_allowed():
    """published_at is nullable — a dateless entry is still worth storing."""
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item><title>Headline with no date</title><link>https://example.ge/a</link></item>
    </channel></rss>"""
    (article,) = parse_feed(xml, SOURCE_ID)
    assert article.published_at is None


def test_empty_feed_returns_nothing():
    """A feed that yields 0 articles is the signal a source broke — must not crash."""
    xml = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
    assert parse_feed(xml, SOURCE_ID) == []
