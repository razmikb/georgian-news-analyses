"""RSS feed parsing.

`parse_feed` takes raw bytes rather than a URL so the whole parsing path is testable
offline against a saved snapshot (tests/fixtures/).

We keep ONLY the headline, the link, and the publish time — never article body text
(copyright; see PLAN.md §9).
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser


@dataclass(frozen=True)
class Article:
    """One headline ready to be written to the `articles` table."""

    source_id: int
    headline: str
    url: str
    published_at: datetime | None

    def to_row(self) -> dict:
        return {
            "source_id": self.source_id,
            "headline": self.headline,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }


def _published_at(entry) -> datetime | None:
    """Convert feedparser's parsed time tuple to an aware UTC datetime.

    feedparser normalises the feed's timezone to UTC already but hands back a naive
    struct_time, so we attach UTC explicitly rather than letting Postgres guess.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=UTC)


def parse_feed(xml: bytes, source_id: int) -> list[Article]:
    """Parse RSS/Atom bytes into Articles, skipping unusable entries.

    An entry with no headline or no link cannot be stored or deduped, so it is
    dropped rather than inserted half-formed.
    """
    feed = feedparser.parse(xml)
    articles: list[Article] = []

    for entry in feed.entries:
        headline = (entry.get("title") or "").strip()
        url = (entry.get("link") or "").strip()
        if not headline or not url:
            continue
        articles.append(
            Article(
                source_id=source_id,
                headline=headline,
                url=url,
                published_at=_published_at(entry),
            )
        )

    return articles
