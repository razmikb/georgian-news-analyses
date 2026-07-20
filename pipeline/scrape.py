"""HTML scrapers for outlets that publish no RSS feed (Imedi, Formula).

Mirrors `pipeline.rss`: each parser takes raw page bytes (not a URL), so the whole
parsing path is testable offline against a saved snapshot (tests/fixtures/). We keep
ONLY the headline, the link, and — where the listing exposes it — the publish time.
Never article body text (copyright; PLAN.md §9).

Scrapers are the fragile part of the system (PLAN.md §9): a site can change its HTML
without warning. So each parser is small and forgiving — a card missing its title or
link is skipped, not fatal — and run.py isolates a broken source so the rest keep
running. A scraper that yields 0 articles is the signal its selectors need a look.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from pipeline.rss import Article

# Georgian outlets show local time. Georgia is UTC+4 year-round (no daylight saving),
# so we read a listing timestamp as +04:00 and convert to UTC — Postgres stores UTC and
# must never have to guess an offset.
_TBILISI = timezone(timedelta(hours=4))


def _clean(text: str) -> str:
    """Collapse the whitespace and newlines inside a headline down to single spaces."""
    return " ".join(text.split())


# --- Imedi (imedinews.ge) -------------------------------------------------------

_IMEDI_BASE = "https://imedinews.ge"
_KA_MONTHS = {
    "იანვარი": 1,
    "თებერვალი": 2,
    "მარტი": 3,
    "აპრილი": 4,
    "მაისი": 5,
    "ივნისი": 6,
    "ივლისი": 7,
    "აგვისტო": 8,
    "სექტემბერი": 9,
    "ოქტომბერი": 10,
    "ნოემბერი": 11,
    "დეკემბერი": 12,
}
# Imedi prints dates as "21 ივლისი 2026, 00:23" (day, Georgian month, year, HH:MM).
_IMEDI_DATE_RE = re.compile(r"(\d{1,2})\s+(\S+)\s+(\d{4}).*?(\d{1,2}):(\d{2})", re.S)


def _imedi_published_at(card) -> datetime | None:
    """Parse Imedi's local date/time into an aware UTC datetime, or None if unreadable."""
    date_el = card.select_one("p.date")
    if not date_el:
        return None
    match = _IMEDI_DATE_RE.search(date_el.get_text(" ", strip=True))
    if not match:
        return None
    day, month_name, year, hour, minute = match.groups()
    month = _KA_MONTHS.get(month_name)
    if not month:
        return None
    try:
        local = datetime(int(year), month, int(day), int(hour), int(minute), tzinfo=_TBILISI)
    except ValueError:
        return None
    return local.astimezone(UTC)


def parse_imedi(html: bytes, source_id: int) -> list[Article]:
    """Extract headlines from Imedi's archive page (imedinews.ge/ge/archive)."""
    soup = BeautifulSoup(html, "html.parser")
    articles: list[Article] = []
    for card in soup.select("a.single-item"):
        href = (card.get("href") or "").strip()
        title_el = card.select_one("h3.title")
        headline = _clean(title_el.get_text()) if title_el else ""
        if not href or not headline:
            continue
        articles.append(
            Article(
                source_id=source_id,
                headline=headline,
                url=urljoin(_IMEDI_BASE, href),
                published_at=_imedi_published_at(card),
            )
        )
    return articles


# --- Formula (formulanews.ge) ---------------------------------------------------

_FORMULA_BASE = "https://formulanews.ge"


def parse_formula(html: bytes, source_id: int) -> list[Article]:
    """Extract headlines from Formula's listing page (formulanews.ge/Category/All).

    Each card links to the article twice — once wrapping the thumbnail (no text) and
    once as the headline itself. We keep only anchors that carry text and dedupe by
    URL, so the image link is dropped rather than double-counted.

    Formula's listing shows only relative times ("1 hour ago"), so published_at is left
    None. The exact time would cost an extra fetch per article — revisit if the 48h
    clustering window needs it (PROGRESS.md).
    """
    soup = BeautifulSoup(html, "html.parser")
    articles: list[Article] = []
    seen: set[str] = set()
    for anchor in soup.select('a[href^="/News/"]'):
        headline = _clean(anchor.get_text())
        href = (anchor.get("href") or "").strip()
        if not headline or not href or href in seen:
            continue
        seen.add(href)
        articles.append(
            Article(
                source_id=source_id,
                headline=headline,
                url=urljoin(_FORMULA_BASE, href),
                published_at=None,
            )
        )
    return articles


# --- Registry -------------------------------------------------------------------


@dataclass(frozen=True)
class Scraper:
    """Where a scraped source's listing lives, and how to parse it.

    The listing URL sits next to its parser on purpose: the two are a matched pair,
    written together against one page's structure and changing together if the site
    restructures. The source *list* and leaning labels stay in the DB (CLAUDE.md);
    this is the per-site recipe, which PLAN.md §9 expects to be code.
    """

    listing_url: str
    parse: Callable[[bytes, int], list[Article]]


SCRAPERS: dict[str, Scraper] = {
    "imedi": Scraper("https://imedinews.ge/ge/archive", parse_imedi),
    "formula": Scraper("https://formulanews.ge/Category/All", parse_formula),
}
