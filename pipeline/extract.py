"""Pull the readable text out of an article page.

Clustering on headlines alone was judged too thin (PLAN.md §6): vague headlines carry
almost no signal, and rival outlets word the same event very differently — which is
precisely this product's subject matter. So we read the article body too.

**Used, never stored.** The text here feeds an embedding and is then dropped. It never
reaches Supabase, the site, or a commit. The only place it touches disk is the
git-ignored `.cache/` (see `pipeline.cache`), so that tuning does not mean re-fetching
120 pages every time.

How much of the body to use is a tuned dial, not an assumption — `lead()` and `body()`
exist so the eval set can decide between them.
"""

from __future__ import annotations

import re

import trafilatura

from pipeline import cache
from pipeline.fetch import FetchError, fetch

CACHE_NAMESPACE = "article-text"

# Roughly how much text an embedding gets from the "full body" variant. Long articles
# are truncated because trailing paragraphs are usually background and boilerplate that
# every story on a topic shares — the exact effect (unrelated stories pulled together)
# that PLAN.md §6 warns full bodies can cause.
BODY_MAX_CHARS = 2000

# How many opening paragraphs count as the "lead". Georgian news leads, like most, put
# who/what/where in the first two.
LEAD_PARAGRAPHS = 2

# Below this many characters, treat the page as having no article at all.
# Extraction degrades rather than fails: on a page with no real prose it will happily
# hand back a nav label or a cookie notice. That scrap is worse than nothing, because a
# vector built from junk sits close to every *other* junk vector — so the pipeline would
# quietly cluster unrelated broken pages together and call it an event. No real Georgian
# news article is this short.
MIN_TEXT_CHARS = 120

# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate removal
#
# Extraction returns the article *and* whatever else the page renders around it.
# On Imedi that is severe: a real article runs ~400 characters and is followed by
# ~9,600 characters of "latest news" — other stories, in full. Left in, every pair
# of Imedi articles would share those 9,600 identical characters and score as nearly
# the same event no matter what they were about. The measurement would look fine and
# mean nothing.
#
# These rules are generic rather than per-site on purpose: the same shapes (a repeated
# headline, a photo credit, a timestamped list of other stories) appear on every news
# site, and per-site selectors are the part of a scraper that rots first.
# ─────────────────────────────────────────────────────────────────────────────

# A standalone "25 July 2026, 23:42" line. Real prose does not consist of a bare
# timestamp, but every article listing stamps one on each entry — so the first such
# line is where the article stopped and the sidebar began.
_TIMESTAMP_LINE = re.compile(r"^(?=.*\d{4})(?=.*\d{1,2}:\d{2})[^.!?]{0,40}$")

# Photo/video credits sit between the headline and the first paragraph.
_CREDIT_LINE = re.compile(r"^\s*(ფოტო|ვიდეო|photo|video|image)\b", re.IGNORECASE)

# Below this, a trailing line is a nav crumb ("all news"), not a paragraph.
_MIN_TRAILING_PARAGRAPH = 20


def paragraphs(text: str) -> list[str]:
    """Split extracted text into non-empty paragraphs."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _is_boilerplate(line: str, headline: str) -> bool:
    """A repeated headline or a photo credit — present on the page, not part of the story."""
    if _CREDIT_LINE.match(line) and len(line) < 60:
        return True
    return bool(headline) and line.casefold() == headline.strip().casefold()


def clean(text: str, headline: str = "") -> str:
    """Strip page furniture and cut everything from the article listing onward.

    Kept separate from fetching so the cache can hold the raw extraction: these rules
    will change as we meet more sites, and re-tuning them must never mean re-fetching
    (or re-knocking on Imedi's bot filter).
    """
    kept: list[str] = []
    for line in paragraphs(text):
        if _TIMESTAMP_LINE.match(line):
            break  # everything after this is other people's stories
        if _is_boilerplate(line, headline):
            continue
        kept.append(line)

    # The listing's own heading ("all news") sits just above the first timestamp.
    while kept and len(kept[-1]) < _MIN_TRAILING_PARAGRAPH:
        kept.pop()
    return "\n".join(kept)


def lead(text: str, count: int | None = None) -> str:
    """The opening paragraphs — the part that says what happened.

    The default is read at call time, not bound at import, so the limits above stay
    genuinely adjustable — they are dials the eval set is meant to turn (PLAN.md §6).
    """
    return "\n".join(paragraphs(text)[: count if count is not None else LEAD_PARAGRAPHS])


def body(text: str, max_chars: int | None = None) -> str:
    """The whole article, truncated at a paragraph boundary where possible."""
    max_chars = max_chars if max_chars is not None else BODY_MAX_CHARS
    joined = "\n".join(paragraphs(text))
    if len(joined) <= max_chars:
        return joined
    cut = joined[:max_chars]
    # Prefer ending on a whole paragraph; fall back to a hard cut if the first paragraph
    # is itself longer than the limit.
    boundary = cut.rfind("\n")
    return cut[:boundary] if boundary > 0 else cut


def article_text(url: str, *, use_cache: bool = True) -> str | None:
    """Fetch `url` and return its raw extracted text, or None if there is nothing usable.

    Raw on purpose — `clean()` is applied by the caller, which knows the headline. What
    is cached is therefore what the site actually served, so changing the cleaning rules
    costs a re-parse rather than 99 re-fetches.

    Returns None rather than raising: a single unreachable article must not sink a run
    of 99, the same per-source isolation rule the ingest pipeline follows (CLAUDE.md).
    Three failure modes end here — the page would not load (Imedi's bot filter still
    answers some requests with a 403), it loaded but held no extractable prose, or what
    came back was too short to be an article (see MIN_TEXT_CHARS).
    """
    if use_cache:
        cached = cache.get(CACHE_NAMESPACE, url)
        if cached is not None:
            # Empty string is a cached "this page has no text" — a real answer worth
            # keeping, so we don't re-fetch a known-dead page on every run.
            return cached or None

    try:
        html = fetch(url)
    except FetchError:
        return None

    # favor_precision drops navigation and related-article blocks that the default
    # settings sweep in. Measured on our own sources: it cut On.ge from 53 paragraphs
    # to 9 without losing article text.
    text = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True
    )
    extracted = "\n".join(paragraphs(text)) if text else ""
    if len(extracted) < MIN_TEXT_CHARS:
        extracted = ""

    if use_cache:
        cache.put(CACHE_NAMESPACE, url, extracted)
    return extracted or None
