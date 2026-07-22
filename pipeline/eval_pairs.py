"""Build the clustering eval set: candidate headline pairs for a human to label.

Phase 2 needs one number — how similar two articles must be before we call them the
same event. That number is tuned against ~50 pairs of real headlines labelled
same-event / different-event by a native Georgian speaker (PLAN.md §6).

This module picks *which* pairs to put in front of them. Picking at random would be
useless: nearly every random pair of headlines is a different event, so the answer
sheet would be 98% "different" and would say nothing about where the boundary sits.
So pairs are **stratified** by a cheap lexical similarity — some obviously-similar,
a lot of middling ones (where the threshold actually gets decided), a few obviously
unrelated as controls.

The similarity used here is deliberately dumb: shared character triples, no API, no
model. It is only a sampling aid — it decides which pairs are *worth a human's
attention*, never whether they match. The real similarity comes from Gemini
embeddings later, and the labels are what judge it.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from pipeline import db

# Only pair articles published within this window of each other — clustering never
# compares beyond 48h either (PLAN.md §6), so pairs further apart aren't real cases.
PAIR_WINDOW = timedelta(hours=48)

# How many pairs of each kind to hand over. Cross-source pairs are the product's actual
# subject (two outlets, one event); a few same-source pairs cover an outlet running
# several stories on the same event.
BAND_SIZES = {
    "A-similar": 20,
    "B-borderline": 22,
    "C-unrelated": 12,
    "D-same-outlet": 6,
}

# Band edges on the lexical score, calibrated against the real corpus: on 33k cross-source
# pairs the median score is 0.04 and the 99th percentile 0.14, so "0.20 and up" is the far
# tail (128 pairs) and "0.03 and under" is ordinary unrelated noise.
SIMILAR_MIN = 0.20
BORDERLINE_RANGE = (0.07, 0.20)
UNRELATED_MAX = 0.03

# Borderline pairs must also be close in time. Two outlets covering one event do it the
# same day, so this filter raises the share of genuinely hard cases in the band — which is
# the whole point of that band — instead of spending a human's attention on coincidences.
SAME_STORY_WINDOW = timedelta(hours=12)

# No headline may appear in more than this many pairs, so the set covers many different
# stories instead of one busy news day seen from twelve angles.
MAX_PAIRS_PER_ARTICLE = 2

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


@dataclass(frozen=True)
class Item:
    """One article, reduced to what pairing needs."""

    id: int
    headline: str
    url: str
    source: str
    leaning: str
    when: datetime
    grams: frozenset[str]


def normalize(headline: str) -> str:
    """Strip punctuation and odd spacing so wording differences aren't drowned by them.

    Georgian has no upper/lower case, so there is no case folding to do. Feeds do emit
    non-breaking spaces, which `\\s` covers.
    """
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", headline)).strip()


def trigrams(text: str) -> frozenset[str]:
    """Character triples of the normalized text.

    Character n-grams rather than whole words because Georgian glues case endings onto
    nouns — "protest", "of the protest" and "at the protest" are three different words
    sharing a stem, and whole-word matching would count them as unrelated.
    """
    cleaned = normalize(text)
    return frozenset(cleaned[i : i + 3] for i in range(len(cleaned) - 2))


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap of two trigram sets: shared triples ÷ triples in either."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_items() -> list[Item]:
    """Read every stored headline, with its source label and a usable timestamp."""
    sources = {s["id"]: s for s in db.client().table("sources").select("*").execute().data}
    rows = (
        db.client()
        .table("articles")
        .select("id, headline, url, source_id, published_at, fetched_at")
        .order("id")
        .execute()
        .data
    )
    items = []
    for row in rows:
        source = sources[row["source_id"]]
        # Formula's listing page shows only relative times, so those rows have no
        # published_at (PROGRESS.md); when we fetched it is the next best thing.
        stamp = row["published_at"] or row["fetched_at"]
        items.append(
            Item(
                id=row["id"],
                headline=row["headline"],
                url=row["url"],
                source=source["name_en"],
                leaning=source["leaning"],
                when=datetime.fromisoformat(stamp),
                grams=trigrams(row["headline"]),
            )
        )
    return items


def candidate_pairs(items: list[Item]) -> list[tuple[float, Item, Item]]:
    """Score every within-window pair. 351 headlines is ~60k pairs — a second's work."""
    scored = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if abs(a.when - b.when) > PAIR_WINDOW:
                continue
            scored.append((similarity(a.grams, b.grams), a, b))
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored


def _pick(
    pool: list[tuple[float, Item, Item]],
    wanted: int,
    used: Counter,
) -> list[tuple[float, Item, Item]]:
    """Take `wanted` pairs from `pool` in order, skipping over-used headlines."""
    chosen = []
    for score, a, b in pool:
        if len(chosen) == wanted:
            break
        if used[a.id] >= MAX_PAIRS_PER_ARTICLE or used[b.id] >= MAX_PAIRS_PER_ARTICLE:
            continue
        used[a.id] += 1
        used[b.id] += 1
        chosen.append((score, a, b))
    return chosen


def select(scored: list[tuple[float, Item, Item]], seed: int = 20260721) -> list[dict]:
    """Draw the stratified sample: similar, borderline, unrelated, same-outlet."""
    rng = random.Random(seed)
    cross = [p for p in scored if p[1].source != p[2].source]
    same = [p for p in scored if p[1].source == p[2].source]

    def same_day(pair: tuple[float, Item, Item]) -> bool:
        return abs(pair[1].when - pair[2].when) <= SAME_STORY_WINDOW

    bands = {
        "A-similar": [p for p in cross if p[0] >= SIMILAR_MIN],
        "B-borderline": [
            p for p in cross if BORDERLINE_RANGE[0] <= p[0] < BORDERLINE_RANGE[1] and same_day(p)
        ],
        "C-unrelated": [p for p in cross if p[0] <= UNRELATED_MAX],
        "D-same-outlet": [p for p in same if p[0] >= SIMILAR_MIN and same_day(p)],
    }
    # Shuffle each band before drawing from it. Taking them in score order would fill band A
    # with nothing but re-published press releases (identical wording, trivially "same") and
    # leave out the interesting near-misses just below them.
    for pool in bands.values():
        rng.shuffle(pool)

    used: Counter = Counter()
    rows = []
    for band, pool in bands.items():
        for score, a, b in _pick(pool, BAND_SIZES[band], used):
            rows.append(
                {
                    "pair_id": len(rows) + 1,
                    "headline_a": a.headline,
                    "headline_b": b.headline,
                    "label": "",
                    "notes": "",
                    "source_a": a.source,
                    "source_b": b.source,
                    "leaning_a": a.leaning,
                    "leaning_b": b.leaning,
                    "url_a": a.url,
                    "url_b": b.url,
                    "published_a": a.when.isoformat(timespec="minutes"),
                    "published_b": b.when.isoformat(timespec="minutes"),
                    "band": band,
                    "word_overlap": round(score, 3),
                }
            )
    return rows


FIELDS = [
    "pair_id",
    "headline_a",
    "headline_b",
    "label",
    "notes",
    "source_a",
    "source_b",
    "leaning_a",
    "leaning_b",
    "url_a",
    "url_b",
    "published_a",
    "published_b",
    "band",
    "word_overlap",
]


def write_csv(rows: list[dict], path: str) -> None:
    """Write the sheet as UTF-8 *with BOM* — without it Excel mangles Georgian text."""
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    items = load_items()
    rows = select(candidate_pairs(items))
    write_csv(rows, "eval/pairs_to_label.csv")
    counts = Counter(r["band"] for r in rows)
    print(f"{len(items)} headlines -> {len(rows)} pairs written to eval/pairs_to_label.csv")
    for band in BAND_SIZES:
        print(f"  {band}: {counts[band]} of {BAND_SIZES[band]} wanted")


if __name__ == "__main__":
    main()
