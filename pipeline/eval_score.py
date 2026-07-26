"""Score the labelled eval pairs and read the clustering threshold off the results.

    python -m pipeline.eval_score --dry-run   # fetch article text only, spend no quota
    python -m pipeline.eval_score             # the real thing
    python -m pipeline.eval_score --clear-cache

Phase 2 rests on one number: how similar two articles must be before we call them the
same event. `eval/labeled_pairs.csv` holds a human's verdict on 60 real pairs. This
module embeds those articles, measures how the machine scores each pair, and compares
the two — so the threshold is *measured*, not guessed (PLAN.md §6).

It answers two questions at once:

1. **How much article text should we embed?** Headline alone, headline + lead, or
   headline + body. PLAN.md §6 records this as a tuned dial rather than an assumption,
   because full bodies can make clustering worse — shared background paragraphs drag
   unrelated stories together.
2. **Where does the threshold go?** Not one number but two: above the upper bound we
   are sure it's the same event, below the lower bound sure it isn't, and the gap
   between them is where a Gemini Flash verifier has to make the call.

Nothing here writes to Supabase. It is a measurement, not part of the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

from pipeline import cache, embed, extract

LABELS_PATH = "eval/labeled_pairs.csv"
RESULTS_PATH = "eval/results.md"

# The three candidate answers to "how much text?", in increasing order of greed.
VARIANTS = ("headline", "headline+lead", "headline+body")


@dataclass(frozen=True)
class Scored:
    """One eval pair, with the human's verdict and the machine's score."""

    pair_id: str
    label: str
    score: float
    headline_a: str
    headline_b: str


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — pure functions, so the maths is testable without network or API keys
# ─────────────────────────────────────────────────────────────────────────────


def _split(scored: list[Scored]) -> tuple[list[float], list[float]]:
    """Scores of the `same` pairs and of the `different` pairs. `unsure` is excluded —
    it has no right answer to be measured against (it is reported separately instead)."""
    same = [s.score for s in scored if s.label == "same"]
    different = [s.score for s in scored if s.label == "different"]
    return same, different


def auc(scored: list[Scored]) -> float:
    """Probability that a random same-pair outscores a random different-pair.

    The cleanest way to compare the three text variants, because it needs no threshold:
    it asks only whether the ordering is right. 1.0 is perfect, 0.5 is a coin flip.
    """
    same, different = _split(scored)
    if not same or not different:
        return float("nan")
    wins = sum((s > d) + 0.5 * (s == d) for s in same for d in different)
    return wins / (len(same) * len(different))


def score_at(scored: list[Scored], threshold: float) -> tuple[float, float, float]:
    """Precision, recall and F1 for calling everything at or above `threshold` the same event."""
    same, different = _split(scored)
    true_positives = sum(s >= threshold for s in same)
    false_positives = sum(d >= threshold for d in different)
    precision = true_positives / (true_positives + false_positives) if true_positives else 0.0
    recall = true_positives / len(same) if same else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def best_threshold(scored: list[Scored]) -> tuple[float, float, float, float]:
    """The threshold with the best F1, and its precision/recall/F1.

    Candidates are the observed scores themselves rather than a fixed grid — the F1 curve
    only changes where a real pair sits, so a grid would either miss the optimum or waste
    steps between them.
    """
    candidates = sorted({s.score for s in scored})
    if not candidates:
        return (float("nan"),) * 4
    best = max(candidates, key=lambda t: score_at(scored, t)[2])
    return (best, *score_at(scored, best))


def ambiguous_band(scored: list[Scored]) -> tuple[float, float, bool]:
    """The zone where the two classes overlap — where a verifier is needed.

    Returns (low, high, separated). Above `high` no `different` pair reaches, so a hit
    there is confidently the same event; below `low` no `same` pair falls, so it is
    confidently different.

    When `separated` is True the classes never overlap: `low` is the top of the different
    pairs and `high` the bottom of the same pairs, and *any* threshold in between is
    perfect on this set. That is a much stronger result than a good F1 score.
    """
    same, different = _split(scored)
    if not same or not different:
        return float("nan"), float("nan"), False
    highest_different = max(different)
    lowest_same = min(same)
    if lowest_same > highest_different:
        return highest_different, lowest_same, True
    return lowest_same, highest_different, False


def share_in_band(scored: list[Scored], low: float, high: float) -> float:
    """Fraction of pairs landing inside the band — i.e. how much work the verifier gets."""
    if not scored:
        return 0.0
    return sum(low <= s.score <= high for s in scored) / len(scored)


def worst_misses(scored: list[Scored], count: int = 3) -> tuple[list[Scored], list[Scored]]:
    """The pairs the scoring finds hardest: different-but-scored-high, same-but-scored-low.

    These are what to actually read. An aggregate says how well it did; these say why.
    """
    same = sorted((s for s in scored if s.label == "same"), key=lambda s: s.score)
    different = sorted(
        (s for s in scored if s.label == "different"), key=lambda s: s.score, reverse=True
    )
    return different[:count], same[:count]


# ─────────────────────────────────────────────────────────────────────────────
# Gathering the data
# ─────────────────────────────────────────────────────────────────────────────


def load_pairs(path: str = LABELS_PATH) -> list[dict]:
    """Read the answer key. UTF-8-with-BOM because that is how Excel saves it."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def article_texts(urls: list[str], *, use_cache: bool = True) -> dict[str, str | None]:
    """Fetch every article once, in order, reporting progress.

    Serial rather than parallel on purpose: Imedi's bot filter scores request *rate*, and
    a burst of 120 concurrent fetches is exactly the shape that trips it (PROGRESS.md).
    """
    texts: dict[str, str | None] = {}
    for index, url in enumerate(urls, start=1):
        text = extract.article_text(url, use_cache=use_cache)
        texts[url] = text
        status = f"{len(text)} chars" if text else "NO TEXT"
        print(f"  [{index}/{len(urls)}] {status}  {url[:80]}")
    return texts


def variant_text(variant: str, headline: str, text: str | None) -> str | None:
    """Build the string to embed, or None if this variant is impossible for this article.

    Cleaning happens here rather than at fetch time because it needs the headline (page
    furniture includes the headline repeated back). What is left after stripping the
    furniture can be too thin to be an article at all — on Imedi, a video stub is mostly
    sidebar — and an article we cannot read has no lead and no body.
    """
    if variant == "headline":
        return headline
    if text is None:
        return None
    cleaned = extract.clean(text, headline)
    if len(cleaned) < extract.MIN_TEXT_CHARS:
        return None
    extra = extract.lead(cleaned) if variant == "headline+lead" else extract.body(cleaned)
    return f"{headline}\n{extra}" if extra else None


def score_variant(
    pairs: list[dict],
    texts: dict[str, str | None],
    variant: str,
    *,
    use_cache: bool = True,
) -> list[Scored]:
    """Embed both sides of every usable pair and score it."""
    usable, to_embed = [], []
    for pair in pairs:
        side_a = variant_text(variant, pair["headline_a"], texts.get(pair["url_a"]))
        side_b = variant_text(variant, pair["headline_b"], texts.get(pair["url_b"]))
        if side_a is None or side_b is None:
            continue
        usable.append(pair)
        to_embed += [side_a, side_b]

    vectors = embed.embed_texts(to_embed, use_cache=use_cache)
    return [
        Scored(
            pair_id=pair["pair_id"],
            label=(pair["label"] or "").strip().lower(),
            score=embed.cosine(vectors[i * 2], vectors[i * 2 + 1]),
            headline_a=pair["headline_a"],
            headline_b=pair["headline_b"],
        )
        for i, pair in enumerate(usable)
    ]


def common_subset(results: dict[str, list[Scored]]) -> set[str]:
    """Pair ids scored by every variant.

    Variants must be compared on identical pairs. An article whose page would not load has
    no lead and no body, so it silently drops out of two of the three — and a variant judged
    on an easier subset would look better for no reason but that.
    """
    id_sets = [{s.pair_id for s in scored} for scored in results.values()]
    return set.intersection(*id_sets) if id_sets else set()


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def variant_report(name: str, scored: list[Scored]) -> list[str]:
    """One variant's numbers, as markdown lines."""
    same, different = _split(scored)
    unsure = [s for s in scored if s.label == "unsure"]
    threshold, precision, recall, f1 = best_threshold(scored)
    low, high, separated = ambiguous_band(scored)

    lines = [
        f"### {name}",
        "",
        f"- Pairs scored: **{len(scored)}** ({len(same)} same, {len(different)} different, "
        f"{len(unsure)} unsure)",
        f"- **Ranking quality (AUC): {auc(scored):.3f}** — the chance it scores a true pair "
        "above a false one. 1.000 is perfect, 0.500 is a coin flip.",
    ]

    if separated:
        lines += [
            f"- **Cleanly separated.** Every `same` pair scored above every `different` one. "
            f"Any threshold between **{low:.3f}** and **{high:.3f}** is perfect on this set; "
            f"the midpoint **{(low + high) / 2:.3f}** leaves the most room for error.",
        ]
    else:
        lines += [
            f"- **Overlap zone: {low:.3f} – {high:.3f}** — inside it, human verdicts go both "
            f"ways, so the pipeline cannot decide alone. "
            f"{_pct(share_in_band(scored, low, high))} of pairs land here and would go to the "
            "Gemini Flash verifier.",
            f"- Best single threshold **{threshold:.3f}** — precision {_pct(precision)}, "
            f"recall {_pct(recall)}, F1 {f1:.3f}.",
        ]

    if unsure:
        marks = ", ".join(f"#{s.pair_id} at {s.score:.3f}" for s in unsure)
        inside = all(low <= s.score <= high for s in unsure) if not separated else False
        verdict = (
            "inside the overlap zone — the maths finds these hard too, which is what we hoped"
            if inside
            else "**outside the overlap zone** — worth a look: the maths is confident "
            "where a human wasn't"
        )
        lines.append(f"- The human's `unsure` pairs scored {marks}: {verdict}.")

    false_alarms, misses = worst_misses(scored)
    if false_alarms:
        lines += ["", "Closest calls it got wrong or nearly did:", ""]
        for s in false_alarms:
            lines.append(
                f"- `different` #{s.pair_id} scored **{s.score:.3f}** — “{s.headline_a[:60]}” / "
                f"“{s.headline_b[:60]}”"
            )
        for s in misses:
            lines.append(
                f"- `same` #{s.pair_id} scored only **{s.score:.3f}** — “{s.headline_a[:60]}” / "
                f"“{s.headline_b[:60]}”"
            )
    return lines + [""]


def build_report(
    results: dict[str, list[Scored]],
    texts: dict[str, str | None],
    total_pairs: int,
) -> str:
    """Assemble results.md — written for a non-developer to read and decide from."""
    shared = common_subset(results)
    got_text = sum(1 for t in texts.values() if t)

    lines = [
        "# Clustering threshold — measured results",
        "",
        "Produced by `python -m pipeline.eval_score` from `eval/labeled_pairs.csv`.",
        "This is the number Phase 2 rests on: how similar two articles must be before the",
        "pipeline calls them the same event.",
        "",
        "## What was measured",
        "",
        f"- {total_pairs} labelled pairs; article text retrieved for "
        f"**{got_text} of {len(texts)}** articles.",
        f"- All three variants compared on the **{len(shared)} pairs every variant could "
        "score** (an article whose page would not load has no lead and no body, so it drops "
        "out of two of the three — comparing on different subsets would flatter whichever "
        "got the easier one).",
        "",
        "## The three variants, head to head",
        "",
    ]

    for name in VARIANTS:
        scored = [s for s in results.get(name, []) if s.pair_id in shared]
        lines += variant_report(name, scored)

    lines += [
        "## Read this before trusting the numbers",
        "",
        "- **Only 15 pairs are `same`.** Every percentage here rests on those 15, so treat the",
        "  thresholds as roughly right, not precise. The fix is a second labelling round with",
        "  candidates chosen *by these embeddings* — far better at surfacing real matches than",
        "  the deliberately dumb word-overlap that picked round 1.",
        "- **This set is not a natural sample of Georgian news.** It was built dense with hard",
        "  cases on purpose, so a good score here does not translate to the same score in",
        "  production. It is a tuning instrument, not a report card.",
        "- **Pairs are not clusters.** A threshold that judges two articles well can still chain",
        "  in production: A matches B, B matches C, and A and C end up in one event without ever",
        "  having matched. That is a real risk to handle when clustering is wired in, not here.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Measure the clustering threshold.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch article text and report coverage, but embed nothing (spends no quota)",
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore cached text and vectors")
    parser.add_argument(
        "--clear-cache", action="store_true", help="delete the local cache and exit"
    )
    args = parser.parse_args(argv)

    if args.clear_cache:
        print(f"Cleared {cache.clear()} cached entries.")
        return 0

    use_cache = not args.no_cache
    pairs = load_pairs()
    labelled = [p for p in pairs if (p["label"] or "").strip()]
    print(f"{len(labelled)} labelled pairs of {len(pairs)}.")

    urls = list(dict.fromkeys([p["url_a"] for p in labelled] + [p["url_b"] for p in labelled]))
    print(f"\nFetching {len(urls)} articles (cached ones are instant)...")
    texts = article_texts(urls, use_cache=use_cache)
    got_text = sum(1 for t in texts.values() if t)
    print(f"\nGot text for {got_text} of {len(urls)} articles.")

    if args.dry_run:
        print("\nDRY RUN — no embeddings requested, no quota spent.")
        for url, text in texts.items():
            if not text:
                print(f"  no text: {url}")
        return 0

    results: dict[str, list[Scored]] = {}
    for name in VARIANTS:
        print(f"\nEmbedding variant: {name}")
        results[name] = score_variant(labelled, texts, name, use_cache=use_cache)
        print(f"  scored {len(results[name])} pairs")

    report = build_report(results, texts, len(labelled))
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"\n{report}\n\nWritten to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
