"""Pipeline entry point.

    python -m pipeline.run                      # every RSS source
    python -m pipeline.run --source netgazeti   # just one
    python -m pipeline.run --dry-run            # fetch + parse, print, write nothing

Failure is ALWAYS per-source (CLAUDE.md): one broken feed must never stop the others.
Every run ends with a plain-language summary the user can paste into a chat.
"""

import argparse
import sys
from dataclasses import dataclass

from pipeline import db
from pipeline.fetch import fetch
from pipeline.rss import parse_feed


def _force_utf8_output() -> None:
    """Print Georgian text safely on any console.

    Windows terminals still default to a legacy codepage that cannot represent
    Georgian script, which turns any headline into a UnicodeEncodeError. Linux/CI
    is already UTF-8, so this is a no-op there.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


@dataclass
class SourceResult:
    name: str
    found: int = 0
    added: int = 0
    error: str | None = None

    def line(self) -> str:
        if self.error:
            return f"  {self.name}: FAILED — {self.error}"
        if self.found == 0:
            return f"  {self.name}: 0 headlines found — feed is probably broken, needs a look"
        already = self.found - self.added
        return (
            f"  {self.name}: {self.found} headlines found, {self.added} new, {already} already had"
        )


def ingest_source(source: dict, *, dry_run: bool) -> SourceResult:
    """Fetch, parse and store one source. Raises nothing — errors land in the result."""
    result = SourceResult(name=source["name_en"])
    try:
        articles = parse_feed(fetch(source["feed_url"]), source["id"])
        result.found = len(articles)

        if dry_run:
            for article in articles:
                published = article.published_at.isoformat() if article.published_at else "no date"
                print(f"    [{published}] {article.headline}\n      {article.url}")
            return result

        before = db.count_articles(source["id"])
        db.insert_articles(articles)
        result.added = db.count_articles(source["id"]) - before
    except Exception as exc:  # noqa: BLE001 — per-source isolation is the whole point
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(description="Ingest Georgian news headlines.")
    parser.add_argument("--source", help="slug of a single source, e.g. netgazeti")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and print headlines without writing to the database",
    )
    args = parser.parse_args(argv)

    sources = db.get_sources(args.source)
    if args.source and not sources:
        print(f"No source with slug '{args.source}'.", file=sys.stderr)
        return 1

    # Sources without a feed_url need an HTML scraper, which does not exist yet.
    feed_sources = [s for s in sources if s.get("feed_url")]
    skipped = [s["name_en"] for s in sources if not s.get("feed_url")]
    if not feed_sources:
        print("No sources with an RSS feed to ingest.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("DRY RUN — nothing will be written to the database.\n")

    results = [ingest_source(s, dry_run=args.dry_run) for s in feed_sources]

    print("\nRun summary")
    for result in results:
        print(result.line())
    if skipped:
        print(f"  Skipped (no RSS feed, needs a scraper): {', '.join(skipped)}")

    failed = [r for r in results if r.error]
    empty = [r for r in results if not r.error and r.found == 0]
    print(f"\n{len(results) - len(failed)} source(s) OK, {len(failed)} failed.")

    # Non-zero exit makes GitHub Actions go red and email us.
    return 1 if failed or empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
