"""Supabase access for the pipeline.

Uses the service-role key, which bypasses row-level security — the pipeline is the
only writer. The frontend uses the anon key and can only read.
"""

from functools import lru_cache

from supabase import Client, create_client

from pipeline.config import supabase_service_role_key, supabase_url
from pipeline.rss import Article


@lru_cache(maxsize=1)
def client() -> Client:
    return create_client(supabase_url(), supabase_service_role_key())


def get_sources(slug: str | None = None) -> list[dict]:
    """Return source rows, optionally just one by slug.

    Sources are config living in the database (CLAUDE.md), so the pipeline always
    reads them rather than hardcoding a list.
    """
    query = client().table("sources").select("id, slug, name_en, feed_url, leaning")
    if slug:
        query = query.eq("slug", slug)
    return query.order("slug").execute().data


def count_articles(source_id: int) -> int:
    """How many articles we already hold for a source (used to measure what a run added)."""
    result = (
        client()
        .table("articles")
        .select("id", count="exact")
        .eq("source_id", source_id)
        .limit(1)
        .execute()
    )
    return result.count or 0


def insert_articles(articles: list[Article]) -> None:
    """Insert headlines, silently skipping any URL we already have.

    Dedupe is enforced by the `articles.url` unique constraint in the database
    (migration 0001), not by application logic — so an hourly re-run that sees the
    same 32 headlines adds nothing.
    """
    if not articles:
        return
    # A feed can list the same link twice; collapse those first, because Postgres
    # rejects a batch that conflicts with itself on the same key.
    unique: dict[str, Article] = {}
    for article in articles:
        unique.setdefault(article.url, article)
    rows = [a.to_row() for a in unique.values()]
    client().table("articles").upsert(rows, on_conflict="url", ignore_duplicates=True).execute()
