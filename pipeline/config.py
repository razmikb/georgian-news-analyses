"""Environment configuration.

Secrets live only in a git-ignored `.env` locally, and in GitHub Actions secrets in CI.
Nothing here ever gets committed with a real value in it.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Identify ourselves honestly to the sites we fetch from, with a contact URL.
USER_AGENT = "GroundNewsGeorgia/0.1 (+https://github.com/razmikb/georgian-news-analyses)"

# Sent with every request. The User-Agent stays honest — we do not disguise ourselves as
# a browser — but we do send the *other* headers a real browser sends. Bot-protection
# filters (Imedi sits behind DDoS-Guard) score a request on how complete its header set
# looks, and a request carrying nothing but a User-Agent is the tell that trips them.
# Accept-Encoding is deliberately absent: httpx sets it from the decoders it actually has,
# and advertising one we cannot decode (brotli) would break the response.
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ka-GE,ka;q=0.9,en-US;q=0.8,en;q=0.7",
    "Upgrade-Insecure-Requests": "1",
}

# How long to wait for a feed before giving up, and how many times to retry.
FETCH_TIMEOUT_SECONDS = 20.0
FETCH_RETRIES = 3

# How long to wait before each retry. Deliberately generous: the thing we retry most is
# Imedi's bot filter, which scores us on how fast we knock. Three tries inside a few
# seconds reads as a bot and re-earns the block; spacing them over three minutes reads as
# a browser someone reloaded. Nothing waits on an ingest run — the cron is every 2 hours —
# so patience here costs only Actions minutes, and the repo is public, where they are free.
FETCH_BACKOFF_SECONDS = (60.0, 120.0)

# ─────────────────────────────────────────────────────────────────────────────
# Gemini embeddings
# ─────────────────────────────────────────────────────────────────────────────
# `text-embedding-004` (named in migration 0001's comment) was deprecated 2026-01-14.
# `gemini-embedding-001` replaces it.
EMBED_MODEL = "gemini-embedding-001"

# The model's native size is 3072, but we ask for 768. That is a hard requirement, not
# a leftover: pgvector's HNSW index refuses anything past 2000 dimensions, so a 3072-wide
# vector could be stored but never searched — which is the only thing we want it for.
# 768 is one of the model's recommended truncations and keeps `articles.embedding
# vector(768)` unchanged.
EMBED_DIMENSIONS = 768

# What we are actually asking the model: "are these two texts about the same thing?"
EMBED_TASK_TYPE = "SEMANTIC_SIMILARITY"

# How much to put in one API call. Two ceilings, and the *character* one is the one that
# matters — measured 2026-07-26, when batching by count alone got the measurement run
# refused. 100 headlines is ~12,000 characters; 100 article bodies is ~174,000. Counting
# texts, those are the same "batch of 100"; in payload they differ fourteen-fold, and the
# big one comes back 429. So a request is capped by size first and count second.
EMBED_BATCH_CHARS = 10_000
EMBED_BATCH_SIZE = 100

# A self-imposed throughput ceiling, because Gemini's free-tier per-minute allowance is
# real but no longer published (the docs now say "see AI Studio"). This sits well under
# where we actually saw a refusal: ~62,000 characters in two back-to-back requests.
# Characters stand in for tokens — the ratio for Georgian is unknown, but a ceiling only
# has to be safely low, not accurate. Raise it if runs feel slow and stay green.
EMBED_CHARS_PER_MINUTE = 25_000

# Backoff for Gemini quota errors (429). Longer than the HTTP schedule because a free-tier
# per-minute quota clears by waiting out the minute, and there is never anything urgent
# about an embedding — the run can always finish next time.
EMBED_BACKOFF_SECONDS = (30.0, 90.0, 180.0)
EMBED_RETRIES = 4

# Quota refusals have two different causes needing opposite responses: waiting fixes a
# per-minute allowance, only sending less fixes a request too big to ever be allowed.
# Google's 429 names neither, so we test the cheaper explanation first — one wait — and
# if the identical request is refused twice, waiting is not the answer and we halve it.
EMBED_SPLIT_AFTER_FAILURES = 2


class MissingConfigError(RuntimeError):
    """Raised when a required environment variable is absent."""


def require(name: str, where: str) -> str:
    """Return env var `name`, or fail with a message that says where to get it.

    `where` matters more than it looks: the person running this is not a developer, so
    "not set" alone is a dead end. The message has to name the page to click.
    """
    value = os.environ.get(name)
    if not value:
        raise MissingConfigError(
            f"{name} is not set. Copy .env.example to .env and fill in the value ({where})."
        )
    return value


def supabase_url() -> str:
    return require("SUPABASE_URL", "Supabase → Project Settings → API")


def supabase_service_role_key() -> str:
    """The write key. Pipeline only — never expose this to the frontend."""
    return require("SUPABASE_SERVICE_ROLE_KEY", "Supabase → Project Settings → API")


def gemini_api_key() -> str:
    """Key for embeddings and event titles.

    Create it in a *new* Google Cloud project rather than an existing one: Gemini's
    free-tier rate limits are per project, so sharing a project with another app means
    sharing one quota with it.
    """
    return require(
        "GEMINI_API_KEY",
        "https://aistudio.google.com/apikey → Create API key → Create in new project",
    )
