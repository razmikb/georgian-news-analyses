"""Turn text into vectors with the Gemini embedding API.

A vector is how the pipeline measures "are these two articles about the same event":
embed both, take the cosine of the angle between them, compare to a threshold.

Two things here are easy to get wrong and silently poison that number:

1. **Normalization.** `gemini-embedding-001` returns unit-length vectors only at its
   native 3072 dimensions. We ask for 768 (see `config.EMBED_DIMENSIONS` for why that
   is forced), and truncated vectors come back *not* normalized. Cosine similarity on
   un-normalized vectors is not cosine similarity. So we normalize here, once, rather
   than hoping every caller remembers.
2. **Order.** Requests are batched; the results must come back in the order they went
   in, or every pair is scored against the wrong partner.

3. **Size.** A batch is budgeted by characters, not by number of texts. Headlines and
   full article bodies differ fourteen-fold in payload for the same "batch of 100", and
   the big one gets refused (see `config.EMBED_BATCH_CHARS`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator

from google import genai
from google.genai import types

from pipeline import cache
from pipeline.config import (
    EMBED_BACKOFF_SECONDS,
    EMBED_BATCH_CHARS,
    EMBED_BATCH_SIZE,
    EMBED_CHARS_PER_MINUTE,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    EMBED_RETRIES,
    EMBED_SPLIT_AFTER_FAILURES,
    EMBED_TASK_TYPE,
    gemini_api_key,
)

CACHE_NAMESPACE = "embeddings"

Vector = list[float]


class EmbedError(RuntimeError):
    """Embedding failed after all retries."""


_client: genai.Client | None = None


def client() -> genai.Client:
    """One client for the process, created on first use so importing needs no API key."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=gemini_api_key())
    return _client


def normalize(vector: Vector) -> Vector:
    """Scale a vector to length 1, so a dot product *is* the cosine similarity."""
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector
    return [value / length for value in vector]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Assumes both vectors came from `embed_texts` (already unit length)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _is_quota_error(exc: Exception) -> bool:
    """True for 'you are going too fast', false for 'your key is wrong'.

    Worth distinguishing: the first clears by waiting, the second never does, and
    retrying a bad key for ten minutes tells the user nothing useful.
    """
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "QUOTA" in text


def _cache_key(text: str) -> str:
    """Model and dimensions are part of the key — vectors from different models
    are not comparable, so a cached one must never be reused across a model change."""
    return f"{EMBED_MODEL}|{EMBED_DIMENSIONS}|{EMBED_TASK_TYPE}|{text}"


class _QuotaRefused(RuntimeError):
    """Internal: the API said 429. Recoverable by waiting, or by sending less."""


# ─────────────────────────────────────────────────────────────────────────────
# Pacing — staying under a per-minute allowance we cannot see
# ─────────────────────────────────────────────────────────────────────────────
# Tracked as "the earliest time the next request may go out". Charging the gap *after*
# a request rather than sleeping before one means the first request is never delayed and
# the last one is never followed by a pointless wait.
_next_request_at = 0.0


def _await_pace() -> None:
    """Sleep off whatever remains of the gap the previous request earned."""
    remaining = _next_request_at - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _charge_pace(chars: int) -> None:
    """Record how long to hold off, given how much text we just sent."""
    global _next_request_at
    _next_request_at = time.monotonic() + 60.0 * chars / EMBED_CHARS_PER_MINUTE


def reset_pace() -> None:
    """Forget the pacing debt. For tests, and for a fresh process's first call."""
    global _next_request_at
    _next_request_at = 0.0


# ─────────────────────────────────────────────────────────────────────────────


def _chunks(texts: list[str]) -> Iterator[list[str]]:
    """Split texts into requests small enough to be accepted.

    Budgeted by characters first, count second — see `config.EMBED_BATCH_CHARS` for why
    counting texts alone is what got a run refused. A single text over the budget still
    goes out on its own: splitting one article's text would change what we are measuring.
    """
    chunk: list[str] = []
    chunk_chars = 0
    for text in texts:
        too_many = len(chunk) >= EMBED_BATCH_SIZE
        too_big = chunk_chars + len(text) > EMBED_BATCH_CHARS
        if chunk and (too_many or too_big):
            yield chunk
            chunk, chunk_chars = [], 0
        chunk.append(text)
        chunk_chars += len(text)
    if chunk:
        yield chunk


def _embed_once(texts: list[str]) -> list[Vector]:
    """Exactly one API call. No retries, no waiting — the caller decides what a failure means."""
    _await_pace()
    try:
        response = client().models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=EMBED_TASK_TYPE,
                output_dimensionality=EMBED_DIMENSIONS,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — the SDK raises several unrelated types
        _charge_pace(sum(len(t) for t in texts))
        if not _is_quota_error(exc):
            raise EmbedError(f"Gemini embedding failed: {exc}") from exc
        raise _QuotaRefused(str(exc)) from exc

    _charge_pace(sum(len(t) for t in texts))
    vectors = [normalize(list(item.values)) for item in response.embeddings]
    if len(vectors) != len(texts):
        raise EmbedError(f"Asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def _embed_batch(texts: list[str]) -> list[Vector]:
    """Get vectors for one chunk, working around whichever quota refused it.

    Waiting and shrinking are both tried, in that order, because a 429 does not say which
    one it needs (`config.EMBED_SPLIT_AFTER_FAILURES`). Halving continues until either the
    request is accepted or a chunk is down to a single text — at which point there is
    nothing left to shrink and patience is all we have.
    """
    last_error: Exception | None = None

    for attempt in range(EMBED_RETRIES):
        if attempt:
            time.sleep(EMBED_BACKOFF_SECONDS[min(attempt, len(EMBED_BACKOFF_SECONDS)) - 1])
        try:
            return _embed_once(texts)
        except _QuotaRefused as exc:
            last_error = exc
            if len(texts) > 1 and attempt + 1 >= EMBED_SPLIT_AFTER_FAILURES:
                middle = len(texts) // 2
                return _embed_batch(texts[:middle]) + _embed_batch(texts[middle:])

    raise EmbedError(f"Gemini quota not clearing after {EMBED_RETRIES} attempts: {last_error}")


def embed_texts(texts: list[str], *, use_cache: bool = True) -> list[Vector]:
    """Embed every text, returning vectors in the same order.

    Anything already cached costs nothing, so re-running an experiment is free and
    offline. Only the genuinely new texts are sent, batched.
    """
    if not texts:
        return []

    results: list[Vector | None] = [None] * len(texts)
    pending: dict[str, list[int]] = {}

    for index, text in enumerate(texts):
        if use_cache:
            cached = cache.get(CACHE_NAMESPACE, _cache_key(text))
            if cached is not None:
                results[index] = cached
                continue
        # Duplicate texts share one API call and are filled into every position they hold.
        pending.setdefault(text, []).append(index)

    for chunk in _chunks(list(pending)):
        for text, vector in zip(chunk, _embed_batch(chunk), strict=True):
            if use_cache:
                cache.put(CACHE_NAMESPACE, _cache_key(text), vector)
            for index in pending[text]:
                results[index] = vector

    missing = [i for i, vector in enumerate(results) if vector is None]
    if missing:
        raise EmbedError(f"No embedding produced for {len(missing)} text(s)")
    return [vector for vector in results if vector is not None]
