"""Turn text into vectors with the Gemini embedding API.

A vector is how the pipeline measures "are these two articles about the same event":
embed both, take the cosine of the angle between them, compare to a threshold.

Two things here are easy to get wrong and silently poison that number:

1. **Normalization.** `gemini-embedding-001` returns unit-length vectors only at its
   native 3072 dimensions. We ask for 768 (see `config.EMBED_DIMENSIONS` for why that
   is forced), and truncated vectors come back *not* normalized. Cosine similarity on
   un-normalized vectors is not cosine similarity. So we normalize here, once, rather
   than hoping every caller remembers.
2. **Order.** Batching sends 100 texts per request; the results must come back in the
   order they went in, or every pair is scored against the wrong partner.
"""

from __future__ import annotations

import math
import time

from google import genai
from google.genai import types

from pipeline import cache
from pipeline.config import (
    EMBED_BACKOFF_SECONDS,
    EMBED_BATCH_SIZE,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    EMBED_RETRIES,
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


def _embed_batch(texts: list[str]) -> list[Vector]:
    """One API call, with backoff on quota errors."""
    last_error: Exception | None = None

    for attempt in range(EMBED_RETRIES):
        if attempt:
            time.sleep(EMBED_BACKOFF_SECONDS[min(attempt, len(EMBED_BACKOFF_SECONDS)) - 1])
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
            if not _is_quota_error(exc):
                raise EmbedError(f"Gemini embedding failed: {exc}") from exc
            last_error = exc
            continue

        vectors = [normalize(list(item.values)) for item in response.embeddings]
        if len(vectors) != len(texts):
            raise EmbedError(f"Asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

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

    unique = list(pending)
    for start in range(0, len(unique), EMBED_BATCH_SIZE):
        chunk = unique[start : start + EMBED_BATCH_SIZE]
        for text, vector in zip(chunk, _embed_batch(chunk), strict=True):
            if use_cache:
                cache.put(CACHE_NAMESPACE, _cache_key(text), vector)
            for index in pending[text]:
                results[index] = vector

    missing = [i for i, vector in enumerate(results) if vector is None]
    if missing:
        raise EmbedError(f"No embedding produced for {len(missing)} text(s)")
    return [vector for vector in results if vector is not None]
