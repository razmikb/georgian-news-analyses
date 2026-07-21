"""HTTP fetching with retries.

Kept separate from parsing so that parsing can be tested offline against a saved
feed snapshot — no network in the test suite.
"""

import time

import httpx

from pipeline.config import DEFAULT_HEADERS, FETCH_RETRIES, FETCH_TIMEOUT_SECONDS

# Statuses worth trying again. 429 and 5xx are the obvious ones. 403 is here because of
# bot-protection WAFs: Imedi's DDoS-Guard answers an occasional request with a 403
# challenge that clears on its own moments later, so treating it as a permanent refusal
# threw away a whole source's run roughly once in four (PROGRESS.md).
RETRYABLE_STATUSES = frozenset({403, 429})


class FetchError(RuntimeError):
    """A URL could not be retrieved after all retries."""


def fetch(url: str, *, retries: int = FETCH_RETRIES) -> bytes:
    """GET `url` and return the raw body.

    Retries with exponential backoff on network errors and retryable statuses.
    A 404 or other 4xx fails immediately — retrying won't fix it.
    """
    last_error: Exception | None = None

    for attempt in range(retries):
        if attempt:
            time.sleep(2**attempt)  # 2s, 4s
        try:
            response = httpx.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=FETCH_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            continue

        if response.status_code in RETRYABLE_STATUSES or response.status_code >= 500:
            last_error = FetchError(f"HTTP {response.status_code} from {url}")
            continue
        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} from {url}")

        return response.content

    raise FetchError(f"Could not fetch {url} after {retries} attempts: {last_error}")
