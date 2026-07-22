"""Tests for the HTTP fetcher's retry and header behaviour.

No network: `httpx.get` is replaced with a stub that returns a scripted sequence of
responses, and the backoff sleep is stubbed out so the suite stays instant. What we are
checking is our own policy — which statuses we retry, which we give up on, and what
headers we send — not that httpx works.
"""

import httpx
import pytest

from pipeline import fetch as fetch_module
from pipeline.fetch import FetchError, fetch


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


@pytest.fixture
def calls(monkeypatch):
    """Script a sequence of responses; return the list of requests actually made."""
    made: list[dict] = []

    def install(responses: list):
        queue = list(responses)

        def fake_get(url, **kwargs):
            made.append({"url": url, **kwargs})
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(httpx, "get", fake_get)

    slept: list[float] = []
    monkeypatch.setattr(fetch_module.time, "sleep", slept.append)
    install.made = made
    install.slept = slept
    return install


# --- Retry policy ---------------------------------------------------------------


def test_403_is_retried_and_can_succeed(calls):
    """Imedi's bot filter 403s intermittently; the retry is the whole point of the fix."""
    calls([FakeResponse(403), FakeResponse(200, b"<html>ok</html>")])
    assert fetch("https://imedinews.ge/ge/archive") == b"<html>ok</html>"
    assert len(calls.made) == 2


def test_403_that_never_clears_fails_with_a_readable_error(calls):
    calls([FakeResponse(403), FakeResponse(403), FakeResponse(403)])
    with pytest.raises(FetchError) as excinfo:
        fetch("https://imedinews.ge/ge/archive")
    assert "403" in str(excinfo.value)
    assert len(calls.made) == 3


def test_retries_are_spaced_minutes_apart(calls):
    """The gap is the fix: knocking three times in seconds re-earns Imedi's block."""
    calls([FakeResponse(403), FakeResponse(403), FakeResponse(403)])
    with pytest.raises(FetchError):
        fetch("https://imedinews.ge/ge/archive")
    assert calls.slept == [60.0, 120.0]


def test_first_attempt_does_not_wait(calls):
    """The happy path — every source, every run — must stay instant."""
    calls([FakeResponse(200, b"ok")])
    fetch("https://example.ge/feed")
    assert calls.slept == []


def test_429_is_retried(calls):
    calls([FakeResponse(429), FakeResponse(200, b"ok")])
    assert fetch("https://example.ge/feed") == b"ok"


def test_500_is_retried(calls):
    calls([FakeResponse(503), FakeResponse(200, b"ok")])
    assert fetch("https://example.ge/feed") == b"ok"


def test_404_fails_immediately_without_retrying(calls):
    """A moved feed is not a temporary condition — retrying only wastes the run."""
    calls([FakeResponse(404)])
    with pytest.raises(FetchError):
        fetch("https://example.ge/gone")
    assert len(calls.made) == 1


def test_network_error_is_retried(calls):
    calls([httpx.ConnectTimeout("timed out"), FakeResponse(200, b"ok")])
    assert fetch("https://example.ge/feed") == b"ok"


# --- Headers --------------------------------------------------------------------


def test_sends_browser_like_headers(calls):
    """A lone User-Agent is what bot filters flag; send the full set a browser sends."""
    calls([FakeResponse(200, b"ok")])
    fetch("https://imedinews.ge/ge/archive")
    headers = calls.made[0]["headers"]
    assert "Accept" in headers
    assert "Accept-Language" in headers


def test_user_agent_stays_honest(calls):
    """We identify ourselves with a contact URL rather than impersonating a browser."""
    calls([FakeResponse(200, b"ok")])
    fetch("https://example.ge/feed")
    assert "GroundNewsGeorgia" in calls.made[0]["headers"]["User-Agent"]


def test_does_not_advertise_an_encoding_we_cannot_decode(calls):
    """httpx sets Accept-Encoding from its real decoders; overriding it breaks brotli."""
    calls([FakeResponse(200, b"ok")])
    fetch("https://example.ge/feed")
    assert "Accept-Encoding" not in calls.made[0]["headers"]
