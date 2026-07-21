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


class MissingConfigError(RuntimeError):
    """Raised when a required environment variable is absent."""


def require(name: str) -> str:
    """Return env var `name`, or fail with a message that says how to fix it."""
    value = os.environ.get(name)
    if not value:
        raise MissingConfigError(
            f"{name} is not set. Copy .env.example to .env and fill in the value "
            f"(Supabase → Project Settings → API)."
        )
    return value


def supabase_url() -> str:
    return require("SUPABASE_URL")


def supabase_service_role_key() -> str:
    """The write key. Pipeline only — never expose this to the frontend."""
    return require("SUPABASE_SERVICE_ROLE_KEY")
