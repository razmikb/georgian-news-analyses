"""Environment configuration.

Secrets live only in a git-ignored `.env` locally, and in GitHub Actions secrets in CI.
Nothing here ever gets committed with a real value in it.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Identify ourselves honestly to the sites we fetch from, with a contact URL.
USER_AGENT = "GroundNewsGeorgia/0.1 (+https://github.com/razmikb/georgian-news-analyses)"

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
