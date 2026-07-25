"""A tiny on-disk cache for things that are slow or rate-limited to fetch twice.

Used by the eval experiment, not by the production pipeline. Two callers: fetched
article text (`extract`) and embedding vectors (`embed`). Both are expensive in ways
that punish iteration — 120 page fetches take minutes and knock on Imedi's bot filter,
and embeddings spend free-tier quota — so tuning a threshold would otherwise mean
paying that cost on every run.

Everything lands under `.cache/`, which is git-ignored. That is deliberate and
load-bearing: article text may be *used* but never stored where it could be published
(PLAN.md §9). Delete the folder any time; it only ever costs a re-fetch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(".cache")


def _path(namespace: str, key: str) -> Path:
    """One file per entry, named by hash.

    Hashed rather than named after the key because keys here are URLs and whole article
    texts — far too long, and full of characters Windows rejects in filenames.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / f"{digest}.json"


def get(namespace: str, key: str) -> Any | None:
    """Return the cached value, or None if it was never stored.

    A corrupt entry (half-written by an interrupted run) is treated as a miss rather
    than an error — the worst case is fetching it again.
    """
    path = _path(namespace, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["value"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def put(namespace: str, key: str, value: Any) -> None:
    """Store a value. The key is saved alongside it purely so the files are debuggable."""
    path = _path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"key": key, "value": value}, ensure_ascii=False)
    # Write to a temp file and move it into place, so an interrupted run leaves either
    # the old entry or none — never a truncated one that reads back as real data.
    temp = path.with_suffix(".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def clear() -> int:
    """Delete the whole cache. Returns how many entries were removed."""
    if not CACHE_ROOT.exists():
        return 0
    files = list(CACHE_ROOT.rglob("*.json"))
    for file in files:
        file.unlink()
    return len(files)
