"""Tests for the on-disk scratch cache.

No network. Every test redirects `CACHE_ROOT` into pytest's temp directory, so nothing
touches the real `.cache/`.
"""

import json

import pytest

from pipeline import cache


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    return tmp_path / "cache"


def test_roundtrip_survives_georgian_text():
    cache.put("text", "https://netgazeti.ge/news/1", "საქართველოს პარლამენტი")
    assert cache.get("text", "https://netgazeti.ge/news/1") == "საქართველოს პარლამენტი"


def test_missing_key_is_none_not_an_error():
    assert cache.get("text", "never stored") is None


def test_namespaces_do_not_collide():
    """Same key, different namespace — a URL's text and its vector must not overwrite
    each other."""
    cache.put("text", "same-key", "prose")
    cache.put("vectors", "same-key", [0.1, 0.2])
    assert cache.get("text", "same-key") == "prose"
    assert cache.get("vectors", "same-key") == [0.1, 0.2]


def test_long_and_awkward_keys_are_usable_as_filenames():
    """Keys are whole article texts and URLs — far past any filename limit, and full of
    characters Windows rejects. Hashing is what makes them storable."""
    key = "https://on.ge/story/" + "?:*<>|" * 100 + "ქ" * 500
    cache.put("text", key, "ok")
    assert cache.get("text", key) == "ok"


def test_empty_string_is_a_real_cached_value():
    """'This page has no extractable text' is an answer worth remembering — otherwise a
    dead page gets re-fetched on every run."""
    cache.put("text", "https://imedinews.ge/dead", "")
    assert cache.get("text", "https://imedinews.ge/dead") == ""


def test_corrupt_entry_reads_as_a_miss(temp_cache):
    """A run killed mid-write must not poison later runs; worst case is re-fetching."""
    cache.put("text", "key", "good")
    path = next(temp_cache.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("text", "key") is None


def test_writes_are_atomic(temp_cache):
    """No half-written file is ever left behind under the real name."""
    cache.put("text", "key", "value")
    assert not list(temp_cache.rglob("*.tmp"))
    stored = json.loads(next(temp_cache.rglob("*.json")).read_text(encoding="utf-8"))
    assert stored["value"] == "value"


def test_clear_removes_everything_and_counts_it():
    cache.put("text", "a", "1")
    cache.put("vectors", "b", [2.0])
    assert cache.clear() == 2
    assert cache.get("text", "a") is None


def test_clear_on_a_missing_cache_is_harmless():
    assert cache.clear() == 0
