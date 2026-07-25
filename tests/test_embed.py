"""Tests for the Gemini embedding client.

No network and no API key: the SDK call is replaced with a stub. What is checked here is
the handful of things that would silently corrupt every similarity score downstream —
normalization, batch ordering, caching — not that Google's API works.
"""

import math

import pytest

from pipeline import cache, embed


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


@pytest.fixture
def stub_api(monkeypatch):
    """Stub the SDK call itself, so our batching, retry and caching logic all stay real.

    `batches` records the texts sent in each API call.
    """
    batches: list[list[str]] = []

    def install(vector_for, *, fail_times=0, error=None, drop_results=False):
        state = {"failures": fail_times}

        def fake_embed_content(*, model, contents, config):
            batches.append(list(contents))
            if state["failures"] > 0:
                state["failures"] -= 1
                raise error or RuntimeError("429 RESOURCE_EXHAUSTED")
            values = [FakeEmbedding(vector_for(t)) for t in contents]
            return FakeResponse(values[:-1] if drop_results else values)

        fake_client = type(
            "FakeClient", (), {"models": type("Models", (), {"embed_content": None})()}
        )()
        fake_client.models.embed_content = fake_embed_content
        monkeypatch.setattr(embed, "client", lambda: fake_client)

    install.batches = batches
    monkeypatch.setattr(embed.time, "sleep", lambda _: None)
    return install


# --- Normalization: the silent score-corrupter -----------------------------------


def test_normalize_gives_unit_length():
    result = embed.normalize([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(v * v for v in result)), 1.0)


def test_normalize_preserves_direction():
    assert embed.normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_normalize_survives_an_all_zero_vector():
    """Dividing by zero here would crash a whole run over one odd input."""
    assert embed.normalize([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_of_identical_vectors_is_one():
    vector = embed.normalize([1.0, 2.0, 3.0])
    assert embed.cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert embed.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_embeddings_come_back_normalized(stub_api):
    """gemini-embedding-001 does NOT normalize truncated (768-dim) output. If we forget,
    cosine similarity stops being cosine similarity and every threshold is wrong."""
    stub_api(lambda t: [10.0, 0.0, 0.0])
    [vector] = embed.embed_texts(["სათაური"])
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


# --- Batching and ordering -------------------------------------------------------


def test_results_keep_the_input_order(stub_api):
    """Scrambled order would silently score every pair against the wrong partner."""
    stub_api(lambda t: [float(len(t)), 1.0])
    texts = ["a", "bbb", "cc", "dddd"]
    vectors = embed.embed_texts(texts)
    for text, vector in zip(texts, vectors, strict=True):
        assert vector == pytest.approx(embed.normalize([float(len(text)), 1.0]))


def test_large_input_is_split_into_batches(stub_api, monkeypatch):
    monkeypatch.setattr(embed, "EMBED_BATCH_SIZE", 3)
    stub_api(lambda t: [1.0, float(len(t))])
    texts = [f"headline {i}" for i in range(7)]
    assert len(embed.embed_texts(texts)) == 7
    assert [len(b) for b in stub_api.batches] == [3, 3, 1]


def test_duplicate_texts_cost_one_api_slot(stub_api):
    """Two outlets can run an identical wire headline; no reason to pay for it twice."""
    stub_api(lambda t: [1.0, float(len(t))])
    vectors = embed.embed_texts(["same", "other", "same"])
    assert vectors[0] == vectors[2]
    assert stub_api.batches == [["same", "other"]]


def test_empty_input_makes_no_call(stub_api):
    stub_api(lambda t: [1.0])
    assert embed.embed_texts([]) == []
    assert stub_api.batches == []


# --- Failure handling ------------------------------------------------------------


def test_quota_error_is_retried_then_succeeds(stub_api):
    """Free-tier quota clears by waiting out the minute; giving up would waste the run."""
    stub_api(lambda t: [1.0, 0.0], fail_times=2)
    assert len(embed.embed_texts(["headline"])) == 1
    assert len(stub_api.batches) == 3


def test_quota_that_never_clears_raises_a_readable_error(stub_api):
    stub_api(lambda t: [1.0], fail_times=99)
    with pytest.raises(embed.EmbedError) as excinfo:
        embed.embed_texts(["headline"])
    assert "quota" in str(excinfo.value).lower()


def test_a_bad_api_key_fails_immediately(stub_api):
    """Not a quota problem — waiting three minutes for it to clear tells the user nothing."""
    stub_api(lambda t: [1.0], fail_times=99, error=RuntimeError("401 API key not valid"))
    with pytest.raises(embed.EmbedError):
        embed.embed_texts(["headline"])
    assert len(stub_api.batches) == 1


def test_quota_error_is_told_apart_from_a_real_error():
    assert embed._is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert embed._is_quota_error(RuntimeError("Quota exceeded for model"))
    assert not embed._is_quota_error(RuntimeError("401 API key not valid"))


# --- Caching ---------------------------------------------------------------------


def test_cached_vectors_are_not_re_requested(stub_api):
    stub_api(lambda t: [1.0, float(len(t))])
    first = embed.embed_texts(["headline"])
    second = embed.embed_texts(["headline"])
    assert first == second
    assert len(stub_api.batches) == 1


def test_only_the_new_texts_are_sent(stub_api):
    stub_api(lambda t: [1.0, float(len(t))])
    embed.embed_texts(["one"])
    embed.embed_texts(["one", "two"])
    assert stub_api.batches == [["one"], ["two"]]


def test_changing_the_model_invalidates_the_cache(stub_api, monkeypatch):
    """Vectors from two different models are not comparable. Reusing one across a model
    change would produce scores that look fine and mean nothing."""
    stub_api(lambda t: [1.0, 2.0])
    embed.embed_texts(["headline"])
    monkeypatch.setattr(embed, "EMBED_MODEL", "some-other-model")
    embed.embed_texts(["headline"])
    assert len(stub_api.batches) == 2


def test_no_cache_always_re_requests(stub_api):
    stub_api(lambda t: [1.0, 2.0])
    embed.embed_texts(["headline"], use_cache=False)
    embed.embed_texts(["headline"], use_cache=False)
    assert len(stub_api.batches) == 2


def test_a_short_response_is_caught_rather_than_misaligned(stub_api):
    """If the API returns fewer vectors than texts, silently zipping them would shift every
    result by one — pairs scored against the wrong partner, with no error to notice."""
    stub_api(lambda t: [1.0, float(len(t))], drop_results=True)
    with pytest.raises(embed.EmbedError):
        embed.embed_texts(["one", "two", "three"])
