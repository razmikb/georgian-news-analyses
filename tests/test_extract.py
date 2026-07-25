"""Tests for article body extraction.

No network: `fetch` is stubbed. The HTML here is invented, not a snapshot of a real
outlet — the point is to exercise our own lead/body/truncation rules, and inventing the
prose keeps real article text out of the repo (PLAN.md §9).
"""

import pytest

from pipeline import cache, extract

PARAGRAPHS = [
    "პარლამენტმა დღეს მიიღო კანონპროექტი პირველი მოსმენით და დეპუტატებმა მხარი დაუჭირეს.",
    "ოპოზიციის წარმომადგენლებმა სხდომა დატოვეს და კენჭისყრაში მონაწილეობა არ მიიღეს.",
    "მესამე აბზაცი შეიცავს ფონურ ინფორმაციას, რომელიც ბევრ სხვა სტატიაშიც მეორდება.",
    "მეოთხე აბზაცი დამატებით დეტალებს აღწერს და სტატიის დასასრულს წარმოადგენს.",
]

ARTICLE_HTML = (
    "<html><body><article>"
    + "".join(f"<p>{p}</p>" for p in PARAGRAPHS)
    + "</article></body></html>"
).encode("utf-8")


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Never touch the real .cache/ during tests."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")


@pytest.fixture
def stub_fetch(monkeypatch):
    """Replace the network with a scripted response; return the URLs actually requested."""
    requested: list[str] = []

    def install(result):
        def fake_fetch(url, **kwargs):
            requested.append(url)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(extract, "fetch", fake_fetch)

    install.requested = requested
    return install


# --- Splitting rules ------------------------------------------------------------


def test_paragraphs_drops_blank_lines():
    assert extract.paragraphs("one\n\n  \ntwo\n") == ["one", "two"]


def test_lead_takes_the_opening_paragraphs():
    text = "\n".join(PARAGRAPHS)
    assert extract.lead(text, count=2) == "\n".join(PARAGRAPHS[:2])


def test_lead_of_a_single_paragraph_article_is_that_paragraph():
    assert extract.lead(PARAGRAPHS[0]) == PARAGRAPHS[0]


def test_body_returns_everything_when_short_enough():
    text = "\n".join(PARAGRAPHS)
    assert extract.body(text, max_chars=10_000) == text


def test_body_truncates_on_a_paragraph_boundary():
    """Cutting mid-sentence would feed the embedding a fragment; whole paragraphs keep
    the text coherent."""
    text = "\n".join(PARAGRAPHS)
    result = extract.body(text, max_chars=len(PARAGRAPHS[0]) + 20)
    assert result == PARAGRAPHS[0]


def test_body_hard_cuts_when_the_first_paragraph_is_itself_too_long():
    """No boundary to fall back on — better a truncated paragraph than nothing at all."""
    result = extract.body("ა" * 500, max_chars=100)
    assert len(result) == 100


# --- Boilerplate removal --------------------------------------------------------
#
# Measured on the real Imedi page behind eval pair #64: a 402-character article followed
# by 9,583 characters of *other stories* from a "latest news" block. Two unrelated Imedi
# articles would have shared those 9,583 characters and scored as near-identical.


def test_listing_tail_is_cut_at_the_first_timestamp():
    article = "რეალური სტატიის პირველი აბზაცი, საკმარისად გრძელი რომ ნამდვილი იყოს."
    text = "\n".join(
        [
            article,
            "25 ივლისი 2026, 23:42",
            "სხვა სტატიის სათაური, რომელიც აქ არ უნდა იყოს.",
            "25 ივლისი 2026, 23:36",
            "კიდევ ერთი უცხო სტატია.",
        ]
    )
    assert extract.clean(text) == article


def test_repeated_headline_is_dropped():
    """Imedi opens the body with the headline again; embedding it twice just doubles
    the weight of words the headline variant already covers."""
    headline = "პრემიერ-მინისტრის ვიზიტი სინგაპურში"
    text = f"{headline}\nსტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."
    assert extract.clean(text, headline) == "სტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."


def test_headline_match_ignores_surrounding_whitespace():
    headline = "  პრემიერის ვიზიტი  "
    text = "პრემიერის ვიზიტი\nსტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."
    assert "პრემიერის ვიზიტი\n" not in extract.clean(text, headline)


def test_photo_credit_is_dropped():
    """The line between headline and first paragraph on Imedi — it would otherwise BE
    the lead."""
    text = "ფოტო: IMEDI/ვიდეოკადრი\nსტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."
    assert extract.clean(text) == "სტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."


def test_english_credit_lines_are_dropped_too():
    text = "Photo: Reuters\nსტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."
    assert extract.clean(text).startswith("სტატიის")


def test_a_long_sentence_starting_with_photo_is_kept():
    """'Photo' at the start does not make it a credit — only a short line does. A real
    sentence about a photograph must survive."""
    sentence = (
        "ფოტო, რომელიც სოციალურ ქსელში გავრცელდა, ბევრმა მომხმარებელმა "
        "გააზიარა და დიდი განხილვა გამოიწვია."
    )
    assert extract.clean(sentence) == sentence


def test_navigation_crumb_above_the_listing_is_dropped():
    text = "\n".join(
        [
            "სტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია.",
            "ყველა სიახლე",
            "25 ივლისი 2026, 23:42",
        ]
    )
    assert extract.clean(text) == "სტატიის ნამდვილი ტექსტი, რომელიც საკმარისად გრძელია."


def test_a_sentence_containing_a_time_is_not_mistaken_for_a_timestamp():
    """The cut rule must key on a *bare* stamp. A sentence that happens to mention a
    time and a year is article text, and cutting there would truncate the story."""
    text = "შეხვედრა 2026 წლის 25 ივლისს, 14:30 საათზე გაიმართა და ორ საათს გაგრძელდა."
    assert extract.clean(text) == text


def test_clean_of_ordinary_text_changes_nothing():
    """Formula and Netgazeti extract cleanly; the rules must not damage them."""
    text = "პირველი აბზაცი საკმარისად გრძელია.\nმეორე აბზაცი ასევე გრძელია და რჩება."
    assert extract.clean(text) == text


# --- Fetch + extract ------------------------------------------------------------


def test_article_text_extracts_prose_from_html(stub_fetch):
    stub_fetch(ARTICLE_HTML)
    text = extract.article_text("https://netgazeti.ge/news/1")
    assert text is not None
    assert PARAGRAPHS[0] in text
    assert "<p>" not in text


def test_unreachable_article_returns_none_instead_of_raising(stub_fetch):
    """One dead link must not sink a run of 120 — the same per-source isolation rule the
    ingest pipeline follows. Imedi's bot filter still 403s occasionally."""
    stub_fetch(extract.FetchError("HTTP 403"))
    assert extract.article_text("https://imedinews.ge/ge/1") is None


def test_page_with_no_prose_returns_none(stub_fetch):
    """Extraction degrades instead of failing — on a page with no article it hands back
    whatever scrap it can find. A vector built from 'menu' sits close to every other junk
    vector, so unrelated broken pages would cluster together and look like an event."""
    stub_fetch(b"<html><body><nav>menu</nav></body></html>")
    assert extract.article_text("https://on.ge/story/1") is None


def test_text_too_short_to_be_an_article_is_rejected(stub_fetch):
    stub_fetch(b"<html><body><article><p>\xe1\x83\x93\xe1\x83\x93\xe1\x83\x93</p></article></body></html>")
    assert extract.article_text("https://on.ge/story/2") is None


# --- Caching --------------------------------------------------------------------


def test_second_call_is_served_from_cache(stub_fetch):
    """The reason the cache exists: re-running the experiment must not re-fetch 120 pages,
    and must not knock on Imedi's bot filter twenty times."""
    stub_fetch(ARTICLE_HTML)
    first = extract.article_text("https://netgazeti.ge/news/1")
    second = extract.article_text("https://netgazeti.ge/news/1")
    assert first == second
    assert len(stub_fetch.requested) == 1


def test_a_dead_page_is_not_refetched_every_run(stub_fetch):
    """'No text here' is cached as an empty string, so it stays one fetch, not one per run."""
    stub_fetch(b"<html><body><nav>menu</nav></body></html>")
    assert extract.article_text("https://on.ge/story/1") is None
    assert extract.article_text("https://on.ge/story/1") is None
    assert len(stub_fetch.requested) == 1


def test_no_cache_always_refetches(stub_fetch):
    stub_fetch(ARTICLE_HTML)
    extract.article_text("https://netgazeti.ge/news/1", use_cache=False)
    extract.article_text("https://netgazeti.ge/news/1", use_cache=False)
    assert len(stub_fetch.requested) == 2


def test_a_failed_fetch_is_not_cached_as_empty(stub_fetch):
    """A 403 is temporary. Caching it as 'no text' would make one bad moment permanent."""
    stub_fetch(extract.FetchError("HTTP 403"))
    assert extract.article_text("https://imedinews.ge/ge/1") is None
    stub_fetch(ARTICLE_HTML)
    assert extract.article_text("https://imedinews.ge/ge/1") is not None
