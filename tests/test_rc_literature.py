# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
"""Unit tests for researchclaw.literature module.

All network-dependent tests mock HTTP responses via monkeypatch.
"""

from __future__ import annotations

import json
import textwrap
import urllib.error
import xml.etree.ElementTree as ET
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import researchclaw.literature._http as _http
from researchclaw.literature.models import Author, Paper
from researchclaw.literature.core_client import (
    _to_paper as _core_to_paper,
    search_core,
)
from researchclaw.literature.crossref_client import (
    _to_paper as _crossref_to_paper,
    search_crossref,
)
from researchclaw.literature.openreview_client import (
    _is_submission,
    _to_paper as _openreview_to_paper,
    search_openreview,
)
from researchclaw.literature.pubmed_client import (
    _parse_articles,
    search_pubmed,
)
from researchclaw.literature.semantic_scholar import (
    _parse_s2_paper,
    search_semantic_scholar,
)
from researchclaw.literature.arxiv_client import (
    _convert_result,
    search_arxiv,
)
from researchclaw.literature.search import (
    _deduplicate,
    _normalise_title,
    papers_to_bibtex,
    search_papers,
    search_papers_multi_query,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────


def _make_paper(**kwargs: Any) -> Paper:
    defaults = {
        "paper_id": "s2-abc",
        "title": "Attention Is All You Need",
        "authors": (Author(name="Ashish Vaswani"),),
        "year": 2017,
        "venue": "NeurIPS",
        "citation_count": 80000,
        "doi": "10.5555/3295222.3295349",
        "arxiv_id": "1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
        "source": "semantic_scholar",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


SAMPLE_S2_RESPONSE = {
    "total": 1,
    "data": [
        {
            "paperId": "abc123",
            "title": "Test Paper on Transformers",
            "abstract": "We study transformers for NLP tasks.",
            "year": 2024,
            "venue": "NeurIPS",
            "citationCount": 42,
            "authors": [
                {"authorId": "1", "name": "Jane Smith"},
                {"authorId": "2", "name": "John Doe"},
            ],
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.00001"},
            "url": "https://www.semanticscholar.org/paper/abc123",
        }
    ],
}


SAMPLE_ARXIV_ATOM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>A Novel Approach to Protein Folding</title>
        <summary>We propose a new method for protein structure prediction.</summary>
        <published>2024-01-15T00:00:00Z</published>
        <author><name>Alice Researcher</name></author>
        <author><name>Bob Scientist</name></author>
        <link href="http://arxiv.org/abs/2401.00001v1" type="text/html"/>
        <arxiv:primary_category term="cs.AI"/>
        <arxiv:doi>10.5678/protein</arxiv:doi>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2402.00002v1</id>
        <title>Deep Reinforcement Learning Survey</title>
        <summary>A comprehensive survey of deep RL methods.</summary>
        <published>2024-02-20T00:00:00Z</published>
        <author><name>Charlie Expert</name></author>
        <link href="http://arxiv.org/abs/2402.00002v1" type="text/html"/>
        <arxiv:primary_category term="cs.LG"/>
      </entry>
    </feed>
""")


# ──────────────────────────────────────────────────────────────────────
# Author tests
# ──────────────────────────────────────────────────────────────────────


class TestAuthor:
    def test_last_name_simple(self) -> None:
        a = Author(name="Jane Smith")
        assert a.last_name() == "smith"

    def test_last_name_accented(self) -> None:
        a = Author(name="José García")
        assert a.last_name() == "garcia"  # accent stripped, but 'i' preserved

    def test_last_name_single(self) -> None:
        a = Author(name="Madonna")
        assert a.last_name() == "madonna"

    def test_last_name_empty(self) -> None:
        a = Author(name="")
        assert a.last_name() == "unknown"


# ──────────────────────────────────────────────────────────────────────
# Paper tests
# ──────────────────────────────────────────────────────────────────────


class TestPaper:
    def test_cite_key_format(self) -> None:
        p = _make_paper()
        key = p.cite_key
        assert key == "vaswani2017attention"

    def test_cite_key_no_authors(self) -> None:
        p = _make_paper(authors=())
        assert p.cite_key.startswith("anon")

    def test_cite_key_no_year(self) -> None:
        p = _make_paper(year=0)
        assert "0000" in p.cite_key

    def test_to_bibtex_contains_fields(self) -> None:
        p = _make_paper()
        bib = p.to_bibtex()
        assert "@inproceedings{vaswani2017attention," in bib
        assert "title = {Attention Is All You Need}" in bib
        assert "author = {Ashish Vaswani}" in bib
        assert "year = {2017}" in bib
        assert "doi = {10.5555/3295222.3295349}" in bib
        assert "eprint = {1706.03762}" in bib

    def test_to_bibtex_override(self) -> None:
        p = _make_paper(_bibtex_override="@article{custom, title={Custom}}")
        assert p.to_bibtex() == "@article{custom, title={Custom}}"

    def test_to_bibtex_article_no_venue(self) -> None:
        p = _make_paper(venue="", arxiv_id="2301.00001")
        bib = p.to_bibtex()
        assert "@article{" in bib
        assert "journal = {arXiv preprint arXiv:2301.00001}" in bib

    def test_to_bibtex_arxiv_category_venue(self) -> None:
        """T1.4: arXiv category codes (cs.CL) must not be used as journal names."""
        p = _make_paper(venue="cs.CL", arxiv_id="2301.00001")
        bib = p.to_bibtex()
        assert "journal = {cs.CL}" not in bib
        assert "arXiv preprint" in bib

    def test_to_dict(self) -> None:
        p = _make_paper()
        d = p.to_dict()
        assert d["paper_id"] == "s2-abc"
        assert d["cite_key"] == "vaswani2017attention"
        assert isinstance(d["authors"], list)
        assert d["authors"][0]["name"] == "Ashish Vaswani"

    def test_paper_frozen(self) -> None:
        p = _make_paper()
        with pytest.raises(AttributeError):
            p.title = "new title"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Semantic Scholar client tests
# ──────────────────────────────────────────────────────────────────────


class TestSemanticScholar:
    def test_parse_s2_paper(self) -> None:
        item = SAMPLE_S2_RESPONSE["data"][0]
        p = _parse_s2_paper(item)
        assert p.paper_id == "s2-abc123"
        assert p.title == "Test Paper on Transformers"
        assert len(p.authors) == 2
        assert p.authors[0].name == "Jane Smith"
        assert p.year == 2024
        assert p.doi == "10.1234/test"
        assert p.arxiv_id == "2401.00001"
        assert p.source == "semantic_scholar"
        assert p.citation_count == 42

    def test_search_semantic_scholar_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock urllib to return sample S2 response."""
        # Reset S2 circuit breaker (may be tripped from prior test API calls)
        from researchclaw.literature.semantic_scholar import _reset_circuit_breaker
        _reset_circuit_breaker()

        response_bytes = json.dumps(SAMPLE_S2_RESPONSE).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.urllib.request.urlopen",
            lambda *a, **kw: mock_resp,
        )

        papers = search_semantic_scholar("transformers", limit=5)
        assert len(papers) == 1
        assert papers[0].title == "Test Paper on Transformers"

    def test_search_semantic_scholar_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return empty list on network error."""
        from researchclaw.literature.semantic_scholar import _reset_circuit_breaker
        _reset_circuit_breaker()

        import urllib.error

        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.urllib.request.urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("timeout")),
        )
        # Patch sleep to speed up test
        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.time.sleep", lambda _: None
        )
        papers = search_semantic_scholar("test", limit=5)
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# arXiv client tests
# ──────────────────────────────────────────────────────────────────────


class TestArxiv:
    def test_convert_result(self) -> None:
        """Test converting arxiv.Result to Paper via the new library."""
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001v1"
        mock_result.title = "A Novel Approach to Protein Folding"
        mock_result.summary = "We study protein folding."
        mock_result.published = datetime(2024, 1, 15)
        mock_result.doi = "10.5678/protein"
        mock_result.primary_category = "q-bio.BM"

        mock_author1 = MagicMock()
        mock_author1.name = "Alice Researcher"
        mock_author2 = MagicMock()
        mock_author2.name = "Bob Scientist"
        mock_result.authors = [mock_author1, mock_author2]

        paper = _convert_result(mock_result)
        assert paper.title == "A Novel Approach to Protein Folding"
        assert paper.arxiv_id == "2401.00001"
        assert paper.year == 2024
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Alice Researcher"
        assert paper.source == "arxiv"
        assert paper.doi == "10.5678/protein"
        assert paper.venue == "q-bio.BM"

    def test_search_arxiv_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test search_arxiv with mocked arxiv library."""
        from unittest.mock import MagicMock
        from datetime import datetime
        import types

        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001v1"
        mock_result.title = "Test Paper"
        mock_result.summary = "Abstract."
        mock_result.published = datetime(2024, 1, 1)
        mock_result.doi = ""
        mock_result.primary_category = "cs.LG"
        mock_author = MagicMock()
        mock_author.name = "Test Author"
        mock_result.authors = [mock_author]

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])

        # Mock the module-level `arxiv` so the `if arxiv is None` guard
        # doesn't short-circuit before the mocked _get_client is reached.
        # Use MagicMock so all attributes (Search, SortOrder, etc.) auto-resolve.
        _fake_arxiv = MagicMock()
        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client.arxiv", _fake_arxiv,
        )
        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client._get_client",
            lambda: mock_client,
        )
        from researchclaw.literature.arxiv_client import _reset_circuit_breaker
        _reset_circuit_breaker()

        papers = search_arxiv("test", limit=10)
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"
        assert papers[0].arxiv_id == "2401.00001"

    def test_search_arxiv_error_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_arxiv returns empty list on error, not raise."""
        from unittest.mock import MagicMock
        import types

        # Build a fake arxiv module with real exception classes so
        # `except arxiv.HTTPError` doesn't TypeError.
        _fake_arxiv = types.ModuleType("arxiv")

        class _FakeHTTPError(Exception):
            pass

        class _FakeUnexpectedEmptyPageError(Exception):
            pass

        _fake_arxiv.HTTPError = _FakeHTTPError
        _fake_arxiv.UnexpectedEmptyPageError = _FakeUnexpectedEmptyPageError
        _fake_arxiv.SortCriterion = MagicMock()
        _fake_arxiv.SortOrder = MagicMock()
        _fake_arxiv.Search = MagicMock()
        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client.arxiv", _fake_arxiv,
        )

        mock_client = MagicMock()
        mock_client.results.side_effect = _FakeHTTPError("Simulated arXiv HTTP error")

        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client._get_client",
            lambda: mock_client,
        )
        from researchclaw.literature.arxiv_client import _reset_circuit_breaker
        _reset_circuit_breaker()

        papers = search_arxiv("test", limit=10)
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# Unified search & deduplication tests
# ──────────────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_dedup_by_doi(self) -> None:
        p1 = _make_paper(paper_id="s2-1", doi="10.1234/a", citation_count=100)
        p2 = _make_paper(
            paper_id="arxiv-1", doi="10.1234/a", citation_count=50, source="arxiv"
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 100  # keeps higher

    def test_dedup_by_arxiv_id(self) -> None:
        p1 = _make_paper(
            paper_id="s2-1", doi="", arxiv_id="2401.00001", citation_count=10
        )
        p2 = _make_paper(
            paper_id="arxiv-1",
            doi="",
            arxiv_id="2401.00001",
            citation_count=20,
            source="arxiv",
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 20  # arxiv version had more

    def test_dedup_by_title(self) -> None:
        p1 = _make_paper(
            paper_id="s2-1",
            doi="",
            arxiv_id="",
            title="My Cool Paper",
            citation_count=5,
        )
        p2 = _make_paper(
            paper_id="s2-2",
            doi="",
            arxiv_id="",
            title="My Cool Paper",
            citation_count=10,
        )
        result = _deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 10

    def test_dedup_no_duplicates(self) -> None:
        p1 = _make_paper(paper_id="s2-1", title="Paper A", doi="10.1/a", arxiv_id="1111.11111")
        p2 = _make_paper(paper_id="s2-2", title="Paper B", doi="10.1/b", arxiv_id="2222.22222")
        result = _deduplicate([p1, p2])
        assert len(result) == 2

    def test_normalise_title(self) -> None:
        assert _normalise_title("  The Great Paper!!! ") == "the great paper"
        assert _normalise_title("A/B Testing: Methods") == "ab testing methods"


class TestPapersToBibtex:
    def test_generates_combined(self) -> None:
        p1 = _make_paper(paper_id="s2-1", title="Paper A")
        p2 = _make_paper(paper_id="s2-2", title="Paper B", venue="ICML 2024")
        bib = papers_to_bibtex([p1, p2])
        assert bib.count("@") == 2
        assert "Paper A" in bib
        assert "Paper B" in bib


class TestSearchPapers:
    def test_search_papers_combines_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock both S2 and arXiv to verify combined search."""
        s2_paper = _make_paper(
            paper_id="s2-1", source="semantic_scholar", citation_count=100
        )
        arxiv_paper = _make_paper(
            paper_id="arxiv-1",
            title="Different Paper",
            doi="10.2/b",
            arxiv_id="2402.99999",
            source="arxiv",
            citation_count=50,
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_semantic_scholar",
            lambda *a, **kw: [s2_paper],
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_arxiv",
            lambda *a, **kw: [arxiv_paper],
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers("test", sources=["semantic_scholar", "arxiv"])
        assert len(papers) == 2
        # Should be sorted by citation_count desc
        assert papers[0].citation_count >= papers[1].citation_count

    def test_default_sources_openalex_first(self) -> None:
        """OpenAlex should be the primary (first) source — least restrictive limits."""
        from researchclaw.literature.search import _DEFAULT_SOURCES
        assert _DEFAULT_SOURCES[0] == "openalex"
        assert "semantic_scholar" in _DEFAULT_SOURCES
        assert "arxiv" in _DEFAULT_SOURCES

    def test_s2_failure_does_not_block_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When S2 fails, other sources should still return results."""
        arxiv_paper = _make_paper(
            paper_id="arxiv-ok", title="ArXiv Paper", source="arxiv",
            doi="10.1/ax", arxiv_id="2401.99991",
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_openalex",
            lambda *a, **kw: [],
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_semantic_scholar",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("S2 down")),
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_arxiv",
            lambda *a, **kw: [arxiv_paper],
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers("test")
        assert len(papers) >= 1
        assert papers[0].source == "arxiv"

    def test_search_papers_unknown_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)
        papers = search_papers("test", sources=["unknown_source"])
        assert papers == []

    def test_search_papers_multi_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def mock_search(*a: Any, **kw: Any) -> list[Paper]:
            nonlocal call_count
            call_count += 1
            return [
                _make_paper(
                    paper_id=f"s2-{call_count}",
                    title=f"Unique Paper {call_count}",
                    doi=f"10.{call_count}/unique",
                    arxiv_id=f"240{call_count}.{call_count:05d}",
                )
            ]

        monkeypatch.setattr(
            "researchclaw.literature.search.search_papers",
            mock_search,
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers_multi_query(["q1", "q2", "q3"])
        assert call_count == 3
        # All unique titles so no dedup
        assert len(papers) == 3


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_paper_with_no_meaningful_title_word(self) -> None:
        """cite_key should still work with stopword-only titles."""
        p = _make_paper(title="The And For With", year=2024)
        # All words are stopwords or <4 chars, keyword should be empty
        key = p.cite_key
        assert key.startswith("vaswani2024")

    def test_paper_multiple_authors_bibtex(self) -> None:
        p = _make_paper(
            authors=(
                Author(name="Alice One"),
                Author(name="Bob Two"),
                Author(name="Charlie Three"),
            )
        )
        bib = p.to_bibtex()
        assert "Alice One and Bob Two and Charlie Three" in bib

    def test_empty_s2_response(self) -> None:
        """_parse_s2_paper shouldn't crash on minimal data."""
        p = _parse_s2_paper({"paperId": "x"})
        assert p.paper_id == "s2-x"
        assert p.title == ""
        assert p.authors == ()


# ──────────────────────────────────────────────────────────────────────
# arXiv circuit breaker tests
# ──────────────────────────────────────────────────────────────────────


class TestArxivCircuitBreaker:
    def setup_method(self) -> None:
        from researchclaw.literature.arxiv_client import _reset_circuit_breaker
        _reset_circuit_breaker()

    def test_failure_triggers_circuit_breaker(self) -> None:
        """Three consecutive failures should trip the circuit breaker."""
        from researchclaw.literature import arxiv_client

        # Simulate 3 consecutive failures
        for _ in range(3):
            arxiv_client._cb_on_failure()

        assert arxiv_client._cb_state == arxiv_client._CB_OPEN
        assert arxiv_client._cb_trip_count == 1

    def test_breaker_open_skips_requests(self) -> None:
        """When breaker is OPEN, requests should be skipped."""
        import time as time_mod
        from researchclaw.literature import arxiv_client

        arxiv_client._cb_state = arxiv_client._CB_OPEN
        arxiv_client._cb_open_since = time_mod.monotonic()
        arxiv_client._cb_cooldown_sec = 999

        assert not arxiv_client._cb_should_allow()

    def test_success_resets_breaker(self) -> None:
        """A successful request should reset the circuit breaker."""
        from researchclaw.literature import arxiv_client

        arxiv_client._cb_state = arxiv_client._CB_HALF_OPEN
        arxiv_client._cb_consecutive_429s = 2

        arxiv_client._cb_on_success()
        assert arxiv_client._cb_state == arxiv_client._CB_CLOSED
        assert arxiv_client._cb_consecutive_429s == 0

    def test_half_open_probe_failure_doubles_cooldown(self) -> None:
        """Probe failure in HALF_OPEN should double the cooldown."""
        from researchclaw.literature import arxiv_client

        arxiv_client._cb_state = arxiv_client._CB_HALF_OPEN
        initial_cooldown = arxiv_client._cb_cooldown_sec

        arxiv_client._cb_on_failure()
        assert arxiv_client._cb_state == arxiv_client._CB_OPEN
        assert arxiv_client._cb_cooldown_sec == min(initial_cooldown * 2, 600)

    def test_search_with_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_arxiv should return empty list on HTTPError."""
        import types

        _fake_arxiv = types.ModuleType("arxiv")

        class _FakeHTTPError(Exception):
            pass

        class _FakeUnexpectedEmptyPageError(Exception):
            pass

        _fake_arxiv.HTTPError = _FakeHTTPError
        _fake_arxiv.UnexpectedEmptyPageError = _FakeUnexpectedEmptyPageError
        _fake_arxiv.SortCriterion = MagicMock()
        _fake_arxiv.SortOrder = MagicMock()
        _fake_arxiv.Search = MagicMock()
        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client.arxiv", _fake_arxiv,
        )

        mock_client = MagicMock()
        mock_client.results.side_effect = _FakeHTTPError("Simulated 429")

        monkeypatch.setattr(
            "researchclaw.literature.arxiv_client._get_client",
            lambda: mock_client,
        )
        from researchclaw.literature.arxiv_client import _reset_circuit_breaker
        _reset_circuit_breaker()

        papers = search_arxiv("test", limit=5)
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# OpenAlex client tests
# ──────────────────────────────────────────────────────────────────────


SAMPLE_OPENALEX_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W123456",
            "title": "Attention Is All You Need",
            "authorships": [
                {
                    "author": {"display_name": "Ashish Vaswani"},
                    "institutions": [{"display_name": "Google Brain"}],
                }
            ],
            "publication_year": 2017,
            "primary_location": {
                "source": {"display_name": "NeurIPS"}
            },
            "cited_by_count": 85000,
            "doi": "https://doi.org/10.5555/3295222.3295349",
            "ids": {
                "openalex": "https://openalex.org/W123456",
                "arxiv": "https://arxiv.org/abs/1706.03762",
            },
            "abstract_inverted_index": {
                "The": [0],
                "dominant": [1],
                "models": [2, 6],
                "are": [3],
                "based": [4],
                "on": [5],
            },
        }
    ]
}


class TestOpenAlex:
    def test_parse_openalex_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock urllib to return sample OpenAlex response."""
        from researchclaw.literature.openalex_client import search_openalex

        response_bytes = json.dumps(SAMPLE_OPENALEX_RESPONSE).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            "researchclaw.literature.openalex_client.urllib.request.urlopen",
            lambda *a, **kw: mock_resp,
        )

        papers = search_openalex("attention", limit=5)
        assert len(papers) == 1
        p = papers[0]
        assert p.title == "Attention Is All You Need"
        assert p.year == 2017
        assert p.citation_count == 85000
        assert p.doi == "10.5555/3295222.3295349"
        assert p.arxiv_id == "1706.03762"
        assert p.source == "openalex"
        assert p.authors[0].name == "Ashish Vaswani"

    def test_openalex_api_key_and_email_are_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAlex API key and mailto should be encoded in request params."""
        import urllib.parse

        from researchclaw.literature.openalex_client import search_openalex

        response_bytes = json.dumps(SAMPLE_OPENALEX_RESPONSE).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        captured: dict[str, str] = {}

        def fake_urlopen(req: Any, *args: Any, **kwargs: Any) -> MagicMock:
            captured["url"] = req.full_url
            return mock_resp

        monkeypatch.setattr(
            "researchclaw.literature.openalex_client.urllib.request.urlopen",
            fake_urlopen,
        )

        papers = search_openalex(
            "attention",
            limit=5,
            email="bot@example.com",
            api_key="oa-test-key",
        )

        params = urllib.parse.parse_qs(
            urllib.parse.urlsplit(captured["url"]).query
        )
        assert len(papers) == 1
        assert params["mailto"] == ["bot@example.com"]
        assert params["api_key"] == ["oa-test-key"]

    def test_abstract_reconstruction(self) -> None:
        from researchclaw.literature.openalex_client import _reconstruct_abstract

        inv_idx = {"Hello": [0], "world": [1], "foo": [3], "bar": [2]}
        result = _reconstruct_abstract(inv_idx)
        assert result == "Hello world bar foo"

    def test_abstract_empty(self) -> None:
        from researchclaw.literature.openalex_client import _reconstruct_abstract
        assert _reconstruct_abstract(None) == ""
        assert _reconstruct_abstract({}) == ""

    def test_openalex_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return empty list on network error."""
        from researchclaw.literature.openalex_client import search_openalex

        monkeypatch.setattr(
            "researchclaw.literature.openalex_client.urllib.request.urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("timeout")),
        )
        monkeypatch.setattr(
            "researchclaw.literature.openalex_client.time.sleep", lambda _: None,
        )

        papers = search_openalex("test", limit=5)
        assert papers == []


# ──────────────────────────────────────────────────────────────────────
# Multi-source fallback tests
# ──────────────────────────────────────────────────────────────────────


class TestMultiSourceFallback:
    def test_openalex_failure_falls_back_to_s2_and_arxiv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When OpenAlex fails, S2 and arXiv should still return results."""
        arxiv_paper = _make_paper(
            paper_id="arxiv-ok", title="ArXiv Paper", source="arxiv",
            doi="10.1/ax", arxiv_id="2401.99999",
        )
        s2_paper = _make_paper(
            paper_id="s2-ok", title="S2 Paper", source="semantic_scholar",
            doi="10.1/s2", arxiv_id="2402.99999",
        )

        monkeypatch.setattr(
            "researchclaw.literature.search.search_openalex",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("OpenAlex down")),
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_semantic_scholar",
            lambda *a, **kw: [s2_paper],
        )
        monkeypatch.setattr(
            "researchclaw.literature.search.search_arxiv",
            lambda *a, **kw: [arxiv_paper],
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers("test")
        assert len(papers) >= 1
        sources = {p.source for p in papers}
        assert "semantic_scholar" in sources or "arxiv" in sources

    def test_search_papers_passes_openalex_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        openalex_paper = _make_paper(
            paper_id="oalex-ok",
            title="OpenAlex Paper",
            source="openalex",
            doi="10.1/oa",
            arxiv_id="2401.12345",
        )

        def fake_openalex(query: str, **kwargs: object) -> list[Paper]:
            captured["query"] = query
            captured.update(kwargs)
            return [openalex_paper]

        monkeypatch.setattr(
            "researchclaw.literature.search.search_openalex",
            fake_openalex,
        )
        monkeypatch.setattr("researchclaw.literature.search.time.sleep", lambda _: None)

        papers = search_papers(
            "test",
            sources=["openalex"],
            openalex_email="bot@example.com",
            openalex_api_key="oa-test-key",
        )

        assert len(papers) == 1
        assert captured["query"] == "test"
        assert captured["email"] == "bot@example.com"
        assert captured["api_key"] == "oa-test-key"


# ──────────────────────────────────────────────────────────────────────
# Cache TTL tests
# ──────────────────────────────────────────────────────────────────────


class TestCacheTTL:
    def test_source_specific_ttl(self, tmp_path: Any) -> None:
        """arXiv cache should expire after 24h, not 7 days."""
        from researchclaw.literature.cache import get_cached, put_cache, _SOURCE_TTL

        assert _SOURCE_TTL["arxiv"] == 86400  # 24h
        assert _SOURCE_TTL["semantic_scholar"] == 86400 * 3

        # Put and get immediately — should hit
        put_cache("test", "arxiv", 10, [{"paper_id": "x", "title": "Y"}], cache_base=tmp_path)
        result = get_cached("test", "arxiv", 10, cache_base=tmp_path)
        assert result is not None
        assert len(result) == 1

    def test_citation_verify_ttl_is_permanent(self) -> None:
        from researchclaw.literature.cache import _SOURCE_TTL
        assert _SOURCE_TTL["citation_verify"] >= 86400 * 365


import urllib.error


class TestS2RetrySleepBudget:
    """The retry loops must not sleep after their final attempt."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception,
        call: str,
    ) -> list[float]:
        import researchclaw.literature.semantic_scholar as s2

        s2._reset_circuit_breaker()
        delays: list[float] = []
        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.urllib.request.urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(exc),
        )
        monkeypatch.setattr(
            "researchclaw.literature.semantic_scholar.time.sleep",
            lambda d: delays.append(d),
        )
        if call == "get":
            assert s2._request_with_retry("https://x/y", {}) is None
        else:
            assert s2._post_with_retry("https://x/y", {}, b"{}") is None
        return delays

    def test_get_connection_error_does_not_sleep_after_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.semantic_scholar import _MAX_RETRIES

        delays = self._run(monkeypatch, ConnectionResetError("boom"), "get")
        assert len(delays) == _MAX_RETRIES - 1

    def test_post_connection_error_does_not_sleep_after_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.semantic_scholar import _MAX_RETRIES

        delays = self._run(monkeypatch, ConnectionResetError("boom"), "post")
        assert len(delays) == _MAX_RETRIES - 1

    def test_get_429_does_not_sleep_after_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.semantic_scholar import _MAX_RETRIES

        err = urllib.error.HTTPError("url", 429, "Too Many", Message(), None)
        delays = self._run(monkeypatch, err, "get")
        assert len(delays) <= _MAX_RETRIES - 1

    def test_post_429_does_not_sleep_after_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.semantic_scholar import _MAX_RETRIES

        err = urllib.error.HTTPError("url", 429, "Too Many", Message(), None)
        delays = self._run(monkeypatch, err, "post")
        assert len(delays) <= _MAX_RETRIES - 1

    def test_retries_still_back_off_between_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard against 'fixing' the sleep by removing backoff entirely."""
        delays = self._run(monkeypatch, ConnectionResetError("boom"), "get")
        assert delays, "expected backoff sleeps between attempts"
        assert delays == sorted(delays), "backoff should be non-decreasing"
        assert all(d > 0 for d in delays)



# ──────────────────────────────────────────────────────────────────────
# Shared HTTP helper: retry / backoff / fatal-status handling
# ──────────────────────────────────────────────────────────────────────


def _http_error(status: int, retry_after: str = "") -> urllib.error.HTTPError:
    headers = Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.test", status, "error", headers, None
    )


class TestHttpRetry:
    def _failing_run(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> list[float]:
        """Run one fetch that always fails; return the recorded sleeps."""
        sleeps: list[float] = []
        monkeypatch.setattr(_http.time, "sleep", lambda s: sleeps.append(s))

        def _raise(*_a: Any, **_kw: Any) -> None:
            raise exc

        monkeypatch.setattr(_http.urllib.request, "urlopen", _raise)
        with pytest.raises(Exception):
            _http.fetch_json("https://example.test", source="Test", base_delay=1.0)
        return sleeps

    def test_retries_transient_connection_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps = self._failing_run(monkeypatch, urllib.error.URLError("boom"))
        assert len(sleeps) == 2  # 3 attempts → 2 waits
        assert all(s >= 1.0 for s in sleeps), "backoff must not be zero"

    def test_retries_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert len(self._failing_run(monkeypatch, _http_error(429))) == 2

    def test_fatal_status_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._failing_run(monkeypatch, _http_error(404)) == []

    def test_retry_after_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps = self._failing_run(monkeypatch, _http_error(429, "0.01"))
        assert len(sleeps) == 2
        assert all(s < 1.0 for s in sleeps), "Retry-After should cap the backoff"

    def test_absurd_retry_after_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 10-minute Retry-After must fail fast, not stall a pipeline stage."""
        assert self._failing_run(monkeypatch, _http_error(429, "600")) == []

    def test_success_after_transient_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_http.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def _flaky(*_a: Any, **_kw: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("temporary")
            resp = MagicMock()
            resp.read.return_value = b'{"ok": true}'
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        monkeypatch.setattr(_http.urllib.request, "urlopen", _flaky)
        assert _http.fetch_json("https://example.test", source="Test") == {"ok": True}
        assert calls["n"] == 2


# ──────────────────────────────────────────────────────────────────────
# Crossref client
# ──────────────────────────────────────────────────────────────────────

SAMPLE_CROSSREF: dict[str, Any] = {
    "message": {
        "items": [
            {
                "DOI": "10.1145/1234.5678",
                "title": ["Efficient KV Cache Eviction"],
                "author": [
                    {"given": "Jane", "family": "Smith"},
                    {"given": "", "family": "Doe"},
                ],
                "issued": {"date-parts": [[2024, 5, 1]]},
                "container-title": ["Proceedings of ASPLOS"],
                "is-referenced-by-count": 12,
                "abstract": "<jats:p>We study <jats:italic>eviction</jats:italic>.</jats:p>",
                "URL": "https://doi.org/10.1145/1234.5678",
            },
            {"DOI": "10.1145/empty", "title": []},
        ]
    }
}


class TestCrossrefClient:
    def test_parse_item(self) -> None:
        paper = _crossref_to_paper(SAMPLE_CROSSREF["message"]["items"][0])
        assert paper is not None
        assert paper.title == "Efficient KV Cache Eviction"
        assert paper.year == 2024
        assert paper.doi == "10.1145/1234.5678"
        assert paper.venue == "Proceedings of ASPLOS"
        assert paper.citation_count == 12
        assert paper.source == "crossref"
        assert [a.name for a in paper.authors] == ["Jane Smith", "Doe"]
        assert "jats" not in paper.abstract, "JATS markup should be stripped"
        assert "We study" in paper.abstract

    def test_item_without_title_is_skipped(self) -> None:
        assert _crossref_to_paper(SAMPLE_CROSSREF["message"]["items"][1]) is None

    def test_search_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        def _fake(url: str, **kw: Any) -> Any:
            captured["url"] = url
            return SAMPLE_CROSSREF

        monkeypatch.setattr("researchclaw.literature.crossref_client.fetch_json", _fake)
        papers = search_crossref("kv cache", limit=5, email="me@example.com")
        assert len(papers) == 1
        assert "mailto=me%40example.com" in captured["url"]
        assert "query.bibliographic=kv+cache" in captured["url"]

    def test_search_returns_empty_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr("researchclaw.literature.crossref_client.fetch_json", _boom)
        assert search_crossref("kv cache") == []


# ──────────────────────────────────────────────────────────────────────
# OpenReview client
# ──────────────────────────────────────────────────────────────────────

SAMPLE_OPENREVIEW: dict[str, Any] = {
    "notes": [
        {
            "id": "abc123",
            "forum": "abc123",
            "invitations": ["ICLR.cc/2025/Conference/-/Submission"],
            "cdate": 1735689600000,  # epoch milliseconds → 2025-01-01
            "content": {
                "title": {"value": "A New KV Cache Method"},
                "abstract": {"value": "Abstract text."},
                "authors": {"value": ["Ada Lovelace", "Alan Turing"]},
                "pdf": {"value": "/pdf/abc123.pdf"},
                "venue": {"value": "ICLR 2025 Poster"},
            },
        },
        {
            "id": "rev1",
            "forum": "abc123",
            "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Official_Review"],
            "content": {"title": {"value": "Official Review of A New KV Cache Method"}},
        },
    ]
}


class TestOpenReviewClient:
    def test_parse_submission(self) -> None:
        paper = _openreview_to_paper(SAMPLE_OPENREVIEW["notes"][0])
        assert paper is not None
        assert paper.paper_id == "openreview:abc123"
        assert paper.year == 2025, "epoch milliseconds must not be truncated"
        assert paper.venue == "ICLR 2025 Poster"
        assert paper.pdf_url == "https://openreview.net/pdf/abc123.pdf"
        assert paper.url == "https://openreview.net/forum?id=abc123"
        assert [a.name for a in paper.authors] == ["Ada Lovelace", "Alan Turing"]

    def test_review_notes_are_not_papers(self) -> None:
        assert _is_submission(SAMPLE_OPENREVIEW["notes"][0]) is True
        assert _is_submission(SAMPLE_OPENREVIEW["notes"][1]) is False

    def test_venue_falls_back_to_invitation(self) -> None:
        note = {
            "id": "x1",
            "forum": "x1",
            "invitations": ["ICLR.cc/2024/Conference/-/Submission"],
            "cdate": 1700000000000,
            "content": {"title": {"value": "T"}},
        }
        paper = _openreview_to_paper(note)
        assert paper is not None
        assert paper.venue == "ICLR 2024"

    def test_search_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        def _fake(url: str, **kw: Any) -> Any:
            captured["url"] = url
            return SAMPLE_OPENREVIEW

        monkeypatch.setattr(
            "researchclaw.literature.openreview_client.fetch_json", _fake
        )
        papers = search_openreview("kv cache", limit=10)
        assert len(papers) == 1, "review notes must be filtered out"
        assert papers[0].pdf_url is not None and papers[0].pdf_url.endswith(".pdf")
        assert "source=forum" in captured["url"]

    def test_search_returns_empty_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr(
            "researchclaw.literature.openreview_client.fetch_json", _boom
        )
        assert search_openreview("kv cache") == []


# ──────────────────────────────────────────────────────────────────────
# PubMed client
# ──────────────────────────────────────────────────────────────────────

SAMPLE_PUBMED_XML = textwrap.dedent(
    """\
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345678</PMID>
          <Article>
            <ArticleTitle>Deep learning for glioma survival</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">Text A.</AbstractText>
              <AbstractText Label="RESULTS">Text B.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <ForeName>Jane</ForeName>
                <LastName>Doe</LastName>
                <AffiliationInfo><Affiliation>Some University</Affiliation></AffiliationInfo>
              </Author>
            </AuthorList>
            <Journal>
              <Title>Journal of Tests</Title>
              <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
            </Journal>
            <ELocationID EIdType="doi">10.1000/xyz</ELocationID>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
)


class TestPubMedClient:
    def test_parse_article(self) -> None:
        root = ET.fromstring(SAMPLE_PUBMED_XML)
        papers = _parse_articles(root, 0)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.paper_id == "pmid:12345678"
        assert paper.title == "Deep learning for glioma survival"
        assert paper.year == 2023
        assert paper.venue == "Journal of Tests"
        assert paper.doi == "10.1000/xyz"
        assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        assert paper.source == "pubmed"
        assert "Text A." in paper.abstract and "Text B." in paper.abstract
        assert paper.authors[0].name == "Jane Doe"
        assert paper.authors[0].affiliation == "Some University"

    def test_year_filter(self) -> None:
        root = ET.fromstring(SAMPLE_PUBMED_XML)
        assert _parse_articles(root, 2024) == []

    def test_search_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.pubmed_client.fetch_json",
            lambda *a, **kw: {"esearchresult": {"idlist": ["12345678"]}},
        )
        monkeypatch.setattr(
            "researchclaw.literature.pubmed_client.fetch_bytes",
            lambda *a, **kw: SAMPLE_PUBMED_XML.encode("utf-8"),
        )
        papers = search_pubmed("glioma survival", limit=5, api_key="k", email="me@x.io")
        assert len(papers) == 1
        assert papers[0].paper_id == "pmid:12345678"

    def test_efetch_is_batched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NCBI rejects over-long URLs, so PMIDs must be fetched in chunks."""
        ids = [str(10_000_000 + i) for i in range(450)]
        monkeypatch.setattr(
            "researchclaw.literature.pubmed_client.fetch_json",
            lambda *a, **kw: {"esearchresult": {"idlist": ids}},
        )
        batches: list[str] = []

        def _fetch_bytes(url: str, **kw: Any) -> bytes:
            batches.append(url)
            return b"<PubmedArticleSet></PubmedArticleSet>"

        monkeypatch.setattr(
            "researchclaw.literature.pubmed_client.fetch_bytes", _fetch_bytes
        )
        search_pubmed("q", limit=450)
        assert len(batches) == 3, "450 ids → 200 + 200 + 50"

    def test_esearch_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr("researchclaw.literature.pubmed_client.fetch_json", _boom)
        assert search_pubmed("q") == []


# ──────────────────────────────────────────────────────────────────────
# CORE client
# ──────────────────────────────────────────────────────────────────────

SAMPLE_CORE: dict[str, Any] = {
    "results": [
        {
            "id": 111,
            "title": "Ada-KV: Adaptive Budget Allocation",
            "yearPublished": 2024,
            "abstract": "abs",
            "downloadUrl": "https://core.ac.uk/download/111.pdf",
            "doi": "10.1000/ada",
            "citationCount": 5,
            "publisher": "ACM",
            "authors": [{"name": "A B"}, "C D"],
        },
        {
            # same work from a second provider, no PDF
            "id": 222,
            "title": "Ada-KV:  Adaptive   Budget Allocation",
            "yearPublished": 2024,
            "downloadUrl": "",
        },
    ]
}


class TestCoreClient:
    def test_parse_item(self) -> None:
        paper = _core_to_paper(SAMPLE_CORE["results"][0])
        assert paper is not None
        assert paper.paper_id == "core:111"
        assert paper.year == 2024
        assert paper.citation_count == 5
        assert paper.doi == "10.1000/ada"
        assert paper.pdf_url == "https://core.ac.uk/download/111.pdf"
        assert [a.name for a in paper.authors] == ["A B", "C D"]

    def test_without_api_key_no_request_is_made(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("must not call the API without a key")

        monkeypatch.setattr("researchclaw.literature.core_client.fetch_json", _boom)
        assert search_core("kv cache") == []

    def test_search_dedupes_same_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.core_client.fetch_json",
            lambda *a, **kw: SAMPLE_CORE,
        )
        papers = search_core("kv cache", api_key="k")
        assert len(papers) == 1

    def test_search_returns_empty_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr("researchclaw.literature.core_client.fetch_json", _boom)
        assert search_core("kv cache", api_key="k") == []


# ──────────────────────────────────────────────────────────────────────
# PubMed domain gate (stage 4 source selection)
# ──────────────────────────────────────────────────────────────────────


class TestPubMedDomainGate:
    SOURCES = ("openalex", "pubmed", "core")

    def test_pubmed_dropped_for_non_biomedical_domain(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _sources_for_domains

        selected = _sources_for_domains(self.SOURCES, ("algorithms", "machine_learning"))
        assert selected == ("openalex", "core")

    def test_pubmed_kept_for_biomedical_domain(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _sources_for_domains

        selected = _sources_for_domains(self.SOURCES, ("biomedicine",))
        assert "pubmed" in selected

    def test_gate_can_be_disabled(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _sources_for_domains

        selected = _sources_for_domains(self.SOURCES, ("algorithms",), gate_pubmed=False)
        assert selected == self.SOURCES
