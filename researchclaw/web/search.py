"""Web search with Tavily, DuckDuckGo and opt-in Parallel Search MCP.

Tavily is the primary search engine (installed as a dependency).
A DuckDuckGo HTML scrape fallback exists for when no API key is set.

Usage::

    client = WebSearchClient(api_key="tvly-...")
    results = client.search("knowledge distillation survey 2024")
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import quote_plus
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchTaskContext:
    """Caller-owned immutable identity for one conversation or research run.

    Reuse across batches, follow-ups, retries and cancellations of the same run.
    Create a fresh context for independent or reset research. Parallel uses this identity
    as its MCP ``session_id``; other providers ignore it.
    """

    session_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""
    score: float = 0.0
    source: str = ""  # "tavily" | "duckduckgo" | "parallel"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class WebSearchResponse:
    """Response from a web search query."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""  # Tavily can provide a direct AI answer
    elapsed_seconds: float = 0.0
    source: str = ""  # "tavily" | "duckduckgo" | "parallel"

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0


class WebSearchClient:
    """General-purpose web search client.

    Uses Tavily (installed) as primary engine. Falls back to DuckDuckGo
    HTML scraping only if no Tavily API key is available.

    Parameters
    ----------
    api_key:
        Tavily API key. Falls back to ``TAVILY_API_KEY`` env var.
    max_results:
        Default number of results per query.
    search_depth:
        Tavily search depth: "basic" or "advanced".
    include_answer:
        Whether to request Tavily's AI-generated answer.
    provider:
        ``auto`` retains Tavily with DuckDuckGo fallback. ``parallel`` opts
        into anonymous Search MCP; explicit providers never switch on errors.
    timeout:
        Overall Parallel MCP request deadline, in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        max_results: int = 10,
        search_depth: str = "advanced",
        include_answer: bool = True,
        provider: str = "auto",
        timeout: float = 30.0,
    ) -> None:
        if provider not in {"auto", "parallel", "tavily", "duckduckgo"}:
            raise ValueError(f"Unknown web search provider: {provider}")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Web search timeout must be finite and positive")
        self.provider = provider
        self.timeout = timeout
        self._parallel_client = None
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        task_context: SearchTaskContext | None = None,
    ) -> WebSearchResponse:
        """Search the web for a query.

        Pass a caller-owned context for related calls in one research run.
        Without a context, Parallel omits the optional session identity.
        """
        limit = max_results or self.max_results
        t0 = time.monotonic()

        if self.provider == "parallel":
            from researchclaw.web.parallel import ParallelSearchClient

            if self._parallel_client is None:
                self._parallel_client = ParallelSearchClient(timeout=self.timeout)
            results = self._parallel_client.search(
                query, limit=limit,
                include_domains=include_domains, exclude_domains=exclude_domains,
                task_context=task_context,
            )
            return WebSearchResponse(
                query=query, results=results, source="parallel",
                elapsed_seconds=time.monotonic() - t0,
            )

        if self.provider == "tavily":
            if not self.api_key:
                raise ValueError("Tavily search requires a Tavily API key")
            return self._search_tavily(query, limit, include_domains, exclude_domains, t0)
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, limit, t0)

        # Tavily is the primary engine
        if self.api_key:
            try:
                return self._search_tavily(query, limit, include_domains, exclude_domains, t0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tavily search failed, falling back to DuckDuckGo: %s", exc)

        return self._search_duckduckgo(query, limit, t0)

    def search_multi(
        self,
        queries: list[str],
        *,
        max_results: int | None = None,
        inter_query_delay: float = 1.0,
        task_context: SearchTaskContext | None = None,
    ) -> list[WebSearchResponse]:
        """Run multiple queries with deduplication and a shared task context.

        Without a caller-owned context, Parallel omits the optional session identity.
        """
        responses = []
        seen_urls: set[str] = set()

        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(inter_query_delay)
            context_kwargs = {"task_context": task_context} if task_context is not None else {}
            resp = self.search(query, max_results=max_results, **context_kwargs)
            unique_results = [r for r in resp.results if r.url not in seen_urls]
            seen_urls.update(r.url for r in unique_results)
            resp.results = unique_results
            responses.append(resp)

        return responses

    # ------------------------------------------------------------------
    # Tavily backend (primary — uses installed tavily-python SDK)
    # ------------------------------------------------------------------

    def _search_tavily(
        self,
        query: str,
        limit: int,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        t0: float,
    ) -> WebSearchResponse:
        """Search using Tavily API (installed SDK)."""
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.api_key)

        kwargs: dict[str, Any] = {
            "query": query,
            "max_results": limit,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        response = client.search(**kwargs)
        elapsed = time.monotonic() - t0

        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                source="tavily",
            ))

        return WebSearchResponse(
            query=query,
            results=results,
            answer=response.get("answer", ""),
            elapsed_seconds=elapsed,
            source="tavily",
        )

    # ------------------------------------------------------------------
    # DuckDuckGo fallback (no API key needed)
    # ------------------------------------------------------------------

    def _search_duckduckgo(
        self, query: str, limit: int, t0: float
    ) -> WebSearchResponse:
        """Fallback: scrape DuckDuckGo HTML search results."""
        encoded = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

        try:
            resp = urlopen(req, timeout=15)  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            logger.warning("DuckDuckGo search failed: %s", exc)
            return WebSearchResponse(query=query, elapsed_seconds=elapsed, source="duckduckgo")

        results = self._parse_ddg_html(html, limit)
        elapsed = time.monotonic() - t0
        return WebSearchResponse(query=query, results=results, elapsed_seconds=elapsed, source="duckduckgo")

    @staticmethod
    def _parse_ddg_html(html: str, limit: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML results page."""
        results = []
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title_html) in enumerate(links[:limit]):
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            if "duckduckgo.com" in url:
                # Extract actual URL from DDG redirect: //duckduckgo.com/l/?uddg=https%3A...
                from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs, unquote as _unquote
                _parsed_ddg = _urlparse(url)
                _uddg = _parse_qs(_parsed_ddg.query).get("uddg")
                if _uddg:
                    url = _unquote(_uddg[0])
                else:
                    continue
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo"))

        return results
