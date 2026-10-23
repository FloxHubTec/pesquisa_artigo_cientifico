"""CORE API v3 client — free academic key, full-text oriented.

CORE aggregates 200M+ open-access records and, unlike the metadata-only
sources, exposes a ``downloadUrl`` for the actual PDF. That is what makes
full-text analysis possible: OpenAlex / Semantic Scholar / arXiv hand back
abstracts, CORE hands back the document.

Public API
----------
- ``search_core(query, limit, year_min, api_key)`` → ``list[Paper]``

Requires ``CORE_API_KEY`` (free at https://core.ac.uk/services/api).
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from researchclaw.literature._http import fetch_json
from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE = "https://api.core.ac.uk/v3/search/works"
_TIMEOUT = 60


def _fetch(url: str, api_key: str) -> dict[str, Any]:
    return fetch_json(
        url,
        source="CORE",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ResearchClaw/1.0",
        },
        timeout=_TIMEOUT,
    )


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _to_paper(item: dict[str, Any]) -> Paper | None:
    title = str(item.get("title") or "").strip()
    if not title:
        return None

    authors: list[Author] = []
    for entry in item.get("authors") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
        else:
            name = str(entry).strip()
        if name:
            authors.append(Author(name=name))

    year = _as_int(item.get("yearPublished"))

    core_id = str(item.get("id") or "")
    pdf_url = str(item.get("downloadUrl") or "")
    doi = str(item.get("doi") or "")

    return Paper(
        paper_id=f"core:{core_id}" if core_id else doi,
        title=title,
        authors=tuple(authors),
        year=year,
        abstract=str(item.get("abstract") or "").strip(),
        venue=str(item.get("publisher") or "").strip(),
        citation_count=_as_int(item.get("citationCount")),
        doi=doi,
        url=pdf_url or (f"https://core.ac.uk/outputs/{core_id}" if core_id else ""),
        pdf_url=pdf_url,
        source="core",
    )


def search_core(
    query: str,
    *,
    limit: int = 50,
    year_min: int = 0,
    api_key: str = "",
) -> list[Paper]:
    """Search CORE for works matching *query*.

    Parameters
    ----------
    query:
        Free-text query (CORE supports its own field syntax, e.g. ``title:``).
    limit:
        Maximum records (API caps ``limit`` at 100).
    year_min:
        If > 0, drop records older than this year.
    api_key:
        CORE API key. Without it the endpoint answers 401, so an empty key
        returns ``[]`` immediately.

    Returns
    -------
    list[Paper]
        Parsed papers, with ``pdf_url`` filled when CORE has a document.
    """
    if not api_key:
        logger.info("CORE: no api_key configured — source skipped")
        return []

    limit = max(1, min(limit, 100))
    params = {"q": query, "limit": str(limit)}
    url = f"{_BASE}?{urllib.parse.urlencode(params)}"

    try:
        payload = _fetch(url, api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CORE search failed for %r: %s", query, exc)
        return []

    papers: list[Paper] = []
    seen_titles: set[str] = set()
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        paper = _to_paper(item)
        if paper is None:
            continue
        if year_min > 0 and paper.year and paper.year < year_min:
            continue
        # CORE indexes the same work once per data provider, so the same title
        # arrives several times under different ids.
        key = " ".join(paper.title.lower().split())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        papers.append(paper)

    with_pdf = sum(1 for p in papers if p.pdf_url)
    logger.info(
        "CORE: found %d papers (%d with PDF) for %r", len(papers), with_pdf, query
    )
    return papers
