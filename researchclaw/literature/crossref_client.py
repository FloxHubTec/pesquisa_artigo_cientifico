"""Crossref REST API client — free, no API key required.

Crossref indexes publisher metadata (ACM, IEEE, Springer, Nature, Elsevier,
PLOS ...), which is the paywalled-venue coverage the preprint-centric
OpenAlex / Semantic Scholar / arXiv trio misses. Most records carry no
abstract, so ``Paper.abstract`` is frequently empty.

Public API
----------
- ``search_crossref(query, limit, year_min, email)`` → ``list[Paper]``

Polite pool: pass ``mailto`` in the query string and a matching User-Agent.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

from researchclaw.literature._http import fetch_json
from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE = "https://api.crossref.org/works"
_TIMEOUT = 45
_SELECT = (
    "DOI,title,author,issued,container-title,is-referenced-by-count,type,URL,abstract"
)

_JATS_RE = re.compile(r"<[^>]+>")


def _strip_jats(text: str) -> str:
    """Crossref abstracts come as JATS XML fragments."""
    return re.sub(r"\s+", " ", _JATS_RE.sub(" ", text)).strip()


def _fetch(url: str, email: str) -> dict[str, Any]:
    ua = f"ResearchClaw/1.0 (mailto:{email})" if email else "ResearchClaw/1.0"
    return fetch_json(
        url, source="Crossref", headers={"User-Agent": ua}, timeout=_TIMEOUT
    )


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _to_paper(item: dict[str, Any]) -> Paper | None:
    title = " ".join(item.get("title") or []).strip()
    if not title:
        return None

    authors = tuple(
        Author(name=f"{a.get('given', '')} {a.get('family', '')}".strip())
        for a in item.get("author") or []
        if isinstance(a, dict)
    )

    year = 0
    parts = (item.get("issued") or {}).get("date-parts") or [[0]]
    if parts and isinstance(parts[0], list) and parts[0]:
        try:
            year = int(parts[0][0])
        except (TypeError, ValueError):
            year = 0

    doi = str(item.get("DOI") or "")
    return Paper(
        paper_id=doi or str(item.get("URL") or ""),
        title=title,
        authors=authors,
        year=year,
        abstract=_strip_jats(str(item.get("abstract") or "")),
        venue=" ".join(item.get("container-title") or []).strip(),
        citation_count=_as_int(item.get("is-referenced-by-count")),
        doi=doi,
        url=str(item.get("URL") or ""),
        source="crossref",
    )


def search_crossref(
    query: str,
    *,
    limit: int = 50,
    year_min: int = 0,
    email: str = "",
) -> list[Paper]:
    """Search Crossref by bibliographic query.

    Parameters
    ----------
    query:
        Free-text query (title / author / venue words).
    limit:
        Maximum number of records (Crossref caps ``rows`` at 1000).
    year_min:
        If > 0, drop records older than this year.
    email:
        Contact address for the polite pool → higher, more stable limits.

    Returns
    -------
    list[Paper]
        Parsed papers. Empty list on failure.
    """
    limit = max(1, min(limit, 1000))
    params: dict[str, str] = {
        "query.bibliographic": query,
        "rows": str(limit),
        "select": _SELECT,
    }
    if email:
        params["mailto"] = email

    url = f"{_BASE}?{urllib.parse.urlencode(params)}"
    try:
        payload = _fetch(url, email)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Crossref search failed for %r: %s", query, exc)
        return []

    items = (payload.get("message") or {}).get("items") or []
    papers: list[Paper] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        paper = _to_paper(item)
        if paper is None:
            continue
        if year_min > 0 and paper.year and paper.year < year_min:
            continue
        papers.append(paper)

    papers.sort(key=lambda p: p.citation_count, reverse=True)
    logger.info("Crossref: found %d papers for %r", len(papers), query)
    return papers
