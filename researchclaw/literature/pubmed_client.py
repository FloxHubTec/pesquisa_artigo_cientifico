"""PubMed E-utilities client (NCBI) — free, API key raises the rate limit.

The only biomedical source in the set: PubMed indexes MEDLINE, which neither
OpenAlex, Semantic Scholar nor arXiv cover with the same recall. It is gated by
``literature_search.pubmed_domain_gate`` because its word-level query
translation floods non-biomedical topics with unrelated clinical hits.

Two calls per query: ``esearch`` returns PMIDs, ``efetch`` (retmode=xml)
returns the metadata. Without a key NCBI allows 3 req/s; with a key, 10 req/s
and a higher daily cap.

Public API
----------
- ``search_pubmed(query, limit, year_min, api_key, email)`` → ``list[Paper]``
"""

from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from researchclaw.literature._http import fetch_bytes, fetch_json
from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TIMEOUT = 45
_TOOL = "researchclaw"
_HEADERS = {"User-Agent": "ResearchClaw/1.0"}
# NCBI answers 414 for over-long GET URLs, so PMIDs are fetched in batches.
_EFETCH_BATCH = 200


def _common_params(api_key: str, email: str) -> dict[str, str]:
    params = {"tool": _TOOL}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    return params


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _to_paper(article: ET.Element) -> Paper | None:
    citation = article.find("MedlineCitation")
    if citation is None:
        return None

    title = _text(citation.find("Article/ArticleTitle"))
    if not title:
        return None

    abstract_parts = [
        _text(node) for node in citation.findall("Article/Abstract/AbstractText")
    ]
    abstract = " ".join(part for part in abstract_parts if part)

    authors: list[Author] = []
    for node in citation.findall("Article/AuthorList/Author"):
        last = _text(node.find("LastName"))
        fore = _text(node.find("ForeName"))
        collective = _text(node.find("CollectiveName"))
        name = " ".join(part for part in (fore, last) if part) or collective
        if name:
            affiliation = _text(node.find("AffiliationInfo/Affiliation"))
            authors.append(Author(name=name, affiliation=affiliation))

    journal = _text(citation.find("Article/Journal/Title"))
    year_text = _text(citation.find("Article/Journal/JournalIssue/PubDate/Year"))
    if not year_text:
        year_text = _text(citation.find("Article/Journal/JournalIssue/PubDate/MedlineDate"))
    year = 0
    if year_text[:4].isdigit():
        year = int(year_text[:4])

    doi = ""
    for node in citation.findall("Article/ELocationID"):
        if node.get("EIdType") == "doi":
            doi = _text(node)
            break
    if not doi:
        for node in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if node.get("IdType") == "doi":
                doi = _text(node)
                break

    pmid = _text(citation.find("PMID"))
    return Paper(
        paper_id=f"pmid:{pmid}" if pmid else doi,
        title=title,
        authors=tuple(authors),
        year=year,
        abstract=abstract,
        venue=journal,
        doi=doi,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        source="pubmed",
    )


def search_pubmed(
    query: str,
    *,
    limit: int = 50,
    year_min: int = 0,
    api_key: str = "",
    email: str = "",
) -> list[Paper]:
    """Search PubMed and return parsed articles.

    Parameters
    ----------
    query:
        PubMed query syntax (Boolean operators, ``[MeSH Terms]``, ``[au]`` ...).
    limit:
        Maximum records; NCBI's ``retmax`` caps at 10000.
    year_min:
        If > 0, only articles published in this year or later.
    api_key:
        NCBI API key — raises 3 req/s to 10 req/s.
    email:
        Contact address NCBI asks for on every E-utilities request.

    Returns
    -------
    list[Paper]
        Parsed articles. Empty list on failure.
    """
    limit = max(1, min(limit, 10_000))
    common = _common_params(api_key, email)

    esearch_params: dict[str, str] = {
        "db": "pubmed",
        "term": query,
        "retmax": str(limit),
        "retmode": "json",
        "sort": "relevance",
        **common,
    }
    if year_min > 0:
        esearch_params["mindate"] = str(year_min)
        esearch_params["maxdate"] = "3000"
        esearch_params["datetype"] = "pdat"

    try:
        payload = fetch_json(
            f"{_BASE}/esearch.fcgi?{urllib.parse.urlencode(esearch_params)}",
            source="PubMed esearch",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        ids = (payload.get("esearchresult") or {}).get("idlist") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("PubMed esearch failed for %r: %s", query, exc)
        return []

    if not ids:
        logger.info("PubMed: no hits for %r", query)
        return []

    papers: list[Paper] = []
    for start in range(0, len(ids), _EFETCH_BATCH):
        batch = ids[start : start + _EFETCH_BATCH]
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(str(i) for i in batch),
            "retmode": "xml",
            **common,
        }
        try:
            raw = fetch_bytes(
                f"{_BASE}/efetch.fcgi?{urllib.parse.urlencode(efetch_params)}",
                source="PubMed efetch",
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            root = ET.fromstring(raw)  # noqa: S314 - trusted NCBI endpoint
        except Exception as exc:  # noqa: BLE001
            # One failing batch must not discard the articles already parsed.
            logger.warning("PubMed efetch failed for %r: %s", query, exc)
            continue
        papers.extend(_parse_articles(root, year_min))

    logger.info("PubMed: found %d papers for %r", len(papers), query)
    return papers


def _parse_articles(root: ET.Element, year_min: int) -> list[Paper]:
    """Papers from one ``efetch`` response, minus anything older than *year_min*."""
    papers: list[Paper] = []
    for article in root.findall("PubmedArticle"):
        paper = _to_paper(article)
        if paper is None:
            continue
        if year_min > 0 and paper.year and paper.year < year_min:
            continue
        papers.append(paper)
    return papers
