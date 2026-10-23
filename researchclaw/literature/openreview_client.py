"""OpenReview API v2 client — free, no API key required.

OpenReview holds conference submissions *and* their reviews, months before the
proceedings exist. That makes it the most useful early-signal source for
novelty checks in fast-moving areas: a paper can be found here when arXiv has
only a stub and Semantic Scholar has nothing.

The public search endpoint returns every note matching the terms — including
official reviews, decisions and rebuttals — so results are filtered down to
actual submissions (a note carrying a ``title`` under an invitation whose name
contains ``Submission``).

Public API
----------
- ``search_openreview(query, limit, year_min)`` → ``list[Paper]``
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from typing import Any

from researchclaw.literature._http import fetch_json
from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE = "https://api2.openreview.net/notes/search"
_TIMEOUT = 45

_VENUE_RE = re.compile(r"^([^/]+)/(\d{4})/([^/]+)")
# v2 submission invitations end with the note type: "ICLR.cc/2025/Conference/-/Submission".
_SUBMISSION_INVITATION_RE = re.compile(r"/(?:-/)?(?:Blind_)?Submission$")
# The rest of what the search endpoint returns: reviews, decisions, comments.
_NON_SUBMISSION_INVITATION_RE = re.compile(
    r"(?:Review|Decision|Comment|Rebuttal|Response)", re.IGNORECASE
)


def _value(content: dict[str, Any], key: str) -> Any:
    """v2 API wraps every field as ``{"value": ...}``; v1 returns it raw."""
    raw = content.get(key)
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _fetch(url: str) -> dict[str, Any]:
    return fetch_json(
        url,
        source="OpenReview",
        headers={"User-Agent": "ResearchClaw/1.0"},
        timeout=_TIMEOUT,
    )


def _year_from_timestamp(raw: Any) -> int:
    """OpenReview timestamps are epoch milliseconds (sometimes seconds)."""
    try:
        value = int(str(raw).strip() or 0)
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    if value > 10**11:  # milliseconds → seconds
        value //= 1000
    try:
        return time.gmtime(value).tm_year
    except (OSError, OverflowError, ValueError):
        return 0


def _venue_from_invitations(invitations: Any) -> str:
    """Fallback venue label, e.g. ``ICLR.cc/2025/Conference/...`` → ``ICLR 2025``."""
    for inv in invitations or []:
        match = _VENUE_RE.match(str(inv))
        if match:
            # Invitation groups carry the venue's domain suffix (ICLR.cc).
            name = match.group(1).rsplit(".", 1)[0] or match.group(1)
            return f"{name} {match.group(2)}"
    return ""


def _is_submission(note: dict[str, Any]) -> bool:
    """True for a submission note, False for a review, decision or comment.

    ``source=forum`` already restricts the hit list server-side, so this is the
    client-side guard for the cases where that filter is not enough — a note
    whose ``invitations`` also list the review forums it belongs to.
    """
    invitations = [str(inv) for inv in (note.get("invitations") or [])]
    if note.get("invitation"):
        invitations.append(str(note["invitation"]))
    if not invitations:
        return True
    if any(_SUBMISSION_INVITATION_RE.search(inv) for inv in invitations):
        return True
    return not any(_NON_SUBMISSION_INVITATION_RE.search(inv) for inv in invitations)


def _to_paper(note: dict[str, Any]) -> Paper | None:
    content = note.get("content") or {}
    if not isinstance(content, dict):
        return None
    title = str(_value(content, "title") or "").strip()
    if not title:
        return None

    authors_raw = _value(content, "authors") or []
    if isinstance(authors_raw, str):
        authors_raw = [a.strip() for a in authors_raw.split(",")]
    authors = tuple(
        Author(name=str(a)) for a in authors_raw if isinstance(a, (str, int))
    )

    forum = str(note.get("forum") or note.get("id") or "")
    year = _year_from_timestamp(
        note.get("pdate") or note.get("cdate") or note.get("tcdate") or 0
    )

    pdf_path = str(_value(content, "pdf") or "")
    if pdf_path and not pdf_path.startswith("http"):
        pdf_path = f"https://openreview.net{pdf_path}"

    venue = str(_value(content, "venue") or "").strip()

    return Paper(
        paper_id=f"openreview:{forum}",
        title=title,
        authors=authors,
        year=year,
        abstract=str(_value(content, "abstract") or ""),
        venue=venue or _venue_from_invitations(note.get("invitations")),
        url=f"https://openreview.net/forum?id={forum}" if forum else "",
        pdf_url=pdf_path,
        source="openreview",
    )


def search_openreview(
    query: str,
    *,
    limit: int = 50,
    year_min: int = 0,
) -> list[Paper]:
    """Search OpenReview submissions by free-text term.

    Parameters
    ----------
    query:
        Free-text query. OpenReview's ``term`` mode does full-text matching
        over titles and abstracts.
    limit:
        Maximum notes to request. The endpoint returns reviews as well, so the
        number of surviving submissions is usually smaller.
    year_min:
        If > 0, drop notes whose posting year is earlier.

    Returns
    -------
    list[Paper]
        Parsed submissions. Empty list on failure.
    """
    limit = max(1, min(limit, 100))
    # ``source=forum`` restricts the hit list to submission *forums*; without it
    # the endpoint happily returns official reviews and rebuttals instead.
    params = {
        "term": query,
        "limit": str(limit),
        "type": "terms",
        "content": "all",
        "group": "all",
        "source": "forum",
    }
    url = f"{_BASE}?{urllib.parse.urlencode(params)}"

    try:
        payload = _fetch(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenReview search failed for %r: %s", query, exc)
        return []

    papers: list[Paper] = []
    seen: set[str] = set()
    for note in payload.get("notes") or []:
        if not isinstance(note, dict) or not _is_submission(note):
            continue
        paper = _to_paper(note)
        if paper is None or paper.paper_id in seen:
            continue
        if year_min > 0 and paper.year and paper.year < year_min:
            continue
        seen.add(paper.paper_id)
        papers.append(paper)

    logger.info("OpenReview: found %d submissions for %r", len(papers), query)
    return papers
