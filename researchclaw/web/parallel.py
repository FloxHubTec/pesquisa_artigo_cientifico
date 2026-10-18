"""Anonymous Parallel Search MCP adapter for the synchronous web-search pipeline."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from researchclaw import __version__
from researchclaw.web.search import SearchResult

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = f"AutoResearchClaw/{__version__} (+https://github.com/aiming-lab/AutoResearchClaw)"


class ParallelSearchClient:
    """Use the optional MCP SDK's Streamable HTTP transport without credentials.

    One stable search session id is shared by all queries from this client.
    The blocking portal also supports callers already running an asyncio loop.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.session_id = uuid4().hex

    def search(
        self, query: str, *, limit: int,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("Parallel result limit must be positive")
        try:
            from anyio.from_thread import start_blocking_portal
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError('Parallel search requires: pip install "researchclaw[parallel]"') from exc

        with start_blocking_portal() as portal:
            payload = portal.call(self._search, query)

        results = []
        for row in payload.get("results", []):
            url = row.get("url", "")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            host = parsed.hostname.lower()

            def matches(domains: list[str]) -> bool:
                return any(
                    host == domain.lstrip(".").lower()
                    or host.endswith("." + domain.lstrip(".").lower())
                    for domain in domains
                )

            # Anonymous MCP settings are server-managed, so filter locally.
            if include_domains and not matches(include_domains):
                continue
            if exclude_domains and matches(exclude_domains):
                continue
            content = "\n\n".join(row.get("excerpts", []))
            results.append(SearchResult(
                title=row.get("title") or url, url=url,
                snippet=content[:500], content=content, source="parallel",
            ))
            if len(results) >= limit:
                break
        return results

    async def _search(self, query: str) -> dict[str, Any]:
        import anyio
        from mcp import ClientSession, types
        from mcp.client.streamable_http import streamablehttp_client

        timeout = timedelta(seconds=self.timeout)
        with anyio.fail_after(self.timeout):
            async with streamablehttp_client(
                MCP_URL, headers={"User-Agent": USER_AGENT},
                timeout=timeout, sse_read_timeout=timeout,
            ) as (read, write, _):
                async with ClientSession(
                    read, write, read_timeout_seconds=timeout,
                    client_info=types.Implementation(name="AutoResearchClaw", version=__version__),
                ) as session:
                    await session.initialize()
                    response = await session.call_tool("web_search", {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": self.session_id,
                    })

        if response.isError:
            message = " ".join(getattr(block, "text", "") for block in response.content)
            raise RuntimeError(f"Parallel search failed: {message}")
        payload = response.structuredContent
        if payload is None:
            for block in response.content:
                if isinstance(block, types.TextContent):
                    payload = json.loads(block.text)
                    break
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Parallel search returned no results array")
        return payload
