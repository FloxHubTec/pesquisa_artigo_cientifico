"""Exercise the real MCP SDK and runtime resolver with an HTTP-level fixture."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import pytest

pytest.importorskip("mcp")  # Installed with researchclaw[parallel].
import anyio
import httpx
import mcp.client.streamable_http as transport

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls._literature import _execute_literature_collect
from researchclaw.web.agent import WebSearchAgent
from researchclaw.web.parallel import MCP_URL, USER_AGENT, ParallelSearchClient
from researchclaw.web.search import WebSearchClient, WebSearchResponse


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for key in ("TAVILY_API_KEY", "PARALLEL_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mcp_http(monkeypatch):
    """Keep real MCP initialize, notification, discovery and tool dispatch."""
    state = {"requests": [], "clients": [], "error": False, "delay": 0, "text_only": False,
             "payload": {"results": [
                 {"title": "Research evidence", "url": "https://papers.example.org/article",
                  "excerpts": ["A useful research excerpt", "Further evidence"]},
                 {"title": None, "url": "https://other.example.org/page", "excerpts": []},
                 {"url": "file:///etc/passwd", "excerpts": ["unsafe URL"]},
             ]}}

    async def handle(request):
        state["requests"].append(request)
        assert str(request.url) == MCP_URL
        assert request.headers["user-agent"] == USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        body = json.loads(request.content)
        method = body["method"]
        if "id" not in body:
            assert method == "notifications/initialized"
            return httpx.Response(202)
        if method == "initialize":
            assert body["params"]["clientInfo"]["name"] == "AutoResearchClaw"
            result = {"protocolVersion": body["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert method == "tools/call"
            assert body["params"]["name"] == "web_search"
            args = body["params"]["arguments"]
            assert args["objective"] == args["search_queries"][0]
            assert len(args["session_id"]) == 32
            if state["delay"]:
                await anyio.sleep(state["delay"])
            payload = state["payload"]
            result = {"content": [{"type": "text", "text": json.dumps(payload)}],
                      "isError": state["error"]}
            if not state["text_only"]:
                result["structuredContent"] = payload
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                              headers={"Mcp-Session-Id": "fixture-session"})

    original = transport.streamablehttp_client

    def factory(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle), **kwargs)
        state["clients"].append(client)
        return client

    def connect(*args, **kwargs):
        return original(*args, **kwargs, httpx_client_factory=factory)

    monkeypatch.setattr(transport, "streamablehttp_client", connect)
    return state


def tool_calls(state):
    bodies = [json.loads(r.content) for r in state["requests"] if r.method == "POST"]
    return [b for b in bodies if b["method"] == "tools/call"]


def test_agent_consumes_mcp_search_and_retains_one_session(mcp_http):
    agent = WebSearchAgent(search_provider="parallel", enable_scholar=False,
                           enable_crawling=False, enable_pdf=False)
    result = agent.search_and_extract("research", search_queries=["research evidence"])
    context = result.to_context_string()
    assert "https://papers.example.org/article" in context
    assert "A useful research excerpt" in context
    assert result.web_results[0].content.endswith("Further evidence")
    assert result.web_results[0].source == "parallel"
    assert "file:" not in context
    agent.web_client.search("another query")
    calls = tool_calls(mcp_http)
    assert calls[0]["params"]["arguments"]["session_id"] == calls[1]["params"]["arguments"]["session_id"]
    assert any(json.loads(r.content)["method"] == "tools/list"
               for r in mcp_http["requests"] if r.method == "POST")


def test_text_content_domains_and_limit(mcp_http):
    mcp_http["text_only"] = True
    client = WebSearchClient(provider="parallel")
    result = client.search("research", max_results=1, include_domains=["example.org"],
                           exclude_domains=["other.example.org"])
    assert len(result.results) == 1
    assert result.results[0].url == "https://papers.example.org/article"
    assert not client.search("research", include_domains=["unrelated.org"]).results


def test_parallel_overrides_configured_tavily(mcp_http, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "saved-tavily-key")
    client = WebSearchClient(provider="parallel")
    client._search_tavily = Mock(side_effect=AssertionError("unexpected Tavily call"))
    client._search_duckduckgo = Mock(side_effect=AssertionError("unexpected DDG call"))
    assert client.search("research").source == "parallel"


@pytest.mark.parametrize("provider,key,expected", [
    ("auto", "", "duckduckgo"), ("auto", "saved-key", "tavily"),
    ("duckduckgo", "saved-key", "duckduckgo"), ("tavily", "saved-key", "tavily"),
])
def test_existing_choices(provider, key, expected, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", key)
    client = WebSearchClient(provider=provider)
    client._search_tavily = Mock(return_value=WebSearchResponse("q", source="tavily"))
    client._search_duckduckgo = Mock(return_value=WebSearchResponse("q", source="duckduckgo"))
    assert client.search("q").source == expected


def test_explicit_errors_never_switch_provider(mcp_http):
    mcp_http["error"] = True
    client = WebSearchClient(provider="parallel")
    client._search_duckduckgo = Mock(side_effect=AssertionError("unexpected fallback"))
    with pytest.raises(RuntimeError, match="Parallel search failed"):
        client.search("research")
    with pytest.raises(ValueError, match="Tavily API key"):
        WebSearchClient(provider="tavily").search("q")
    tavily = WebSearchClient(provider="tavily", api_key="saved")
    tavily._search_tavily = Mock(side_effect=RuntimeError("provider failed"))
    with pytest.raises(RuntimeError, match="provider failed"):
        tavily.search("q")


def test_malformed_response_is_an_error(mcp_http):
    mcp_http["payload"] = {"unexpected": "shape"}
    with pytest.raises(ValueError, match="results array"):
        WebSearchClient(provider="parallel").search("q")


def test_timeout_and_cancellation_close_transport(mcp_http):
    mcp_http["delay"] = 10
    with pytest.raises(TimeoutError):
        ParallelSearchClient(timeout=0.1).search("q", limit=1)

    async def cancel():
        client = ParallelSearchClient(timeout=10)
        with anyio.move_on_after(0.1) as scope:
            await client._search("q")
        assert scope.cancel_called

    asyncio.run(cancel())
    assert all(client.is_closed for client in mcp_http["clients"])


def test_sync_agent_inside_asyncio_loop(mcp_http):
    async def run():
        return WebSearchAgent(search_provider="parallel", enable_scholar=False,
                              enable_crawling=False, enable_pdf=False).search_and_extract(
                                  "q", search_queries=["q"])
    assert asyncio.run(run()).web_results[0].source == "parallel"


@pytest.mark.parametrize("kwargs", [{"provider": "typo"}, {"timeout": 0},
                                   {"timeout": float("nan")}, {"timeout": float("inf")}])
def test_invalid_selection_fails(kwargs):
    with pytest.raises(ValueError):
        WebSearchClient(**kwargs)


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(ImportError, match=r"researchclaw\[parallel\]"):
        WebSearchClient(provider="parallel").search("q")


@pytest.mark.parametrize("enabled", [True, False])
def test_stage4_config_to_context_without_other_network(enabled, mcp_http, monkeypatch, tmp_path):
    data = {
        "project": {"name": "test"}, "research": {"topic": "research"},
        "runtime": {"timezone": "UTC"}, "notifications": {"channel": "console"},
        "knowledge_base": {"root": str(tmp_path / "kb")},
        "llm": {"provider": "openai-compatible", "base_url": "http://localhost", "api_key_env": "TEST"},
        "web_search": {"enabled": enabled, "provider": "parallel", "search_timeout_sec": 5,
                       "enable_scholar": False, "enable_crawling": False,
                       "enable_pdf_extraction": False},
    }
    config = RCConfig.from_dict(data, check_paths=False)
    assert config.web_search.provider == "parallel"
    assert config.to_dict()["web_search"]["search_timeout_sec"] == 5
    from researchclaw.literature.models import Paper
    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query",
                        lambda *args, **kwargs: [Paper(paper_id="test", title="Academic paper", year=2025)])
    run_dir = tmp_path / "run"
    prior = run_dir / "stage-03"
    prior.mkdir(parents=True)
    (prior / "queries.json").write_text(json.dumps({"queries": ["research evidence"]}))
    stage_dir = run_dir / "stage-04"
    stage_dir.mkdir()
    _execute_literature_collect(stage_dir, run_dir, config, AdapterBundle(), llm=None)
    context = stage_dir / "web_context.md"
    if enabled:
        assert "A useful research excerpt" in context.read_text()
        assert "https://papers.example.org/article" in context.read_text()
        metadata = json.loads((stage_dir / "web_search_result.json").read_text())
        assert metadata["web_results"][0]["source"] == "parallel"
    else:
        assert not context.exists()
        assert not mcp_http["requests"]
