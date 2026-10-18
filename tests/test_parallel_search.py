"""Exercise the real MCP SDK and runtime resolver with an HTTP-level fixture."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

pytest.importorskip("mcp")  # Installed with researchclaw[parallel].
import anyio
import httpx
import mcp.client.streamable_http as transport

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.web.agent import WebSearchAgent
from researchclaw.web.parallel import MCP_URL, USER_AGENT, ParallelSearchClient
from researchclaw.web.search import SearchTaskContext, WebSearchClient, WebSearchResponse


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
            if "session_id" in args:
                assert len(args["session_id"]) == 32
            if state.get("on_call"):
                await state["on_call"](args)
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


def test_reusable_agent_scopes_each_topic_at_mcp_boundary(mcp_http):
    agent = WebSearchAgent(search_provider="parallel", enable_scholar=False,
                           enable_crawling=False, enable_pdf=False)
    results = [agent.search_and_extract(topic, search_queries=[f"{topic} evidence", f"{topic} survey"])
               for topic in ("vision", "language")]
    for topic, result in zip(("vision", "language"), results):
        assert result.topic == topic
        context = result.to_context_string()
        assert "https://papers.example.org/article" in context
        assert "A useful research excerpt" in context
        assert result.web_results[0].content.endswith("Further evidence")
        assert result.web_results[0].source == "parallel"
        assert "file:" not in context
        assert len(result.web_results) == 2  # Cross-query URL deduplication still applies.
    calls = tool_calls(mcp_http)
    args = [c["params"]["arguments"] for c in calls]
    assert [a["search_queries"][0] for a in args] == [
        "vision evidence", "vision survey", "language evidence", "language survey"]
    assert args[0]["session_id"] == args[1]["session_id"]
    assert args[2]["session_id"] == args[3]["session_id"]
    assert args[0]["session_id"] != args[2]["session_id"]
    assert any(json.loads(r.content)["method"] == "tools/list"
               for r in mcp_http["requests"] if r.method == "POST")


@pytest.mark.parametrize("caller_owned", [False, True])
def test_reusable_agent_interleaved_topics_keep_local_context(mcp_http, caller_owned):
    agent = WebSearchAgent(search_provider="parallel", enable_scholar=False,
                           enable_crawling=False, enable_pdf=False)
    a_first, b_first, a_second = (threading.Event() for _ in range(3))

    async def interleave(args):
        query = args["search_queries"][0]
        if query == "a1":
            a_first.set()
            assert await anyio.to_thread.run_sync(b_first.wait, 10)
        elif query == "b1":
            b_first.set()
            assert await anyio.to_thread.run_sync(a_second.wait, 10)
        elif query == "a2":
            a_second.set()

    mcp_http["on_call"] = interleave
    with ThreadPoolExecutor(max_workers=2) as pool:
        context_a = {"task_context": SearchTaskContext()} if caller_owned else {}
        context_b = {"task_context": SearchTaskContext()} if caller_owned else {}
        first = pool.submit(agent.search_and_extract, "a", search_queries=["a1", "a2"], **context_a)
        try:
            assert a_first.wait(10)
            second = pool.submit(agent.search_and_extract, "b", search_queries=["b1", "b2"], **context_b)
            for future in (first, second):
                result = future.result(timeout=20)
                assert "A useful research excerpt" in result.to_context_string()
        finally:
            b_first.set()
            a_second.set()
    args = [c["params"]["arguments"] for c in tool_calls(mcp_http)]
    assert [a["search_queries"][0] for a in args] == ["a1", "b1", "a2", "b2"]
    ids = {a["search_queries"][0]: a["session_id"] for a in args}
    assert ids["a1"] == ids["a2"]
    assert ids["b1"] == ids["b2"]
    assert ids["a1"] != ids["b1"]
    assert all(client.is_closed for client in mcp_http["clients"])


@pytest.mark.parametrize("failure", ["error", "timeout", "cancel"])
@pytest.mark.parametrize("same_run", [False, True])
def test_reusable_agent_context_after_failed_invocation(mcp_http, monkeypatch, failure, same_run):
    agent = WebSearchAgent(search_provider="parallel", search_timeout_sec=0.1 if failure == "timeout" else 5,
                           enable_scholar=False, enable_crawling=False, enable_pdf=False)
    mcp_http["error"] = failure == "error"
    mcp_http["delay"] = 10 if failure in {"timeout", "cancel"} else 0
    if failure == "cancel":
        original = ParallelSearchClient._search

        async def cancel_first(self, query, *args, **kwargs):
            if query == "failed1":
                with anyio.move_on_after(0.1) as scope:
                    await original(self, query, *args, **kwargs)
                assert scope.cancel_called
                raise asyncio.CancelledError
            return await original(self, query, *args, **kwargs)

        monkeypatch.setattr(ParallelSearchClient, "_search", cancel_first)
    context_kwargs = {"task_context": SearchTaskContext()} if same_run else {}
    failed = agent.search_and_extract("failed", search_queries=["failed1", "failed2"], **context_kwargs)
    assert not failed.web_results
    mcp_http["error"] = False
    mcp_http["delay"] = 0
    result = agent.search_and_extract("new", search_queries=["new1", "new2"], **context_kwargs)
    assert "https://papers.example.org/article" in result.to_context_string()
    assert "A useful research excerpt" in result.to_context_string()
    args = [c["params"]["arguments"] for c in tool_calls(mcp_http)]
    assert [a["search_queries"][0] for a in args] == ["failed1", "new1", "new2"]
    assert (args[0]["session_id"] == args[1]["session_id"]) is same_run
    assert args[1]["session_id"] == args[2]["session_id"]
    assert all(client.is_closed for client in mcp_http["clients"])


def test_direct_web_client_omits_unowned_identity_and_reuses_batch_context(mcp_http):
    client = WebSearchClient(provider="parallel")
    client.search("default1")
    context = SearchTaskContext()
    batch = client.search_multi(["batch1", "batch2"], inter_query_delay=0, task_context=context)
    assert len(batch[0].results) == 2
    assert not batch[1].results  # Existing cross-query deduplication.
    client.search("batch3", task_context=context)
    client.search("independent", task_context=SearchTaskContext())
    client.search("default2")
    args = [c["params"]["arguments"] for c in tool_calls(mcp_http)]
    assert "session_id" not in args[0] and "session_id" not in args[5]
    assert [a["session_id"] for a in args[1:4]] == [context.session_id] * 3
    assert args[4]["session_id"] != context.session_id
    client.search_multi(["unowned1", "unowned2"], inter_query_delay=0)
    assert all("session_id" not in c["params"]["arguments"] for c in tool_calls(mcp_http)[6:])


def test_direct_parallel_client_omits_unowned_identity(mcp_http):
    client = ParallelSearchClient()
    context = SearchTaskContext()
    for query, task in [("default1", None), ("batch1", context),
                        ("batch2", context), ("default2", None)]:
        assert client.search(query, limit=1, task_context=task)[0].content.endswith("Further evidence")
    args = [c["params"]["arguments"] for c in tool_calls(mcp_http)]
    assert "session_id" not in args[0] and "session_id" not in args[3]
    assert args[1]["session_id"] == args[2]["session_id"] == context.session_id



def test_public_agent_same_run_followups_and_retries_reuse_context(mcp_http):
    from dataclasses import FrozenInstanceError

    agent = WebSearchAgent(search_provider="parallel", enable_scholar=False,
                           enable_crawling=False, enable_pdf=False)
    context = SearchTaskContext()
    for queries in (["initial1", "initial2"], ["followup"], ["followup"]):
        result = agent.search_and_extract("same research", search_queries=queries,
                                          task_context=context)
        assert "A useful research excerpt" in result.to_context_string()
    args = [c["params"]["arguments"] for c in tool_calls(mcp_http)]
    assert [a["session_id"] for a in args] == [context.session_id] * 4
    with pytest.raises(FrozenInstanceError):
        context.session_id = "changed"
    agent.search_and_extract("same research", search_queries=["reset"],
                             task_context=SearchTaskContext())
    assert tool_calls(mcp_http)[-1]["params"]["arguments"]["session_id"] != context.session_id

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
    from researchclaw.pipeline.executor import execute_stage
    from researchclaw.pipeline.stages import Stage, StageStatus
    (prior / "search_plan.yaml").write_text("queries: [research evidence]\n")
    identities = []
    for run_id in ("run-a", "run-a", "run-b"):
        result = execute_stage(Stage.LITERATURE_COLLECT, run_dir=run_dir, run_id=run_id,
                               config=config, adapters=AdapterBundle())
        assert result.status == StageStatus.DONE
        if enabled:
            identities.append(tool_calls(mcp_http)[-1]["params"]["arguments"]["session_id"])
    if enabled:
        assert identities[0] == identities[1]
        assert identities[0] != identities[2]
    context = stage_dir / "web_context.md"
    if enabled:
        assert "A useful research excerpt" in context.read_text()
        assert "https://papers.example.org/article" in context.read_text()
        metadata = json.loads((stage_dir / "web_search_result.json").read_text())
        assert metadata["web_results"][0]["source"] == "parallel"
    else:
        assert not context.exists()
        assert not mcp_http["requests"]
