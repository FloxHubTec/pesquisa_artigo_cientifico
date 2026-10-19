"""Capture serialized MiniMax requests without sending paid API calls."""

from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace

import httpx
import pytest

from researchclaw.cli import _PROVIDER_MODELS
from researchclaw.llm.client import LLMClient


_PROFILES = [
    ("minimax-global", "https://api.minimax.io/v1/chat/completions"),
    ("minimax", "https://api.minimaxi.com/v1/chat/completions"),
    ("minimax-anthropic", "https://api.minimax.io/anthropic/v1/messages"),
    ("minimax-anthropic-cn", "https://api.minimaxi.com/anthropic/v1/messages"),
]


def _config(provider, primary, fallbacks):
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            base_url="",
            api_key="test-key",
            api_key_env="",
            primary_model=primary,
            fallback_models=fallbacks,
        ),
    )


@pytest.mark.parametrize(("provider", "url"), _PROFILES)
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_current_models_preserve_api_thinking_defaults(provider, url, model, monkeypatch):
    primary, fallbacks = _PROVIDER_MODELS[provider]
    client = LLMClient.from_rc_config(_config(provider, primary, fallbacks))
    assert client._model_chain == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    captured = {}

    def capture_anthropic(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "type": "message",
            "model": model,
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "pong"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "model": model,
                "choices": [{
                    "message": {"content": "pong", "reasoning_content": "private reasoning"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }).encode()

    def capture_openai(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", capture_openai)
    if client._anthropic is not None:
        client._anthropic._client = httpx.Client(transport=httpx.MockTransport(capture_anthropic))
    try:
        response = client.chat(
            [{"role": "user", "content": "ping"}],
            model=model,
            max_tokens=1024,
            system="Answer briefly.",
        )
    finally:
        if client._anthropic is not None:
            client._anthropic.close()

    assert captured["url"] == url
    body = captured["body"]
    assert body["model"] == model
    assert body["max_tokens"] == 1024
    assert response.content == "pong"
    assert response.model == model
    # Neither config nor chat exposes thinking/effort controls. Omission keeps
    # Flash adaptive (default effort max); M3 uses each protocol's API default.
    for field in ("thinking", "reasoning_effort", "output_config", "service_tier", "reasoning_split"):
        assert field not in body
    if "anthropic" in provider:
        assert body["system"] == "Answer briefly."
        assert captured["headers"]["x-api-key"] == "test-key"
    else:
        assert captured["headers"]["authorization"] == "Bearer test-key"


@pytest.mark.parametrize(("provider", "url"), _PROFILES)
@pytest.mark.parametrize("fallbacks", [[], ["MiniMax-M2.7", "custom-fallback"]])
def test_explicit_saved_model_chain_is_unchanged(provider, url, fallbacks):
    client = LLMClient.from_rc_config(_config(provider, "custom-primary", fallbacks))
    assert client.config.primary_model == "custom-primary"
    assert client.config.fallback_models == fallbacks
    assert client._model_chain == ["custom-primary", *fallbacks]
