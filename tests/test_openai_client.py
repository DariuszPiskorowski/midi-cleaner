from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from midi_cleaner.ai_completion.openai_client import (
    OpenAIPatternCompletionClient,
    OpenAIPatternCompletionClientError,
    calculate_max_output_tokens,
)


def _install_fake_openai(monkeypatch, create_callback):
    class _OpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = SimpleNamespace(create=create_callback)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))


def test_complete_pattern_uses_input_text_content_type(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text=json.dumps({"notes": []}))

    _install_fake_openai(monkeypatch, _create)

    client = OpenAIPatternCompletionClient()
    result = client.complete_pattern(
        api_key="sk-test",
        model="gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        temperature=0.2,
        max_completion_notes=64,
    )

    assert result.raw_response_text == json.dumps({"notes": []})
    assert result.parsed_payload == {"notes": []}
    assert result.max_output_tokens_used == 12000

    payload = captured["input"]
    assert isinstance(payload, list)
    assert [entry["content"][0]["type"] for entry in payload] == ["input_text", "input_text"]
    assert captured["max_output_tokens"] == calculate_max_output_tokens(64)


def test_complete_pattern_fallback_uses_input_text_content_type(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _create(**kwargs):
        calls.append(kwargs)
        if "text" in kwargs:
            raise TypeError("text format argument unsupported")
        return SimpleNamespace(output_text=json.dumps({"notes": []}))

    _install_fake_openai(monkeypatch, _create)

    client = OpenAIPatternCompletionClient()
    result = client.complete_pattern(
        api_key="sk-test",
        model="gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        temperature=0.2,
        max_completion_notes=64,
    )

    assert result.raw_response_text == json.dumps({"notes": []})
    assert result.parsed_payload == {"notes": []}
    assert result.max_output_tokens_used == 12000
    assert len(calls) == 2
    for call in calls:
        payload = call["input"]
        assert isinstance(payload, list)
        assert [entry["content"][0]["type"] for entry in payload] == [
            "input_text",
            "input_text",
        ]
        assert call["max_output_tokens"] == calculate_max_output_tokens(64)


def test_complete_pattern_wraps_and_sanitizes_openai_errors(monkeypatch) -> None:
    class _FakeAPIError(Exception):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.status_code = status_code

    def _create(**kwargs):
        _ = kwargs
        raise _FakeAPIError(
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
            "sk-secret-token-12345***************************************Ugjn. "
            "You can find your API key at https://platform.openai.com/account/api-keys.'}}",
            status_code=400,
        )

    _install_fake_openai(monkeypatch, _create)

    client = OpenAIPatternCompletionClient()

    with pytest.raises(OpenAIPatternCompletionClientError) as exc_info:
        client.complete_pattern(
            api_key="sk-test",
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=0.2,
            max_completion_notes=64,
        )

    message = str(exc_info.value)
    assert "OpenAI Responses API request failed (HTTP 400)" in message
    assert "Incorrect API key provided: [REDACTED]" in message
    assert "sk-secret-token-12345" not in message
    assert "Ugjn" not in message
    assert "sk-test" not in message
