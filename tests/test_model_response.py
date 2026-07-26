from types import SimpleNamespace

import pytest

from core.model_response import ModelResponseParseError, create_message, extract_text, provider_extra_body


def test_extract_text_combines_non_empty_blocks():
    response = SimpleNamespace(content=[
        SimpleNamespace(text=None),
        SimpleNamespace(text=" 第一段 "),
        SimpleNamespace(text="第二段"),
    ])
    assert extract_text(response) == "第一段\n第二段"


def test_extract_text_rejects_empty_response():
    response = SimpleNamespace(content=[SimpleNamespace(text=None)])
    with pytest.raises(ModelResponseParseError, match="no text content"):
        extract_text(response)


def test_deepseek_thinking_is_disabled_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.delenv("DEEPSEEK_THINKING_MODE", raising=False)
    assert provider_extra_body() == {"thinking": {"type": "disabled"}}


def test_create_message_injects_deepseek_thinking_control(monkeypatch):
    import asyncio

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    captured = {}

    class Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    client = SimpleNamespace(messages=Messages())
    asyncio.run(create_message(client, component="test", model="deepseek-v4-pro", max_tokens=8, messages=[]))
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
