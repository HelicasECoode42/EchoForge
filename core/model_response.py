"""Normalize and observe responses from Anthropic-compatible providers."""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class ModelResponseParseError(ValueError):
    """Raised when a successful provider response contains no usable text."""


def provider_extra_body() -> dict[str, Any] | None:
    """Return provider controls without leaking them into business code."""
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").lower()
    if "api.deepseek.com" not in base_url:
        return None
    mode = os.getenv("DEEPSEEK_THINKING_MODE", "disabled").strip().lower()
    if mode not in {"enabled", "disabled"}:
        raise ValueError("DEEPSEEK_THINKING_MODE must be enabled or disabled")
    return {"thinking": {"type": mode}}


async def create_message(client: Any, *, component: str, **kwargs: Any) -> Any:
    """Call a provider and record latency without logging prompts or secrets."""
    extra_body = provider_extra_body()
    if extra_body is not None and "extra_body" not in kwargs:
        kwargs["extra_body"] = extra_body
    started = time.monotonic()
    try:
        return await client.messages.create(**kwargs)
    finally:
        logger.info(
            "model_call component=%s model=%s thinking=%s latency_ms=%.1f",
            component,
            kwargs.get("model", "unknown"),
            (kwargs.get("extra_body") or {}).get("thinking", {}).get("type", "provider_default"),
            (time.monotonic() - started) * 1000,
        )


def extract_text(response: Any) -> str:
    """Extract non-empty text blocks from an Anthropic SDK response.

    Providers such as DeepSeek may return multiple content blocks, or a block
    whose ``text`` is null.  Business code should receive a validated string
    instead of failing later with a confusing ``NoneType`` exception.
    """
    content = getattr(response, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    result = "\n".join(parts).strip()
    if not result:
        raise ModelResponseParseError(
            f"model returned no text content: response_type={type(response).__name__}"
        )
    return result
