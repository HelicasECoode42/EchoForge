from __future__ import annotations

import asyncio

from mcp.tool_manager import MCPToolManager, Tool


class InstrumentedManager(MCPToolManager):
    def __init__(self):
        super().__init__(api_key="test-key")
        self.rewrite_count = 0
        self.rerank_count = 0

    async def rewrite_query(self, query: str, n: int = 3):
        self.rewrite_count += 1
        await asyncio.sleep(0.005)
        return [query, f"{query} 改写一", f"{query} 改写二"]

    async def _rerank(self, query, items, top_k):
        self.rerank_count += 1
        await asyncio.sleep(0.005)
        return items[:top_k]


def _tool(handler):
    return Tool(
        name="search",
        description="test",
        handler=handler,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    )


def test_high_confidence_vector_result_skips_two_llm_stages():
    calls = []

    async def handler(params, context):
        calls.append(params["query"])
        return [
            {"chunk_id": "a", "content": "answer", "score": 0.78},
            {"chunk_id": "b", "content": "other", "score": 0.52},
            {"chunk_id": "c", "content": "other", "score": 0.40},
        ]

    manager = InstrumentedManager()
    manager.register(_tool(handler))
    result = asyncio.run(manager.search_with_rewrite("search", "query", top_k=2))

    assert result.success
    assert result.diagnostics["path"] == "vector_fast_path"
    assert calls == ["query"]
    assert manager.rewrite_count == 0
    assert manager.rerank_count == 0


def test_low_confidence_result_keeps_rewrite_and_rerank_quality_path():
    async def handler(params, context):
        suffix = params["query"]
        return [
            {"chunk_id": f"{suffix}-a", "content": "candidate", "score": 0.55},
            {"chunk_id": f"{suffix}-b", "content": "candidate", "score": 0.53},
        ]

    manager = InstrumentedManager()
    manager.register(_tool(handler))
    result = asyncio.run(manager.search_with_rewrite("search", "query", top_k=2))

    assert result.success
    assert result.diagnostics["path"] == "rewrite_and_rerank"
    assert result.diagnostics["vector_calls"] == 3
    assert manager.rewrite_count == 1
    assert manager.rerank_count == 1
