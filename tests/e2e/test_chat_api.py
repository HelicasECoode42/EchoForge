from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest

import api.main as main
from agents.agent_orchestrator import AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory
from memory.conversation_memory import MemoryContext


@dataclass
class FakeMemory:
    writes: list[tuple[str, str]]

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        return MemoryContext([], [], {}, "")

    async def add_message(self, user_id, conv_id, role, content, metadata=None):
        self.writes.append((role.value, content))

    async def update_profile(self, user_id, conv_id):
        return None


class FakeOrchestrator:
    async def run(self, request, progress_sink=None):
        trace = SimpleNamespace(
            trace_id=f"route-{request.request_id}",
            attempts=1,
            reroutes=0,
            stop_reason="completed",
            success=True,
            total_latency_ms=1.0,
        )
        return OrchestratorResult(
            request_id=request.request_id,
            response="你好，我可以帮助你。" if not request.evidence_ids else "退款通常需要 3-5 个工作日。",
            agent_type=AgentType.GENERAL,
            intent=IntentCategory.GREETING,
            success=True,
            route_trace=trace,
            confidence=0.95,
        )


@pytest.fixture
def isolated_app(monkeypatch):
    memory = FakeMemory([])
    monkeypatch.setattr(main, "_memory", memory)
    monkeypatch.setattr(main, "_orchestrator", FakeOrchestrator())
    monkeypatch.setattr(main, "_tool_manager", None)
    monkeypatch.setattr(main, "_trace_store", None)
    monkeypatch.setattr(main, "_graph_trace_store", None)
    monkeypatch.setattr(main, "_chat_graph", main._create_chat_graph(None))
    return memory


@pytest.mark.anyio
async def test_http_chat_completes_and_persists_verified_response(isolated_app):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        graph_response = await client.get("/graph")
        chat_response = await client.post("/chat", json={"message": "你好", "user_id": "e2e"})

    assert graph_response.status_code == 200
    graph = graph_response.json()
    assert {"complete", "blocked", "failed"}.issubset(set(graph["terminal_nodes"]))
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["status"] == "completed"
    assert payload["execution_status"] == "verified"
    assert len(isolated_app.writes) == 2


@pytest.mark.anyio
async def test_http_chat_blocks_retrieval_answer_without_evidence(isolated_app):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "退款多久到账？", "user_id": "e2e"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["verification_reason"] == "insufficient_evidence"
    assert payload["grounding_status"] == "blocked"
    assert isolated_app.writes == []
