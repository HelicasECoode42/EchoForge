"""Deterministic route replay that never calls an external model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from evidence.route_trace import RoutingBudget


class ScriptedAgent:
    """Minimal deterministic agent used only by route replay."""

    def __init__(self, agent_type: AgentType, outcomes: Iterable[bool]):
        self.agent_type = agent_type
        self._outcomes = list(outcomes) or [True]
        self._index = 0
        self.stats = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        success = self._outcomes[min(self._index, len(self._outcomes) - 1)]
        self._index += 1
        latency_ms = 5.0
        self.stats.total += 1
        self.stats.success += int(success)
        self.stats.total_ms += latency_ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=f"scripted:{self.agent_type.value}:{'ok' if success else 'failed'}",
            success=success,
            latency_ms=latency_ms,
        )


@dataclass
class ReplayResult:
    case_id: str
    passed: bool
    expected: Dict[str, Any]
    observed: Dict[str, Any]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "expected": self.expected,
            "observed": self.observed,
            "trace": self.trace,
        }


async def run_replay_case(case: Dict[str, Any]) -> ReplayResult:
    pool: Dict[AgentType, List[ScriptedAgent]] = {}
    for agent_name, outcomes in case.get("agent_outcomes", {}).items():
        agent_type = AgentType(agent_name)
        pool[agent_type] = [ScriptedAgent(agent_type, outcomes)]

    budget = RoutingBudget(**case.get("budget", {}))
    orchestrator = AgentOrchestrator(
        api_key="replay-only",
        agent_pool=pool,
        default_budget=budget,
    )
    request = Request(
        message=case.get("message", case["id"]),
        user_id="replay",
        conv_id=f"replay-{case['id']}",
        intent=IntentCategory(case.get("intent", "other")),
        urgency=UrgencyLevel[case.get("urgency", "LOW").upper()],
        request_id=f"replay-{case['id']}",
    )
    result = await orchestrator.run(request)
    trace = result.route_trace
    if trace is None:
        raise RuntimeError("route replay did not produce evidence")

    observed = {
        "final_agent": result.agent_type.value,
        "success": trace.success,
        "attempts": trace.attempts,
        "reroutes": trace.reroutes,
        "stop_reason": trace.stop_reason,
        "escalated": result.escalated,
    }
    expected = case["expected"]
    passed = all(observed.get(key) == value for key, value in expected.items())
    return ReplayResult(
        case_id=case["id"],
        passed=passed,
        expected=expected,
        observed=observed,
        trace=trace.to_dict(),
    )


async def run_replay_suite(cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results = [await run_replay_case(case) for case in cases]
    passed = sum(result.passed for result in results)
    return {
        "mode": "deterministic-route-replay",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": [result.to_dict() for result in results],
    }


def load_replay_cases(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("replay case file must contain a JSON array")
    return data
