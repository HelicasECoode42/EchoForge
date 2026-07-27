"""Deterministic replay for EchoForge graph paths; no model or datastore needed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from core.execution_graph import EdgeSpec, ExecutionGraph, GraphBudget, NodeSpec


@dataclass(frozen=True)
class GraphReplayResult:
    case_id: str
    passed: bool
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    trace: Dict[str, Any]


def load_graph_cases(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cases"] if isinstance(payload, dict) else payload


def _build_replay_graph(case: Dict[str, Any]) -> ExecutionGraph:
    outcomes = {name: list(values) for name, values in case.get("node_outcomes", {}).items()}
    calls: Dict[str, int] = {}

    def handler(name: str):
        async def run(state: Dict[str, Any]):
            index = calls.get(name, 0)
            calls[name] = index + 1
            planned = outcomes.get(name) or [True]
            success = planned[index] if index < len(planned) else planned[-1]
            if not success:
                raise RuntimeError(f"deterministic_{name}_failure")
            if name == "decide_retrieval":
                return {"should_retrieve": bool(case.get("use_retrieval", False))}
            return {f"{name}_done": True}
        return run

    raw_budget = case.get("budget", {})
    return ExecutionGraph(
        name="echoforge_chat_pipeline_replay",
        nodes=[
            NodeSpec("load_memory", handler("load_memory")),
            NodeSpec("decide_retrieval", handler("decide_retrieval")),
            NodeSpec("retrieve", handler("retrieve"), max_retries=1),
            NodeSpec("execute_agent", handler("execute_agent")),
            NodeSpec("persist_memory", handler("persist_memory")),
            NodeSpec("complete", handler("complete")),
        ],
        edges=[
            EdgeSpec("load_memory", "decide_retrieval", label="memory_loaded"),
            EdgeSpec("decide_retrieval", "retrieve", lambda state: state["should_retrieve"], "retrieval_required"),
            EdgeSpec("decide_retrieval", "execute_agent", lambda state: not state["should_retrieve"], "retrieval_skipped"),
            EdgeSpec("retrieve", "execute_agent", label="context_ready"),
            EdgeSpec("execute_agent", "persist_memory", label="agent_completed"),
            EdgeSpec("persist_memory", "complete", label="memory_persisted"),
        ],
        start_node="load_memory",
        terminal_nodes={"complete"},
        budget=GraphBudget(
            max_node_executions=int(raw_budget.get("max_node_executions", 8)),
            max_transitions=int(raw_budget.get("max_transitions", 7)),
            max_total_runtime_ms=float(raw_budget.get("max_total_runtime_ms", 1000)),
        ),
    )


async def run_graph_case(case: Dict[str, Any]) -> GraphReplayResult:
    result = await _build_replay_graph(case).run({})
    actual = {
        "stop_reason": result.trace.stop_reason,
        "path": [run.node for run in result.trace.node_runs if run.status == "completed"],
        "transitions": len(result.trace.transitions),
        "failed_nodes": [run.node for run in result.trace.node_runs if run.status != "completed"],
    }
    expected = case["expected"]
    passed = all(actual.get(key) == value for key, value in expected.items())
    return GraphReplayResult(
        case_id=case["id"],
        passed=passed,
        expected=expected,
        actual=actual,
        trace=result.trace.to_dict(),
    )


async def run_graph_suite(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = [await run_graph_case(case) for case in cases]
    passed = sum(1 for result in results if result.passed)
    return {
        "scope": "deterministic_graph_path_replay",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": [
            {
                "case_id": result.case_id,
                "passed": result.passed,
                "expected": result.expected,
                "actual": result.actual,
                "trace": result.trace,
            }
            for result in results
        ],
    }
