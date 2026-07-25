from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from core.execution_graph import (
    EdgeSpec,
    ExecutionGraph,
    GraphBudget,
    GraphValidationError,
    NodeSpec,
)
from evaluation.graph_replay import load_graph_cases, run_graph_suite


ROOT = Path(__file__).resolve().parents[1]


class ExecutionGraphTests(unittest.TestCase):
    def test_production_path_replay_suite_passes(self):
        cases = load_graph_cases(ROOT / "data" / "replay" / "graph_cases.json")
        report = asyncio.run(run_graph_suite(cases))
        self.assertEqual(report["total"], 5)
        self.assertEqual(report["passed"], 5)
        self.assertEqual(report["failed"], 0)

    def test_conditional_path_is_traced(self):
        async def mark(state):
            return {"visited": state.get("visited", []) + ["start"]}

        graph = ExecutionGraph(
            name="conditional",
            nodes=[
                NodeSpec("start", mark),
                NodeSpec("retrieve", lambda state: {"retrieved": True}),
                NodeSpec("skip", lambda state: {"retrieved": False}),
                NodeSpec("done", lambda state: None),
            ],
            edges=[
                EdgeSpec("start", "retrieve", lambda state: state["use_retrieval"], "retrieve"),
                EdgeSpec("start", "skip", lambda state: not state["use_retrieval"], "skip"),
                EdgeSpec("retrieve", "done"),
                EdgeSpec("skip", "done"),
            ],
            start_node="start",
            terminal_nodes={"done"},
        )

        result = asyncio.run(graph.run({"use_retrieval": True}))
        self.assertEqual(result.trace.stop_reason, "completed")
        self.assertTrue(result.state["retrieved"])
        self.assertEqual(
            [(step.source, step.target) for step in result.trace.transitions],
            [("start", "retrieve"), ("retrieve", "done")],
        )

    def test_retry_is_bounded_and_recorded(self):
        attempts = {"count": 0}

        def flaky(state):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        graph = ExecutionGraph(
            name="retry",
            nodes=[NodeSpec("work", flaky, max_retries=1)],
            edges=[],
            start_node="work",
            terminal_nodes={"work"},
        )
        result = asyncio.run(graph.run({}))
        self.assertEqual(result.trace.stop_reason, "completed")
        self.assertEqual([run.status for run in result.trace.node_runs], ["failed", "completed"])

    def test_cycle_stops_at_transition_budget(self):
        graph = ExecutionGraph(
            name="cycle",
            nodes=[
                NodeSpec("a", lambda state: None),
                NodeSpec("b", lambda state: None),
                NodeSpec("done", lambda state: None),
            ],
            edges=[
                EdgeSpec("a", "b"),
                EdgeSpec("b", "a"),
                EdgeSpec("b", "done", lambda state: False),
            ],
            start_node="a",
            terminal_nodes={"done"},
            budget=GraphBudget(max_node_executions=8, max_transitions=3, max_total_runtime_ms=1000),
        )
        result = asyncio.run(graph.run({}))
        self.assertEqual(result.trace.stop_reason, "transition_budget_exhausted")
        self.assertEqual(len(result.trace.transitions), 3)

    def test_validation_rejects_unreachable_node(self):
        with self.assertRaisesRegex(GraphValidationError, "unreachable"):
            ExecutionGraph(
                name="invalid",
                nodes=[
                    NodeSpec("start", lambda state: None),
                    NodeSpec("done", lambda state: None),
                    NodeSpec("orphan", lambda state: None),
                ],
                edges=[EdgeSpec("start", "done")],
                start_node="start",
                terminal_nodes={"done"},
            )


if __name__ == "__main__":
    unittest.main()
