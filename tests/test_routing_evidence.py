from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.route_replay import load_replay_cases, run_replay_case, run_replay_suite
from evidence.route_trace import JsonlRouteTraceStore, RouteTrace, RoutingBudget


ROOT = Path(__file__).resolve().parents[1]


class RouteReplayTests(unittest.TestCase):
    def test_sample_replay_suite_passes(self):
        cases = load_replay_cases(ROOT / "data" / "replay" / "route_cases.json")
        report = asyncio.run(run_replay_suite(cases))
        self.assertEqual(report["total"], 5)
        self.assertEqual(report["passed"], 5)
        self.assertEqual(report["failed"], 0)

    def test_failure_reroutes_once_and_preserves_evidence(self):
        case = {
            "id": "one-fallback",
            "message": "退款失败",
            "intent": "billing",
            "urgency": "low",
            "agent_outcomes": {"billing": [False], "general": [True]},
            "expected": {
                "final_agent": "general",
                "success": True,
                "attempts": 2,
                "reroutes": 1,
                "stop_reason": "completed",
            },
        }
        result = asyncio.run(run_replay_case(case))
        self.assertTrue(result.passed)
        stages = [step["stage"] for step in result.trace["steps"]]
        self.assertEqual(stages, ["attempt", "reroute", "attempt"])


class TraceStoreTests(unittest.TestCase):
    def test_trace_store_never_persists_raw_message(self):
        raw_message = "银行卡尾号 1234 的扣款有问题"
        trace = RouteTrace.start(
            request_id="privacy",
            message=raw_message,
            intent="billing",
            urgency="low",
            budget=RoutingBudget(),
        )
        trace.stop_reason = "completed"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traces.jsonl"
            store = JsonlRouteTraceStore(path)
            store.append(trace)
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn(raw_message, payload)
            self.assertIn(trace.message_sha256, payload)
            self.assertEqual(store.recent(1)[0]["request_id"], "privacy")

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            RoutingBudget(max_attempts=0)


if __name__ == "__main__":
    unittest.main()
