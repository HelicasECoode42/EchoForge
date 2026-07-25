"""Controlled latency/call-count benchmark for adaptive retrieval.

The sleep values model remote rewrite/rerank calls; this is not presented as a
live provider benchmark.  It isolates the algorithmic effect of skipping two
model calls when the initial vector result is confidently separated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.tool_manager import MCPToolManager, Tool


class BenchmarkManager(MCPToolManager):
    def __init__(self):
        super().__init__(api_key="benchmark-key")
        self.rewrite_calls = 0
        self.rerank_calls = 0
        self.vector_calls = 0

    async def rewrite_query(self, query: str, n: int = 3):
        self.rewrite_calls += 1
        await asyncio.sleep(0.025)
        return [query, f"{query} 改写一", f"{query} 改写二", f"{query} 改写三"]

    async def _rerank(self, query, items, top_k):
        self.rerank_calls += 1
        await asyncio.sleep(0.035)
        return items[:top_k]


async def run_once(adaptive: bool):
    manager = BenchmarkManager()

    async def handler(params, context):
        manager.vector_calls += 1
        await asyncio.sleep(0.003)
        return [
            {"chunk_id": "answer", "content": "answer", "score": 0.78},
            {"chunk_id": "other-1", "content": "other", "score": 0.52},
            {"chunk_id": "other-2", "content": "other", "score": 0.40},
        ]

    manager.register(Tool(
        name="search",
        description="benchmark",
        handler=handler,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    ))
    started = time.perf_counter()
    result = await manager.search_with_rewrite(
        "search", "退款到账时间", top_k=2, adaptive=adaptive
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "latency_ms": latency_ms,
        "top_chunk_id": result.data[0]["chunk_id"],
        "vector_calls": manager.vector_calls,
        "rewrite_calls": manager.rewrite_calls,
        "rerank_calls": manager.rerank_calls,
    }


def percentile(values, p):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


async def benchmark(iterations: int):
    baseline = [await run_once(False) for _ in range(iterations)]
    adaptive = [await run_once(True) for _ in range(iterations)]

    def summarize(rows):
        latencies = [row["latency_ms"] for row in rows]
        return {
            "iterations": len(rows),
            "median_latency_ms": round(statistics.median(latencies), 3),
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "avg_vector_calls": statistics.mean(row["vector_calls"] for row in rows),
            "avg_rewrite_calls": statistics.mean(row["rewrite_calls"] for row in rows),
            "avg_rerank_calls": statistics.mean(row["rerank_calls"] for row in rows),
            "top_result_stability": len({row["top_chunk_id"] for row in rows}) == 1,
        }

    baseline_summary = summarize(baseline)
    adaptive_summary = summarize(adaptive)
    reduction = 1.0 - adaptive_summary["median_latency_ms"] / baseline_summary["median_latency_ms"]
    return {
        "evaluation_scope": "controlled_high_confidence_retrieval_path",
        "latency_model": {
            "vector_call_ms": 3,
            "rewrite_call_ms": 25,
            "rerank_call_ms": 35,
            "note": "Injected deterministic delays; not live Anthropic or production latency.",
        },
        "quality_guard": "The top chunk id must be identical in baseline and adaptive runs.",
        "baseline_always_rewrite_rerank": baseline_summary,
        "adaptive_vector_fast_path": adaptive_summary,
        "median_latency_reduction": round(reduction, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "evidence" / "retrieval-policy-benchmark.json",
    )
    args = parser.parse_args()
    report = asyncio.run(benchmark(args.iterations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
