from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.chunking_harness import load_dataset, run_harness
from retrieval.chunking import FixedCharacterChunker, SlidingWindowChunker, StructureAwareTokenChunker


ROOT = Path(__file__).resolve().parents[1]


class ChunkingTests(unittest.TestCase):
    def test_structure_chunker_respects_token_budget_and_links_neighbors(self):
        text = "# 任务规则\n" + "旅行者完成委托后获得声望。" * 40
        chunks = StructureAwareTokenChunker(max_tokens=32).chunk(
            document_id="rules",
            title="规则",
            text=text,
            document_version="7",
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 32 for chunk in chunks))
        self.assertEqual(chunks[0].document_version, "7")
        self.assertEqual(chunks[0].section_path, "任务规则")
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].chunk_id)
        self.assertEqual(chunks[1].previous_chunk_id, chunks[0].chunk_id)

    def test_sliding_window_preserves_boundary_context(self):
        chunks = SlidingWindowChunker(chunk_size=10, overlap=3).chunk(
            document_id="d",
            title="t",
            text="abcdefghijklmnopqrstuvwxyz",
        )
        self.assertEqual(chunks[0].content[-3:], chunks[1].content[:3])

    def test_harness_reports_quality_cost_and_latency(self):
        dataset = load_dataset(ROOT / "data" / "chunking" / "cases.json")
        report = run_harness(
            [
                FixedCharacterChunker(chunk_size=220),
                SlidingWindowChunker(chunk_size=220, overlap=60),
                StructureAwareTokenChunker(max_tokens=140),
            ],
            dataset,
            top_k=3,
        )
        self.assertEqual(len(report["strategies"]), 3)
        self.assertIn(report["winner"], {"fixed_char", "sliding_window", "structure_token"})
        for strategy in report["strategies"]:
            self.assertIn("recall_at_k", strategy)
            self.assertIn("mrr", strategy)
            self.assertIn("evidence_coverage", strategy)
            self.assertIn("avg_context_tokens", strategy)
            self.assertIn("avg_retrieval_latency_ms", strategy)


if __name__ == "__main__":
    unittest.main()
