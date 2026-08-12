from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.improvement import FailureType, ImprovementProposal, ProposalStatus, ProposalTarget
from evaluation.improvement_harness import (
    load_replay_dataset,
    replay_chunk_proposal,
    summarize_case_deltas,
)


ROOT = Path(__file__).resolve().parents[1]


class ImprovementHarnessTests(unittest.TestCase):
    def test_structure_chunk_proposal_passes_against_fixed_baseline(self):
        dataset = load_replay_dataset(ROOT / "data" / "chunking" / "cases.json")
        proposal = ImprovementProposal(
            proposal_version="chunking.structure-token.v1",
            source_trace_ids=("route-no-recall-1",),
            target=ProposalTarget.CHUNK_STRATEGY,
            parameters={"strategy": "structure_token", "max_tokens": 140},
            failure_types=(FailureType.NO_RECALL,),
        )

        report = replay_chunk_proposal(
            proposal,
            baseline_parameters={"strategy": "fixed_char", "chunk_size": 220},
            dataset=dataset,
            dataset_id="chunking-cases",
        )

        self.assertEqual(report.proposal.status, ProposalStatus.CANDIDATE)
        self.assertTrue(report.evaluation.passed_regression)
        self.assertEqual(report.evaluation.context.dataset_id, "chunking-cases")
        self.assertEqual(len(report.evaluation.context.dataset_hash), 64)
        self.assertEqual(len(report.evaluation.context.baseline_config_hash), 64)
        self.assertEqual(report.evaluation.context.top_k, 3)
        self.assertEqual(report.evaluation.context.adapter_version, "chunking-adapter.v1")
        self.assertEqual(report.evaluation.context.evaluator_version, "improvement-evaluator.v1")
        self.assertGreater(
            report.evaluation.comparisons["evidence_coverage"].proposal,
            report.evaluation.comparisons["evidence_coverage"].baseline,
        )
        self.assertLess(
            report.evaluation.comparisons["token_cost"].proposal,
            report.evaluation.comparisons["token_cost"].baseline,
        )

    def test_sliding_window_cost_regression_is_rejected(self):
        dataset = load_replay_dataset(ROOT / "data" / "chunking" / "cases.json")
        proposal = ImprovementProposal(
            proposal_version="chunking.sliding-window.v1",
            source_trace_ids=("route-cost-regression-1",),
            target=ProposalTarget.CHUNK_STRATEGY,
            parameters={"strategy": "sliding_window", "chunk_size": 220, "overlap": 60},
        )

        report = replay_chunk_proposal(
            proposal,
            baseline_parameters={"strategy": "structure_token", "max_tokens": 140},
            dataset=dataset,
            dataset_id="chunking-cases",
        )

        self.assertEqual(report.proposal.status, ProposalStatus.REJECTED)
        self.assertIn("token_cost", report.evaluation.reason)

    def test_report_contains_improved_and_regressed_case_evidence(self):
        dataset = load_replay_dataset(ROOT / "data" / "chunking" / "cases.json")
        better = replay_chunk_proposal(
            ImprovementProposal(
                proposal_version="chunking.structure-token.v1",
                source_trace_ids=("trace-better",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
            ),
            baseline_parameters={"strategy": "fixed_char", "chunk_size": 220},
            dataset=dataset,
            dataset_id="chunking-cases",
        )
        worse = replay_chunk_proposal(
            ImprovementProposal(
                proposal_version="chunking.sliding-window.v1",
                source_trace_ids=("trace-worse",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "sliding_window", "chunk_size": 220, "overlap": 60},
            ),
            baseline_parameters={"strategy": "structure_token", "max_tokens": 140},
            dataset=dataset,
            dataset_id="chunking-cases",
        )

        summary = summarize_case_deltas((better, worse))
        self.assertGreater(summary["improved_case_count"], 0)
        self.assertGreater(summary["regressed_case_count"], 0)


if __name__ == "__main__":
    unittest.main()
