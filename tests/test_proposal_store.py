from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.improvement import (
    EvaluationMetrics,
    ImprovementProposal,
    ProposalStatus,
    ProposalTarget,
    EvaluationContext,
    build_evaluation,
)
from evaluation.proposal_store import ProposalStore


class ProposalStoreTests(unittest.TestCase):
    def test_store_round_trips_and_only_approves_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProposalStore(Path(directory) / "proposals.json")
            proposal = ImprovementProposal(
                proposal_version="chunking.v1",
                source_trace_ids=("trace-1",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
            )
            store.save(proposal)
            loaded = store.get(proposal.proposal_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.status, ProposalStatus.PROPOSED)
            with self.assertRaisesRegex(ValueError, "offline regression"):
                store.approve(proposal.proposal_id)

    def test_candidate_status_can_be_approved_without_apply_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProposalStore(Path(directory) / "proposals.json")
            proposal = ImprovementProposal(
                proposal_version="chunking.v1",
                source_trace_ids=("trace-1",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
            )
            candidate = proposal.with_evaluation(
                build_evaluation(
                    EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0, 0.8),
                    EvaluationMetrics(0.9, 0.9, 0.1, 90.0, 350.0, 0.9),
                )
            )
            store.save(candidate)
            approved = store.approve(candidate.proposal_id, approved_at="2026-08-12T00:00:00+00:00")
            self.assertEqual(approved.status, ProposalStatus.APPROVED)
            self.assertEqual(store.get(candidate.proposal_id).approved_at, "2026-08-12T00:00:00+00:00")
            with self.assertRaisesRegex(ValueError, "terminal"):
                store.reject(candidate.proposal_id)

    def test_ledger_rejects_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProposalStore(Path(directory) / "proposals.json")
            proposal = ImprovementProposal(
                proposal_version="chunking.v1",
                source_trace_ids=("trace-identity",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
            )
            store.save(proposal)
            replacement = ImprovementProposal(
                proposal_version="chunking.v1",
                source_trace_ids=("trace-identity",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 160},
                proposal_id=proposal.proposal_id,
            )
            with self.assertRaisesRegex(ValueError, "identity fields"):
                store.save(replacement)

    def test_re_evaluation_preserves_prior_evidence_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProposalStore(Path(directory) / "proposals.json")
            proposal = ImprovementProposal(
                proposal_version="chunking.audit.v1",
                source_trace_ids=("trace-audit",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
            )
            baseline = EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0, 0.8)
            improved = EvaluationMetrics(0.9, 0.9, 0.1, 90.0, 350.0, 0.9)
            first = proposal.with_evaluation(build_evaluation(
                baseline,
                improved,
                context=EvaluationContext(
                    dataset_id="dataset-a",
                    dataset_hash="hash-a",
                    baseline_config_hash="baseline-a",
                    top_k=3,
                ),
            ))
            store.save(first)

            second = proposal.with_evaluation(build_evaluation(
                baseline,
                improved,
                context=EvaluationContext(
                    dataset_id="dataset-b",
                    dataset_hash="hash-b",
                    baseline_config_hash="baseline-b",
                    top_k=5,
                ),
            ))
            persisted = store.save(second)
            loaded = store.get(proposal.proposal_id)
            assert loaded is not None
            self.assertEqual(len(persisted.evaluation_history), 2)
            self.assertEqual(len(loaded.evaluation_history), 2)
            self.assertEqual(loaded.evaluation_history[0].context.top_k, 3)
            self.assertEqual(loaded.evaluation_history[1].context.top_k, 5)


if __name__ == "__main__":
    unittest.main()
