from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from evaluation.improvement import (
    EvaluationMetrics,
    FailureType,
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
            with self.assertRaisesRegex(ValueError, "proposal_id does not match"):
                replace(proposal, parameters={"strategy": "structure_token", "max_tokens": 160})

    def test_store_instances_share_path_lock_without_lost_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.json"
            proposals = tuple(
                ImprovementProposal(
                    proposal_version="chunking.concurrent.v1",
                    source_trace_ids=(f"trace-{index}",),
                    target=ProposalTarget.CHUNK_STRATEGY,
                    parameters={"strategy": "structure_token", "max_tokens": 140},
                )
                for index in range(2)
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda item: ProposalStore(path).save(item), proposals))
            self.assertEqual(
                {item.proposal_id for item in ProposalStore(path).list()},
                {item.proposal_id for item in proposals},
            )

    def test_concurrent_terminal_decisions_only_commit_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.json"
            proposal = ImprovementProposal(
                proposal_version="chunking.decision.v1",
                source_trace_ids=("trace-decision",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
                failure_types=(FailureType.NO_RECALL,),
            )
            candidate = proposal.with_evaluation(build_evaluation(
                EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0),
                EvaluationMetrics(0.9, 0.9, 0.1, 90.0, 350.0),
            ))
            ProposalStore(path).save(candidate)

            def decide(action):
                try:
                    action(ProposalStore(path))
                    return "committed"
                except ValueError:
                    return "conflict"

            actions = (
                lambda store: store.approve(candidate.proposal_id, approved_at="2026-08-12T00:00:00+00:00"),
                lambda store: store.reject(candidate.proposal_id),
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(decide, actions))
            self.assertEqual(outcomes.count("committed"), 1)
            self.assertEqual(outcomes.count("conflict"), 1)
            self.assertIn(
                ProposalStore(path).get(candidate.proposal_id).status,
                {ProposalStatus.APPROVED, ProposalStatus.REJECTED},
            )

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
