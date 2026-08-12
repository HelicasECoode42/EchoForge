from __future__ import annotations

import unittest

from evaluation.improvement import (
    EvaluationMetrics,
    FailureType,
    ImprovementProposal,
    ProposalStatus,
    ProposalTarget,
    build_evaluation,
    classify_failure,
    generate_proposals_from_traces,
)


class ImprovementDomainTests(unittest.TestCase):
    def test_proposal_has_stable_id_and_serializable_provenance(self):
        proposal = ImprovementProposal(
            proposal_version="chunking.v2",
            source_trace_ids=("route-123", "graph-456"),
            target=ProposalTarget.CHUNK_STRATEGY,
            parameters={"strategy": "structure_token", "max_tokens": 140},
            failure_types=(FailureType.NO_RECALL,),
        )

        payload = proposal.to_dict()
        self.assertTrue(proposal.proposal_id.startswith("proposal-"))
        self.assertEqual(payload["source_trace_ids"], ["route-123", "graph-456"])
        self.assertEqual(payload["status"], "proposed")
        self.assertEqual(payload["failure_types"], ["no_recall"])

    def test_production_bound_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            ImprovementProposal(
                proposal_version="unsafe.v1",
                source_trace_ids=("trace-1",),
                target=ProposalTarget.PROMPT_VERSION,
                parameters={"system_prompt": "change production prompt"},
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            ImprovementProposal(
                proposal_version="unsafe.v2",
                source_trace_ids=("trace-1",),
                target=ProposalTarget.RETRIEVAL_POLICY,
                parameters={"nested": [{"production_config": {"chunk_size": 10}}]},
            )
        with self.assertRaisesRegex(ValueError, "privacy-safe"):
            ImprovementProposal(
                proposal_version="unsafe.v3",
                source_trace_ids=("trace/with/raw/path",),
                target=ProposalTarget.RETRIEVAL_POLICY,
                parameters={"mode": "adaptive"},
            )
        with self.assertRaisesRegex(ValueError, "secret or prompt"):
            ImprovementProposal(
                proposal_version="unsafe.v4",
                source_trace_ids=("trace-safe",),
                target=ProposalTarget.RETRIEVAL_POLICY,
                parameters={"mode": "adaptive"},
                description="contains system prompt material",
            )

    def test_only_passing_offline_evaluation_becomes_candidate_and_approved(self):
        baseline = EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0)
        better = EvaluationMetrics(0.9, 0.9, 0.1, 90.0, 350.0)
        proposal = ImprovementProposal(
            proposal_version="retrieval.v2",
            source_trace_ids=("trace-1",),
            target=ProposalTarget.RETRIEVAL_POLICY,
            parameters={"mode": "adaptive"},
        )

        evaluated = proposal.with_evaluation(build_evaluation(baseline, better))
        self.assertEqual(evaluated.status, ProposalStatus.CANDIDATE)
        approved = evaluated.approve(approved_at="2026-08-12T00:00:00+00:00")
        self.assertEqual(approved.status, ProposalStatus.APPROVED)

        worse = EvaluationMetrics(0.9, 0.9, 0.1, 120.0, 350.0)
        rejected = proposal.with_evaluation(build_evaluation(baseline, worse))
        self.assertEqual(rejected.status, ProposalStatus.REJECTED)
        with self.assertRaisesRegex(ValueError, "offline regression"):
            rejected.approve()

    def test_no_op_proposal_is_not_a_candidate(self):
        metrics = EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0)
        proposal = ImprovementProposal(
            proposal_version="retrieval.no-op.v1",
            source_trace_ids=("trace-no-op",),
            target=ProposalTarget.RETRIEVAL_POLICY,
            parameters={"mode": "adaptive"},
        )

        evaluated = proposal.with_evaluation(build_evaluation(metrics, metrics))
        self.assertEqual(evaluated.status, ProposalStatus.NO_CHANGE)
        with self.assertRaisesRegex(ValueError, "only a proposal"):
            evaluated.approve()

    def test_approved_proposal_is_terminal(self):
        metrics = EvaluationMetrics(0.8, 0.8, 0.2, 100.0, 400.0)
        better = EvaluationMetrics(0.9, 0.8, 0.2, 100.0, 400.0)
        proposal = ImprovementProposal(
            proposal_version="retrieval.terminal.v1",
            source_trace_ids=("trace-terminal",),
            target=ProposalTarget.RETRIEVAL_POLICY,
            parameters={"mode": "adaptive"},
        )
        approved = proposal.with_evaluation(build_evaluation(metrics, better)).approve(
            approved_at="2026-08-12T00:00:00+00:00"
        )
        with self.assertRaisesRegex(ValueError, "only proposed"):
            approved.with_evaluation(build_evaluation(metrics, better))
        with self.assertRaisesRegex(ValueError, "terminal"):
            approved.reject()

    def test_failure_classifier_uses_privacy_safe_trace_signals(self):
        trace = {
            "status": "blocked",
            "stop_reason": "node_timeout",
            "verification_reason": "insufficient_evidence",
            "should_retrieve": True,
            "evidence_ids": [],
        }
        self.assertEqual(
            classify_failure(trace),
            (FailureType.NO_RECALL, FailureType.TIMEOUT),
        )

        citation_trace = {
            "status": "blocked",
            "stop_reason": "completed",
            "verification_reason": "grounding_failed",
            "evidence_ids": ["chunk-1"],
            "citations": ["chunk-outside-retrieval"],
        }
        self.assertEqual(classify_failure(citation_trace), (FailureType.INCORRECT_CITATION,))

    def test_trace_clusters_generate_versioned_review_proposals(self):
        proposals = generate_proposals_from_traces([
            {
                "trace_id": "trace-no-recall-1",
                "status": "blocked",
                "verification_reason": "insufficient_evidence",
                "should_retrieve": True,
                "evidence_ids": [],
            },
            {
                "trace_id": "trace-no-recall-2",
                "status": "blocked",
                "verification_reason": "insufficient_evidence",
                "should_retrieve": True,
                "evidence_ids": [],
            },
        ])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].failure_types, (FailureType.NO_RECALL,))
        self.assertEqual(proposals[0].source_trace_ids, ("trace-no-recall-1", "trace-no-recall-2"))
        self.assertEqual(proposals[0].status, ProposalStatus.PROPOSED)


if __name__ == "__main__":
    unittest.main()
