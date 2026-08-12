"""Deterministic replay harness for offline improvement proposals.

The first supported proposal surface is document chunking because the
repository already has a deterministic chunking evaluator.  Keeping the
adapter explicit is intentional: a proposal for rewrite/rerank or prompt
versions must not be presented as evaluated until a corresponding isolated
harness exists.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.chunking_harness import evaluate_chunker, load_dataset
from evaluation.improvement import (
    CaseEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    FailureType,
    ImprovementProposal,
    ProposalEvaluation,
    ProposalTarget,
    build_evaluation,
    compare_metrics,
)
from retrieval.chunking import (
    Chunker,
    FixedCharacterChunker,
    SlidingWindowChunker,
    StructureAwareTokenChunker,
)


class UnsupportedReplayTarget(ValueError):
    """Raised when no isolated replay adapter exists for a proposal target."""


ADAPTER_VERSION = "chunking-adapter.v1"
EVALUATOR_VERSION = "improvement-evaluator.v1"


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    configured = os.getenv("GIT_REVISION", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class ReplayReport:
    """Serializable result of one baseline/proposal offline comparison."""

    dataset_id: str
    evaluation_scope: str
    baseline_parameters: Mapping[str, Any]
    proposal: ImprovementProposal
    evaluation: ProposalEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "evaluation_scope": self.evaluation_scope,
            "baseline_parameters": dict(self.baseline_parameters),
            "proposal": self.proposal.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def build_chunker(parameters: Mapping[str, Any]) -> Chunker:
    """Build one chunker from an allow-listed offline parameter payload."""
    strategy = str(parameters.get("strategy", "")).strip()
    if strategy == "fixed_char":
        return FixedCharacterChunker(chunk_size=int(parameters.get("chunk_size", 500)))
    if strategy == "sliding_window":
        return SlidingWindowChunker(
            chunk_size=int(parameters.get("chunk_size", 500)),
            overlap=int(parameters.get("overlap", 100)),
        )
    if strategy == "structure_token":
        return StructureAwareTokenChunker(max_tokens=int(parameters.get("max_tokens", 320)))
    raise UnsupportedReplayTarget(f"unsupported chunk strategy: {strategy or '<empty>'}")


def metrics_from_chunk_report(report: Mapping[str, Any]) -> EvaluationMetrics:
    """Map existing chunking metrics to the Issue #4 regression contract.

    A case with zero evidence coverage is considered blocked for this offline
    retrieval gate.  Partial coverage remains visible as a quality metric but
    is not silently converted into a blocked response.
    """
    cases = list(report.get("cases", []))
    blocked_count = sum(float(case.get("evidence_coverage", 0.0)) <= 0.0 for case in cases)
    total = len(cases)
    return EvaluationMetrics(
        recall_at_k=float(report.get("recall_at_k", 0.0)),
        mrr=float(report.get("mrr", 0.0)),
        evidence_coverage=float(report.get("evidence_coverage", 0.0)),
        blocked_rate=blocked_count / total if total else 1.0,
        latency_ms=float(report.get("avg_retrieval_latency_ms", 0.0)),
        token_cost=float(report.get("avg_context_tokens", 0.0)),
    )


def _case_metrics(case: Mapping[str, Any]) -> EvaluationMetrics:
    coverage = float(case.get("evidence_coverage", 0.0))
    return EvaluationMetrics(
        recall_at_k=float(case.get("recall_at_k", 0.0)),
        mrr=float(case.get("reciprocal_rank", 0.0)),
        evidence_coverage=coverage,
        blocked_rate=1.0 if coverage <= 0.0 else 0.0,
        latency_ms=float(case.get("retrieval_latency_ms", 0.0)),
        token_cost=float(case.get("context_tokens", 0.0)),
    )


def _case_evaluations(
    baseline_report: Mapping[str, Any],
    proposal_report: Mapping[str, Any],
    *,
    latency_tolerance_ms: float,
) -> tuple[CaseEvaluation, ...]:
    baseline_cases = {str(item["case_id"]): item for item in baseline_report.get("cases", [])}
    proposal_cases = {str(item["case_id"]): item for item in proposal_report.get("cases", [])}
    if set(baseline_cases) != set(proposal_cases):
        raise ValueError("baseline and proposal replay must evaluate the same case IDs")

    cases: list[CaseEvaluation] = []
    for case_id in baseline_cases:
        before = _case_metrics(baseline_cases[case_id])
        after = _case_metrics(proposal_cases[case_id])
        comparisons = compare_metrics(before, after, tolerances={"latency_ms": latency_tolerance_ms})
        cases.append(CaseEvaluation(
            case_id=case_id,
            baseline=before,
            proposal=after,
            improved_metrics=tuple(name for name, item in comparisons.items() if item.improved),
            regressed_metrics=tuple(name for name, item in comparisons.items() if item.regressed),
        ))
    return tuple(cases)


def replay_chunk_proposal(
    proposal: ImprovementProposal,
    *,
    baseline_parameters: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_id: str,
    top_k: int = 3,
    latency_tolerance_ms: float = 1.0,
) -> ReplayReport:
    """Replay a chunking proposal without changing runtime configuration."""
    if proposal.target is not ProposalTarget.CHUNK_STRATEGY:
        raise UnsupportedReplayTarget(
            f"no isolated replay adapter for target: {proposal.target.value}"
        )
    baseline_report = evaluate_chunker(build_chunker(baseline_parameters), dict(dataset), top_k=top_k)
    proposal_report = evaluate_chunker(build_chunker(proposal.parameters), dict(dataset), top_k=top_k)
    baseline_metrics = metrics_from_chunk_report(baseline_report)
    proposal_metrics = metrics_from_chunk_report(proposal_report)
    cases = _case_evaluations(
        baseline_report,
        proposal_report,
        latency_tolerance_ms=latency_tolerance_ms,
    )
    tolerances = {"latency_ms": latency_tolerance_ms}
    context = EvaluationContext(
        dataset_id=dataset_id,
        dataset_hash=_stable_hash(dict(dataset)),
        baseline_config_hash=_stable_hash(dict(baseline_parameters)),
        adapter_version=ADAPTER_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        top_k=top_k,
        tolerances=tolerances,
        git_revision=_git_revision(),
    )
    evaluation = build_evaluation(
        baseline_metrics,
        proposal_metrics,
        cases=cases,
        tolerances=tolerances,
        target_metrics=("recall_at_k", "mrr", "evidence_coverage", "blocked_rate", "token_cost"),
        context=context,
    )
    evaluated_proposal = proposal.with_evaluation(evaluation)
    return ReplayReport(
        dataset_id=dataset_id,
        evaluation_scope="offline_chunk_boundary_proposal_replay",
        baseline_parameters=dict(baseline_parameters),
        proposal=evaluated_proposal,
        evaluation=evaluation,
    )


def load_replay_dataset(path: str | Path) -> dict[str, Any]:
    """Load a replay dataset with a clear error for malformed input."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("documents") or not payload.get("cases"):
        raise ValueError("replay dataset must contain documents and cases")
    return payload


def summarize_case_deltas(reports: Sequence[ReplayReport]) -> dict[str, Any]:
    """Return compact evidence for improved and regressed example cases."""
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for report in reports:
        for case in report.evaluation.cases:
            item = {
                "proposal_id": report.proposal.proposal_id,
                "case_id": case.case_id,
                "improved_metrics": list(case.improved_metrics),
                "regressed_metrics": list(case.regressed_metrics),
            }
            if case.improved_metrics:
                improved.append(item)
            if case.regressed_metrics:
                regressed.append(item)
    return {
        "improved_cases": improved,
        "regressed_cases": regressed,
        "improved_case_count": len(improved),
        "regressed_case_count": len(regressed),
    }


def default_replay_scenarios() -> tuple[tuple[ImprovementProposal, Mapping[str, Any]], ...]:
    """Return two deterministic review fixtures required by Issue #4."""
    return (
        (
            ImprovementProposal(
                proposal_version="chunking.structure-token.v1",
                source_trace_ids=("route-no-recall-fixture",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "structure_token", "max_tokens": 140},
                failure_types=(FailureType.NO_RECALL,),
                description="Use structure-aware chunks to recover boundary evidence.",
            ),
            {"strategy": "fixed_char", "chunk_size": 220},
        ),
        (
            ImprovementProposal(
                proposal_version="chunking.sliding-window.v1",
                source_trace_ids=("route-cost-regression-fixture",),
                target=ProposalTarget.CHUNK_STRATEGY,
                parameters={"strategy": "sliding_window", "chunk_size": 220, "overlap": 60},
                description="Add overlap to protect chunk boundaries.",
            ),
            {"strategy": "structure_token", "max_tokens": 140},
        ),
    )


def run_default_suite(dataset: Mapping[str, Any], *, dataset_id: str) -> tuple[ReplayReport, ...]:
    """Replay the built-in improved and regressed examples."""
    return tuple(
        replay_chunk_proposal(
            proposal,
            baseline_parameters=baseline_parameters,
            dataset=dataset,
            dataset_id=dataset_id,
        )
        for proposal, baseline_parameters in default_replay_scenarios()
    )
