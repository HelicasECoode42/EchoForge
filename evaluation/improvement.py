"""Offline proposal-based improvement contracts.

This module contains only deterministic domain logic.  It deliberately does
not know how to mutate production configuration, prompts, memory or model
providers.  Replay adapters may evaluate a proposal in an isolated harness,
but approval remains a human-controlled state transition.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional


class FailureType(str, Enum):
    """Auditable failure buckets used to group improvement proposals."""

    NO_RECALL = "no_recall"
    INCORRECT_CITATION = "incorrect_citation"
    STALE_EVIDENCE = "stale_evidence"
    LOW_CONFIDENCE = "low_confidence"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ProposalStatus(str, Enum):
    """Allowed lifecycle states for an offline proposal."""

    PROPOSED = "proposed"
    CANDIDATE = "candidate"
    NO_CHANGE = "no_change"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProposalTarget(str, Enum):
    """Configuration surfaces that may be discussed offline."""

    CHUNK_STRATEGY = "chunk_strategy"
    RETRIEVAL_POLICY = "retrieval_policy"
    REWRITE_RERANK = "rewrite_rerank"
    REFUSAL_RULE = "refusal_rule"
    PROMPT_VERSION = "prompt_version"


METRIC_DIRECTIONS: Mapping[str, str] = {
    "recall_at_k": "higher",
    "mrr": "higher",
    "evidence_coverage": "higher",
    "blocked_rate": "lower",
    "latency_ms": "lower",
    "token_cost": "lower",
}

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_DESCRIPTION_CODE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,119}$")
_SAFE_VERSION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,120}$")
_SENSITIVE_KEY_RE = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|credential|password|passwd|prompt|secret|token)($|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(bearer\s+|-----begin|raw[-_.\s]+user[-_.\s]+prompt|system[-_.\s]+prompt|sk-[a-z0-9_-]{6,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics used by the offline regression gate."""

    recall_at_k: float
    evidence_coverage: float
    blocked_rate: float
    latency_ms: float
    token_cost: float
    mrr: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("recall_at_k", "evidence_coverage", "blocked_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("latency_ms", "token_cost"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.mrr is not None and (
            not math.isfinite(float(self.mrr)) or not 0.0 <= float(self.mrr) <= 1.0
        ):
            raise ValueError("mrr must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationMetrics":
        return cls(
            recall_at_k=float(payload["recall_at_k"]),
            evidence_coverage=float(payload["evidence_coverage"]),
            blocked_rate=float(payload["blocked_rate"]),
            latency_ms=float(payload["latency_ms"]),
            token_cost=float(payload["token_cost"]),
            mrr=float(payload["mrr"]) if payload.get("mrr") is not None else None,
        )


@dataclass(frozen=True)
class MetricComparison:
    """One baseline/proposal metric comparison."""

    baseline: float
    proposal: float
    delta: float
    direction: str
    improved: bool
    regressed: bool


@dataclass(frozen=True)
class CaseEvaluation:
    """Per-case evidence for a proposal evaluation."""

    case_id: str
    baseline: EvaluationMetrics
    proposal: EvaluationMetrics
    improved_metrics: tuple[str, ...] = ()
    regressed_metrics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["improved_metrics"] = list(self.improved_metrics)
        payload["regressed_metrics"] = list(self.regressed_metrics)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseEvaluation":
        return cls(
            case_id=str(payload["case_id"]),
            baseline=EvaluationMetrics.from_dict(payload["baseline"]),
            proposal=EvaluationMetrics.from_dict(payload["proposal"]),
            improved_metrics=tuple(str(item) for item in payload.get("improved_metrics", [])),
            regressed_metrics=tuple(str(item) for item in payload.get("regressed_metrics", [])),
        )


@dataclass(frozen=True)
class EvaluationContext:
    """Reproducibility metadata for one offline evaluation."""

    dataset_id: str = ""
    dataset_hash: str = ""
    baseline_config_hash: str = ""
    adapter_version: str = ""
    evaluator_version: str = ""
    top_k: Optional[int] = None
    tolerances: Mapping[str, float] = field(default_factory=dict)
    git_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "baseline_config_hash": self.baseline_config_hash,
            "adapter_version": self.adapter_version,
            "evaluator_version": self.evaluator_version,
            "top_k": self.top_k,
            "tolerances": dict(self.tolerances),
            "git_revision": self.git_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationContext":
        return cls(
            dataset_id=str(payload.get("dataset_id", "")),
            dataset_hash=str(payload.get("dataset_hash", "")),
            baseline_config_hash=str(payload.get("baseline_config_hash", "")),
            adapter_version=str(payload.get("adapter_version", "")),
            evaluator_version=str(payload.get("evaluator_version", "")),
            top_k=int(payload["top_k"]) if payload.get("top_k") is not None else None,
            tolerances={str(name): float(value) for name, value in payload.get("tolerances", {}).items()},
            git_revision=str(payload.get("git_revision", "")),
        )


@dataclass(frozen=True)
class ProposalEvaluation:
    """Result of replaying one proposal against an immutable baseline."""

    baseline: EvaluationMetrics
    proposal: EvaluationMetrics
    comparisons: Mapping[str, MetricComparison]
    passed_regression: bool
    reason: str
    cases: tuple[CaseEvaluation, ...] = ()
    evaluated_at: str = ""
    target_metrics: tuple[str, ...] = ()
    context: EvaluationContext = field(default_factory=EvaluationContext)

    def __post_init__(self) -> None:
        expected = compare_metrics(
            self.baseline,
            self.proposal,
            tolerances=self.context.tolerances,
        )
        if dict(self.comparisons) != expected:
            raise ValueError("evaluation comparisons do not match metrics and tolerances")
        targets = set(self.target_metrics) or set(expected)
        regressions = any(
            name in targets and item.regressed
            for name, item in expected.items()
        )
        improvements = any(
            name in targets and item.improved
            for name, item in expected.items()
        )
        if self.passed_regression != (not regressions and improvements):
            raise ValueError("evaluation gate result does not match metric comparisons")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "proposal": self.proposal.to_dict(),
            "comparisons": {name: asdict(value) for name, value in self.comparisons.items()},
            "passed_regression": self.passed_regression,
            "reason": self.reason,
            "cases": [case.to_dict() for case in self.cases],
            "evaluated_at": self.evaluated_at,
            "target_metrics": list(self.target_metrics),
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalEvaluation":
        return cls(
            baseline=EvaluationMetrics.from_dict(payload["baseline"]),
            proposal=EvaluationMetrics.from_dict(payload["proposal"]),
            comparisons={
                str(name): MetricComparison(**item)
                for name, item in payload.get("comparisons", {}).items()
            },
            passed_regression=bool(payload["passed_regression"]),
            reason=str(payload.get("reason", "")),
            cases=tuple(CaseEvaluation.from_dict(item) for item in payload.get("cases", [])),
            evaluated_at=str(payload.get("evaluated_at", "")),
            target_metrics=tuple(str(item) for item in payload.get("target_metrics", [])),
            context=EvaluationContext.from_dict(payload.get("context", {})),
        )


@dataclass(frozen=True)
class ImprovementProposal:
    """A reviewable offline change proposal.

    The payload describes a candidate configuration only.  There is no apply
    method by design: this object can be replayed and approved, but it cannot
    mutate a live runtime.
    """

    proposal_version: str
    source_trace_ids: tuple[str, ...]
    target: ProposalTarget
    parameters: Mapping[str, Any]
    failure_types: tuple[FailureType, ...] = ()
    description: str = ""
    proposal_id: str = ""
    status: ProposalStatus = ProposalStatus.PROPOSED
    created_at: str = ""
    evaluation: Optional[ProposalEvaluation] = None
    evaluation_history: tuple[ProposalEvaluation, ...] = ()
    approved_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.proposal_version.strip():
            raise ValueError("proposal_version must not be empty")
        if len(self.proposal_version) > 120 or any(ord(char) < 32 for char in self.proposal_version):
            raise ValueError("proposal_version must be <= 120 characters without control characters")
        if not self.source_trace_ids or any(not item.strip() for item in self.source_trace_ids):
            raise ValueError("source_trace_ids must contain at least one non-empty trace id")
        if any(not _TRACE_ID_RE.fullmatch(item) for item in self.source_trace_ids):
            raise ValueError("source_trace_ids must use privacy-safe identifier characters")
        if self.description:
            if not _DESCRIPTION_CODE_RE.fullmatch(self.description):
                raise ValueError("description must be an auditable machine-readable code")
            if _SENSITIVE_VALUE_RE.search(self.description):
                raise ValueError("description contains sensitive string material")
        if not isinstance(self.target, ProposalTarget):
            object.__setattr__(self, "target", ProposalTarget(self.target))
        if not isinstance(self.status, ProposalStatus):
            object.__setattr__(self, "status", ProposalStatus(self.status))
        normalized_failures = tuple(
            item if isinstance(item, FailureType) else FailureType(item)
            for item in self.failure_types
        )
        object.__setattr__(self, "failure_types", normalized_failures)
        self._validate_parameters(self.target, self.parameters)
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())
        expected_id = self._make_id()
        if not self.proposal_id:
            object.__setattr__(self, "proposal_id", expected_id)
        elif self.proposal_id != expected_id:
            raise ValueError("proposal_id does not match immutable identity fields")
        if self.evaluation is not None and not self.evaluation_history:
            object.__setattr__(self, "evaluation_history", (self.evaluation,))
        if self.evaluation is None and self.evaluation_history:
            raise ValueError("evaluation_history requires a current evaluation")
        if self.evaluation is not None and self.evaluation_history[-1] != self.evaluation:
            raise ValueError("current evaluation must be the latest evaluation_history item")
        if self.status in {ProposalStatus.CANDIDATE, ProposalStatus.NO_CHANGE} and self.evaluation is None:
            raise ValueError(f"{self.status.value} proposals must include evaluation")
        if self.status is ProposalStatus.CANDIDATE and (
            self.evaluation is None or not self.evaluation.passed_regression
        ):
            raise ValueError("candidate proposals must pass offline regression")
        if self.status is ProposalStatus.NO_CHANGE and (
            self.evaluation is None or self.evaluation.passed_regression
        ):
            raise ValueError("no_change proposals must fail the improvement gate")
        if self.status is ProposalStatus.APPROVED:
            if not self.approved_at:
                raise ValueError("approved proposals must include approved_at")
            if self.evaluation is None or not self.evaluation.passed_regression:
                raise ValueError("approved proposals must have passed offline regression")
        elif self.approved_at is not None:
            raise ValueError("only approved proposals may include approved_at")

    def _make_id(self) -> str:
        payload = {
            "proposal_version": self.proposal_version,
            "source_trace_ids": list(self.source_trace_ids),
            "target": self.target.value,
            "parameters": self.parameters,
            "failure_types": [item.value for item in self.failure_types],
            "description": self.description,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"proposal-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _validate_parameters(target: ProposalTarget, parameters: Mapping[str, Any]) -> None:
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError("parameters must be a non-empty mapping")
        if any(not isinstance(key, str) for key in parameters):
            raise ValueError("proposal parameter keys must be strings")

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    key_text = str(key)
                    if _SENSITIVE_KEY_RE.search(key_text):
                        raise ValueError(f"parameters contain sensitive key: {key_text}")
                    visit(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    visit(nested)
            elif isinstance(value, str) and _SENSITIVE_VALUE_RE.search(value):
                raise ValueError("parameters contain sensitive string material")

        visit(parameters)
        keys = set(parameters)

        def exact(allowed: set[str], required: set[str]) -> None:
            extras = keys - allowed
            missing = required - keys
            if extras or missing:
                detail = []
                if extras:
                    detail.append(f"unexpected={sorted(extras)}")
                if missing:
                    detail.append(f"missing={sorted(missing)}")
                raise ValueError("invalid proposal parameter schema: " + ", ".join(detail))

        def integer(name: str, minimum: int, maximum: int) -> int:
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
            return value

        if target is ProposalTarget.CHUNK_STRATEGY:
            strategy = parameters.get("strategy")
            if strategy == "fixed_char":
                exact({"strategy", "chunk_size"}, {"strategy"})
                if "chunk_size" in parameters:
                    integer("chunk_size", 1, 100_000)
            elif strategy == "sliding_window":
                exact({"strategy", "chunk_size", "overlap"}, {"strategy"})
                chunk_size = integer("chunk_size", 1, 100_000) if "chunk_size" in parameters else 500
                overlap = integer("overlap", 0, 99_999) if "overlap" in parameters else 100
                if overlap >= chunk_size:
                    raise ValueError("overlap must be smaller than chunk_size")
            elif strategy == "structure_token":
                exact({"strategy", "max_tokens"}, {"strategy"})
                if "max_tokens" in parameters:
                    integer("max_tokens", 8, 100_000)
            else:
                raise ValueError("unsupported chunk strategy")
            return

        if target is ProposalTarget.RETRIEVAL_POLICY:
            if keys == {"mode"} and parameters["mode"] == "adaptive":
                return
            if keys == {"retry_limit", "fallback"}:
                integer("retry_limit", 0, 10)
                if parameters["fallback"] != "deterministic":
                    raise ValueError("fallback must be deterministic")
                return
            if keys == {"adaptive_retrieval", "max_rerank_candidates"}:
                if parameters["adaptive_retrieval"] is not True:
                    raise ValueError("adaptive_retrieval must be true")
                integer("max_rerank_candidates", 1, 100)
                return
            raise ValueError("invalid retrieval_policy parameter schema")

        if target is ProposalTarget.REFUSAL_RULE:
            if keys in ({"require_citation_in_evidence"}, {"reject_stale_evidence"}):
                if next(iter(parameters.values())) is not True:
                    raise ValueError("refusal rule flag must be true")
                return
            if keys == {"minimum_confidence"}:
                value = parameters["minimum_confidence"]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError("minimum_confidence must be between 0 and 1")
                return
            raise ValueError("invalid refusal_rule parameter schema")

        if target is ProposalTarget.PROMPT_VERSION:
            exact({"version"}, {"version"})
            if not isinstance(parameters["version"], str) or not _SAFE_VERSION_RE.fullmatch(parameters["version"]):
                raise ValueError("prompt version must be a safe version identifier")
            return

        if target is ProposalTarget.REWRITE_RERANK:
            exact({"rewrite_enabled", "rerank_top_k"}, {"rewrite_enabled", "rerank_top_k"})
            if parameters["rewrite_enabled"] is not True:
                raise ValueError("rewrite_enabled must be true")
            integer("rerank_top_k", 1, 100)
            return

        raise ValueError(f"unsupported proposal target: {target.value}")

    def with_evaluation(self, evaluation: ProposalEvaluation) -> "ImprovementProposal":
        """Attach one offline result; approved/rejected artifacts are immutable."""
        if self.status is not ProposalStatus.PROPOSED:
            raise ValueError("only proposed artifacts can be evaluated")
        targets = set(evaluation.target_metrics) or set(evaluation.comparisons)
        has_regression = any(
            name in targets and item.regressed
            for name, item in evaluation.comparisons.items()
        )
        if evaluation.passed_regression:
            next_status = ProposalStatus.CANDIDATE
        elif not has_regression:
            next_status = ProposalStatus.NO_CHANGE
        else:
            next_status = ProposalStatus.REJECTED
        return replace(
            self,
            evaluation=evaluation,
            evaluation_history=self.evaluation_history + (evaluation,),
            status=next_status,
        )

    def approve(self, approved_at: Optional[str] = None) -> "ImprovementProposal":
        if self.status is not ProposalStatus.CANDIDATE:
            raise ValueError("only a proposal that passed offline regression can be approved")
        return replace(
            self,
            status=ProposalStatus.APPROVED,
            approved_at=approved_at or datetime.now(timezone.utc).isoformat(),
        )

    def reject(self) -> "ImprovementProposal":
        if self.status in {ProposalStatus.APPROVED, ProposalStatus.REJECTED, ProposalStatus.NO_CHANGE}:
            raise ValueError(f"{self.status.value} is a terminal proposal state")
        return replace(self, status=ProposalStatus.REJECTED)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target"] = self.target.value
        payload["status"] = self.status.value
        payload["failure_types"] = [item.value for item in self.failure_types]
        payload["source_trace_ids"] = list(self.source_trace_ids)
        payload["parameters"] = dict(self.parameters)
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation.to_dict()
        payload["evaluation_history"] = [item.to_dict() for item in self.evaluation_history]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImprovementProposal":
        evaluation_payload = payload.get("evaluation")
        return cls(
            proposal_version=str(payload["proposal_version"]),
            source_trace_ids=tuple(str(item) for item in payload["source_trace_ids"]),
            target=ProposalTarget(payload["target"]),
            parameters=dict(payload["parameters"]),
            failure_types=tuple(FailureType(item) for item in payload.get("failure_types", [])),
            description=str(payload.get("description", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            status=ProposalStatus(payload.get("status", ProposalStatus.PROPOSED.value)),
            created_at=str(payload.get("created_at", "")),
            evaluation=ProposalEvaluation.from_dict(evaluation_payload) if evaluation_payload else None,
            evaluation_history=tuple(
                ProposalEvaluation.from_dict(item)
                for item in payload.get("evaluation_history", [])
            ),
            approved_at=payload.get("approved_at"),
        )


def compare_metrics(
    baseline: EvaluationMetrics,
    proposal: EvaluationMetrics,
    *,
    tolerances: Optional[Mapping[str, float]] = None,
) -> dict[str, MetricComparison]:
    """Compare metrics using explicit higher-is-better/lower-is-better rules."""
    tolerances = tolerances or {}
    comparisons: dict[str, MetricComparison] = {}
    for name, direction in METRIC_DIRECTIONS.items():
        baseline_value = getattr(baseline, name)
        proposal_value = getattr(proposal, name)
        if baseline_value is None or proposal_value is None:
            continue
        before = float(baseline_value)
        after = float(proposal_value)
        delta = after - before
        tolerance = max(0.0, float(tolerances.get(name, 0.0)))
        improved = delta > tolerance if direction == "higher" else delta < -tolerance
        regressed = delta < -tolerance if direction == "higher" else delta > tolerance
        comparisons[name] = MetricComparison(
            baseline=round(before, 6),
            proposal=round(after, 6),
            delta=round(delta, 6),
            direction=direction,
            improved=improved,
            regressed=regressed,
        )
    return comparisons


def build_evaluation(
    baseline: EvaluationMetrics,
    proposal: EvaluationMetrics,
    *,
    cases: tuple[CaseEvaluation, ...] = (),
    tolerances: Optional[Mapping[str, float]] = None,
    target_metrics: tuple[str, ...] = (),
    context: Optional[EvaluationContext] = None,
) -> ProposalEvaluation:
    """Build the regression and measurable-improvement gate result."""
    if context is None:
        context = EvaluationContext(tolerances=dict(tolerances or {}))
    elif tolerances is not None and dict(context.tolerances) != dict(tolerances):
        raise ValueError("tolerances must match evaluation context tolerances")

    comparisons = compare_metrics(baseline, proposal, tolerances=context.tolerances)
    required_improvements = set(target_metrics) or set(comparisons)
    regressions = [
        name for name, item in comparisons.items()
        if name in required_improvements and item.regressed
    ]
    improvements = [
        name for name, item in comparisons.items()
        if name in required_improvements and item.improved
    ]
    if regressions:
        reason = f"offline regression detected: {', '.join(regressions)}"
        passed_regression = False
    elif not improvements:
        reason = "no measurable improvement in target metrics"
        passed_regression = False
    else:
        reason = "offline regression gate passed"
        passed_regression = True
    return ProposalEvaluation(
        baseline=baseline,
        proposal=proposal,
        comparisons=comparisons,
        passed_regression=passed_regression,
        reason=reason,
        cases=cases,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        target_metrics=tuple(target_metrics),
        context=context,
    )


def classify_failure(trace: Mapping[str, Any], *, low_confidence_threshold: float = 0.5) -> tuple[FailureType, ...]:
    """Classify failures from privacy-safe route/graph/verifier evidence.

    Multiple classifications are allowed because a timeout can also leave a
    retrieval request without evidence.  The returned order is stable for
    grouping and snapshot tests.
    """
    signals: set[FailureType] = set()
    stop_reason = str(trace.get("stop_reason") or trace.get("pipeline_stop_reason") or "").lower()
    verification_reason = str(trace.get("verification_reason") or "").lower()
    error_types = {
        str(item.get("error_type") or "").lower()
        for item in trace.get("node_runs", [])
        if isinstance(item, Mapping)
    }
    step_reasons = {
        str(item.get("reason") or "").lower()
        for item in trace.get("steps", [])
        if isinstance(item, Mapping)
    }

    if (
        "timeout" in stop_reason
        or "latency_budget_exhausted" in stop_reason
        or "timeout" in verification_reason
        or "timeouterror" in error_types
    ):
        signals.add(FailureType.TIMEOUT)
    if (
        "tool" in verification_reason
        or "tool_failed" in step_reasons
        or "tool_failure" in stop_reason
        or any("tool" in item for item in error_types)
    ):
        signals.add(FailureType.TOOL_FAILURE)
    evidence_items = trace.get("evidence_items") or []
    has_stale_item = any(
        isinstance(item, Mapping) and str(item.get("freshness", "")).lower() == "stale"
        for item in evidence_items
    )
    if "stale" in verification_reason or bool(trace.get("stale_evidence")) or has_stale_item:
        signals.add(FailureType.STALE_EVIDENCE)
    evidence_ids = trace.get("evidence_ids") or []
    evidence_id_set = {str(item) for item in evidence_ids}
    citation_ids = {str(item) for item in (trace.get("citations") or [])}
    if (
        "citation_not_in_retrieval" in verification_reason
        or bool(trace.get("incorrect_citation"))
        or bool(citation_ids - evidence_id_set)
    ):
        signals.add(FailureType.INCORRECT_CITATION)

    confidence = trace.get("confidence")
    if "confidence" in verification_reason or (
        isinstance(confidence, (int, float)) and float(confidence) < low_confidence_threshold
    ):
        signals.add(FailureType.LOW_CONFIDENCE)

    retrieval_requested = bool(
        trace.get("should_retrieve")
        or trace.get("retrieval_requested")
        or trace.get("knowledge_requested")
    )
    if retrieval_requested and not evidence_ids:
        signals.add(FailureType.NO_RECALL)
    if "insufficient_evidence" in verification_reason:
        signals.add(FailureType.NO_RECALL)

    terminal_failure = (
        trace.get("success") is False
        or str(trace.get("status") or "").lower() in {"blocked", "failed"}
        or stop_reason not in {"", "completed", "complete", "success"}
    )
    if terminal_failure and not signals:
        signals.add(FailureType.UNKNOWN)

    order = tuple(FailureType)
    return tuple(item for item in order if item in signals)


_FAILURE_PROPOSAL_RECIPES: Mapping[FailureType, tuple[ProposalTarget, Mapping[str, Any], str]] = {
    FailureType.NO_RECALL: (
        ProposalTarget.CHUNK_STRATEGY,
        {"strategy": "structure_token", "max_tokens": 140},
        "chunking.structure-aware-boundaries",
    ),
    FailureType.INCORRECT_CITATION: (
        ProposalTarget.REFUSAL_RULE,
        {"require_citation_in_evidence": True},
        "refusal.citation-in-evidence",
    ),
    FailureType.STALE_EVIDENCE: (
        ProposalTarget.REFUSAL_RULE,
        {"reject_stale_evidence": True},
        "refusal.reject-stale-evidence",
    ),
    FailureType.LOW_CONFIDENCE: (
        ProposalTarget.REFUSAL_RULE,
        {"minimum_confidence": 0.6},
        "refusal.minimum-confidence",
    ),
    FailureType.TOOL_FAILURE: (
        ProposalTarget.RETRIEVAL_POLICY,
        {"retry_limit": 1, "fallback": "deterministic"},
        "retrieval.bounded-tool-fallback",
    ),
    FailureType.TIMEOUT: (
        ProposalTarget.RETRIEVAL_POLICY,
        {"adaptive_retrieval": True, "max_rerank_candidates": 8},
        "retrieval.adaptive-latency-budget",
    ),
}


def generate_proposals_from_traces(
    traces: list[Mapping[str, Any]],
    *,
    proposal_version: str = "offline-recipes.v1",
) -> tuple[ImprovementProposal, ...]:
    """Cluster privacy-safe traces and create deterministic review proposals.

    This is intentionally recipe-based rather than LLM-generated.  It gives
    the offline loop a reproducible first proposal source; richer proposal
    synthesis can be added behind the same contract without changing approval
    or replay boundaries.
    """
    grouped: dict[FailureType, list[str]] = {}
    for trace in traces:
        trace_id = str(trace.get("trace_id") or "").strip()
        if not trace_id:
            continue
        for failure_type in classify_failure(trace):
            grouped.setdefault(failure_type, []).append(trace_id)

    proposals: list[ImprovementProposal] = []
    for failure_type in FailureType:
        trace_ids = tuple(dict.fromkeys(grouped.get(failure_type, [])))
        recipe = _FAILURE_PROPOSAL_RECIPES.get(failure_type)
        if not trace_ids or recipe is None:
            continue
        target, parameters, description = recipe
        proposals.append(ImprovementProposal(
            proposal_version=proposal_version,
            source_trace_ids=trace_ids,
            target=target,
            parameters=parameters,
            failure_types=(failure_type,),
            description=description,
        ))
    return tuple(proposals)
