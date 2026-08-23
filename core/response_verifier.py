"""Layered verification for execution, grounding, and task completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Mapping

try:
    from agent_runtime import AgentRuntime, Budget, Observation, Verification
except ModuleNotFoundError:
    # Local checkout fallback first tries the sibling shared package; the
    # container-safe bridge is bundled with EchoForge when that package is not
    # present in the image.
    _path = Path(__file__).resolve()
    _SHARED = (_path.parents[3] / "agent-runtime") if len(_path.parents) > 3 else Path("/__missing_agent_runtime__")
    if _SHARED.exists() and str(_SHARED) not in sys.path:
        sys.path.insert(0, str(_SHARED))
    try:
        from agent_runtime import AgentRuntime, Budget, Observation, Verification
    except ModuleNotFoundError:
        from core.agent_runtime_compat import AgentRuntime, Budget, Observation, Verification


@dataclass(frozen=True)
class ResponseVerification:
    status: str
    reason: str
    evidence: dict[str, Any]
    execution_status: str = "unknown"
    grounding_status: str = "not_checked"
    task_status: str = "not_checked"
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _execution_verifier(result: Any) -> tuple[str, str, dict[str, Any]]:
    trace = getattr(result, "route_trace", None)
    if trace is None:
        return "blocked", "missing_route_trace", {}
    if not getattr(result, "response", "").strip():
        return "blocked", "empty_response", {"trace_id": trace.trace_id}
    if not getattr(result, "success", True):
        return "failed", "agent_execution_failed", {"trace_id": trace.trace_id}
    if getattr(trace, "stop_reason", "") != "completed" or not getattr(trace, "success", False):
        return "blocked", "route_not_verified", {"trace_id": trace.trace_id}
    return "verified", "execution_and_route_verified", {
        "trace_id": trace.trace_id,
        "attempts": trace.attempts,
        "reroutes": trace.reroutes,
    }


_ASCII_TOKEN_RE = re.compile(r"[a-z_]+|\d+(?:\.\d+)?", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_MIN_GROUNDING_COVERAGE = 0.55


def _tokens(text: str) -> set[str]:
    """Return lexical features suitable for Chinese and English overlap.

    A whole Chinese sentence must not become one token: that produced false
    blocks for supported paraphrases while a single shared number could still
    produce a false accept. Character bigrams give the deterministic verifier
    a bounded, inspectable overlap signal without pretending to be entailment.
    """
    lowered = (text or "").lower()
    features = set(_ASCII_TOKEN_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            features.add(run)
        else:
            features.update(run[index:index + 2] for index in range(len(run) - 1))
    return features


def _polarity(text: str) -> set[str]:
    lowered = (text or "").lower()
    negative_markers = ("不能", "不可以", "不会", "不得", "禁止", "无需", "没有", "未能")
    positive_markers = ("可以", "能够", "允许")
    polarity: set[str] = set()
    if any(marker in lowered for marker in negative_markers) or re.search(r"\b(cannot|can't|not|never)\b", lowered):
        polarity.add("negative")
    positive_text = lowered
    for marker in negative_markers:
        positive_text = positive_text.replace(marker, "")
    if any(marker in positive_text for marker in positive_markers) or re.search(r"\b(can|may|allowed)\b", positive_text):
        polarity.add("positive")
    return polarity


def _grounding_support(answer: str, cited_text: str) -> tuple[bool, str, dict[str, Any]]:
    if not cited_text.strip():
        return False, "cited_evidence_empty", {"support_coverage": 0.0}

    answer_numbers = set(_NUMBER_RE.findall(answer))
    evidence_numbers = set(_NUMBER_RE.findall(cited_text))
    missing_numbers = sorted(answer_numbers - evidence_numbers)
    if missing_numbers:
        return False, "answer_number_not_in_evidence", {"missing_numbers": missing_numbers}

    answer_polarity = _polarity(answer)
    evidence_polarity = _polarity(cited_text)
    if answer_polarity and evidence_polarity and answer_polarity.isdisjoint(evidence_polarity):
        return False, "answer_polarity_conflicts_with_evidence", {
            "answer_polarity": sorted(answer_polarity),
            "evidence_polarity": sorted(evidence_polarity),
        }

    answer_terms = _tokens(answer)
    evidence_terms = _tokens(cited_text)
    overlap = answer_terms & evidence_terms
    coverage = len(overlap) / len(answer_terms) if answer_terms else 0.0
    details = {
        "support_coverage": round(coverage, 4),
        "answer_feature_count": len(answer_terms),
        "overlap_feature_count": len(overlap),
    }
    if coverage < _MIN_GROUNDING_COVERAGE:
        return False, "citation_support_below_threshold", details
    return True, "citation_lexically_supported", details


def _grounding_verifier(
    result: Any,
    *,
    evidence_ids: list[str],
    evidence_items: Mapping[str, Mapping[str, Any]],
    knowledge_required: bool,
) -> tuple[str, str, dict[str, Any]]:
    if not knowledge_required:
        return "not_required", "grounding_not_required", {}
    if not evidence_ids:
        return "blocked", "insufficient_evidence", {"evidence_ids": []}

    citations = list(getattr(result, "citations", []) or [])
    if not citations:
        return "blocked", "citation_missing", {"evidence_ids": evidence_ids}
    unknown = sorted(set(citations) - set(evidence_ids))
    if unknown:
        return "blocked", "citation_not_in_retrieval", {"unknown_citations": unknown}

    empty_citations = [
        citation
        for citation in citations
        if not str(evidence_items.get(citation, {}).get("content", "")).strip()
    ]
    if empty_citations:
        return "blocked", "cited_evidence_empty", {"empty_citations": empty_citations}
    cited_text = " ".join(str(evidence_items.get(cid, {}).get("content", "")) for cid in citations)
    supported, support_reason, support_evidence = _grounding_support(
        getattr(result, "response", ""),
        cited_text,
    )
    if not supported:
        return "blocked", support_reason, {"citations": citations, **support_evidence}
    return "verified", "grounding_verified", {
        "citations": citations,
        "evidence_ids": evidence_ids,
        **support_evidence,
    }


def _task_verifier(result: Any) -> tuple[str, str, dict[str, Any]]:
    if bool(getattr(result, "needs_human", False)):
        return "blocked", "needs_human", {"needs_human": True}
    unresolved = list(getattr(result, "unresolved", []) or [])
    if unresolved:
        return "blocked", "unresolved_task", {"unresolved": unresolved}
    confidence = getattr(result, "confidence", None)
    if confidence is not None and confidence < 0.7:
        return "blocked", "low_confidence", {"confidence": confidence}
    return "verified", "task_resolution_signal_verified", {"confidence": confidence}


def verify_orchestrator_result(
    result: Any,
    *,
    evidence_ids: list[str] | None = None,
    evidence_items: Mapping[str, Mapping[str, Any]] | None = None,
    knowledge_required: bool = False,
) -> ResponseVerification:
    """Run independent layered checks; no LLM judge is used here."""
    evidence_ids = evidence_ids or []
    evidence_items = evidence_items or {}

    def step(state):
        state.values["execution"] = _execution_verifier(result)
        return Observation("response_observed", {"has_text": bool(getattr(result, "response", "").strip())})

    def verify(state, _observation):
        status, reason, evidence = state.values["execution"]
        if status == "failed":
            return Verification(False, reason, evidence, failed=True)
        if status == "blocked":
            return Verification(False, reason, evidence, blocked=True)
        grounding = _grounding_verifier(
            result,
            evidence_ids=evidence_ids,
            evidence_items=evidence_items,
            knowledge_required=knowledge_required,
        )
        state.values["grounding"] = grounding
        if grounding[0] == "blocked":
            return Verification(False, grounding[1], {**evidence, **grounding[2]}, blocked=True)
        task = _task_verifier(result)
        state.values["task"] = task
        if task[0] == "blocked":
            return Verification(False, task[1], {**evidence, **grounding[2], **task[2]}, blocked=True)
        return Verification(True, "execution_grounding_task_verified", {**evidence, **grounding[2], **task[2]})

    outcome = AgentRuntime(budget=Budget(max_steps=1)).run({}, step=step, verifier=verify)
    execution = outcome.state.values.get("execution", ("unknown", "unknown", {}))
    grounding = outcome.state.values.get("grounding", ("not_checked", "not_checked", {}))
    task = outcome.state.values.get("task", ("not_checked", "not_checked", {}))
    status = "completed" if outcome.status.value == "completed" else outcome.status.value
    return ResponseVerification(
        status=status,
        reason=outcome.reason,
        evidence=dict(outcome.state.values.get("verification", {})),
        execution_status=execution[0],
        grounding_status=grounding[0],
        task_status=task[0],
        checks=(
            {"verifier": "execution", "status": execution[0], "reason": execution[1]},
            {"verifier": "grounding", "status": grounding[0], "reason": grounding[1]},
            {"verifier": "task", "status": task[0], "reason": task[1]},
        ),
    )
