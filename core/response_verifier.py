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


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", (text or "").lower()))


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

    answer_terms = _tokens(getattr(result, "response", ""))
    cited_text = " ".join(str(evidence_items.get(cid, {}).get("content", "")) for cid in citations)
    if cited_text and not answer_terms.intersection(_tokens(cited_text)):
        return "blocked", "citation_not_supporting_answer", {"citations": citations}
    return "verified", "grounding_verified", {"citations": citations, "evidence_ids": evidence_ids}


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
