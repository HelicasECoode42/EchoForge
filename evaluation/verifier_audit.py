"""Labeled audit harness for the deterministic response verifier."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.response_verifier import verify_orchestrator_result


def load_audit_cases(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result(case: dict[str, Any]) -> SimpleNamespace:
    if "route_trace" in case and case["route_trace"] is None:
        trace = None
    else:
        trace = SimpleNamespace(
            trace_id=f"audit-{case['id']}",
            stop_reason=case.get("route_stop_reason", "completed"),
            success=case.get("route_success", True),
            attempts=1,
            reroutes=0,
        )
    return SimpleNamespace(
        response=case.get("answer", ""),
        success=case.get("success", True),
        route_trace=trace,
        citations=list(case.get("citations", [])),
        needs_human=case.get("needs_human", False),
        unresolved=list(case.get("unresolved", [])),
        confidence=case.get("confidence"),
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _legacy_status(case: dict[str, Any]) -> tuple[str, str]:
    """Reproduce the pre-audit gate so before/after metrics stay executable."""
    if "route_trace" in case and case["route_trace"] is None:
        return "blocked", "missing_route_trace"
    if not case.get("answer", "").strip():
        return "blocked", "empty_response"
    if not case.get("success", True):
        return "failed", "agent_execution_failed"
    if case.get("route_stop_reason", "completed") != "completed" or not case.get("route_success", True):
        return "blocked", "route_not_verified"
    if case.get("knowledge_required", False):
        evidence = case.get("evidence", {})
        citations = list(case.get("citations", []))
        if not evidence:
            return "blocked", "insufficient_evidence"
        if not citations:
            return "blocked", "citation_missing"
        if set(citations) - set(evidence):
            return "blocked", "citation_not_in_retrieval"
        answer_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", case.get("answer", "").lower()))
        cited_text = " ".join(str(evidence.get(citation, "")) for citation in citations)
        cited_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", cited_text.lower()))
        if cited_text and not answer_terms.intersection(cited_terms):
            return "blocked", "citation_not_supporting_answer"
    if case.get("needs_human", False):
        return "blocked", "needs_human"
    if case.get("unresolved"):
        return "blocked", "unresolved_task"
    confidence = case.get("confidence")
    if confidence is not None and confidence < 0.7:
        return "blocked", "low_confidence"
    return "completed", "execution_grounding_task_verified"


def _metrics(rows: list[dict[str, Any]], observed_key: str) -> dict[str, Any]:
    expected_accept = [row for row in rows if row["expected_status"] == "completed"]
    expected_reject = [row for row in rows if row["expected_status"] != "completed"]
    observed_accept = [row for row in rows if row[observed_key] == "completed"]
    true_accepts = sum(row["expected_status"] == "completed" and row[observed_key] == "completed" for row in rows)
    true_rejects = sum(row["expected_status"] != "completed" and row[observed_key] == row["expected_status"] for row in rows)
    false_accepts = sum(row["expected_status"] != "completed" and row[observed_key] == "completed" for row in rows)
    false_blocks = sum(row["expected_status"] == "completed" and row[observed_key] != "completed" for row in rows)
    return {
        "accuracy": _safe_ratio(true_accepts + true_rejects, len(rows)),
        "accept_precision": _safe_ratio(true_accepts, len(observed_accept)),
        "accept_recall": _safe_ratio(true_accepts, len(expected_accept)),
        "false_accept_rate": _safe_ratio(false_accepts, len(expected_reject)),
        "false_block_rate": _safe_ratio(false_blocks, len(expected_accept)),
        "dangerous_false_accepts": false_accepts,
        "false_blocks": false_blocks,
    }


def run_verifier_audit(path: str | Path) -> dict[str, Any]:
    dataset = load_audit_cases(path)
    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in dataset["cases"]:
        evidence = {
            evidence_id: {"content": content}
            for evidence_id, content in case.get("evidence", {}).items()
        }
        verification = verify_orchestrator_result(
            _result(case),
            evidence_ids=list(evidence),
            evidence_items=evidence,
            knowledge_required=case.get("knowledge_required", False),
        )
        expected_status = case["expected_status"]
        legacy_status, legacy_reason = _legacy_status(case)
        row = {
            "case_id": case["id"],
            "category": case["category"],
            "expected_status": expected_status,
            "observed_status": verification.status,
            "observed_reason": verification.reason,
            "legacy_status": legacy_status,
            "legacy_reason": legacy_reason,
            "passed": verification.status == expected_status,
            "dangerous_false_accept": expected_status != "completed" and verification.status == "completed",
            "false_block": expected_status == "completed" and verification.status != "completed",
        }
        rows.append(row)
        by_category[row["category"]].append(row)

    return {
        "evaluation_scope": "deterministic_verifier_labeled_audit",
        "dataset_version": dataset["version"],
        "dataset_status": dataset["status"],
        "cases": len(rows),
        "legacy_metrics": _metrics(rows, "legacy_status"),
        "metrics": _metrics(rows, "observed_status"),
        "categories": {
            category: {
                "cases": len(category_rows),
                "passed": sum(row["passed"] for row in category_rows),
                "dangerous_false_accepts": sum(row["dangerous_false_accept"] for row in category_rows),
                "false_blocks": sum(row["false_block"] for row in category_rows),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "boundary": (
            "This audit measures only labeled deterministic cases. It cannot establish semantic entailment, "
            "real-user quality, or production readiness."
        ),
        "cases_detail": rows,
    }
