"""Deterministic RAG golden-set evaluation with a calibrated holdout gate.

This harness intentionally measures retrieval and answerability gating only.
It does not claim that a retrieved passage entails a generated answer, nor
does it validate refusal wording or production latency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from evaluation.chunking_harness import _score
from retrieval.chunking import Chunk, StructureAwareTokenChunker


def load_golden_set(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_source_documents(golden_path: str | Path, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(golden_path).resolve().parents[2] / dataset["source_documents"]
    return json.loads(source.read_text(encoding="utf-8"))["documents"]


def _build_chunks(documents: Iterable[dict[str, Any]]) -> list[Chunk]:
    chunker = StructureAwareTokenChunker(max_tokens=140)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunker.chunk(
                document_id=document["id"],
                title=document["title"],
                text=document["content"],
                document_version=str(document.get("version", "1")),
            )
        )
    return chunks


def _rank(query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    return sorted(
        ((chunk, _score(query, chunk)) for chunk in chunks),
        key=lambda item: item[1],
        reverse=True,
    )


def _balanced_accuracy(rows: list[dict[str, Any]], threshold: float) -> float:
    positive = [row for row in rows if row["expected_retrievable"]]
    negative = [row for row in rows if not row["expected_retrievable"]]
    true_positive_rate = mean(row["top_score"] >= threshold for row in positive) if positive else 1.0
    true_negative_rate = mean(row["top_score"] < threshold for row in negative) if negative else 1.0
    return (true_positive_rate + true_negative_rate) / 2


def _select_threshold(calibration_rows: list[dict[str, Any]]) -> tuple[float, float]:
    scores = sorted({float(row["top_score"]) for row in calibration_rows})
    if not scores:
        return 0.0, 0.0
    candidates = [0.0, scores[-1] + 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    ranked = [(_balanced_accuracy(calibration_rows, threshold), threshold) for threshold in candidates]
    # Prefer the more conservative (higher) threshold when calibration scores tie.
    accuracy, threshold = max(ranked, key=lambda item: (item[0], item[1]))
    return threshold, accuracy


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _aggregate(rows: list[dict[str, Any]], threshold: float, top_k: int) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_retrievable"]]
    no_answer = [row for row in rows if not row["expected_retrievable"]]
    answerability = [row["top_score"] >= threshold for row in rows]
    expected = [row["expected_retrievable"] for row in rows]
    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "no_answer_cases": len(no_answer),
        "answerability_accuracy": _safe_mean([float(pred == truth) for pred, truth in zip(answerability, expected)]),
        "answerable_detection_rate": _safe_mean([float(row["top_score"] >= threshold) for row in answerable]),
        "no_answer_rejection_rate": _safe_mean([float(row["top_score"] < threshold) for row in no_answer]),
        f"full_document_recall_at_{top_k}": _safe_mean([row["full_document_recall_at_k"] for row in answerable]),
        f"mean_document_recall_at_{top_k}": _safe_mean([row["document_recall_at_k"] for row in answerable]),
        "mrr": _safe_mean([row["reciprocal_rank"] for row in answerable]),
        "evidence_coverage": _safe_mean([row["evidence_coverage"] for row in answerable]),
        f"gated_full_recall_at_{top_k}": _safe_mean([
            float(row["top_score"] >= threshold) * row["full_document_recall_at_k"]
            for row in answerable
        ]),
    }


def run_golden_harness(
    golden_path: str | Path,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    dataset = load_golden_set(golden_path)
    documents = load_source_documents(golden_path, dataset)
    chunks = _build_chunks(documents)
    rows: list[dict[str, Any]] = []

    for case in dataset["cases"]:
        ranked = _rank(case["query"], chunks)
        selected = ranked[:top_k]
        relevant_ids = set(case.get("relevant_document_ids", []))
        selected_ids = {chunk.document_id for chunk, _ in selected}
        relevant_ranks = [
            index
            for index, (chunk, _) in enumerate(ranked, start=1)
            if chunk.document_id in relevant_ids
        ]
        selected_text = "\n".join(chunk.content for chunk, _ in selected)
        evidence = [phrase for phrase in case.get("evidence", []) if phrase]
        document_recall = len(relevant_ids & selected_ids) / len(relevant_ids) if relevant_ids else 1.0
        rows.append({
            "case_id": case["id"],
            "split": case["split"],
            "category": case["category"],
            "expected_decision": case["expected_decision"],
            "expected_retrievable": bool(relevant_ids),
            "top_score": round(selected[0][1], 6) if selected else 0.0,
            "top_document_ids": [chunk.document_id for chunk, _ in selected],
            "relevant_document_ids": sorted(relevant_ids),
            "document_recall_at_k": round(document_recall, 4),
            "full_document_recall_at_k": float(document_recall == 1.0),
            "reciprocal_rank": round(1 / min(relevant_ranks), 4) if relevant_ranks else 0.0,
            "evidence_coverage": round(
                sum(phrase in selected_text for phrase in evidence) / len(evidence), 4
            ) if evidence else 1.0,
        })

    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    threshold, calibration_balanced_accuracy = _select_threshold(calibration_rows)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["predicted_retrievable"] = row["top_score"] >= threshold
        by_category[row["category"]].append(row)

    holdout_metrics = _aggregate(holdout_rows, threshold, top_k)
    gated_recall_key = f"gated_full_recall_at_{top_k}"
    gate_passed = bool(
        (holdout_metrics.get(gated_recall_key) or 0.0) >= 0.80
        and (holdout_metrics.get("no_answer_rejection_rate") or 0.0) >= 0.80
        and (holdout_metrics.get("evidence_coverage") or 0.0) >= 0.80
    )
    return {
        "evaluation_scope": "synthetic_retrieval_and_answerability_baseline",
        "dataset_version": dataset["version"],
        "dataset_status": dataset["status"],
        "documents": len(documents),
        "chunks": len(chunks),
        "cases": len(rows),
        "top_k": top_k,
        "threshold": round(threshold, 6),
        "threshold_selection": {
            "split": "calibration",
            "objective": "balanced_accuracy",
            "balanced_accuracy": round(calibration_balanced_accuracy, 4),
        },
        "calibration": _aggregate(calibration_rows, threshold, top_k),
        "holdout": holdout_metrics,
        "categories": {
            category: _aggregate(category_rows, threshold, top_k)
            for category, category_rows in sorted(by_category.items())
        },
        "readiness_gate": {
            "status": "research_baseline_only" if gate_passed else "no_go",
            "passed": gate_passed,
            "criteria": {
                f"holdout_{gated_recall_key}": 0.80,
                "holdout_no_answer_rejection_rate": 0.80,
                "holdout_evidence_coverage": 0.80,
            },
            "boundary": (
                "Passing this synthetic gate does not authorize production use. "
                "Human-reviewed domain cases, live retrieval, answer verification, latency, cost, and tail-risk audits remain required."
            ),
        },
        "cases_detail": rows,
    }
