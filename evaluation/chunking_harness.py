"""Offline, deterministic comparison harness for chunking strategies.

The lexical ranker is a controlled proxy, not an embedding benchmark.  The
harness isolates the effect of chunk boundaries and metadata while exposing a
retriever-shaped evaluation surface that can later accept production search.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from retrieval.chunking import Chunk, Chunker, estimate_tokens


ASCII_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


def _features(text: str) -> Counter[str]:
    lowered = (text or "").lower()
    features: List[str] = ASCII_WORD_RE.findall(lowered)
    for run in CJK_RUN_RE.findall(lowered):
        features.extend(run[i:i + 2] for i in range(max(0, len(run) - 1)))
        if len(run) == 1:
            features.append(run)
    return Counter(features)


def _score(query: str, chunk: Chunk) -> float:
    query_features = _features(query)
    chunk_features = _features(f"{chunk.title} {chunk.section_path} {chunk.content}")
    overlap = sum(min(count, chunk_features.get(feature, 0)) for feature, count in query_features.items())
    normalization = math.sqrt(max(1, sum(chunk_features.values())))
    phrase_bonus = 2.0 if query.strip().lower() in chunk.content.lower() else 0.0
    return overlap / normalization + phrase_bonus


@dataclass(frozen=True)
class QueryMetric:
    case_id: str
    first_relevant_rank: int
    recall_at_k: float
    reciprocal_rank: float
    evidence_coverage: float
    context_tokens: int
    retrieval_latency_ms: float


def load_dataset(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_chunker(chunker: Chunker, dataset: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    started = time.perf_counter()
    chunks: List[Chunk] = []
    for document in dataset["documents"]:
        chunks.extend(chunker.chunk(
            document_id=document["id"],
            title=document["title"],
            text=document["content"],
            document_version=str(document.get("version", "1")),
        ))
    indexing_latency_ms = (time.perf_counter() - started) * 1000

    metrics: List[QueryMetric] = []
    for case in dataset["cases"]:
        retrieval_started = time.perf_counter()
        ranked = sorted(chunks, key=lambda chunk: _score(case["query"], chunk), reverse=True)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        evidence = [item for item in case.get("evidence", []) if item]

        def relevant(chunk: Chunk) -> bool:
            if chunk.document_id != case["relevant_document_id"]:
                return False
            return any(item in chunk.content for item in evidence)

        first_rank = next((index for index, chunk in enumerate(ranked, start=1) if relevant(chunk)), 0)
        selected = ranked[:top_k]
        selected_text = "\n".join(chunk.content for chunk in selected)
        coverage = (
            sum(1 for item in evidence if item in selected_text) / len(evidence)
            if evidence else 0.0
        )
        metrics.append(QueryMetric(
            case_id=case["id"],
            first_relevant_rank=first_rank,
            recall_at_k=1.0 if first_rank and first_rank <= top_k else 0.0,
            reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
            evidence_coverage=coverage,
            context_tokens=sum(chunk.token_count for chunk in selected),
            retrieval_latency_ms=retrieval_latency_ms,
        ))

    return {
        "strategy": chunker.name,
        "ranker": "deterministic_lexical_proxy",
        "top_k": top_k,
        "documents": len(dataset["documents"]),
        "queries": len(metrics),
        "chunk_count": len(chunks),
        "indexing_latency_ms": round(indexing_latency_ms, 3),
        "avg_retrieval_latency_ms": round(mean(m.retrieval_latency_ms for m in metrics), 3),
        "recall_at_k": round(mean(m.recall_at_k for m in metrics), 4),
        "mrr": round(mean(m.reciprocal_rank for m in metrics), 4),
        "evidence_coverage": round(mean(m.evidence_coverage for m in metrics), 4),
        "avg_context_tokens": round(mean(m.context_tokens for m in metrics), 2),
        "cases": [asdict(metric) for metric in metrics],
    }


def run_harness(chunkers: Sequence[Chunker], dataset: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    reports = [evaluate_chunker(chunker, dataset, top_k=top_k) for chunker in chunkers]
    winner = max(
        reports,
        key=lambda report: (
            report["recall_at_k"],
            report["mrr"],
            report["evidence_coverage"],
            -report["avg_context_tokens"],
        ),
    )
    return {
        "evaluation_scope": "offline_chunk_boundary_comparison",
        "metric_note": "Latency is local harness overhead; it is not vector DB or model latency.",
        "winner": winner["strategy"],
        "strategies": reports,
    }
