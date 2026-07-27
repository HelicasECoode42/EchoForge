"""Real embedding + Chroma retrieval evaluation for chunking strategies."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence

import chromadb

from retrieval.chunking import Chunker
from retrieval.embedding import EmbeddingConfig, EmbeddingProvider


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


@dataclass(frozen=True)
class VectorQueryMetric:
    case_id: str
    first_relevant_rank: int
    recall_at_k: float
    reciprocal_rank: float
    evidence_coverage: float
    context_tokens: int
    query_embedding_latency_ms: float
    vector_search_latency_ms: float
    top_score: float
    score_margin: float


def evaluate_vector_chunker(
    chunker: Chunker,
    dataset: Dict[str, Any],
    embedding: EmbeddingProvider,
    *,
    top_k: int = 3,
    fast_path_min_score: float = 0.60,
    fast_path_min_margin: float = 0.08,
) -> Dict[str, Any]:
    client = chromadb.EphemeralClient(
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection(
        name=f"eval_{chunker.name}_{uuid.uuid4().hex[:8]}",
        metadata={"hnsw:space": embedding.config.distance},
    )

    chunks = []
    for document in dataset["documents"]:
        chunks.extend(chunker.chunk(
            document_id=document["id"],
            title=document["title"],
            text=document["content"],
            document_version=str(document.get("version", "1")),
        ))

    embedding_started = time.perf_counter()
    document_embeddings = embedding.embed_documents(chunk.content for chunk in chunks)
    document_embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000

    indexing_started = time.perf_counter()
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        embeddings=document_embeddings,
        metadatas=[chunk.metadata() for chunk in chunks],
    )
    vector_index_latency_ms = (time.perf_counter() - indexing_started) * 1000

    metrics: List[VectorQueryMetric] = []
    for case in dataset["cases"]:
        query_started = time.perf_counter()
        query_embedding = embedding.embed_queries([case["query"]])[0]
        query_embedding_latency_ms = (time.perf_counter() - query_started) * 1000

        search_started = time.perf_counter()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(chunks)),
            include=["documents", "metadatas", "distances"],
        )
        vector_search_latency_ms = (time.perf_counter() - search_started) * 1000

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scores = [1.0 - float(distance) for distance in distances]
        evidence = [item for item in case.get("evidence", []) if item]

        relevant_ranks = [
            rank
            for rank, (document, metadata) in enumerate(zip(documents, metadatas), start=1)
            if metadata.get("document_id") == case["relevant_document_id"]
            and any(item in document for item in evidence)
        ]
        first_rank = relevant_ranks[0] if relevant_ranks else 0
        selected_text = "\n".join(documents)
        coverage = (
            sum(1 for item in evidence if item in selected_text) / len(evidence)
            if evidence else 0.0
        )
        metrics.append(VectorQueryMetric(
            case_id=case["id"],
            first_relevant_rank=first_rank,
            recall_at_k=1.0 if first_rank else 0.0,
            reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
            evidence_coverage=coverage,
            context_tokens=sum(int(meta.get("token_count", 0)) for meta in metadatas),
            query_embedding_latency_ms=query_embedding_latency_ms,
            vector_search_latency_ms=vector_search_latency_ms,
            top_score=scores[0] if scores else -1.0,
            score_margin=(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
        ))

    fast_path_cases = [
        metric
        for metric in metrics
        if metric.top_score >= fast_path_min_score
        and metric.score_margin >= fast_path_min_margin
    ]
    if not metrics:
        return {
            "strategy": chunker.name,
            "retriever": "fastembed_query_passage_plus_chroma_hnsw",
            "embedding": embedding.config.metadata(),
            "top_k": top_k,
            "documents": len(dataset["documents"]),
            "queries": 0,
            "chunk_count": len(chunks),
            "document_embedding_latency_ms": round(document_embedding_latency_ms, 3),
            "vector_index_latency_ms": round(vector_index_latency_ms, 3),
            "avg_query_embedding_latency_ms": 0.0,
            "p95_query_embedding_latency_ms": 0.0,
            "avg_vector_search_latency_ms": 0.0,
            "p95_vector_search_latency_ms": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "evidence_coverage": 0.0,
            "avg_context_tokens": 0.0,
            "fast_path_calibration": {
                "min_score": fast_path_min_score,
                "min_margin": fast_path_min_margin,
                "eligible_queries": 0,
                "eligible_rate": 0.0,
                "top1_relevant_rate": 0.0,
                "note": "No query cases were supplied.",
            },
            "cases": [],
        }
    return {
        "strategy": chunker.name,
        "retriever": "fastembed_query_passage_plus_chroma_hnsw",
        "embedding": embedding.config.metadata(),
        "top_k": top_k,
        "documents": len(dataset["documents"]),
        "queries": len(metrics),
        "chunk_count": len(chunks),
        "document_embedding_latency_ms": round(document_embedding_latency_ms, 3),
        "vector_index_latency_ms": round(vector_index_latency_ms, 3),
        "avg_query_embedding_latency_ms": round(mean(m.query_embedding_latency_ms for m in metrics), 3),
        "p95_query_embedding_latency_ms": round(_percentile([m.query_embedding_latency_ms for m in metrics], 0.95), 3),
        "avg_vector_search_latency_ms": round(mean(m.vector_search_latency_ms for m in metrics), 3),
        "p95_vector_search_latency_ms": round(_percentile([m.vector_search_latency_ms for m in metrics], 0.95), 3),
        "recall_at_k": round(mean(m.recall_at_k for m in metrics), 4),
        "mrr": round(mean(m.reciprocal_rank for m in metrics), 4),
        "evidence_coverage": round(mean(m.evidence_coverage for m in metrics), 4),
        "avg_context_tokens": round(mean(m.context_tokens for m in metrics), 2),
        "fast_path_calibration": {
            "min_score": fast_path_min_score,
            "min_margin": fast_path_min_margin,
            "eligible_queries": len(fast_path_cases),
            "eligible_rate": round(len(fast_path_cases) / len(metrics), 4),
            "top1_relevant_rate": round(
                mean(1.0 if metric.first_relevant_rank == 1 else 0.0 for metric in fast_path_cases),
                4,
            ) if fast_path_cases else 0.0,
            "note": "Calibration evidence only; the dataset is too small for a production threshold claim.",
        },
        "cases": [asdict(metric) for metric in metrics],
    }


def run_vector_harness(
    chunkers: Sequence[Chunker],
    dataset: Dict[str, Any],
    embedding: EmbeddingProvider,
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    reports = [
        evaluate_vector_chunker(chunker, dataset, embedding, top_k=top_k)
        for chunker in chunkers
    ]
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
        "evaluation_scope": "real_local_embedding_and_chroma_retrieval",
        "metric_note": (
            "Latencies are local warm-process measurements on this machine; they exclude "
            "LLM rewrite/rerank, network and answer generation."
        ),
        "winner": winner["strategy"],
        "strategies": reports,
    }
