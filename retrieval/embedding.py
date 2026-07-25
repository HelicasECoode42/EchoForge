"""Explicit, versioned embedding configuration for EchoForge retrieval.

The previous implementation delegated embedding selection to Chroma's default.
That made model upgrades, vector dimensions and index migration implicit.  This
module keeps those choices in application code and exposes a stable fingerprint
that can be attached to every collection and evaluation report.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Protocol


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "fastembed"
    provider_version: str = "0.8.0"
    model: str = "BAAI/bge-small-zh-v1.5"
    dimensions: int = 512
    distance: str = "cosine"
    cache_dir: str = "./data/models/fastembed"
    schema_version: str = "2"

    @property
    def fingerprint(self) -> str:
        # cache_dir is an operational path and does not change vector semantics.
        semantic_config = asdict(self)
        semantic_config.pop("cache_dir")
        payload = json.dumps(semantic_config, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def metadata(self) -> Dict[str, str | int]:
        return {
            "embedding_provider": self.provider,
            "embedding_provider_version": self.provider_version,
            "embedding_model": self.model,
            "embedding_dimensions": self.dimensions,
            "embedding_distance": self.distance,
            "embedding_schema_version": self.schema_version,
            "embedding_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "fastembed"),
            provider_version=os.getenv("EMBEDDING_PROVIDER_VERSION", "0.8.0"),
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "512")),
            distance=os.getenv("EMBEDDING_DISTANCE", "cosine"),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "./data/models/fastembed"),
            schema_version=os.getenv("EMBEDDING_SCHEMA_VERSION", "2"),
        )


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]: ...

    def embed_queries(self, texts: Iterable[str]) -> List[List[float]]: ...


class FastEmbedProvider:
    """Local ONNX BGE embeddings with separate query/passage encoders."""

    def __init__(self, config: EmbeddingConfig):
        if config.provider != "fastembed":
            raise ValueError(f"unsupported embedding provider: {config.provider}")
        if config.distance != "cosine":
            raise ValueError("EchoForge currently calibrates scores only for cosine distance")

        from fastembed import TextEmbedding

        self.config = config
        self._model = TextEmbedding(
            model_name=config.model,
            cache_dir=config.cache_dir,
            lazy_load=True,
        )
        self._lock = threading.Lock()

    def _as_lists(self, vectors: Iterable[object]) -> List[List[float]]:
        result = [vector.tolist() for vector in vectors]  # type: ignore[attr-defined]
        if result and len(result[0]) != self.config.dimensions:
            raise ValueError(
                f"embedding dimension mismatch: configured={self.config.dimensions}, "
                f"actual={len(result[0])}, model={self.config.model}"
            )
        return result

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        with self._lock:
            return self._as_lists(self._model.passage_embed(values))

    def embed_queries(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        with self._lock:
            return self._as_lists(self._model.query_embed(values))


@lru_cache(maxsize=4)
def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    return FastEmbedProvider(config)
