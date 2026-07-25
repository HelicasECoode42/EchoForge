from __future__ import annotations

from dataclasses import replace

import chromadb

from mcp.knowledge_base import KnowledgeBase
from retrieval.chunking import FixedCharacterChunker
from retrieval.embedding import EmbeddingConfig


def test_embedding_fingerprint_changes_with_model_or_dimension():
    base = EmbeddingConfig()
    assert base.fingerprint != replace(base, model="another-model").fingerprint
    assert base.fingerprint != replace(base, dimensions=384).fingerprint
    assert base.fingerprint == replace(base, cache_dir="/another/machine/cache").fingerprint


def test_embedding_metadata_is_complete_for_reindex_audit():
    config = EmbeddingConfig()
    metadata = config.metadata()
    assert metadata["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert metadata["embedding_provider_version"] == "0.8.0"
    assert metadata["embedding_dimensions"] == 512
    assert metadata["embedding_distance"] == "cosine"
    assert metadata["embedding_fingerprint"] == config.fingerprint


class FakeEmbeddingProvider:
    def __init__(self):
        self.config = EmbeddingConfig(
            provider="fastembed",
            model="fake-test-model",
            dimensions=2,
            cache_dir="unused",
        )

    @staticmethod
    def _embed(text):
        return [1.0, 0.0] if "退款" in text else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_queries(self, texts):
        return [self._embed(text) for text in texts]


def test_knowledge_base_supplies_explicit_embeddings_to_chroma():
    provider = FakeEmbeddingProvider()
    kb = KnowledgeBase(
        client=chromadb.EphemeralClient(
            settings=chromadb.Settings(anonymized_telemetry=False),
        ),
        chunker=FixedCharacterChunker(chunk_size=200),
        embedding_config=provider.config,
        embedding_provider=provider,
        load_default_docs=False,
    )
    kb.add_documents([
        {"id": "refund", "title": "退款", "content": "退款将在三天内到账。"},
        {"id": "login", "title": "登录", "content": "登录失败请重置密码。"},
    ])

    results = kb.search("退款多久到账", top_k=2)

    assert results[0]["title"] == "退款"
    assert results[0]["score"] == 1.0
    assert kb.embedding_info["embedding_dimensions"] == 2
