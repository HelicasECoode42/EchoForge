from mcp.knowledge_base import KnowledgeBase


class _Collection:
    def get(self, ids, include):
        docs = {
            "a": "前一段证据",
            "b": "命中的核心证据",
            "c": "后一段证据",
        }
        metas = {
            "a": {"chunk_id": "a"},
            "b": {"chunk_id": "b"},
            "c": {"chunk_id": "c"},
        }
        return {"documents": [docs[i] for i in ids], "metadatas": [metas[i] for i in ids]}


def test_neighbor_expansion_is_budgeted():
    kb = object.__new__(KnowledgeBase)
    kb._neighbor_radius = 1
    kb._expansion_token_budget = 100
    kb._collection = _Collection()
    item = {
        "chunk_id": "b",
        "content": "命中的核心证据",
        "previous_chunk_id": "a",
        "next_chunk_id": "c",
    }
    expanded = kb._expand_with_neighbors(item)
    assert expanded["expanded_context"] is True
    assert expanded["source_chunk_ids"] == ["a", "b", "c"]
