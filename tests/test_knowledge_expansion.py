from mcp.knowledge_base import KnowledgeBase


class _Collection:
    def get(self, ids, include):
        docs = {
            "a": "前一段证据",
            "b": "命中的核心证据",
            "c": "后一段证据",
        }
        metas = {
            "a": {"chunk_id": "a", "previous_chunk_id": ""},
            "b": {"chunk_id": "b", "previous_chunk_id": "a", "next_chunk_id": "c"},
            "c": {"chunk_id": "c", "next_chunk_id": ""},
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


def test_neighbor_expansion_follows_the_configured_radius():
    class ChainCollection(_Collection):
        def get(self, ids, include):
            docs = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
            metas = {
                "a": {"chunk_id": "a", "previous_chunk_id": ""},
                "b": {"chunk_id": "b", "previous_chunk_id": "a", "next_chunk_id": "c"},
                "c": {"chunk_id": "c", "previous_chunk_id": "b", "next_chunk_id": "d"},
                "d": {"chunk_id": "d", "previous_chunk_id": "c", "next_chunk_id": "e"},
                "e": {"chunk_id": "e", "previous_chunk_id": "d", "next_chunk_id": ""},
            }
            return {"documents": [docs[i] for i in ids], "metadatas": [metas[i] for i in ids]}

    kb = object.__new__(KnowledgeBase)
    kb._neighbor_radius = 2
    kb._expansion_token_budget = 100
    kb._collection = ChainCollection()
    expanded = kb._expand_with_neighbors({
        "chunk_id": "c", "content": "C", "previous_chunk_id": "b", "next_chunk_id": "d",
    })

    assert expanded["source_chunk_ids"] == ["a", "b", "c", "d", "e"]
