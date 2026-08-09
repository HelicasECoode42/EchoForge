# Retrieval module context

Retrieval owns chunking, embeddings and knowledge-base search.

Every usable result must retain a stable `chunk_id` (or `source_chunk_ids`). The ID is passed to the model context and preserved through the response and route trace. A retrieval hit without lineage is not usable evidence for automatic completion.

Changes to chunking or ranking require the deterministic chunking evaluation and, when dependencies are available, vector retrieval evaluation.
