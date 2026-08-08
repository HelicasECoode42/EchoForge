# Evidence lineage

Retrieval results retain `chunk_id`, title, content and score. The same chunk IDs are rendered into model context, passed in the structured-answer contract, stored in route evidence, and checked by `GroundingVerifier` before memory persistence.
