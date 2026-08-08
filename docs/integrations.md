# Integrations

- LLM provider: configured through the existing environment settings and wrapped by `core/model_response.py`.
- Knowledge retrieval: `mcp/tool_manager.py` and `mcp/knowledge_base.py`.
- Memory: `memory/`.
- Deterministic substitutes: `evaluation/route_replay.py`, `evaluation/graph_replay.py` and the local verifier tests.

CI must use deterministic substitutes. Live provider and vector-store checks are opt-in evaluation jobs, not merge blockers.
