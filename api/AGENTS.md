# API module context

`api/main.py` owns the production chat graph and HTTP/SSE adapters.

Invariants:

- Graph terminal states are `complete`, `blocked`, and `failed`.
- `persist_memory` is reachable only after `verification_status == completed`.
- Streaming tokens are buffered until verification succeeds.
- API responses expose graph trace, route trace, evidence IDs, citations and verifier statuses.

When changing graph edges, update `tests/test_execution_graph.py` or add a graph replay case.
