# EchoForge execution graph and chunking harness

EchoForge is a reliability harness around an Agent request path. The product
API is not described as "Graph Engineering" merely because it contains if/else
routing: `/chat` now executes the validated directed graph below.

```text
load_memory -> decide_retrieval -> retrieve ------> execute_agent
                                \-----------------> execute_agent
execute_agent -> persist_memory -> complete
```

## Graph runtime

`core/execution_graph.py` provides typed node and edge specifications,
conditional guards, per-node timeout/retry policy, mutable graph state and
three hard budgets:

- maximum node executions;
- maximum edge transitions;
- maximum total runtime.

Validation rejects missing nodes, unreachable nodes and non-terminal dead
ends. Cycles are allowed only because execution budgets guarantee a stop.
Every run records node attempts, selected transitions, error types,
`stop_reason` and per-node/total latency. Graph state and user prompts are not
written to the graph JSONL evidence file.

The production definition is available at `GET /graph`, and recent execution
paths are available at `GET /graph/traces/recent`. `/chat` also returns the
graph trace id, stop reason, total pipeline latency and node timing map.

Run the no-model regression suite with:

```bash
python scripts/replay_graph.py
```

The five deterministic cases cover retrieval/skip branches, one bounded
retrieval retry, Agent failure preventing persistence and transition-budget
exhaustion. The generated report is an offline control-path result, not a live
LLM success-rate or production-latency claim.

## Chunking strategies

The previous implementation grouped sentences under a 500-character target.
It had no overlap, did not enforce a hard limit for oversized sentences and
stored only title/index metadata. `retrieval/chunking.py` replaces it with
three explicit strategies:

1. `fixed_char`: strict non-overlapping character baseline;
2. `sliding_window`: character windows with overlap;
3. `structure_token`: Markdown/paragraph-aware packing under an estimated
   token budget.

All strategies produce stable chunk ids and record document version, character
offsets, estimated token count, content SHA-256, section path and neighbor ids.
The production knowledge base defaults to `structure_token` and can be changed
with `CHUNK_STRATEGY`.

Run the controlled comparison with:

```bash
python scripts/evaluate_chunking.py
```

The checked-in dataset contains four documents and eight queries. At Top-3,
the first run produced:

| Strategy | Recall@3 | MRR | Evidence coverage | Avg context tokens |
| --- | ---: | ---: | ---: | ---: |
| fixed_char | 0.875 | 0.8281 | 0.875 | 428.50 |
| sliding_window | 1.000 | 1.0000 | 1.000 | 467.88 |
| structure_token | 1.000 | 1.0000 | 1.000 | 339.12 |

The harness uses a deterministic lexical proxy to isolate boundary behavior.
Its timing numbers are local Python overhead and must not be presented as
ChromaDB, embedding, rerank or live end-to-end latency. A production decision
still requires the same query set to be replayed through the real embedding
and rerank path.
