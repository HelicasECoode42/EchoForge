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

## Explicit embedding and real vector evaluation

Production retrieval no longer delegates model selection to Chroma defaults.
`retrieval/embedding.py` fixes the provider, provider version, Chinese model,
vector dimension and cosine distance. The semantic configuration produces a
fingerprint that is stored in collection metadata and the collection name.
Changing the model, dimension or distance therefore creates a new index rather
than silently mixing incompatible vectors. Cache paths are excluded from the
fingerprint because they do not affect vector semantics.

The default model is `BAAI/bge-small-zh-v1.5` through FastEmbed 0.8.0. Document
and query vectors are generated explicitly in the application and supplied to
Chroma; Chroma is responsible only for vector indexing and nearest-neighbor
search. `/knowledge/stats` exposes the active embedding metadata.

Run the real local retrieval comparison with:

```bash
python scripts/evaluate_vector_retrieval.py
```

The current 4-document/8-query controlled set produced this warm-process run:

| Strategy | Recall@3 | MRR | Evidence coverage | Avg context tokens |
| --- | ---: | ---: | ---: | ---: |
| fixed_char | 1.000 | 0.8125 | 0.9375 | 395.62 |
| sliding_window | 0.875 | 0.8125 | 0.8750 | 477.00 |
| structure_token | 1.000 | 0.8125 | 1.0000 | 332.38 |

This supports `structure_token` for the checked-in set: it preserved full
Recall@3 and evidence coverage while using about 16% fewer context tokens than
`fixed_char` and about 30% fewer than `sliding_window`. The set is intentionally
small and is not a production-quality generalization claim. The report includes
per-query scores, margins, embedding latency and Chroma search latency in
`data/evidence/vector-retrieval-report.json`.

## Adaptive rewrite/rerank policy

The previous pipeline paid for query rewrite and LLM rerank on every retrieval.
The current policy first executes one vector query. It uses a conservative
score-plus-margin gate to return clearly separated results directly; ambiguous
queries retain multi-query recall and reranking. Original-query results are
reused, duplicate chunks use stable ids, and at most 12 candidates are sent to
the reranker.

On the checked-in real-vector set, the current `0.60` score / `0.08` margin gate
selected 2 of 8 `structure_token` queries, and both had the relevant evidence at
rank 1. This is calibration evidence only, not enough data to freeze a production
threshold.

Run the controlled call-count benchmark with:

```bash
python scripts/benchmark_retrieval_policy.py
```

With injected deterministic delays, the high-confidence path reduced calls from
4 vector + 1 rewrite + 1 rerank to 1 vector + 0 rewrite + 0 rerank while keeping
the same top chunk. Median latency changed from 69.099 ms to 3.461 ms in that
simulation. The 94.99% number is not live provider latency and must not be used
as a resume production metric; the durable claim is that two LLM stages are
skipped for gated queries without removing the full path for ambiguous ones.
