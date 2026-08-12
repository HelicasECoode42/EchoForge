# Offline proposal-based improvement

EchoForge now supports a bounded, offline improvement loop:

```text
privacy-safe trace
  -> deterministic failure classification
  -> versioned proposal
  -> isolated replay/evaluation
  -> candidate or rejected
  -> explicit human approval
```

## Stage delivery

| Stage | Delivery | Independent acceptance |
|---|---|---|
| 1 | `evaluation/improvement.py` domain model, failure buckets, metric directions and approval state machine | `tests/test_improvement_domain.py` |
| 2 | `evaluation/improvement_harness.py` chunking proposal replay | `tests/test_improvement_harness.py` |
| 3 | `ProposalStore`, CLI and `/improvement/*` API | `tests/test_proposal_store.py`, `tests/test_improvement_api.py` |
| 4 | fixed improved/regressed fixtures, documentation and full deterministic gate | `scripts/evaluate_improvement.py` and `bash scripts/ci_check.sh` |

## Technology and current scope

- Python 3 with explicit `dataclass` contracts and deterministic functions;
- existing chunking harness and `data/chunking/cases.json` as the first replay adapter;
- JSON report plus an atomic local JSON proposal ledger;
- FastAPI endpoints for evaluation, listing, approval and rejection;
- pytest and the existing CI replay scripts.

The currently implemented replay target is `chunk_strategy`. Supported
strategies are `fixed_char`, `sliding_window` and `structure_token`. Other
proposal targets remain valid as discussion metadata, but replay returns an
explicit unsupported-target error until an isolated evaluator is implemented.

For this retrieval harness, a case with zero evidence coverage counts as
blocked. Partial coverage remains visible as a quality metric. Retrieval
latency is the existing local harness overhead, not a production latency
claim; a small tolerance is used for the regression gate. Context tokens are
used as the deterministic token-cost proxy.

## Run the acceptance suite

```bash
cd /Users/wangyufan/Desktop/code/EchoForge/EchoForge
./.venv/bin/python scripts/evaluate_improvement.py
```

The built-in suite contains two explicit examples:

- `fixed_char -> structure_token`: evidence coverage/recall improve and token
  cost falls, so the proposal becomes `candidate`;
- `structure_token -> sliding_window`: token cost increases, so the proposal
  is `rejected` despite being a plausible retrieval change.

To persist review artifacts locally:

```bash
./.venv/bin/python scripts/evaluate_improvement.py \
  --store /tmp/echoforge-proposals.json \
  --output /tmp/echoforge-improvement-report.json
```

To generate proposals from privacy-safe trace metadata:

```bash
./.venv/bin/python scripts/evaluate_improvement.py \
  --traces /tmp/privacy-safe-traces.json \
  --store /tmp/echoforge-proposals.json
```

A candidate can only be marked approved by an explicit human action:

```bash
./.venv/bin/python scripts/evaluate_improvement.py \
  --store /tmp/echoforge-proposals.json \
  --approve proposal-<id>
```

## API surface

- `POST /improvement/evaluate` — replay a proposal against the configured
  offline dataset;
- `GET /improvement/proposals` — list stored proposals, optionally filtered by
  status;
- `POST /improvement/proposals/generate` — cluster privacy-safe trace metadata
  and create deterministic recipe proposals; repeated generation returns the
  existing ledger artifact for the same proposal identity;
- `POST /improvement/proposals/{proposal_id}/approve` — record human approval;
- `POST /improvement/proposals/{proposal_id}/reject` — record human rejection.

The API only writes the offline review ledger. There is intentionally no
`apply`, `publish`, production config mutation, prompt mutation or memory write
operation in this feature.

Each proposal keeps its latest evaluation plus an append-only
`evaluation_history`. Re-running a mutable proposal with a different dataset,
baseline configuration, or `top_k` therefore preserves the prior evaluation
context instead of silently replacing the evidence.

These endpoints are local/controlled-offline APIs, not production approval
services. They are disabled when `APP_ENV=production`; they have no
authentication layer, and `ProposalStore` only provides a process-local lock.
Use a separate authenticated review service and a multi-process storage/locking
design before exposing this workflow outside a controlled environment.

## Safety boundaries

- Proposal provenance uses trace IDs and structured metadata; raw user text and
  prompts are not added to the proposal ledger.
- The first proposal generator is deterministic and recipe-based. It never
  asks an LLM to judge its own output or invent a release decision.
- Candidate status requires the offline regression gate to pass.
- Approval does not change the live runtime; a future publish workflow would
  need a separate authorization boundary and deployment review.
- LLM self-evaluation, a single user like, or a single successful response is
  not a release signal.
- No external model call is required for the current replay adapter.
