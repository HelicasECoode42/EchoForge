# Evaluation gates

EchoForge treats retrieval, answerability and response verification as separate decisions.

## RAG golden set

`data/eval/rag_golden_set.v1.json` contains 40 repository-authored synthetic cases across eight categories. Twenty calibration cases select a lexical answerability threshold; twenty holdout cases report the primary result. Hard-negative no-answer cases deliberately share vocabulary with the four source documents.

Run:

```bash
python scripts/evaluate_rag_golden.py
```

The checked-in report is `data/evidence/rag-golden-report.json`. The current holdout retrieves every expected document within Top-3 before gating, but gated full Recall@3 is 0.625 and no-answer rejection is 0.750. The resulting `no-go` is intentional evidence: lexical relevance alone cannot decide whether the system should answer.

The harness does not evaluate generated answer quality, refusal wording, live vector behavior, latency or cost. Conflict and refusal labels identify downstream expectations; their retrieval metrics only ask whether the required evidence was present.

## Verifier audit

`data/eval/verifier_audit.v1.json` contains 24 labeled deterministic cases. The audit reproduces the legacy verifier in an isolated baseline and compares it with the current gate.

Run:

```bash
python scripts/audit_response_verifier.py --fail-on-dangerous-false-accept
```

The checked-in report is `data/evidence/verifier-audit-report.json`. The current gate removes all four labeled dangerous false accepts while reducing false blocks from five to two. The two remaining false blocks are kept visible rather than weakened away: deterministic lexical support cannot safely verify every paraphrase.

The dataset status remains `author_review_required`. These results are useful for regression and design review, not as production accuracy or semantic entailment claims.

## Request metrics

`monitor/request_metrics.py` exports bounded-label Prometheus metrics through `/metrics`:

- `echoforge_pipeline_requests_total{status}` for completed / blocked / failed;
- `echoforge_pipeline_request_duration_seconds{status}`;
- `echoforge_pipeline_stage_duration_seconds{stage}` for fixed graph nodes;
- `echoforge_verifier_outcomes_total{layer,status}`;
- `echoforge_memory_persistence_total{outcome}`.

Prompts, user IDs, evidence text and trace IDs are forbidden as labels. A blocked or failed response must appear as skipped persistence, making the verifier-to-memory gate observable at request level.

## Release boundary

Passing either deterministic gate does not authorize deployment. A production decision still requires human-reviewed domain data, live retrieval and model runs, repeated latency/cost measurements, tail-risk cases, privacy review, and a stated permission scope.
