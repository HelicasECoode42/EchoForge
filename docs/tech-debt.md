# Technical debt inventory

This is a living, evidence-backed list. Add an owner and a regression test before removing an item.

| Priority | Item | Evidence / impact | Next action |
|---|---|---|---|
| P1 | Live vector retrieval is not a default CI gate | Requires provider/model dependencies and can be environment-sensitive | Keep deterministic chunking in CI; run vector eval as an opt-in job with a report |
| P1 | API integration tests need isolated service fakes | `/chat` depends on memory, retrieval and provider wiring | Add graph-level fake integration fixtures before enabling HTTP E2E |
| P2 | Legacy evaluation scripts have separate CLI conventions | Replay and retrieval scripts use different output flags | Route all deterministic checks through `scripts/ci_check.sh` |
| P2 | Structured-answer parsing remains compatibility-first | Plain text is accepted outside retrieval flows | Add contract cases for malformed JSON and partial fields |
| P3 | Module context can drift from implementation | AGENTS files are maintained manually | Review module indexes quarterly and on boundary changes |
