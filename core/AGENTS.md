# Core module context

Core contains bounded graph execution, replay boundaries and independent response verification.

Invariants:

- Execution is bounded by node, transition and total-runtime budgets.
- `blocked` means a known missing prerequisite or business condition; `failed` means execution failure.
- Verifier outcomes are evidence-bearing and must not depend only on model self-report.
- Any response-contract change requires tests for success, blocked and failed paths.
