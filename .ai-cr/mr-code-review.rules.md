# EchoForge MR review rules

## Blocking rules

- Do not persist a response unless the graph reaches `complete` after independent verification.
- Do not equate `trace.success`, non-empty text or retrieval hit with task completion.
- Retrieval-required responses must preserve evidence IDs and validate citations against the current retrieval set.
- Graph edge, terminal-state and verifier changes require deterministic regression tests.
- Do not add secrets, raw prompts or raw user text to traces, fixtures or reports.

## Required review questions

1. What new observation or evidence proves the changed behavior?
2. Which `completed`, `blocked` and `failed` paths were tested?
3. Does the change expand permissions, side effects or persistence scope?
4. If a verifier changed, which previous false-positive or false-negative does the test cover?
