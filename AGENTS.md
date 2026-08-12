# EchoForge AI Context Index

## Architecture

- `api/main.py` — chat request graph, retrieval context, and terminal state handling.
- `core/execution_graph.py` — bounded graph execution, budgets, trace and replay boundaries.
- `core/response_verifier.py` — independent execution, grounding and task verifiers.
- `agents/agent_orchestrator.py` — routing, structured answer parsing and route evidence.
- `evidence/route_trace.py` — privacy-preserving route trace with evidence/citation metadata.
- `evaluation/improvement.py` and `evaluation/improvement_harness.py` — offline proposal contracts, failure classification and isolated replay.

## Knowledge index

- `docs/business-context.md` — project scope and non-goals.
- `docs/architecture/verification-boundary.md` — execution, grounding and task completion contract.
- `docs/standards/engineering.md` — coding, state and evidence rules.
- `docs/integrations.md` — external providers and local deterministic substitutes.
- `docs/tech-debt.md` — bounded debt inventory and next cleanup targets.
- `api/AGENTS.md`, `core/AGENTS.md`, `agents/AGENTS.md`, `retrieval/AGENTS.md`, `memory/AGENTS.md` — module-specific context.

## Runtime contract

- Every graph run terminates in `complete`, `blocked` or `failed`.
- Only a completed, independently verified response may be persisted to memory.
- `route success` is not `answer success`.
- `answer non-empty` is not `task completed`.
- `retrieval hit` is not `evidence supports answer`.
- Blocked and failed responses must not be persisted.

## Tests and change rules

- Run the full deterministic gate with `bash scripts/ci_check.sh`.
- Run tests directly with `PYTHONPATH=. ./.venv/bin/pytest -q core/test_response_verifier.py tests`.
- Changes to graph transitions require graph/replay regression tests.
- Response contract or verifier changes require verifier tests.
- Retrieval/evidence changes require evidence-ID and grounding tests.
- Offline improvement changes require proposal state, metric-direction and improved/regressed replay tests.
- Keep raw user text and prompts out of persisted route traces.
- New reusable conclusions must be added to the relevant module `AGENTS.md` and linked here.
