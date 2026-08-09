# Replay and regression entry points

Run deterministic verifier and graph regressions from this directory:

```bash
PYTHONPATH=. ./.venv/bin/pytest -q core/test_response_verifier.py tests
```

Changes to graph edges, terminal states or response contracts should add a deterministic test before changing replay/evaluation behavior.
