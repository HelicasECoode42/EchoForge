#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$(command -v "$PYTHON" 2>/dev/null || true)"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
fi

if [[ -n "${PYTEST:-}" ]]; then
  PYTEST_BIN="$(command -v "$PYTEST" 2>/dev/null || true)"
elif [[ -x "$ROOT/.venv/bin/pytest" ]]; then
  PYTEST_BIN="$ROOT/.venv/bin/pytest"
else
  PYTEST_BIN="$(command -v pytest 2>/dev/null || true)"
fi

if [[ -z "$PYTHON_BIN" || -z "$PYTEST_BIN" ]]; then
  echo "EchoForge requires python and pytest from .venv or PATH" >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TEST_PATHS=()
if git ls-files --error-unmatch tests >/dev/null 2>&1; then
  TEST_PATHS+=(tests)
fi
if git ls-files --error-unmatch core/test_response_verifier.py >/dev/null 2>&1; then
  TEST_PATHS+=(core/test_response_verifier.py)
fi

if (( ${#TEST_PATHS[@]} > 0 )); then
  "$PYTEST_BIN" -q "${TEST_PATHS[@]}"
else
  echo "No tracked pytest suites found; running replay and evaluation checks only."
fi
"$PYTHON_BIN" scripts/replay_routes.py --output "$(mktemp -t echoforge-route-replay.XXXXXX.json)"
"$PYTHON_BIN" scripts/replay_graph.py --output "$(mktemp -t echoforge-graph-replay.XXXXXX.json)"
"$PYTHON_BIN" scripts/evaluate_chunking.py --output "$(mktemp -t echoforge-chunking.XXXXXX.json)"
