#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PYTEST="${PYTEST:-$ROOT/.venv/bin/pytest}"
if [[ ! -x "$PYTHON" || ! -x "$PYTEST" ]]; then
  echo "EchoForge virtualenv is required: $ROOT/.venv/bin/{python,pytest}" >&2
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
  "$PYTEST" -q "${TEST_PATHS[@]}"
else
  echo "No tracked pytest suites found; running replay and evaluation checks only."
fi
"$PYTHON" scripts/replay_routes.py --output "$(mktemp -t echoforge-route-replay.XXXXXX.json)"
"$PYTHON" scripts/replay_graph.py --output "$(mktemp -t echoforge-graph-replay.XXXXXX.json)"
"$PYTHON" scripts/evaluate_chunking.py --output "$(mktemp -t echoforge-chunking.XXXXXX.json)"
