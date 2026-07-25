"""CLI for deterministic routing replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.route_replay import load_replay_cases, run_replay_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay EchoMind route decisions without an LLM")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "data" / "replay" / "route_cases.json"),
        help="JSON replay case file",
    )
    parser.add_argument("--output", help="Optional report JSON path")
    args = parser.parse_args()

    report = asyncio.run(run_replay_suite(load_replay_cases(args.cases)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
