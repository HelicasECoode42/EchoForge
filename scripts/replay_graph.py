from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.graph_replay import load_graph_cases, run_graph_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay EchoForge graph paths")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "replay" / "graph_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evidence" / "graph-replay-report.json")
    args = parser.parse_args()
    report = asyncio.run(run_graph_suite(load_graph_cases(args.cases)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
