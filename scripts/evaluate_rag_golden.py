from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.rag_golden_harness import run_golden_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the versioned EchoForge RAG golden set")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "eval" / "rag_golden_set.v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "evidence" / "rag-golden-report.json",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    report = run_golden_harness(args.dataset, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
