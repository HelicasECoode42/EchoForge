from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.chunking_harness import load_dataset, run_harness
from retrieval.chunking import FixedCharacterChunker, SlidingWindowChunker, StructureAwareTokenChunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EchoForge chunking strategies")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "chunking" / "cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evidence" / "chunking-report.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    report = run_harness(
        [
            FixedCharacterChunker(chunk_size=220),
            SlidingWindowChunker(chunk_size=220, overlap=60),
            StructureAwareTokenChunker(max_tokens=140),
        ],
        load_dataset(args.dataset),
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
