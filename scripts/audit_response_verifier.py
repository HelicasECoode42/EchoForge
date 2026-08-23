from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.verifier_audit import run_verifier_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EchoForge verifier against labeled cases")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "eval" / "verifier_audit.v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "evidence" / "verifier-audit-report.json",
    )
    parser.add_argument(
        "--fail-on-dangerous-false-accept",
        action="store_true",
        help="Exit non-zero when a labeled blocked/failed case is accepted as completed",
    )
    args = parser.parse_args()
    report = run_verifier_audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_dangerous_false_accept and report["metrics"]["dangerous_false_accepts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
