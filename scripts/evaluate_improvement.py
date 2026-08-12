#!/usr/bin/env python3
"""Run the deterministic offline proposal review suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.improvement import ImprovementProposal, generate_proposals_from_traces
from evaluation.improvement_harness import (
    load_replay_dataset,
    run_default_suite,
    summarize_case_deltas,
)
from evaluation.proposal_store import ProposalStore


def validate_default_acceptance(payload: dict) -> None:
    """Fail the command if the two required acceptance examples drift."""
    statuses = {item["proposal"]["status"] for item in payload.get("proposals", [])}
    summary = payload.get("summary", {})
    if statuses != {"candidate", "rejected"}:
        raise RuntimeError(f"unexpected default proposal statuses: {sorted(statuses)}")
    if summary.get("improved_case_count", 0) < 1:
        raise RuntimeError("default suite has no improved case evidence")
    if summary.get("regressed_case_count", 0) < 1:
        raise RuntimeError("default suite has no regressed case evidence")


def build_report(dataset_path: Path) -> dict:
    dataset = load_replay_dataset(dataset_path)
    reports = run_default_suite(dataset, dataset_id=dataset_path.stem)
    return {
        "evaluation_scope": "offline_proposal_based_improvement",
        "dataset": str(dataset_path),
        "proposals": [report.to_dict() for report in reports],
        "summary": summarize_case_deltas(reports),
    }


def build_proposal_report(trace_path: Path) -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    traces = payload.get("traces") if isinstance(payload, dict) else payload
    if not isinstance(traces, list):
        raise ValueError("trace input must be a JSON list or an object with a traces list")
    proposals = generate_proposals_from_traces(traces)
    return {
        "evaluation_scope": "offline_proposal_generation",
        "traces": str(trace_path),
        "proposals": [{"proposal": proposal.to_dict()} for proposal in proposals],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "chunking" / "cases.json",
        help="deterministic replay dataset",
    )
    parser.add_argument("--traces", type=Path, help="optional privacy-safe trace JSON for proposal generation")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--store", type=Path, help="optional proposal ledger path")
    parser.add_argument("--approve", help="approve one candidate already stored in the ledger")
    args = parser.parse_args()

    if args.approve:
        if not args.store:
            parser.error("--approve requires --store")
        approved = ProposalStore(args.store).approve(args.approve)
        payload = {"proposal": approved.to_dict(), "action": "approved"}
    else:
        payload = build_proposal_report(args.traces) if args.traces else build_report(args.dataset)
        if not args.traces:
            validate_default_acceptance(payload)
        if args.store:
            store = ProposalStore(args.store)
            for item in payload["proposals"]:
                store.save(ImprovementProposal.from_dict(item["proposal"]))

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
