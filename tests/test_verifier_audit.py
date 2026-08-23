from pathlib import Path

from evaluation.verifier_audit import load_audit_cases, run_verifier_audit


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval" / "verifier_audit.v1.json"


def test_verifier_audit_is_versioned_and_labeled():
    dataset = load_audit_cases(DATASET)
    assert dataset["version"] == "verifier-audit.v1"
    assert dataset["status"] == "author_review_required"
    assert len(dataset["cases"]) == 24
    assert {case["expected_status"] for case in dataset["cases"]} == {"completed", "blocked", "failed"}


def test_verifier_audit_reports_false_accepts_separately():
    report = run_verifier_audit(DATASET)
    assert report["cases"] == 24
    assert report["legacy_metrics"]["dangerous_false_accepts"] >= 1
    assert report["metrics"]["dangerous_false_accepts"] == 0
    assert report["metrics"]["accept_precision"] > report["legacy_metrics"]["accept_precision"]
    assert "semantic entailment" in report["boundary"]
