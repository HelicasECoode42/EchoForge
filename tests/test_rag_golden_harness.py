from pathlib import Path

from evaluation.rag_golden_harness import load_golden_set, run_golden_harness


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval" / "rag_golden_set.v1.json"


def test_golden_set_has_versioned_balanced_coverage():
    dataset = load_golden_set(DATASET)
    assert dataset["version"] == "rag-golden.v1"
    assert len(dataset["cases"]) == 40
    assert {case["split"] for case in dataset["cases"]} == {"calibration", "holdout"}
    assert {
        "normal",
        "no_answer",
        "conflict",
        "expired",
        "refusal",
        "multi_hop",
        "temporal",
        "cross_session",
    } <= {case["category"] for case in dataset["cases"]}


def test_golden_harness_reports_holdout_and_boundaries():
    report = run_golden_harness(DATASET, top_k=3)
    assert report["cases"] == 40
    assert report["calibration"]["cases"] == 20
    assert report["holdout"]["cases"] == 20
    assert report["threshold_selection"]["split"] == "calibration"
    assert report["dataset_status"] == "author_review_required"
    assert "does not authorize production use" in report["readiness_gate"]["boundary"]


def test_no_answer_cases_are_not_counted_as_retrieval_recall_wins():
    report = run_golden_harness(DATASET, top_k=3)
    no_answer = report["categories"]["no_answer"]
    assert no_answer["answerable_cases"] == 0
    assert no_answer["no_answer_cases"] == 8
    assert no_answer["full_document_recall_at_3"] is None
