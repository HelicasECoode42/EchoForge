from types import SimpleNamespace

from core.response_verifier import verify_orchestrator_result


def _result(**kwargs):
    trace = SimpleNamespace(trace_id="t1", stop_reason="completed", success=True, attempts=1, reroutes=0)
    values = dict(response="verified answer", success=True, route_trace=trace)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_response_verifier_accepts_external_route_and_text_evidence():
    result = verify_orchestrator_result(_result())
    assert result.status == "completed"
    assert result.reason == "execution_grounding_task_verified"
    assert result.execution_status == "verified"
    assert result.grounding_status == "not_required"
    assert result.task_status == "verified"


def test_response_verifier_blocks_missing_route_instead_of_persisting():
    result = verify_orchestrator_result(_result(route_trace=None))
    assert result.status == "blocked"
    assert result.reason == "missing_route_trace"


def test_response_verifier_requires_evidence_for_retrieval_question():
    result = verify_orchestrator_result(_result(), knowledge_required=True)
    assert result.status == "blocked"
    assert result.reason == "insufficient_evidence"
    assert result.grounding_status == "blocked"


def test_response_verifier_accepts_retrieved_supporting_citation():
    result = _result(
        response="退款需要 3-5 个工作日",
        citations=["chunk_12"],
    )
    verification = verify_orchestrator_result(
        result,
        evidence_ids=["chunk_12"],
        evidence_items={"chunk_12": {"content": "退款到账通常需要 3-5 个工作日。"}},
        knowledge_required=True,
    )
    assert verification.status == "completed"
    assert verification.reason == "execution_grounding_task_verified"
    assert verification.grounding_status == "verified"


def test_response_verifier_rejects_citation_outside_retrieval():
    verification = verify_orchestrator_result(
        _result(response="退款需要 3-5 个工作日", citations=["chunk_other"]),
        evidence_ids=["chunk_12"],
        evidence_items={"chunk_12": {"content": "退款到账通常需要 3-5 个工作日。"}},
        knowledge_required=True,
    )
    assert verification.status == "blocked"
    assert verification.reason == "citation_not_in_retrieval"


def test_response_verifier_blocks_task_that_requests_human():
    verification = verify_orchestrator_result(_result(needs_human=True))
    assert verification.status == "blocked"
    assert verification.reason == "needs_human"
    assert verification.task_status == "blocked"
