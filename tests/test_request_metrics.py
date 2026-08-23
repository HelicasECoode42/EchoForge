from types import SimpleNamespace

from prometheus_client import CollectorRegistry, generate_latest

from monitor.request_metrics import RequestMetrics


def _result(status: str, *, persisted: bool):
    verification = SimpleNamespace(checks=(
        {"verifier": "execution", "status": "verified", "reason": "not_exported"},
        {"verifier": "grounding", "status": "blocked" if status == "blocked" else "verified", "reason": "not_exported"},
    ))
    return SimpleNamespace(
        state={
            "pipeline_status": status,
            "verification": verification,
            "memory_persisted": persisted,
            "request": "must-not-be-exported",
        },
        trace=SimpleNamespace(
            stop_reason="completed",
            total_latency_ms=1250,
            node_timings_ms={"execute_agent": 1000, "verify_response": 20},
            trace_id="must-not-be-exported",
        ),
    )


def test_request_metrics_record_completed_and_persisted_without_sensitive_labels():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.record(_result("completed", persisted=True))
    payload = generate_latest(registry).decode()
    assert 'echoforge_pipeline_requests_total{status="completed"} 1.0' in payload
    assert 'echoforge_memory_persistence_total{outcome="persisted"} 1.0' in payload
    assert 'echoforge_verifier_outcomes_total{layer="execution",status="verified"} 1.0' in payload
    assert "must-not-be-exported" not in payload


def test_request_metrics_make_blocked_persistence_skip_visible():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.record(_result("blocked", persisted=False))
    payload = generate_latest(registry).decode()
    assert 'echoforge_pipeline_requests_total{status="blocked"} 1.0' in payload
    assert 'echoforge_memory_persistence_total{outcome="skipped_blocked"} 1.0' in payload
    assert 'echoforge_verifier_outcomes_total{layer="grounding",status="blocked"} 1.0' in payload
