"""Privacy-safe request-level metrics for the verified chat pipeline."""

from __future__ import annotations

from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram


_PIPELINE_STATUSES = {"completed", "blocked", "failed"}
_VERIFIER_STATUSES = {"verified", "blocked", "failed", "not_checked", "not_required", "unknown"}


def _bounded(value: Any, allowed: set[str], fallback: str = "unknown") -> str:
    normalized = str(value or fallback)
    return normalized if normalized in allowed else fallback


class RequestMetrics:
    """Record only bounded labels; never include prompts, users, evidence text or trace IDs."""

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.requests = Counter(
            "echoforge_pipeline_requests_total",
            "Verified chat-pipeline requests by terminal business status",
            ["status"],
            registry=registry,
        )
        self.latency = Histogram(
            "echoforge_pipeline_request_duration_seconds",
            "End-to-end graph duration by terminal business status",
            ["status"],
            registry=registry,
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30),
        )
        self.stage_latency = Histogram(
            "echoforge_pipeline_stage_duration_seconds",
            "Graph-node duration for the fixed EchoForge pipeline",
            ["stage"],
            registry=registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20),
        )
        self.verifier_outcomes = Counter(
            "echoforge_verifier_outcomes_total",
            "Layered verifier outcomes using bounded layer and status labels",
            ["layer", "status"],
            registry=registry,
        )
        self.persistence = Counter(
            "echoforge_memory_persistence_total",
            "Whether a verified pipeline response reached memory persistence",
            ["outcome"],
            registry=registry,
        )

    def record(self, graph_result: Any) -> None:
        state = getattr(graph_result, "state", {}) or {}
        trace = getattr(graph_result, "trace", None)
        status = _bounded(state.get("pipeline_status"), _PIPELINE_STATUSES, "failed")
        if trace is not None and getattr(trace, "stop_reason", "completed") != "completed":
            status = "failed"

        self.requests.labels(status=status).inc()
        total_ms = max(0.0, float(getattr(trace, "total_latency_ms", 0.0) or 0.0))
        self.latency.labels(status=status).observe(total_ms / 1000)

        node_timings = getattr(trace, "node_timings_ms", {}) or {}
        for stage, duration_ms in node_timings.items():
            # Stage names come from the validated static graph, not user input.
            self.stage_latency.labels(stage=str(stage)).observe(max(0.0, float(duration_ms)) / 1000)

        verification = state.get("verification")
        for check in getattr(verification, "checks", ()) or ():
            layer = str(check.get("verifier", "unknown"))
            verifier_status = _bounded(check.get("status"), _VERIFIER_STATUSES)
            self.verifier_outcomes.labels(layer=layer, status=verifier_status).inc()

        persistence_outcome = "persisted" if state.get("memory_persisted") else f"skipped_{status}"
        self.persistence.labels(outcome=persistence_outcome).inc()
