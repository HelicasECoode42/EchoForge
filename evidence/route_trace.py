"""Bounded routing budgets and privacy-preserving execution evidence."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RoutingBudget:
    """Hard limits that keep fallback/reroute behavior predictable."""

    max_attempts: int = 2
    max_reroutes: int = 1
    max_total_latency_ms: float = 15_000.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_reroutes < 0:
            raise ValueError("max_reroutes must be >= 0")
        if self.max_total_latency_ms <= 0:
            raise ValueError("max_total_latency_ms must be > 0")


@dataclass
class RouteStep:
    stage: str
    requested_agent: str
    selected_agent: Optional[str]
    outcome: str
    reason: str
    latency_ms: float = 0.0
    routing_score: Optional[float] = None


@dataclass
class RouteTrace:
    """Compact evidence record; raw user text is intentionally excluded."""

    trace_id: str
    request_id: str
    created_at: str
    message_sha256: str
    message_length: int
    intent: str
    urgency: str
    budget: RoutingBudget
    steps: List[RouteStep] = field(default_factory=list)
    attempts: int = 0
    reroutes: int = 0
    stop_reason: str = "running"
    final_agent: Optional[str] = None
    success: bool = False
    total_latency_ms: float = 0.0

    @classmethod
    def start(
        cls,
        *,
        request_id: str,
        message: str,
        intent: str,
        urgency: str,
        budget: RoutingBudget,
    ) -> "RouteTrace":
        encoded = message.encode("utf-8", errors="ignore")
        return cls(
            trace_id=f"route-{request_id}-{uuid.uuid4().hex[:8]}",
            request_id=request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            message_sha256=hashlib.sha256(encoded).hexdigest(),
            message_length=len(message),
            intent=intent,
            urgency=urgency,
            budget=budget,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlRouteTraceStore:
    """Append-only JSONL evidence store with bounded readback."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, trace: RouteTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trace.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
