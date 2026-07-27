"""A small, dependency-free execution graph for bounded Agent workflows.

The runtime intentionally separates graph mechanics from Agent behavior.  A
node receives mutable state and may return a mapping to merge into that state;
edges decide the next node from the resulting state.  Every attempt and
transition is recorded so production paths can be replayed without persisting
raw user prompts.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional


GraphState = MutableMapping[str, Any]
NodeHandler = Callable[[GraphState], Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]]
EdgeGuard = Callable[[GraphState], bool]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphBudget:
    """Hard limits that prevent cycles and retries from becoming unbounded."""

    max_node_executions: int = 12
    max_transitions: int = 10
    max_total_runtime_ms: float = 30_000.0

    def __post_init__(self) -> None:
        if self.max_node_executions < 1:
            raise ValueError("max_node_executions must be >= 1")
        if self.max_transitions < 0:
            raise ValueError("max_transitions must be >= 0")
        if self.max_total_runtime_ms <= 0:
            raise ValueError("max_total_runtime_ms must be > 0")


@dataclass(frozen=True)
class NodeSpec:
    name: str
    handler: NodeHandler
    timeout_ms: float = 10_000.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("node name must not be empty")
        if self.timeout_ms <= 0:
            raise ValueError("node timeout_ms must be > 0")
        if self.max_retries < 0:
            raise ValueError("node max_retries must be >= 0")


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    guard: Optional[EdgeGuard] = None
    label: str = "next"


@dataclass
class NodeRun:
    node: str
    attempt: int
    status: str
    latency_ms: float
    error_type: Optional[str] = None


@dataclass
class TransitionRun:
    source: str
    target: str
    label: str


@dataclass
class GraphTrace:
    trace_id: str
    graph_name: str
    created_at: str
    start_node: str
    terminal_nodes: List[str]
    budget: GraphBudget
    node_runs: List[NodeRun] = field(default_factory=list)
    transitions: List[TransitionRun] = field(default_factory=list)
    stop_reason: str = "running"
    total_latency_ms: float = 0.0

    @classmethod
    def start(
        cls,
        *,
        graph_name: str,
        start_node: str,
        terminal_nodes: set[str],
        budget: GraphBudget,
        trace_id: Optional[str] = None,
    ) -> "GraphTrace":
        return cls(
            trace_id=trace_id or f"graph-{uuid.uuid4().hex[:16]}",
            graph_name=graph_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            start_node=start_node,
            terminal_nodes=sorted(terminal_nodes),
            budget=budget,
        )

    @property
    def node_timings_ms(self) -> Dict[str, float]:
        timings: Dict[str, float] = {}
        for run in self.node_runs:
            timings[run.node] = timings.get(run.node, 0.0) + run.latency_ms
        return {name: round(value, 3) for name, value in timings.items()}

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["node_timings_ms"] = self.node_timings_ms
        return payload


@dataclass
class GraphResult:
    state: GraphState
    trace: GraphTrace


class GraphValidationError(ValueError):
    pass


class JsonlGraphTraceStore:
    """Append-only graph evidence. Graph state is deliberately not persisted."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, trace: GraphTrace) -> None:
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


class ExecutionGraph:
    """Validated directed graph with conditional edges and bounded execution."""

    def __init__(
        self,
        *,
        name: str,
        nodes: List[NodeSpec],
        edges: List[EdgeSpec],
        start_node: str,
        terminal_nodes: set[str],
        budget: Optional[GraphBudget] = None,
        trace_store: Optional[JsonlGraphTraceStore] = None,
    ) -> None:
        self.name = name
        self.nodes = {node.name: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise GraphValidationError("duplicate node name")
        self.edges = list(edges)
        self.start_node = start_node
        self.terminal_nodes = set(terminal_nodes)
        self.budget = budget or GraphBudget()
        self.trace_store = trace_store
        self.validate()

    def validate(self) -> None:
        if self.start_node not in self.nodes:
            raise GraphValidationError(f"start node not found: {self.start_node}")
        missing_terminals = self.terminal_nodes - self.nodes.keys()
        if missing_terminals:
            raise GraphValidationError(f"terminal nodes not found: {sorted(missing_terminals)}")
        if not self.terminal_nodes:
            raise GraphValidationError("at least one terminal node is required")

        outgoing: Dict[str, List[str]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise GraphValidationError(f"edge references missing node: {edge.source}->{edge.target}")
            outgoing[edge.source].append(edge.target)

        reachable: set[str] = set()
        stack = [self.start_node]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(outgoing[current])
        unreachable = sorted(self.nodes.keys() - reachable)
        if unreachable:
            raise GraphValidationError(f"unreachable nodes: {unreachable}")

        dead_ends = sorted(
            name for name, targets in outgoing.items()
            if not targets and name not in self.terminal_nodes
        )
        if dead_ends:
            raise GraphValidationError(f"non-terminal dead ends: {dead_ends}")

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_node": self.start_node,
            "terminal_nodes": sorted(self.terminal_nodes),
            "nodes": [
                {
                    "name": node.name,
                    "timeout_ms": node.timeout_ms,
                    "max_retries": node.max_retries,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "conditional": edge.guard is not None,
                }
                for edge in self.edges
            ],
            "budget": asdict(self.budget),
        }

    async def run(
        self,
        initial_state: Optional[GraphState] = None,
        *,
        trace_id: Optional[str] = None,
    ) -> GraphResult:
        state: GraphState = initial_state if initial_state is not None else {}
        trace = GraphTrace.start(
            graph_name=self.name,
            start_node=self.start_node,
            terminal_nodes=self.terminal_nodes,
            budget=self.budget,
            trace_id=trace_id,
        )
        started = time.monotonic()
        current = self.start_node

        while True:
            elapsed_ms = (time.monotonic() - started) * 1000
            if elapsed_ms >= self.budget.max_total_runtime_ms:
                trace.stop_reason = "runtime_budget_exhausted"
                break
            if len(trace.node_runs) >= self.budget.max_node_executions:
                trace.stop_reason = "node_budget_exhausted"
                break

            node = self.nodes[current]
            succeeded = False
            for attempt in range(1, node.max_retries + 2):
                if len(trace.node_runs) >= self.budget.max_node_executions:
                    trace.stop_reason = "node_budget_exhausted"
                    break
                remaining_ms = self.budget.max_total_runtime_ms - ((time.monotonic() - started) * 1000)
                if remaining_ms <= 0:
                    trace.stop_reason = "runtime_budget_exhausted"
                    break

                attempt_started = time.monotonic()
                try:
                    output = await asyncio.wait_for(
                        self._invoke_handler(node.handler, state),
                        timeout=min(node.timeout_ms, remaining_ms) / 1000,
                    )
                    if output:
                        state.update(output)
                    trace.node_runs.append(NodeRun(
                        node=current,
                        attempt=attempt,
                        status="completed",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                    ))
                    succeeded = True
                    break
                except asyncio.TimeoutError:
                    trace.node_runs.append(NodeRun(
                        node=current,
                        attempt=attempt,
                        status="timeout",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type="TimeoutError",
                    ))
                except Exception as exc:  # trace type only; state/error text may contain secrets
                    trace.node_runs.append(NodeRun(
                        node=current,
                        attempt=attempt,
                        status="failed",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        error_type=type(exc).__name__,
                    ))

            if not succeeded:
                if trace.stop_reason == "running":
                    last_status = trace.node_runs[-1].status if trace.node_runs else "failed"
                    trace.stop_reason = "node_timeout" if last_status == "timeout" else "node_failed"
                break

            if current in self.terminal_nodes:
                trace.stop_reason = "completed"
                break
            if len(trace.transitions) >= self.budget.max_transitions:
                trace.stop_reason = "transition_budget_exhausted"
                break

            selected: Optional[EdgeSpec] = None
            for edge in self.edges:
                if edge.source != current:
                    continue
                try:
                    if edge.guard is None or edge.guard(state):
                        selected = edge
                        break
                except Exception as exc:
                    logger.warning(
                        "graph edge guard failed graph=%s edge=%s->%s error=%s",
                        self.name,
                        edge.source,
                        edge.target,
                        type(exc).__name__,
                    )
                    continue
            if selected is None:
                trace.stop_reason = "no_matching_edge"
                break

            trace.transitions.append(TransitionRun(
                source=selected.source,
                target=selected.target,
                label=selected.label,
            ))
            current = selected.target

        trace.total_latency_ms = (time.monotonic() - started) * 1000
        if self.trace_store:
            await asyncio.to_thread(self.trace_store.append, trace)
        return GraphResult(state=state, trace=trace)

    @staticmethod
    async def _invoke_handler(handler: NodeHandler, state: GraphState):
        """Run synchronous handlers off-loop without exposing live graph state."""
        if inspect.iscoroutinefunction(handler):
            return await handler(state)
        # A timed-out thread cannot be forcibly stopped. Supplying a new mapping
        # prevents a late synchronous handler from mutating the graph's
        # authoritative state after its attempt has already failed or retried.
        output = await asyncio.to_thread(handler, dict(state))
        if inspect.isawaitable(output):
            return await output
        return output
