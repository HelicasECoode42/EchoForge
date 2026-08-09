"""Container-safe compatibility copy of the public agent-runtime primitives.

The standalone ``agent-runtime`` package is preferred in local development.
Docker builds EchoForge as an isolated context, so this dependency-free bridge
keeps the verifier's terminal-state contract available inside the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class OutcomeStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class Budget:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")


@dataclass
class AgentState:
    values: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Observation:
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verification:
    ok: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    blocked: bool = False
    failed: bool = False


Step = Callable[[AgentState], Observation]
Verifier = Callable[[AgentState, Observation], Verification]


class AgentRuntime:
    """Run a bounded step and verify its observation before completing."""

    def __init__(self, *, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()

    def run(self, initial: Mapping[str, Any], *, step: Step, verifier: Verifier):
        state = AgentState(values=dict(initial))
        for index in range(1, self.budget.max_steps + 1):
            try:
                observation = step(state)
                state.steps.append({"index": index, "kind": observation.kind, "data": dict(observation.data)})
                verdict = verifier(state, observation)
            except Exception as exc:
                state.values["verification"] = {}
                return type("Outcome", (), {
                    "status": OutcomeStatus.FAILED,
                    "state": state,
                    "reason": f"runtime_error:{type(exc).__name__}",
                })()
            if verdict.ok:
                state.values["verification"] = dict(verdict.evidence)
                return type("Outcome", (), {
                    "status": OutcomeStatus.COMPLETED,
                    "state": state,
                    "reason": verdict.reason,
                })()
            if verdict.failed or verdict.blocked:
                state.values["verification"] = dict(verdict.evidence)
                return type("Outcome", (), {
                    "status": OutcomeStatus.FAILED if verdict.failed else OutcomeStatus.BLOCKED,
                    "state": state,
                    "reason": verdict.reason,
                })()
        return type("Outcome", (), {
            "status": OutcomeStatus.BLOCKED,
            "state": state,
            "reason": "step_budget_exhausted",
        })()
