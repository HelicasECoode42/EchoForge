# EchoForge business context

EchoForge is a bounded customer-support style Agent Workflow: load memory, optionally retrieve knowledge, route to an agent, verify the response, and persist only verified outcomes.

## Scope

- Evidence-aware question answering.
- Auditable routing and graph traces.
- Explicit `completed`, `blocked`, and `failed` terminal states.

## Non-goals

- Treating free-form model output as a source of truth.
- Autonomous irreversible external actions.
- Training or self-improvement from unverified trajectories.
