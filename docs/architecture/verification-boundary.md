# Verification boundary

EchoForge separates workflow execution from business completion:

1. `ExecutionVerifier` checks route trace, non-empty output and agent execution.
2. `GroundingVerifier` checks retrieved evidence IDs, citations and basic support overlap.
3. `TaskVerifier` checks escalation, unresolved work and confidence signals.

Only the combined `completed` result reaches `persist_memory`. A successful model call alone never authorizes persistence.
