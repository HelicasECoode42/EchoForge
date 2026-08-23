# Verification boundary

EchoForge separates workflow execution from business completion:

1. `ExecutionVerifier` checks route trace, non-empty output and agent execution.
2. `GroundingVerifier` checks retrieved evidence IDs, citations, numeric preservation, polarity conflicts and conservative lexical support coverage.
3. `TaskVerifier` checks escalation, unresolved work and confidence signals.

Only the combined `completed` result reaches `persist_memory`. A successful model call alone never authorizes persistence.

The grounding verifier is not a semantic entailment model. Its versioned audit intentionally reports both dangerous false accepts and false blocks; unknown semantic cases must remain blocked or be escalated rather than treated as verified.
