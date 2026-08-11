# Agent module context

The orchestrator owns intent routing, bounded agent execution and structured answer parsing.

Response contract when retrieval evidence is present:

- `answer`: final answer text;
- `citations`: retrieved evidence IDs only;
- `confidence`: optional numeric signal;
- `needs_human`: escalation signal;
- `unresolved`: remaining task gaps.

An LLM call returning text is execution success, not task completion. Keep route evidence privacy-preserving.
