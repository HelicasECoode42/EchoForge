# Engineering standards

- Prefer a small explicit state contract over hidden globals.
- Every graph edge must have a named label and a regression case for important branches.
- Keep execution, grounding and task verification separate.
- Preserve evidence lineage from retrieval to final response.
- Never persist blocked or failed output.
- Do not put raw user text, prompts or secrets in traces.
- New behavior requires a deterministic test before it becomes a CI gate.
