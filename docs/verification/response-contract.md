# Response contract

When evidence is supplied, the agent is asked for:

```json
{
  "answer": "...",
  "citations": ["chunk_id"],
  "confidence": 0.82,
  "needs_human": false,
  "unresolved": []
}
```

Plain text remains accepted for non-retrieval flows. Retrieval-required flows must provide citations that belong to the current retrieval set.
