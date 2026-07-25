# EchoForge Control Deck

React + TypeScript dashboard for the production graph and its JSONL-backed
execution traces. It loads `GET /graph` and `GET /graph/traces/recent`, renders
the validated topology, attributes latency by node and replays the selected
path step by step.

```bash
npm install
npm run check
npm run build
npm run dev
```

During local development only, an unavailable API activates a clearly marked
`DEMO TRACE` so layout and replay controls can be inspected without starting
Redis, ChromaDB and the model service. Production builds do not substitute demo
data for an API failure.
