# EchoForge Control Deck

React + TypeScript dashboard for the production graph, JSONL-backed execution
traces and route/evidence traces. It loads `GET /graph`,
`GET /graph/traces/recent` and `GET /traces/recent`, renders the validated
topology, attributes latency by node, and exposes evidence IDs/citations and
verifier status beside a step-by-step replay.

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

For a live view, start the API first from the EchoForge project directory:

```bash
docker compose up -d --build
# or, when Redis/ChromaDB are already available:
uvicorn api.main:app --reload --port 8000
```

Then start the dashboard in another terminal:

```bash
npm run dev
```

The Vite dev server proxies `/api/*` to the Docker/Nginx entrypoint
`http://localhost:80` by default. When running only Uvicorn, use:

```bash
VITE_API_TARGET=http://localhost:8000 npm run dev
```

A 404 on `/api/graph` usually means the dashboard is pointing at the wrong API
entrypoint; a 404 only on `/api/traces/recent` means an older API instance is
running and the graph view remains usable without route evidence.
