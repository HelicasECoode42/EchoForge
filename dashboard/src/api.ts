import type { GraphDefinition, GraphTrace } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const loadGraph = () => getJson<GraphDefinition>("/graph");

export async function loadTraces(limit = 40): Promise<GraphTrace[]> {
  const payload = await getJson<{ items: GraphTrace[] }>(`/graph/traces/recent?limit=${limit}`);
  // JsonlGraphTraceStore.recent() returns the selected JSONL slice in
  // chronological order; the dashboard intentionally shows newest first.
  return payload.items.slice().reverse();
}
