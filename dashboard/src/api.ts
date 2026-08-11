import type { GraphDefinition, GraphTrace, RouteTrace } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} at ${path}`);
  }
  return response.json() as Promise<T>;
}

export const loadGraph = () => getJson<GraphDefinition>("/graph");

export async function loadTraces(limit = 40): Promise<GraphTrace[]> {
  const payload = await getJson<{ items: GraphTrace[] }>(`/graph/traces/recent?limit=${limit}`);
  return payload.items.slice().sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

export async function loadRouteTraces(limit = 40): Promise<RouteTrace[]> {
  const payload = await getJson<{ items: RouteTrace[] }>(`/traces/recent?limit=${limit}`);
  // Older JSONL records predate evidence lineage fields. Normalize them so
  // historical traces remain renderable after the response-contract upgrade.
  return payload.items.slice().sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)).map((trace) => ({
    ...trace,
    evidence_ids: trace.evidence_ids ?? [],
    citations: trace.citations ?? [],
    verification_status: trace.verification_status ?? "not_checked",
  }));
}
