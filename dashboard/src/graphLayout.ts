import type { GraphDefinition } from "./types";

export type Point = { x: number; y: number };

const preferred: Record<string, Point> = {
  load_memory: { x: 110, y: 190 },
  decide_retrieval: { x: 300, y: 190 },
  retrieve: { x: 490, y: 95 },
  execute_agent: { x: 620, y: 190 },
  verify_response: { x: 760, y: 190 },
  persist_memory: { x: 900, y: 190 },
  complete: { x: 1030, y: 95 },
  blocked: { x: 1030, y: 190 },
  failed: { x: 1030, y: 285 },
};

export function layoutGraph(graph: GraphDefinition): Record<string, Point> {
  const fallbackGap = 820 / Math.max(1, graph.nodes.length - 1);
  return Object.fromEntries(
    graph.nodes.map((node, index) => [
      node.name,
      preferred[node.name] ?? { x: 90 + index * fallbackGap, y: 190 },
    ]),
  );
}
