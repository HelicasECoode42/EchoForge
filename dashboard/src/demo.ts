import type { GraphDefinition, GraphTrace } from "./types";

export const DEMO_GRAPH: GraphDefinition = {
  name: "echoforge_chat_pipeline",
  start_node: "load_memory",
  terminal_nodes: ["complete"],
  nodes: [
    { name: "load_memory", timeout_ms: 5000, max_retries: 0 },
    { name: "decide_retrieval", timeout_ms: 250, max_retries: 0 },
    { name: "retrieve", timeout_ms: 8000, max_retries: 1 },
    { name: "execute_agent", timeout_ms: 20000, max_retries: 0 },
    { name: "persist_memory", timeout_ms: 5000, max_retries: 0 },
    { name: "complete", timeout_ms: 250, max_retries: 0 },
  ],
  edges: [
    { source: "load_memory", target: "decide_retrieval", label: "memory_loaded", conditional: false },
    { source: "decide_retrieval", target: "retrieve", label: "retrieval_required", conditional: true },
    { source: "decide_retrieval", target: "execute_agent", label: "retrieval_skipped", conditional: true },
    { source: "retrieve", target: "execute_agent", label: "context_ready", conditional: false },
    { source: "execute_agent", target: "persist_memory", label: "agent_completed", conditional: false },
    { source: "persist_memory", target: "complete", label: "memory_persisted", conditional: false },
  ],
  budget: { max_node_executions: 8, max_transitions: 7, max_total_runtime_ms: 30000 },
};

export const DEMO_TRACES: GraphTrace[] = [
  {
    trace_id: "demo-knowledge-retrieval-path",
    graph_name: "echoforge_chat_pipeline",
    created_at: new Date().toISOString(),
    node_runs: [
      { node: "load_memory", attempt: 1, status: "completed", latency_ms: 31.8, error_type: null },
      { node: "decide_retrieval", attempt: 1, status: "completed", latency_ms: 0.3, error_type: null },
      { node: "retrieve", attempt: 1, status: "completed", latency_ms: 86.4, error_type: null },
      { node: "execute_agent", attempt: 1, status: "completed", latency_ms: 734.2, error_type: null },
      { node: "persist_memory", attempt: 1, status: "completed", latency_ms: 24.7, error_type: null },
      { node: "complete", attempt: 1, status: "completed", latency_ms: 0.1, error_type: null },
    ],
    transitions: [
      { source: "load_memory", target: "decide_retrieval", label: "memory_loaded" },
      { source: "decide_retrieval", target: "retrieve", label: "retrieval_required" },
      { source: "retrieve", target: "execute_agent", label: "context_ready" },
      { source: "execute_agent", target: "persist_memory", label: "agent_completed" },
      { source: "persist_memory", target: "complete", label: "memory_persisted" },
    ],
    stop_reason: "completed",
    total_latency_ms: 878.1,
    node_timings_ms: {
      load_memory: 31.8,
      decide_retrieval: 0.3,
      retrieve: 86.4,
      execute_agent: 734.2,
      persist_memory: 24.7,
      complete: 0.1,
    },
  },
];
