export type GraphNode = {
  name: string;
  timeout_ms: number;
  max_retries: number;
};

export type GraphEdge = {
  source: string;
  target: string;
  label: string;
  conditional: boolean;
};

export type GraphDefinition = {
  name: string;
  start_node: string;
  terminal_nodes: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  budget: {
    max_node_executions: number;
    max_transitions: number;
    max_total_runtime_ms: number;
  };
};

export type NodeRun = {
  node: string;
  attempt: number;
  status: "completed" | "failed" | "timeout";
  latency_ms: number;
  error_type: string | null;
};

export type TransitionRun = {
  source: string;
  target: string;
  label: string;
};

export type GraphTrace = {
  trace_id: string;
  graph_name: string;
  created_at: string;
  node_runs: NodeRun[];
  transitions: TransitionRun[];
  stop_reason: string;
  total_latency_ms: number;
  node_timings_ms: Record<string, number>;
};
