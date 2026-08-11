import { useCallback, useEffect, useMemo, useState } from "react";
import { loadGraph, loadRouteTraces, loadTraces } from "./api";
import { DEMO_GRAPH, DEMO_TRACES } from "./demo";
import { layoutGraph } from "./graphLayout";
import type { GraphDefinition, GraphTrace, RouteTrace } from "./types";

const short = (value: string) => value.length > 22 ? `${value.slice(0, 12)}…${value.slice(-7)}` : value;
const milliseconds = (value: number) => `${value < 10 ? value.toFixed(2) : value.toFixed(1)} ms`;

export function App() {
  const [graph, setGraph] = useState<GraphDefinition | null>(null);
  const [traces, setTraces] = useState<GraphTrace[]>([]);
  const [routeTraces, setRouteTraces] = useState<RouteTrace[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [usingDemo, setUsingDemo] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [definition, recent] = await Promise.all([loadGraph(), loadTraces()]);
      let routes: RouteTrace[] = [];
      let routeError = "";
      try {
        routes = await loadRouteTraces();
      } catch (cause) {
        // Older API instances may not expose route evidence yet. The graph
        // view is still useful, so keep it live and surface the exact gap.
        routeError = cause instanceof Error ? cause.message : "route trace unavailable";
      }
      setGraph(definition);
      setTraces(recent);
      setRouteTraces(routes);
      setSelectedId((current) => current || recent[0]?.trace_id || "");
      setError(routeError);
      setUsingDemo(false);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Unknown API error";
      if (import.meta.env.DEV) {
        setGraph(DEMO_GRAPH);
        setTraces(DEMO_TRACES);
        setSelectedId(DEMO_TRACES[0].trace_id);
        setUsingDemo(true);
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const trace = traces.find((item) => item.trace_id === selectedId) ?? traces[0];
  const legacyTrace = Boolean(trace && !trace.node_runs.some((run) => run.node === "verify_response"));
  useEffect(() => { setStep(trace?.node_runs.length ?? 0); }, [trace?.trace_id, trace?.node_runs.length]);

  const visibleRuns = trace?.node_runs.slice(0, step) ?? [];
  const visibleTransitions = trace?.transitions.slice(0, Math.max(0, step - 1)) ?? [];
  const visited = new Map(visibleRuns.map((run) => [run.node, run]));
  const traversed = new Set(visibleTransitions.map((edge) => `${edge.source}->${edge.target}`));
  const positions = useMemo(() => graph ? layoutGraph(graph) : {}, [graph]);
  const maxTiming = Math.max(1, ...Object.values(trace?.node_timings_ms ?? {}));

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AGENT RELIABILITY HARNESS / LIVE EVIDENCE</p>
          <h1>Echo<span>Forge</span> Control Deck</h1>
          <p className="subtitle">条件路径、预算停止与节点时延的可回放视图</p>
        </div>
        <div className="actions">
          <div className={`status ${usingDemo ? "warn" : error ? "bad" : "good"}`}>
            <i /> {usingDemo ? "DEMO TRACE" : error ? "API DEGRADED" : loading ? "SYNCING" : "TRACE LINKED"}
          </div>
          <button type="button" onClick={() => void refresh()} disabled={loading}>刷新证据</button>
        </div>
      </header>

      {error && <section className="error">EchoForge API 证据接口提示：{error}{usingDemo && "；开发模式正在展示明确标记的演示路径。"}</section>}
      {legacyTrace && <section className="history-note">当前选中的是 verifier 改造前的历史 trace；它可以用于回放旧行为，不代表当前生产图的完整路径。</section>}

      <section className="metrics">
        <Metric label="STOP REASON" value={trace?.stop_reason ?? "—"} tone={trace?.stop_reason === "completed" ? "cyan" : "amber"} />
        <Metric label="PIPELINE" value={trace ? milliseconds(trace.total_latency_ms) : "—"} />
        <Metric label="NODE RUNS" value={trace ? String(trace.node_runs.length) : "—"} />
        <Metric label="GRAPH BUDGET" value={graph ? `${graph.budget.max_node_executions}N / ${graph.budget.max_transitions}E` : "—"} />
        <Metric label="EVIDENCE LINKS" value={routeTraces[0] ? `${routeTraces[0].citations.length}/${routeTraces[0].evidence_ids.length}` : "—"} tone="cyan" />
      </section>

      <section className="workspace">
        <article className="panel graph-panel">
          <div className="panel-title">
            <div><small>PRODUCTION TOPOLOGY</small><h2>{graph?.name ?? "Waiting for graph"}</h2></div>
            {trace && <code>{short(trace.trace_id)}</code>}
          </div>
          <div className="canvas-wrap">
            {graph && (
              <svg viewBox="0 0 1180 350" role="img" aria-label="EchoForge execution graph">
                <defs>
                  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L8,3 z" className="arrow" />
                  </marker>
                </defs>
                {graph.edges.map((edge) => {
                  const from = positions[edge.source];
                  const to = positions[edge.target];
                  const active = traversed.has(`${edge.source}->${edge.target}`);
                  const bend = edge.source === "decide_retrieval" && edge.target === "retrieve";
                  const path = bend
                    ? `M${from.x + 65},${from.y} C${from.x + 110},${from.y - 85} ${to.x - 100},${to.y} ${to.x - 65},${to.y}`
                    : `M${from.x + 65},${from.y} L${to.x - 65},${to.y}`;
                  return <g key={`${edge.source}-${edge.target}`}>
                    <path d={path} className={`edge ${active ? "active" : ""}`} markerEnd="url(#arrow)" />
                    <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 10} className="edge-label">{edge.label}</text>
                  </g>;
                })}
                {graph.nodes.map((node) => {
                  const point = positions[node.name];
                  const run = visited.get(node.name);
                  const terminal = graph.terminal_nodes.includes(node.name);
                  return <g key={node.name} transform={`translate(${point.x - 65} ${point.y - 34})`}>
                    <rect width="130" height="68" rx="12" className={`node ${run ? run.status : ""} ${terminal ? "terminal" : ""}`} />
                    <text x="12" y="27" className="node-name">{node.name}</text>
                    <text x="12" y="49" className="node-meta">{run ? milliseconds(run.latency_ms) : `${node.timeout_ms}ms budget`}</text>
                  </g>;
                })}
              </svg>
            )}
          </div>
          <div className="replay">
            <label htmlFor="replay">REPLAY STEP <strong>{step}/{trace?.node_runs.length ?? 0}</strong></label>
            <input id="replay" type="range" min="0" max={trace?.node_runs.length ?? 0} value={step} onChange={(event) => setStep(Number(event.target.value))} />
            <div className="replay-buttons">
              <button type="button" onClick={() => setStep((value) => Math.max(0, value - 1))} disabled={step === 0}>上一步</button>
              <button type="button" onClick={() => setStep((value) => Math.min(trace?.node_runs.length ?? 0, value + 1))} disabled={step === (trace?.node_runs.length ?? 0)}>下一步</button>
            </div>
          </div>
        </article>

        <aside className="panel trace-list">
          <div className="panel-title"><div><small>EVIDENCE LOG</small><h2>Recent traces</h2></div><b>{traces.length}</b></div>
          <div className="trace-scroll">
            {traces.map((item) => (
              <button key={item.trace_id} type="button" className={item.trace_id === trace?.trace_id ? "selected" : ""} onClick={() => setSelectedId(item.trace_id)}>
                <span><i className={item.stop_reason === "completed" ? "ok" : "warn"} />{short(item.trace_id)}</span>
                <em>{milliseconds(item.total_latency_ms)}</em>
                <small>{new Date(item.created_at).toLocaleString("zh-CN", { hour12: false })}</small>
              </button>
            ))}
            {!traces.length && <p className="empty">发送一次 /chat 请求后，路径证据会出现在这里。</p>}
          </div>
        </aside>
      </section>

      <section className="panel timing-panel">
        <div className="panel-title"><div><small>LATENCY ATTRIBUTION</small><h2>Node timing</h2></div><code>state payload excluded</code></div>
        <div className="timings">
          {graph?.nodes.map((node) => {
            const value = trace?.node_timings_ms[node.name] ?? 0;
            return <div className="timing" key={node.name}>
              <span>{node.name}</span><div><i style={{ width: `${(value / maxTiming) * 100}%` }} /></div><strong>{milliseconds(value)}</strong>
            </div>;
          })}
        </div>
      </section>

      <section className="panel evidence-panel">
        <div className="panel-title"><div><small>ANSWER QUALITY GATE</small><h2>Evidence lineage</h2></div><code>route trace / verifier</code></div>
        {routeTraces[0] ? <div className="evidence-grid">
          <div><span>VERIFICATION</span><strong className={routeTraces[0].verification_status === "completed" ? "cyan" : "amber"}>{routeTraces[0].verification_status}</strong></div>
          <div><span>ROUTE</span><strong>{routeTraces[0].final_agent ?? "—"}</strong></div>
          <div><span>RETRIEVED IDS</span><strong>{routeTraces[0].evidence_ids.length ? routeTraces[0].evidence_ids.join(", ") : "none"}</strong></div>
          <div><span>CITATIONS</span><strong>{routeTraces[0].citations.length ? routeTraces[0].citations.join(", ") : "none"}</strong></div>
        </div> : <p className="empty">发送一次请求后，这里会显示 evidence IDs、citations 和 verifier 终态。</p>}
      </section>
    </main>
  );
}

function Metric({ label, value, tone = "plain" }: { label: string; value: string; tone?: string }) {
  return <article><small>{label}</small><strong className={tone}>{value}</strong></article>;
}
