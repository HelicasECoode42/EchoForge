"""
EchoMind 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   EchoMind  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_trace_store  = None
_graph_trace_store = None
_chat_graph   = None


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _trace_store
    global _graph_trace_store, _chat_graph

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.execution_graph import JsonlGraphTraceStore
    from core.intent_recognizer import IntentRecognizer
    from evidence.route_trace import JsonlRouteTraceStore, RoutingBudget
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Agent 编排器
    _trace_store = JsonlRouteTraceStore(
        os.getenv("ROUTE_TRACE_PATH", "/app/data/evidence/route_traces.jsonl")
    )
    route_budget = RoutingBudget(
        max_attempts=int(os.getenv("ROUTE_MAX_ATTEMPTS", "2")),
        max_reroutes=int(os.getenv("ROUTE_MAX_REROUTES", "1")),
        max_total_latency_ms=float(os.getenv("ROUTE_MAX_LATENCY_MS", "15000")),
    )
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        trace_store=_trace_store,
        default_budget=route_budget,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        chunk_strategy=os.getenv("CHUNK_STRATEGY", "structure_token"),
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    _graph_trace_store = JsonlGraphTraceStore(
        os.getenv("GRAPH_TRACE_PATH", "/app/data/evidence/graph_traces.jsonl")
    )
    _chat_graph = _create_chat_graph(_graph_trace_store)

    logger.info("EchoForge 已就绪")
    yield

    await _monitor.stop()
    logger.info("EchoForge 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoForge Agent Reliability Harness",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    trace_id: Optional[str] = None
    route_attempts: int = 0
    reroutes: int = 0
    stop_reason: str = "unknown"
    graph_trace_id: Optional[str] = None
    pipeline_stop_reason: str = "unknown"
    pipeline_total_latency_ms: float = 0.0
    pipeline_timings_ms: Dict[str, float] = Field(default_factory=dict)


def _create_chat_graph(trace_store):
    """Build the production graph; handlers use initialized lifespan services."""
    from agents.agent_orchestrator import Request as OrchestratorRequest
    from core.execution_graph import EdgeSpec, ExecutionGraph, GraphBudget, NodeSpec
    from memory.conversation_memory import MsgRole

    async def load_memory(state: Dict[str, Any]):
        req: ChatRequest = state["request"]
        memory_context = await _memory.get_context(req.user_id, state["conv_id"], query=req.message)
        history = [
            {"role": message.role.value, "content": message.content}
            for message in memory_context.recent_messages[-5:]
        ] if memory_context.recent_messages else None
        return {"memory_context": memory_context, "history": history}

    def decide_retrieval(state: Dict[str, Any]):
        req: ChatRequest = state["request"]
        return {"should_retrieve": _should_use_knowledge(req.message)}

    async def retrieve(state: Dict[str, Any]):
        req: ChatRequest = state["request"]
        knowledge_text, knowledge_used = await _build_knowledge_context(req.message)
        return {"knowledge_text": knowledge_text, "knowledge_used": knowledge_used}

    async def execute_agent(state: Dict[str, Any]):
        req: ChatRequest = state["request"]
        context_parts = [state["memory_context"].to_prompt_text()]
        if state.get("knowledge_text"):
            context_parts.append(state["knowledge_text"])
        request = OrchestratorRequest(
            message=req.message,
            user_id=req.user_id,
            conv_id=state["conv_id"],
            context="\n\n".join(part for part in context_parts if part),
            history=state.get("history"),
        )
        return {"orchestrator_result": await _orchestrator.run(request)}

    async def persist_memory(state: Dict[str, Any]):
        req: ChatRequest = state["request"]
        result = state["orchestrator_result"]
        await _memory.add_message(req.user_id, state["conv_id"], MsgRole.USER, req.message)
        await _memory.add_message(req.user_id, state["conv_id"], MsgRole.ASSISTANT, result.response)
        asyncio.create_task(_memory.update_profile(req.user_id, state["conv_id"]))
        return {"memory_persisted": True}

    return ExecutionGraph(
        name="echoforge_chat_pipeline",
        nodes=[
            NodeSpec("load_memory", load_memory, timeout_ms=5_000),
            NodeSpec("decide_retrieval", decide_retrieval, timeout_ms=250),
            NodeSpec("retrieve", retrieve, timeout_ms=8_000, max_retries=1),
            NodeSpec("execute_agent", execute_agent, timeout_ms=20_000),
            NodeSpec("persist_memory", persist_memory, timeout_ms=5_000),
            NodeSpec("complete", lambda state: None, timeout_ms=250),
        ],
        edges=[
            EdgeSpec("load_memory", "decide_retrieval", label="memory_loaded"),
            EdgeSpec(
                "decide_retrieval",
                "retrieve",
                guard=lambda state: bool(state.get("should_retrieve")),
                label="retrieval_required",
            ),
            EdgeSpec(
                "decide_retrieval",
                "execute_agent",
                guard=lambda state: not bool(state.get("should_retrieve")),
                label="retrieval_skipped",
            ),
            EdgeSpec("retrieve", "execute_agent", label="context_ready"),
            EdgeSpec("execute_agent", "persist_memory", label="agent_completed"),
            EdgeSpec("persist_memory", "complete", label="memory_persisted"),
        ],
        start_node="load_memory",
        terminal_nodes={"complete"},
        budget=GraphBudget(
            max_node_executions=int(os.getenv("GRAPH_MAX_NODE_EXECUTIONS", "8")),
            max_transitions=int(os.getenv("GRAPH_MAX_TRANSITIONS", "7")),
            max_total_runtime_ms=float(os.getenv("GRAPH_MAX_RUNTIME_MS", "30000")),
        ),
        trace_store=trace_store,
    )


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run the production request through the validated EchoForge graph."""
    if _chat_graph is None:
        raise HTTPException(503, "服务未就绪")
    conv_id = req.conv_id or str(uuid.uuid4())
    graph_result = await _chat_graph.run({"request": req, "conv_id": conv_id})
    if graph_result.trace.stop_reason != "completed" or "orchestrator_result" not in graph_result.state:
        logger.error(
            "chat graph failed trace=%s stop_reason=%s",
            graph_result.trace.trace_id,
            graph_result.trace.stop_reason,
        )
        raise HTTPException(
            503,
            detail={
                "message": "Agent pipeline did not complete",
                "graph_trace_id": graph_result.trace.trace_id,
                "stop_reason": graph_result.trace.stop_reason,
            },
        )

    result = graph_result.state["orchestrator_result"]
    trace = result.route_trace
    return ChatResponse(
        conv_id=conv_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        agent_type=result.agent_type.value,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=bool(graph_result.state.get("knowledge_used", False)),
        trace_id=trace.trace_id if trace else None,
        route_attempts=trace.attempts if trace else 0,
        reroutes=trace.reroutes if trace else 0,
        stop_reason=trace.stop_reason if trace else "unknown",
        graph_trace_id=graph_result.trace.trace_id,
        pipeline_stop_reason=graph_result.trace.stop_reason,
        pipeline_total_latency_ms=round(graph_result.trace.total_latency_ms, 1),
        pipeline_timings_ms=graph_result.trace.node_timings_ms,
    )


@app.get("/traces/recent", tags=["路由证据"])
async def recent_route_traces(limit: int = 20):
    """读取最近路由证据；只包含消息哈希和决策元数据，不返回用户原文。"""
    if _trace_store is None:
        raise HTTPException(503, "路由证据存储未初始化")
    return {"items": _trace_store.recent(limit)}


@app.get("/graph", tags=["执行图"])
async def graph_definition():
    """Expose the validated production graph for the trace dashboard."""
    if _chat_graph is None:
        raise HTTPException(503, "执行图未初始化")
    return _chat_graph.describe()


@app.get("/graph/traces/recent", tags=["执行图"])
async def recent_graph_traces(limit: int = 20):
    """Return paths and node timings only; graph state and prompts are excluded."""
    if _graph_trace_store is None:
        raise HTTPException(503, "执行图证据存储未初始化")
    return {"items": _graph_trace_store.recent(limit)}


async def _build_knowledge_context(message: str, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message):
        return "", False
    try:
        result = await _tool_manager.search_with_rewrite("knowledge_search", message, top_k=top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用客服能力说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "退款", "订单", "物流", "配送", "发票", "扣款", "支付", "账单", "订阅",
        "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址",
        "refund", "order", "invoice", "payment", "error", "login",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档按 CHUNK_STRATEGY 切分；默认保留章节路径并受 token budget 约束。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count, "chunk_strategy": kb.chunk_strategy}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole

    cfg = _anthropic_cfg()
    orch = AgentOrchestrator(api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"])
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
