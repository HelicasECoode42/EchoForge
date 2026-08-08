"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.model_response import create_message, extract_text, provider_extra_body
from evidence.route_trace import JsonlRouteTraceStore, RouteStep, RouteTrace, RoutingBudget

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 账单/退款
    ESCALATION = "escalation" # 人工升级（占位）


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  Optional[float] = None
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    citations:   List[str] = field(default_factory=list)
    needs_human: bool = False
    unresolved:  List[str] = field(default_factory=list)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    routing_budget: Optional[RoutingBudget] = None
    evidence_ids: List[str] = field(default_factory=list)
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    success:      bool  = False
    latency_ms:  float = 0.0
    route_trace: Optional[RouteTrace] = None
    citations: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    needs_human: bool = False
    unresolved: List[str] = field(default_factory=list)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            raw_content = await self._call_llm(req)
            payload = self._parse_answer_payload(raw_content)
            content = str(payload.get("answer", raw_content)) if payload else raw_content
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                citations=list(payload.get("citations", [])) if payload else [],
                confidence=payload.get("confidence") if payload else None,
                needs_human=bool(payload.get("needs_human", False)) if payload else False,
                unresolved=list(payload.get("unresolved", [])) if payload else [],
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def handle_stream(self, req: Request, progress_sink) -> AgentResponse:
        """Stream final-agent text while preserving the normal result contract."""
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            raw_content = await self._call_llm_stream(req, progress_sink)
            payload = self._parse_answer_payload(raw_content)
            content = str(payload.get("answer", raw_content)) if payload else raw_content
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=self._needs_escalation(content),
                citations=list(payload.get("citations", [])) if payload else [],
                confidence=payload.get("confidence") if payload else None,
                needs_human=bool(payload.get("needs_human", False)) if payload else False,
                unresolved=list(payload.get("unresolved", [])) if payload else [],
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 流式处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )
    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await create_message(
            self._client,
            component=f"agent.{self.agent_type.value}",
            model=self._model,
            max_tokens=1024,
            system=self._system_prompt(req),
            messages=messages,
        )
        return extract_text(resp)

    @staticmethod
    def _parse_answer_payload(content: str) -> Optional[Dict[str, Any]]:
        """Accept structured answers while preserving compatibility with plain text."""
        try:
            value = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            return None
        citations = value.get("citations", [])
        unresolved = value.get("unresolved", [])
        if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
            return None
        if not isinstance(unresolved, list) or not all(isinstance(item, str) for item in unresolved):
            return None
        confidence = value.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            return None
        return value

    async def _call_llm_stream(self, req: Request, progress_sink) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        chunks = []
        stream_kwargs = {}
        extra_body = provider_extra_body()
        if extra_body is not None:
            stream_kwargs["extra_body"] = extra_body
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=self._system_prompt(req),
            messages=messages,
            **stream_kwargs,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    chunks.append(text)
                    await progress_sink(text)
            final_message = await stream.get_final_message()

        # Some compatible providers may emit no text events but still return
        # a valid final response; normalize that response as a fallback.
        content = "".join(chunks).strip()
        return content or extract_text(final_message)

    def _system_prompt(self, req: Request) -> str:
        if not req.evidence_ids:
            return self.system_prompt
        return self.system_prompt + (
            " 这是一次需要知识库依据的回答。只能使用提供的证据 ID，不要编造来源。"
            "请严格返回 JSON：{\"answer\": string, \"citations\": string[], "
            "\"confidence\": number, \"needs_human\": boolean, \"unresolved\": string[]}。"
            f"可用证据 ID：{json.dumps(req.evidence_ids, ensure_ascii=False)}"
        )

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        content = content or ""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    system_prompt = (
        "你是 EchoMind 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    system_prompt = (
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    system_prompt = (
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.ACCOUNT:    AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        *,
        intent_recognizer: Optional[Any] = None,
        agent_pool: Optional[Dict[AgentType, List[Any]]] = None,
        trace_store: Optional[JsonlRouteTraceStore] = None,
        default_budget: Optional[RoutingBudget] = None,
    ):
        if agent_pool is None:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncAnthropic(**kwargs)
            self._intent_recognizer = intent_recognizer or IntentRecognizer(
                api_key=api_key, base_url=base_url, model=model
            )
            # Agent 池：每种类型可有多个实例（水平扩展）
            self._pool: Dict[AgentType, List[Any]] = {
                AgentType.GENERAL:   [GeneralAgent(client, model)],
                AgentType.TECHNICAL: [TechnicalAgent(client, model)],
                AgentType.BILLING:   [BillingAgent(client, model)],
            }
        else:
            # 允许回放/测试注入确定性 Agent，不触发任何远端模型调用。
            self._intent_recognizer = intent_recognizer
            self._pool = agent_pool
        self._trace_store = trace_store
        self._default_budget = default_budget or RoutingBudget()

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request, progress_sink=None) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency

        # 2. 路由：选择一个主 Agent 类型。
        # 面试版/稳定版采用“三选一”策略：General / Technical / Billing。
        # 复合问题先按主意图处理，必要时在回复中建议转人工或补充信息。
        agent_type = self._route(req.intent, req.urgency)

        # 3. 执行（含降级）
        budget = req.routing_budget or self._default_budget
        trace = RouteTrace.start(
            request_id=req.request_id,
            message=req.message,
            intent=req.intent.value if req.intent else "other",
            urgency=req.urgency.name.lower() if req.urgency else "low",
            budget=budget,
            evidence_ids=req.evidence_ids,
        )
        response, trace = await self._execute_bounded(req, agent_type, budget, trace, progress_sink=progress_sink)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知人工客服

        trace.final_agent = response.agent_type.value
        trace.success = response.success
        trace.citations = list(response.citations)
        trace.total_latency_ms = (time.monotonic() - t0) * 1000

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            success=response.success,
            latency_ms=trace.total_latency_ms,
            route_trace=trace,
            citations=list(response.citations),
            confidence=response.confidence,
            needs_human=response.needs_human or escalated,
            unresolved=list(response.unresolved),
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，所有 Agent 均处理失败。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            success=bool(parts),
            latency_ms=(time.monotonic() - t0) * 1000,
            citations=[citation for response in responses if isinstance(response, AgentResponse) for citation in response.citations],
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"登录报错且被重复扣款"需要技术和账单 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        technical_kws = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401"]
        billing_kws = ["退款", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice"]

        if req.intent == IntentCategory.TECHNICAL or any(kw in msg for kw in technical_kws):
            targets.append(AgentType.TECHNICAL)
        if req.intent in (IntentCategory.BILLING, IntentCategory.ACCOUNT) or any(kw in msg for kw in billing_kws):
            targets.append(AgentType.BILLING)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    async def _execute_bounded(
        self,
        req: Request,
        agent_type: AgentType,
        budget: RoutingBudget,
        trace: RouteTrace,
        progress_sink=None,
    ) -> Tuple[AgentResponse, RouteTrace]:
        """Execute with explicit attempt, reroute, and latency limits."""
        started = time.monotonic()
        current = agent_type
        last_response: Optional[AgentResponse] = None

        while trace.attempts < budget.max_attempts:
            elapsed_ms = (time.monotonic() - started) * 1000
            if elapsed_ms >= budget.max_total_latency_ms:
                trace.stop_reason = "latency_budget_exhausted"
                break

            agent = self._best_agent(current)
            if agent is None:
                trace.steps.append(RouteStep(
                    stage="select",
                    requested_agent=current.value,
                    selected_agent=None,
                    outcome="unavailable",
                    reason="no_agent_instance",
                ))
                if current != AgentType.GENERAL and trace.reroutes < budget.max_reroutes:
                    trace.reroutes += 1
                    trace.steps.append(RouteStep(
                        stage="reroute",
                        requested_agent=current.value,
                        selected_agent=AgentType.GENERAL.value,
                        outcome="scheduled",
                        reason="target_unavailable",
                    ))
                    current = AgentType.GENERAL
                    continue
                trace.stop_reason = "reroute_budget_exhausted"
                break

            score = agent.stats.routing_score() if hasattr(agent, "stats") else None
            response = (
                await agent.handle_stream(req, progress_sink)
                if progress_sink is not None
                else await agent.handle(req)
            )
            trace.attempts += 1
            last_response = response
            trace.steps.append(RouteStep(
                stage="attempt",
                requested_agent=current.value,
                selected_agent=response.agent_type.value,
                outcome="success" if response.success else "failed",
                reason="agent_completed" if response.success else "agent_failed",
                latency_ms=response.latency_ms,
                routing_score=round(score, 6) if score is not None else None,
            ))

            if response.success:
                trace.stop_reason = "completed"
                return response, trace

            if current != AgentType.GENERAL and trace.reroutes < budget.max_reroutes:
                if trace.attempts >= budget.max_attempts:
                    trace.stop_reason = "attempt_budget_exhausted"
                    break
                trace.reroutes += 1
                trace.steps.append(RouteStep(
                    stage="reroute",
                    requested_agent=current.value,
                    selected_agent=AgentType.GENERAL.value,
                    outcome="scheduled",
                    reason="target_failed",
                ))
                current = AgentType.GENERAL
                continue

            trace.stop_reason = (
                "reroute_budget_exhausted"
                if current != AgentType.GENERAL
                else "fallback_failed"
            )
            break

        if trace.stop_reason == "running":
            trace.stop_reason = "attempt_budget_exhausted"
        if last_response is not None:
            return last_response, trace
        return AgentResponse(
            agent_type=AgentType.GENERAL,
            content="路由预算已用尽，请稍后重试或转人工客服。",
            success=False,
        ), trace

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
