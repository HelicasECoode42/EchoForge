# 路由预算、证据 Trace 与离线回放

EchoMind 将专业 Agent 失败后的 fallback 视为一次显式 reroute，并通过三项硬预算限制行为：

- `ROUTE_MAX_ATTEMPTS`：单次请求最多执行多少次 Agent；
- `ROUTE_MAX_REROUTES`：最多允许多少次重新选路；
- `ROUTE_MAX_LATENCY_MS`：路由链路的总延迟上限。

默认预算为 2 次执行、1 次 reroute、15 秒总延迟。达到预算时系统会返回明确的 `stop_reason`，不会继续隐式重试。

## 证据记录

每次 `/chat` 响应新增：

- `trace_id`
- `route_attempts`
- `reroutes`
- `stop_reason`

完整路由证据以 JSONL 追加到 `ROUTE_TRACE_PATH`，并可通过 `GET /traces/recent?limit=20` 查看。Trace 记录目标 Agent、实际 Agent、路由评分、结果、延迟、reroute 原因和停止原因。

为避免把客服对话写入运行日志，Trace 只保存消息 SHA-256 与字符数，不保存用户原文。

## 确定性回放

回放不调用外部模型，也不需要 API Key：

```bash
python scripts/replay_routes.py
python scripts/replay_routes.py --output reports/route-replay.json
```

默认 5 条案例覆盖：

1. 专业 Agent 一次成功；
2. 账单 Agent 失败后仅 fallback 一次；
3. 紧急请求缺少人工 Agent 实例时的降级；
4. reroute 配额为 0 时停止；
5. 尝试次数用尽时不再执行 fallback。

回放报告逐项比较最终 Agent、成功状态、尝试次数、reroute 次数、停止原因和升级状态，可作为路由策略改动后的回归证据。
