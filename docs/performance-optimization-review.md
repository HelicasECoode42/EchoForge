# EchoForge 性能与检索优化复盘

## 当前结论

一次真实请求曾出现约 10.6 秒端到端耗时，其中：

| 阶段 | 观测耗时 |
|---|---:|
| `load_memory` | 约 1.03 秒 |
| `retrieve` | 约 0.11 秒 |
| `execute_agent` | 约 9.46 秒 |
| 总耗时 | 约 10.60 秒 |

因此当前主要瓶颈是 DeepSeek 模型调用，而不是 chunk 或向量检索。后续应以多次请求的 p50/p95 为准，不能用单次请求证明优化效果。

关闭 VPN 后，一次同类请求从约 11.43 秒下降到约 9.46 秒；其中 `execute_agent` 仍占约 9.30 秒。这说明网络路径会影响延迟，但关闭 VPN 没有消除主要瓶颈，模型请求与服务端调度仍需继续拆分观测。

## 2026-07-26：DeepSeek 官方文档调研

进一步排查发现，DeepSeek V4 的思考模式默认开启，普通请求默认使用较高思考强度；在思考模式下，`temperature` 不生效。此前代码虽然把分类温度设为 `0.1`，但没有显式关闭思考模式，因此不能据此推断输出应当稳定。

参考：

- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [DeepSeek Anthropic API Compatibility](https://api-docs.deepseek.com/guides/anthropic_api)
- [DeepSeek Rate Limit and Keep-Alive](https://api-docs.deepseek.com/quick_start/rate_limit/)
- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)

官方文档还说明：请求可能在开始推理前保持连接并发送 keep-alive；因此 HTTP 连接成功不等于模型已经开始生成。上下文缓存默认开启，但属于 best-effort，不能直接视为稳定的延迟优化。

## 意图可靠性澄清

本轮没有删除原有三路意图信号：

```text
LLM 语义判断 + Embedding 相似度 + 关键词信号 → 加权投票
```

合并的是两次 LLM 任务：`意图识别` 与 `实体提取`。合并可能存在多任务干扰，但目前没有 A/B 数据证明它导致准确率下降。同一句订单问题出现 `query` 与 `complaint` 两种结果，更直接的问题包括：标签边界不清晰、LLM 权重过高，以及此前默认思考模式下温度参数不生效。

## 已完成的改动

### 1. 可审计评测集

新增 `data/eval/intent_cases.v1.json`，包含 40 条覆盖 10 类意图的合成种子样本。数据标记为 `needs_human_validation`，只用于回归和类别覆盖基线，不代表真实线上分布。

`/eval/run` 会返回数据集 provenance，包括数据集版本、来源、标签状态和样本数量。

### 2. 合并模型请求

意图识别与实体提取合并为一次 DeepSeek 请求：

```text
旧：意图请求 → 实体请求 → Agent 回答
新：意图+实体请求 → Agent 回答
```

通过单元测试验证主链路只发起一次意图分析请求。

### 3. 并行准备上下文

memory 和知识检索在 `load_memory` 节点内部并发执行，原有图节点名称保留用于回放兼容。新增模型调用日志：

```text
model_call component=intent_and_entities latency_ms=...
model_call component=agent.general latency_ms=...
```

这能区分“意图请求慢”和“最终回答慢”。

### 4. 子块命中与邻块回填

检索仍以结构感知 chunk 为索引单元，命中后最多回填一层前后邻块，并受 `RAG_EXPANSION_TOKEN_BUDGET=700` 限制。这样可以减少证据被切分边界截断的概率，同时避免无界扩大上下文。

### 5. SSE 流式响应

新增 `POST /chat/stream`。它仍执行同一套 memory、检索、路由和持久化流程，只将最终 Agent 的文本通过 SSE token 事件逐段返回。

事件类型：

- `started`
- `token`
- `complete`
- `error`

SSE 主要改善首字时间（TTFT）和用户感知，不等价于降低模型完整生成时间。

### 6. 显式关闭 DeepSeek 思考模式

模型适配层现在对 DeepSeek Anthropic 兼容接口统一注入：

```json
{"thinking":{"type":"disabled"}}
```

配置项：

```dotenv
DEEPSEEK_THINKING_MODE=disabled
```

该配置覆盖普通请求和 SSE 流式请求。非 DeepSeek 的 Anthropic 地址不会收到这个 provider-specific 参数。日志会显示实际模式：

```text
model_call component=intent_and_entities model=deepseek-v4-pro thinking=disabled latency_ms=...
model_call component=agent.general model=deepseek-v4-pro thinking=disabled latency_ms=...
```

这次改动不使用规则引擎替代模型判断；意图与回复仍由 DeepSeek 生成。预期收益是假设：简单客服任务无需隐藏思考过程，可以减少 TTFT 与总生成时间。该假设必须通过改动前后的重复实验验证。

## 如何验证

启动服务后执行：

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"我的订单为什么还没有发货？","user_id":"demo-user"}'
```

观察服务端日志中的 `model_call`，并记录：

- 意图+实体模型调用耗时；
- Agent 模型调用耗时；
- `pipeline_total_latency_ms`；
- 首个 `token` 到达时间；
- 完整回答到达时间；
- 回答是否仍然有知识库证据支撑。

建议使用至少 20 条固定问题、每条重复 3 次，分别比较 `/chat` 和 `/chat/stream` 的 p50/p95。模型网络延迟可能波动，单次请求不作为结论。

关闭思考模式后的第一轮验证还应记录：

- 同一输入重复 3 次的意图一致率；
- `intent_and_entities` 与 `agent.*` 各自耗时；
- VPN 开启/关闭两组网络条件，避免把网络变化错误归因于代码；
- SSE 的 TTFT、完整生成耗时和输出长度；
- 失败率、空响应率和 JSON 解析成功率。

## 当前限制

- 40 条评测数据是合成数据，尚未经过真实用户样本和人工复核。
- 当前 chunk 实验规模较小，不能代表生产语料。
- 邻块回填提升的是证据完整性，是否提升最终回答质量需要额外标注评测。
- SSE 降低的是等待体验，不保证降低 total latency。
