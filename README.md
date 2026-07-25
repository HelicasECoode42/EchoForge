# EchoForge

EchoForge 是一个面向智能客服场景的多 Agent 应用工程项目，重点不在堆叠 Agent 数量，而在于让路由、检索、降级与评测具备可解释、可回放和可验证的工程边界。

项目提供 FastAPI 服务，将意图识别、知识检索、专业 Agent 路由、短期/长期记忆和执行证据串成完整请求链路；同时提供无需外部模型的确定性回放，以及基于真实本地 Embedding + ChromaDB 的检索评测。

## 核心能力

- **多 Agent 路由**：General、Technical、Billing 三类 Agent，专业 Agent 失败时按显式预算回退。
- **有界执行图**：节点超时、重试、最大执行次数、最大边跳转数和总运行时间共同限制执行链路。
- **执行证据与回放**：记录脱敏 Route Trace / Graph Trace，通过固定案例复现路由与状态图行为。
- **显式 Embedding 管理**：固定 Provider、模型、维度、距离度量和 Schema 版本，使用 fingerprint 隔离不兼容索引。
- **可评测 Chunking**：支持 fixed-char、sliding-window 和 structure-token 三种策略，并比较 Recall@K、MRR、证据覆盖率和上下文 Token。
- **自适应检索策略**：高置信查询跳过 Rewrite / Rerank；不确定查询保留多查询召回和重排链路。
- **工程可靠性**：工具调用支持缓存、熔断、降级和调用预算，Prometheus 暴露运行指标。

## 请求链路

```text
请求
  -> load_memory
  -> decide_retrieval
  -> retrieve（按需）
  -> execute_agent
  -> persist_memory（仅成功后）
  -> complete
```

执行图会记录每个节点的尝试次数、状态、耗时、选择的边和最终 `stop_reason`。Trace 不保存用户原始消息，只保留必要的结构化信息与内容摘要。

## 检索设计

### Embedding

默认配置：

| 配置 | 默认值 |
| --- | --- |
| Provider | FastEmbed 0.8.0 |
| Model | `BAAI/bge-small-zh-v1.5` |
| Dimension | 512 |
| Distance | cosine |

文档向量和查询向量由应用显式生成，ChromaDB 仅负责索引与近邻搜索。模型、维度或距离配置发生变化时，会生成新的 fingerprint 和集合名，避免不同语义空间的向量被写入同一索引。

### Chunking 校准结果

在仓库内置的 4 文档、8 查询小型校准集上，真实本地 Embedding + ChromaDB 的一次 warm-process 结果如下：

| Strategy | Recall@3 | MRR | Evidence coverage | Avg context tokens |
| --- | ---: | ---: | ---: | ---: |
| fixed-char | 1.000 | 0.8125 | 0.9375 | 395.62 |
| sliding-window | 0.875 | 0.8125 | 0.8750 | 477.00 |
| structure-token | 1.000 | 0.8125 | 1.0000 | 332.38 |

该结果只用于校准当前数据集上的策略选择，不代表生产环境泛化能力。完整逐查询结果位于：

```text
data/evidence/vector-retrieval-report.json
```

### Rewrite / Rerank 门控

检索首先执行一次向量查询。只有当 Top-1 分数或候选间隔不足时，才进入 Query Rewrite、多路召回和 LLM Rerank。高置信路径复用首次检索结果，避免重复向量查询；进入重排的候选最多保留 12 条。

受控基准中的延迟来自注入的确定性等待，仅用于验证调用次数和分支行为，不等同于真实模型或线上端到端延迟。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. 配置环境变量

复制示例配置并按需填写模型 Key：

```bash
cp .env.example .env
```

常用配置：

```ini
CHUNK_STRATEGY=structure_token
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
ROUTE_MAX_ATTEMPTS=2
ROUTE_MAX_REROUTES=1
ROUTE_MAX_LATENCY_MS=15000
```

### 3. 启动服务

```bash
uvicorn api.main:app --reload
```

也可以使用 Docker Compose：

```bash
docker compose up --build
```

## 评测与回放

```bash
# 路由回放：不调用外部模型
python scripts/replay_routes.py

# 执行图回放：不调用外部模型
python scripts/replay_graph.py

# Chunk 边界代理评测
python scripts/evaluate_chunking.py

# 真实本地 Embedding + ChromaDB 检索评测
python scripts/evaluate_vector_retrieval.py

# 自适应 Rewrite / Rerank 调用次数基准
python scripts/benchmark_retrieval_policy.py
```

运行测试：

```bash
pytest -q
```

## 主要目录

```text
agents/       专业 Agent 与路由编排
api/          FastAPI 服务入口
core/         意图识别与有界执行图
evidence/     Route / Graph Trace
evaluation/   检索、路由与图回放评测
mcp/          知识库与工具管理
memory/       会话记忆与长期记忆
retrieval/    Chunking 与 Embedding
scripts/      回放和评测脚本
tests/        自动化测试
```

## 当前边界

- 检索评测集只有 4 篇文档和 8 条查询，属于校准集而非生产基准。
- 自适应检索延迟实验使用确定性模拟等待，不能作为线上延迟提升结论。
- 回放测试验证控制流与失败边界，不代表真实 LLM 回复质量。
- 项目当前更适合作为可运行、可评测的 Agent 工程原型，生产使用仍需补充多租户权限、数据治理、真实流量压测和在线效果监控。

## 进一步阅读

- [执行图、Chunking 与检索评测](docs/graph-and-chunking-harness.md)
- [路由预算、证据 Trace 与离线回放](docs/route-budget-and-replay.md)

