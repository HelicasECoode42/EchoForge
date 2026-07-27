# EchoMind 智能客服系统 — Docker 多阶段构建
# 目标：生产镜像尽量精简，开发镜像包含调试工具

# ── 阶段 1：基础环境 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# curl 用于健康检查；Embedding 使用 FastEmbed ONNX，无需 gcc/g++。
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 阶段 2：安装 Python 依赖 ──────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 预下载固定的中文 BGE ONNX 模型，避免应用首次请求时联网。
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-small-zh-v1.5', cache_dir='/root/.cache/fastembed').query_embed(['warmup']))"

# ── 阶段 3：生产镜像 ──────────────────────────────────────────────────────────
FROM base AS production

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
# 复制预下载的 ONNX 模型缓存
COPY --from=dependencies /root/.cache/fastembed /home/echomind/.cache/fastembed

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p /app/data/chroma /app/logs /app/config

# 非 root 用户运行
RUN useradd -m -u 1000 echomind && \
    chown -R echomind:echomind /app && \
    chown -R echomind:echomind /home/echomind/.cache
USER echomind

ENV EMBEDDING_PROVIDER=fastembed \
    EMBEDDING_PROVIDER_VERSION=0.8.0 \
    EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
    EMBEDDING_DIMENSIONS=512 \
    EMBEDDING_DISTANCE=cosine \
    EMBEDDING_CACHE_DIR=/home/echomind/.cache/fastembed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── 阶段 4：开发镜像 ──────────────────────────────────────────────────────────
FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    useradd -m -u 1000 echomind && \
    mkdir -p /home/echomind/.cache && \
    if [ -d /root/.cache/fastembed ]; then cp -R /root/.cache/fastembed /home/echomind/.cache/fastembed; else echo "Warning: fastembed cache not found; runtime download may be required"; fi && \
    chown -R echomind:echomind /app/data /app/logs /home/echomind/.cache

USER echomind

ENV EMBEDDING_PROVIDER=fastembed \
    EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
    EMBEDDING_DIMENSIONS=512 \
    EMBEDDING_DISTANCE=cosine \
    EMBEDDING_CACHE_DIR=/home/echomind/.cache/fastembed

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
