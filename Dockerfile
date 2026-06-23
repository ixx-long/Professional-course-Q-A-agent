# 多阶段构建 — 跳过重型依赖（sentence-transformers→torch 800MB）
FROM python:3.10-slim as builder

WORKDIR /build

# 编译依赖（chromadb 的 hnswlib 需要 C++ 编译器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（跳过 sentence-transformers/torch）
COPY requirements.txt .
RUN grep -v "sentence-transformers" requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir --user -r /tmp/reqs.txt

# ---- 运行阶段 ----
FROM python:3.10-slim

WORKDIR /app

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 从 builder 复制已安装的 Python 包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制项目代码
COPY --chown=appuser:appuser . .

# 运行时目录
RUN mkdir -p data logs models && chown -R appuser:appuser /app

USER appuser
EXPOSE 5000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_OFFLINE=1 \
    QA_CONFIG=/app/config.yaml

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "web_server:create_app()"]
