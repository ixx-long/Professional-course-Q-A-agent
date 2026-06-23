"""
Prometheus 监控指标模块。

提供系统运行时的关键指标收集，包括：
- 请求计数和延迟
- LLM 调用统计
- 检索性能指标
- 错误统计
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from flask import Response


# 请求相关指标
REQUEST_COUNT = Counter(
    'qa_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'qa_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# LLM 相关指标
LLM_REQUEST_COUNT = Counter(
    'qa_llm_requests_total',
    'Total number of LLM requests',
    ['model', 'status']
)

LLM_TOKEN_COUNT = Counter(
    'qa_llm_tokens_total',
    'Total number of tokens generated',
    ['model']
)

LLM_LATENCY = Histogram(
    'qa_llm_duration_seconds',
    'LLM request latency in seconds',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

# 检索相关指标
RETRIEVAL_COUNT = Counter(
    'qa_retrievals_total',
    'Total number of retrieval operations',
    ['course']
)

RETRIEVAL_LATENCY = Histogram(
    'qa_retrieval_duration_seconds',
    'Retrieval latency in seconds',
    ['course'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

RETRIEVED_DOCS = Histogram(
    'qa_retrieved_docs_count',
    'Number of documents retrieved',
    ['course'],
    buckets=[1, 2, 4, 8, 16, 32]
)

# 会话相关指标
ACTIVE_SESSIONS = Gauge(
    'qa_active_sessions',
    'Number of active sessions'
)

# 错误统计
ERROR_COUNT = Counter(
    'qa_errors_total',
    'Total number of errors',
    ['error_type']
)

# 缓存相关指标
CACHE_HIT_COUNT = Counter(
    'qa_cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISS_COUNT = Counter(
    'qa_cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

CACHE_SIZE = Gauge(
    'qa_cache_size',
    'Current cache size',
    ['cache_type']
)

# 向量库操作指标
VECTORSTORE_OPERATION_COUNT = Counter(
    'qa_vectorstore_operations_total',
    'Total number of vector store operations',
    ['operation_type']
)

VECTORSTORE_OPERATION_LATENCY = Histogram(
    'qa_vectorstore_operation_duration_seconds',
    'Vector store operation latency in seconds',
    ['operation_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# 文档处理指标
DOCUMENT_LOAD_COUNT = Counter(
    'qa_documents_loaded_total',
    'Total number of documents loaded',
    ['document_type']
)

DOCUMENT_CHUNK_COUNT = Histogram(
    'qa_document_chunks_count',
    'Number of chunks per document',
    ['document_type'],
    buckets=[1, 5, 10, 20, 50, 100, 200]
)

# 重排序指标
RERANK_COUNT = Counter(
    'qa_rerank_operations_total',
    'Total number of rerank operations'
)

RERANK_LATENCY = Histogram(
    'qa_rerank_duration_seconds',
    'Rerank operation latency in seconds',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# 系统资源指标
MEMORY_USAGE_BYTES = Gauge(
    'qa_memory_usage_bytes',
    'Current memory usage in bytes'
)

CPU_USAGE_PERCENT = Gauge(
    'qa_cpu_usage_percent',
    'Current CPU usage percentage'
)


def record_request(method: str, endpoint: str, status: int, latency: float) -> None:
    """记录请求指标。"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)


def record_llm_request(model: str, status: str, latency: float, tokens: int = 0) -> None:
    """记录 LLM 请求指标。"""
    LLM_REQUEST_COUNT.labels(model=model, status=status).inc()
    LLM_LATENCY.labels(model=model).observe(latency)
    if tokens > 0:
        LLM_TOKEN_COUNT.labels(model=model).inc(tokens)


def record_retrieval(course: str, latency: float, doc_count: int) -> None:
    """记录检索指标。"""
    RETRIEVAL_COUNT.labels(course=course).inc()
    RETRIEVAL_LATENCY.labels(course=course).observe(latency)
    RETRIEVED_DOCS.labels(course=course).observe(doc_count)


def record_error(error_type: str) -> None:
    """记录错误指标。"""
    ERROR_COUNT.labels(error_type=error_type).inc()


def set_active_sessions(count: int) -> None:
    """设置活跃会话数。"""
    ACTIVE_SESSIONS.set(count)


def record_cache_hit(cache_type: str) -> None:
    """记录缓存命中。"""
    CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """记录缓存未命中。"""
    CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()


def set_cache_size(cache_type: str, size: int) -> None:
    """设置缓存大小。"""
    CACHE_SIZE.labels(cache_type=cache_type).set(size)


def record_vectorstore_operation(operation_type: str, latency: float) -> None:
    """记录向量库操作。"""
    VECTORSTORE_OPERATION_COUNT.labels(operation_type=operation_type).inc()
    VECTORSTORE_OPERATION_LATENCY.labels(operation_type=operation_type).observe(latency)


def record_document_load(document_type: str, chunk_count: int) -> None:
    """记录文档加载。"""
    DOCUMENT_LOAD_COUNT.labels(document_type=document_type).inc()
    DOCUMENT_CHUNK_COUNT.labels(document_type=document_type).observe(chunk_count)


def record_rerank(latency: float) -> None:
    """记录重排序操作。"""
    RERANK_COUNT.inc()
    RERANK_LATENCY.observe(latency)


def record_memory_usage(bytes_used: int) -> None:
    """记录内存使用量。"""
    MEMORY_USAGE_BYTES.set(bytes_used)


def record_cpu_usage(percent: float) -> None:
    """记录 CPU 使用率。"""
    CPU_USAGE_PERCENT.set(percent)


def metrics_endpoint() -> Response:
    """Flask 路由：返回 Prometheus 指标。"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
