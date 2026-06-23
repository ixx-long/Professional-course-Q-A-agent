"""
检索与重排序模块。

流程:
  1. 使用 Chroma Retriever 检索 top_k 个候选文档
  2. 使用 CrossEncoder 对候选文档重新打分
  3. 返回 top_n 个最相关文档

CrossEncoder 模型首次运行时自动下载至 cache_dir。
"""

# 模块加载时设置 HF 环境（必须在导入 sentence_transformers 之前）
import os as _os
_os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
_os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '10')

import logging
from pathlib import Path
from typing import List, Tuple, Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

logger = logging.getLogger(__name__)


def load_cross_encoder(model_name: str, cache_dir: str = "./models", download_timeout: int = 15, device: str = None) -> "CrossEncoder":
    """加载 CrossEncoder 模型（带缓存处理和下载超时）。

    Args:
        model_name: HuggingFace 模型名称。
        cache_dir: 模型缓存目录。
        download_timeout: 模型下载超时秒数（默认 15s，内网场景避免卡死）。
        device: 设备选择（'cuda'/'cpu'/None，None 表示自动检测）。

    Returns:
        CrossEncoder 实例。

    Raises:
        ImportError: sentence-transformers 未安装时抛出。
        TimeoutError: 模型下载超时。
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers 未安装，请运行: pip install sentence-transformers"
        )

    # 自动检测设备
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    
    logger.info(f"CrossEncoder 使用设备: {device}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 检查是否已缓存，若已缓存则离线加载（跳过网络版本检查）
    model_dir = cache_path / ("models--" + model_name.replace("/", "--"))
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        logger.info(f"CrossEncoder 模型已缓存，离线加载: {model_dir}")
        _os.environ['HF_HUB_OFFLINE'] = '1'
        return CrossEncoder(model_name, cache_dir=str(cache_path), device=device)

    logger.info(f"下载 CrossEncoder 模型: {model_name}（超时 {download_timeout}s）")
    # 使用线程 + 超时避免网络问题卡死启动
    import threading
    result = {"model": None, "error": None}

    def _download():
        try:
            result["model"] = CrossEncoder(model_name, cache_dir=str(cache_path), device=device)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    t.join(timeout=download_timeout)

    if result["model"] is not None:
        logger.info("CrossEncoder 模型加载完成")
        return result["model"]
    if result["error"] is not None:
        raise result["error"]
    from src.errors import RequestTimeoutError
    raise RequestTimeoutError(f"CrossEncoder 模型下载超时（{download_timeout}s）")


def _rerank_with_cross_encoder(
    query: str,
    documents: list[Document],
    cross_encoder: "CrossEncoder",
    top_n: int = 4,
) -> list[Document]:
    """
    使用 CrossEncoder 对候选文档重新排序。

    Args:
        query: 用户问题。
        documents: 候选文档列表。
        cross_encoder: CrossEncoder 实例。
        top_n: 保留的文档数量。

    Returns:
        按相关性降序排列的 top_n 个 Document。
    """
    if not documents:
        return []

    # 构造 (query, doc_content) 对
    pairs = [(query, doc.page_content) for doc in documents]

    # CrossEncoder 打分
    scores = cross_encoder.predict(pairs)

    # 按分数降序排列
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 取 top_n，复制文档避免污染原始对象 metadata
    top_docs: list[Document] = []
    for score, doc in scored_docs[:top_n]:
        doc_copy = doc.model_copy()
        doc_copy.metadata["rerank_score"] = float(score)
        top_docs.append(doc_copy)

    logger.debug(
        f"重排序: {len(documents)} -> {len(top_docs)} 个文档, "
        f"最高分={scored_docs[0][0]:.4f}"
    )

    return top_docs


def retrieve_and_rerank(
    query: str,
    retriever: VectorStoreRetriever,
    cross_encoder: "CrossEncoder",
    top_n: int = 4,
) -> Tuple[list[Document], list[Document]]:
    """
    执行检索 + 重排序的完整流程。

    Args:
        query: 用户问题。
        retriever: LangChain Retriever 实例。
        cross_encoder: CrossEncoder 实例。
        top_n: 重排序后保留的文档数。

    Returns:
        (reranked_docs, raw_docs): 重排序后的文档列表和原始检索结果列表。

    用法:
        docs, raw = retrieve_and_rerank("什么是死锁？", retriever, ce, top_n=4)
        for doc in docs:
            print(f"[{doc.metadata['source']}] {doc.page_content[:100]}...")
    """
    # Step 1: 基础检索
    raw_docs = retriever.invoke(query)
    logger.debug(f"检索: 召回 {len(raw_docs)} 个候选文档")

    # Step 2: 重排序
    if cross_encoder is not None and len(raw_docs) > top_n:
        reranked = _rerank_with_cross_encoder(query, raw_docs, cross_encoder, top_n)
        return reranked, raw_docs
    else:
        # 候选数不足 top_n 时，仍需打分以保持 source.retrieval_score 一致性
        if cross_encoder is not None and raw_docs:
            pairs = [(query, doc.page_content) for doc in raw_docs]
            scores = cross_encoder.predict(pairs)
            for score, doc in zip(scores, raw_docs):
                doc.metadata["rerank_score"] = float(score)
        logger.debug(f"候选文档不足 {top_n}，全部保留")
        return raw_docs[:top_n], raw_docs


# 兼容 LangChain ContextualCompressionRetriever 的封装（用于链内集成）
def create_compression_retriever(
    retriever: VectorStoreRetriever,
    cross_encoder: "CrossEncoder",
    config: dict[str, Any],
) -> ContextualCompressionRetriever:
    """
    使用 LangChain 的 ContextualCompressionRetriever 封装重排序。

    这是 retrieve_and_rerank 的替代方案，适用于希望完全走 LangChain 链的场景。

    Args:
        retriever: LangChain Retriever。
        cross_encoder: CrossEncoder 实例。
        config: 检索配置段。

    Returns:
        ContextualCompressionRetriever 实例。
    """
    top_n = config.get("rerank_top_n", 4)
    logger.info(f"创建压缩 Retriever: top_n={top_n}")

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=top_n,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
