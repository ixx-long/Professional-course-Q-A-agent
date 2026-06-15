"""
向量库管理模块。

职责:
  - 根据配置创建 Embedding 模型（支持阿里云百炼 API 和本地 sentence-transformers）
  - 初始化/加载 Chroma 向量库（持久化模式）
  - 文档去重入库（基于 chunk_id）
  - 获取配置好的 Retriever
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Any

from openai import OpenAI

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


# ============================================================
# 阿里云百炼 Embedding 封装（避免 LangChain OpenAIEmbeddings 兼容性问题）
# ============================================================

class BailianEmbeddings(Embeddings):
    """阿里云百炼 Embedding 封装，基于 OpenAI 兼容接口。

    百炼的 /compatible-mode/v1 端点支持 OpenAI 风格的 embedding 请求，
    但 LangChain 的 OpenAIEmbeddings 在批量调用时会附加额外参数导致 400 错误。
    此类使用原生 openai 库直接调用，避免兼容性问题。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v3"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=30.0)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档嵌入。每批最多 25 条，自动重试 3 次。"""
        results: list[list[float]] = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    results.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Embedding 请求失败 (重试 {attempt+1}/3): {e}")
                    time.sleep(1.5 ** attempt)
            logger.debug(f"Embedding 进度: {min(i + batch_size, len(texts))}/{len(texts)}")
        return results

    def embed_query(self, text: str) -> list[float]:
        """生成查询嵌入（自动重试）。"""
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(model=self.model, input=text)
                return resp.data[0].embedding
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 ** attempt)


def get_embedding_model(config: dict[str, Any]) -> Embeddings:
    """
    根据配置创建 Embedding 模型。

    支持两种模式:
      1. API 模式（阿里云百炼）: 通过 BailianEmbeddings（OpenAI 兼容）调用
      2. 本地模式（预留）: 通过 sentence-transformers 本地加载

    Args:
        config: embedding 段配置字典，需包含:
            - api_key
            - api_base
            - model_name

    Returns:
        LangChain 兼容的 Embedding 实例。

    用法:
        embedder = get_embedding_model(config["embedding"])
    """
    emb_config = config["embedding"]
    # 判断是本地模式还是 API 模式：本地模式下 api_base 可设置为 "local"
    if emb_config.get("api_base", "").strip() == "local":
        # 预留本地 sentence-transformers 切换
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "本地 Embedding 模式需要安装 langchain-community 和 sentence-transformers，"
                "请运行: pip install langchain-community sentence-transformers"
            )
        model_name = emb_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"使用本地 Embedding 模型: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        logger.info(f"使用 API Embedding: {emb_config.get('model_name')} @ {emb_config.get('api_base')}")
        return BailianEmbeddings(
            api_key=emb_config["api_key"],
            base_url=emb_config["api_base"],
            model=emb_config.get("model_name", "text-embedding-v3"),
        )


def get_vectorstore(config: dict[str, Any], embedder: Optional[Embeddings] = None) -> Chroma:
    """
    初始化或加载 Chroma 向量库（持久化模式）。

    Args:
        config: 完整配置字典（需包含 chroma 段）。
        embedder: Embedding 实例，若为 None 则自动创建。

    Returns:
        Chroma 向量库实例。

    用法:
        vectorstore = get_vectorstore(config)
    """
    if embedder is None:
        embedder = get_embedding_model(config)

    chroma_config = config["chroma"]
    persist_dir = chroma_config["persist_dir"]
    collection_name = chroma_config.get("collection_name", "course_materials")

    # 确保持久化目录存在
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"加载 Chroma 向量库: collection={collection_name}, persist_dir={persist_dir}")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )

    return vectorstore


def get_existing_chunk_ids(vectorstore: Chroma) -> set[str]:
    """
    获取向量库中已存在的 chunk_id 集合（用于去重）。

    Args:
        vectorstore: Chroma 向量库实例。

    Returns:
        已存在的 chunk_id 集合。若集合为空或不可获取，返回空集。

    Raises:
        RuntimeError: Chroma 数据库异常（非空集合错误）时抛出。
    """
    try:
        results = vectorstore.get()
        if results and results["metadatas"]:
            chunk_ids = {m.get("chunk_id", "") for m in results["metadatas"] if m.get("chunk_id")}
            logger.debug(f"向量库中已存在 {len(chunk_ids)} 个 chunk_id")
            return chunk_ids
    except (IndexError, KeyError, TypeError) as e:
        # 空集合或元数据缺失：正常情况，返回空集
        logger.debug(f"集合为空或元数据缺失: {e}")
        return set()
    except Exception as e:
        # 数据库损坏或其他意外异常：抛出便于排查
        logger.error(f"获取已存在 chunk_id 失败（可能数据库异常）: {e}", exc_info=True)
        raise RuntimeError(f"无法读取 Chroma 向量库数据: {e}") from e
    return set()


def add_documents(
    vectorstore: Chroma,
    documents: list[Document],
    skip_existing: bool = True,
) -> int:
    """
    向向量库添加文档，支持去重。

    去重逻辑: 检查文档的 chunk_id 是否已存在于向量库中。

    Args:
        vectorstore: Chroma 向量库实例。
        documents: 待添加的 Document 列表。
        skip_existing: 是否跳过已存在的文档（True=去重，False=全量添加）。

    Returns:
        实际新增的文档数量。

    用法:
        added = add_documents(vectorstore, docs)
        print(f"新增 {added} 个文本块")
    """
    if not documents:
        logger.info("没有需要添加的文档")
        return 0

    if skip_existing:
        existing_ids = get_existing_chunk_ids(vectorstore)
        new_docs = []
        skipped = 0
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id and chunk_id in existing_ids:
                skipped += 1
            else:
                new_docs.append(doc)

        if skipped > 0:
            logger.info(f"去重: 跳过 {skipped} 个已存在的文本块，新增 {len(new_docs)} 个")

        if not new_docs:
            logger.info("所有文档已存在，无需更新")
            return 0
        documents = new_docs

    # 批量添加（Chroma 自动生成 embedding）
    logger.info(f"正在添加 {len(documents)} 个文档到向量库...")
    vectorstore.add_documents(documents)
    logger.info(f"成功添加 {len(documents)} 个文档")
    return len(documents)


def get_retriever(vectorstore: Chroma, top_k: int = 8) -> VectorStoreRetriever:
    """
    获取配置了检索参数的 Retriever。

    Args:
        vectorstore: Chroma 向量库实例。
        top_k: 检索返回的候选文档数量。

    Returns:
        Chroma Retriever 实例。

    用法:
        retriever = get_retriever(vectorstore, top_k=8)
        docs = retriever.invoke("什么是堆排序？")
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    logger.debug(f"创建 Retriever: top_k={top_k}")
    return retriever
