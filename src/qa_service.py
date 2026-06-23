"""
问答服务层 — 封装所有业务逻辑，供路由层调用。

职责:
  - 系统初始化（RAG 组件加载）
  - 问答核心逻辑（文本问答、图片问答、流式问答）
  - 语义缓存管理
  - 课程 Chain 缓存管理
  - 输入校验
"""

import json
import re
import base64
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

from src.utils import load_config, setup_logger
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.chain import create_qa_chain, ChatHistory
from src.errors import (
    QAError, ConfigError, RequestTimeoutError,
    LLMError, RateLimitError
)
from src.metrics import (
    record_request, record_llm_request, record_retrieval,
    record_error, set_active_sessions
)
from src.semantic_cache import SemanticCache
from src.session_manager import SessionManager

# 企业级增强模块（可选导入）
try:
    from src.config_manager import get_config_manager, ConfigChangeHandler
    from src.structured_logging import setup_structured_logger
    from src.monitoring import get_metric_collector
    from src.redis_cache import get_redis_cache
    ENTERPRISE_ENABLED = True
except ImportError:
    ENTERPRISE_ENABLED = False


class QAService:
    """问答业务逻辑层，不依赖 Flask request 上下文。"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path: str = config_path

        # 核心组件（延迟初始化）
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        self.compression_retriever: Optional[VectorStoreRetriever] = None
        self.llm: Optional[ChatOpenAI] = None
        self.embedder: Optional[Embeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.config: Optional[Dict[str, Any]] = None
        self.logger: Optional[Any] = None
        self.cross_encoder: Optional[Any] = None

        # Session 管理
        self.session_manager: SessionManager = SessionManager(
            sessions_file=Path(__file__).parent.parent / "data" / "sessions.json",
            max_sessions=100,
            max_turns=4,
            save_interval=30,
        )

        # 课程筛选 Chain 缓存
        self._filtered_chains: Dict[str, ConversationalRetrievalChain] = {}
        self._filtered_chains_lock: threading.Lock = threading.Lock()
        self._filtered_chains_times: Dict[str, float] = {}
        self._MAX_FILTERED_CHAINS: int = 20

        # 语义缓存
        self.semantic_cache: SemanticCache = SemanticCache(
            cache_dir="./data/semantic_cache",
            similarity_threshold=0.95,
            max_cache_size=1000
        )

    def init_system(self) -> None:
        """初始化 RAG 系统组件。"""
        self.config = load_config(self.config_path)

        log_cfg = self.config.get("logging", {})
        self.logger = setup_logger(
            name="qa_service",
            log_file=log_cfg.get("file", "logs/qa.log"),
            level=log_cfg.get("level", "INFO"),
        )
        self.logger.info("LLM 和 Embedding API Key 已加载")

        # Embedding + 向量库
        self.embedder = get_embedding_model(self.config)
        self.vectorstore = get_vectorstore(self.config, self.embedder)
        self.retriever = get_retriever(
            self.vectorstore,
            top_k=self.config.get("retrieval", {}).get("top_k", 8)
        )

        # CrossEncoder 可选
        try:
            from src.retriever import load_cross_encoder, create_compression_retriever
            reranker_cfg = self.config.get("reranker", {})
            cross_encoder = load_cross_encoder(
                model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-base"),
                cache_dir=reranker_cfg.get("cache_dir", "./models"),
            )
            self.compression_retriever = create_compression_retriever(
                self.retriever, cross_encoder, self.config.get("retrieval", {})
            )
            self.logger.info("重排序模型加载成功")
        except Exception as e:
            self.logger.warning(f"重排序模型加载失败，使用基础检索: {e}")
            self.compression_retriever = self.retriever

        # LLM 实例
        from src.chain import get_llm
        self.llm = get_llm(self.config)

        # 对话链
        self.qa_chain = create_qa_chain(self.compression_retriever, self.config, llm=self.llm)

        # 恢复持久化的 session 数据
        self.session_manager.load(self.config, self.logger)

        self.logger.info("系统初始化完成")

    # ============================================================
    # 输入校验
    # ============================================================

    def validate_image(self, image_b64: str) -> Optional[str]:
        """校验图片 base64。通过返回 None，失败返回错误消息。"""
        if not image_b64 or not isinstance(image_b64, str):
            return "图片数据为空或格式错误"
        if len(image_b64) > 10 * 1024 * 1024:  # 10MB base64 ≈ 7.5MB raw
            return "图片过大（最大 10MB）"
        if image_b64.startswith("data:image/"):
            if not re.match(r'^data:image/(png|jpeg|jpg|gif|webp);base64,', image_b64):
                return "不支持的图片格式（支持 PNG/JPEG/GIF/WebP）"
            try:
                base64.b64decode(image_b64.split(",", 1)[1], validate=True)
            except Exception:
                return "图片 base64 编码无效"
        else:
            try:
                base64.b64decode(image_b64, validate=True)
            except Exception:
                return "图片 base64 编码无效"
        return None

    def check_vision_support(self) -> bool:
        """检查当前 LLM 是否支持图片问答。"""
        model_name = self.config.get("llm", {}).get("model_name", "")
        return any(kw in model_name.lower() for kw in ("vision", "gpt-4", "claude", "gemini"))

    # ============================================================
    # 核心业务方法
    # ============================================================

    def invoke_with_retry(self, chain: Any, inputs: Dict[str, Any], max_retries: int = 2) -> Any:
        """带自动重试的 chain.invoke 调用（指数退避）。"""
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    delay = 2 ** attempt
                    if self.logger:
                        self.logger.warning(
                            f"Chain 调用失败 (重试 {attempt+1}/{max_retries}，等待 {delay:.1f}s): {e}"
                        )
                    _time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"Chain 调用失败（已达最大重试次数）: {e}")
        raise last_err

    def get_chain_for_course(self, course: str) -> ConversationalRetrievalChain:
        """获取指定课程的 QA Chain（带 LRU 缓存，线程安全）。"""
        with self._filtered_chains_lock:
            if course not in self._filtered_chains:
                if len(self._filtered_chains) >= self._MAX_FILTERED_CHAINS:
                    oldest_course = min(
                        self._filtered_chains_times,
                        key=self._filtered_chains_times.get
                    )
                    del self._filtered_chains[oldest_course]
                    del self._filtered_chains_times[oldest_course]
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.config.get("retrieval", {}).get("top_k", 8),
                        "filter": {"course": course},
                    },
                )
                self._filtered_chains[course] = create_qa_chain(
                    retriever, self.config, llm=self.llm
                )
            self._filtered_chains_times[course] = _time.time()
            return self._filtered_chains[course]

    def check_semantic_cache(self, question: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        """查询语义缓存。返回 (cached_result, question_embedding) 或 (None, None)。"""
        if not self.semantic_cache:
            return None, None
        try:
            question_embedding = self.embedder.embed_query(question)
            cached_result = self.semantic_cache.get(question, question_embedding)
            return cached_result, question_embedding
        except Exception as e:
            if self.logger:
                self.logger.warning(f"语义缓存查询失败: {e}")
            return None, None

    def write_semantic_cache(self, question: str, question_embedding: Optional[List[float]], answer: str, sources: List[Document]) -> None:
        """写入语义缓存（失败不抛异常）。"""
        if self.semantic_cache and question_embedding is not None:
            try:
                self.semantic_cache.put(
                    question, question_embedding, answer,
                    self.extract_sources(sources)
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"语义缓存写入失败: {e}")

    def classify_llm_error(self, e: Exception) -> QAError:
        """根据异常信息分类为具体的 QAError 子类。"""
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            return RequestTimeoutError("请求超时，请稍后重试")
        if "401" in str(e) or "403" in str(e) or "invalid api key" in error_msg:
            return ConfigError("API Key 无效，请检查 config.yaml 配置")
        if "rate" in error_msg:
            return RateLimitError("请求过于频繁，请稍后重试")
        return LLMError("服务内部错误，请稍后重试")

    def extract_sources(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """从 Document 列表提取去重的来源信息。"""
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for doc in (source_docs or []):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "N/A")
            score = doc.metadata.get("rerank_score")
            key = f"{source}#p{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                    "score": round(float(score), 4) if score is not None else None,
                    "content": doc.page_content[:300],
                })
        return sources

    def ask_with_image(self, question: str, image_b64: str, chat_history: ChatHistory) -> Tuple[str, List[Document]]:
        """多模态问答：检索 + 图片 + LLM 直调。"""
        search_query = question or "请描述这张图片的内容"
        raw_docs = self.compression_retriever.invoke(
            search_query
        )[:self.config.get("retrieval", {}).get("rerank_top_n", 4)]
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n"
            f"{d.page_content[:600]}"
            for d in raw_docs
        ) if raw_docs else "（知识库中未找到相关内容）"

        system_msg = SystemMessage(content=(
            "你是一个专业课程答疑助手。请根据提供的知识库上下文回答用户问题。"
            "如果知识库信息不足，请如实说明。回答格式：\n"
            "【依据】…\n【解答】…\n【来源】…"
        ))

        history_str = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in chat_history.get_history()
        ) if chat_history.get_history() else "（无历史）"

        content_parts = [
            {
                "type": "text",
                "text": (
                    f"## 知识库上下文\n{context}\n\n"
                    f"## 对话历史\n{history_str}\n\n"
                    f"## 用户问题\n{question or '请分析这张图片，结合知识库内容回答。'}"
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": image_b64}
            },
        ]
        human_msg = HumanMessage(content=content_parts)

        try:
            resp = self.llm.invoke([system_msg, human_msg])
            answer = resp.content
        except Exception as e:
            if self.logger:
                self.logger.error(f"多模态问答失败: {e}", exc_info=True)
            raise self.classify_llm_error(e)

        return answer, raw_docs

    def shutdown(self) -> None:
        """优雅关闭：保存所有持久化数据。"""
        if self.logger:
            self.logger.info("正在优雅关闭...")
        try:
            self.session_manager.save()
            if self.semantic_cache:
                self.semantic_cache.save()
            if self.logger:
                self.logger.info("数据保存完成")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存数据时出错: {e}")
