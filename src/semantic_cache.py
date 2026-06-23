"""
语义缓存模块。

基于向量相似度缓存问答结果，提高响应速度。
当新问题与缓存问题的相似度超过阈值时，直接返回缓存答案。
"""

import logging
import hashlib
import json
import time
import threading
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """语义缓存管理器。"""
    
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        similarity_threshold: float = 0.95,
        max_cache_size: int = 1000,
        ttl: Optional[int] = None
    ) -> None:
        """
        初始化语义缓存。
        
        Args:
            cache_dir: 缓存目录
            similarity_threshold: 相似度阈值（0-1），超过此值认为问题相似
            max_cache_size: 最大缓存条目数
            ttl: 缓存过期时间（秒），None 表示不过期
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = ttl
        
        # 内存缓存：question_hash -> {embedding, answer, sources, timestamp}
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 写盘节流
        self._last_save = 0
        self._save_interval = 30  # 30秒
        
        # 加载持久化缓存
        self._load_cache()
        
        ttl_str = f"，TTL={ttl}秒" if ttl else "，无过期"
        logger.info(f"语义缓存初始化完成，阈值={similarity_threshold}，最大条目={max_cache_size}{ttl_str}")
    
    def _load_cache(self) -> None:
        """从磁盘加载缓存。"""
        cache_file = self.cache_dir / "semantic_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get("cache", {})
                logger.info(f"从磁盘加载了 {len(self.cache)} 条缓存")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """持久化缓存到磁盘。"""
        cache_file = self.cache_dir / "semantic_cache.json"
        try:
            # 在锁内复制缓存数据，避免序列化时数据被修改
            with self.lock:
                cache_data = {"cache": self.cache.copy()}
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"缓存已保存到磁盘，共 {len(cache_data['cache'])} 条")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _compute_hash(self, text: str) -> str:
        """计算文本的哈希值。"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        a = np.array(emb1)
        b = np.array(emb2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # 防止除零
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def get(self, question: str, embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        查询缓存。
        
        Args:
            question: 问题文本
            embedding: 问题的向量表示
            
        Returns:
            缓存的问答结果，如果未命中则返回 None
        """
        with self.lock:
            if not self.cache:
                return None
            
            # 清理过期条目
            if self.ttl:
                current_time = time.time()
                expired_keys = []
                for q_hash, cached_item in self.cache.items():
                    timestamp = cached_item.get("timestamp", 0)
                    if current_time - timestamp > self.ttl:
                        expired_keys.append(q_hash)
                
                for q_hash in expired_keys:
                    del self.cache[q_hash]
                    logger.debug(f"清理过期缓存: {q_hash[:8]}...")
                
                if expired_keys:
                    self._save_cache_throttled()
            
            if not self.cache:
                return None
            
            # 批量计算余弦相似度（使用 numpy 矩阵运算）
            query_vec = np.array(embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return None
            
            # 提取所有缓存的 embedding 到矩阵
            cache_keys = []
            embeddings_list = []
            for q_hash, cached_item in self.cache.items():
                cached_emb = cached_item.get("embedding")
                if cached_emb:
                    cache_keys.append(q_hash)
                    embeddings_list.append(cached_emb)
            
            if not embeddings_list:
                return None
            
            # 转换为矩阵并批量计算
            embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            norms = np.linalg.norm(embeddings_matrix, axis=1)
            
            # 防止除零
            valid_mask = norms > 0
            if not np.any(valid_mask):
                return None
            
            # 计算相似度（只计算有效的）
            similarities = np.zeros(len(cache_keys))
            similarities[valid_mask] = np.dot(embeddings_matrix[valid_mask], query_vec) / (norms[valid_mask] * query_norm)
            
            # 找到最佳匹配
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # 检查是否超过阈值
            if best_similarity >= self.similarity_threshold:
                best_match = self.cache[cache_keys[best_idx]]
                logger.info(f"语义缓存命中，相似度={best_similarity:.4f}")
                # 确保 sources 始终是 List[Dict] 类型
                sources = best_match.get("sources", [])
                if not isinstance(sources, list):
                    sources = []
                return {
                    "answer": best_match.get("answer"),
                    "sources": sources,
                    "cached": True,
                    "similarity": float(best_similarity)
                }
            
            return None
    
    def put(self, question: str, embedding: List[float], answer: str, sources: List[Dict]) -> None:
        """
        添加缓存条目。
        
        Args:
            question: 问题文本
            embedding: 问题的向量表示
            answer: 答案文本
            sources: 来源文档列表
        """
        q_hash = self._compute_hash(question)
        
        with self.lock:
            # 如果缓存已满，删除最旧的条目
            if len(self.cache) >= self.max_cache_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get("timestamp", 0))
                del self.cache[oldest_key]
                logger.debug(f"缓存已满，删除最旧条目: {oldest_key[:8]}...")
            
            # 添加新条目
            self.cache[q_hash] = {
                "question": question,
                "embedding": embedding,
                "answer": answer,
                "sources": sources,
                "timestamp": time.time()
            }
            
            logger.debug(f"缓存已添加，当前条目数: {len(self.cache)}")
        
        # 节流保存到磁盘（在锁外执行）
        self._save_cache_throttled()
    
    def _save_cache_throttled(self) -> None:
        """节流保存缓存到磁盘（30秒间隔）。"""
        current_time = time.time()
        with self.lock:
            if current_time - self._last_save >= self._save_interval:
                self._last_save = current_time
                # 在锁外执行实际的文件写入操作
                should_save = True
            else:
                should_save = False
        
        if should_save:
            self._save_cache()
    
    def clear(self) -> None:
        """清空缓存。"""
        with self.lock:
            self.cache.clear()
        self._save_cache()
        logger.info("语义缓存已清空")
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_cache_size,
                "similarity_threshold": self.similarity_threshold,
                "cache_dir": str(self.cache_dir)
            }
