"""
Redis 分布式缓存模块。

提供基于 Redis 的分布式缓存实现，支持：
- 会话存储
- 语义缓存
- 限流计数
- 配置缓存
"""

import json
import time
import logging
from typing import Any, Optional, Dict, List
from datetime import timedelta

try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis 未安装，将使用内存缓存作为降级方案")


class RedisCache:
    """
    Redis 缓存管理器。
    
    提供统一的缓存接口，支持自动降级到内存缓存。
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = True,
    ):
        """
        初始化 Redis 缓存。
        
        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库编号
            password: 密码
            socket_timeout: 套接字超时（秒）
            socket_connect_timeout: 连接超时（秒）
            decode_responses: 是否自动解码响应
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        if not REDIS_AVAILABLE:
            logging.warning("Redis 不可用，使用内存缓存降级")
            self._client = None
            self._fallback_cache: Dict[str, Any] = {}
            return
        
        try:
            self._client = Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses,
            )
            # 测试连接
            self._client.ping()
            logging.info(f"Redis 连接成功：{host}:{port}")
        except Exception as e:
            logging.warning(f"Redis 连接失败：{e}，使用内存缓存降级")
            self._client = None
            self._fallback_cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，不存在则返回 None
        """
        if self._client is None:
            return self._fallback_cache.get(key)
        
        try:
            value = self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logging.error(f"Redis GET 失败：{e}")
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None):
        """
        设置缓存值。
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒），None 表示永不过期
        """
        if self._client is None:
            self._fallback_cache[key] = value
            return
        
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            if expire:
                self._client.setex(key, expire, serialized)
            else:
                self._client.set(key, serialized)
        except Exception as e:
            logging.error(f"Redis SET 失败：{e}")
            self._fallback_cache[key] = value
    
    def delete(self, key: str):
        """
        删除缓存。
        
        Args:
            key: 缓存键
        """
        if self._client is None:
            self._fallback_cache.pop(key, None)
            return
        
        try:
            self._client.delete(key)
        except Exception as e:
            logging.error(f"Redis DELETE 失败：{e}")
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在。
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        if self._client is None:
            return key in self._fallback_cache
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logging.error(f"Redis EXISTS 失败：{e}")
            return False
    
    def incr(self, key: str, amount: int = 1) -> int:
        """
        原子递增。
        
        Args:
            key: 缓存键
            amount: 递增量
            
        Returns:
            递增后的值
        """
        if self._client is None:
            current = self._fallback_cache.get(key, 0)
            new_value = current + amount
            self._fallback_cache[key] = new_value
            return new_value
        
        try:
            return self._client.incr(key, amount)
        except Exception as e:
            logging.error(f"Redis INCR 失败：{e}")
            return 0
    
    def expire(self, key: str, seconds: int):
        """
        设置过期时间。
        
        Args:
            key: 缓存键
            seconds: 过期时间（秒）
        """
        if self._client is None:
            return
        
        try:
            self._client.expire(key, seconds)
        except Exception as e:
            logging.error(f"Redis EXPIRE 失败：{e}")
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        获取匹配的键列表（使用 SCAN 避免阻塞）。
        
        Args:
            pattern: 匹配模式
            
        Returns:
            键列表
        """
        if self._client is None:
            return list(self._fallback_cache.keys())
        
        try:
            # 使用 SCAN 代替 KEYS，避免阻塞 Redis
            cursor = 0
            keys = []
            while True:
                cursor, batch = self._client.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            logging.error(f"Redis SCAN 失败：{e}")
            return []
    
    def flush_db(self):
        """清空当前数据库。"""
        if self._client is None:
            self._fallback_cache.clear()
            return
        
        try:
            self._client.flushdb()
        except Exception as e:
            logging.error(f"Redis FLUSHDB 失败：{e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查。
        
        Returns:
            健康状态字典
        """
        if self._client is None:
            return {
                "status": "degraded",
                "mode": "memory",
                "cache_size": len(self._fallback_cache),
            }
        
        try:
            info = self._client.info()
            return {
                "status": "healthy",
                "mode": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "N/A"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


class SessionStore:
    """
    基于 Redis 的会话存储。
    
    提供会话的 CRUD 操作，支持自动过期。
    """
    
    def __init__(
        self,
        redis_cache: RedisCache,
        session_prefix: str = "session:",
        default_ttl: int = 3600,
    ):
        """
        初始化会话存储。
        
        Args:
            redis_cache: Redis 缓存实例
            session_prefix: 会话键前缀
            default_ttl: 默认会话过期时间（秒）
        """
        self.redis = redis_cache
        self.prefix = session_prefix
        self.default_ttl = default_ttl
    
    def create_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        创建会话。
        
        Args:
            session_id: 会话 ID
            data: 会话数据
            ttl: 过期时间（秒），None 使用默认值
        """
        key = f"{self.prefix}{session_id}"
        expire = ttl or self.default_ttl
        
        session_data = {
            "data": data,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
        self.redis.set(key, session_data, expire=expire)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            会话数据，不存在则返回 None
        """
        key = f"{self.prefix}{session_id}"
        session_data = self.redis.get(key)
        
        if session_data is None:
            return None
        
        # 更新访问时间
        if "data" in session_data:
            session_data["updated_at"] = time.time()
            self.redis.set(key, session_data, expire=self.default_ttl)
        
        return session_data.get("data")
    
    def update_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        更新会话。
        
        Args:
            session_id: 会话 ID
            data: 新的会话数据
            ttl: 过期时间（秒）
        """
        key = f"{self.prefix}{session_id}"
        expire = ttl or self.default_ttl
        
        session_data = {
            "data": data,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
        self.redis.set(key, session_data, expire=expire)
    
    def delete_session(self, session_id: str):
        """
        删除会话。
        
        Args:
            session_id: 会话 ID
        """
        key = f"{self.prefix}{session_id}"
        self.redis.delete(key)
    
    def session_exists(self, session_id: str) -> bool:
        """
        检查会话是否存在。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            是否存在
        """
        key = f"{self.prefix}{session_id}"
        return self.redis.exists(key)
    
    def list_sessions(self) -> List[str]:
        """
        列出所有会话 ID。
        
        Returns:
            会话 ID 列表
        """
        pattern = f"{self.prefix}*"
        keys = self.redis.keys(pattern)
        return [key.replace(self.prefix, "") for key in keys]


# 全局 Redis 缓存实例
_redis_cache: Optional[RedisCache] = None
_session_store: Optional[SessionStore] = None


def get_redis_cache() -> RedisCache:
    """获取全局 Redis 缓存实例。"""
    global _redis_cache
    if _redis_cache is None:
        raise RuntimeError("Redis 缓存未初始化，请先调用 init_redis_cache")
    return _redis_cache


def init_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
) -> RedisCache:
    """初始化全局 Redis 缓存。"""
    global _redis_cache
    _redis_cache = RedisCache(
        host=host,
        port=port,
        db=db,
        password=password,
    )
    return _redis_cache


def get_session_store() -> SessionStore:
    """获取全局会话存储实例。"""
    global _session_store
    if _session_store is None:
        redis_cache = get_redis_cache()
        _session_store = SessionStore(redis_cache)
    return _session_store
