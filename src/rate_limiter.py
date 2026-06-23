"""
请求限流和防刷模块。

提供基于滑动窗口的限流器，支持多维度限流策略。
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import defaultdict
from functools import wraps
from flask import request, jsonify, g

from src.errors import RateLimitError


class RateLimiter:
    """
    滑动窗口限流器。
    
    支持多维度限流：
    - 基于 IP 地址
    - 基于用户 ID
    - 基于 API Key
    - 自定义维度
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        cleanup_interval: int = 300,
    ):
        """
        初始化限流器。
        
        Args:
            max_requests: 窗口内最大请求数
            window_seconds: 时间窗口（秒）
            cleanup_interval: 清理过期记录的间隔（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        
        # 存储请求记录：{key: [timestamp1, timestamp2, ...]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
    
    def _cleanup_expired(self):
        """清理过期的请求记录。"""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        with self._lock:
            expired_keys = []
            for key, timestamps in self._requests.items():
                # 移除窗口外的时间戳
                valid_timestamps = [
                    ts for ts in timestamps 
                    if current_time - ts < self.window_seconds
                ]
                if valid_timestamps:
                    self._requests[key] = valid_timestamps
                else:
                    expired_keys.append(key)
            
            # 删除空的键
            for key in expired_keys:
                del self._requests[key]
            
            self._last_cleanup = current_time
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, int]]:
        """
        检查请求是否被允许。
        
        Args:
            key: 限流键（如 IP、用户 ID 等）
            
        Returns:
            (是否允许, 统计信息字典)
        """
        self._cleanup_expired()
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            # 获取窗口内的请求
            timestamps = self._requests[key]
            valid_timestamps = [ts for ts in timestamps if ts >= window_start]
            
            # 统计信息
            stats = {
                "current_count": len(valid_timestamps),
                "max_requests": self.max_requests,
                "remaining": max(0, self.max_requests - len(valid_timestamps)),
                "window_seconds": self.window_seconds,
            }
            
            # 检查是否超限
            if len(valid_timestamps) >= self.max_requests:
                self._requests[key] = valid_timestamps
                return False, stats
            
            # 记录本次请求
            valid_timestamps.append(current_time)
            self._requests[key] = valid_timestamps
            stats["current_count"] = len(valid_timestamps)
            stats["remaining"] = max(0, self.max_requests - len(valid_timestamps))
            
            return True, stats
    
    def get_stats(self, key: str) -> Dict[str, int]:
        """
        获取指定键的统计信息。
        
        Args:
            key: 限流键
            
        Returns:
            统计信息字典
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            timestamps = self._requests.get(key, [])
            valid_timestamps = [ts for ts in timestamps if ts >= window_start]
            
            return {
                "current_count": len(valid_timestamps),
                "max_requests": self.max_requests,
                "remaining": max(0, self.max_requests - len(valid_timestamps)),
                "window_seconds": self.window_seconds,
            }
    
    def reset(self, key: str):
        """
        重置指定键的计数。
        
        Args:
            key: 限流键
        """
        with self._lock:
            if key in self._requests:
                del self._requests[key]


class MultiTierRateLimiter:
    """
    多层限流器。
    
    支持同时应用多个限流策略，例如：
    - 每分钟 60 次
    - 每小时 1000 次
    - 每天 10000 次
    """
    
    def __init__(self):
        """初始化多层限流器。"""
        self._limiters: list = []
    
    def add_limiter(self, limiter: RateLimiter, name: str = ""):
        """
        添加限流器。
        
        Args:
            limiter: 限流器实例
            name: 限流器名称（用于日志）
        """
        self._limiters.append({
            "name": name or f"limiter_{len(self._limiters)}",
            "limiter": limiter,
        })
    
    def is_allowed(self, key: str) -> Tuple[bool, Optional[Dict]]:
        """
        检查请求是否被所有层允许。
        
        Args:
            key: 限流键
            
        Returns:
            (是否允许, 被拒绝的限流器信息)
        """
        for limiter_info in self._limiters:
            limiter = limiter_info["limiter"]
            allowed, stats = limiter.is_allowed(key)
            
            if not allowed:
                return False, {
                    "limiter_name": limiter_info["name"],
                    "stats": stats,
                }
        
        return True, None
    
    def get_all_stats(self, key: str) -> Dict[str, Dict]:
        """
        获取所有层的统计信息。
        
        Args:
            key: 限流键
            
        Returns:
            各层统计信息字典
        """
        return {
            info["name"]: info["limiter"].get_stats(key)
            for info in self._limiters
        }


# 全局限流器实例
_rate_limiters: Dict[str, MultiTierRateLimiter] = {}
_limiters_lock = threading.Lock()


def get_rate_limiter(name: str = "default") -> MultiTierRateLimiter:
    """
    获取指定名称的限流器。
    
    Args:
        name: 限流器名称
        
    Returns:
        多层限流器实例
    """
    with _limiters_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = MultiTierRateLimiter()
        return _rate_limiters[name]


def configure_rate_limiter(
    name: str = "default",
    per_minute: int = 60,
    per_hour: int = 1000,
    per_day: int = 10000,
):
    """
    配置限流策略。
    
    Args:
        name: 限流器名称
        per_minute: 每分钟限制
        per_hour: 每小时限制
        per_day: 每天限制
    """
    limiter = get_rate_limiter(name)
    
    if per_minute > 0:
        limiter.add_limiter(
            RateLimiter(max_requests=per_minute, window_seconds=60),
            name="per_minute"
        )
    
    if per_hour > 0:
        limiter.add_limiter(
            RateLimiter(max_requests=per_hour, window_seconds=3600),
            name="per_hour"
        )
    
    if per_day > 0:
        limiter.add_limiter(
            RateLimiter(max_requests=per_day, window_seconds=86400),
            name="per_day"
        )


def rate_limit(
    limiter_name: str = "default",
    key_func: str = "ip",
):
    """
    Flask 路由装饰器：应用限流。
    
    Args:
        limiter_name: 限流器名称
        key_func: 限流键类型（ip/user/api_key）
    
    用法：
        @app.route('/api')
        @rate_limit('default', 'ip')
        def api_route():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 确定限流键
            if key_func == "ip":
                key = request.remote_addr or "unknown"
            elif key_func == "user":
                key = getattr(g, "user_id", None) or request.remote_addr or "unknown"
            elif key_func == "api_key":
                key = request.headers.get("X-API-Key", "unknown")
            else:
                key = "unknown"
            
            # 检查限流
            limiter = get_rate_limiter(limiter_name)
            allowed, rejected_info = limiter.is_allowed(key)
            
            if not allowed:
                raise RateLimitError(
                    f"请求过于频繁，请稍后重试。"
                    f"限制：{rejected_info['stats']['max_requests']} 次/"
                    f"{rejected_info['stats']['window_seconds']} 秒"
                )
            
            # 添加限流信息到响应头
            response = f(*args, **kwargs)
            if hasattr(response, "headers"):
                stats = limiter.get_all_stats(key)
                for limiter_name_key, limiter_stats in stats.items():
                    response.headers[f"X-RateLimit-{limiter_name_key}-Limit"] = str(
                        limiter_stats["max_requests"]
                    )
                    response.headers[f"X-RateLimit-{limiter_name_key}-Remaining"] = str(
                        limiter_stats["remaining"]
                    )
            
            return response
        
        return decorated_function
    
    return decorator


class AbuseDetector:
    """
    滥用检测器。
    
    检测异常行为模式：
    - 短时间内大量失败请求
    - 重复的无效输入
    - 异常的用户行为
    """
    
    def __init__(
        self,
        failure_threshold: int = 10,
        failure_window: int = 300,
        block_duration: int = 3600,
    ):
        """
        初始化滥用检测器。
        
        Args:
            failure_threshold: 失败次数阈值
            failure_window: 失败统计窗口（秒）
            block_duration: 封禁时长（秒）
        """
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.block_duration = block_duration
        
        # 失败记录：{key: [timestamp1, timestamp2, ...]}
        self._failures: Dict[str, list] = defaultdict(list)
        # 封禁记录：{key: unblock_timestamp}
        self._blocked: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record_failure(self, key: str):
        """
        记录失败。
        
        Args:
            key: 用户标识
        """
        current_time = time.time()
        
        with self._lock:
            # 添加失败记录
            self._failures[key].append(current_time)
            
            # 清理过期记录
            window_start = current_time - self.failure_window
            self._failures[key] = [
                ts for ts in self._failures[key] if ts >= window_start
            ]
            
            # 检查是否超过阈值
            if len(self._failures[key]) >= self.failure_threshold:
                self._blocked[key] = current_time + self.block_duration
    
    def is_blocked(self, key: str) -> bool:
        """
        检查是否被封禁。
        
        Args:
            key: 用户标识
            
        Returns:
            是否被封禁
        """
        current_time = time.time()
        
        with self._lock:
            if key not in self._blocked:
                return False
            
            unblock_time = self._blocked[key]
            if current_time >= unblock_time:
                # 解封
                del self._blocked[key]
                if key in self._failures:
                    del self._failures[key]
                return False
            
            return True
    
    def get_failure_count(self, key: str) -> int:
        """
        获取失败次数。
        
        Args:
            key: 用户标识
            
        Returns:
            失败次数
        """
        current_time = time.time()
        window_start = current_time - self.failure_window
        
        with self._lock:
            if key not in self._failures:
                return 0
            
            return len([
                ts for ts in self._failures[key] if ts >= window_start
            ])
    
    def unblock(self, key: str):
        """
        手动解封。
        
        Args:
            key: 用户标识
        """
        with self._lock:
            if key in self._blocked:
                del self._blocked[key]
            if key in self._failures:
                del self._failures[key]


# 全局滥用检测器实例
_abuse_detector: Optional[AbuseDetector] = None


def get_abuse_detector() -> AbuseDetector:
    """获取全局滥用检测器实例。"""
    global _abuse_detector
    if _abuse_detector is None:
        _abuse_detector = AbuseDetector()
    return _abuse_detector


def init_rate_limiter(
    name: str = "default",
    per_minute: int = 60,
    per_hour: int = 1000,
    per_day: int = 10000,
) -> MultiTierRateLimiter:
    """初始化限流器。"""
    configure_rate_limiter(name, per_minute, per_hour, per_day)
    return get_rate_limiter(name)


def init_abuse_detector(
    failure_threshold: int = 10,
    failure_window: int = 300,
    block_duration: int = 3600,
) -> AbuseDetector:
    """初始化全局滥用检测器。"""
    global _abuse_detector
    _abuse_detector = AbuseDetector(
        failure_threshold=failure_threshold,
        failure_window=failure_window,
        block_duration=block_duration,
    )
    return _abuse_detector
