"""
JWT 认证授权模块。

提供基于 JWT 的 Token 生成、验证和权限管理。
"""

import jwt
import logging
import time
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import request, jsonify, g

from src.errors import AuthError

logger = logging.getLogger(__name__)


class JWTAuthManager:
    """
    JWT 认证管理器。

    功能：
    - Token 生成和验证
    - 支持访问令牌和刷新令牌
    - 权限声明和校验
    - Token 黑名单机制
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        redis_cache=None,
    ):
        """
        初始化 JWT 认证管理器。

        Args:
            secret_key: JWT 签名密钥
            algorithm: 签名算法
            access_token_expire_minutes: 访问令牌过期时间（分钟）
            refresh_token_expire_days: 刷新令牌过期时间（天）
            redis_cache: Redis 缓存实例（可选，用于持久化黑名单）
        """
        # 检查密钥长度，HS256 建议至少 32 字节
        if algorithm == "HS256" and len(secret_key.encode('utf-8')) < 32:
            logger.warning(
                f"JWT 密钥长度不足（当前 {len(secret_key.encode('utf-8'))} 字节，建议至少 32 字节）。"
                "将使用 SHA256 哈希扩展密钥以满足安全要求。"
            )
            # 使用 SHA256 哈希扩展密钥到 32 字节
            secret_key = hashlib.sha256(secret_key.encode('utf-8')).hexdigest()

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.redis_cache = redis_cache

        # Token 黑名单（内存缓存，Redis 作为持久化后端）
        self._blacklist: set = set()
        self._blacklist_key = "jwt:blacklist"
    
    def create_access_token(
        self,
        user_id: str,
        permissions: list = None,
        extra_claims: Dict[str, Any] = None,
    ) -> str:
        """
        创建访问令牌。
        
        Args:
            user_id: 用户 ID
            permissions: 权限列表
            extra_claims: 额外的声明
            
        Returns:
            JWT Token 字符串
        """
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=self.access_token_expire_minutes),
            "type": "access",
            "permissions": permissions or [],
        }
        
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        创建刷新令牌。
        
        Args:
            user_id: 用户 ID
            
        Returns:
            JWT Token 字符串
        """
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=self.refresh_token_expire_days),
            "type": "refresh",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def _is_blacklisted(self, token: str) -> bool:
        """检查 Token 是否在黑名单中（内存 + Redis）。"""
        if token in self._blacklist:
            return True
        if self.redis_cache:
            try:
                if self.redis_cache.exists(f"{self._blacklist_key}:{token}"):
                    self._blacklist.add(token)  # 回填内存缓存
                    return True
            except Exception as e:
                logger.debug(f"Redis 黑名单查询失败: {e}")
        return False

    def verify_token(self, token: str, expected_type: str = "access") -> Dict[str, Any]:
        """
        验证 Token。
        
        Args:
            token: JWT Token 字符串
            expected_type: 期望的 Token 类型（access/refresh）
            
        Returns:
            Token 载荷字典
            
        Raises:
            AuthError: Token 无效或已过期
        """
        if self._is_blacklisted(token):
            raise AuthError("Token 已失效")
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise AuthError("Token 已过期")
        except jwt.InvalidTokenError as e:
            raise AuthError(f"无效的 Token: {e}")
        
        if payload.get("type") != expected_type:
            raise AuthError(f"Token 类型错误，期望 {expected_type}")
        
        return payload
    
    def revoke_token(self, token: str):
        """注销 Token（加入黑名单，同时写入 Redis 持久化）。"""
        self._blacklist.add(token)
        if self.redis_cache:
            try:
                # 计算 Token 剩余有效期作为 Redis TTL
                try:
                    payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
                    exp = payload.get("exp", 0)
                    ttl = max(int(exp - time.time()), 1)
                except Exception:
                    ttl = 86400  # 默认 24 小时
                self.redis_cache.set(f"{self._blacklist_key}:{token}", "1", ex=ttl)
            except Exception as e:
                logger.error(f"Redis 黑名单写入失败: {e}")
    
    def get_user_id_from_token(self, token: str) -> str:
        """
        从 Token 中提取用户 ID。
        
        Args:
            token: JWT Token 字符串
            
        Returns:
            用户 ID
        """
        payload = self.verify_token(token)
        return payload.get("sub")
    
    def get_permissions_from_token(self, token: str) -> list:
        """
        从 Token 中提取权限列表。
        
        Args:
            token: JWT Token 字符串
            
        Returns:
            权限列表
        """
        payload = self.verify_token(token)
        return payload.get("permissions", [])
    
    def has_permission(self, token: str, required_permission: str) -> bool:
        """
        检查 Token 是否拥有指定权限。
        
        Args:
            token: JWT Token 字符串
            required_permission: 需要的权限
            
        Returns:
            是否拥有权限
        """
        permissions = self.get_permissions_from_token(token)
        return required_permission in permissions


# 全局认证管理器实例
_auth_manager: Optional[JWTAuthManager] = None


def get_auth_manager() -> JWTAuthManager:
    """获取全局认证管理器实例。"""
    global _auth_manager
    if _auth_manager is None:
        raise RuntimeError("认证管理器未初始化，请先调用 init_auth_manager")
    return _auth_manager


def init_auth_manager(
    secret_key: str,
    algorithm: str = "HS256",
    access_token_expire_minutes: int = 30,
    refresh_token_expire_days: int = 7,
    redis_cache=None,
) -> JWTAuthManager:
    """初始化全局认证管理器。"""
    global _auth_manager
    _auth_manager = JWTAuthManager(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_minutes=access_token_expire_minutes,
        refresh_token_expire_days=refresh_token_expire_days,
        redis_cache=redis_cache,
    )
    return _auth_manager


def require_auth(f):
    """
    Flask 路由装饰器：要求 JWT 认证。
    
    用法：
        @app.route('/protected')
        @require_auth
        def protected_route():
            user_id = g.user_id
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise AuthError("缺少认证头")
        
        # 支持 Bearer Token 格式
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthError("无效的认证头格式，应为: Bearer <token>")
        
        token = parts[1]
        
        try:
            auth_manager = get_auth_manager()
            payload = auth_manager.verify_token(token)
            
            # 将用户信息存储到 Flask g 对象
            g.user_id = payload.get("sub")
            g.permissions = payload.get("permissions", [])
            g.token = token
            
        except AuthError as e:
            raise e
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(permission: str):
    """
    Flask 路由装饰器：要求特定权限。
    
    用法：
        @app.route('/admin')
        @require_auth
        @require_permission('admin')
        def admin_route():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, "permissions"):
                raise AuthError("未认证")
            
            if permission not in g.permissions:
                raise AuthError(f"缺少权限：{permission}")
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator
