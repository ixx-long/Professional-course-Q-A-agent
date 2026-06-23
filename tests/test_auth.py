"""
JWT 认证授权模块单元测试。
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.auth import (
    JWTAuthManager,
    init_auth_manager,
    get_auth_manager,
)
from src.errors import AuthError


class TestJWTAuthManager:
    """JWTAuthManager 单元测试类。"""

    @pytest.fixture
    def auth_manager(self):
        """创建测试用 JWTAuthManager 实例。"""
        return JWTAuthManager(
            secret_key="test_secret_key",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
        )

    def test_create_access_token(self, auth_manager):
        """测试创建访问令牌。"""
        token = auth_manager.create_access_token(
            user_id="user123",
            permissions=["read", "write"],
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 验证 token 内容
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "user123"
        assert payload["type"] == "access"
        assert "read" in payload["permissions"]
        assert "write" in payload["permissions"]

    def test_create_access_token_with_extra_claims(self, auth_manager):
        """测试创建带额外声明的访问令牌。"""
        token = auth_manager.create_access_token(
            user_id="user123",
            extra_claims={"custom_field": "custom_value"},
        )
        
        payload = auth_manager.verify_token(token)
        assert payload["custom_field"] == "custom_value"

    def test_create_refresh_token(self, auth_manager):
        """测试创建刷新令牌。"""
        token = auth_manager.create_refresh_token(user_id="user123")
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 验证 token 类型
        payload = auth_manager.verify_token(token, expected_type="refresh")
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"

    def test_verify_token_expired(self, auth_manager):
        """测试验证过期令牌。"""
        # 创建一个已过期的 token
        auth_manager.access_token_expire_minutes = 0
        token = auth_manager.create_access_token(user_id="user123")
        
        # 等待 1 秒确保过期
        time.sleep(1)
        
        with pytest.raises(AuthError, match="Token 已过期"):
            auth_manager.verify_token(token)

    def test_verify_token_invalid(self, auth_manager):
        """测试验证无效令牌。"""
        with pytest.raises(AuthError, match="无效的 Token"):
            auth_manager.verify_token("invalid_token")

    def test_verify_token_wrong_type(self, auth_manager):
        """测试验证类型错误的令牌。"""
        token = auth_manager.create_access_token(user_id="user123")
        
        with pytest.raises(AuthError, match="Token 类型错误"):
            auth_manager.verify_token(token, expected_type="refresh")

    def test_revoke_token(self, auth_manager):
        """测试注销令牌。"""
        token = auth_manager.create_access_token(user_id="user123")
        
        # 注销前可以验证
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "user123"
        
        # 注销令牌
        auth_manager.revoke_token(token)
        
        # 注销后应该失败
        with pytest.raises(AuthError, match="Token 已失效"):
            auth_manager.verify_token(token)

    def test_revoke_token_with_redis(self, auth_manager):
        """测试使用 Redis 持久化黑名单。"""
        mock_redis = Mock()
        auth_manager.redis_cache = mock_redis
        
        token = auth_manager.create_access_token(user_id="user123")
        auth_manager.revoke_token(token)
        
        # 验证 Redis 被调用
        assert mock_redis.set.called
        call_args = mock_redis.set.call_args
        assert "jwt:blacklist:" in call_args[0][0]
        assert call_args[0][1] == "1"
        assert "ex" in call_args[1]  # 设置了 TTL

    def test_is_blacklisted_memory(self, auth_manager):
        """测试内存黑名单检查。"""
        token = "test_token"
        
        # 初始不在黑名单
        assert not auth_manager._is_blacklisted(token)
        
        # 添加到黑名单
        auth_manager._blacklist.add(token)
        
        # 应该在黑名单
        assert auth_manager._is_blacklisted(token)

    def test_is_blacklisted_redis(self, auth_manager):
        """测试 Redis 黑名单检查。"""
        mock_redis = Mock()
        mock_redis.exists.return_value = True
        auth_manager.redis_cache = mock_redis
        
        token = "test_token"
        
        # 应该从 Redis 查询并回填内存
        assert auth_manager._is_blacklisted(token)
        assert token in auth_manager._blacklist

    def test_is_blacklisted_redis_failure(self, auth_manager):
        """测试 Redis 查询失败时降级到内存。"""
        mock_redis = Mock()
        mock_redis.exists.side_effect = Exception("Redis error")
        auth_manager.redis_cache = mock_redis
        
        token = "test_token"
        
        # Redis 失败，应该返回 False（不在内存黑名单中）
        assert not auth_manager._is_blacklisted(token)

    def test_get_user_id_from_token(self, auth_manager):
        """测试从令牌提取用户 ID。"""
        token = auth_manager.create_access_token(user_id="user123")
        
        user_id = auth_manager.get_user_id_from_token(token)
        assert user_id == "user123"

    def test_get_permissions_from_token(self, auth_manager):
        """测试从令牌提取权限列表。"""
        token = auth_manager.create_access_token(
            user_id="user123",
            permissions=["read", "write", "admin"],
        )
        
        permissions = auth_manager.get_permissions_from_token(token)
        assert permissions == ["read", "write", "admin"]

    def test_has_permission_true(self, auth_manager):
        """测试拥有权限。"""
        token = auth_manager.create_access_token(
            user_id="user123",
            permissions=["read", "write"],
        )
        
        assert auth_manager.has_permission(token, "read")
        assert auth_manager.has_permission(token, "write")

    def test_has_permission_false(self, auth_manager):
        """测试缺少权限。"""
        token = auth_manager.create_access_token(
            user_id="user123",
            permissions=["read"],
        )
        
        assert not auth_manager.has_permission(token, "admin")


class TestGlobalAuthManager:
    """全局认证管理器测试。"""

    def test_init_auth_manager(self):
        """测试初始化全局认证管理器。"""
        # 使用足够长的密钥（>=32字节），避免触发哈希扩展
        test_secret = "this_is_a_very_long_secret_key_for_testing_32_bytes"
        manager = init_auth_manager(
            secret_key=test_secret,
            algorithm="HS256",
            access_token_expire_minutes=60,
        )

        assert manager.secret_key == test_secret
        assert manager.algorithm == "HS256"
        assert manager.access_token_expire_minutes == 60
        assert get_auth_manager() is manager

    def test_get_auth_manager_not_initialized(self):
        """测试未初始化时获取全局管理器失败。"""
        import src.auth
        original = src.auth._auth_manager
        
        try:
            src.auth._auth_manager = None
            
            with pytest.raises(RuntimeError, match="认证管理器未初始化"):
                get_auth_manager()
        finally:
            src.auth._auth_manager = original
