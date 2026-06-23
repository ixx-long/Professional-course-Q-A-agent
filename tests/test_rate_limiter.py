"""
请求限流和防刷模块单元测试。
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.rate_limiter import (
    RateLimiter,
    MultiTierRateLimiter,
    AbuseDetector,
    get_rate_limiter,
    configure_rate_limiter,
    get_abuse_detector,
    init_abuse_detector,
)
from src.errors import RateLimitError


class TestRateLimiter:
    """RateLimiter 单元测试类。"""

    @pytest.fixture
    def limiter(self):
        """创建测试用限流器实例。"""
        return RateLimiter(
            max_requests=5,
            window_seconds=60,
            cleanup_interval=300,
        )

    def test_is_allowed_within_limit(self, limiter):
        """测试在限制内允许请求。"""
        allowed, stats = limiter.is_allowed("user1")
        
        assert allowed is True
        assert stats["current_count"] == 1
        assert stats["max_requests"] == 5
        assert stats["remaining"] == 4

    def test_is_allowed_exceed_limit(self, limiter):
        """测试超过限制拒绝请求。"""
        # 发送 5 个请求（达到限制）
        for i in range(5):
            limiter.is_allowed("user1")
        
        # 第 6 个请求应该被拒绝
        allowed, stats = limiter.is_allowed("user1")
        
        assert allowed is False
        assert stats["current_count"] == 5
        assert stats["remaining"] == 0

    def test_is_allowed_different_keys(self, limiter):
        """测试不同键独立计数。"""
        # user1 发送 3 个请求
        for i in range(3):
            limiter.is_allowed("user1")
        
        # user2 发送 2 个请求
        for i in range(2):
            limiter.is_allowed("user2")
        
        # 两个用户都应该被允许
        allowed1, stats1 = limiter.is_allowed("user1")
        allowed2, stats2 = limiter.is_allowed("user2")
        
        assert allowed1 is True
        assert allowed2 is True
        assert stats1["current_count"] == 4
        assert stats2["current_count"] == 3

    def test_get_stats(self, limiter):
        """测试获取统计信息。"""
        # 发送 3 个请求
        for i in range(3):
            limiter.is_allowed("user1")
        
        stats = limiter.get_stats("user1")
        
        assert stats["current_count"] == 3
        assert stats["max_requests"] == 5
        assert stats["remaining"] == 2
        assert stats["window_seconds"] == 60

    def test_get_stats_no_requests(self, limiter):
        """测试无请求时的统计信息。"""
        stats = limiter.get_stats("nonexistent_user")
        
        assert stats["current_count"] == 0
        assert stats["remaining"] == 5

    def test_reset(self, limiter):
        """测试重置计数。"""
        # 发送 3 个请求
        for i in range(3):
            limiter.is_allowed("user1")
        
        # 重置
        limiter.reset("user1")
        
        # 应该回到 0
        stats = limiter.get_stats("user1")
        assert stats["current_count"] == 0

    def test_window_expiration(self, limiter):
        """测试时间窗口过期。"""
        # 发送 5 个请求（达到限制）
        for i in range(5):
            limiter.is_allowed("user1")
        
        # 模拟时间流逝（窗口外）
        current_time = time.time()
        for key in limiter._requests:
            limiter._requests[key] = [current_time - 61]  # 61 秒前
        
        # 应该允许新请求
        allowed, stats = limiter.is_allowed("user1")
        assert allowed is True
        assert stats["current_count"] == 1

    def test_cleanup_expired(self, limiter):
        """测试清理过期记录。"""
        # 添加一些过期记录
        current_time = time.time()
        limiter._requests["user1"] = [current_time - 120]  # 2 分钟前
        limiter._requests["user2"] = [current_time - 30]   # 30 秒前
        
        # 强制触发清理
        limiter._last_cleanup = current_time - 400  # 超过 cleanup_interval
        
        limiter._cleanup_expired()
        
        # user1 应该被清理，user2 应该保留
        assert "user1" not in limiter._requests or len(limiter._requests["user1"]) == 0
        assert "user2" in limiter._requests


class TestMultiTierRateLimiter:
    """MultiTierRateLimiter 单元测试类。"""

    @pytest.fixture
    def multi_limiter(self):
        """创建多层限流器。"""
        limiter = MultiTierRateLimiter()
        limiter.add_limiter(
            RateLimiter(max_requests=5, window_seconds=60),
            name="per_minute"
        )
        limiter.add_limiter(
            RateLimiter(max_requests=10, window_seconds=3600),
            name="per_hour"
        )
        return limiter

    def test_is_allowed_all_tiers_pass(self, multi_limiter):
        """测试所有层都允许。"""
        allowed, rejected_info = multi_limiter.is_allowed("user1")
        
        assert allowed is True
        assert rejected_info is None

    def test_is_allowed_first_tier_reject(self, multi_limiter):
        """测试第一层拒绝。"""
        # 达到每分钟限制
        for i in range(5):
            multi_limiter.is_allowed("user1")
        
        # 第 6 个请求应该被第一层拒绝
        allowed, rejected_info = multi_limiter.is_allowed("user1")
        
        assert allowed is False
        assert rejected_info is not None
        assert rejected_info["limiter_name"] == "per_minute"

    def test_get_all_stats(self, multi_limiter):
        """测试获取所有层的统计信息。"""
        # 发送 3 个请求
        for i in range(3):
            multi_limiter.is_allowed("user1")
        
        stats = multi_limiter.get_all_stats("user1")
        
        assert "per_minute" in stats
        assert "per_hour" in stats
        assert stats["per_minute"]["current_count"] == 3
        assert stats["per_hour"]["current_count"] == 3


class TestAbuseDetector:
    """AbuseDetector 单元测试类。"""

    @pytest.fixture
    def detector(self):
        """创建滥用检测器。"""
        return AbuseDetector(
            failure_threshold=3,
            failure_window=300,
            block_duration=3600,
        )

    def test_record_failure(self, detector):
        """测试记录失败。"""
        detector.record_failure("user1")
        
        count = detector.get_failure_count("user1")
        assert count == 1

    def test_is_blocked_after_threshold(self, detector):
        """测试达到阈值后被封禁。"""
        # 记录 3 次失败（达到阈值）
        for i in range(3):
            detector.record_failure("user1")
        
        # 应该被封禁
        assert detector.is_blocked("user1")

    def test_is_not_blocked_below_threshold(self, detector):
        """测试未达到阈值不被封禁。"""
        # 记录 2 次失败（未达到阈值）
        for i in range(2):
            detector.record_failure("user1")
        
        # 不应该被封禁
        assert not detector.is_blocked("user1")

    def test_unblock(self, detector):
        """测试手动解封。"""
        # 封禁用户
        for i in range(3):
            detector.record_failure("user1")
        
        assert detector.is_blocked("user1")
        
        # 手动解封
        detector.unblock("user1")
        
        # 应该不再被封禁
        assert not detector.is_blocked("user1")
        assert detector.get_failure_count("user1") == 0

    def test_block_expiration(self, detector):
        """测试封禁过期。"""
        # 封禁用户
        for i in range(3):
            detector.record_failure("user1")
        
        # 模拟封禁过期
        detector._blocked["user1"] = time.time() - 3601  # 1 小时前
        
        # 应该不再被封禁
        assert not detector.is_blocked("user1")

    def test_get_failure_count_window(self, detector):
        """测试失败计数在时间窗口内。"""
        # 记录一些失败
        detector.record_failure("user1")
        detector.record_failure("user1")
        
        # 模拟一条记录过期
        current_time = time.time()
        detector._failures["user1"] = [current_time - 301, current_time - 10]
        
        # 应该只计算窗口内的
        count = detector.get_failure_count("user1")
        assert count == 1


class TestGlobalRateLimiter:
    """全局流器测试。"""

    def test_get_rate_limiter(self):
        """测试获取限流器。"""
        limiter = get_rate_limiter("test")
        assert isinstance(limiter, MultiTierRateLimiter)

    def test_configure_rate_limiter(self):
        """测试配置限流器。"""
        configure_rate_limiter(
            name="test_config",
            per_minute=60,
            per_hour=1000,
            per_day=10000,
        )
        
        limiter = get_rate_limiter("test_config")
        stats = limiter.get_all_stats("test_user")
        assert "per_minute" in stats
        assert "per_hour" in stats
        assert "per_day" in stats


class TestGlobalAbuseDetector:
    """全局滥用检测器测试。"""

    def test_get_abuse_detector(self):
        """测试获取滥用检测器。"""
        detector = get_abuse_detector()
        assert isinstance(detector, AbuseDetector)

    def test_init_abuse_detector(self):
        """测试初始化滥用检测器。"""
        detector = init_abuse_detector(
            failure_threshold=5,
            failure_window=600,
            block_duration=7200,
        )
        
        assert detector.failure_threshold == 5
        assert detector.failure_window == 600
        assert detector.block_duration == 7200
