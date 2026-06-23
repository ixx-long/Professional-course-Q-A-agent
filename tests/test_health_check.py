"""
健康检查和生命周期管理模块单元测试。
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.health_check import (
    ServiceStatus,
    ComponentHealth,
    HealthCheckResult,
    HealthChecker,
    GracefulShutdown,
    ServiceLifecycle,
    init_lifecycle,
    get_lifecycle,
)


class TestServiceStatus:
    """ServiceStatus 单元测试类。"""

    def test_status_values(self):
        """测试状态枚举值。"""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.STARTING.value == "starting"
        assert ServiceStatus.STOPPING.value == "stopping"
        assert ServiceStatus.STOPPED.value == "stopped"


class TestComponentHealth:
    """ComponentHealth 单元测试类。"""

    def test_init(self):
        """测试初始化组件健康状态。"""
        health = ComponentHealth(
            name="test_component",
            status=ServiceStatus.HEALTHY,
            message="OK",
            details={"key": "value"},
        )
        
        assert health.name == "test_component"
        assert health.status == ServiceStatus.HEALTHY
        assert health.message == "OK"
        assert health.details == {"key": "value"}
        assert health.last_check > 0

    def test_init_defaults(self):
        """测试默认值。"""
        health = ComponentHealth(
            name="test",
            status=ServiceStatus.HEALTHY,
        )
        
        assert health.message == ""
        assert health.details == {}


class TestHealthCheckResult:
    """HealthCheckResult 单元测试类。"""

    def test_to_dict(self):
        """测试转换为字典。"""
        components = {
            "db": ComponentHealth(
                name="db",
                status=ServiceStatus.HEALTHY,
                message="Connected",
            ),
            "cache": ComponentHealth(
                name="cache",
                status=ServiceStatus.DEGRADED,
                message="Slow",
            ),
        }
        
        result = HealthCheckResult(
            status=ServiceStatus.DEGRADED,
            components=components,
            uptime_seconds=3600.0,
            version="1.0.0",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "degraded"
        assert "db" in result_dict["components"]
        assert "cache" in result_dict["components"]
        assert result_dict["components"]["db"]["status"] == "healthy"
        assert result_dict["components"]["cache"]["status"] == "degraded"
        assert result_dict["uptime_seconds"] == 3600.0
        assert result_dict["version"] == "1.0.0"


class TestHealthChecker:
    """HealthChecker 单元测试类。"""

    @pytest.fixture
    def checker(self):
        """创建测试用 HealthChecker 实例。"""
        return HealthChecker()

    def test_register_checker(self, checker):
        """测试注册检查器。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        checker.register_checker("db", db_check)
        
        assert "db" in checker._checkers

    def test_unregister_checker(self, checker):
        """测试注销检查器。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        checker.register_checker("db", db_check)
        checker.unregister_checker("db")
        
        assert "db" not in checker._checkers

    def test_unregister_nonexistent_checker(self, checker):
        """测试注销不存在的检查器。"""
        # 不应该抛出异常
        checker.unregister_checker("nonexistent")

    def test_check_component_healthy(self, checker):
        """测试检查健康组件。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "Connected", {"latency": 10})
        
        checker.register_checker("db", db_check)
        
        health = checker.check_component("db")
        
        assert health.status == ServiceStatus.HEALTHY
        assert health.message == "Connected"
        assert health.details == {"latency": 10}

    def test_check_component_unhealthy(self, checker):
        """测试检查不健康组件。"""
        def db_check():
            return (ServiceStatus.UNHEALTHY, "Connection failed", {})
        
        checker.register_checker("db", db_check)
        
        health = checker.check_component("db")
        
        assert health.status == ServiceStatus.UNHEALTHY
        assert health.message == "Connection failed"

    def test_check_component_exception(self, checker):
        """测试检查组件时异常。"""
        def db_check():
            raise Exception("Database error")
        
        checker.register_checker("db", db_check)
        
        health = checker.check_component("db")
        
        assert health.status == ServiceStatus.UNHEALTHY
        assert "Database error" in health.message

    def test_check_component_not_registered(self, checker):
        """测试检查未注册的组件。"""
        health = checker.check_component("nonexistent")
        
        assert health.status == ServiceStatus.UNHEALTHY
        assert "检查器未注册" in health.message

    def test_check_all_healthy(self, checker):
        """测试所有组件健康。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        def cache_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        checker.register_checker("db", db_check)
        checker.register_checker("cache", cache_check)
        
        result = checker.check_all()
        
        assert result.status == ServiceStatus.HEALTHY
        assert len(result.components) == 2
        assert result.uptime_seconds >= 0

    def test_check_all_degraded(self, checker):
        """测试组件降级。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        def cache_check():
            return (ServiceStatus.DEGRADED, "Slow", {})
        
        checker.register_checker("db", db_check)
        checker.register_checker("cache", cache_check)
        
        result = checker.check_all()
        
        assert result.status == ServiceStatus.DEGRADED

    def test_check_all_unhealthy(self, checker):
        """测试组件不健康。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        def cache_check():
            return (ServiceStatus.UNHEALTHY, "Down", {})
        
        checker.register_checker("db", db_check)
        checker.register_checker("cache", cache_check)
        
        result = checker.check_all()
        
        assert result.status == ServiceStatus.UNHEALTHY

    def test_get_last_result(self, checker):
        """测试获取最后检查结果。"""
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        checker.register_checker("db", db_check)
        
        # 检查前应该没有结果
        assert checker.get_last_result("db") is None
        
        # 检查后应该有结果
        checker.check_component("db")
        last_result = checker.get_last_result("db")
        
        assert last_result is not None
        assert last_result.status == ServiceStatus.HEALTHY

    def test_set_version(self, checker):
        """测试设置版本号。"""
        checker.set_version("2.0.0")
        
        assert checker._version == "2.0.0"


class TestGracefulShutdown:
    """GracefulShutdown 单元测试类。"""

    @pytest.fixture
    def shutdown(self):
        """创建测试用 GracefulShutdown 实例。"""
        return GracefulShutdown(timeout=5.0)

    def test_register_hook(self, shutdown):
        """测试注册关闭钩子。"""
        def cleanup():
            pass
        
        shutdown.register_hook(cleanup, priority=0)
        
        assert len(shutdown._hooks) == 1
        assert shutdown._hooks[0][1] == cleanup

    def test_register_hook_priority(self, shutdown):
        """测试钩子优先级排序。"""
        def cleanup1():
            pass
        
        def cleanup2():
            pass
        
        def cleanup3():
            pass
        
        shutdown.register_hook(cleanup2, priority=2)
        shutdown.register_hook(cleanup1, priority=1)
        shutdown.register_hook(cleanup3, priority=3)
        
        # 应该按优先级排序
        assert shutdown._hooks[0][1] == cleanup1
        assert shutdown._hooks[1][1] == cleanup2
        assert shutdown._hooks[2][1] == cleanup3

    def test_is_shutting_down(self, shutdown):
        """测试检查是否正在关闭。"""
        assert not shutdown.is_shutting_down()
        
        shutdown._shutting_down = True
        
        assert shutdown.is_shutting_down()

    def test_shutdown(self, shutdown):
        """测试执行关闭。"""
        call_order = []
        
        def cleanup1():
            call_order.append(1)
        
        def cleanup2():
            call_order.append(2)
        
        shutdown.register_hook(cleanup1, priority=1)
        shutdown.register_hook(cleanup2, priority=2)
        
        shutdown.shutdown()
        
        assert call_order == [1, 2]
        assert shutdown.is_shutting_down()

    def test_shutdown_timeout(self, shutdown):
        """测试关闭超时。"""
        shutdown._timeout = 0.1
        
        def slow_cleanup():
            time.sleep(0.2)
        
        def fast_cleanup():
            pass
        
        shutdown.register_hook(slow_cleanup, priority=1)
        shutdown.register_hook(fast_cleanup, priority=2)
        
        # 应该超时并跳过第二个钩子
        shutdown.shutdown()
        
        assert shutdown.is_shutting_down()

    def test_shutdown_exception_handling(self, shutdown):
        """测试关闭时异常处理。"""
        def failing_cleanup():
            raise Exception("Cleanup error")
        
        def successful_cleanup():
            pass
        
        shutdown.register_hook(failing_cleanup, priority=1)
        shutdown.register_hook(successful_cleanup, priority=2)
        
        # 不应该抛出异常，应该继续执行下一个钩子
        shutdown.shutdown()
        
        assert shutdown.is_shutting_down()

    def test_shutdown_idempotent(self, shutdown):
        """测试关闭幂等性。"""
        call_count = 0
        
        def cleanup():
            nonlocal call_count
            call_count += 1
        
        shutdown.register_hook(cleanup)
        
        shutdown.shutdown()
        shutdown.shutdown()  # 第二次调用
        
        # 钩子应该只被调用一次
        assert call_count == 1

    def test_wait_for_shutdown(self, shutdown):
        """测试等待关闭。"""
        import threading
        
        def trigger_shutdown():
            time.sleep(0.1)
            shutdown.shutdown()
        
        thread = threading.Thread(target=trigger_shutdown)
        thread.start()
        
        result = shutdown.wait_for_shutdown(timeout=1.0)
        
        assert result is True
        thread.join()

    def test_wait_for_shutdown_timeout(self, shutdown):
        """测试等待关闭超时。"""
        result = shutdown.wait_for_shutdown(timeout=0.1)
        
        assert result is False

    def test_setup_signal_handlers(self, shutdown):
        """测试设置信号处理器。"""
        with patch('signal.signal') as mock_signal:
            shutdown.setup_signal_handlers()
            
            # 应该注册了 SIGTERM 和 SIGINT
            assert mock_signal.call_count == 2


class TestServiceLifecycle:
    """ServiceLifecycle 单元测试类。"""

    @pytest.fixture
    def lifecycle(self):
        """创建测试用 ServiceLifecycle 实例。"""
        return ServiceLifecycle(service_name="test_service")

    def test_init(self, lifecycle):
        """测试初始化。"""
        assert lifecycle.service_name == "test_service"
        assert lifecycle._status == ServiceStatus.STARTING
        assert isinstance(lifecycle.health_checker, HealthChecker)
        assert isinstance(lifecycle.shutdown_manager, GracefulShutdown)

    def test_start(self, lifecycle):
        """测试启动服务。"""
        lifecycle.start()
        
        assert lifecycle._status == ServiceStatus.HEALTHY

    def test_stop(self, lifecycle):
        """测试停止服务。"""
        lifecycle.start()
        lifecycle.stop()
        
        assert lifecycle._status == ServiceStatus.STOPPED

    def test_stop_already_stopped(self, lifecycle):
        """测试重复停止。"""
        lifecycle.start()
        lifecycle.stop()
        lifecycle.stop()  # 第二次调用
        
        assert lifecycle._status == ServiceStatus.STOPPED

    def test_get_status(self, lifecycle):
        """测试获取状态。"""
        assert lifecycle.get_status() == ServiceStatus.STARTING
        
        lifecycle.start()
        assert lifecycle.get_status() == ServiceStatus.HEALTHY
        
        lifecycle.stop()
        assert lifecycle.get_status() == ServiceStatus.STOPPED

    def test_health_check(self, lifecycle):
        """测试健康检查。"""
        lifecycle.start()
        
        # 注册一个检查器
        def db_check():
            return (ServiceStatus.HEALTHY, "OK", {})
        
        lifecycle.health_checker.register_checker("db", db_check)
        
        result = lifecycle.health_check()
        
        assert result.status == ServiceStatus.HEALTHY
        assert "db" in result.components
        assert result.metadata["service_name"] == "test_service"
        assert result.metadata["service_status"] == "healthy"


class TestGlobalLifecycle:
    """全局生命周期管理器测试。"""

    def test_init_lifecycle(self):
        """测试初始化全局生命周期管理器。"""
        lifecycle = init_lifecycle("test_service")
        
        assert isinstance(lifecycle, ServiceLifecycle)
        assert lifecycle.service_name == "test_service"
        assert get_lifecycle() is lifecycle

    def test_get_lifecycle_auto_init(self):
        """测试自动初始化全局生命周期管理器。"""
        import src.health_check
        original = src.health_check._lifecycle
        
        try:
            src.health_check._lifecycle = None
            
            lifecycle = get_lifecycle()
            
            assert lifecycle is not None
            assert isinstance(lifecycle, ServiceLifecycle)
        finally:
            src.health_check._lifecycle = original
