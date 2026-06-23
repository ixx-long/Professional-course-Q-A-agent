"""
健康检查和生命周期管理模块。

提供：
- 增强的健康检查端点
- 优雅关闭机制
- 服务生命周期管理
- 依赖服务检查
"""

import logging
import signal
import sys
import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举。"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ComponentHealth:
    """组件健康状态。"""
    name: str
    status: ServiceStatus
    message: str = ""
    details: Dict = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)


@dataclass
class HealthCheckResult:
    """健康检查结果。"""
    status: ServiceStatus
    components: Dict[str, ComponentHealth]
    uptime_seconds: float
    version: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            "status": self.status.value,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "details": comp.details,
                    "last_check": comp.last_check,
                }
                for name, comp in self.components.items()
            },
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "metadata": self.metadata,
        }


class HealthChecker:
    """
    健康检查管理器。
    
    支持：
    - 注册多个组件检查器
    - 定期健康检查
    - 状态聚合
    - 自定义检查逻辑
    """
    
    def __init__(self):
        """初始化健康检查器。"""
        self._checkers: Dict[str, Callable] = {}
        self._last_results: Dict[str, ComponentHealth] = {}
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._version = "1.0.0"
    
    def register_checker(self, name: str, checker: Callable):
        """
        注册组件检查器。
        
        Args:
            name: 组件名称
            checker: 检查函数，返回 (status, message, details)
        """
        self._checkers[name] = checker
        logger.info(f"注册健康检查器：{name}")
    
    def unregister_checker(self, name: str):
        """
        注销组件检查器。
        
        Args:
            name: 组件名称
        """
        if name in self._checkers:
            del self._checkers[name]
            logger.info(f"注销健康检查器：{name}")
    
    def check_component(self, name: str) -> ComponentHealth:
        """
        检查单个组件的健康状态。
        
        Args:
            name: 组件名称
            
        Returns:
            组件健康状态
        """
        if name not in self._checkers:
            return ComponentHealth(
                name=name,
                status=ServiceStatus.UNHEALTHY,
                message="检查器未注册",
            )
        
        try:
            checker = self._checkers[name]
            result = checker()
            
            if isinstance(result, tuple):
                status, message, details = result
            else:
                status = ServiceStatus.HEALTHY if result else ServiceStatus.UNHEALTHY
                message = ""
                details = {}
            
            health = ComponentHealth(
                name=name,
                status=status if isinstance(status, ServiceStatus) else ServiceStatus.HEALTHY,
                message=message,
                details=details,
                last_check=time.time(),
            )
            
        except Exception as e:
            logger.error(f"健康检查失败 [{name}]: {e}", exc_info=True)
            health = ComponentHealth(
                name=name,
                status=ServiceStatus.UNHEALTHY,
                message=f"检查异常：{str(e)}",
                last_check=time.time(),
            )
        
        with self._lock:
            self._last_results[name] = health
        
        return health
    
    def check_all(self) -> HealthCheckResult:
        """
        检查所有组件的健康状态。
        
        Returns:
            综合健康检查结果
        """
        components = {}
        
        for name in self._checkers:
            components[name] = self.check_component(name)
        
        # 聚合状态
        statuses = [comp.status for comp in components.values()]
        
        if ServiceStatus.UNHEALTHY in statuses:
            overall_status = ServiceStatus.UNHEALTHY
        elif ServiceStatus.DEGRADED in statuses:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.HEALTHY
        
        uptime = time.time() - self._start_time
        
        return HealthCheckResult(
            status=overall_status,
            components=components,
            uptime_seconds=uptime,
            version=self._version,
        )
    
    def get_last_result(self, name: str) -> Optional[ComponentHealth]:
        """
        获取组件的最后一次检查结果。
        
        Args:
            name: 组件名称
            
        Returns:
            最后的健康状态
        """
        with self._lock:
            return self._last_results.get(name)
    
    def set_version(self, version: str):
        """
        设置服务版本号。
        
        Args:
            version: 版本号字符串
        """
        self._version = version


class GracefulShutdown:
    """
    优雅关闭管理器。
    
    支持：
    - 注册关闭钩子
    - 信号处理
    - 超时控制
    - 依赖顺序关闭
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        初始化优雅关闭管理器。
        
        Args:
            timeout: 关闭超时时间（秒）
        """
        self._hooks: List[Callable] = []
        self._timeout = timeout
        self._shutting_down = False
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
    
    def register_hook(self, hook: Callable, priority: int = 0):
        """
        注册关闭钩子。
        
        Args:
            hook: 关闭函数
            priority: 优先级（数字越小优先级越高）
        """
        self._hooks.append((priority, hook))
        self._hooks.sort(key=lambda x: x[0])
        logger.info(f"注册关闭钩子：{hook.__name__}，优先级：{priority}")
    
    def is_shutting_down(self) -> bool:
        """
        检查是否正在关闭。
        
        Returns:
            是否正在关闭
        """
        return self._shutting_down
    
    def wait_for_shutdown(self, timeout: Optional[float] = None):
        """
        等待关闭信号。
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            是否收到关闭信号
        """
        return self._shutdown_event.wait(timeout=timeout)
    
    def shutdown(self):
        """
        执行优雅关闭。
        
        按优先级顺序调用所有注册的钩子，并处理超时。
        """
        with self._lock:
            if self._shutting_down:
                logger.warning("关闭已在进行中")
                return
            self._shutting_down = True
        
        logger.info("开始优雅关闭...")
        start_time = time.time()
        
        # 调用所有钩子
        for priority, hook in self._hooks:
            elapsed = time.time() - start_time
            if elapsed >= self._timeout:
                logger.warning(f"关闭超时（{self._timeout}s），跳过剩余钩子")
                break
            
            try:
                logger.info(f"执行关闭钩子：{hook.__name__}")
                hook()
            except Exception as e:
                logger.error(f"关闭钩子失败 [{hook.__name__}]: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        logger.info(f"优雅关闭完成，耗时：{elapsed:.2f}s")
        
        # 触发关闭事件
        self._shutdown_event.set()
    
    def setup_signal_handlers(self):
        """
        设置信号处理器。
        
        监听 SIGTERM 和 SIGINT 信号，触发优雅关闭。
        """
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"收到信号：{sig_name}")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("信号处理器已设置（SIGTERM, SIGINT）")


class ServiceLifecycle:
    """
    服务生命周期管理器。
    
    整合健康检查和优雅关闭，提供完整的服务生命周期管理。
    """
    
    def __init__(self, service_name: str = "course_qa_service"):
        """
        初始化服务生命周期管理器。
        
        Args:
            service_name: 服务名称
        """
        self.service_name = service_name
        self.health_checker = HealthChecker()
        self.shutdown_manager = GracefulShutdown()
        self._status = ServiceStatus.STARTING
        self._lock = threading.Lock()
    
    def start(self):
        """启动服务。"""
        with self._lock:
            self._status = ServiceStatus.STARTING
        
        logger.info(f"服务启动：{self.service_name}")
        
        # 注意：信号处理器由调用方（如 web_server.py）统一管理
        # 避免多处设置信号导致冲突
        
        with self._lock:
            self._status = ServiceStatus.HEALTHY
        
        logger.info(f"服务就绪：{self.service_name}")
    
    def stop(self):
        """停止服务。"""
        with self._lock:
            if self._status == ServiceStatus.STOPPED:
                logger.warning("服务已停止")
                return
            self._status = ServiceStatus.STOPPING
        
        logger.info(f"服务停止：{self.service_name}")
        
        # 执行优雅关闭
        self.shutdown_manager.shutdown()
        
        with self._lock:
            self._status = ServiceStatus.STOPPED
        
        logger.info(f"服务已停止：{self.service_name}")
    
    def get_status(self) -> ServiceStatus:
        """
        获取服务状态。
        
        Returns:
            当前服务状态
        """
        with self._lock:
            return self._status
    
    def health_check(self) -> HealthCheckResult:
        """
        执行健康检查。
        
        Returns:
            健康检查结果
        """
        result = self.health_checker.check_all()
        result.metadata["service_name"] = self.service_name
        result.metadata["service_status"] = self._status.value
        return result


# 全局生命周期管理器实例
_lifecycle: Optional[ServiceLifecycle] = None


def get_lifecycle() -> ServiceLifecycle:
    """获取全局生命周期管理器实例。"""
    global _lifecycle
    if _lifecycle is None:
        _lifecycle = ServiceLifecycle()
    return _lifecycle


def init_lifecycle(service_name: str = "course_qa_service") -> ServiceLifecycle:
    """初始化全局生命周期管理器。"""
    global _lifecycle
    _lifecycle = ServiceLifecycle(service_name)
    return _lifecycle
