"""
结构化日志模块。

提供 JSON 格式的日志输出，便于日志收集和分析。
支持：
- JSON 格式化输出
- 请求上下文追踪
- 日志级别动态控制
- 敏感信息脱敏
"""

import json
import logging
import sys
import time
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    JSON 格式化器。
    
    将日志记录转换为 JSON 格式，包含：
    - 时间戳
    - 日志级别
    - 模块名
    - 函数名
    - 行号
    - 消息内容
    - 额外字段
    """
    
    def __init__(
        self,
        include_context: bool = True,
        mask_fields: list = None,
    ):
        """
        初始化 JSON 格式化器。
        
        Args:
            include_context: 是否包含上下文信息
            mask_fields: 需要脱敏的字段列表
        """
        super().__init__()
        self.include_context = include_context
        self.mask_fields = mask_fields or ["password", "api_key", "token", "secret"]
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对敏感数据进行脱敏处理。
        
        Args:
            data: 原始数据字典
            
        Returns:
            脱敏后的数据字典
        """
        masked = {}
        for key, value in data.items():
            if any(field in key.lower() for field in self.mask_fields):
                masked[key] = "***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive_data(value)
            else:
                masked[key] = value
        return masked
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录。
        
        Args:
            record: 日志记录
            
        Returns:
            JSON 格式的日志字符串
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        # 添加上下文信息
        if self.include_context:
            context = {}
            
            # 请求 ID
            if hasattr(record, "request_id"):
                context["request_id"] = record.request_id
            
            # 用户 ID
            if hasattr(record, "user_id"):
                context["user_id"] = record.user_id
            
            # 会话 ID
            if hasattr(record, "session_id"):
                context["session_id"] = record.session_id
            
            # IP 地址
            if hasattr(record, "ip_address"):
                context["ip_address"] = record.ip_address
            
            # 用户代理
            if hasattr(record, "user_agent"):
                context["user_agent"] = record.user_agent
            
            if context:
                log_data["context"] = context
        
        # 添加额外字段
        if hasattr(record, "extra_data"):
            extra_data = self._mask_sensitive_data(record.extra_data)
            log_data["extra"] = extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """
    结构化日志记录器。
    
    提供便捷的日志方法，自动添加上下文信息。
    """
    
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        初始化结构化日志记录器。
        
        Args:
            name: 日志记录器名称
            logger: 底层 logging.Logger 实例
            request_id: 请求 ID
            user_id: 用户 ID
            session_id: 会话 ID
        """
        self.name = name
        self.logger = logger
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加上下文信息到额外字段。
        
        Args:
            extra: 额外字段字典
            
        Returns:
            包含上下文的额外字段字典
        """
        context_extra = extra.copy()
        
        if self.request_id:
            context_extra["request_id"] = self.request_id
        if self.user_id:
            context_extra["user_id"] = self.user_id
        if self.session_id:
            context_extra["session_id"] = self.session_id
        
        return context_extra
    
    def debug(self, message: str, **kwargs):
        """记录 DEBUG 级别日志。"""
        extra = self._add_context(kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """记录 INFO 级别日志。"""
        extra = self._add_context(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """记录 WARNING 级别日志。"""
        extra = self._add_context(kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """记录 ERROR 级别日志。"""
        extra = self._add_context(kwargs)
        self.logger.error(message, exc_info=exc_info, extra=extra)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """记录 CRITICAL 级别日志。"""
        extra = self._add_context(kwargs)
        self.logger.critical(message, exc_info=exc_info, extra=extra)
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """
        绑定上下文信息，返回新的日志记录器。
        
        Args:
            **kwargs: 上下文键值对
            
        Returns:
            新的结构化日志记录器
        """
        return StructuredLogger(
            name=self.name,
            logger=self.logger,
            request_id=kwargs.get("request_id", self.request_id),
            user_id=kwargs.get("user_id", self.user_id),
            session_id=kwargs.get("session_id", self.session_id),
        )


def setup_structured_logger(
    name: str = "course_qa",
    log_file: str = "logs/qa.log",
    level: str = "INFO",
    json_output: bool = True,
    include_context: bool = True,
    mask_fields: list = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> StructuredLogger:
    """
    初始化结构化日志记录器。
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        json_output: 是否使用 JSON 格式输出
        include_context: 是否包含上下文信息
        mask_fields: 需要脱敏的字段列表
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的旧日志文件数量
        
    Returns:
        结构化日志记录器
    """
    logger = logging.getLogger(name)
    
    # 清理旧处理器
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 创建格式化器
    if json_output:
        formatter = JSONFormatter(
            include_context=include_context,
            mask_fields=mask_fields,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return StructuredLogger(name=name, logger=logger)


class RequestContext:
    """
    请求上下文管理器。
    
    用于在请求生命周期内追踪上下文信息。
    """
    
    def __init__(self):
        """初始化请求上下文。"""
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.ip_address: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.start_time: Optional[float] = None
    
    def start_request(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """
        开始新请求。
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            ip_address: IP 地址
            user_agent: 用户代理
        """
        self.request_id = str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.start_time = time.time()
    
    def end_request(self) -> Dict[str, Any]:
        """
        结束请求，返回请求摘要。
        
        Returns:
            请求摘要字典
        """
        duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "duration": duration,
        }
    
    def get_logger(self, name: str = "course_qa") -> StructuredLogger:
        """
        获取绑定当前上下文的日志记录器。
        
        Args:
            name: 日志记录器名称
            
        Returns:
            结构化日志记录器
        """
        logger = logging.getLogger(name)
        return StructuredLogger(
            name=name,
            logger=logger,
            request_id=self.request_id,
            user_id=self.user_id,
            session_id=self.session_id,
        )


# 线程安全的请求上下文存储
_thread_local = threading.local()


def get_request_context() -> RequestContext:
    """获取当前线程的请求上下文。"""
    if not hasattr(_thread_local, "request_context") or _thread_local.request_context is None:
        _thread_local.request_context = RequestContext()
    return _thread_local.request_context


def init_request_context() -> RequestContext:
    """初始化当前线程的请求上下文。"""
    _thread_local.request_context = RequestContext()
    return _thread_local.request_context


class LogLevelManager:
    """
    日志级别动态管理器。
    
    支持在运行时动态调整日志级别,无需重启应用。
    线程安全,支持按模块设置不同级别。
    """
    
    def __init__(self) -> None:
        """初始化日志级别管理器。"""
        self._lock = threading.RLock()
        self._global_level: int = logging.INFO
        self._module_levels: Dict[str, int] = {}
        self._loggers: Dict[str, logging.Logger] = {}
    
    def set_global_level(self, level: str | int) -> None:
        """
        设置全局日志级别。
        
        Args:
            level: 日志级别(字符串如'INFO'或整数如logging.INFO)
        """
        with self._lock:
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)
            self._global_level = level
            
            # 更新所有已注册的logger
            for logger in self._loggers.values():
                logger.setLevel(level)
    
    def set_module_level(self, module: str, level: str | int) -> None:
        """
        设置特定模块的日志级别。
        
        Args:
            module: 模块名称(如'src.qa_service')
            level: 日志级别
        """
        with self._lock:
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)
            self._module_levels[module] = level
            
            # 更新对应的logger
            if module in self._loggers:
                self._loggers[module].setLevel(level)
    
    def get_level(self, module: Optional[str] = None) -> int:
        """
        获取日志级别。
        
        Args:
            module: 模块名称,为None时返回全局级别
            
        Returns:
            日志级别整数
        """
        with self._lock:
            if module and module in self._module_levels:
                return self._module_levels[module]
            return self._global_level
    
    def register_logger(self, name: str, logger: logging.Logger) -> None:
        """
        注册logger以便后续动态调整。
        
        Args:
            name: logger名称
            logger: logger实例
        """
        with self._lock:
            self._loggers[name] = logger
            
            # 应用当前级别设置
            if name in self._module_levels:
                logger.setLevel(self._module_levels[name])
            else:
                logger.setLevel(self._global_level)
    
    def reset_module_level(self, module: str) -> None:
        """
        重置模块日志级别为全局级别。
        
        Args:
            module: 模块名称
        """
        with self._lock:
            if module in self._module_levels:
                del self._module_levels[module]
                if module in self._loggers:
                    self._loggers[module].setLevel(self._global_level)
    
    def get_all_levels(self) -> Dict[str, int]:
        """
        获取所有模块的日志级别配置。
        
        Returns:
            模块名称到日志级别的映射字典
        """
        with self._lock:
            result = {"__global__": self._global_level}
            result.update(self._module_levels)
            return result


# 全局日志级别管理器实例
_log_level_manager: Optional[LogLevelManager] = None
_manager_lock = threading.Lock()


def get_log_level_manager() -> LogLevelManager:
    """获取全局日志级别管理器实例。"""
    global _log_level_manager
    with _manager_lock:
        if _log_level_manager is None:
            _log_level_manager = LogLevelManager()
        return _log_level_manager


def set_log_level(level: str | int, module: Optional[str] = None) -> None:
    """
    便捷函数:动态设置日志级别。
    
    Args:
        level: 日志级别(如'DEBUG','INFO','WARNING')
        module: 模块名称,为None时设置全局级别
    """
    manager = get_log_level_manager()
    if module:
        manager.set_module_level(module, level)
    else:
        manager.set_global_level(level)


def get_log_level(module: Optional[str] = None) -> int:
    """
    便捷函数:获取当前日志级别。
    
    Args:
        module: 模块名称,为None时返回全局级别
        
    Returns:
        日志级别整数
    """
    manager = get_log_level_manager()
    return manager.get_level(module)
