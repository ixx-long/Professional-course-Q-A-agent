"""
配置热加载和动态更新模块。

提供：
- 配置文件监控（文件变化检测）
- 动态重新加载配置
- 配置变更通知
- 线程安全的配置访问
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import yaml

logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """配置变更事件。"""
    
    def __init__(self, old_config: Dict, new_config: Dict, changed_keys: List[str]):
        """
        初始化配置变更事件。
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            changed_keys: 变更的配置键列表
        """
        self.old_config = old_config
        self.new_config = new_config
        self.changed_keys = changed_keys
        self.timestamp = time.time()
    
    def __repr__(self):
        return f"ConfigChangeEvent(changed_keys={self.changed_keys}, timestamp={self.timestamp})"


class ConfigChangeHandler:
    """配置变更处理器接口。"""
    
    def on_config_change(self, event: ConfigChangeEvent):
        """
        配置变更回调。
        
        Args:
            event: 配置变更事件
        """
        raise NotImplementedError


class DynamicConfig:
    """
    动态配置管理器。
    
    支持：
    - 配置文件监控和自动重载
    - 线程安全的配置访问
    - 配置变更通知
    - 配置版本管理
    """
    
    def __init__(self, config_path: str):
        """
        初始化动态配置管理器。
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._version = 0
        self._handlers: List[ConfigChangeHandler] = []
        self._observer: Optional[Observer] = None
        self._last_modified = 0
        
        # 初始加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件。"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f) or {}
            
            with self._lock:
                old_config = self._config.copy()
                self._config = new_config
                self._version += 1
                self._last_modified = self.config_path.stat().st_mtime
            
            # 检测变更的键
            changed_keys = self._detect_changes(old_config, new_config)
            
            if changed_keys:
                logger.info(f"配置已更新，变更键：{changed_keys}")
                self._notify_handlers(old_config, new_config, changed_keys)
            
        except Exception as e:
            logger.error(f"加载配置文件失败：{e}", exc_info=True)
            raise
    
    def _detect_changes(self, old_config: Dict, new_config: Dict) -> List[str]:
        """
        检测配置变更。
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            
        Returns:
            变更的配置键列表
        """
        changed_keys = []
        
        # 检测所有可能的键
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                changed_keys.append(key)
        
        return changed_keys
    
    def _notify_handlers(self, old_config: Dict, new_config: Dict, changed_keys: List[str]):
        """
        通知配置变更处理器。
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            changed_keys: 变更的配置键列表
        """
        event = ConfigChangeEvent(old_config, new_config, changed_keys)
        
        for handler in self._handlers:
            try:
                handler.on_config_change(event)
            except Exception as e:
                logger.error(f"配置变更处理器失败：{e}", exc_info=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（线程安全）。
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        with self._lock:
            return self._config.get(key, default)
    
    def get_nested(self, *keys, default: Any = None) -> Any:
        """
        获取嵌套配置值。
        
        Args:
            *keys: 配置键路径
            default: 默认值
            
        Returns:
            配置值
            
        Example:
            config.get_nested("database", "host", default="localhost")
        """
        with self._lock:
            value = self._config
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                    if value is None:
                        return default
                else:
                    return default
            return value
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置（线程安全）。
        
        Returns:
            配置字典的副本
        """
        with self._lock:
            return self._config.copy()
    
    def reload(self):
        """手动重新加载配置。"""
        logger.info("手动触发配置重新加载")
        self._load_config()
    
    def register_handler(self, handler: ConfigChangeHandler):
        """
        注册配置变更处理器。
        
        Args:
            handler: 配置变更处理器
        """
        self._handlers.append(handler)
        logger.info(f"注册配置变更处理器：{handler.__class__.__name__}")
    
    def unregister_handler(self, handler: ConfigChangeHandler):
        """
        注销配置变更处理器。
        
        Args:
            handler: 配置变更处理器
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            logger.info(f"注销配置变更处理器：{handler.__class__.__name__}")
    
    @property
    def version(self) -> int:
        """获取配置版本号。"""
        with self._lock:
            return self._version
    
    def start_watching(self):
        """启动配置文件监控。"""
        if self._observer is not None:
            logger.warning("配置监控已在运行")
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager: DynamicConfig):
                self.config_manager = config_manager
            
            def on_modified(self, event: FileModifiedEvent):
                if event.src_path == str(self.config_manager.config_path):
                    logger.info(f"检测到配置文件变化：{event.src_path}")
                    # 延迟加载，避免文件写入未完成
                    time.sleep(0.1)
                    self.config_manager.reload()
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        self._observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self._observer.start()
        logger.info(f"开始监控配置文件：{self.config_path}")
    
    def stop_watching(self):
        """停止配置文件监控。"""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("停止配置文件监控")


class ConfigSection:
    """
    配置段访问器。
    
    提供对特定配置段的便捷访问。
    """
    
    def __init__(self, config: DynamicConfig, section: str):
        """
        初始化配置段访问器。
        
        Args:
            config: 动态配置管理器
            section: 配置段名称
        """
        self._config = config
        self._section = section
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置段中的值。
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get_nested(self._section, key, default=default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取整个配置段。
        
        Returns:
            配置段字典
        """
        return self._config.get(self._section, {})


# 全局配置管理器实例
_config_manager: Optional[DynamicConfig] = None


def get_config_manager() -> DynamicConfig:
    """获取全局配置管理器实例。"""
    global _config_manager
    if _config_manager is None:
        raise RuntimeError("配置管理器未初始化，请先调用 init_config_manager")
    return _config_manager


def init_config_manager(config_path: str) -> DynamicConfig:
    """
    初始化全局配置管理器。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    _config_manager = DynamicConfig(config_path)
    return _config_manager
