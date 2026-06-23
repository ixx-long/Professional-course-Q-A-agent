"""
配置热加载和动态更新模块单元测试。
"""
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from src.config_manager import (
    DynamicConfig,
    ConfigChangeEvent,
    ConfigChangeHandler,
    ConfigSection,
    init_config_manager,
    get_config_manager,
)


class TestConfigChangeEvent:
    """ConfigChangeEvent 单元测试类。"""

    def test_init(self):
        """测试初始化配置变更事件。"""
        old_config = {"key1": "value1"}
        new_config = {"key1": "value2"}
        changed_keys = ["key1"]
        
        event = ConfigChangeEvent(old_config, new_config, changed_keys)
        
        assert event.old_config == old_config
        assert event.new_config == new_config
        assert event.changed_keys == changed_keys
        assert event.timestamp > 0

    def test_repr(self):
        """测试字符串表示。"""
        event = ConfigChangeEvent({}, {}, ["key1", "key2"])
        
        repr_str = repr(event)
        assert "ConfigChangeEvent" in repr_str
        assert "key1" in repr_str
        assert "key2" in repr_str


class TestConfigChangeHandler:
    """ConfigChangeHandler 单元测试类。"""

    def test_on_config_change_not_implemented(self):
        """测试 on_config_change 方法未实现。"""
        handler = ConfigChangeHandler()
        
        with pytest.raises(NotImplementedError):
            handler.on_config_change(Mock())


class TestDynamicConfig:
    """DynamicConfig 单元测试类。"""

    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key1: value1\nkey2: value2\n")
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        if Path(temp_path).exists():
            Path(temp_path).unlink()

    @pytest.fixture
    def dynamic_config(self, temp_config_file):
        """创建测试用 DynamicConfig 实例。"""
        return DynamicConfig(temp_config_file)

    def test_init_loads_config(self, dynamic_config):
        """测试初始化时加载配置。"""
        assert dynamic_config.get("key1") == "value1"
        assert dynamic_config.get("key2") == "value2"

    def test_get_default(self, dynamic_config):
        """测试获取不存在的键返回默认值。"""
        assert dynamic_config.get("nonexistent", "default") == "default"

    def test_get_nested(self, temp_config_file):
        """测试获取嵌套配置。"""
        # 创建包含嵌套结构的配置
        with open(temp_config_file, 'w') as f:
            f.write("database:\n  host: localhost\n  port: 5432\n")
        
        config = DynamicConfig(temp_config_file)
        
        assert config.get_nested("database", "host") == "localhost"
        assert config.get_nested("database", "port") == 5432
        assert config.get_nested("database", "nonexistent", default="default") == "default"

    def test_get_nested_deep_path(self, temp_config_file):
        """测试获取深层嵌套配置。"""
        with open(temp_config_file, 'w') as f:
            f.write("level1:\n  level2:\n    level3: value\n")
        
        config = DynamicConfig(temp_config_file)
        
        assert config.get_nested("level1", "level2", "level3") == "value"

    def test_get_nested_invalid_path(self, dynamic_config):
        """测试获取无效嵌套路径。"""
        assert dynamic_config.get_nested("nonexistent", "path") is None
        assert dynamic_config.get_nested("key1", "nested") is None

    def test_get_all(self, dynamic_config):
        """测试获取所有配置。"""
        all_config = dynamic_config.get_all()
        
        assert isinstance(all_config, dict)
        assert "key1" in all_config
        assert "key2" in all_config

    def test_reload(self, dynamic_config, temp_config_file):
        """测试手动重新加载配置。"""
        # 修改配置文件
        with open(temp_config_file, 'w') as f:
            f.write("key1: new_value1\nkey3: value3\n")
        
        dynamic_config.reload()
        
        assert dynamic_config.get("key1") == "new_value1"
        assert dynamic_config.get("key3") == "value3"

    def test_reload_detects_changes(self, dynamic_config, temp_config_file):
        """测试重新加载检测变更。"""
        # 注册处理器
        handler = Mock(spec=ConfigChangeHandler)
        dynamic_config.register_handler(handler)
        
        # 修改配置
        with open(temp_config_file, 'w') as f:
            f.write("key1: changed_value\n")
        
        dynamic_config.reload()
        
        # 验证处理器被调用
        assert handler.on_config_change.called
        event = handler.on_config_change.call_args[0][0]
        assert "key1" in event.changed_keys

    def test_register_handler(self, dynamic_config):
        """测试注册配置变更处理器。"""
        handler = Mock(spec=ConfigChangeHandler)
        
        dynamic_config.register_handler(handler)
        
        assert handler in dynamic_config._handlers

    def test_unregister_handler(self, dynamic_config):
        """测试注销配置变更处理器。"""
        handler = Mock(spec=ConfigChangeHandler)
        dynamic_config.register_handler(handler)
        
        dynamic_config.unregister_handler(handler)
        
        assert handler not in dynamic_config._handlers

    def test_unregister_nonexistent_handler(self, dynamic_config):
        """测试注销不存在的处理器。"""
        handler = Mock(spec=ConfigChangeHandler)
        
        # 不应该抛出异常
        dynamic_config.unregister_handler(handler)

    def test_version_increments(self, dynamic_config, temp_config_file):
        """测试版本号递增。"""
        initial_version = dynamic_config.version
        
        # 修改配置
        with open(temp_config_file, 'w') as f:
            f.write("key1: new_value\n")
        
        dynamic_config.reload()
        
        assert dynamic_config.version == initial_version + 1

    def test_detect_changes(self, dynamic_config):
        """测试变更检测。"""
        old_config = {"key1": "value1", "key2": "value2"}
        new_config = {"key1": "new_value1", "key3": "value3"}
        
        changed_keys = dynamic_config._detect_changes(old_config, new_config)
        
        assert "key1" in changed_keys  # 值改变
        assert "key2" in changed_keys  # 被删除
        assert "key3" in changed_keys  # 新增

    def test_detect_changes_no_change(self, dynamic_config):
        """测试无变更时的检测。"""
        config = {"key1": "value1", "key2": "value2"}
        
        changed_keys = dynamic_config._detect_changes(config, config.copy())
        
        assert len(changed_keys) == 0

    def test_notify_handlers(self, dynamic_config):
        """测试通知处理器。"""
        handler1 = Mock(spec=ConfigChangeHandler)
        handler2 = Mock(spec=ConfigChangeHandler)
        
        dynamic_config.register_handler(handler1)
        dynamic_config.register_handler(handler2)
        
        dynamic_config._notify_handlers({}, {}, ["key1"])
        
        assert handler1.on_config_change.called
        assert handler2.on_config_change.called

    def test_notify_handlers_handles_exception(self, dynamic_config):
        """测试通知处理器时异常处理。"""
        handler = Mock(spec=ConfigChangeHandler)
        handler.on_config_change.side_effect = Exception("Handler error")
        
        dynamic_config.register_handler(handler)
        
        # 不应该抛出异常
        dynamic_config._notify_handlers({}, {}, ["key1"])

    def test_start_watching(self, dynamic_config):
        """测试启动配置文件监控。"""
        with patch('src.config_manager.Observer') as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance
            
            dynamic_config.start_watching()
            
            assert mock_observer_instance.start.called
            assert dynamic_config._observer is not None

    def test_start_watching_already_running(self, dynamic_config):
        """测试重复启动监控。"""
        dynamic_config._observer = Mock()
        
        # 不应该抛出异常，应该直接返回
        dynamic_config.start_watching()

    def test_stop_watching(self, dynamic_config):
        """测试停止配置文件监控。"""
        mock_observer = Mock()
        dynamic_config._observer = mock_observer
        
        dynamic_config.stop_watching()
        
        assert mock_observer.stop.called
        assert mock_observer.join.called
        assert dynamic_config._observer is None

    def test_stop_watching_not_running(self, dynamic_config):
        """测试停止未运行的监控。"""
        # 不应该抛出异常
        dynamic_config.stop_watching()

    def test_load_config_file_not_found(self):
        """测试加载不存在的配置文件。"""
        with pytest.raises(Exception):
            DynamicConfig("/nonexistent/config.yaml")

    def test_load_config_invalid_yaml(self):
        """测试加载无效 YAML 文件。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):
                DynamicConfig(temp_path)
        finally:
            Path(temp_path).unlink()


class TestConfigSection:
    """ConfigSection 单元测试类。"""

    @pytest.fixture
    def dynamic_config(self):
        """创建测试用 DynamicConfig 实例。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("database:\n  host: localhost\n  port: 5432\n")
            temp_path = f.name
        
        config = DynamicConfig(temp_path)
        yield config
        
        Path(temp_path).unlink()

    def test_get(self, dynamic_config):
        """测试获取配置段中的值。"""
        section = ConfigSection(dynamic_config, "database")
        
        assert section.get("host") == "localhost"
        assert section.get("port") == 5432
        assert section.get("nonexistent", "default") == "default"

    def test_get_all(self, dynamic_config):
        """测试获取整个配置段。"""
        section = ConfigSection(dynamic_config, "database")
        
        all_config = section.get_all()
        
        assert isinstance(all_config, dict)
        assert all_config["host"] == "localhost"
        assert all_config["port"] == 5432


class TestGlobalConfigManager:
    """全局配置管理器测试。"""

    def test_init_config_manager(self):
        """测试初始化全局配置管理器。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\n")
            temp_path = f.name
        
        try:
            manager = init_config_manager(temp_path)
            
            assert isinstance(manager, DynamicConfig)
            assert get_config_manager() is manager
        finally:
            Path(temp_path).unlink()

    def test_get_config_manager_not_initialized(self):
        """测试未初始化时获取全局管理器失败。"""
        import src.config_manager
        original = src.config_manager._config_manager
        
        try:
            src.config_manager._config_manager = None
            
            with pytest.raises(RuntimeError, match="配置管理器未初始化"):
                get_config_manager()
        finally:
            src.config_manager._config_manager = original
