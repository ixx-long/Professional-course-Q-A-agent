"""
单元测试 - utils 模块
"""
import pytest
from src.utils import load_config, setup_logger


def test_load_config_success():
    """测试成功加载配置文件"""
    config = load_config("config.example.yaml")
    assert "llm" in config
    assert "embedding" in config
    assert "retrieval" in config
    assert config["retrieval"]["top_k"] > 0


def test_load_config_missing_file():
    """测试配置文件不存在"""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_setup_logger():
    """测试日志器创建"""
    logger = setup_logger("test_logger", "logs/test.log", "DEBUG")
    assert logger.name == "test_logger"
    assert logger.level == 10  # DEBUG level
