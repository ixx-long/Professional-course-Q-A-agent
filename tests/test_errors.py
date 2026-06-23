"""
单元测试 - errors 模块
"""
import pytest
from src.errors import (
    QAError, ConfigError, AuthError, RateLimitError,
    RequestTimeoutError, ValidationError, NotFoundError, LLMError
)


def test_qa_error_base():
    """测试基础错误类"""
    error = QAError("测试错误", "TEST_ERROR", 500)
    assert error.message == "测试错误"
    assert error.code == "TEST_ERROR"
    assert error.status_code == 500
    
    error_dict = error.to_dict()
    assert error_dict["error"] == "测试错误"
    assert error_dict["code"] == "TEST_ERROR"


def test_config_error():
    """测试配置错误"""
    error = ConfigError("配置错误")
    assert error.code == "CONFIG_ERROR"
    assert error.status_code == 500


def test_auth_error():
    """测试认证错误"""
    error = AuthError("无权操作")
    assert error.code == "AUTH_ERROR"
    assert error.status_code == 403


def test_rate_limit_error():
    """测试限流错误"""
    error = RateLimitError()
    assert error.code == "RATE_LIMIT"
    assert error.status_code == 429


def test_timeout_error():
    """测试超时错误"""
    error = RequestTimeoutError()
    assert error.code == "TIMEOUT"
    assert error.status_code == 504


def test_validation_error():
    """测试校验错误"""
    error = ValidationError("参数无效")
    assert error.code == "VALIDATION_ERROR"
    assert error.status_code == 400


def test_not_found_error():
    """测试未找到错误"""
    error = NotFoundError()
    assert error.code == "NOT_FOUND"
    assert error.status_code == 404


def test_llm_error():
    """测试 LLM 错误"""
    error = LLMError()
    assert error.code == "LLM_ERROR"
    assert error.status_code == 500
