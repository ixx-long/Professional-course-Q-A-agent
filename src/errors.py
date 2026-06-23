"""
统一错误定义模块。

提供标准化的错误类型，便于全局错误处理和前端统一解析。
"""

from typing import Dict


class QAError(Exception):
    """基础错误类。"""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", status_code: int = 500) -> None:
        self.message: str = message
        self.code: str = code
        self.status_code: int = status_code
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式（用于 JSON 响应）。"""
        return {
            "error": self.message,
            "code": self.code
        }


class ConfigError(QAError):
    """配置错误。"""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR", 500)


class AuthError(QAError):
    """认证错误。"""
    
    def __init__(self, message: str = "无权操作"):
        super().__init__(message, "AUTH_ERROR", 403)


class RateLimitError(QAError):
    """限流错误。"""
    
    def __init__(self, message: str = "请求过于频繁，请稍后重试"):
        super().__init__(message, "RATE_LIMIT", 429)


class RequestTimeoutError(QAError):
    """请求超时错误。"""
    
    def __init__(self, message: str = "请求超时，请稍后重试"):
        super().__init__(message, "TIMEOUT", 504)


class ValidationError(QAError):
    """参数校验错误。"""
    
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR", 400)


class NotFoundError(QAError):
    """资源未找到错误。"""
    
    def __init__(self, message: str = "资源未找到"):
        super().__init__(message, "NOT_FOUND", 404)


class LLMError(QAError):
    """LLM 调用错误。"""
    
    def __init__(self, message: str = "LLM 调用失败"):
        super().__init__(message, "LLM_ERROR", 500)


class RetrievalError(QAError):
    """检索错误。"""
    
    def __init__(self, message: str = "检索失败"):
        super().__init__(message, "RETRIEVAL_ERROR", 500)


class ImageError(QAError):
    """图片处理错误。"""
    
    def __init__(self, message: str):
        super().__init__(message, "IMAGE_ERROR", 400)
