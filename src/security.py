"""
安全验证模块。

提供输入验证、XSS 防护、SQL 注入防护、敏感信息脱敏等安全检查。
"""

import re
import html
import logging
from typing import Optional, Any, Dict, List
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecurityValidator:
    """安全验证器。"""
    
    # 危险字符模式
    DANGEROUS_PATTERNS = [
        r"<script.*?>.*?</script>",  # Script 标签
        r"javascript:",  # JavaScript 协议
        r"on\w+\s*=",  # 事件处理器
        r"<iframe.*?>",  # iframe 标签
        r"<object.*?>",  # object 标签
        r"<embed.*?>",  # embed 标签
        r"eval\s*\(",  # eval 函数
        r"expression\s*\(",  # CSS expression
        r"url\s*\(",  # CSS url
        r"import\s*\(",  # 动态导入
    ]
    
    # SQL 注入模式
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b.*\b(FROM|INTO|WHERE|SET|VALUES|TABLE)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
        r"(EXEC\s*\(|EXECUTE\s*\()",
    ]
    
    # 路径遍历模式
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",  # Unix 路径遍历
        r"\.\.\\",  # Windows 路径遍历
        r"%2e%2e",  # URL 编码的 ..
        r"%252e%252e",  # 双重 URL 编码
    ]
    
    @classmethod
    def validate_question(cls, question: str, max_length: int = 5000) -> Optional[str]:
        """
        验证问题文本。
        
        Args:
            question: 问题文本
            max_length: 最大长度
            
        Returns:
            错误信息，如果验证通过则返回 None
        """
        if not question or not question.strip():
            return "问题不能为空"
        
        if len(question) > max_length:
            return f"问题文本过长（最大 {max_length} 字符）"
        
        # 检查危险内容
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE | re.DOTALL):
                logger.warning(f"检测到危险内容: {pattern}")
                return "问题包含不安全的内容"
        
        return None
    
    @classmethod
    def validate_session_id(cls, session_id: str) -> Optional[str]:
        """
        验证会话 ID。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            错误信息，如果验证通过则返回 None
        """
        if not session_id:
            return None  # 允许空值，使用默认值
        
        # 只允许字母、数字、下划线和连字符
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return "会话 ID 只能包含字母、数字、下划线和连字符"
        
        if len(session_id) > 128:
            return "会话 ID 过长（最大 128 字符）"
        
        return None
    
    @classmethod
    def validate_course(cls, course: Optional[str]) -> Optional[str]:
        """
        验证课程名称。
        
        Args:
            course: 课程名称
            
        Returns:
            错误信息，如果验证通过则返回 None
        """
        if not course:
            return None  # 允许空值
        
        # 只允许中文、字母、数字、下划线和连字符
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]+$', course):
            return "课程名称只能包含中文、字母、数字、下划线和连字符"
        
        if len(course) > 64:
            return "课程名称过长（最大 64 字符）"
        
        return None
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        清理文本内容，防止 XSS。
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # HTML 转义
        sanitized = html.escape(text)
        
        # 移除危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @classmethod
    def check_sql_injection(cls, text: str) -> bool:
        """
        检查是否包含 SQL 注入。
        
        Args:
            text: 待检查文本
            
        Returns:
            True 如果检测到 SQL 注入
        """
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"检测到 SQL 注入尝试: {pattern}")
                return True
        return False
    
    @classmethod
    def check_path_traversal(cls, path: str) -> bool:
        """
        检查是否包含路径遍历。
        
        Args:
            path: 文件路径
            
        Returns:
            True 如果检测到路径遍历
        """
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"检测到路径遍历尝试: {pattern}")
                return True
        return False
    
    @classmethod
    def validate_file_path(cls, file_path: str, allowed_dirs: list = None) -> Optional[str]:
        """
        验证文件路径。
        
        Args:
            file_path: 文件路径
            allowed_dirs: 允许的目录列表
            
        Returns:
            错误信息，如果验证通过则返回 None
        """
        # 检查路径遍历
        if cls.check_path_traversal(file_path):
            return "文件路径包含非法内容"
        
        # 检查是否在允许的目录内
        if allowed_dirs:
            import os
            abs_path = os.path.abspath(file_path)
            is_allowed = any(
                abs_path.startswith(os.path.abspath(d))
                for d in allowed_dirs
            )
            if not is_allowed:
                return "文件路径不在允许的目录内"
        
        return None
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理字典中的所有字符串值。
        
        Args:
            data: 原始字典
            
        Returns:
            清理后的字典
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = cls.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_text(item) if isinstance(item, str)
                    else cls.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


def validate_request_data(data: Dict[str, Any]) -> Optional[str]:
    """
    验证请求数据。
    
    Args:
        data: 请求数据字典
        
    Returns:
        错误信息，如果验证通过则返回 None
    """
    # 验证问题
    if "question" in data:
        error = SecurityValidator.validate_question(data["question"])
        if error:
            return error
    
    # 验证会话 ID
    if "session_id" in data:
        error = SecurityValidator.validate_session_id(data["session_id"])
        if error:
            return error
    
    # 验证课程
    if "course" in data:
        error = SecurityValidator.validate_course(data["course"])
        if error:
            return error
    
    return None


class SensitiveDataMasker:
    """敏感信息脱敏器。"""
    
    # API Key 模式
    API_KEY_PATTERNS = [
        (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9]{20,})["\']?', r'\1***'),
        (r'(sk-[a-zA-Z0-9]{20,})', r'sk-***'),  # OpenAI 风格
        (r'(key-[a-zA-Z0-9]{20,})', r'key-***'),
    ]
    
    # 密码模式
    PASSWORD_PATTERNS = [
        (r'(password["\']?\s*[:=]\s*["\']?)([^"\',\s]+)["\']?', r'\1***'),
        (r'(passwd["\']?\s*[:=]\s*["\']?)([^"\',\s]+)["\']?', r'\1***'),
        (r'(pwd["\']?\s*[:=]\s*["\']?)([^"\',\s]+)["\']?', r'\1***'),
    ]
    
    # Token 模式
    TOKEN_PATTERNS = [
        (r'(token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9._-]{20,})["\']?', r'\1***'),
        (r'(Bearer\s+)([a-zA-Z0-9._-]{20,})', r'\1***'),
    ]
    
    # 邮箱模式
    EMAIL_PATTERN = r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    
    # 手机号模式（中国大陆）
    PHONE_PATTERN = r'\b(1[3-9]\d)(\d{4})(\d{4})\b'
    
    # 身份证号模式
    ID_CARD_PATTERN = r'\b(\d{6})(\d{8})(\d{4})\b'
    
    @classmethod
    def mask_text(cls, text: str) -> str:
        """
        脱敏文本中的敏感信息。
        
        Args:
            text: 原始文本
            
        Returns:
            脱敏后的文本
        """
        masked = text
        
        # 脱敏 API Key
        for pattern, replacement in cls.API_KEY_PATTERNS:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        # 脱敏密码
        for pattern, replacement in cls.PASSWORD_PATTERNS:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        # 脱敏 Token
        for pattern, replacement in cls.TOKEN_PATTERNS:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        # 脱敏邮箱（保留首字母）
        masked = re.sub(
            cls.EMAIL_PATTERN,
            lambda m: f"{m.group(1)[0]}***@{m.group(2)}",
            masked
        )
        
        # 脱敏手机号（保留前3后4）
        masked = re.sub(
            cls.PHONE_PATTERN,
            lambda m: f"{m.group(1)}****{m.group(3)}",
            masked
        )
        
        # 脱敏身份证号（保留前6后4）
        masked = re.sub(
            cls.ID_CARD_PATTERN,
            lambda m: f"{m.group(1)}********{m.group(3)}",
            masked
        )
        
        return masked
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        脱敏字典中的敏感信息。
        
        Args:
            data: 原始字典
            
        Returns:
            脱敏后的字典
        """
        masked = {}
        for key, value in data.items():
            # 检查键名是否包含敏感词
            key_lower = key.lower()
            if any(word in key_lower for word in ['password', 'passwd', 'pwd', 'secret', 'api_key', 'apikey', 'token']):
                masked[key] = '***'
            elif isinstance(value, str):
                masked[key] = cls.mask_text(value)
            elif isinstance(value, dict):
                masked[key] = cls.mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    cls.mask_text(item) if isinstance(item, str)
                    else cls.mask_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                masked[key] = value
        return masked


class RateLimitAbuseDetector:
    """请求频率异常检测器。"""
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        """
        初始化检测器。
        
        Args:
            window_size: 时间窗口（秒）
            max_requests: 窗口内最大请求数
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.request_history: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> 解封时间
        self.lock = __import__('threading').Lock()
    
    def check_request(self, client_id: str) -> bool:
        """
        检查请求是否异常。
        
        Args:
            client_id: 客户端标识（IP 或用户 ID）
            
        Returns:
            True 如果请求正常，False 如果异常
        """
        import time
        
        now = time.time()
        
        with self.lock:
            # 检查是否被封禁
            if client_id in self.blocked_ips:
                if now < self.blocked_ips[client_id]:
                    logger.warning(f"客户端 {client_id} 处于封禁状态")
                    return False
                else:
                    del self.blocked_ips[client_id]
            
            # 清理过期记录
            if client_id in self.request_history:
                self.request_history[client_id] = [
                    t for t in self.request_history[client_id]
                    if now - t < self.window_size
                ]
            else:
                self.request_history[client_id] = []
            
            # 检查频率
            if len(self.request_history[client_id]) >= self.max_requests:
                logger.warning(f"客户端 {client_id} 请求频率异常，触发封禁")
                self.blocked_ips[client_id] = now + 300  # 封禁 5 分钟
                return False
            
            # 记录请求
            self.request_history[client_id].append(now)
            return True
    
    def get_request_count(self, client_id: str) -> int:
        """
        获取客户端在时间窗口内的请求次数。
        
        Args:
            client_id: 客户端标识
            
        Returns:
            请求次数
        """
        import time
        
        now = time.time()
        
        with self.lock:
            if client_id not in self.request_history:
                return 0
            
            return len([
                t for t in self.request_history[client_id]
                if now - t < self.window_size
            ])
    
    def is_blocked(self, client_id: str) -> bool:
        """
        检查客户端是否被封禁。
        
        Args:
            client_id: 客户端标识
            
        Returns:
            True 如果被封禁
        """
        import time
        
        with self.lock:
            if client_id not in self.blocked_ips:
                return False
            
            if time.time() < self.blocked_ips[client_id]:
                return True
            else:
                del self.blocked_ips[client_id]
                return False


def log_securityEvent(event_type: str, details: Dict[str, Any], severity: str = "WARNING"):
    """
    记录安全事件。
    
    Args:
        event_type: 事件类型（如 "XSS_ATTEMPT", "SQL_INJECTION", "RATE_LIMIT_EXCEEDED"）
        details: 事件详情
        severity: 严重程度（"INFO", "WARNING", "ERROR", "CRITICAL"）
    """
    log_func = getattr(logger, severity.lower(), logger.warning)
    
    # 脱敏详情
    masked_details = SensitiveDataMasker.mask_dict(details)
    
    log_func(
        f"安全事件: {event_type} | 详情: {masked_details}",
        extra={
            'event_type': event_type,
            'severity': severity,
            'details': masked_details
        }
    )
