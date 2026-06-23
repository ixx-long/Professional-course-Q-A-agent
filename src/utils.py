"""
工具模块：配置加载、日志初始化、API Key 脱敏、密钥管理。

提供项目级别的通用工具函数，被所有其他 src 模块依赖。
"""

import os
import re
import logging
import sys
import hashlib
import hmac
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

import yaml


# 环境变量 → 配置字段映射
# 优先级：环境变量 > config.yaml 值 > 默认值
_ENV_MAP = {
    "DEEPSEEK_API_KEY":    ("llm", "api_key"),
    "DEEPSEEK_API_BASE":   ("llm", "api_base"),
    "DEEPSEEK_MODEL":      ("llm", "model_name"),
    "BAILIAN_API_KEY":     ("embedding", "api_key"),
    "BAILIAN_API_BASE":    ("embedding", "api_base"),
    "BAILIAN_MODEL":       ("embedding", "model_name"),
}


def _resolve_from_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """用环境变量覆盖配置中的敏感字段。"""
    for env_var, (section, key) in _ENV_MAP.items():
        val: Optional[str] = os.environ.get(env_var)
        if val:
            config.setdefault(section, {})[key] = val

    # 也支持 ${VAR} 占位符语法
    _unresolved: list[str] = []

    def _resolve(val: Any) -> Any:
        if isinstance(val, str):
            m = re.match(r'^\$\{(\w+)\}$', val)
            if m:
                var_name = m.group(1)
                if var_name in os.environ:
                    return os.environ[var_name]
                else:
                    _unresolved.append(var_name)
        return val

    for section in config:
        if isinstance(config[section], dict):
            for key in config[section]:
                config[section][key] = _resolve(config[section][key])

    if _unresolved:
        logging.getLogger(__name__).warning(
            f"以下环境变量未设置，占位符未替换: {', '.join(_unresolved)}"
        )

    return config


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载 YAML 配置文件并返回字典。

    Args:
        config_path: 配置文件路径，支持相对路径和绝对路径。

    Returns:
        配置字典，包含 llm、embedding、chroma、retrieval、reranker、memory、logging 段。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        yaml.YAMLError: YAML 格式错误时抛出。
        ValueError: 配置校验失败时抛出。

    用法:
        config = load_config("config.yaml")
        api_key = config["llm"]["api_key"]
    """
    from .config import validate_config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {path.absolute()}\n"
            f"请复制 config.example.yaml 为 config.yaml 并填入你的 API Key。"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"配置文件 {path} 格式错误: {e}")

    if config is None:
        raise ValueError(f"配置文件 {path} 内容为空")

    # 用环境变量覆盖配置中的敏感字段
    config = _resolve_from_env(config)

    # 使用 Pydantic 进行结构化校验
    try:
        validated_config = validate_config(config)
        # 转换回字典以保持向后兼容
        return validated_config.model_dump()
    except Exception as e:
        raise ValueError(f"配置校验失败: {e}")


def mask_key(key: str) -> str:
    """脱敏 API Key，仅显示前 6 和后 4 个字符。"""
    if not key or len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def setup_logger(
    name: str = "course_qa",
    log_file: str = "logs/qa.log",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    初始化日志记录器，同时输出到文件和控制台。

    Args:
        name: 日志记录器名称。
        log_file: 日志文件路径，None 则仅控制台输出。
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）。
        max_bytes: 单个日志文件最大字节数（默认 10MB）。
        backup_count: 保留的旧日志文件数量（默认 5 个）。

    Returns:
        配置完成的 logging.Logger 实例。

    用法:
        logger = setup_logger("course_qa", "logs/qa.log", "INFO")
        logger.info("系统启动")
    """
    logger = logging.getLogger(name)

    # 如果已配置过且名称相同，先清理旧 handler 再重新配置
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 格式
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件输出（使用 RotatingFileHandler 实现日志轮转）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# 企业级密钥管理服务
# ============================================================


class SecretManager:
    """
    企业级密钥管理器。
    
    功能：
    - 从环境变量或加密文件加载密钥
    - 运行时解密，内存中明文存储
    - 支持密钥轮换和版本管理
    - 自动检测密钥泄露风险
    """
    
    def __init__(self, master_key: Optional[str] = None, max_audit_log_size: int = 1000):
        """
        初始化密钥管理器。
        
        Args:
            master_key: 主密钥，用于加密/解密。优先从环境变量 SECRET_MASTER_KEY 获取。
            max_audit_log_size: 审计日志最大条目数，超过后自动清理最旧记录（默认 1000）。
        """
        self.master_key = master_key or os.environ.get("SECRET_MASTER_KEY")
        self._secrets: Dict[str, str] = {}
        self._secret_versions: Dict[str, int] = {}
        self._secret_access_times: Dict[str, float] = {}
        self._audit_log: list = []
        self._max_audit_log_size = max_audit_log_size
        
        if not self.master_key:
            logging.warning("未设置主密钥，将使用环境变量直接读取模式")
    
    def _trim_audit_log(self) -> None:
        """清理审计日志，保持不超过最大限制。"""
        if len(self._audit_log) > self._max_audit_log_size:
            # 保留最新的记录
            self._audit_log = self._audit_log[-self._max_audit_log_size:]
    
    def _derive_key(self, salt: bytes) -> bytes:
        """从主密钥派生加密密钥。"""
        if not self.master_key:
            raise ValueError("未设置主密钥")
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(
            kdf.derive(self.master_key.encode())
        )
    
    def encrypt_secret(self, plaintext: str) -> str:
        """
        加密敏感信息。
        
        Args:
            plaintext: 明文
            
        Returns:
            加密后的字符串（格式：salt:ciphertext）
        """
        if not self.master_key:
            raise ValueError("未设置主密钥，无法加密")
        
        salt = os.urandom(16)
        key = self._derive_key(salt)
        f = Fernet(key)
        ciphertext = f.encrypt(plaintext.encode())
        return f"{base64.b64encode(salt).decode()}:{ciphertext.decode()}"
    
    def decrypt_secret(self, encrypted: str) -> str:
        """
        解密敏感信息。
        
        Args:
            encrypted: 加密字符串（格式：salt:ciphertext）
            
        Returns:
            解密后的明文
        """
        if not self.master_key:
            raise ValueError("未设置主密钥，无法解密")
        
        try:
            salt_b64, ciphertext = encrypted.split(":", 1)
            salt = base64.b64decode(salt_b64)
            key = self._derive_key(salt)
            f = Fernet(key)
            return f.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            raise ValueError(f"解密失败：{e}")
    
    def set_secret(self, name: str, value: str, encrypted: bool = False) -> None:
        """
        设置密钥。
        
        Args:
            name: 密钥名称
            value: 密钥值（明文或加密）
            encrypted: 是否已加密
        """
        if encrypted:
            decrypted_value = self.decrypt_secret(value)
        else:
            decrypted_value = value
        
        self._secrets[name] = decrypted_value
        self._secret_versions[name] = self._secret_versions.get(name, 0) + 1
        self._secret_access_times[name] = time.time()
        
        # 记录审计日志
        self._audit_log.append({
            "action": "set",
            "name": name,
            "version": self._secret_versions[name],
            "timestamp": time.time()
        })
        self._trim_audit_log()
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取密钥。
        
        Args:
            name: 密钥名称
            default: 默认值
            
        Returns:
            密钥值
        """
        value = self._secrets.get(name, default)
        self._secret_access_times[name] = time.time()
        
        # 记录审计日志
        self._audit_log.append({
            "action": "get",
            "name": name,
            "timestamp": time.time()
        })
        self._trim_audit_log()
        
        return value
    
    def load_from_env(self, env_mapping: Dict[str, str]) -> None:
        """
        从环境变量批量加载密钥。
        
        Args:
            env_mapping: 环境变量名到密钥名的映射
        """
        for env_var, secret_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                self.set_secret(secret_name, value)
    
    def load_from_encrypted_file(self, file_path: str) -> None:
        """
        从加密文件加载密钥。
        
        Args:
            file_path: 加密文件路径
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"加密密钥文件不存在：{file_path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                encrypted_data = yaml.safe_load(f)
            
            for name, encrypted_value in encrypted_data.items():
                self.set_secret(name, encrypted_value, encrypted=True)
                
        except Exception as e:
            raise ValueError(f"加载加密密钥文件失败：{e}")
    
    def rotate_secret(self, name: str, new_value: str) -> None:
        """
        轮换密钥。
        
        Args:
            name: 密钥名称
            new_value: 新密钥值
        """
        old_version = self._secret_versions.get(name, 0)
        self.set_secret(name, new_value)
        
        # 记录审计日志
        self._audit_log.append({
            "action": "rotate",
            "name": name,
            "old_version": old_version,
            "new_version": self._secret_versions[name],
            "timestamp": time.time()
        })
        self._trim_audit_log()
    
    def audit_report(self) -> Dict[str, Any]:
        """
        生成审计报告。
        
        Returns:
            审计报告字典
        """
        return {
            "total_secrets": len(self._secrets),
            "access_times": self._secret_access_times.copy(),
            "versions": self._secret_versions.copy(),
            "recent_audit": self._audit_log[-50:],  # 最近 50 条记录
        }
    
    def check_leak_risk(self) -> List[Dict[str, Any]]:
        """
        检查密钥泄露风险。
        
        Returns:
            风险列表
        """
        risks = []
        
        # 检查长时间未访问的密钥
        current_time = time.time()
        for name, access_time in self._secret_access_times.items():
            days_since_access = (current_time - access_time) / 86400
            if days_since_access > 90:
                risks.append({
                    "type": "unused_secret",
                    "name": name,
                    "days_since_access": days_since_access
                })
        
        # 检查版本过多的密钥（可能频繁轮换）
        for name, version in self._secret_versions.items():
            if version > 10:
                risks.append({
                    "type": "frequent_rotation",
                    "name": name,
                    "version_count": version
                })
        
        return risks


# 全局密钥管理器实例
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """获取全局密钥管理器实例。"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


def init_secret_manager(master_key: Optional[str] = None) -> SecretManager:
    """初始化全局密钥管理器。"""
    global _secret_manager
    _secret_manager = SecretManager(master_key)
    return _secret_manager
