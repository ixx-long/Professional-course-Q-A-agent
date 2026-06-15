"""
工具模块：配置加载、日志初始化、API Key 脱敏。

提供项目级别的通用工具函数，被所有其他 src 模块依赖。
"""

import os
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml


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

    用法:
        config = load_config("config.yaml")
        api_key = config["llm"]["api_key"]
    """
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

    # 校验必须的顶层键
    required_sections = ["llm", "embedding", "chroma", "retrieval", "reranker", "memory", "logging"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"配置文件缺少以下段: {', '.join(missing)}")

    # 深度校验：关键字段不能为空
    if not config["llm"].get("api_key"):
        raise ValueError("llm.api_key 不能为空")
    if not config["llm"].get("api_base"):
        raise ValueError("llm.api_base 不能为空")
    if not config["embedding"].get("api_key"):
        raise ValueError("embedding.api_key 不能为空")
    if not config["embedding"].get("api_base"):
        raise ValueError("embedding.api_base 不能为空")

    # 数值合理性校验
    top_k = config["retrieval"].get("top_k", 8)
    rerank_top_n = config["retrieval"].get("rerank_top_n", 4)
    if not (1 <= top_k <= 100):
        raise ValueError(f"retrieval.top_k 应在 1-100 之间，当前值: {top_k}")
    if not (1 <= rerank_top_n <= top_k):
        raise ValueError(f"retrieval.rerank_top_n 应在 1-top_k 之间，当前值: {rerank_top_n}")

    # memory.max_turns 合理性
    max_turns = config["memory"].get("max_turns", 4)
    if not (1 <= max_turns <= 50):
        raise ValueError(f"memory.max_turns 应在 1-50 之间，当前值: {max_turns}")

    return config


def mask_key(key: str) -> str:
    """脱敏 API Key，仅显示前 6 和后 4 个字符。"""
    if not key or len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def setup_logger(
    name: str = "course_qa",
    log_file: str = "logs/qa.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    初始化日志记录器，同时输出到文件和控制台。

    Args:
        name: 日志记录器名称。
        log_file: 日志文件路径，None 则仅控制台输出。
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）。

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

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
