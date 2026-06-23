"""
pytest 全局配置 — 为 CI 环境创建假 config.yaml，避免测试因缺少配置文件而崩溃。
"""

import os
import tempfile
from pathlib import Path
import pytest


# 测试用假配置模板
_CI_CONFIG = """
llm:
  api_key: "sk-test-deepseek-key-for-ci"
  api_base: "https://api.deepseek.com"
  model_name: "deepseek-chat"
  temperature: 0.1
  max_tokens: 2048

embedding:
  api_key: "sk-test-bailian-key-for-ci"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model_name: "text-embedding-v3"
  batch_size: 25

chroma:
  persist_dir: "./data/test_chroma_db"
  collection_name: "test_course_materials"

retrieval:
  top_k: 8
  rerank_top_n: 4

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-4-v2"
  cache_dir: "./models"

memory:
  max_turns: 4

logging:
  level: "WARNING"
  file: null
"""


@pytest.fixture(autouse=True)
def setup_ci_config(monkeypatch):
    """在 CI 环境中自动创建假 config.yaml，避免 FileNotFoundError。"""
    cwd = Path.cwd()
    config_path = cwd / "config.yaml"

    if not config_path.exists():
        config_path.write_text(_CI_CONFIG, encoding="utf-8")

    # 确保测试不会意外调用真实 API
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key-for-ci")
    monkeypatch.setenv("BAILIAN_API_KEY", "sk-test-bailian-key-for-ci")

    yield

    # 清理（仅在 CI 创建的假文件）
    if config_path.exists() and "test-deepseek-key-for-ci" in config_path.read_text():
        config_path.unlink(missing_ok=True)

    # 清理测试用 Chroma 数据
    test_chroma = cwd / "data" / "test_chroma_db"
    if test_chroma.exists():
        import shutil
        shutil.rmtree(test_chroma, ignore_errors=True)
