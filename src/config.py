"""
配置模型定义（使用 Pydantic 进行结构化校验）。
"""

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from typing import Optional, List


class LLMConfig(BaseModel):
    """LLM 配置。"""
    api_key: str = Field(..., min_length=8, description="LLM API Key")
    api_base: str = Field(..., description="LLM API 基础 URL")
    model_name: str = Field(default="deepseek-chat", description="模型名称")
    temperature: float = Field(default=0.1, ge=0, le=2, description="生成温度")
    max_tokens: int = Field(default=2048, ge=1, le=32000, description="最大 token 数")
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证 API Key 格式。"""
        if not v or v.isspace():
            raise ValueError("API Key 不能为空")
        if len(v) < 8:
            raise ValueError("API Key 长度至少为 8 个字符")
        return v
    
    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str) -> str:
        """验证 API 基础 URL。"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("API 基础 URL 必须以 http:// 或 https:// 开头")
        return v.rstrip("/")


class EmbeddingConfig(BaseModel):
    """Embedding 配置。"""
    api_key: str = Field(..., min_length=8, description="Embedding API Key")
    api_base: str = Field(..., description="Embedding API 基础 URL")
    model_name: str = Field(default="text-embedding-v3", description="模型名称")
    dimensions: int = Field(default=1024, ge=128, le=4096, description="向量维度")
    batch_size: int = Field(default=25, ge=1, le=100, description="批量大小")
    max_concurrent: int = Field(default=4, ge=1, le=16, description="最大并发数")
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证 API Key 格式。"""
        if not v or v.isspace():
            raise ValueError("API Key 不能为空")
        if len(v) < 8:
            raise ValueError("API Key 长度至少为 8 个字符")
        return v
    
    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str) -> str:
        """验证 API 基础 URL。"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("API 基础 URL 必须以 http:// 或 https:// 开头")
        return v.rstrip("/")


class ChromaConfig(BaseModel):
    """Chroma 向量库配置。"""
    persist_dir: str = Field(default="./data/chroma_db", description="持久化目录")
    collection_name: str = Field(default="course_qa", min_length=1, max_length=63, description="集合名称")
    
    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """验证集合名称格式。"""
        if not v or v.isspace():
            raise ValueError("集合名称不能为空")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("集合名称只能包含字母、数字、下划线和连字符")
        return v


class RetrievalConfig(BaseModel):
    """检索配置。"""
    top_k: int = Field(default=8, ge=1, le=100, description="初始召回数")
    rerank_top_n: int = Field(default=4, ge=1, description="重排序保留数")

    @model_validator(mode='after')
    def validate_rerank_top_n(self):
        """校验 rerank_top_n 不能大于 top_k。"""
        if self.rerank_top_n > self.top_k:
            raise ValueError("rerank_top_n 不能大于 top_k")
        return self


class RerankerConfig(BaseModel):
    """重排序模型配置。"""
    model_name: str = Field(default="BAAI/bge-reranker-base", description="模型名称")
    cache_dir: str = Field(default="./models", description="模型缓存目录")
    device: Optional[str] = Field(default=None, description="设备（cuda/cpu）")


class MemoryConfig(BaseModel):
    """对话记忆配置。"""
    max_turns: int = Field(default=4, ge=1, le=50, description="最大对话轮数")


class LoggingConfig(BaseModel):
    """日志配置。"""
    level: str = Field(default="INFO", description="日志级别")
    file: str = Field(default="logs/qa.log", description="日志文件路径")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """验证日志级别。"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"日志级别必须是 {', '.join(valid_levels)} 之一")
        return v_upper


class AppConfig(BaseModel):
    """应用总配置。"""
    model_config = ConfigDict(validate_assignment=True)
    
    llm: LLMConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    memory: MemoryConfig
    logging: LoggingConfig
    course_keywords: list[str] = Field(default_factory=list, description="课程关键词列表")


def validate_config(config_dict: dict) -> AppConfig:
    """校验配置字典并返回 Pydantic 模型实例。

    Args:
        config_dict: 从 YAML 加载的配置字典。

    Returns:
        校验通过的 AppConfig 实例。

    Raises:
        pydantic.ValidationError: 配置校验失败时抛出。
    """
    return AppConfig(**config_dict)
