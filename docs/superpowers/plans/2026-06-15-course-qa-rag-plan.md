# RAG 课程答疑智能体 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 RAG 的专业课程答疑 CLI 工具，支持 PDF/Word/Markdown 知识库、DeepSeek 生成、百炼 Embedding、CrossEncoder 重排序。

**Architecture:** 离线阶段通过 build_kb.py 将文档向量化存入 Chroma；在线阶段通过 qa_cli.py 交互式问答，经检索→重排序→生成链路返回带引用来源的回答。

**Tech Stack:** Python 3.10+, LangChain, DeepSeek API, 阿里云百炼 Embedding, Chroma, CrossEncoder

---

## 文件结构

```
course-qa-rag/
├── requirements.txt          # 创建
├── config.example.yaml       # 创建（安全模板，不含真实 Key）
├── .gitignore                # 创建
├── build_kb.py               # 创建
├── qa_cli.py                 # 创建
├── src/
│   ├── __init__.py           # 创建
│   ├── utils.py              # 创建
│   ├── loader.py             # 创建
│   ├── vectorstore.py        # 创建
│   ├── retriever.py          # 创建
│   └── chain.py              # 创建
├── data/
│   └── documents/            # 放入 .gitkeep
└── logs/                     # 运行时自动创建
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `requirements.txt`
- Create: `config.example.yaml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `data/documents/.gitkeep`

- [ ] **Step 1: 创建 requirements.txt**

```txt
# LangChain 核心
langchain==0.3.13
langchain-community==0.3.13
langchain-openai==0.2.14
langchain-chroma==0.2.1

# Chroma 向量数据库
chromadb==0.5.23

# 文档解析
pypdf==5.1.0
docx2txt==0.8

# 重排序
sentence-transformers==3.3.1

# 配置与工具
pyyaml==6.0.2
tqdm==4.67.1

# CLI 增强（可选）
colorama==0.4.6
```

- [ ] **Step 2: 创建 config.example.yaml**

```yaml
# ============================================================
# 生成模型配置（DeepSeek）
# ============================================================
llm:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  api_base: "https://api.deepseek.com"
  model_name: "deepseek-chat"
  temperature: 0.1
  max_tokens: 2048

# ============================================================
# Embedding 模型配置（阿里云百炼）
# ============================================================
embedding:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model_name: "text-embedding-v3"

# ============================================================
# 向量库配置
# ============================================================
chroma:
  persist_dir: "./data/chroma_db"
  collection_name: "course_materials"

# ============================================================
# 检索配置
# ============================================================
retrieval:
  top_k: 8
  rerank_top_n: 4

# ============================================================
# 重排序模型配置
# ============================================================
reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-4-v2"
  cache_dir: "./models"

# ============================================================
# 对话记忆配置
# ============================================================
memory:
  max_turns: 4

# ============================================================
# 日志配置
# ============================================================
logging:
  level: "INFO"
  file: "./logs/qa.log"
```

- [ ] **Step 3: 创建 .gitignore**

```gitignore
# 配置文件（含 API Key）
config.yaml

# 向量库持久化数据
data/chroma_db/

# 下载的模型
models/

# 日志
logs/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# IDE
.vscode/
.idea/
```

- [ ] **Step 4: 创建 src/__init__.py**

```python
"""基于 RAG 的专业课程答疑智能体"""
```

- [ ] **Step 5: 创建占位目录**

```bash
mkdir -p data/documents logs
touch data/documents/.gitkeep
```

- [ ] **Step 6: 初始化 Git 仓库**

```bash
cd "d:\lenovo\Documents\vs\Professional course Q&A agent"
git init
git add .
git commit -m "chore: 初始化项目脚手架"
```

---

### Task 2: utils.py — 配置加载与日志

**Files:**
- Create: `src/utils.py`

- [ ] **Step 1: 编写完整模块**

```python
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

    # 避免重复添加 handler
    if logger.handlers:
        return logger

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
```

- [ ] **Step 2: 验证模块可导入**

```bash
cd "d:\lenovo\Documents\vs\Professional course Q&A agent"
python -c "from src.utils import load_config, setup_logger, mask_key; print('✓ utils 模块正常')"
```

- [ ] **Step 3: 提交**

```bash
git add src/utils.py
git commit -m "feat: 添加 utils 模块（配置加载+日志）"
```

---

### Task 3: loader.py — 文档加载与分块

**Files:**
- Create: `src/loader.py`

- [ ] **Step 1: 编写完整模块**

```python
"""
文档加载与分块模块。

支持的格式:
  - PDF (.pdf): 使用 PyPDFLoader 逐页加载
  - Word (.docx): 使用 Docx2txtLoader 加载
  - Markdown / 纯文本 (.md / .txt): 使用 TextLoader 加载

分块策略:
  - 使用 RecursiveCharacterTextSplitter
  - chunk_size=1000, chunk_overlap=200
  - 每个 chunk 附带 metadata: source(文件名), page(页码), chunk_id(唯一编号)
"""

import os
import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# 支持的文件扩展名 → Loader 映射
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def _get_loader(file_path: Path):
    """
    根据文件扩展名返回对应的 LangChain Loader 类。

    Args:
        file_path: 文件路径。

    Returns:
        Loader 类（未实例化）。

    Raises:
        ValueError: 不支持的文件格式时抛出。
    """
    ext = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        supported = ", ".join(LOADER_MAP.keys())
        raise ValueError(f"不支持的文件格式 '{ext}'，支持: {supported}")
    return loader_cls


def load_single_document(file_path: Path) -> List[Document]:
    """
    加载单个文件并返回 Document 列表（未分块）。

    Args:
        file_path: 文件路径。

    Returns:
        List[Document]: 每个元素代表一个页面或整个文档。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不支持的文件格式。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loader_cls = _get_loader(file_path)

    # PyPDFLoader 需要特殊处理（逐页加载）
    if file_path.suffix.lower() == ".pdf":
        loader = loader_cls(str(file_path))
        docs = loader.load()
        # 为每页补充 source 元数据
        for i, doc in enumerate(docs):
            doc.metadata["source"] = file_path.name
            doc.metadata["page"] = i + 1
        return docs
    else:
        loader = loader_cls(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["page"] = 1  # 非 PDF 文件视为单页
        return docs


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档列表进行分块。

    Args:
        docs: 原始 Document 列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 相邻块重叠字符数。

    Returns:
        分块后的 Document 列表，每个 chunk 的 metadata 包含:
        - source: 来源文件名
        - page: 页码
        - chunk_id: 基于 source+page+content 的 MD5 哈希（用于去重）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", ".", ";", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(docs)

    # 为每个 chunk 生成唯一 ID（用于去重）
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:8]
        chunk.metadata["chunk_id"] = f"{source}#p{page}#{content_hash}"
        chunk.metadata["chunk_index"] = i

    return chunks


def load_documents(input_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    遍历目录下所有支持的文档，加载并分块。

    Args:
        input_dir: 文档目录路径。
        chunk_size: 分块大小。
        chunk_overlap: 重叠大小。

    Returns:
        分块后的 Document 列表。

    用法:
        docs = load_documents("./data/documents")
        print(f"共加载 {len(docs)} 个文本块")
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {input_path.absolute()}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {input_path}")

    all_docs = []
    supported_exts = tuple(LOADER_MAP.keys())

    # 递归遍历目录
    for root, _, files in os.walk(input_path):
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in supported_exts:
                try:
                    docs = load_single_document(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    # 单个文件加载失败不中断整体流程
                    import logging
                    logging.getLogger("course_qa").warning(f"跳过文件 {file_path}: {e}")

    if not all_docs:
        raise RuntimeError(f"在 {input_dir} 中未找到任何支持的文档文件")

    # 分块
    chunks = split_documents(all_docs, chunk_size, chunk_overlap)
    return chunks
```

- [ ] **Step 2: 验证模块**

```bash
python -c "from src.loader import load_documents; print('✓ loader 模块正常')"
```

- [ ] **Step 3: 提交**

```bash
git add src/loader.py
git commit -m "feat: 添加 loader 模块（PDF/Word/MD 加载+分块）"
```

---

### Task 4: vectorstore.py — 向量库管理

**Files:**
- Create: `src/vectorstore.py`

- [ ] **Step 1: 编写完整模块**

```python
"""
向量库管理模块。

职责:
  - 根据配置创建 Embedding 模型（支持阿里云百炼 API 和本地 sentence-transformers）
  - 初始化/加载 Chroma 向量库（持久化模式）
  - 文档去重入库（基于 chunk_id）
  - 获取配置好的 Retriever
"""

from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_embedding_model(config: dict):
    """
    根据配置创建 Embedding 模型。

    支持两种模式:
      1. API 模式（阿里云百炼）: 通过 OpenAIEmbeddings + 自定义 api_base 调用
      2. 本地模式（预留）: 通过 sentence-transformers 本地加载

    Args:
        config: embedding 段配置字典，需包含:
            - api_key
            - api_base
            - model_name

    Returns:
        OpenAIEmbeddings 或其他 LangChain 兼容的 Embedding 实例。

    用法:
        embedder = get_embedding_model(config["embedding"])
    """
    emb_config = config["embedding"]
    # 判断是本地模式还是 API 模式：本地模式下 api_base 可设置为 "local"
    if emb_config.get("api_base", "").strip() == "local":
        # 预留本地 sentence-transformers 切换
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = emb_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        return OpenAIEmbeddings(
            model=emb_config.get("model_name", "text-embedding-v3"),
            api_key=emb_config.get("api_key"),
            base_url=emb_config.get("api_base"),
        )


def get_vectorstore(config: dict, embedder=None) -> Chroma:
    """
    初始化或加载 Chroma 向量库（持久化模式）。

    Args:
        config: 完整配置字典（需包含 chroma 段）。
        embedder: Embedding 实例，若为 None 则自动创建。

    Returns:
        Chroma 向量库实例。

    用法:
        vectorstore = get_vectorstore(config)
    """
    if embedder is None:
        embedder = get_embedding_model(config)

    chroma_config = config["chroma"]
    persist_dir = chroma_config["persist_dir"]
    collection_name = chroma_config.get("collection_name", "course_materials")

    # 确保持久化目录存在
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )

    return vectorstore


def get_existing_chunk_ids(vectorstore: Chroma) -> set:
    """
    获取向量库中已存在的 chunk_id 集合（用于去重）。

    Args:
        vectorstore: Chroma 向量库实例。

    Returns:
        已存在的 chunk_id 集合。若集合为空，返回空集。
    """
    try:
        results = vectorstore.get()
        if results and results["metadatas"]:
            return {m.get("chunk_id", "") for m in results["metadatas"] if m.get("chunk_id")}
    except Exception:
        pass
    return set()


def add_documents(
    vectorstore: Chroma,
    documents: List[Document],
    skip_existing: bool = True,
) -> int:
    """
    向向量库添加文档，支持去重。

    去重逻辑: 检查文档的 chunk_id 是否已存在于向量库中。

    Args:
        vectorstore: Chroma 向量库实例。
        documents: 待添加的 Document 列表。
        skip_existing: 是否跳过已存在的文档（True=去重，False=全量添加）。

    Returns:
        实际新增的文档数量。

    用法:
        added = add_documents(vectorstore, docs)
        print(f"新增 {added} 个文本块")
    """
    if not documents:
        return 0

    if skip_existing:
        existing_ids = get_existing_chunk_ids(vectorstore)
        new_docs = []
        skipped = 0
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id and chunk_id in existing_ids:
                skipped += 1
            else:
                new_docs.append(doc)

        if skipped > 0:
            import logging
            logging.getLogger("course_qa").info(
                f"去重: 跳过 {skipped} 个已存在的文本块，新增 {len(new_docs)} 个"
            )

        if not new_docs:
            return 0
        documents = new_docs

    # 批量添加（Chroma 自动生成 embedding）
    vectorstore.add_documents(documents)
    return len(documents)


def get_retriever(vectorstore: Chroma, top_k: int = 8):
    """
    获取配置了检索参数的 Retriever。

    Args:
        vectorstore: Chroma 向量库实例。
        top_k: 检索返回的候选文档数量。

    Returns:
        Chroma Retriever 实例。

    用法:
        retriever = get_retriever(vectorstore, top_k=8)
        docs = retriever.invoke("什么是堆排序？")
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever
```

- [ ] **Step 2: 验证模块**

```bash
python -c "from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever; print('✓ vectorstore 模块正常')"
```

- [ ] **Step 3: 提交**

```bash
git add src/vectorstore.py
git commit -m "feat: 添加 vectorstore 模块（Chroma 管理+去重）"
```

---

### Task 5: retriever.py — 检索与重排序

**Files:**
- Create: `src/retriever.py`

- [ ] **Step 1: 编写完整模块**

```python
"""
检索与重排序模块。

流程:
  1. 使用 Chroma Retriever 检索 top_k 个候选文档
  2. 使用 CrossEncoder 对候选文档重新打分
  3. 返回 top_n 个最相关文档

CrossEncoder 模型首次运行时自动下载至 cache_dir。
"""

import logging
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document

logger = logging.getLogger("course_qa")


def load_cross_encoder(model_name: str, cache_dir: str = "./models"):
    """
    加载 CrossEncoder 模型（带缓存处理）。

    Args:
        model_name: HuggingFace 模型名称，如 'cross-encoder/ms-marco-MiniLM-L-4-v2'。
        cache_dir: 模型缓存目录。

    Returns:
        CrossEncoder 实例。

    Raises:
        ImportError: sentence-transformers 未安装时抛出。

    用法:
        ce = load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-4-v2", "./models")
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers 未安装，请运行: pip install sentence-transformers"
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载 CrossEncoder 模型: {model_name}")
    model = CrossEncoder(model_name, cache_folder=str(cache_path))
    logger.info("CrossEncoder 模型加载完成")

    return model


def _rerank_with_cross_encoder(
    query: str,
    documents: List[Document],
    cross_encoder,
    top_n: int = 4,
) -> List[Document]:
    """
    使用 CrossEncoder 对候选文档重新排序。

    Args:
        query: 用户问题。
        documents: 候选文档列表。
        cross_encoder: CrossEncoder 实例。
        top_n: 保留的文档数量。

    Returns:
        按相关性降序排列的 top_n 个 Document。
    """
    if not documents:
        return []

    # 构造 (query, doc_content) 对
    pairs = [(query, doc.page_content) for doc in documents]

    # CrossEncoder 打分
    scores = cross_encoder.predict(pairs)

    # 按分数降序排列
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 取 top_n
    top_docs = [doc for _, doc in scored_docs[:top_n]]

    # 将 CrossEncoder 分数写入 metadata 供引用
    for score, doc in scored_docs[:top_n]:
        doc.metadata["rerank_score"] = float(score)

    logger.debug(
        f"重排序: {len(documents)} → {len(top_docs)} 个文档, "
        f"最高分={scored_docs[0][0]:.4f}"
    )

    return top_docs


def retrieve_and_rerank(
    query: str,
    retriever,
    cross_encoder,
    top_n: int = 4,
) -> Tuple[List[Document], List[Document]]:
    """
    执行检索 + 重排序的完整流程。

    Args:
        query: 用户问题。
        retriever: LangChain Retriever 实例。
        cross_encoder: CrossEncoder 实例。
        top_n: 重排序后保留的文档数。

    Returns:
        (reranked_docs, raw_docs): 重排序后的文档列表和原始检索结果列表。

    用法:
        docs, raw = retrieve_and_rerank("什么是死锁？", retriever, ce, top_n=4)
        for doc in docs:
            print(f"[{doc.metadata['source']}] {doc.page_content[:100]}...")
    """
    # Step 1: 基础检索
    raw_docs = retriever.invoke(query)
    logger.debug(f"检索: 召回 {len(raw_docs)} 个候选文档")

    # Step 2: 重排序
    if cross_encoder is not None and len(raw_docs) > top_n:
        reranked = _rerank_with_cross_encoder(query, raw_docs, cross_encoder, top_n)
        return reranked, raw_docs
    else:
        # 候选数不足 top_n 时，全部保留
        return raw_docs[:top_n], raw_docs


# 兼容 LangChain ContextualCompressionRetriever 的封装（可选使用）
def create_compression_retriever(retriever, cross_encoder, config: dict):
    """
    使用 LangChain 的 ContextualCompressionRetriever 封装重排序。

    这是 retrieve_and_rerank 的替代方案，适用于希望完全走 LangChain 链的场景。

    Args:
        retriever: LangChain Retriever。
        cross_encoder: CrossEncoder 实例。
        config: 检索配置段。

    Returns:
        ContextualCompressionRetriever 实例。
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=config.get("rerank_top_n", 4),
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
```

- [ ] **Step 2: 验证模块**

```bash
python -c "from src.retriever import retrieve_and_rerank, load_cross_encoder; print('✓ retriever 模块正常')"
```

- [ ] **Step 3: 提交**

```bash
git add src/retriever.py
git commit -m "feat: 添加 retriever 模块（检索+CrossEncoder 重排序）"
```

---

### Task 6: chain.py — 对话链与 Prompt

**Files:**
- Create: `src/chain.py`

- [ ] **Step 1: 编写完整模块**

```python
"""
对话链模块。

职责:
  - 构建包含学术诚信规则的 System Prompt
  - 创建 ConversationalRetrievalChain，整合检索、记忆和自定义 Prompt
  - 支持 Few-shot 示例引导模型行为
"""

import logging
from typing import Tuple

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder

logger = logging.getLogger("course_qa")

# ============================================================
# System Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业课程答疑助手，服务于软件工程专业核心课程（数据结构、操作系统、计算机网络、软件工程概论等）。

## 核心规则（严格遵循，优先级从上到下）

### 1. 以知识库为准
- 优先根据下方「参考知识库片段」回答问题。
- 知识库未涉及的内容，如实告知："根据课程资料暂未找到答案，建议查阅教材或咨询老师。"
- 严禁编造不存在于知识库中的具体数据、公式或结论。

### 2. 学术诚信（三级分类）
| 场景 | 允许 | 禁止 |
|------|------|------|
| 直接求作业答案/完整代码 | 仅给出概念提示、方向引导 | 完整代码、可直接提交的答案、逐行详解 |
| 问思路、不知如何开始 | 分析解题思路、伪代码、方法对比 | 完整可运行代码 |
| 问概念、原理、知识点 | 完整解释、示例代码片段、图解说明 | 无 |

### 3. 回答格式
按以下结构输出回答：
```
【依据】引用知识库中的关键信息（1-2句话概括）。
【解答】给出你的回答。
【来源】列出引用的文件名或段落编号。
```

### 4. 补充规则
- 若知识库信息不足以回答，请在【解答】中说明原因并给出学习建议。
- 对不确定的内容，标注"以上分析仅供参考，请以教材为准"。
- 回答简洁、准确，避免冗长铺垫。

---

## Few-shot 示例

### 示例 1（概念问题）
用户: 什么是死锁？
【依据】知识库"操作系统-第03章-进程管理.md"指出：死锁是指两个或多个进程互相等待对方释放资源而陷入无限等待的状态，产生条件包括互斥、持有并等待、非抢占、循环等待。
【解答】死锁是两个或多个进程因循环等待资源而无法推进的状态。四个必要条件：互斥→持有并等待→非抢占→循环等待。破坏任一条件即可预防死锁。
【来源】操作系统-第03章-进程管理.md

### 示例 2（思路求助）
用户: 这道二叉树遍历题我完全没有思路，能帮帮我吗？
【依据】知识库"数据结构-第04章-树与二叉树.md"介绍了二叉树的前序、中序、后续遍历的递归定义。
【解答】二叉树遍历题的通用思路：1) 先明确题目要求的是哪种遍历（前/中/后序）；2) 写出递归三要素：终止条件→处理当前节点→递归子树；3) 用一个小例子手动走一遍验证逻辑。递归伪代码：
```
def traverse(root):
    if root is None: return
    # 前序: 先处理 root
    traverse(root.left)
    # 中序: 在这里处理 root
    traverse(root.right)
    # 后序: 最后处理 root
```
以上分析仅供参考，请以教材为准。
【来源】数据结构-第04章-树与二叉树.md

### 示例 3（知识库无法回答）
用户: 某篇论文里的最新量子计算算法是怎么实现的？
【依据】知识库中未找到与"量子计算算法"相关的内容。
【解答】根据课程资料暂未找到答案。当前知识库覆盖数据结构、操作系统、计算机网络等基础课程，不包含量子计算相关内容。建议查阅相关学术论文或咨询老师。
【来源】无
"""


def get_llm(config: dict):
    """
    根据配置创建 LLM 实例。

    Args:
        config: 完整配置字典，使用 config["llm"] 段。

    Returns:
        ChatOpenAI 实例（配置为 DeepSeek 兼容接口）。
    """
    llm_config = config["llm"]
    return ChatOpenAI(
        model=llm_config.get("model_name", "deepseek-chat"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("api_base"),
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 2048),
    )


def create_qa_chain(
    retriever,
    memory: ConversationBufferMemory,
    config: dict,
):
    """
    创建整合检索、记忆和自定义 Prompt 的对话链。

    使用 LangChain 的 ConversationalRetrievalChain，将检索到的知识库片段
    与对话历史、System Prompt 一同送入 LLM。

    Args:
        retriever: LangChain Retriever 实例（可以是基础 Retriever 或压缩 Retriever）。
        memory: ConversationBufferMemory 实例。
        config: 完整配置字典。

    Returns:
        ConversationalRetrievalChain 实例，可直接调用 .invoke({"question": "..."})。

    用法:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        chain = create_qa_chain(retriever, memory, config)
        result = chain.invoke({"question": "什么是死锁？"})
        print(result["answer"])
    """
    llm = get_llm(config)

    # 构建 ChatPromptTemplate
    # 使用 MessagesPlaceholder 为 chat_history 预留位置
    messages = [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # 创建 ConversationalRetrievalChain
    # combine_docs_chain_kwargs 用于自定义 Prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    logger.info("对话链创建完成")
    return chain


def create_memory(max_turns: int = 4) -> ConversationBufferMemory:
    """
    创建对话记忆实例。

    Args:
        max_turns: 保留的最大对话轮数。但 ConversationBufferMemory 本身不限制轮数，
                   通过记忆管理在 CLI 层控制（手动 truncate）。

    Returns:
        ConversationBufferMemory 实例。

    用法:
        memory = create_memory(max_turns=4)
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
    )


def format_source_documents(source_docs: list) -> str:
    """
    格式化来源文档引用信息。

    Args:
        source_docs: ConversationalRetrievalChain 返回的 source_documents。

    Returns:
        格式化的引用字符串。

    用法:
        sources = format_source_documents(result["source_documents"])
        print(sources)
    """
    if not source_docs:
        return "无来源引用"

    lines = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score")
        score_str = f" (相关度: {score:.3f})" if score else ""
        key = f"{source}#p{page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  • {source} 第{page}页{score_str}")

    return "\n".join(lines)
```

- [ ] **Step 2: 验证模块**

```bash
python -c "from src.chain import create_qa_chain, create_memory, SYSTEM_PROMPT; print('✓ chain 模块正常')"
```

- [ ] **Step 3: 提交**

```bash
git add src/chain.py
git commit -m "feat: 添加 chain 模块（对话链+学术诚信 Prompt+记忆）"
```

---

### Task 7: build_kb.py — 知识库构建脚本

**Files:**
- Create: `build_kb.py`

- [ ] **Step 1: 编写完整脚本**

```python
#!/usr/bin/env python3
"""
知识库构建脚本。

遍历指定目录下所有支持的文档（PDF/Word/Markdown），
加载、分块、生成向量嵌入并存入 Chroma 向量库。

用法:
    python build_kb.py --input_dir ./data/documents --config config.yaml
    python build_kb.py --input_dir ./data/documents --config config.yaml --chunk_size 800 --chunk_overlap 150
    python build_kb.py --input_dir ./data/documents --config config.yaml --force  # 强制全量重建
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 src 模块可导入
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, mask_key
from src.loader import load_documents
from src.vectorstore import get_embedding_model, get_vectorstore, add_documents


def main():
    parser = argparse.ArgumentParser(
        description="构建课程知识库 — 将文档向量化存入 Chroma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_kb.py --input_dir ./data/documents
  python build_kb.py --input_dir ./data/documents --config my_config.yaml
  python build_kb.py --input_dir ./data/documents --force  # 跳过去重，全量重建
        """,
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="文档目录路径（将递归遍历子目录）",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="文本分块大小（默认: 1000）",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="文本分块重叠大小（默认: 200）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制全量重建，不进行去重检查",
    )

    args = parser.parse_args()

    # ---- 加载配置 ----
    print("=" * 60)
    print("  课程知识库构建工具")
    print("=" * 60)

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[错误] 加载配置失败: {e}")
        sys.exit(1)

    # ---- 初始化日志 ----
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="build_kb",
        log_file=log_config.get("file", "logs/build_kb.log"),
        level=log_config.get("level", "INFO"),
    )

    # 脱敏显示 API Key
    llm_key = config.get("llm", {}).get("api_key", "")
    emb_key = config.get("embedding", {}).get("api_key", "")
    logger.info(f"LLM API Key: {mask_key(llm_key)}")
    logger.info(f"Embedding API Key: {mask_key(emb_key)}")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"分块大小: {args.chunk_size}, 重叠: {args.chunk_overlap}")

    # ---- 加载文档 ----
    print("\n[1/3] 加载文档...")
    try:
        documents = load_documents(
            input_dir=args.input_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        sys.exit(1)

    logger.info(f"加载完成: {len(documents)} 个文本块")
    print(f"  ✓ 共加载 {len(documents)} 个文本块")

    # 统计文件来源
    sources = set(doc.metadata.get("source", "未知") for doc in documents)
    print(f"  ✓ 来源文件数: {len(sources)}")
    for src in sorted(sources):
        print(f"    - {src}")

    # ---- 初始化向量库 ----
    print("\n[2/3] 初始化向量库...")
    try:
        embedder = get_embedding_model(config)
        vectorstore = get_vectorstore(config, embedder)
    except Exception as e:
        logger.error(f"初始化向量库失败: {e}")
        sys.exit(1)
    print("  ✓ Chroma 向量库已就绪")

    # ---- 添加文档 ----
    print("\n[3/3] 生成嵌入并写入向量库...")
    try:
        added = add_documents(
            vectorstore=vectorstore,
            documents=documents,
            skip_existing=not args.force,
        )
    except Exception as e:
        logger.error(f"写入向量库失败: {e}")
        sys.exit(1)

    if added == 0:
        print("  ✓ 所有文档已存在，无需更新")
    else:
        print(f"  ✓ 新增 {added} 个文本块")
        persist_dir = config.get("chroma", {}).get("persist_dir", "./data/chroma_db")
        print(f"  ✓ 向量库已持久化至: {persist_dir}")

    print("\n" + "=" * 60)
    print("  构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add build_kb.py
git commit -m "feat: 添加 build_kb.py 知识库构建脚本"
```

---

### Task 8: qa_cli.py — 命令行交互问答

**Files:**
- Create: `qa_cli.py`

- [ ] **Step 1: 编写完整脚本**

```python
#!/usr/bin/env python3
"""
命令行交互问答脚本。

启动后进入交互式循环，用户可以输入课程问题，
系统执行 检索→重排序→生成 链路并打印答案。

支持特殊命令:
  /reset  - 清空对话记忆
  /sources - 切换是否显示详细引用
  /exit   - 退出程序

用法:
    python qa_cli.py
    python qa_cli.py --config my_config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, mask_key
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.retriever import load_cross_encoder, retrieve_and_rerank, create_compression_retriever
from src.chain import create_qa_chain, create_memory, format_source_documents

# ANSI 颜色（跨平台）
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = YELLOW = RED = RESET = ""


def print_banner():
    """打印欢迎横幅。"""
    print(CYAN + "=" * 60)
    print("   📚 专业课程答疑智能体")
    print("   输入问题开始问答，/exit 退出，/reset 清空记忆")
    print("=" * 60 + RESET)


def print_sources(source_docs):
    """打印来源引用。"""
    if not source_docs:
        print(YELLOW + "  (无知识库来源引用)" + RESET)
        return
    print(GREEN + "\n📖 参考来源:" + RESET)
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score")
        score_str = f" [相关度: {score:.3f}]" if score else ""
        key = f"{source}#p{page}"
        if key not in seen:
            seen.add(key)
            print(f"  • {source}  第{page}页{score_str}")


def main():
    parser = argparse.ArgumentParser(
        description="专业课程答疑 — 命令行交互问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--show_raw",
        action="store_true",
        help="显示检索到的原始文本片段（调试用）",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"{RED}[错误] 加载配置失败: {e}{RESET}")
        sys.exit(1)

    # ---- 初始化日志 ----
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="qa_cli",
        log_file=log_config.get("file", "logs/qa.log"),
        level=log_config.get("level", "INFO"),
    )
    logger.info(f"LLM API Key: {mask_key(config['llm']['api_key'])}")
    logger.info(f"Embedding API Key: {mask_key(config['embedding']['api_key'])}")

    # ---- 初始化组件 ----
    print(CYAN + "正在初始化系统组件..." + RESET)

    try:
        # Embedding + 向量库
        embedder = get_embedding_model(config)
        vectorstore = get_vectorstore(config, embedder)
        retriever = get_retriever(vectorstore, top_k=config["retrieval"]["top_k"])

        # CrossEncoder
        reranker_config = config["reranker"]
        cross_encoder = load_cross_encoder(
            model_name=reranker_config["model_name"],
            cache_dir=reranker_config.get("cache_dir", "./models"),
        )

        # 对话链（压缩型 retriever 嵌入链内）
        compression_retriever = create_compression_retriever(
            retriever, cross_encoder, config["retrieval"]
        )
        memory = create_memory(max_turns=config["memory"].get("max_turns", 4))
        qa_chain = create_qa_chain(compression_retriever, memory, config)

    except Exception as e:
        print(f"{RED}[错误] 初始化失败: {e}{RESET}")
        logger.error(f"初始化失败: {e}", exc_info=True)
        sys.exit(1)

    print(GREEN + "✓ 系统就绪" + RESET)
    logger.info("系统初始化完成")

    # ---- 交互循环 ----
    print_banner()
    show_raw = args.show_raw

    while True:
        try:
            user_input = input(CYAN + "\n🧑 你的问题: " + RESET).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input.lower() == "/exit":
            print("再见！")
            break

        if user_input.lower() == "/reset":
            memory.clear()
            print(GREEN + "✓ 对话记忆已清空" + RESET)
            logger.info("用户清空对话记忆")
            continue

        if user_input.lower() == "/sources":
            show_raw = not show_raw
            status = "开启" if show_raw else "关闭"
            print(GREEN + f"✓ 详细来源显示已{status}" + RESET)
            continue

        # ---- 执行问答 ----
        logger.info(f"用户问题: {user_input}")
        print(YELLOW + "⏳ 正在检索知识库..." + RESET)

        try:
            result = qa_chain.invoke({"question": user_input})
        except Exception as e:
            print(f"{RED}[错误] 问答失败: {e}{RESET}")
            logger.error(f"问答失败: {e}", exc_info=True)
            continue

        answer = result.get("answer", "（生成回答失败）")
        source_docs = result.get("source_documents", [])

        # ---- 输出回答 ----
        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60)

        # 来源
        print_sources(source_docs)

        # 调试模式：显示原始片段
        if show_raw and source_docs:
            print(YELLOW + "\n🔍 检索片段（原始）:" + RESET)
            for i, doc in enumerate(source_docs, 1):
                content_preview = doc.page_content[:200].replace("\n", " ")
                source = doc.metadata.get("source", "未知")
                print(f"  [{i}] {source}: {content_preview}...")

        # 日志记录
        sources_summary = [f"{d.metadata.get('source','?')}:{d.metadata.get('page','?')}" for d in source_docs]
        logger.info(f"回答生成完成 | 来源: {', '.join(sources_summary[:4])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add qa_cli.py
git commit -m "feat: 添加 qa_cli.py 命令行问答脚本"
```

---

### Task 9: 配置真实 config.yaml 并跑通集成测试

**Files:**
- Create: `config.yaml`（从 example 复制并填入真实 Key）
- Create: `data/documents/demo.md`（测试用 Markdown 文档）

- [ ] **Step 1: 创建测试文档**

创建 `data/documents/demo.md`:

```markdown
# 数据结构测试文档

## 堆排序

堆排序（Heap Sort）是一种基于二叉堆数据结构的比较排序算法。它的基本思想是将待排序序列构造成一个大顶堆，此时整个序列的最大值就是堆顶的根节点。将其与末尾元素进行交换，此时末尾就为最大值。然后将剩余 n-1 个元素重新构造成一个堆，重复此过程直到有序。

堆排序的时间复杂度为 O(nlogn)，空间复杂度为 O(1)，是一种不稳定排序算法。

## 死锁

死锁（Deadlock）是指两个或两个以上的进程在执行过程中，由于竞争资源或者由于彼此通信而造成的一种阻塞现象。若无外力作用，它们都将无法推进下去。

死锁产生的四个必要条件是：
1. 互斥条件
2. 请求与保持条件
3. 不可抢占条件
4. 循环等待条件
```

- [ ] **Step 2: 创建 config.yaml**

从 config.example.yaml 复制并填入 API Key，通过以下方式确保不提交真实 Key:

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入真实 API Key
```

- [ ] **Step 3: 构建知识库**

```bash
python build_kb.py --input_dir ./data/documents --config config.yaml
```

预期输出：成功加载并写入文本块。

- [ ] **Step 4: 启动问答测试**

```bash
python qa_cli.py --config config.yaml
```

测试以下问题：
1. "什么是堆排序？它的时间复杂度是多少？"
2. "死锁的四个必要条件是什么？"
3. "/reset" → 确认记忆清空 → "/exit"

- [ ] **Step 5: 验证日志**

```bash
ls logs/
cat logs/qa.log
```

确认日志包含问答记录和检索来源。

- [ ] **Step 6: 提交**

```bash
git add data/documents/demo.md
git commit -m "feat: 添加测试文档 demo.md"
```

---

## 计划自审清单

- [x] Spec 覆盖：所有 MVP 功能（build_kb、qa_cli、六模块、记忆、Prompt、日志）均有对应任务
- [x] 无占位符：所有步骤包含完整代码，无 TBD/TODO
- [x] 类型一致性：模块间接口（load_config → dict, retriever → List[Document], chain → ConversationalRetrievalChain）前后一致
- [x] 第二阶段（eval.py、Web UI）不在本次计划范围内，符合 MVP 边界
