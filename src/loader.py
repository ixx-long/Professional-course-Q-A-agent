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

import logging
import os
import hashlib
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 支持的文件扩展名 → Loader 映射
LOADER_MAP: Dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


# 已知的课程名称关键词（用于从路径自动检测课程标签）
_COURSE_KEYWORDS = ["数据结构", "操作系统", "计算机网络", "软件工程", "计算机组成", "数据库", "编译原理"]

def _detect_course(file_path: Path) -> str | None:
    """从文件路径中自动检测所属课程。

    检查路径的每一级目录名是否包含已知课程关键词。
    """
    for part in file_path.parts:
        for kw in _COURSE_KEYWORDS:
            if kw in part:
                return kw
    return None

def _load_text_with_fallback(file_path: str, loader_cls: type) -> list:
    """加载文本文件，UTF-8 优先，失败回退 GBK/GB2312。"""
    encodings = ["utf-8", "gbk", "gb2312"]
    for enc in encodings:
        try:
            loader = loader_cls(file_path, encoding=enc)
            docs = loader.load()
            logger.debug(f"文件 {file_path} 使用 {enc} 编码加载成功")
            return docs
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 所有常见编码失败：使用 Python open + errors=replace 兜底
    logger.warning(f"文件 {file_path} 编码检测失败，使用 utf-8 + errors=replace 兜底")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": Path(file_path).name})]


def _get_loader(file_path: Path) -> type:
    """根据文件扩展名返回对应的 LangChain Loader 类。

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
    # PDF 用默认参数，文本类 Loader 尝试 UTF-8 优先，失败回退 GBK
    if file_path.suffix.lower() == ".pdf":
        loader = loader_cls(str(file_path))
        docs = loader.load()
    else:
        docs = _load_text_with_fallback(str(file_path), loader_cls)

    is_pdf = file_path.suffix.lower() == ".pdf"

    for i, doc in enumerate(docs):
        doc.metadata["source"] = file_path.name
        doc.metadata["page"] = i + 1 if is_pdf else 1

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
                    # 自动检测课程标签（从目录结构）
                    course = _detect_course(file_path)
                    if course:
                        for doc in docs:
                            doc.metadata["course"] = course
                    all_docs.extend(docs)
                except Exception as e:
                    # 单个文件加载失败不中断整体流程
                    logger.warning(f"跳过文件 {file_path}: {e}")

    if not all_docs:
        raise RuntimeError(f"在 {input_dir} 中未找到任何支持的文档文件")

    # 分块
    chunks = split_documents(all_docs, chunk_size, chunk_overlap)
    return chunks
