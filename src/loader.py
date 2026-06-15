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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
