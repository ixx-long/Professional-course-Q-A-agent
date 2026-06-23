"""
单元测试 - loader 模块
"""
import pytest
from pathlib import Path
from src.loader import load_documents, _detect_course


def test_detect_course_data_structure():
    """测试数据结构课程检测"""
    # _detect_course 接受 Path 参数，检查目录名是否包含课程关键词
    file_path = Path("/data/数据结构/chapter1/intro.md")
    course = _detect_course(file_path)
    assert course == "数据结构"


def test_detect_course_no_match():
    """测试无匹配课程"""
    file_path = Path("/data/无关目录/intro.md")
    course = _detect_course(file_path)
    assert course is None


def test_load_documents_missing_dir():
    """测试加载不存在的目录"""
    with pytest.raises(FileNotFoundError):
        load_documents("nonexistent_dir")


def test_load_documents_not_a_directory():
    """测试传入文件路径而非目录"""
    # 创建临时测试文件
    test_file = Path("test_temp.txt")
    test_file.write_text("测试内容", encoding="utf-8")
    
    try:
        with pytest.raises(NotADirectoryError):
            load_documents(str(test_file))
    finally:
        test_file.unlink()


def test_load_documents_empty_dir(tmp_path):
    """测试空目录（无支持的文件）"""
    with pytest.raises(RuntimeError, match="未找到任何支持的文档文件"):
        load_documents(str(tmp_path))


def test_load_documents_txt(tmp_path):
    """测试加载文本文件"""
    # 创建临时测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("测试内容", encoding="utf-8")
    
    docs = load_documents(str(tmp_path))
    assert len(docs) > 0
    assert "测试内容" in docs[0].page_content
