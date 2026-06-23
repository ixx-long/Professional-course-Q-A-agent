"""
检索模块单元测试。
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.retriever import (
    load_cross_encoder,
    _rerank_with_cross_encoder,
    retrieve_and_rerank,
    create_compression_retriever,
)


class TestLoadCrossEncoder:
    """load_cross_encoder 单元测试。"""

    def test_import_error(self):
        """测试 sentence-transformers 未安装时抛出 ImportError。"""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with pytest.raises(ImportError, match="sentence-transformers 未安装"):
                load_cross_encoder("test-model")

    def test_cached_model_offline_load(self):
        """测试缓存模型离线加载。"""
        # 模拟 Path 链式调用：cache_path / name → model_dir
        mock_model_dir = Mock()
        mock_model_dir.exists.return_value = True
        mock_model_dir.rglob.return_value = [Mock()]  # 有 safetensors 文件

        mock_cache_path = Mock()
        mock_cache_path.__truediv__ = Mock(return_value=mock_model_dir)
        mock_cache_path.mkdir = Mock()

        with patch('src.retriever.Path', return_value=mock_cache_path):
            with patch('sentence_transformers.CrossEncoder') as mock_ce:
                mock_ce.return_value = Mock()
                result = load_cross_encoder("test-model", cache_dir="/tmp/models")
                assert result is not None

    def test_download_timeout(self):
        """测试模型下载超时。"""
        from src.errors import RequestTimeoutError
        
        # 模拟 Path 链式调用：cache_path / name → model_dir（不存在，触发下载）
        mock_model_dir = Mock()
        mock_model_dir.exists.return_value = False

        mock_cache_path = Mock()
        mock_cache_path.__truediv__ = Mock(return_value=mock_model_dir)
        mock_cache_path.mkdir = Mock()

        with patch('src.retriever.Path', return_value=mock_cache_path):
            with patch('sentence_transformers.CrossEncoder') as mock_ce:
                # 模拟下载失败（返回 None）
                mock_ce.return_value = None
                
                with pytest.raises(RequestTimeoutError, match="下载超时"):
                    load_cross_encoder("test-model", download_timeout=1)


class TestRerankWithCrossEncoder:
    """_rerank_with_cross_encoder 单元测试。"""

    def test_empty_documents(self):
        """测试空文档列表。"""
        mock_ce = Mock()
        result = _rerank_with_cross_encoder("测试", [], mock_ce)
        assert result == []

    def test_rerank_top_n(self):
        """测试重排序返回 top_n。"""
        mock_ce = Mock()
        mock_ce.predict.return_value = [0.9, 0.5, 0.7, 0.3]
        
        docs = [
            Document(page_content="文档1"),
            Document(page_content="文档2"),
            Document(page_content="文档3"),
            Document(page_content="文档4"),
        ]
        
        result = _rerank_with_cross_encoder("测试", docs, mock_ce, top_n=2)
        
        assert len(result) == 2
        assert result[0].metadata["rerank_score"] == 0.9
        assert result[1].metadata["rerank_score"] == 0.7

    def test_rerank_preserves_metadata(self):
        """测试重排序保留原始元数据。"""
        mock_ce = Mock()
        mock_ce.predict.return_value = [0.8, 0.6]
        
        docs = [
            Document(page_content="文档1", metadata={"source": "file1.pdf", "page": 1}),
            Document(page_content="文档2", metadata={"source": "file2.pdf", "page": 2}),
        ]
        
        result = _rerank_with_cross_encoder("测试", docs, mock_ce, top_n=2)
        
        assert result[0].metadata["source"] == "file1.pdf"
        assert result[0].metadata["page"] == 1
        assert result[0].metadata["rerank_score"] == 0.8


class TestRetrieveAndRerank:
    """retrieve_and_rerank 单元测试。"""

    def test_retrieve_and_rerank(self):
        """测试检索 + 重排序完整流程。"""
        mock_retriever = Mock()
        mock_ce = Mock()
        
        raw_docs = [
            Document(page_content="文档1", metadata={"source": "file1.pdf"}),
            Document(page_content="文档2", metadata={"source": "file2.pdf"}),
            Document(page_content="文档3", metadata={"source": "file3.pdf"}),
        ]
        mock_retriever.invoke.return_value = raw_docs
        mock_ce.predict.return_value = [0.9, 0.5, 0.7]
        
        reranked, raw = retrieve_and_rerank("测试", mock_retriever, mock_ce, top_n=2)
        
        assert len(reranked) == 2
        assert len(raw) == 3
        assert reranked[0].metadata["rerank_score"] == 0.9

    def test_retrieve_without_rerank(self):
        """测试无重排序模型时直接返回。"""
        mock_retriever = Mock()
        raw_docs = [
            Document(page_content="文档1"),
            Document(page_content="文档2"),
        ]
        mock_retriever.invoke.return_value = raw_docs
        
        reranked, raw = retrieve_and_rerank("测试", mock_retriever, None, top_n=4)
        
        assert reranked == raw_docs[:4]
        assert raw == raw_docs

    def test_retrieve_insufficient_candidates(self):
        """测试候选文档不足 top_n 时仍打分。"""
        mock_retriever = Mock()
        mock_ce = Mock()
        
        raw_docs = [
            Document(page_content="文档1"),
            Document(page_content="文档2"),
        ]
        mock_retriever.invoke.return_value = raw_docs
        mock_ce.predict.return_value = [0.8, 0.6]
        
        reranked, raw = retrieve_and_rerank("测试", mock_retriever, mock_ce, top_n=5)
        
        assert len(reranked) == 2
        assert raw_docs[0].metadata["rerank_score"] == 0.8
        assert raw_docs[1].metadata["rerank_score"] == 0.6


class TestCreateCompressionRetriever:
    """create_compression_retriever 单元测试。"""

    def test_create_compression_retriever(self):
        """测试创建压缩 Retriever。"""
        mock_retriever = Mock()
        mock_ce = Mock()
        config = {"rerank_top_n": 3}
        
        with patch('src.retriever.CrossEncoderReranker') as mock_reranker:
            with patch('src.retriever.ContextualCompressionRetriever') as mock_compression:
                result = create_compression_retriever(mock_retriever, mock_ce, config)
                
                mock_reranker.assert_called_once_with(model=mock_ce, top_n=3)
                mock_compression.assert_called_once()
                assert result is not None

    def test_default_top_n(self):
        """测试默认 top_n 值。"""
        mock_retriever = Mock()
        mock_ce = Mock()
        config = {}
        
        with patch('src.retriever.CrossEncoderReranker') as mock_reranker:
            with patch('src.retriever.ContextualCompressionRetriever'):
                create_compression_retriever(mock_retriever, mock_ce, config)
                
                mock_reranker.assert_called_once_with(model=mock_ce, top_n=4)
