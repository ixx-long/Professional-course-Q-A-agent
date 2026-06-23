"""
向量库模块单元测试。
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.vectorstore import (
    BailianEmbeddings,
    get_embedding_model,
    get_vectorstore,
    get_existing_chunk_ids,
    add_documents,
    get_retriever,
)


class TestBailianEmbeddings:
    """BailianEmbeddings 单元测试。"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI 客户端。"""
        with patch('src.vectorstore.OpenAI') as mock:
            yield mock

    def test_embed_documents_single_batch(self, mock_openai_client):
        """测试单批次文档嵌入。"""
        # 准备 Mock
        mock_client = Mock()
        mock_openai_client.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        # 执行
        embedder = BailianEmbeddings(
            api_key="test_key",
            base_url="https://test.api",
            model="test-model",
            batch_size=25
        )
        result = embedder.embed_documents(["测试文本"])
        
        # 验证
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_embed_documents_multiple_batches(self, mock_openai_client):
        """测试多批次文档嵌入。"""
        mock_client = Mock()
        mock_openai_client.return_value = mock_client
        
        # Mock 返回不同的嵌入
        def create_embedding(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in kwargs['input']]
            return mock_resp
        
        mock_client.embeddings.create.side_effect = create_embedding
        
        embedder = BailianEmbeddings(
            api_key="test_key",
            base_url="https://test.api",
            batch_size=2
        )
        texts = ["文本1", "文本2", "文本3"]
        result = embedder.embed_documents(texts)
        
        # 验证
        assert len(result) == 3
        assert mock_client.embeddings.create.call_count == 2  # 2 个批次

    def test_embed_query_success(self, mock_openai_client):
        """测试查询嵌入成功。"""
        mock_client = Mock()
        mock_openai_client.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5, 0.6, 0.7])]
        mock_client.embeddings.create.return_value = mock_response
        
        embedder = BailianEmbeddings(
            api_key="test_key",
            base_url="https://test.api"
        )
        result = embedder.embed_query("测试查询")
        
        assert result == [0.5, 0.6, 0.7]

    def test_embed_query_retry_on_failure(self, mock_openai_client):
        """测试查询嵌入失败时重试。"""
        mock_client = Mock()
        mock_openai_client.return_value = mock_client
        
        # 前两次失败，第三次成功
        mock_client.embeddings.create.side_effect = [
            Exception("API 错误"),
            Exception("API 错误"),
            Mock(data=[Mock(embedding=[0.1, 0.2])])
        ]
        
        embedder = BailianEmbeddings(
            api_key="test_key",
            base_url="https://test.api"
        )
        
        with patch('src.vectorstore.time.sleep'):  # 跳过 sleep
            result = embedder.embed_query("测试")
        
        assert result == [0.1, 0.2]
        assert mock_client.embeddings.create.call_count == 3

    def test_embed_query_max_retries_exceeded(self, mock_openai_client):
        """测试查询嵌入超过最大重试次数。"""
        mock_client = Mock()
        mock_openai_client.return_value = mock_client
        
        mock_client.embeddings.create.side_effect = Exception("持续错误")
        
        embedder = BailianEmbeddings(
            api_key="test_key",
            base_url="https://test.api"
        )
        
        with patch('src.vectorstore.time.sleep'):
            with pytest.raises(Exception, match="持续错误"):
                embedder.embed_query("测试")


class TestGetEmbeddingModel:
    """get_embedding_model 单元测试。"""

    def test_api_mode(self):
        """测试 API 模式。"""
        config = {
            "embedding": {
                "api_key": "test_key",
                "api_base": "https://test.api",
                "model_name": "test-model"
            }
        }
        
        with patch('src.vectorstore.BailianEmbeddings') as mock_class:
            result = get_embedding_model(config)
            mock_class.assert_called_once()
            assert result is not None

    def test_local_mode(self):
        """测试本地模式。"""
        config = {
            "embedding": {
                "api_base": "local",
                "model_name": "test-model"
            }
        }
        
        with patch('langchain_huggingface.HuggingFaceEmbeddings') as mock_class:
            result = get_embedding_model(config)
            mock_class.assert_called_once_with(model_name="test-model")
            assert result is not None


class TestGetVectorstore:
    """get_vectorstore 单元测试。"""

    @pytest.fixture
    def mock_chroma(self):
        """Mock Chroma。"""
        with patch('src.vectorstore.Chroma') as mock:
            yield mock

    def test_with_embedder(self, mock_chroma):
        """测试提供 embedder。"""
        config = {
            "chroma": {
                "persist_dir": "/tmp/test_chroma",
                "collection_name": "test_collection"
            }
        }
        embedder = Mock()
        
        result = get_vectorstore(config, embedder)
        
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=embedder,
            persist_directory="/tmp/test_chroma"
        )
        assert result is not None

    def test_without_embedder(self, mock_chroma):
        """测试不提供 embedder（自动创建）。"""
        config = {
            "embedding": {
                "api_key": "test_key",
                "api_base": "https://test.api",
                "model_name": "test-model"
            },
            "chroma": {
                "persist_dir": "/tmp/test_chroma",
                "collection_name": "test_collection"
            }
        }
        
        with patch('src.vectorstore.get_embedding_model') as mock_get_emb:
            mock_get_emb.return_value = Mock()
            result = get_vectorstore(config)
            
            mock_get_emb.assert_called_once_with(config)
            assert result is not None


class TestGetExistingChunkIds:
    """get_existing_chunk_ids 单元测试。"""

    def test_empty_vectorstore(self):
        """测试空向量库。"""
        mock_vs = Mock()
        mock_vs.get.return_value = {"metadatas": []}
        
        result = get_existing_chunk_ids(mock_vs)
        
        assert result == set()

    def test_with_chunk_ids(self):
        """测试有 chunk_id 的向量库。"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "metadatas": [
                {"chunk_id": "chunk_1"},
                {"chunk_id": "chunk_2"},
                {"chunk_id": "chunk_1"}  # 重复
            ]
        }
        
        result = get_existing_chunk_ids(mock_vs)
        
        assert result == {"chunk_1", "chunk_2"}

    def test_missing_chunk_id(self):
        """测试缺少 chunk_id 的元数据。"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "metadatas": [
                {"chunk_id": "chunk_1"},
                {"other_field": "value"}  # 缺少 chunk_id
            ]
        }
        
        result = get_existing_chunk_ids(mock_vs)
        
        assert result == {"chunk_1"}

    def test_database_error(self):
        """测试数据库错误。"""
        mock_vs = Mock()
        mock_vs.get.side_effect = RuntimeError("数据库损坏")
        
        with pytest.raises(RuntimeError, match="无法读取 Chroma 向量库数据"):
            get_existing_chunk_ids(mock_vs)


class TestAddDocuments:
    """add_documents 单元测试。"""

    def test_empty_documents(self):
        """测试空文档列表。"""
        mock_vs = Mock()
        
        result = add_documents(mock_vs, [])
        
        assert result == 0
        mock_vs.add_documents.assert_not_called()

    def test_add_without_dedup(self):
        """测试不去重添加。"""
        mock_vs = Mock()
        docs = [
            Document(page_content="内容1", metadata={"chunk_id": "id1"}),
            Document(page_content="内容2", metadata={"chunk_id": "id2"})
        ]
        
        result = add_documents(mock_vs, docs, skip_existing=False)
        
        assert result == 2
        mock_vs.add_documents.assert_called_once_with(docs)

    def test_add_with_dedup(self):
        """测试去重添加。"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "metadatas": [{"chunk_id": "id1"}]
        }
        
        docs = [
            Document(page_content="内容1", metadata={"chunk_id": "id1"}),  # 已存在
            Document(page_content="内容2", metadata={"chunk_id": "id2"})   # 新增
        ]
        
        result = add_documents(mock_vs, docs, skip_existing=True)
        
        assert result == 1
        mock_vs.add_documents.assert_called_once()
        # 验证只添加了 id2
        added_docs = mock_vs.add_documents.call_args[0][0]
        assert len(added_docs) == 1
        assert added_docs[0].metadata["chunk_id"] == "id2"

    def test_all_documents_exist(self):
        """测试所有文档已存在。"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "metadatas": [{"chunk_id": "id1"}, {"chunk_id": "id2"}]
        }
        
        docs = [
            Document(page_content="内容1", metadata={"chunk_id": "id1"}),
            Document(page_content="内容2", metadata={"chunk_id": "id2"})
        ]
        
        result = add_documents(mock_vs, docs, skip_existing=True)
        
        assert result == 0
        mock_vs.add_documents.assert_not_called()


class TestGetRetriever:
    """get_retriever 单元测试。"""

    def test_default_top_k(self):
        """测试默认 top_k。"""
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        
        result = get_retriever(mock_vs)
        
        mock_vs.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        assert result == mock_retriever

    def test_custom_top_k(self):
        """测试自定义 top_k。"""
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        
        result = get_retriever(mock_vs, top_k=10)
        
        mock_vs.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        assert result == mock_retriever
