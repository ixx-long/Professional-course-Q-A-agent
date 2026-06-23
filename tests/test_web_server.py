"""
单元测试 - QAService 关键逻辑模块

使用 mock 隔离 Flask 和 LangChain 外部依赖，测试 QAService 中的纯逻辑方法。
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestValidateImage:
    """图片校验逻辑测试"""

    @pytest.fixture
    def qa_service(self):
        """创建 QAService 实例（不初始化系统）"""
        from src.qa_service import QAService
        service = QAService.__new__(QAService)
        service.config = {}
        service.logger = MagicMock()
        return service

    def test_empty_image(self, qa_service):
        """测试空图片数据"""
        result = qa_service.validate_image("")
        assert result == "图片数据为空或格式错误"

    def test_none_image(self, qa_service):
        """测试 None 图片"""
        result = qa_service.validate_image(None)
        assert result == "图片数据为空或格式错误"

    def test_non_string_image(self, qa_service):
        """测试非字符串图片"""
        result = qa_service.validate_image(12345)
        assert result == "图片数据为空或格式错误"

    def test_oversized_image(self, qa_service):
        """测试超大图片"""
        result = qa_service.validate_image("a" * (10 * 1024 * 1024 + 1))
        assert result == "图片过大（最大 10MB）"

    def test_valid_base64_png(self, qa_service):
        """测试有效的 base64 PNG"""
        import base64
        # 最小有效 PNG（1x1 像素）
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        b64 = base64.b64encode(png_bytes).decode()
        result = qa_service.validate_image(b64)
        assert result is None

    def test_valid_data_url_png(self, qa_service):
        """测试有效的 data URL PNG"""
        import base64
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        b64 = base64.b64encode(png_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"
        result = qa_service.validate_image(data_url)
        assert result is None

    def test_invalid_data_url_format(self, qa_service):
        """测试无效的 data URL 格式"""
        result = qa_service.validate_image("data:image/bmp;base64,abc")
        assert result == "不支持的图片格式（支持 PNG/JPEG/GIF/WebP）"

    def test_invalid_base64_encoding(self, qa_service):
        """测试无效的 base64 编码"""
        result = qa_service.validate_image("not_valid_base64!!!")
        assert result == "图片 base64 编码无效"


class TestExtractSources:
    """来源提取逻辑测试"""

    @pytest.fixture
    def qa_service(self):
        """创建 QAService 实例"""
        from src.qa_service import QAService
        service = QAService.__new__(QAService)
        service.config = {}
        service.logger = MagicMock()
        return service

    def test_empty_sources(self, qa_service):
        """测试空来源列表"""
        result = qa_service.extract_sources([])
        assert result == []

    def test_none_sources(self, qa_service):
        """测试 None 来源"""
        result = qa_service.extract_sources(None)
        assert result == []

    def test_extract_with_metadata(self, qa_service):
        """测试从 Document 提取来源信息"""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="测试内容1",
                metadata={"source": "test.pdf", "page": 1, "rerank_score": 0.95}
            ),
            Document(
                page_content="测试内容2",
                metadata={"source": "test.pdf", "page": 2, "rerank_score": 0.85}
            ),
        ]
        result = qa_service.extract_sources(docs)
        assert len(result) == 2
        assert result[0]["file"] == "test.pdf"
        assert result[0]["page"] == 1
        assert result[0]["score"] == 0.95
        assert result[1]["page"] == 2

    def test_dedup_sources(self, qa_service):
        """测试来源去重"""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="内容1",
                metadata={"source": "test.pdf", "page": 1}
            ),
            Document(
                page_content="内容2",
                metadata={"source": "test.pdf", "page": 1}  # 重复
            ),
        ]
        result = qa_service.extract_sources(docs)
        assert len(result) == 1  # 去重后只有 1 个

    def test_missing_metadata(self, qa_service):
        """测试缺失 metadata 的默认值"""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="内容", metadata={})
        ]
        result = qa_service.extract_sources(docs)
        assert len(result) == 1
        assert result[0]["file"] == "未知"
        assert result[0]["page"] == "N/A"
        assert result[0]["score"] is None


class TestSessionManagement:
    """Session 管理测试"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def session_manager(self, temp_dir):
        """创建 SessionManager 实例（带临时目录）"""
        from src.session_manager import SessionManager
        manager = SessionManager(
            sessions_file=Path(temp_dir) / "sessions.json",
            max_sessions=3,
            max_turns=4,
            save_interval=30
        )
        return manager

    def test_get_session_creates_new(self, session_manager):
        """测试获取不存在的 session 时创建新的"""
        from src.chain import ChatHistory
        session = session_manager.get_session("test_session")
        assert isinstance(session, ChatHistory)
        assert "test_session" in session_manager.sessions

    def test_get_session_returns_existing(self, session_manager):
        """测试获取已存在的 session"""
        from src.chain import ChatHistory
        session_manager.sessions["existing"] = ChatHistory(max_turns=4)
        session = session_manager.get_session("existing")
        assert session is session_manager.sessions["existing"]

    def test_lru_eviction(self, session_manager):
        """测试 LRU 淘汰"""
        from src.chain import ChatHistory
        # 填满 session 上限
        for i in range(3):
            session_manager.sessions[f"session_{i}"] = ChatHistory(max_turns=4)
            session_manager._session_access_times[f"session_{i}"] = float(i)

        # 访问 session_0 使其变为最新
        session_manager._session_access_times["session_0"] = 999.0

        # 添加新 session，应淘汰最久未访问的 session_1
        session_manager.get_session("new_session")
        assert "session_1" not in session_manager.sessions
        assert "session_0" in session_manager.sessions  # 因为刚被访问过

    def test_save_and_load_sessions(self, session_manager):
        """测试 session 持久化和恢复"""
        from src.chain import ChatHistory
        from langchain_core.messages import HumanMessage, AIMessage

        # 添加对话历史
        ch = ChatHistory(max_turns=4)
        ch.add_user("你好")
        ch.add_ai("你好！有什么可以帮你的？")
        session_manager.sessions["test"] = ch

        # 保存
        session_manager.save()
        assert session_manager.SESSIONS_FILE.exists()

        # 清空并重新加载
        session_manager.sessions.clear()
        session_manager.load({"memory": {"max_turns": 4}})
        assert "test" in session_manager.sessions
        loaded = session_manager.sessions["test"]
        assert len(loaded.messages) == 2

    def test_validate_token_first_request(self, session_manager):
        """测试首次请求自动注册 token"""
        result = session_manager.validate_token("new_session", "test_token")
        assert result is True
        assert session_manager._session_tokens["new_session"] == "test_token"

    def test_validate_token_match(self, session_manager):
        """测试 token 匹配"""
        session_manager._session_tokens["existing"] = "correct_token"
        result = session_manager.validate_token("existing", "correct_token")
        assert result is True

    def test_validate_token_mismatch(self, session_manager):
        """测试 token 不匹配"""
        session_manager._session_tokens["existing"] = "correct_token"
        result = session_manager.validate_token("existing", "wrong_token")
        assert result is False


class TestInvokeWithRetry:
    """重试逻辑测试"""

    @pytest.fixture
    def qa_service(self):
        """创建 QAService 实例"""
        from src.qa_service import QAService
        service = QAService.__new__(QAService)
        service.logger = MagicMock()
        return service

    def test_invoke_success(self, qa_service):
        """测试调用成功"""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "测试回答"}

        result = qa_service.invoke_with_retry(mock_chain, {"question": "测试"})
        assert result == {"answer": "测试回答"}
        assert mock_chain.invoke.call_count == 1

    def test_invoke_retry_then_success(self, qa_service):
        """测试重试后成功"""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            Exception("临时错误"),
            {"answer": "测试回答"}
        ]

        result = qa_service.invoke_with_retry(mock_chain, {"question": "测试"}, max_retries=2)
        assert result == {"answer": "测试回答"}
        assert mock_chain.invoke.call_count == 2

    def test_invoke_max_retries_exceeded(self, qa_service):
        """测试超过最大重试次数"""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("持续错误")

        with pytest.raises(Exception, match="持续错误"):
            qa_service.invoke_with_retry(mock_chain, {"question": "测试"}, max_retries=2)
        assert mock_chain.invoke.call_count == 3  # 1 次初始 + 2 次重试
