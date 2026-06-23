"""
测试 chain 模块。
"""
import pytest
from src.chain import ChatHistory, format_source_documents
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class TestChatHistory:
    """测试对话历史管理。"""

    def test_init_default(self):
        """测试默认初始化。"""
        ch = ChatHistory()
        assert ch.max_turns == 4
        assert len(ch.messages) == 0

    def test_init_custom_turns(self):
        """测试自定义轮数。"""
        ch = ChatHistory(max_turns=2)
        assert ch.max_turns == 2

    def test_add_user_message(self):
        """测试添加用户消息。"""
        ch = ChatHistory()
        ch.add_user("你好")
        assert len(ch.messages) == 1
        assert isinstance(ch.messages[0], HumanMessage)
        assert ch.messages[0].content == "你好"

    def test_add_ai_message(self):
        """测试添加 AI 消息。"""
        ch = ChatHistory()
        ch.add_ai("你好，有什么可以帮助你的？")
        assert len(ch.messages) == 1
        assert isinstance(ch.messages[0], AIMessage)
        assert ch.messages[0].content == "你好，有什么可以帮助你的？"

    def test_get_history_empty(self):
        """测试获取空历史。"""
        ch = ChatHistory()
        history = ch.get_history()
        assert history == []

    def test_get_history_within_limit(self):
        """测试获取未超限的历史。"""
        ch = ChatHistory(max_turns=2)
        ch.add_user("问题1")
        ch.add_ai("回答1")
        ch.add_user("问题2")
        ch.add_ai("回答2")
        
        history = ch.get_history()
        assert len(history) == 4

    def test_get_history_exceeds_limit(self):
        """测试获取超限的历史（应截断）。"""
        ch = ChatHistory(max_turns=2)
        # 添加 3 轮对话（6 条消息）
        for i in range(3):
            ch.add_user(f"问题{i}")
            ch.add_ai(f"回答{i}")
        
        history = ch.get_history()
        # 应该只返回最近 2 轮（4 条消息）
        assert len(history) == 4
        assert history[0].content == "问题1"
        assert history[1].content == "回答1"

    def test_clear(self):
        """测试清空历史。"""
        ch = ChatHistory()
        ch.add_user("问题")
        ch.add_ai("回答")
        assert len(ch.messages) == 2
        
        ch.clear()
        assert len(ch.messages) == 0


class TestFormatSourceDocuments:
    """测试来源文档格式化。"""

    def test_format_empty_list(self):
        """测试空列表。"""
        result = format_source_documents([])
        assert result == "无来源引用"

    def test_format_single_source(self):
        """测试单个来源。"""
        docs = [
            Document(
                page_content="内容",
                metadata={"source": "test.md", "page": 1}
            )
        ]
        result = format_source_documents(docs)
        assert "test.md" in result
        assert "第1页" in result

    def test_format_with_rerank_score(self):
        """测试带重排序分数。"""
        docs = [
            Document(
                page_content="内容",
                metadata={"source": "test.md", "page": 1, "rerank_score": 0.95}
            )
        ]
        result = format_source_documents(docs)
        assert "相关度: 0.950" in result

    def test_format_deduplicate(self):
        """测试去重。"""
        docs = [
            Document(
                page_content="内容1",
                metadata={"source": "test.md", "page": 1}
            ),
            Document(
                page_content="内容2",
                metadata={"source": "test.md", "page": 1}  # 相同来源
            ),
        ]
        result = format_source_documents(docs)
        # 应该只出现一次
        assert result.count("test.md") == 1
