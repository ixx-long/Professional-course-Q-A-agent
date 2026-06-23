"""
集成测试 - 端到端测试

测试完整的问答流程，从用户请求到最终响应。
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def temp_workspace():
    """创建临时工作空间"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    @pytest.fixture
    def mock_qa_service(self, temp_workspace):
        """创建模拟的 QAService"""
        with patch('src.utils.load_config') as mock_load_config, \
             patch('src.qa_service.get_embedding_model') as mock_get_embedding, \
             patch('src.qa_service.get_vectorstore') as mock_get_vectorstore, \
             patch('src.vectorstore.get_retriever') as mock_get_retriever, \
             patch('src.chain.get_llm') as mock_get_llm, \
             patch('src.qa_service.create_qa_chain') as mock_create_chain:
            
            # 配置加载
            mock_config = {
                "llm": {
                    "api_key": "test-key",
                    "api_base": "https://api.test.com",
                    "model_name": "test-model"
                },
                "embedding": {
                    "api_key": "test-key",
                    "api_base": "https://api.test.com",
                    "model_name": "test-embedding"
                },
                "retrieval": {
                    "top_k": 5
                },
                "logging": {
                    "level": "INFO",
                    "file": "./logs/test.log"
                }
            }
            mock_load_config.return_value = mock_config
            
            # Embedding 模型
            mock_embedder = Mock()
            mock_embedder.embed_query.return_value = [0.1] * 768
            mock_get_embedding.return_value = mock_embedder
            
            # 向量库
            mock_vectorstore = Mock()
            mock_get_vectorstore.return_value = mock_vectorstore
            
            # 检索器
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = []
            mock_get_retriever.return_value = mock_retriever
            
            # LLM
            mock_llm = Mock()
            mock_llm.invoke.return_value = Mock(content="测试回答")
            mock_get_llm.return_value = mock_llm
            
            # Chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "answer": "测试回答",
                "source_documents": []
            }
            mock_create_chain.return_value = mock_chain
            
            from src.qa_service import QAService
            service = QAService(config_path="test_config.yaml")
            service.init_system()
            
            yield {
                'service': service,
                'config': mock_config,
                'embedder': mock_embedder,
                'vectorstore': mock_vectorstore,
                'retriever': mock_retriever,
                'llm': mock_llm,
                'chain': mock_chain
            }
    
    def test_complete_qa_workflow(self, mock_qa_service):
        """测试完整的问答工作流"""
        service = mock_qa_service['service']
        
        # 执行问答
        session_id = "test_session"
        question = "什么是机器学习？"
        
        # 获取会话
        chat_history = service.session_manager.get_session(session_id)
        
        # 调用 chain
        chain = mock_qa_service['chain']
        result = service.invoke_with_retry(chain, {
            "question": question,
            "chat_history": chat_history.get_history()
        })
        
        # 验证结果
        assert result is not None
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        
        # 验证组件被正确调用
        chain.invoke.assert_called_once()
    
    def test_multi_turn_conversation(self, mock_qa_service):
        """测试多轮对话"""
        service = mock_qa_service['service']
        session_id = "test_session"
        
        # 获取会话
        chat_history = service.session_manager.get_session(session_id)
        
        # 第一轮对话
        chain = mock_qa_service['chain']
        result1 = service.invoke_with_retry(chain, {
            "question": "什么是机器学习？",
            "chat_history": chat_history.get_history()
        })
        chat_history.add_user("什么是机器学习？")
        chat_history.add_ai(result1["answer"])
        
        # 第二轮对话（应该包含上下文）
        result2 = service.invoke_with_retry(chain, {
            "question": "它有哪些应用？",
            "chat_history": chat_history.get_history()
        })
        
        # 验证 chain 被调用了两次
        assert chain.invoke.call_count == 2
    
    def test_session_isolation(self, mock_qa_service):
        """测试会话隔离"""
        service = mock_qa_service['service']
        
        # 两个不同的会话
        session1 = "session_1"
        session2 = "session_2"
        
        # 获取两个会话
        history1 = service.session_manager.get_session(session1)
        history2 = service.session_manager.get_session(session2)
        
        # 验证会话是独立的
        assert session1 != session2
        assert history1 is not history2
    
    def test_error_handling(self, mock_qa_service):
        """测试错误处理"""
        service = mock_qa_service['service']
        
        # 模拟 LLM 错误
        chain = mock_qa_service['chain']
        chain.invoke.side_effect = Exception("LLM 错误")
        
        # 应该捕获错误
        with pytest.raises(Exception):
            service.invoke_with_retry(chain, {
                "question": "测试问题",
                "chat_history": []
            })
    
    def test_empty_question(self, mock_qa_service):
        """测试空问题"""
        service = mock_qa_service['service']
        
        # 空问题应该被处理
        chain = mock_qa_service['chain']
        result = service.invoke_with_retry(chain, {
            "question": "",
            "chat_history": []
        })
        
        # chain 应该被调用
        chain.invoke.assert_called_once()


class TestSessionManagement:
    """会话管理集成测试"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_session_persistence(self, temp_workspace):
        """测试会话持久化"""
        from src.session_manager import SessionManager
        
        sessions_file = Path(temp_workspace) / "sessions.json"
        
        # 第一次初始化并创建会话
        manager1 = SessionManager(
            sessions_file=sessions_file,
            max_sessions=100,
            max_turns=4,
            save_interval=30
        )
        
        session_id = "test_session"
        chat_history = manager1.get_session(session_id)
        chat_history.add_user("问题1")
        chat_history.add_ai("回答1")
        
        # 保存会话
        manager1.save()
        
        # 第二次初始化，应该能恢复会话
        manager2 = SessionManager(
            sessions_file=sessions_file,
            max_sessions=100,
            max_turns=4,
            save_interval=30
        )
        manager2.load({"memory": {"max_turns": 4}})
        
        # 验证会话被恢复
        session = manager2.get_session(session_id)
        assert session is not None
        assert len(session.messages) > 0
    
    def test_session_token_validation(self):
        """测试会话令牌验证"""
        from src.session_manager import SessionManager
        
        manager = SessionManager()
        
        # 注册令牌
        assert manager.validate_token("session1", "token1") is True
        
        # 验证令牌
        assert manager.validate_token("session1", "token1") is True
        assert manager.validate_token("session1", "token2") is False


class TestRateLimiting:
    """限流集成测试"""
    
    def test_rate_limiting(self):
        """测试限流"""
        from src.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # 前 5 个请求应该通过
        for i in range(5):
            allowed, info = limiter.is_allowed("test_client")
            assert allowed is True
        
        # 第 6 个请求应该被拒绝
        allowed, info = limiter.is_allowed("test_client")
        assert allowed is False


class TestSecurityIntegration:
    """安全集成测试"""
    
    def test_input_validation(self):
        """测试输入验证"""
        from src.security import SecurityValidator
        
        # 正常输入
        assert SecurityValidator.validate_question("什么是机器学习？") is None
        
        # 过长输入
        long_input = "a" * 10000
        result = SecurityValidator.validate_question(long_input)
        assert result is not None
        
        # 空输入
        result = SecurityValidator.validate_question("")
        assert result is not None
    
    def test_sensitive_data_masking(self):
        """测试敏感信息脱敏"""
        from src.security import SensitiveDataMasker
        
        # 测试 API Key 脱敏（使用更长的 key 以匹配正则）
        text = "api_key: sk-1234567890abcdef1234567890"
        masked = SensitiveDataMasker.mask_text(text)
        assert "sk-1234567890abcdef1234567890" not in masked
        
        # 测试密码脱敏
        text = "password: mysecretpassword123"
        masked = SensitiveDataMasker.mask_text(text)
        assert "mysecretpassword123" not in masked


class TestPerformanceIntegration:
    """性能集成测试"""
    
    def test_concurrent_sessions(self, temp_workspace):
        """测试并发会话"""
        import concurrent.futures
        from src.session_manager import SessionManager
        
        manager = SessionManager(
            sessions_file=Path(temp_workspace) / "sessions.json",
            max_sessions=100,
            max_turns=4,
            save_interval=30
        )
        
        def create_session(session_id):
            return manager.get_session(session_id)
        
        # 并发创建 10 个会话
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_session, f"session_{i}")
                for i in range(10)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # 所有会话都应该被创建
        assert len(results) == 10
        assert manager.session_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
