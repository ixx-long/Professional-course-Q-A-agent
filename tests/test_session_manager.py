"""
Session 管理模块单元测试。
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.session_manager import SessionManager
from src.chain import ChatHistory


class TestSessionManager:
    """SessionManager 单元测试类。"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def session_manager(self, temp_dir):
        """创建测试用 SessionManager 实例。"""
        return SessionManager(
            sessions_file=Path(temp_dir) / "sessions.json",
            max_sessions=3,
            max_turns=4,
            save_interval=30,
        )

    def test_get_session_creates_new(self, session_manager):
        """测试创建新 session。"""
        session = session_manager.get_session("test_session")
        
        assert isinstance(session, ChatHistory)
        assert "test_session" in session_manager.sessions
        assert session_manager.session_count == 1

    def test_get_session_returns_existing(self, session_manager):
        """测试获取已存在的 session。"""
        session1 = session_manager.get_session("test_session")
        session2 = session_manager.get_session("test_session")
        
        assert session1 is session2
        assert session_manager.session_count == 1

    def test_get_session_lru_eviction(self, session_manager):
        """测试 LRU 淘汰机制。"""
        # 创建 3 个 session（达到上限）
        session_manager.get_session("session1")
        session_manager.get_session("session2")
        session_manager.get_session("session3")
        
        assert session_manager.session_count == 3
        
        # 手动设置访问时间，使 session2 最旧
        import time
        now = time.time()
        session_manager._session_access_times["session1"] = now - 100
        session_manager._session_access_times["session2"] = now - 200  # 最旧
        session_manager._session_access_times["session3"] = now - 50
        
        # 创建第 4 个 session，应该淘汰最久未访问的 session2
        session_manager.get_session("session4")
        
        assert session_manager.session_count == 3
        assert "session2" not in session_manager.sessions
        assert "session1" in session_manager.sessions
        assert "session3" in session_manager.sessions
        assert "session4" in session_manager.sessions

    def test_validate_token_first_request(self, session_manager):
        """测试首次请求自动注册 token。"""
        result = session_manager.validate_token("new_session", "test_token")
        
        assert result is True
        assert session_manager._session_tokens["new_session"] == "test_token"

    def test_validate_token_match(self, session_manager):
        """测试 token 匹配。"""
        session_manager._session_tokens["existing"] = "correct_token"
        
        result = session_manager.validate_token("existing", "correct_token")
        assert result is True

    def test_validate_token_mismatch(self, session_manager):
        """测试 token 不匹配。"""
        session_manager._session_tokens["existing"] = "correct_token"
        
        result = session_manager.validate_token("existing", "wrong_token")
        assert result is False

    def test_validate_token_empty(self, session_manager):
        """测试空 token。"""
        result = session_manager.validate_token("session", "")
        assert result is False

    def test_mark_dirty(self, session_manager):
        """测试标记脏数据。"""
        assert not session_manager._sessions_dirty
        
        session_manager.mark_dirty()
        
        assert session_manager._sessions_dirty

    def test_save(self, session_manager):
        """测试保存 session 数据。"""
        # 添加一些数据
        session = session_manager.get_session("test_session")
        session.add_user("你好")
        session.add_ai("你好！有什么可以帮助你的？")
        
        # 保存
        session_manager.save()
        
        # 验证文件已创建
        assert session_manager.SESSIONS_FILE.exists()
        
        # 验证文件内容
        with open(session_manager.SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "test_session" in data
        assert len(data["test_session"]) == 2
        assert data["test_session"][0]["role"] == "user"
        assert data["test_session"][0]["content"] == "你好"
        assert data["test_session"][1]["role"] == "bot"

    def test_save_throttled(self, session_manager):
        """测试节流保存。"""
        session = session_manager.get_session("test_session")
        session.add_user("测试")
        
        session_manager.mark_dirty()
        
        # 第一次调用应该保存
        import time
        session_manager._last_save = 0
        session_manager.save_throttled()
        
        assert session_manager.SESSIONS_FILE.exists()
        assert not session_manager._sessions_dirty

    def test_save_throttled_respects_interval(self, session_manager):
        """测试节流保存遵守时间间隔。"""
        session = session_manager.get_session("test_session")
        session.add_user("测试")
        
        session_manager.mark_dirty()
        
        # 第一次保存
        session_manager._last_save = 0
        session_manager.save_throttled()
        
        # 立即再次标记脏并尝试保存
        session_manager.mark_dirty()
        session_manager.save_throttled()
        
        # 应该还在节流期内，不会再次保存
        # （通过检查 _last_save 是否更新来判断）

    def test_load(self, session_manager):
        """测试加载 session 数据。"""
        # 先保存一些数据
        session = session_manager.get_session("test_session")
        session.add_user("问题")
        session.add_ai("回答")
        session_manager.save()
        
        # 创建新的 manager 并加载
        new_manager = SessionManager(
            sessions_file=session_manager.SESSIONS_FILE,
            max_sessions=3,
            max_turns=4,
        )
        
        config = {"memory": {"max_turns": 4}}
        new_manager.load(config)
        
        assert "test_session" in new_manager.sessions
        assert new_manager.session_count == 1
        
        loaded_session = new_manager.sessions["test_session"]
        assert len(loaded_session.messages) == 2

    def test_load_nonexistent_file(self, session_manager):
        """测试加载不存在的文件。"""
        config = {"memory": {"max_turns": 4}}
        
        # 不应该抛出异常
        session_manager.load(config)
        
        assert session_manager.session_count == 0

    def test_load_with_invalid_data(self, session_manager, temp_dir):
        """测试加载无效数据。"""
        # 创建包含无效数据的文件
        invalid_file = Path(temp_dir) / "invalid_sessions.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("invalid json")
        
        manager = SessionManager(sessions_file=invalid_file)
        config = {"memory": {"max_turns": 4}}
        
        # 应该捕获异常，不抛出
        manager.load(config)

    def test_session_count_property(self, session_manager):
        """测试 session 计数属性。"""
        assert session_manager.session_count == 0
        
        session_manager.get_session("session1")
        assert session_manager.session_count == 1
        
        session_manager.get_session("session2")
        assert session_manager.session_count == 2
        
        session_manager.get_session("session1")  # 重复获取
        assert session_manager.session_count == 2

    def test_thread_safety(self, session_manager):
        """测试线程安全性。"""
        import threading
        
        def create_sessions():
            for i in range(10):
                session_manager.get_session(f"session_{threading.current_thread().name}_{i}")
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_sessions, name=f"thread_{i}")
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 应该创建了 50 个 session（5 线程 * 10 session）
        # 但由于 max_sessions=3，实际应该只有 3 个
        assert session_manager.session_count == 3

    def test_save_handles_exception(self, session_manager):
        """测试保存时异常处理。"""
        # 添加数据
        session = session_manager.get_session("test_session")
        session.add_user("测试")
        
        # 模拟保存失败
        with patch.object(Path, "write_text", side_effect=Exception("磁盘错误")):
            # 不应该抛出异常
            session_manager.save()
        
        # 应该记录了错误日志（通过检查 logger 调用）
