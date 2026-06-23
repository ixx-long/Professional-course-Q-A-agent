"""
Session 管理模块。

负责对话历史的创建、LRU 淘汰、Token 鉴权和持久化。
"""

import json
import logging
import time as _time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from src.chain import ChatHistory

_logger = logging.getLogger(__name__)


class SessionManager:
    """Session 管理器，封装对话历史的创建、LRU 淘汰、Token 鉴权和持久化。"""

    def __init__(
        self,
        sessions_file: str | Path = "data/sessions.json",
        max_sessions: int = 100,
        max_turns: int = 4,
        save_interval: int = 30,
    ) -> None:
        self.sessions: Dict[str, ChatHistory] = {}
        self.sessions_lock = threading.Lock()
        self.SESSIONS_FILE = Path(sessions_file)
        self._session_access_times: Dict[str, float] = {}
        self._session_tokens: Dict[str, str] = {}
        self._MAX_SESSIONS = max_sessions
        self._max_turns = max_turns
        self._sessions_dirty = False
        self._last_save = 0.0
        self._SESSION_SAVE_INTERVAL = save_interval

    def get_session(self, session_id: str) -> ChatHistory:
        """获取或创建 session 对应的 ChatHistory（线程安全，LRU 淘汰）。"""
        with self.sessions_lock:
            if session_id not in self.sessions:
                if len(self.sessions) >= self._MAX_SESSIONS:
                    oldest_sid = min(
                        self._session_access_times,
                        key=self._session_access_times.get,
                    )
                    del self.sessions[oldest_sid]
                    del self._session_access_times[oldest_sid]
                    self._session_tokens.pop(oldest_sid, None)
                self.sessions[session_id] = ChatHistory(max_turns=self._max_turns)
            self._session_access_times[session_id] = _time.time()
            return self.sessions[session_id]

    def validate_token(self, session_id: str, token: str) -> bool:
        """校验 session 归属 Token。

        首次请求（session 无已注册 token）时自动注册，后续请求必须携带一致 token。
        空 token 不予注册，防止被任意覆盖。
        """
        if not token:
            return False
        expected = self._session_tokens.get(session_id)
        if expected is None:
            self._session_tokens[session_id] = token
            return True
        return token == expected

    def mark_dirty(self) -> None:
        """标记 session 数据已变更，需要持久化。"""
        self._sessions_dirty = True

    def save_throttled(self) -> None:
        """节流写盘 + 脏标记（线程安全）。"""
        if not self._sessions_dirty:
            return
        now = _time.time()
        with self.sessions_lock:
            if now - self._last_save < self._SESSION_SAVE_INTERVAL:
                return
            self._last_save = now
            self._sessions_dirty = False
            # 在锁内执行保存，避免并发写入
            self._save_locked()

    def save(self) -> None:
        """持久化所有 session 对话历史到 JSON 文件（线程安全）。"""
        with self.sessions_lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """在锁内执行实际的文件写入（调用前必须持有 sessions_lock）。"""
        try:
            data = {}
            for sid, ch in self.sessions.items():
                msgs = []
                for m in ch.messages[-100:]:
                    if isinstance(m, HumanMessage):
                        role = "user"
                    elif isinstance(m, AIMessage):
                        role = "bot"
                    else:
                        continue
                    msgs.append({"role": role, "content": m.content})
                if msgs:
                    data[sid] = msgs

            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            self.SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.SESSIONS_FILE.write_text(json_data, encoding="utf-8")
        except Exception as e:
            _logger.error(f"保存会话数据失败: {e}", exc_info=True)

    def load(self, config: Optional[Dict[str, Any]], logger: Optional[Any] = None) -> None:
        """从 JSON 文件恢复对话历史。"""
        if not self.SESSIONS_FILE.exists():
            return
        if config is None:
            if logger:
                logger.warning("配置未加载，跳过 session 恢复")
            return
        try:
            data = json.loads(self.SESSIONS_FILE.read_text(encoding="utf-8"))
            now = _time.time()
            max_turns = config.get("memory", {}).get("max_turns", self._max_turns)
            success_count = 0
            for sid, msgs in data.items():
                try:
                    ch = ChatHistory(max_turns=max_turns)
                    for m in msgs:
                        if m["role"] == "user":
                            ch.add_user(m["content"])
                        else:
                            ch.add_ai(m["content"])
                    self.sessions[sid] = ch
                    self._session_access_times[sid] = now
                    success_count += 1
                except Exception as e:
                    if logger:
                        logger.warning(f"恢复 session {sid} 失败: {e}，跳过")
            if logger:
                logger.info(f"从文件恢复了 {success_count}/{len(data)} 个 session")
        except Exception as e:
            if logger:
                logger.warning(f"加载 sessions 失败: {e}")

    async def save_async(self) -> None:
        """异步持久化所有 session 对话历史到 JSON 文件。"""
        try:
            data = {}
            for sid, ch in self.sessions.items():
                msgs = []
                for m in ch.messages[-100:]:
                    if isinstance(m, HumanMessage):
                        role = "user"
                    elif isinstance(m, AIMessage):
                        role = "bot"
                    else:
                        continue
                    msgs.append({"role": role, "content": m.content})
                if msgs:
                    data[sid] = msgs

            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            self.SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 asyncio.to_thread 将阻塞的文件写入操作放到线程池
            await asyncio.to_thread(self.SESSIONS_FILE.write_text, json_data, encoding="utf-8")
        except Exception as e:
            _logger.error(f"异步保存会话数据失败: {e}", exc_info=True)

    async def load_async(self, config: Optional[Dict[str, Any]], logger: Optional[Any] = None) -> None:
        """异步从 JSON 文件恢复对话历史。"""
        if not self.SESSIONS_FILE.exists():
            return
        if config is None:
            if logger:
                logger.warning("配置未加载，跳过 session 恢复")
            return
        try:
            # 使用 asyncio.to_thread 将阻塞的文件读取操作放到线程池
            content = await asyncio.to_thread(self.SESSIONS_FILE.read_text, encoding="utf-8")
            data = json.loads(content)
            now = _time.time()
            max_turns = config.get("memory", {}).get("max_turns", self._max_turns)
            success_count = 0
            for sid, msgs in data.items():
                try:
                    ch = ChatHistory(max_turns=max_turns)
                    for m in msgs:
                        if m["role"] == "user":
                            ch.add_user(m["content"])
                        else:
                            ch.add_ai(m["content"])
                    self.sessions[sid] = ch
                    self._session_access_times[sid] = now
                    success_count += 1
                except Exception as e:
                    if logger:
                        logger.warning(f"恢复 session {sid} 失败: {e}，跳过")
            if logger:
                logger.info(f"从文件恢复了 {success_count}/{len(data)} 个 session")
        except Exception as e:
            if logger:
                logger.warning(f"加载 sessions 失败: {e}")

    @property
    def session_count(self) -> int:
        """当前活跃 session 数量。"""
        with self.sessions_lock:
            return len(self.sessions)
