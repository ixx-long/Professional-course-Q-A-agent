#!/usr/bin/env python3
"""
Web 问答服务器 — Flask API + 前端页面。

启动后访问 http://localhost:5000 进入问答界面。

用法:
    python web_server.py
    python web_server.py --config my_config.yaml --port 8080
"""

import sys
import argparse
import logging
import json
import atexit
import threading
import re
import base64
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_file

from src.utils import load_config, setup_logger
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.chain import create_qa_chain, ChatHistory

app = Flask(__name__)

# ---- 全局组件（延迟初始化）----
qa_chain = None
sessions: dict = {}       # session_id → ChatHistory（多用户隔离）
sessions_lock = threading.Lock()  # 并发保护
config = None
logger = None
retriever = None          # 供 _ask_with_image 和课程筛选使用
llm = None                # 供 _ask_with_image 使用
vectorstore = None        # 供课程筛选检索使用
SESSIONS_FILE = Path(__file__).parent / "data" / "sessions.json"
_filtered_chains: dict = {}           # 课程筛选 Chain 缓存
_filtered_chains_lock = threading.Lock()   # _filtered_chains 并发保护
_filtered_chains_times: dict = {}          # 各 Chain 缓存时间戳（LRU 淘汰用）
_MAX_FILTERED_CHAINS = 20                 # 缓存上限（防内存泄漏）
_sessions_dirty = False                    # 脏标记：True 表示有待保存的修改
_MAX_SESSIONS = 100                        # session 数量上限
_session_access_times: dict = {}           # session 最后访问时间（LRU 淘汰用）
_session_tokens: dict = {}                  # session_id → token（简单鉴权）


def _save_sessions():
    """持久化所有 session 对话历史到 JSON 文件（线程安全）。"""
    global sessions
    try:
        with sessions_lock:
            data = {}
            for sid, ch in sessions.items():
                msgs = []
                for m in ch.messages[-100:]:
                    msgs.append({
                        "role": "user" if m.__class__.__name__ == "HumanMessage" else "bot",
                        "content": m.content,
                    })
                if msgs:
                    data[sid] = msgs
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        if logger:
            logger.warning(f"保存 sessions 失败: {e}")


def _load_sessions():
    """从 JSON 文件恢复对话历史。"""
    global sessions, _session_access_times
    if not SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        now = _time.time()
        for sid, msgs in data.items():
            ch = ChatHistory(max_turns=config["memory"].get("max_turns", 4))
            for m in msgs:
                if m["role"] == "user":
                    ch.add_user(m["content"])
                else:
                    ch.add_ai(m["content"])
            sessions[sid] = ch
            _session_access_times[sid] = now  # 恢复的 session 也纳入 LRU 追踪
        if logger:
            logger.info(f"从文件恢复了 {len(sessions)} 个 session")
    except Exception as e:
        if logger:
            logger.warning(f"加载 sessions 失败: {e}")


# 写盘节流：30 秒内最多写一次
_last_save = 0
_SESSION_SAVE_INTERVAL = 30

def _save_sessions_throttled():
    """节流写盘 + 脏标记。

    设计权衡：为减少磁盘 I/O，30 秒内最多写入一次。若进程在此间隔内被
    强制终止（SIGKILL/OOM），本次修改丢失——atexit 无法覆盖此类场景。
    正常退出（Ctrl+C/SIGTERM）由 atexit 兜底，不会丢数据。
    """
    global _last_save, _sessions_dirty
    if not _sessions_dirty:
        return
    now = _time.time()
    if now - _last_save < _SESSION_SAVE_INTERVAL:
        return  # 还没到间隔，但脏标记保持为 True，下次调用会重试
    _last_save = now
    _sessions_dirty = False
    _save_sessions()

# 注册退出时强制写入
atexit.register(_save_sessions)


def init_system(config_path: str):
    """初始化 RAG 系统组件。"""
    global qa_chain, config, logger, retriever, llm, vectorstore

    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    logger = setup_logger(
        name="web_server",
        log_file=log_cfg.get("file", "logs/qa.log"),
        level=log_cfg.get("level", "INFO"),
    )
    logger.info("LLM 和 Embedding API Key 已加载")

    # Embedding + 向量库
    embedder = get_embedding_model(config)
    vectorstore = get_vectorstore(config, embedder)
    retriever = get_retriever(vectorstore, top_k=config["retrieval"]["top_k"])

    # CrossEncoder 可选
    try:
        from src.retriever import load_cross_encoder, create_compression_retriever
        reranker_cfg = config["reranker"]
        cross_encoder = load_cross_encoder(
            model_name=reranker_cfg["model_name"],
            cache_dir=reranker_cfg.get("cache_dir", "./models"),
        )
        compression_retriever = create_compression_retriever(
            retriever, cross_encoder, config["retrieval"]
        )
        logger.info("重排序模型加载成功")
    except Exception as e:
        logger.warning(f"重排序模型加载失败，使用基础检索: {e}")
        compression_retriever = retriever

    # LLM 实例（供图片问答使用）
    from src.chain import get_llm
    llm = get_llm(config)

    # 对话链（每个 session 独立 ChatHistory）
    qa_chain = create_qa_chain(compression_retriever, config)

    # 恢复持久化的 session 数据
    _load_sessions()

    logger.info("系统初始化完成")


def _invoke_with_retry(chain, inputs: dict, max_retries: int = 2):
    """带自动重试的 chain.invoke 调用（指数退避）。"""
    import time
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = 1.5 ** attempt
                logger.warning(f"Chain 调用失败 (重试 {attempt+1}/{max_retries}，等待 {delay:.1f}s): {e}")
                time.sleep(delay)
            else:
                logger.error(f"Chain 调用失败（已达最大重试次数）: {e}")
    raise last_err


def _validate_token(session_id: str) -> bool:
    """校验 session 归属：请求头 X-Session-Token 须与首次注册一致。

    首次请求（session 无已注册 token）时自动注册，后续请求必须携带一致 token。
    不传 token 且 session 已注册 → 拒绝。
    """
    token = request.headers.get("X-Session-Token", "")
    expected = _session_tokens.get(session_id)
    if expected is None:
        if token:
            # 首次请求：注册 token
            _session_tokens[session_id] = token
        # 无 token 也允许首次（兼容未配置鉴权的旧客户端首次访问）
        return True
    # session 已注册 token：必须匹配
    return token == expected


def _get_session(session_id: str) -> ChatHistory:
    """获取或创建 session 对应的 ChatHistory（线程安全，LRU 淘汰）。"""
    global sessions, _session_access_times
    with sessions_lock:
        if session_id not in sessions:
            # LRU 淘汰：超过上限时删除最久未访问的 session
            if len(sessions) >= _MAX_SESSIONS:
                oldest_sid = min(_session_access_times, key=_session_access_times.get)
                del sessions[oldest_sid]
                del _session_access_times[oldest_sid]
                _session_tokens.pop(oldest_sid, None)  # 同步清理 token
            sessions[session_id] = ChatHistory(max_turns=config["memory"].get("max_turns", 4))
        _session_access_times[session_id] = _time.time()
        return sessions[session_id]


# ---- API 路由 ----

@app.route("/")
def index():
    """前端页面（使用 send_file 避免 Jinja2 模板注入风险）。"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return send_file(str(template_path))
    return _build_inline_template(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """问答接口。支持文本问答、图片提问、课程筛选。"""
    global qa_chain, retriever, llm, vectorstore

    if qa_chain is None:
        return jsonify({"error": "系统未初始化，请等待服务启动完成"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    question = (data.get("question") or "").strip()
    image_b64 = data.get("image")
    course = data.get("course")
    session_id = data.get("session_id") or "default"
    if not _validate_token(session_id):
        return jsonify({"error": "无权操作此会话"}), 403

    if not question and not image_b64:
        return jsonify({"error": "问题不能为空"}), 400

    # 图片校验
    if image_b64:
        img_err = _validate_image(image_b64)
        if img_err:
            return jsonify({"error": f"图片无效: {img_err}"}), 400
        # 视觉模型检查
        model_name = config.get("llm", {}).get("model_name", "")
        if "vision" not in model_name.lower() and "gpt-4" not in model_name.lower() \
           and "claude" not in model_name.lower() and "gemini" not in model_name.lower():
            return jsonify({"error": "当前模型不支持图片问答，请使用文本提问"}), 400

    chat_history = _get_session(session_id)
    display_text = question or "[图片提问]"
    logger.info(f"[{session_id[:8]}] 问题: {display_text}" + (f" | 课程: {course}" if course else ""))

    try:
        if image_b64:
            answer, source_docs = _ask_with_image(question, image_b64, chat_history)
        else:
            # 课程筛选：使用缓存 Chain（避免每次重建，线程安全，有上限）
            if course and course != "all":
                with _filtered_chains_lock:
                    if course not in _filtered_chains:
                        # 缓存满时删除最久未使用的条目
                        if len(_filtered_chains) >= _MAX_FILTERED_CHAINS:
                            oldest_course = min(_filtered_chains_times, key=_filtered_chains_times.get)
                            del _filtered_chains[oldest_course]
                            del _filtered_chains_times[oldest_course]
                        _filtered_chains[course] = create_qa_chain(
                            _get_filtered_retriever(course), config
                        )
                    _filtered_chains_times[course] = _time.time()
                    chain = _filtered_chains[course]
            else:
                chain = qa_chain

            result = _invoke_with_retry(chain, {
                "question": question,
                "chat_history": chat_history.get_history(),
            })
            answer = result.get("answer", "（生成回答失败）")
            source_docs = result.get("source_documents", [])
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            logger.error(f"LLM 调用超时: {e}")
            return jsonify({"error": "请求超时，请稍后重试"}), 504
        if "401" in error_msg or "403" in error_msg or "invalid api key" in error_msg.lower():
            logger.error(f"API Key 无效: {e}")
            return jsonify({"error": "API Key 无效，请检查 config.yaml 配置"}), 500
        if "rate" in error_msg.lower():
            logger.error(f"API 限流: {e}")
            return jsonify({"error": "请求过于频繁，请稍后重试"}), 429
        logger.error(f"问答失败: {e}", exc_info=True)
        return jsonify({"error": "服务内部错误，请稍后重试"}), 500

    # 更新对话历史
    chat_history.add_user(display_text)
    chat_history.add_ai(answer)
    global _sessions_dirty
    _sessions_dirty = True
    _save_sessions_throttled()

    return jsonify({
        "answer": answer,
        "sources": _extract_sources(source_docs),
    })


def _validate_image(image_b64: str) -> str | None:
    """校验图片 base64。通过返回 None，失败返回错误消息。"""
    if not image_b64 or not isinstance(image_b64, str):
        return "图片数据为空或格式错误"
    if len(image_b64) > 10 * 1024 * 1024:  # 10MB base64 ≈ 7.5MB raw
        return "图片过大（最大 10MB）"
    # 检查是否为有效 data URL 或纯 base64
    if image_b64.startswith("data:image/"):
        # data URL 格式
        if not re.match(r'^data:image/(png|jpeg|jpg|gif|webp);base64,', image_b64):
            return "不支持的图片格式（支持 PNG/JPEG/GIF/WebP）"
        try:
            base64.b64decode(image_b64.split(",", 1)[1], validate=True)
        except Exception:
            return "图片 base64 编码无效"
    else:
        try:
            base64.b64decode(image_b64, validate=True)
        except Exception:
            return "图片 base64 编码无效"
    return None  # OK


def _get_filtered_retriever(course: str):
    """获取带课程筛选的 retriever。"""
    global vectorstore
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": config["retrieval"]["top_k"],
            "filter": {"course": course},
        },
    )


def _extract_sources(source_docs: list) -> list:
    """从 Document 列表提取去重的来源信息。"""
    sources = []
    seen = set()
    for doc in (source_docs or []):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score")
        key = f"{source}#p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source,
                "page": page,
                "score": round(float(score), 4) if score is not None else None,
            })
    return sources


def _ask_with_image(question: str, image_b64: str, chat_history: ChatHistory):
    """多模态问答：使用与文本一致的检索链路 + 图片 + LLM 直调。

    chat_history 由调用方 api_ask 传入，避免重复 _get_session 查找。
    """
    global retriever, llm, config

    # 1. 使用统一检索链路（与文本路径一致）
    search_query = question or "请描述这张图片的内容"
    raw_docs = retriever.invoke(search_query)[:config["retrieval"]["rerank_top_n"]]
    context = "\n\n".join(
        f"[来源: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n{d.page_content[:600]}"
        for d in raw_docs
    ) if raw_docs else "（知识库中未找到相关内容）"

    # 2. 构建多模态消息
    from langchain_core.messages import HumanMessage, SystemMessage

    system_msg = SystemMessage(content=(
        "你是一个专业课程答疑助手。请根据提供的知识库上下文回答用户问题。"
        "如果知识库信息不足，请如实说明。回答格式：\n"
        "【依据】…\n【解答】…\n【来源】…"
    ))

    history_str = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in chat_history.get_history()
    ) if chat_history.get_history() else "（无历史）"

    # 多模态 HumanMessage
    content_parts = []
    content_parts.append({
        "type": "text",
        "text": (
            f"## 知识库上下文\n{context}\n\n"
            f"## 对话历史\n{history_str}\n\n"
            f"## 用户问题\n{question or '请分析这张图片，结合知识库内容回答。'}"
        )
    })
    content_parts.append({
        "type": "image_url",
        "image_url": {"url": image_b64}
    })

    human_msg = HumanMessage(content=content_parts)

    # 3. 调用 LLM
    resp = llm.invoke([system_msg, human_msg])
    answer = resp.content

    return answer, raw_docs


@app.route("/api/history", methods=["GET"])
def api_history():
    """获取当前 session 的对话历史（页面刷新后恢复，需 token 鉴权）。"""
    session_id = request.args.get("session_id") or "default"
    if not _validate_token(session_id):
        return jsonify({"error": "无权访问此会话"}), 403
    chat_history = _get_session(session_id)

    from langchain_core.messages import HumanMessage, AIMessage
    messages = []
    for msg in chat_history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "bot", "content": msg.content})
    return jsonify({"messages": messages})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """重置当前 session 的对话历史（需 token 鉴权）。"""
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    if not _validate_token(session_id):
        return jsonify({"error": "无权操作此会话"}), 403
    _get_session(session_id).clear()
    global _sessions_dirty
    _sessions_dirty = True
    _save_sessions_throttled()
    logger.info(f"[{session_id[:8]}] 对话历史已重置")
    return jsonify({"status": "ok", "message": "对话历史已清空"})


# ---- 启动 ----

def main():
    parser = argparse.ArgumentParser(description="专业课程答疑 Web 服务器")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--port", "-p", type=int, default=5000, help="服务端口（默认: 5000）")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（默认: 127.0.0.1）")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 60)
    print("  专业课程答疑 Web 服务器")
    print("=" * 60)

    print("正在初始化系统组件...")
    try:
        init_system(args.config)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        sys.exit(1)

    print(f"  访问地址: http://{args.host}:{args.port}")
    print(f"  按 Ctrl+C 停止服务")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()


# ---- 内联 HTML 模板（作为后备）----

def _build_inline_template() -> str:
    """构建内联 HTML 模板字符串（标准模式应使用 templates/index.html）。"""
    return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>专业课程答疑</title>
</head>
<body>
  <p>模板未加载。请将 index.html 放入 templates/ 目录。</p>
</body>
</html>"""
