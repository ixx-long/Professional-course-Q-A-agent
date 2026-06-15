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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template_string

from src.utils import load_config, setup_logger, mask_key
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.chain import create_qa_chain, ChatHistory

app = Flask(__name__)

# ---- 全局组件（延迟初始化）----
qa_chain = None
chat_history = None
config = None
logger = None
retriever = None   # 供 _ask_with_image 使用
llm = None         # 供 _ask_with_image 使用

# ---- 内联 HTML 模板（从 templates/index.html 加载或使用内联模板）----
INDEX_HTML = None  # 延迟加载


def init_system(config_path: str):
    """初始化 RAG 系统组件。"""
    global qa_chain, chat_history, config, logger, retriever, llm

    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    logger = setup_logger(
        name="web_server",
        log_file=log_cfg.get("file", "logs/qa.log"),
        level=log_cfg.get("level", "INFO"),
    )
    logger.info(f"LLM API Key: {mask_key(config['llm']['api_key'])}")
    logger.info(f"Embedding API Key: {mask_key(config['embedding']['api_key'])}")

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

    # 对话历史 + 链
    chat_history = ChatHistory(max_turns=config["memory"].get("max_turns", 4))
    qa_chain = create_qa_chain(compression_retriever, config)

    logger.info("系统初始化完成")


def load_index_html() -> str:
    """加载前端 HTML 页面。"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    # 回退：使用内联模板字符串（由 build_frontend.py 生成）
    return _build_inline_template()


# ---- API 路由 ----

@app.route("/")
def index():
    """前端页面。"""
    global INDEX_HTML
    if INDEX_HTML is None:
        INDEX_HTML = load_index_html()
    return render_template_string(INDEX_HTML)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """问答接口。支持文本问答和图片提问。"""
    global qa_chain, chat_history, retriever, llm

    if qa_chain is None:
        return jsonify({"error": "系统未初始化"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    question = (data.get("question") or "").strip()
    image_b64 = data.get("image")  # 可选：base64 图片

    if not question and not image_b64:
        return jsonify({"error": "问题不能为空"}), 400

    display_text = question or "[图片提问]"
    logger.info(f"用户问题: {display_text}")

    try:
        # 有图片：走多模态直调（检索文档 + 图片 + 问题）
        if image_b64:
            answer, source_docs = _ask_with_image(question, image_b64)
        else:
            result = qa_chain.invoke({
                "question": question,
                "chat_history": chat_history.get_history(),
            })
            answer = result.get("answer", "（生成回答失败）")
            source_docs = result.get("source_documents", [])
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        return jsonify({"error": f"问答失败: {str(e)}"}), 500

    # 更新对话历史
    chat_history.add_user(display_text)
    chat_history.add_ai(answer)

    # 提取来源信息
    sources = _extract_sources(source_docs)

    return jsonify({
        "answer": answer,
        "sources": sources,
    })


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
                "score": round(float(score), 4) if score else None,
            })
    return sources


def _ask_with_image(question: str, image_b64: str):
    """多模态问答：检索文档 + 图片 + LLM 直调。"""
    global retriever, llm, chat_history, config

    # 1. 用文本检索相关文档
    search_query = question or "请描述这张图片的内容"
    raw_docs = retriever.invoke(search_query)[:4]
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


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """重置对话历史。"""
    global chat_history
    if chat_history:
        chat_history.clear()
    logger.info("对话历史已重置")
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

    app.run(host=args.host, port=args.port, debug=args.debug)


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
