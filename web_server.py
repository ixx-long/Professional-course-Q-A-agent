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

# ---- 内联 HTML 模板（从 templates/index.html 加载或使用内联模板）----
INDEX_HTML = None  # 延迟加载


def init_system(config_path: str):
    """初始化 RAG 系统组件。"""
    global qa_chain, chat_history, config, logger

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
    """问答接口。"""
    global qa_chain, chat_history

    if qa_chain is None:
        return jsonify({"error": "系统未初始化"}), 500

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "缺少 question 参数"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    logger.info(f"用户问题: {question}")

    try:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history.get_history(),
        })
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        return jsonify({"error": f"问答失败: {str(e)}"}), 500

    answer = result.get("answer", "（生成回答失败）")
    source_docs = result.get("source_documents", [])

    # 更新对话历史
    chat_history.add_user(question)
    chat_history.add_ai(answer)

    # 提取来源信息
    sources = []
    seen = set()
    for doc in source_docs:
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

    return jsonify({
        "answer": answer,
        "sources": sources,
    })


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
