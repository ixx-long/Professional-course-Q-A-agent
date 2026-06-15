#!/usr/bin/env python3
"""
命令行交互问答脚本。

启动后进入交互式循环，用户可以输入课程问题，
系统执行 检索->重排序->生成 链路并打印答案。

支持特殊命令:
  /reset  - 清空对话记忆
  /sources - 切换是否显示详细引用
  /exit   - 退出程序

用法:
    python qa_cli.py
    python qa_cli.py --config my_config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, mask_key
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.retriever import load_cross_encoder, create_compression_retriever
from src.chain import create_qa_chain, ChatHistory, format_source_documents

# ANSI 颜色（跨平台）
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = YELLOW = RED = RESET = ""


def print_banner():
    """打印欢迎横幅。"""
    print(CYAN + "=" * 60)
    print("   专业课程答疑智能体")
    print("   输入问题开始问答，/exit 退出，/reset 清空记忆，/sources 切换来源显示")
    print("=" * 60 + RESET)


def print_sources(source_docs):
    """打印来源引用。"""
    if not source_docs:
        print(YELLOW + "  (无知识库来源引用)" + RESET)
        return
    print(GREEN + "\n参考来源:" + RESET)
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score")
        score_str = f" [相关度: {score:.3f}]" if score else ""
        key = f"{source}#p{page}"
        if key not in seen:
            seen.add(key)
            print(f"  • {source}  第{page}页{score_str}")


def main():
    parser = argparse.ArgumentParser(
        description="专业课程答疑 — 命令行交互问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--show_raw",
        action="store_true",
        help="显示检索到的原始文本片段（调试用）",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"{RED}[错误] 加载配置失败: {e}{RESET}")
        sys.exit(1)

    # ---- 初始化日志 ----
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="qa_cli",
        log_file=log_config.get("file", "logs/qa.log"),
        level=log_config.get("level", "INFO"),
    )
    logger.info(f"LLM API Key: {mask_key(config['llm']['api_key'])}")
    logger.info(f"Embedding API Key: {mask_key(config['embedding']['api_key'])}")

    # ---- 初始化组件 ----
    print(CYAN + "正在初始化系统组件..." + RESET)

    try:
        # Embedding + 向量库
        embedder = get_embedding_model(config)
        vectorstore = get_vectorstore(config, embedder)
        retriever = get_retriever(vectorstore, top_k=config["retrieval"]["top_k"])

        # CrossEncoder（可选：加载失败时降级为无重排序模式）
        cross_encoder = None
        try:
            reranker_config = config["reranker"]
            cross_encoder = load_cross_encoder(
                model_name=reranker_config["model_name"],
                cache_dir=reranker_config.get("cache_dir", "./models"),
            )
            compression_retriever = create_compression_retriever(
                retriever, cross_encoder, config["retrieval"]
            )
            print(f"{GREEN}[OK] 重排序模型加载成功{RESET}")
        except Exception as ce_err:
            print(f"{YELLOW}[警告] 重排序模型加载失败，使用基础检索: {ce_err}{RESET}")
            logger.warning(f"CrossEncoder 加载失败，降级为无重排序: {ce_err}")
            compression_retriever = retriever  # 降级：不使用重排序

        # 对话历史 + 对话链
        chat_history = ChatHistory(max_turns=config["memory"].get("max_turns", 4))
        qa_chain = create_qa_chain(compression_retriever, config)

    except Exception as e:
        print(f"{RED}[错误] 初始化失败: {e}{RESET}")
        logger.error(f"初始化失败: {e}", exc_info=True)
        sys.exit(1)

    print(GREEN + "[OK] 系统就绪" + RESET)
    logger.info("系统初始化完成")

    # ---- 交互循环 ----
    print_banner()
    show_raw = args.show_raw

    while True:
        try:
            user_input = input(CYAN + "\n你的问题: " + RESET).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input.lower() == "/exit":
            print("再见！")
            break

        if user_input.lower() == "/reset":
            chat_history.clear()
            print(GREEN + "[OK] 对话记忆已清空" + RESET)
            continue

        if user_input.lower() == "/sources":
            show_raw = not show_raw
            status = "开启" if show_raw else "关闭"
            print(GREEN + f"[OK] 详细来源显示已{status}" + RESET)
            continue

        # ---- 执行问答 ----
        logger.info(f"用户问题: {user_input}")
        print(YELLOW + "正在检索知识库..." + RESET)

        try:
            result = qa_chain.invoke({
                "question": user_input,
                "chat_history": chat_history.get_history(),
            })
        except Exception as e:
            print(f"{RED}[错误] 问答失败: {e}{RESET}")
            logger.error(f"问答失败: {e}", exc_info=True)
            continue

        answer = result.get("answer", "（生成回答失败）")
        source_docs = result.get("source_documents", [])

        # 更新对话历史
        chat_history.add_user(user_input)
        chat_history.add_ai(answer)

        # ---- 输出回答 ----
        print("\n" + "-" * 60)
        print(answer)
        print("-" * 60)

        # 来源
        print_sources(source_docs)

        # 调试模式：显示原始片段
        if show_raw and source_docs:
            print(YELLOW + "\n[Debug]检索片段（原始）:" + RESET)
            for i, doc in enumerate(source_docs, 1):
                content_preview = doc.page_content[:200].replace("\n", " ")
                source = doc.metadata.get("source", "未知")
                print(f"  [{i}] {source}: {content_preview}...")

        # 日志记录
        sources_summary = [f"{d.metadata.get('source','?')}:{d.metadata.get('page','?')}" for d in source_docs]
        logger.info(f"回答生成完成 | 来源: {', '.join(sources_summary[:4])}")


if __name__ == "__main__":
    main()
