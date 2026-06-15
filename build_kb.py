#!/usr/bin/env python3
"""
知识库构建脚本。

遍历指定目录下所有支持的文档（PDF/Word/Markdown），
加载、分块、生成向量嵌入并存入 Chroma 向量库。

用法:
    python build_kb.py --input_dir ./data/documents --config config.yaml
    python build_kb.py --input_dir ./data/documents --config config.yaml --chunk_size 800 --chunk_overlap 150
    python build_kb.py --input_dir ./data/documents --config config.yaml --force  # 强制全量重建
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 src 模块可导入
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, mask_key
from src.loader import load_documents
from src.vectorstore import get_embedding_model, get_vectorstore, add_documents


def main():
    parser = argparse.ArgumentParser(
        description="构建课程知识库 — 将文档向量化存入 Chroma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_kb.py --input_dir ./data/documents
  python build_kb.py --input_dir ./data/documents --config my_config.yaml
  python build_kb.py --input_dir ./data/documents --force  # 跳过去重，全量重建
        """,
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="文档目录路径（将递归遍历子目录）",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="文本分块大小（默认: 1000）",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="文本分块重叠大小（默认: 200）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制全量重建，不进行去重检查",
    )

    args = parser.parse_args()

    # ---- 加载配置 ----
    print("=" * 60)
    print("  课程知识库构建工具")
    print("=" * 60)

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[错误] 加载配置失败: {e}")
        sys.exit(1)

    # ---- 初始化日志 ----
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="build_kb",
        log_file=log_config.get("file", "logs/build_kb.log"),
        level=log_config.get("level", "INFO"),
    )

    # 脱敏显示 API Key
    llm_key = config.get("llm", {}).get("api_key", "")
    emb_key = config.get("embedding", {}).get("api_key", "")
    logger.info(f"LLM API Key: {mask_key(llm_key)}")
    logger.info(f"Embedding API Key: {mask_key(emb_key)}")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"分块大小: {args.chunk_size}, 重叠: {args.chunk_overlap}")

    # ---- 加载文档 ----
    print("\n[1/3] 加载文档...")
    try:
        documents = load_documents(
            input_dir=args.input_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        sys.exit(1)

    logger.info(f"加载完成: {len(documents)} 个文本块")
    print(f"  [OK] 共加载 {len(documents)} 个文本块")

    # 统计文件来源
    sources = set(doc.metadata.get("source", "未知") for doc in documents)
    print(f"  [OK] 来源文件数: {len(sources)}")
    for src in sorted(sources):
        print(f"    - {src}")

    # ---- 初始化向量库 ----
    print("\n[2/3] 初始化向量库...")
    try:
        embedder = get_embedding_model(config)
        vectorstore = get_vectorstore(config, embedder)
    except Exception as e:
        logger.error(f"初始化向量库失败: {e}")
        sys.exit(1)
    print("  [OK] Chroma 向量库已就绪")

    # ---- 添加文档 ----
    print("\n[3/3] 生成嵌入并写入向量库...")
    try:
        added = add_documents(
            vectorstore=vectorstore,
            documents=documents,
            skip_existing=not args.force,
        )
    except Exception as e:
        logger.error(f"写入向量库失败: {e}")
        sys.exit(1)

    if added == 0:
        print("  [OK] 所有文档已存在，无需更新")
    else:
        print(f"  [OK] 新增 {added} 个文本块")
        persist_dir = config.get("chroma", {}).get("persist_dir", "./data/chroma_db")
        print(f"  [OK] 向量库已持久化至: {persist_dir}")

    print("\n" + "=" * 60)
    print("  构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
