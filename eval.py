#!/usr/bin/env python3
"""
评估脚本 — 自动计算 RAG 问答准确率。

读取 JSON 测试集，逐条问答后用 LLM 评判答案质量（1-5分），输出评估报告。

用法:
    python eval.py --test_file ./tests/questions.json --config config.yaml
    python eval.py --test_file ./tests/questions.json --output report.md

测试集 JSON 格式:
[
  {"question": "什么是堆排序？", "reference_answer": "堆排序是基于二叉堆的比较排序…"}
]
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger
from src.vectorstore import get_embedding_model, get_vectorstore, get_retriever
from src.chain import create_qa_chain, ChatHistory, get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def load_test_cases(file_path: str) -> list:
    """加载测试用例。"""
    with open(file_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("测试文件必须是 JSON 数组")
    for i, c in enumerate(cases):
        if "question" not in c:
            raise ValueError(f"第 {i+1} 条缺少 question 字段")
    return cases


def judge_answer(llm, question: str, answer: str, reference: str | None) -> dict:
    """用 LLM 评判回答质量，返回评分和理由。"""
    ref_text = f"参考答案: {reference}" if reference else "（无参考答案，请根据常识评判）"
    prompt = f"""你是一名专业课程助教评审。请对以下问答进行评分。

**问题**: {question}

**系统回答**: {answer}

**{ref_text}**

请从以下维度综合评分（1-5分，5分为满分）：
1. 准确性：回答内容是否正确
2. 完整性：是否覆盖关键知识点
3. 引用质量：是否标注知识库来源
4. 学术诚信：是否避免了直接给完整代码/作业答案

输出格式（严格按此格式）：
评分: X
理由: （一句话说明）"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()

    score = 3  # 默认
    reason = text
    for line in text.split("\n"):
        if line.startswith("评分:") or line.startswith("评分："):
            try:
                score = int("".join(c for c in line if c.isdigit()))
            except ValueError:
                pass
        if line.startswith("理由:") or line.startswith("理由："):
            reason = line.split(":", 1)[-1].split("：", 1)[-1].strip()

    return {"score": min(5, max(1, score)), "reason": reason}


def main():
    parser = argparse.ArgumentParser(description="RAG 问答评估")
    parser.add_argument("--test_file", "-t", required=True, help="测试集 JSON 文件")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件")
    parser.add_argument("--output", "-o", default=None, help="输出报告文件（可选，支持 .md/.txt）")
    parser.add_argument("--sample", type=int, default=0, help="仅评估前 N 条（0=全部）")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 问答评估")
    print("=" * 60)

    # 加载配置和初始化
    config = load_config(args.config)
    logger = setup_logger(
        name="eval",
        log_file=config.get("logging", {}).get("file", "logs/eval.log"),
        level="INFO",
    )

    print("初始化系统组件...")
    embedder = get_embedding_model(config)
    vectorstore = get_vectorstore(config, embedder)
    retriever = get_retriever(vectorstore, top_k=config["retrieval"]["top_k"])
    llm = get_llm(config)
    qa_chain = create_qa_chain(retriever, config)

    # 加载测试用例
    cases = load_test_cases(args.test_file)
    if args.sample > 0:
        cases = cases[:args.sample]
    print(f"加载 {len(cases)} 条测试用例")

    # 逐条评估
    results = []
    total_score = 0
    chat_history = ChatHistory()

    for i, case in enumerate(cases, 1):
        question = case["question"]
        reference = case.get("reference_answer", "")
        print(f"\n[{i}/{len(cases)}] {question[:50]}...")

        try:
            # QA
            start = time.time()
            result = qa_chain.invoke({
                "question": question,
                "chat_history": chat_history.get_history(),
            })
            elapsed = time.time() - start
            answer = result.get("answer", "")
            sources = result.get("source_documents", [])

            # 评判
            judgment = judge_answer(llm, question, answer, reference)
            score = judgment["score"]
            total_score += score

            results.append({
                "index": i,
                "question": question,
                "answer": answer[:300],
                "reference": reference[:200] if reference else "",
                "score": score,
                "reason": judgment["reason"],
                "sources": [d.metadata.get("source", "?") for d in sources],
                "time": round(elapsed, 2),
            })

            print(f"  评分: {score}/5 | 耗时: {elapsed:.1f}s | {judgment['reason'][:50]}")

            # 更新历史（仅保留上一轮，避免上下文混淆）
            chat_history.add_user(question)
            chat_history.add_ai(answer)

        except Exception as e:
            logger.error(f"[{i}] 评估失败: {e}")
            results.append({
                "index": i,
                "question": question,
                "error": str(e),
                "score": 0,
            })
            print(f"  失败: {e}")

    # 汇总
    valid = [r for r in results if r.get("score", 0) > 0]
    avg_score = total_score / len(valid) if valid else 0
    pass_count = sum(1 for r in valid if r["score"] >= 4)

    report = f"""# RAG 问答评估报告

- **测试用例**: {len(cases)} 条
- **有效评估**: {len(valid)} 条
- **平均分**: {avg_score:.2f}/5
- **优秀率** (≥4分): {pass_count}/{len(valid)} ({pass_count/len(valid)*100:.1f}%)
- **失败**: {len(cases) - len(valid)} 条

## 逐条详情

| # | 问题 | 评分 | 耗时 | 来源 |
|---|------|------|------|------|
"""
    for r in results:
        q = r["question"][:30]
        if "error" in r:
            report += f"| {r['index']} | {q} | ❌ {r['error'][:30]} | - | - |\n"
        else:
            srcs = ", ".join(set(r.get("sources", [])))[:30]
            report += f"| {r['index']} | {q} | {r['score']}/5 | {r['time']}s | {srcs} |\n"

    print("\n" + "=" * 60)
    print(f"  平均分: {avg_score:.2f}/5 | 优秀率: {pass_count}/{len(valid)}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已保存至: {out_path}")

    # 返回非零退出码如果分数太低
    if avg_score < 3.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
