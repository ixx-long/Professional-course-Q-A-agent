# 专业课程答疑 RAG 智能体 — 设计文档

> 日期：2026-06-15 | 状态：已确认

---

## 1. 概述

构建面向软件工程专业核心课程（数据结构、操作系统、计算机网络、软件工程概论等）的 RAG 答疑助手。核心指标：

- 专业问题回答准确率 ≥ 86%
- 错误率相比原生大模型下降 ≥ 42%
- 支持 PDF/Word/Markdown 批量导入
- 多轮对话 + 上下文记忆

---

## 2. 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 生成模型 | DeepSeek `deepseek-chat` | OpenAI 兼容接口 |
| Embedding | 阿里云百炼 `text-embedding-v3` | 1024 维 |
| 向量库 | Chroma（持久化） | 轻量，Python 原生 |
| 重排序 | CrossEncoder `ms-marco-MiniLM-L-6-v2` | 本地 CPU 运行 |
| 编排 | LangChain | ConversationalRetrievalChain |
| 记忆 | ConversationBufferMemory | 最近 4 轮 |
| Web UI | Gradio（第二阶段） | — |

---

## 3. 架构

### 3.1 数据流

```
离线：documents/ → loader → split → embed → Chroma
在线：用户问题 → retriever(top8) → reranker(top4) → chain(历史+片段+Prompt) → DeepSeek → 回答
```

### 3.2 模块依赖

```
utils ← loader ← vectorstore ← retriever ← chain
(无依赖)  (utils)   (loader,utils)  (vectorstore) (retriever,utils)
```

### 3.3 模块职责

| 模块 | 职责 | 对外接口 |
|------|------|----------|
| `utils.py` | YAML 配置加载、日志初始化 | `load_config()`, `setup_logger()` |
| `loader.py` | 多格式加载 + 分块 → `List[Document]` | `load_documents(input_dir)` |
| `vectorstore.py` | Chroma 初始化、添加（去重）、retriever | `get_vectorstore()`, `add_documents()`, `get_retriever()` |
| `retriever.py` | 检索 + CrossEncoder 重排序 | `retrieve_and_rerank(query, retriever, k=4)` |
| `chain.py` | 创建 ConversationalRetrievalChain | `create_qa_chain(retriever, memory)` |

---

## 4. 配置文件结构

```yaml
llm:        # DeepSeek
  api_key, api_base, model_name, temperature: 0.1, max_tokens: 2048
embedding:  # 阿里云百炼
  api_key, api_base, model_name
chroma:
  persist_dir, collection_name
retrieval:
  top_k: 8, rerank_top_n: 4
reranker:
  model_name, cache_dir
memory:
  max_turns: 4
logging:
  level, file
```

---

## 5. 分阶段交付

### MVP
- `build_kb.py` + `qa_cli.py` + `src/` 全部模块
- CLI 交互问答
- 会话内记忆（不跨会话持久化）
- 问答日志

### 第二阶段
- `eval.py` 评估脚本
- Web 界面（Gradio）
- 元数据过滤
- 持久化记忆

---

## 6. 学术诚信规则

| 级别 | 场景 | 允许 | 禁止 |
|------|------|------|------|
| 🔴 | 作业原题 | 概念提示 | 完整答案、逐行详解 |
| 🟡 | 思路求助 | 思路分析、伪代码 | 完整可运行代码 |
| 🟢 | 概念知识点 | 完整解释、示例片段 | 无 |

---

## 7. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 两套 API 分离配置 | 独立 section | Key/URL/模型名均不同 |
| 重排序本地运行 | CrossEncoder | CPU 可用，无额外网络成本 |
| 去重依据 | source + chunk_id | 简单可靠 |
| temperature=0.1 | 低温度 | 答疑需稳定可复现 |
| 记忆暂不持久化 | BufferMemory | MVP 降复杂度 |
