# 基于 RAG 的专业课程答疑智能体 — 可执行规格书

> 原始需求见 `prompt.md`，本文件为优化后可执行版本。

---

## 〇、环境配置确认（已澄清）

| 组件 | 提供商 | 配置项 | 值 |
|------|--------|--------|-----|
| 生成模型 | DeepSeek | api_base | `https://api.deepseek.com` |
| 生成模型 | DeepSeek | model_name | `deepseek-chat` |
| Embedding | 阿里云百炼 | api_base | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Embedding | 阿里云百炼 | model_name | `text-embedding-v3` |
| Embedding | 阿里云百炼 | dimension | `1024` |
| 重排序 | 本地 | model_name | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

> **注意**：生成模型和 Embedding 使用**两套不同的 API Key 和 Base URL**，config.yaml 须分别配置。

---

## 分阶段交付计划

### MVP（优先实现）

- `build_kb.py`：知识库构建脚本
- `qa_cli.py`：命令行交互问答
- `src/` 下全部核心模块：`loader.py`、`vectorstore.py`、`retriever.py`、`chain.py`、`utils.py`
- 记忆模块：会话内保持，跨会话不持久化
- 日志：记录每次问答的检索片段和生成答案

### 第二阶段

- `eval.py`：评估脚本
- Web 界面（Gradio / Streamlit，配合 frontend-design 技能）
- 元数据过滤（按课程名、文档类型筛选）
- 持久化记忆

---

## 一、项目背景

构建面向软件工程专业核心课程（数据结构、操作系统、计算机网络、软件工程概论等）的智能答疑助手。核心指标：

- 专业知识问答，避免通用大模型幻觉与资料过时
- 支持 PDF、Word (.docx)、Markdown 三种格式批量导入
- 多轮对话与上下文记忆
- 专业问题回答准确率 ≥ 86%，错误率相比原生大模型下降 ≥ 42%

---

## 二、技术栈

| 层次 | 组件 | 说明 |
|------|------|------|
| 语言 | Python 3.10+ | — |
| 编排框架 | LangChain | 核心编排 |
| 生成模型 | DeepSeek API（`deepseek-chat`） | OpenAI 兼容接口，预留本地模型切换空间 |
| 向量数据库 | Chroma | 持久化模式 |
| 文档加载 | PyPDFLoader / Docx2txtLoader / TextLoader | Markdown 作为纯文本加载 |
| 文本分割 | RecursiveCharacterTextSplitter | chunk_size≈1000, chunk_overlap≈200 |
| Embedding | 阿里云百炼 `text-embedding-v3` | 1024 维；预留 sentence-transformers 切换 |
| 重排序 | CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` | 本地运行；或 LangChain ContextualCompressionRetriever |
| 记忆 | ConversationBufferMemory / ConversationSummaryBufferMemory | — |
| Web 界面 | Gradio 或 Streamlit（第二阶段） | — |

---

## 三、核心功能

### 1. 知识库构建（`build_kb.py`）

- 支持 **PDF、.docx、.md** 三种格式
- 按逻辑块（标题、段落）分块，chunk_size≈1000，chunk_overlap≈200
- 向量嵌入存入 Chroma（collection 名可配置）
- **增量入库**：以 `source + chunk_id` 去重，避免重复
- CLI：`python build_kb.py --input_dir ./data/documents --config config.yaml`

### 2. 检索链路

- Chroma retriever，top_k=8
- CrossEncoder 重排序，保留 top_n=4
- 可选元数据过滤（第二阶段）

### 3. 多轮对话与记忆

- 保留最近 4 轮交互
- 使用 `ConversationalRetrievalChain` 整合检索与记忆
- 回答以知识库内容为准，历史仅作上下文辅助
- 支持 `/reset` 清空记忆，`/exit` 退出

### 4. Prompt 工程与生成控制

**系统 Prompt 核心规则**：

1. 优先根据提供的知识库片段回答
2. 信息不足时明确告知"根据课程资料暂未找到答案"，禁止编造
3. 引用知识库来源（文件名或段落编号）
4. 遵守学术诚信三级分类规则

**生成参数**：

- temperature：默认 0.1（可配置）
- max_tokens：默认 2048（可配置）

### 5. 学术诚信规则（三级分类）

| 级别 | 问题类型 | 允许 | 禁止 |
|------|----------|------|------|
| 🔴 作业/考试原题 | 用户直接问某道作业题的完整答案 | 仅给出概念提示 | 完整代码、可直接提交的答案、逐行详解 |
| 🟡 思路求助 | 用户说"没思路""不知如何开始" | 分析思路、伪代码、方法对比 | 完整可运行代码 |
| 🟢 概念/知识点 | 概念含义、算法原理、协议流程 | 完整解释、示例代码片段 | 无 |

> 通过 Few-shot 示例让模型区分三类场景。

### 6. 评估（`eval.py`，第二阶段）

- 读取 `question` / `reference_answer` JSON 文件
- 以 LLM 评判为主（评分 1-5），人工抽样验证为辅
- 输出准确率报告和典型错误案例

### 7. 日志

- 记录每次问答的：时间、问题、检索片段（带来源）、生成答案
- 日志文件路径可配置

---

## 四、项目文件结构

```
course-qa-rag/
├── requirements.txt
├── config.yaml              # 双 API 配置
├── build_kb.py              # 知识库构建
├── qa_cli.py                # CLI 交互问答
├── eval.py                  # 评估脚本（第二阶段）
├── src/
│   ├── __init__.py
│   ├── loader.py            # 文档加载与分块
│   ├── vectorstore.py       # Chroma 初始化与增/查
│   ├── retriever.py         # 检索 + CrossEncoder 重排序
│   ├── chain.py             # 对话链 + 记忆 + Prompt
│   └── utils.py             # 日志、配置加载
├── prompt/
│   ├── prompt.md            # 原始需求
│   └── prompt-optimized.md  # 本文件
├── data/
│   ├── documents/           # 课程资料
│   └── chroma_db/           # 持久化向量库
└── logs/                    # 问答日志
```

---

## 五、关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 两套 API 分离配置 | DeepSeek + 百炼各自独立 section | API Key、Base URL、模型名均不同，混在一起易出错 |
| 重排序用本地模型 | CrossEncoder，不用 API | 避免额外网络调用，CPU 可运行 |
| 去重依据 | `source + chunk_id` | 简单可靠，无需额外哈希 |
| 记忆跨会话不持久化 | ConversationBufferMemory | MVP 聚焦核心功能，降低复杂度 |
| temperature=0.1 | 低温度 | 答疑场景需要稳定、可复现的输出 |

---

## 六、运行要求

- 无 GPU 可运行（Embedding 用 API，CrossEncoder 用 CPU）
- Chroma 持久化，避免重复计算 Embedding
- 重排序模型首次运行自动下载，需处理缓存路径
- 所有模块提供错误处理和中文注释
