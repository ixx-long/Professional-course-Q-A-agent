# CourseQA Code Wiki

> 专业课程答疑智能体 — 完整代码参考文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [项目目录结构](#3-项目目录结构)
4. [核心模块详解](#4-核心模块详解)
5. [关键类与函数说明](#5-关键类与函数说明)
6. [模块依赖关系](#6-模块依赖关系)
7. [配置文件说明](#7-配置文件说明)
8. [运行方式](#8-运行方式)
9. [API 接口文档](#9-api-接口文档)
10. [前端架构](#10-前端架构)
11. [数据流与处理流程](#11-数据流与处理流程)

---

## 1. 项目概述

**CourseQA** 是一个基于 RAG（检索增强生成）的专业课程答疑系统，面向软件工程核心课程（数据结构、操作系统、计算机网络、软件工程概论等），解决通用大模型在专业领域的幻觉和知识过时问题。

### 核心处理流程

```
用户问题 → Embedding 检索 (Chroma, top_k=8) → CrossEncoder 重排序 (top_n=4) → LLM 生成 (DeepSeek) → 结构化回答
```

### 核心特性

| 特性 | 说明 |
|------|------|
| 学术诚信三级分类 | 红色（禁给答案）/ 黄色（给思路）/ 绿色（完整解答） |
| 课程筛选 | 按课程标签过滤检索范围 |
| 多模态问答 | 支持图片 + 文本提问 |
| 语音输入 | Web Speech API 中文语音识别 |
| 多用户隔离 | session_id + token 鉴权，对话持久化 |
| 降级策略 | CrossEncoder 加载失败自动回退基础检索 |
| 编码容错 | UTF-8 → GBK → GB2312 → errors=replace |
| 自动重试 | LLM 调用失败指数退避重试 |

---

## 2. 技术架构

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 生成模型 | DeepSeek `deepseek-chat` | OpenAI 兼容接口 |
| Embedding | 阿里云百炼 `text-embedding-v3` | 1024 维，中文优化 |
| 向量库 | Chroma（持久化） | 轻量，Python 原生 |
| 重排序 | `BAAI/bge-reranker-base` | 中文 CrossEncoder |
| 框架 | LangChain | ConversationalRetrievalChain |
| 前端 | Vanilla JS + Flask | 零依赖，ES 模块 |
| 文档解析 | PyPDF / docx2txt / TextLoader | 支持 PDF/Word/Markdown |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (index.html)                     │
│   Vanilla JS · Markdown 渲染 · 语音输入 · 图片上传            │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST API
┌──────────────────────▼──────────────────────────────────────┐
│                   web_server.py (Flask)                       │
│   路由管理 · Session 管理 · Token 鉴权 · 课程筛选缓存         │
└──┬──────────┬──────────┬──────────┬─────────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌────────┐  ┌────────┐
│utils │  │loader│  │vector  │  │chain   │
│.py   │  │.py   │  │store.py│  │.py     │
│      │  │      │  │        │  │        │
│配置  │  │文档  │  │Embed   │  │Prompt  │
│日志  │  │加载  │  │Chroma  │  │LLM     │
│环境变量│ │分块  │  │Retriever│ │对话历史 │
└──────┘  └──────┘  └────┬───┘  └────────┘
                         │
                    ┌────▼────┐
                    │retriever│
                    │.py      │
                    │         │
                    │CrossEnc │
                    │重排序   │
                    └─────────┘
```

---

## 3. 项目目录结构

```
Professional course Q&A agent/
├── src/                          # 核心源码包
│   ├── __init__.py               # 包初始化
│   ├── utils.py                  # 配置加载、环境变量注入、日志、API Key 脱敏
│   ├── loader.py                 # 文档加载（PDF/Word/Markdown）、分块、课程自动检测
│   ├── vectorstore.py            # BailianEmbeddings 封装、Chroma 向量库管理
│   ├── retriever.py              # CrossEncoder 重排序、ContextualCompressionRetriever
│   └── chain.py                  # System Prompt、LLM 创建、ConversationalRetrievalChain、ChatHistory
│
├── templates/
│   └── index.html                # Web 前端（学术书房风格，Vanilla JS）
│
├── data/
│   └── documents/                # 课程文档存放目录（按子目录自动标注课程标签）
│       └── demo.md
│
├── prompt/
│   ├── prompt.md                 # 原始需求 Prompt
│   └── prompt-optimized.md       # 优化后的 Prompt
│
├── docs/
│   └── superpowers/
│       ├── plans/                # 实施计划文档
│       └── specs/                # 设计规格文档
│
├── web_server.py                 # Flask Web 服务入口（API + 前端页面）
├── qa_cli.py                     # 命令行交互问答入口
├── build_kb.py                   # 知识库构建脚本（文档 → 向量化 → Chroma）
├── eval.py                       # RAG 评估脚本（LLM 评判答案质量）
├── config.example.yaml           # 配置模板
├── requirements.txt              # Python 依赖清单
├── demo.html                     # 演示页面
├── .gitignore
└── README.md
```

---

## 4. 核心模块详解

### 4.1 `src/utils.py` — 工具模块

**职责**：配置加载、环境变量注入、日志初始化、API Key 脱敏。被所有其他 src 模块依赖。

**环境变量映射**（优先级：环境变量 > config.yaml > 默认值）：

| 环境变量 | 配置路径 |
|---------|---------|
| `DEEPSEEK_API_KEY` | `llm.api_key` |
| `DEEPSEEK_API_BASE` | `llm.api_base` |
| `DEEPSEEK_MODEL` | `llm.model_name` |
| `BAILIAN_API_KEY` | `embedding.api_key` |
| `BAILIAN_API_BASE` | `embedding.api_base` |
| `BAILIAN_MODEL` | `embedding.model_name` |

**关键函数**：

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_config` | `(config_path: str) -> Dict[str, Any]` | 加载 YAML 配置，解析 `${VAR}` 占位符，校验必填段和数值合理性 |
| `_resolve_from_env` | `(config: Dict) -> Dict` | 用环境变量覆盖配置中的敏感字段，支持 `${VAR}` 占位符语法 |
| `mask_key` | `(key: str) -> str` | API Key 脱敏，仅显示前 6 和后 4 个字符 |
| `setup_logger` | `(name, log_file, level) -> Logger` | 初始化日志记录器，同时输出到文件和控制台 |

**配置校验规则**：
- 必须包含 7 个顶层段：`llm`, `embedding`, `chroma`, `retrieval`, `reranker`, `memory`, `logging`
- `llm.api_key` 和 `embedding.api_key` 不能为空
- `retrieval.top_k` 范围 1-100
- `retrieval.rerank_top_n` 范围 1-top_k
- `memory.max_turns` 范围 1-50

---

### 4.2 `src/loader.py` — 文档加载与分块模块

**职责**：遍历目录加载文档、自动检测课程标签、文本分块。

**支持格式**：

| 扩展名 | Loader | 说明 |
|--------|--------|------|
| `.pdf` | `PyPDFLoader` | 逐页加载 |
| `.docx` | `Docx2txtLoader` | Word 文档 |
| `.md` | `TextLoader` | Markdown / 纯文本 |
| `.txt` | `TextLoader` | 纯文本 |

**关键函数**：

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_documents` | `(input_dir, chunk_size=1000, chunk_overlap=200) -> List[Document]` | 主入口：递归遍历目录，加载所有文档并分块 |
| `load_single_document` | `(file_path: Path) -> List[Document]` | 加载单个文件，返回 Document 列表（未分块） |
| `split_documents` | `(docs, chunk_size, chunk_overlap) -> List[Document]` | 使用 RecursiveCharacterTextSplitter 分块，生成 chunk_id 用于去重 |
| `_detect_course` | `(file_path: Path) -> str \| None` | 从路径中自动检测课程标签（基于关键词匹配目录名） |
| `_load_text_with_fallback` | `(file_path, loader_cls) -> list` | 编码容错加载：UTF-8 → GBK → GB2312 → errors=replace |
| `_get_loader` | `(file_path: Path) -> type` | 根据扩展名返回对应 Loader 类 |

**自动课程检测关键词**：`数据结构`, `操作系统`, `计算机网络`, `软件工程`, `计算机组成`, `数据库`, `编译原理`

**分块策略**：
- 分隔符优先级：`\n\n` > `\n` > `。` > `；` > `.` > `;` > ` ` > `""`
- 每个 chunk 的 metadata 包含：`source`（文件名）、`page`（页码）、`chunk_id`（MD5 哈希去重）、`chunk_index`

---

### 4.3 `src/vectorstore.py` — 向量库管理模块

**职责**：Embedding 模型创建、Chroma 向量库初始化/加载、文档去重入库、获取 Retriever。

#### `BailianEmbeddings` 类

阿里云百炼 Embedding 封装，基于 OpenAI 兼容接口。绕过 LangChain `OpenAIEmbeddings` 的兼容性问题。

| 方法 | 说明 |
|------|------|
| `__init__(api_key, base_url, model, batch_size=25)` | 初始化 OpenAI 客户端，配置重试和超时 |
| `embed_documents(texts) -> list[list[float]]` | 批量生成文档嵌入，自动分批 + 重试 3 次 |
| `embed_query(text) -> list[float]` | 生成查询嵌入，自动重试 |

**关键函数**：

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_embedding_model` | `(config) -> Embeddings` | 根据配置创建 Embedding（API 模式 or 本地 sentence-transformers） |
| `get_vectorstore` | `(config, embedder=None) -> Chroma` | 初始化/加载 Chroma 向量库（持久化模式） |
| `get_existing_chunk_ids` | `(vectorstore) -> set[str]` | 获取已存在的 chunk_id 集合（用于去重） |
| `add_documents` | `(vectorstore, documents, skip_existing=True) -> int` | 向向量库添加文档，支持基于 chunk_id 的去重 |
| `get_retriever` | `(vectorstore, top_k=8) -> VectorStoreRetriever` | 创建配置了检索参数的 Retriever |

**去重机制**：每个 chunk 的 `chunk_id` 格式为 `{source}#p{page}#{md5_hash[:8]}`，入库前检查是否已存在。

---

### 4.4 `src/retriever.py` — 检索与重排序模块

**职责**：CrossEncoder 加载、候选文档重排序、ContextualCompressionRetriever 封装。

**关键函数**：

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_cross_encoder` | `(model_name, cache_dir="./models") -> CrossEncoder` | 加载 CrossEncoder 模型，首次运行自动下载 |
| `_rerank_with_cross_encoder` | `(query, documents, cross_encoder, top_n=4) -> list[Document]` | 对候选文档重新打分排序，返回 top_n |
| `retrieve_and_rerank` | `(query, retriever, cross_encoder, top_n=4) -> (docs, raw_docs)` | 完整检索 + 重排序流程 |
| `create_compression_retriever` | `(retriever, cross_encoder, config) -> ContextualCompressionRetriever` | LangChain 压缩 Retriever 封装，用于链内集成 |

**重排序流程**：
1. Chroma Retriever 检索 top_k 个候选文档
2. CrossEncoder 对 `(query, doc_content)` 对打分
3. 按分数降序排列，取 top_n
4. 将 `rerank_score` 写入文档 metadata

**降级策略**：CrossEncoder 加载失败时，自动回退到基础 Retriever（无重排序）。

---

### 4.5 `src/chain.py` — 对话链模块

**职责**：System Prompt 构建、LLM 创建、ConversationalRetrievalChain 组装、对话历史管理。

#### System Prompt 结构

```
你是一个专业课程答疑助手...

## 核心规则
### 1. 以知识库为准
### 2. 学术诚信（三级分类）
  - 🔴 红色：直接求答案 → 仅给概念提示
  - 🟡 黄色：问思路 → 分析思路、伪代码
  - 🟢 绿色：问概念 → 完整解释
### 3. 回答格式
  【依据】...
  【解答】...
  【来源】...
### 4. 补充规则

## Few-shot 示例（3个）

{context}  ← 检索到的知识库片段
```

#### `ChatHistory` 类

管理对话历史，支持滑动窗口。

| 方法 | 说明 |
|------|------|
| `__init__(max_turns=4)` | 初始化，设置最大轮次 |
| `add_user(content)` | 记录用户消息（HumanMessage） |
| `add_ai(content)` | 记录 AI 回复（AIMessage） |
| `get_history() -> list` | 获取最近 max_turns 轮消息 |
| `clear()` | 清空历史 |

**关键函数**：

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_llm` | `(config) -> ChatOpenAI` | 创建 LLM 实例（DeepSeek 兼容接口） |
| `create_qa_chain` | `(retriever, config) -> ConversationalRetrievalChain` | 创建检索问答链，含 condense question 压缩 |
| `format_source_documents` | `(source_docs) -> str` | 格式化来源引用信息 |

**ConversationalRetrievalChain 配置**：
- `return_source_documents=True`：返回检索到的原始文档
- `condense_question_prompt`：将历史对话 + 当前问题压缩为独立问句
- 手动管理 `chat_history`（非 LangChain Memory）

---

### 4.6 `web_server.py` — Web 服务入口

**职责**：Flask API 服务、Session 管理、Token 鉴权、课程筛选缓存、多模态问答。

**全局状态**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `qa_chain` | `ConversationalRetrievalChain` | 全局对话链 |
| `sessions` | `dict[str, ChatHistory]` | session_id → 对话历史（多用户隔离） |
| `_filtered_chains` | `dict[str, Chain]` | 课程筛选 Chain 缓存（LRU，上限 20） |
| `_session_tokens` | `dict[str, str]` | session_id → token（简单鉴权） |
| `SESSIONS_FILE` | `Path` | 对话历史持久化文件（`data/sessions.json`） |

**关键内部函数**：

| 函数 | 说明 |
|------|------|
| `init_system(config_path)` | 初始化所有 RAG 组件（Embedding、向量库、Retriever、CrossEncoder、LLM、Chain） |
| `_invoke_with_retry(chain, inputs, max_retries=2)` | 带指数退避重试的 chain.invoke |
| `_validate_token(session_id) -> bool` | 校验 session 归属，首次请求自动注册 token |
| `_get_session(session_id) -> ChatHistory` | 获取/创建 session（线程安全，LRU 淘汰，上限 100） |
| `_save_sessions()` | 持久化所有 session 到 JSON（线程安全） |
| `_load_sessions()` | 从 JSON 恢复对话历史 |
| `_save_sessions_throttled()` | 节流写盘（30 秒间隔）+ 脏标记 |
| `_validate_image(image_b64) -> str \| None` | 校验图片 base64（格式、大小 10MB） |
| `_get_filtered_retriever(course)` | 获取带课程筛选的 retriever（metadata filter） |
| `_ask_with_image(question, image_b64, chat_history)` | 多模态问答：统一检索 + 图片 + LLM 直调 |
| `_extract_sources(source_docs) -> list` | 提取去重的来源信息 |

**并发安全机制**：
- `sessions_lock`：保护 sessions 字典的读写
- `_filtered_chains_lock`：保护课程 Chain 缓存
- LRU 淘汰：sessions 上限 100，Chain 缓存上限 20

---

### 4.7 `build_kb.py` — 知识库构建脚本

**职责**：遍历文档目录 → 加载 → 分块 → 生成向量嵌入 → 写入 Chroma。

**命令行参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir / -i` | 文档目录路径（必填） | - |
| `--config / -c` | 配置文件路径 | `config.yaml` |
| `--chunk_size` | 文本分块大小 | 1000 |
| `--chunk_overlap` | 文本分块重叠 | 200 |
| `--force` | 强制全量重建（清空旧集合） | False |
| `--course` | 手动标注课程标签 | None |

**执行流程**：
1. 加载配置 → 2. 加载文档（自动检测课程标签）→ 3. 初始化向量库 → 4. 生成嵌入并写入

---

### 4.8 `qa_cli.py` — 命令行交互问答

**职责**：命令行交互式问答，支持特殊命令。

**特殊命令**：

| 命令 | 说明 |
|------|------|
| `/reset` | 清空对话记忆 |
| `/sources` | 切换是否显示详细引用 |
| `/exit` | 退出程序 |

**初始化流程**：加载配置 → Embedding + 向量库 → CrossEncoder（可选降级）→ ChatHistory → QA Chain → 交互循环

---

### 4.9 `eval.py` — RAG 评估脚本

**职责**：读取 JSON 测试集，逐条问答后用 LLM 评判答案质量（1-5 分），输出评估报告。

**评判维度**：
1. 准确性：回答内容是否正确
2. 完整性：是否覆盖关键知识点
3. 引用质量：是否标注知识库来源
4. 学术诚信：是否避免了直接给完整代码/作业答案

**测试集格式**：
```json
[
  {"question": "什么是堆排序？", "reference_answer": "堆排序是基于二叉堆的比较排序…"}
]
```

**输出指标**：平均分、优秀率（≥4分）、失败数、逐条详情表

---

## 5. 关键类与函数说明

### 类清单

| 类名 | 所在模块 | 说明 |
|------|---------|------|
| `BailianEmbeddings` | `src/vectorstore.py` | 阿里云百炼 Embedding 封装，实现 LangChain `Embeddings` 接口 |
| `ChatHistory` | `src/chain.py` | 对话历史管理器，滑动窗口保留最近 N 轮消息 |

### 函数清单（按模块）

#### `src/utils.py`
| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load_config` | `config_path: str` | `Dict[str, Any]` | 加载 YAML 配置 + 环境变量注入 + 校验 |
| `_resolve_from_env` | `config: Dict` | `Dict` | 解析 `${VAR}` 占位符 |
| `mask_key` | `key: str` | `str` | API Key 脱敏 |
| `setup_logger` | `name, log_file, level` | `Logger` | 初始化日志 |

#### `src/loader.py`
| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load_documents` | `input_dir, chunk_size, chunk_overlap` | `List[Document]` | 主入口：加载 + 分块 |
| `load_single_document` | `file_path: Path` | `List[Document]` | 加载单个文件 |
| `split_documents` | `docs, chunk_size, chunk_overlap` | `List[Document]` | 分块 + 生成 chunk_id |
| `_detect_course` | `file_path: Path` | `str \| None` | 从路径检测课程 |
| `_load_text_with_fallback` | `file_path, loader_cls` | `list` | 编码容错加载 |

#### `src/vectorstore.py`
| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `get_embedding_model` | `config` | `Embeddings` | 创建 Embedding 实例 |
| `get_vectorstore` | `config, embedder` | `Chroma` | 初始化 Chroma 向量库 |
| `get_existing_chunk_ids` | `vectorstore` | `set[str]` | 获取已有 chunk_id |
| `add_documents` | `vectorstore, documents, skip_existing` | `int` | 去重入库 |
| `get_retriever` | `vectorstore, top_k` | `VectorStoreRetriever` | 创建 Retriever |

#### `src/retriever.py`
| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load_cross_encoder` | `model_name, cache_dir` | `CrossEncoder` | 加载重排序模型 |
| `_rerank_with_cross_encoder` | `query, documents, cross_encoder, top_n` | `list[Document]` | CrossEncoder 重排序 |
| `retrieve_and_rerank` | `query, retriever, cross_encoder, top_n` | `(docs, raw_docs)` | 检索 + 重排序 |
| `create_compression_retriever` | `retriever, cross_encoder, config` | `ContextualCompressionRetriever` | LangChain 压缩封装 |

#### `src/chain.py`
| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `get_llm` | `config` | `ChatOpenAI` | 创建 LLM 实例 |
| `create_qa_chain` | `retriever, config` | `ConversationalRetrievalChain` | 创建对话链 |
| `format_source_documents` | `source_docs` | `str` | 格式化来源引用 |

---

## 6. 模块依赖关系

### 内部依赖图

```
build_kb.py ──→ src/utils.py (load_config, setup_logger)
            ──→ src/loader.py (load_documents)
            ──→ src/vectorstore.py (get_embedding_model, get_vectorstore, add_documents)

qa_cli.py ──→ src/utils.py (load_config, setup_logger)
          ──→ src/vectorstore.py (get_embedding_model, get_vectorstore, get_retriever)
          ──→ src/retriever.py (load_cross_encoder, create_compression_retriever)
          ──→ src/chain.py (create_qa_chain, ChatHistory, format_source_documents)

web_server.py ──→ src/utils.py (load_config, setup_logger)
            ──→ src/vectorstore.py (get_embedding_model, get_vectorstore, get_retriever)
            ──→ src/chain.py (create_qa_chain, ChatHistory, get_llm)
            ──→ src/retriever.py (load_cross_encoder, create_compression_retriever) [可选]

eval.py ──→ src/utils.py (load_config, setup_logger)
        ──→ src/vectorstore.py (get_embedding_model, get_vectorstore, get_retriever)
        ──→ src/retriever.py (load_cross_encoder, create_compression_retriever) [可选]
        ──→ src/chain.py (create_qa_chain, ChatHistory, get_llm)
```

### 模块间依赖矩阵

| 被依赖方 → | utils | loader | vectorstore | retriever | chain |
|-----------|-------|--------|-------------|-----------|-------|
| **依赖方 ↓** | | | | | |
| `utils` | - | | | | |
| `loader` | | - | | | |
| `vectorstore` | | | - | | |
| `retriever` | | | | - | |
| `chain` | | | | | - |
| `build_kb.py` | ✓ | ✓ | ✓ | | |
| `qa_cli.py` | ✓ | | ✓ | ✓ | ✓ |
| `web_server.py` | ✓ | | ✓ | ✓ | ✓ |
| `eval.py` | ✓ | | ✓ | ✓ | ✓ |

### 外部依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `langchain` | 0.3.13 | RAG 框架核心 |
| `langchain-openai` | 0.2.14 | ChatOpenAI（LLM 调用） |
| `langchain-chroma` | 0.2.1 | Chroma 向量库集成 |
| `langchain-text-splitters` | 0.3.5 | 文本分块 |
| `langchain-classic` | 0.3.13 | ConversationalRetrievalChain、ContextualCompressionRetriever |
| `chromadb` | 0.5.23 | 向量数据库 |
| `sentence-transformers` | 3.3.1 | CrossEncoder 重排序 |
| `openai` | >=1.0 | BailianEmbeddings 底层调用 |
| `pypdf` | 5.1.0 | PDF 文档解析 |
| `docx2txt` | 0.8 | Word 文档解析 |
| `pyyaml` | 6.0.2 | YAML 配置解析 |
| `flask` | 3.1.0 | Web 服务 |
| `tqdm` | 4.67.1 | 进度条 |
| `colorama` | 0.4.6 | CLI 彩色输出 |

---

## 7. 配置文件说明

配置文件为 YAML 格式，模板见 `config.example.yaml`。

### 完整配置段

```yaml
# 生成模型配置（DeepSeek）
llm:
  api_key: "${DEEPSEEK_API_KEY}"      # 支持环境变量或 ${VAR} 占位符
  api_base: "https://api.deepseek.com"
  model_name: "deepseek-chat"
  temperature: 0.1                     # 低温度保证准确性
  max_tokens: 2048

# Embedding 模型配置（阿里云百炼）
embedding:
  api_key: "${BAILIAN_API_KEY}"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model_name: "text-embedding-v3"      # 1024 维，中文优化

# 向量库配置
chroma:
  persist_dir: "./data/chroma_db"      # 持久化目录
  collection_name: "course_materials"  # 集合名称

# 检索配置
retrieval:
  top_k: 8                             # 初始检索候选数
  rerank_top_n: 4                      # 重排序后保留数

# 重排序模型
reranker:
  model_name: "BAAI/bge-reranker-base" # 中文 CrossEncoder
  cache_dir: "./models"

# 对话记忆
memory:
  max_turns: 4                         # 保留最近 4 轮对话

# 日志
logging:
  level: "INFO"
  file: "./logs/qa.log"
```

### 配置校验规则

| 字段 | 校验规则 |
|------|---------|
| `llm.api_key` | 不能为空 |
| `embedding.api_key` | 不能为空 |
| `llm.api_base` | 不能为空 |
| `embedding.api_base` | 不能为空；设为 `"local"` 时切换本地 sentence-transformers |
| `retrieval.top_k` | 1-100 |
| `retrieval.rerank_top_n` | 1-top_k |
| `memory.max_turns` | 1-50 |

---

## 8. 运行方式

### 前置准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
# 方式 A：环境变量（推荐）
export DEEPSEEK_API_KEY=sk-xxx
export BAILIAN_API_KEY=sk-xxx

# 方式 B：配置文件
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入 API Key

# 3. 准备课程文档
# 将 PDF/Word/Markdown 文件放入 data/documents/ 子目录
```

### 构建知识库

```bash
python build_kb.py --input_dir ./data/documents

# 可选参数
python build_kb.py -i ./data/documents --chunk_size 800 --chunk_overlap 150
python build_kb.py -i ./data/documents --force          # 全量重建
python build_kb.py -i ./data/documents --course 数据结构  # 手动标注课程
```

### 启动 Web 服务

```bash
python web_server.py
python web_server.py --port 8080 --host 0.0.0.0  # 自定义端口
python web_server.py --debug                       # 调试模式
```

访问 `http://127.0.0.1:5000`

### 命令行问答

```bash
python qa_cli.py
python qa_cli.py --show_raw   # 显示原始检索片段
```

### 运行评估

```bash
python eval.py --test_file ./tests/questions.json --output report.md
python eval.py -t ./tests/questions.json --sample 10  # 仅评估前 10 条
```

---

## 9. API 接口文档

### 基础信息

- 基础 URL：`http://127.0.0.1:5000`
- 鉴权方式：请求头 `X-Session-Token`（首次请求自动注册）

### 接口列表

#### `GET /` — 前端页面

返回 `templates/index.html`，使用 `send_file` 避免 Jinja2 模板注入风险。

#### `POST /api/ask` — 问答接口

**请求体**（JSON）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 否* | 问题文本 |
| `image` | string | 否* | 图片 base64（data URL 或纯 base64） |
| `course` | string | 否 | 课程筛选（如 "数据结构"），"all" 或省略表示全部 |
| `session_id` | string | 否 | 会话 ID，默认 "default" |

> *`question` 和 `image` 至少提供一个

**响应**（JSON）：

```json
{
  "answer": "【依据】...【解答】...【来源】...",
  "sources": [
    {"file": "tree.pdf", "page": 42, "score": 0.8765}
  ]
}
```

**错误码**：

| HTTP 状态码 | 说明 |
|------------|------|
| 400 | 请求体为空 / 问题为空 / 图片无效 / 模型不支持图片 |
| 403 | 无权操作此会话（token 不匹配） |
| 429 | API 限流 |
| 500 | API Key 无效 / 服务内部错误 |
| 503 | 系统未初始化 |
| 504 | 请求超时 |

#### `GET /api/history` — 获取对话历史

**查询参数**：`session_id`（可选）

**响应**：

```json
{
  "messages": [
    {"role": "user", "content": "什么是死锁？"},
    {"role": "bot", "content": "【依据】..."}
  ]
}
```

#### `POST /api/reset` — 重置对话记忆

**请求体**：`{"session_id": "xxx"}`

**响应**：`{"status": "ok", "message": "对话历史已清空"}`

---

## 10. 前端架构

前端为单文件 `templates/index.html`，零框架依赖（Vanilla JS）。

### 布局结构

```
┌──────────┬────────────────────────────────────┐
│ Sidebar  │  Header（状态栏）                    │
│          ├────────────────────────────────────┤
│ ·品牌    │                                    │
│ ·新对话  │  Chat（消息列表）                    │
│ ·课程筛选│  · Welcome 欢迎页                   │
│ ·历史列表│  · 用户消息（右对齐，绿色气泡）       │
│          │  · Bot 回复（左对齐，白色气泡）       │
│          │  · 参考文献脚注                     │
│          ├────────────────────────────────────┤
│          │  Input Zone                         │
│          │  [语音] [图片] [文本输入...] [发送]   │
└──────────┴────────────────────────────────────┘
```

### 核心功能模块

| 模块 | 实现方式 |
|------|---------|
| Session 管理 | `localStorage` 持久化 session_id + token |
| 课程筛选 | 侧边栏 chip 按钮，切换 `currentCourse` 状态 |
| 语音输入 | Web Speech API（`SpeechRecognition`），中文识别 |
| 图片上传 | `FileReader` → base64 → 预览 → 发送 |
| Markdown 渲染 | 自实现解析器（标题/粗体/代码块/表格/列表/引用/链接） |
| 对话恢复 | 页面加载时调用 `/api/history` 恢复消息 |
| 历史管理 | `localStorage` 存储会话标题列表 |
| 复制功能 | `navigator.clipboard` API |

### 设计风格

- **主题**：学术书房风格（深色侧边栏 + 羊皮纸色主区域）
- **字体**：Playfair Display（标题）+ Noto Sans SC（正文）+ JetBrains Mono（代码）
- **配色**：墨色 `#1a1a2e` + 青绿 `#0f766e` + 羊皮纸 `#f8f5f0`
- **响应式**：720px 以下隐藏侧边栏

---

## 11. 数据流与处理流程

### 文本问答完整流程

```
1. 用户在前端输入问题
   │
2. POST /api/ask {question, session_id, course?}
   │
3. _validate_token(session_id) → 鉴权
   │
4. _get_session(session_id) → 获取 ChatHistory
   │
5. 课程筛选？
   ├─ 是 → 从 _filtered_chains 缓存获取/创建带 filter 的 Chain
   └─ 否 → 使用全局 qa_chain
   │
6. chain.invoke({question, chat_history})
   │
   ├─ 6a. condense_question_prompt: 将历史+问题压缩为独立问句
   │
   ├─ 6b. Retriever 检索:
   │      Chroma similarity search → top_k=8 候选文档
   │      ↓
   │      CrossEncoder 重排序 → top_n=4 最相关文档
   │
   ├─ 6c. 构建 Prompt:
   │      System Prompt + {context}(检索片段) + 对话历史 + 用户问题
   │
   └─ 6d. LLM 生成回答 (DeepSeek)
   │
7. 更新 ChatHistory（add_user + add_ai）
   │
8. _save_sessions_throttled()（节流持久化）
   │
9. 返回 {answer, sources}
```

### 图片问答流程

```
1. 用户上传图 + 输入问题
   │
2. _validate_image(image_b64) → 校验格式/大小
   │
3. 检查模型是否支持视觉（vision/gpt-4/claude/gemini）
   │
4. 使用统一 Retriever 检索知识库上下文
   │
5. 构建多模态消息:
   SystemMessage + HumanMessage([text + image_url])
   │
6. LLM 直调（不经过 Chain）
   │
7. 返回回答 + 来源文档
```

### 知识库构建流程

```
1. 遍历 data/documents/ 目录
   │
2. 按扩展名选择 Loader（PDF/Word/MD/TXT）
   │
3. 自动检测课程标签（从目录名匹配关键词）
   │
4. 编码容错加载（UTF-8 → GBK → GB2312 → replace）
   │
5. RecursiveCharacterTextSplitter 分块
   │  chunk_size=1000, overlap=200
   │  生成 chunk_id = {source}#p{page}#{md5[:8]}
   │
6. 去重检查（对比已有 chunk_id）
   │
7. BailianEmbeddings 批量生成向量
   │  batch_size=25, 自动重试 3 次
   │
8. 写入 Chroma 向量库（持久化）
```

---

> **文档版本**：基于项目当前代码状态生成
> **Python 版本要求**：>= 3.10
> **License**：MIT
