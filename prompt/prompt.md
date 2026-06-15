你是一名资深 AI 应用开发工程师，请基于以下需求，从零设计并实现一个**基于 RAG 的专业课程答疑智能体**。请给出完整项目代码、文件结构、依赖清单和运行说明。

## 一、项目背景
目标是构建一个面向软件工程专业核心课程（如数据结构、操作系统、计算机网络、软件工程概论等）的智能答疑助手。项目已在班级内部使用，需具备：
- 专业知识问答能力，避免通用大模型的幻觉与资料过时问题
- 支持多格式知识库批量导入
- 支持多轮对话与上下文记忆
- 高准确率：专业问题回答准确率 86%，错误率相比原生大模型下降 42%

## 二、技术栈要求
- Python 3.10+
- LangChain（核心编排）
- OpenAI API（或兼容接口）作为生成模型，也可使用本地模型，但需预留切换空间
- Chroma 向量数据库（持久化存储）
- 文档加载：PyPDFLoader、Docx2txtLoader、TextLoader（Markdown 作为纯文本）
- 文本分割：RecursiveCharacterTextSplitter
- Embedding：OpenAIEmbeddings（或 sentence-transformers 本地模型，需可配置）
- 重排序：使用 CrossEncoder（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）或 LangChain 的 `ContextualCompressionRetriever` 配合 `LLMChainExtractor` / `CrossEncoderReranker`
- 记忆模块：`ConversationBufferMemory` 或 `ConversationSummaryBufferMemory`
- Web 界面（可选，但提供命令行交互脚本作为核心演示）

## 三、核心功能要求

### 1. 知识库构建模块（离线/在线）
- 支持 **PDF、Word (.docx)、Markdown (.md)** 三种格式的文档上传
- 自动解析文档内容，按逻辑块（如按标题、段落）进行分块（chunk_size≈1000，chunk_overlap≈200）
- 生成向量嵌入并存入 Chroma 集合（collection 名可配置）
- 支持增量入库（检查已有文档避免重复，可用 source+chunk_id 去重）
- 提供一个脚本 `build_kb.py`，接收 `--input_dir` 参数，遍历目录下所有支持的文件，处理后存入向量库

### 2. 检索链路
- 使用 Chroma 作为 retriever，检索 top_k=8 个候选文档
- 引入**重排序机制**：对检索结果用 CrossEncoder 重新打分，保留 top_n=4 个最相关片段送入生成模型
- 支持检索时附带元数据过滤（如按课程名、文档类型过滤，可选实现）

### 3. 上下文记忆与多轮对话
- 实现对话记忆，保留最近 k 轮交互（例如最近 4 轮）
- 使用 LangChain 的 `ConversationChain` 或 `ConversationalRetrievalChain`，将记忆与检索问答结合
- 系统需将历史问答和当前问题一起考虑，但最终生成时以检索到的知识库内容为准

### 4. Prompt 工程与生成控制
- 设计高质量系统 Prompt，要求模型：
  - 优先根据提供的知识库片段回答问题
  - 若知识库信息不足，明确告知“根据课程资料暂未找到答案”，避免编造
  - 引用知识库来源（文件名或段落编号，若元数据中有则展示）
  - 对作业思路类问题，只给分析思路和方法，不直接给完整代码或答案（学术诚信导向）
- 提供清晰的结构化输出：先总结依据，再给出解答，最后附引用
- 可调节生成温度（默认 0.1，保证稳定性）

### 5. 命令行交互问答接口
- 提供 `qa_cli.py`，启动后进入交互式循环
- 用户输入问题，程序执行检索→重排序→生成回答，并打印引用来源
- 支持特殊命令：`/reset` 清空记忆，`/exit` 退出
- 记忆在会话内保持，跨会话不保存（持久化记忆可选实现）

### 6. 评估与日志（可选，但须预留接口）
- 提供 `eval.py` 脚本，读取一份包含 `question`、`reference_answer` 的 JSON 文件，自动计算回答准确率（基于 LLM 评判或简单的关键词匹配+人工抽样说明）。输出评估报告。
- 系统记录每次问答的检索片段、生成答案、用户反馈（可选），写入日志文件。

## 四、项目文件结构建议
course-qa-rag/
├── requirements.txt
├── config.yaml # API key, 模型名称, 路径等配置
├── build_kb.py # 知识库构建脚本
├── qa_cli.py # 命令行问答交互
├── eval.py # 评估脚本（可选）
├── src/
│ ├── init.py
│ ├── loader.py # 文档加载与分块
│ ├── vectorstore.py # Chroma 初始化与增/查
│ ├── retriever.py # 检索 + 重排序
│ ├── chain.py # 对话链 + 记忆
│ └── utils.py # 日志、配置加载等
└── data/
├── documents/ # 示例课程资料
└── chroma_db/ # 持久化向量库


## 五、输出要求
请按以下顺序生成内容并保证代码可运行：

1. **设计思路概述**（1-2 段）：说明整体架构、数据流、为什么这么设计。
2. **环境配置**：`requirements.txt` 中列出所有依赖及版本号建议。
3. **配置文件示例**：`config.yaml`，包含 openai_api_key、embedding_model、chroma_persist_dir、cross_encoder_model 等。
4. **核心代码实现**：
   - 文档加载与分块逻辑（`loader.py`），处理不同格式，返回 LangChain Document 列表，包含 metadata（source、page、chunk_id）。
   - 向量库管理（`vectorstore.py`），支持初始化、添加文档（去重）、获取 retriever。
   - 检索与重排序（`retriever.py`），实现基础检索 + CrossEncoder 重排序的封装函数，或使用 LangChain 的 `ContextualCompressionRetriever`。
   - 对话链（`chain.py`），基于 `ConversationalRetrievalChain`，整合上述 retriever、记忆和自定义 Prompt。
   - 入口脚本 `build_kb.py` 和 `qa_cli.py`，提供命令行参数解析（如 `--input_dir`、`--config`），并打印详细运行日志。
5. **运行指南**：说明从环境安装、配置 API key、准备文档到启动问答的全步骤。
6. **测试示例**：给出 2-3 个模拟的课程问题和期望的生成回答风格，用于验证系统效果。

## 六、特别注意事项
- 所有代码需添加适当的错误处理和注释，说明关键设计意图。
- Chroma 使用持久化模式，避免每次重新计算 embedding。
- 重排序模型首次运行时会自动下载，需在代码中处理模型缓存路径。
- 保证系统在无 GPU 的环境下也能运行（embedding 和 cross-encoder 使用 CPU，或可配置）。
- Prompt 模板需使用 LangChain 的 `PromptTemplate` 或 `ChatPromptTemplate`，并展示完整内容。

请开始输出，先给出架构概述，然后逐一交付各文件完整代码。