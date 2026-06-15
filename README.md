# CourseQA — 专业课程答疑智能体

基于 RAG（检索增强生成）的专业课程答疑系统，面向软件工程核心课程（数据结构、操作系统、计算机网络、软件工程概论等），解决通用大模型在专业领域的幻觉和知识过时问题。

## 技术架构

```
用户问题 → Embedding 检索 (Chroma, top_k=8) → CrossEncoder 重排序 (top_n=4) → LLM 生成 (DeepSeek) → 结构化回答
```

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 生成模型 | DeepSeek `deepseek-chat` | OpenAI 兼容接口 |
| Embedding | 阿里云百炼 `text-embedding-v3` | 1024 维，中文优化 |
| 向量库 | Chroma（持久化） | 轻量，Python 原生 |
| 重排序 | `BAAI/bge-reranker-base` | 中文 CrossEncoder |
| 框架 | LangChain | ConversationalRetrievalChain |
| 前端 | Vanilla JS + Flask | 零依赖，ES 模块 |
| 文档解析 | PyPDF / docx2txt / TextLoader | 支持 PDF/Word/Markdown |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填入你的 API Key（或通过环境变量注入）：

```yaml
llm:
  api_key: "${DEEPSEEK_API_KEY}"      # DeepSeek API Key
embedding:
  api_key: "${BAILIAN_API_KEY}"       # 阿里云百炼 API Key
```

环境变量方式（推荐，避免 Key 落盘）：

```bash
export DEEPSEEK_API_KEY=sk-xxx
export BAILIAN_API_KEY=sk-xxx
```

### 3. 准备课程资料

将课程文档按子目录组织（自动标注课程标签）：

```
data/documents/
  数据结构/tree.pdf
  操作系统/process.md
  计算机网络/tcp.docx
  软件工程/design.txt
```

### 4. 构建知识库

```bash
python build_kb.py --input_dir ./data/documents
```

指定课程标签（仅用于未自动检测的文档）：

```bash
python build_kb.py --input_dir ./data/documents --course 数据结构 --force
```

### 5. 启动服务

```bash
python web_server.py
```

访问 **http://127.0.0.1:5000**

### 6. 命令行模式（可选）

```bash
python qa_cli.py
```

支持命令：`/reset` 清空记忆、`/sources` 切换来源显示、`/exit` 退出。

## 项目结构

```
├── src/
│   ├── utils.py          # 配置加载、环境变量注入、日志
│   ├── loader.py         # 文档加载（PDF/Word/Markdown）、分块
│   ├── vectorstore.py    # BailianEmbeddings、Chroma 向量库
│   ├── retriever.py      # CrossEncoder 重排序
│   └── chain.py          # System Prompt、对话链、ChatHistory
├── templates/
│   └── index.html        # Web 前端（学术书房风格）
├── web_server.py         # Flask API 服务
├── qa_cli.py             # 命令行问答
├── build_kb.py           # 知识库构建
├── eval.py               # RAG 评估脚本
├── config.example.yaml   # 配置模板
└── prompt/
    └── prompt.md         # 原始需求 Prompt
```

## 核心特性

### 学术诚信三级分类

| 级别 | 场景 | 响应策略 |
|------|------|---------|
| 🔴 红色 | 直接求作业答案 / 完整代码 | 仅给概念提示、方向引导 |
| 🟡 黄色 | 问思路、不知如何开始 | 分析解题思路、伪代码、方法对比 |
| 🟢 绿色 | 问概念、原理、知识点 | 完整解释、示例代码片段 |

### 其他特性

- **课程筛选**：按数据结构/操作系统/计算机网络/软件工程过滤检索范围
- **多模态问答**：支持图片 + 文本提问
- **语音输入**：Web Speech API，中文语音识别
- **多用户隔离**：session_id + token 鉴权，对话持久化
- **Markdown 渲染**：标题、列表、表格、代码块、引用块完整支持
- **降级策略**：CrossEncoder 加载失败自动回退基础检索
- **编码容错**：UTF-8 → GBK → GB2312 → errors=replace
- **自动重试**：LLM 调用失败指数退避重试

## 评估

```bash
python eval.py --test_file ./tests/questions.json --output report.md
```

测试集格式：

```json
[
  {"question": "什么是堆排序？", "reference_answer": "堆排序是基于二叉堆的比较排序，时间复杂度 O(n log n)…"}
]
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `POST` | `/api/ask` | 问答（支持 text + image + course） |
| `GET` | `/api/history` | 获取对话历史 |
| `POST` | `/api/reset` | 重置对话记忆 |

所有 API 需携带 `X-Session-Token` 头进行鉴权。

## License

MIT
