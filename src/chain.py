"""
对话链模块。

职责:
  - 构建包含学术诚信规则的 System Prompt
  - 创建 ConversationalRetrievalChain，手动管理对话历史
  - 支持 Few-shot 示例引导模型行为
"""

import logging
from pathlib import Path
from typing import Any, Optional, List

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

# ============================================================
# System Prompt 加载
# ============================================================

# 默认 Prompt 文件路径
_PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "prompt.md"

def _load_system_prompt() -> str:
    """从外部文件加载 System Prompt，支持热更新。

    优先从 prompt/prompt.md 加载，文件不存在时使用内置默认模板。
    自动追加 {context} 占位符（LangChain 检索文档注入点）。
    """
    if _PROMPT_FILE.exists():
        try:
            prompt_text = _PROMPT_FILE.read_text(encoding="utf-8").strip()
            if prompt_text:
                # 确保 prompt 末尾有 {context} 占位符，供 LangChain 注入检索文档
                if "{context}" not in prompt_text:
                    prompt_text += "\n\n{context}"
                logger.info(f"从 {_PROMPT_FILE} 加载 System Prompt")
                return prompt_text
        except Exception as e:
            logger.warning(f"加载 Prompt 文件失败: {e}，使用内置默认模板")

    # 内置默认模板（当外部文件不存在时的兜底）
    return _DEFAULT_SYSTEM_PROMPT


# 内置默认 System Prompt（兜底）
_DEFAULT_SYSTEM_PROMPT = """你是一个专业课程答疑助手，服务于软件工程专业核心课程（数据结构、操作系统、计算机网络、软件工程概论等）。

## 核心规则（严格遵循，优先级从上到下）

### 1. 以知识库为准
- 优先根据下方「参考知识库片段」回答问题。
- 知识库未涉及的内容，如实告知："根据课程资料暂未找到答案，建议查阅教材或咨询老师。"
- 严禁编造不存在于知识库中的具体数据、公式或结论。

### 2. 学术诚信（三级分类）
| 级别 | 场景 | 允许 | 禁止 |
|------|------|------|------|
| 🔴 红色 | 直接求作业答案/完整代码 | 仅给出概念提示、方向引导 | 完整代码、可直接提交的答案、逐行详解 |
| 🟡 黄色 | 问思路、不知如何开始 | 分析解题思路、伪代码、方法对比 | 完整可运行代码 |
| 🟢 绿色 | 问概念、原理、知识点 | 完整解释、示例代码片段、图解说明 | 无 |

### 3. 回答格式
按以下结构输出回答：
```
【依据】引用知识库中的关键信息（1-2句话概括）。
【解答】给出你的回答。
【来源】列出引用的文件名或段落编号。
```

### 4. 补充规则
- 若知识库信息不足以回答，请在【解答】中说明原因并给出学习建议。
- 对不确定的内容，标注"以上分析仅供参考，请以教材为准"。
- 回答简洁、准确，避免冗长铺垫。

---

{context}
"""

# 模块加载时加载 Prompt
SYSTEM_PROMPT = _load_system_prompt()


def get_llm(config: dict[str, Any]) -> ChatOpenAI:
    """
    根据配置创建 LLM 实例。

    Args:
        config: 完整配置字典，使用 config["llm"] 段。

    Returns:
        ChatOpenAI 实例（配置为 DeepSeek 兼容接口）。

    Raises:
        KeyError: config 中缺少 llm 段时抛出。
    """
    if "llm" not in config:
        raise KeyError("配置中缺少 [llm] 段，请检查 config.yaml")
    llm_config = config["llm"]
    logger.info(
        f"初始化 LLM: {llm_config.get('model_name')} @ {llm_config.get('api_base')}, "
        f"temperature={llm_config.get('temperature', 0.1)}"
    )
    
    # 构建 LLM 参数
    llm_kwargs = {
        "model": llm_config.get("model_name", "deepseek-chat"),
        "api_key": llm_config.get("api_key"),
        "base_url": llm_config.get("api_base"),
        "temperature": llm_config.get("temperature", 0.1),
        "max_tokens": llm_config.get("max_tokens", 2048),
    }
    
    # 添加请求超时配置（防止 LLM 调用无限阻塞）
    request_timeout = llm_config.get("request_timeout")
    if request_timeout:
        llm_kwargs["request_timeout"] = request_timeout
        logger.info(f"LLM 请求超时: {request_timeout} 秒")
    
    return ChatOpenAI(**llm_kwargs)


def create_qa_chain(
    retriever: VectorStoreRetriever,
    config: dict[str, Any],
    llm: Optional[ChatOpenAI] = None,
) -> ConversationalRetrievalChain:
    """
    创建检索问答链（手动管理对话历史）。

    使用 LangChain 的 ConversationalRetrievalChain，将检索到的知识库片段
    与对话历史、System Prompt 一同送入 LLM。

    调用方需手动传入 chat_history（list[BaseMessage]）：
        chat_history = []
        chain = create_qa_chain(retriever, config)
        result = chain.invoke({"question": "...", "chat_history": chat_history})
        chat_history.append(HumanMessage(content=result["question"]))
        chat_history.append(AIMessage(content=result["answer"]))

    Args:
        retriever: LangChain Retriever 实例（可以是基础 Retriever 或压缩 Retriever）。
        config: 完整配置字典。
        llm: 可选的 LLM 实例，传入则复用，否则从 config 创建新实例。

    Returns:
        ConversationalRetrievalChain 实例。
    """
    if llm is None:
        llm = get_llm(config)

    # 动态加载最新 System Prompt（支持热更新）
    system_prompt = _load_system_prompt()

    # 构建问答 Prompt（chat_history 使用字符串模板，因为 ConversationalRetrievalChain 内部将其转为 str）
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "对话历史：\n{chat_history}"),
        ("human", "{question}"),
    ])

    # Condense question prompt（用于将历史对话压缩为独立问题）
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于结合上下文理解问题的助手。"),
        ("human", "对话历史：\n{chat_history}\n\n请将以下问题重新表述为一个独立、完整的问句：\n{question}"),
    ])

    # 创建 ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        verbose=False,
    )

    logger.info("对话链创建完成")
    return chain


class ChatHistory:
    """对话历史管理器。

    ConversationalRetrievalChain 接受 list[BaseMessage] 作为 chat_history 输入，
    内部自动转为字符串后注入 Prompt 模板的 {chat_history} 变量。
    """

    def __init__(self, max_turns: int = 4) -> None:
        self.messages: List[BaseMessage] = []
        self.max_turns: int = max_turns

    def add_user(self, content: str) -> None:
        """记录用户消息。"""
        self.messages.append(HumanMessage(content=content))

    def add_ai(self, content: str) -> None:
        """记录 AI 回复。"""
        self.messages.append(AIMessage(content=content))

    def get_history(self) -> List[BaseMessage]:
        """获取最近 max_turns 轮的消息列表（供 ConversationalRetrievalChain 使用）。"""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            return self.messages[-max_messages:]
        return list(self.messages)

    def clear(self) -> None:
        """清空历史。"""
        self.messages = []
        logger.info("对话历史已清空")


def format_source_documents(source_docs: list[Document]) -> str:
    """
    格式化来源文档引用信息。

    Args:
        source_docs: ConversationalRetrievalChain 返回的 source_documents。

    Returns:
        格式化的引用字符串。

    用法:
        sources = format_source_documents(result["source_documents"])
        print(sources)
    """
    if not source_docs:
        return "无来源引用"

    lines = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score")
        score_str = f" (相关度: {score:.3f})" if score is not None else ""
        key = f"{source}#p{page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  • {source} 第{page}页{score_str}")

    return "\n".join(lines)
