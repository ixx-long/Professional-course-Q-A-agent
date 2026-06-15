"""
对话链模块。

职责:
  - 构建包含学术诚信规则的 System Prompt
  - 创建 ConversationalRetrievalChain，手动管理对话历史
  - 支持 Few-shot 示例引导模型行为
"""

import logging
from typing import Any

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# ============================================================
# System Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业课程答疑助手，服务于软件工程专业核心课程（数据结构、操作系统、计算机网络、软件工程概论等）。

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

## Few-shot 示例

### 示例 1（概念问题）
用户: 什么是死锁？
【依据】知识库"操作系统-第03章-进程管理.md"指出：死锁是指两个或多个进程互相等待对方释放资源而陷入无限等待的状态，产生条件包括互斥、持有并等待、非抢占、循环等待。
【解答】死锁是两个或多个进程因循环等待资源而无法推进的状态。四个必要条件：互斥→持有并等待→非抢占→循环等待。破坏任一条件即可预防死锁。
【来源】操作系统-第03章-进程管理.md

### 示例 2（思路求助）
用户: 这道二叉树遍历题我完全没有思路，能帮帮我吗？
【依据】知识库"数据结构-第04章-树与二叉树.md"介绍了二叉树的前序、中序、后续遍历的递归定义。
【解答】二叉树遍历题的通用思路：1) 先明确题目要求的是哪种遍历（前/中/后序）；2) 写出递归三要素：终止条件→处理当前节点→递归子树；3) 用一个小例子手动走一遍验证逻辑。递归伪代码：
```
def traverse(root):
    if root is None: return
    # 前序: 先处理 root
    traverse(root.left)
    # 中序: 在这里处理 root
    traverse(root.right)
    # 后序: 最后处理 root
```
以上分析仅供参考，请以教材为准。
【来源】数据结构-第04章-树与二叉树.md

### 示例 3（知识库无法回答）
用户: 某篇论文里的最新量子计算算法是怎么实现的？
【依据】知识库中未找到与"量子计算算法"相关的内容。
【解答】根据课程资料暂未找到答案。当前知识库覆盖数据结构、操作系统、计算机网络等基础课程，不包含量子计算相关内容。建议查阅相关学术论文或咨询老师。
【来源】无

---

{context}
"""


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
    return ChatOpenAI(
        model=llm_config.get("model_name", "deepseek-chat"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("api_base"),
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 2048),
    )


def create_qa_chain(
    retriever,
    config: dict[str, Any],
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

    Returns:
        ConversationalRetrievalChain 实例。
    """
    llm = get_llm(config)

    # 构建问答 Prompt（chat_history 使用字符串模板，因为 ConversationalRetrievalChain 内部将其转为 str）
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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

    def __init__(self, max_turns: int = 4):
        self.messages: list = []
        self.max_turns = max_turns

    def add_user(self, content: str) -> None:
        """记录用户消息。"""
        self.messages.append(HumanMessage(content=content))

    def add_ai(self, content: str) -> None:
        """记录 AI 回复。"""
        self.messages.append(AIMessage(content=content))

    def get_history(self) -> list:
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
