P0 问题

敏感信息已落盘：真实 API Key 明文存在于 config.yaml，虽然 .gitignore 忽略了它，但工作区本身已经泄露，存在误提交、拷贝扩散和日志旁路暴露风险。
Web 会话全局共享：服务端把 qa_chain、chat_history、retriever、llm 放在全局变量里，所有用户共用同一份会话状态，任何人的提问、重置都会影响别人，web_server.py api_ask api_reset。
存在并发竞态：请求先读历史再写历史，中间没有锁、没有 session 隔离，两个请求并发时会出现串话、覆盖、顺序错乱、被 /api/reset 抢断等问题，api_ask ChatHistory。
全局关闭 HTTPS 校验：加载重排序模型时直接 monkey patch requests.Session.request 并禁用 TLS 校验，这会污染整个进程内后续网络请求，属于高风险安全实现，load_cross_encoder。
内部异常直接回显给前端：/api/ask 出错时把 str(e) 原样返回客户端，可能泄露上游模型接口、运行环境和内部错误细节，api_ask。
图片接口缺少服务端校验：后端对 image 只做“有无”检查，没有验证类型、尺寸、base64 长度和 MIME，前端的 20MB 限制不能阻止直接打 API，存在资源放大和拒绝服务风险，api_ask _ask_with_image。
P1 问题

图片问答默认配置很可能不可用：代码把图片作为 image_url 发给 LLM，但默认模型配置是 deepseek-chat，没有明确声明为视觉模型，默认契约下图片问答大概率报错或表现不稳定，config.example.yaml _ask_with_image。
课程筛选是假的：前端会发送 course 且展示筛选 UI，但后端根本不读取该字段，入库元数据里也没有课程维度，用户以为在按课程检索，实际完全没生效，index.html sendMessage api_ask load_single_document。
对话历史切换是假的：左侧历史列表只是前端内存数组，点击后直接弹“开发中”，没有任何恢复会话的后端能力，index.html。
“新对话”语义是错的：UI 看起来像新建一个会话，实际上调用的是全局 /api/reset，会清空当前服务上共享的唯一历史，不是新建会话，newChat api_reset。
图片问答和文本问答走了两套不一致链路：文本走 compression_retriever + ConversationalRetrievalChain，图片却直接用基础 retriever.invoke(... )[:4] 手拼 prompt，导致同一问题从两个入口进入会得到不同质量和不同行为，init_system _ask_with_image。
``--force 语义与实现不符：文案说“强制全量重建”，但实际只是关闭去重后继续往现有 Chroma 里加数据，不会清空旧库，反而会制造重复 chunk，build_kb.py argparse add_documents 调用 add_documents。
请求 JSON 解析异常不统一：request.get_json() 放在主异常处理之外，非法 JSON 时 Flask 可能直接返回默认 HTML 400，而不是项目自己的 JSON 错误格式，api_ask。
前端错误提示会误导用户：fetch 后不检查 resp.ok 和返回结构，后端 500/400 或非 JSON 响应时，前端会统一提示“网络连接失败”，掩盖真实问题，sendMessage。
单文件加载失败会被静默跳过：构建知识库时单个文件异常只写 warning 不终止，容易生成一个“部分缺资料但继续成功”的知识库，结果偏差不易被发现，load_documents。
P2 问题

依赖声明不完整且脆弱：代码明确导入了 openai、langchain_classic、langchain_text_splitters，但 requirements.txt 没有显式声明这些直接依赖，环境复现依赖传递安装，容易在新环境翻车，imports imports imports requirements.txt。
没有自动化测试：仓库中没有 tests/、test_*.py 或 pytest/unittest 用例，核心链路改动后没有回归保护。
评估脚本缺失：规格里提到 eval.py，但仓库里没有该文件，项目无法验证自己宣称的准确率或回归质量，prompt-optimized.md。
部署方案缺失：Web 端直接 app.run(...)，没有生产 WSGI、Docker、CI/CD、README 部署说明和健康检查，当前只适合开发态运行，main。
配置校验过浅：只检查顶层 section 是否存在，不检查关键字段是否为空、URL 是否合法、目录是否可写、top_k/rerank_top_n 是否合理，load_config。
规格与实现不一致：规格文档多处写 cross-encoder/ms-marco-MiniLM-L-6-v2，配置文件却用 L-4-v2，会造成性能预期和实验复现不一致，prompt-optimized.md config.example.yaml。
前端历史并不持久：所谓“历史会话”只存在页面内存，刷新即丢，既没有服务端持久化也没有 localStorage 持久化，index.html。
文档缺口明显：没有正式 README、安装步骤、最小启动方式、故障排查和接口说明，当前只有需求/设计文档，不适合别人快速接手。
P3 问题

Embedding 建库效率低：BailianEmbeddings.embed_documents() 逐条调用外部 API，没有批处理、重试、限流或并发控制，文档量大时会很慢，BailianEmbeddings。
去重前会全量读取已有元数据：get_existing_chunk_ids() 直接 vectorstore.get()，数据量大时会有明显内存和启动成本，get_existing_chunk_ids。
来源分数字段对 0.0 处理错误：if score else None 会把合法的 0.0 当作无值，属于边界 bug，extract_sources format_source_documents。
日志初始化会吞掉后续配置变化：setup_logger() 发现已有 handler 就直接返回，之后即使换日志路径或级别也不会生效，setup_logger。
有未落地或遗留实现痕迹：headerFilters 是空容器、build_frontend.py 被注释提及但仓库中不存在、部分 import 未使用，这些都说明代码处于半成品状态，index.html load_index_html。
开放问题

config.yaml 是否已经提交到任何远端仓库：如果提交过，需要按泄露事件处理，不只是本地删除。
图片问答 目标模型到底是不是多模态：如果本来就要支持视觉，需要把模型、校验和错误提示一起改正；如果不支持，前端入口应先移除。
课程筛选 预期是基于目录名、文件名前缀还是显式 metadata：这关系到入库结构是否要重做。