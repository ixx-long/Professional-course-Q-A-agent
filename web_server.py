#!/usr/bin/env python3
"""
Web 问答服务器 — Flask API + 前端页面（薄路由层）。

业务逻辑委托给 src/qa_service.QAService。

用法:
    python web_server.py
    python web_server.py --config my_config.yaml --port 8080
"""

# 必须在所有导入之前设置 HF 环境变量
import os as _os
_os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
_os.environ.setdefault('HF_HUB_OFFLINE', '1')

import sys
import argparse
import json
import queue
import atexit
import signal
import threading
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from src.errors import (
    QAError, ConfigError, AuthError, ValidationError, ImageError
)
from src.security import SecurityValidator, validate_request_data
from src.metrics import (
    record_request, record_llm_request, record_retrieval,
    record_error, set_active_sessions, metrics_endpoint
)
from src.qa_service import QAService

# 企业级增强模块
from src.auth import init_auth_manager, require_auth, require_permission
from src.rate_limiter import init_rate_limiter, rate_limit
from src.redis_cache import init_redis_cache, get_redis_cache
from src.structured_logging import setup_structured_logger
from src.health_check import init_lifecycle, get_lifecycle
from src.monitoring import init_monitoring, get_metric_collector, get_alert_manager
from src.config_manager import init_config_manager, get_config_manager


class StreamCallback(BaseCallbackHandler):
    """流式响应回调处理器（继承 BaseCallbackHandler 以获得所有必需方法）。"""

    def __init__(self, top_k: int):
        super().__init__()
        self.queue = queue.Queue()
        self.sources = []
        self.raw_recall_count = top_k
        self.retrieval_done = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_retriever_end(self, documents, **kwargs):
        self.sources = documents
        self.retrieval_done = True


class QAApp:
    """课程答疑应用主类 — 薄路由层，业务逻辑委托给 QAService。"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.app = Flask(__name__)
        # 限制请求体最大为 10MB，防止恶意大 payload 耗尽内存
        self.app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

        # 初始化配置管理器
        self.config_manager = init_config_manager(config_path)
        
        # 初始化结构化日志
        log_config = self.config_manager.get_nested('logging', default={})
        self.logger = setup_structured_logger(
            name="course_qa",
            log_file=log_config.get('file', './logs/qa.log'),
            level=log_config.get('level', 'INFO'),
            json_output=True,
            include_context=True
        )
        
        # 初始化生命周期管理
        self.lifecycle = init_lifecycle("course_qa_service")
        
        # 初始化监控系统
        monitoring_config = self.config_manager.get_nested('enterprise', 'monitoring', default={})
        if monitoring_config.get('enabled', True):
            init_monitoring(
                webhook_url=monitoring_config.get('webhook_url'),
                eval_interval=monitoring_config.get('eval_interval', 10)
            )
        
        # 初始化 Redis 缓存（如果启用）
        redis_config = self.config_manager.get_nested('enterprise', 'redis', default={})
        if redis_config.get('enabled', False):
            init_redis_cache(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password')
            )
            self.logger.info("Redis 缓存已启用")
        
        # 初始化认证管理器（如果启用）
        jwt_config = self.config_manager.get_nested('enterprise', 'jwt', default={})
        if jwt_config.get('enabled', False):
            init_auth_manager(
                secret_key=jwt_config.get('secret_key', 'change-this-secret'),
                algorithm=jwt_config.get('algorithm', 'HS256'),
                access_token_expire_minutes=jwt_config.get('access_token_expire_minutes', 30),
                refresh_token_expire_days=jwt_config.get('refresh_token_expire_days', 7)
            )
            self.logger.info("JWT 认证已启用")
        
        # 初始化限流器（如果启用）
        rate_limit_config = self.config_manager.get_nested('enterprise', 'rate_limit', default={})
        if rate_limit_config.get('enabled', True):
            init_rate_limiter(
                per_minute=rate_limit_config.get('per_minute', 60),
                per_hour=rate_limit_config.get('per_hour', 1000),
                per_day=rate_limit_config.get('per_day', 10000)
            )
            self.logger.info("限流器已启用")

        # CORS
        cors_origins = _os.environ.get("CORS_ORIGINS", "*")
        CORS(self.app, resources={r"/api/*": {"origins": cors_origins}})

        # 频率限制
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"],
            storage_uri="memory://"
        )

        # 业务逻辑层
        self.service = QAService(config_path)

        self._register_routes()
        
        # 启动服务生命周期
        self.lifecycle.start()
        self.logger.info("服务生命周期已启动")

        atexit.register(self._shutdown_hook)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _shutdown_hook(self):
        """应用关闭时的清理钩子。"""
        if self.service.logger:
            self.service.logger.info("应用正在关闭，执行清理...")
        self.service.shutdown()
        self.lifecycle.stop()
        if self.service.logger:
            self.service.logger.info("应用已完全关闭")

    def _handle_shutdown(self, signum, frame):
        """优雅关闭。"""
        if self.service.logger:
            self.service.logger.info(f"收到关闭信号 {signum}，正在优雅关闭...")
        self._shutdown_hook()
        sys.exit(0)

    # ============================================================
    # 路由注册
    # ============================================================

    def _register_routes(self):
        """注册 Flask 路由和错误处理器。"""
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule(
            "/api/ask", "api_ask",
            self.limiter.limit("30 per minute")(self.api_ask),
            methods=["POST"]
        )
        self.app.add_url_rule(
            "/api/ask/stream", "api_ask_stream",
            self.limiter.limit("20 per minute")(self.api_ask_stream),
            methods=["POST"]
        )
        self.app.add_url_rule("/api/history", "api_history", self.api_history, methods=["GET"])
        self.app.add_url_rule("/api/reset", "api_reset", self.api_reset, methods=["POST"])
        self.app.add_url_rule("/metrics", "metrics", metrics_endpoint)
        self.app.add_url_rule("/health", "health", self.health_check)
        self.app.add_url_rule("/api/config", "api_config", self.api_config, methods=["GET", "PUT"])
        self.app.add_url_rule("/api/system-info", "api_system_info", self.api_system_info, methods=["GET"])

        self.app.register_error_handler(QAError, self._handle_qa_error)
        self.app.register_error_handler(404, self._handle_404)
        self.app.register_error_handler(500, self._handle_500)
        self.app.register_error_handler(429, self._handle_rate_limit)

        # CSRF 保护
        self.app.before_request(self._validate_origin)

    # ============================================================
    # 中间件 & 错误处理
    # ============================================================

    def _validate_origin(self):
        """CSRF 保护：验证 Origin 头。"""
        if request.method not in ("POST", "PUT", "DELETE"):
            return None

        allowed_origins = _os.environ.get("ALLOWED_ORIGINS", "")
        if not allowed_origins:
            return None

        origin = request.headers.get("Origin", "")
        if not origin:
            return None

        allowed_list = [o.strip() for o in allowed_origins.split(",")]
        if origin not in allowed_list and "*" not in allowed_list:
            if self.service.logger:
                self.service.logger.warning(f"CSRF 保护：拒绝来自 {origin} 的请求")
            return jsonify({"error": "请求来源不被允许", "code": "CSRF_BLOCKED"}), 403

        return None

    def _handle_qa_error(self, error: QAError):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def _handle_404(self, error):
        return jsonify({"error": "资源未找到", "code": "NOT_FOUND"}), 404

    def _handle_500(self, error):
        if self.service.logger:
            self.service.logger.error(f"服务器内部错误: {error}")
        return jsonify({"error": "服务器内部错误", "code": "INTERNAL_ERROR"}), 500

    def _handle_rate_limit(self, error):
        return jsonify({"error": "请求过于频繁，请稍后再试", "code": "RATE_LIMIT"}), 429

    # ============================================================
    # 初始化 & 鉴权
    # ============================================================

    def init_system(self):
        """初始化 RAG 系统组件（委托给 service）。"""
        self.service.init_system()

    def _validate_token(self, session_id: str) -> bool:
        """校验 session 归属。"""
        token = request.headers.get("X-Session-Token", "")
        return self.service.session_manager.validate_token(session_id, token)

    def _prepare_request(self, allow_image: bool = True):
        """
        公共请求预处理逻辑。
        
        Args:
            allow_image: 是否允许图片输入
            
        Returns:
            tuple: (question, image_b64, course, session_id, chat_history, question_embedding)
            
        Raises:
            ConfigError: 系统未初始化
            ValidationError: 请求体无效
            AuthError: Token 验证失败
        """
        svc = self.service
        
        if svc.qa_chain is None:
            raise ConfigError("系统未初始化，请等待服务启动完成")
        
        data = request.get_json()
        if not data:
            raise ValidationError("请求体为空")
        
        question = (data.get("question") or "").strip()
        image_b64 = data.get("image") if allow_image else None
        course = data.get("course")
        session_id = data.get("session_id") or "default"
        
        if not self._validate_token(session_id):
            raise AuthError("无权操作此会话")
        
        if not question and not image_b64:
            raise ValidationError("问题不能为空")
        
        if question and len(question) > 5000:
            raise ValidationError("问题文本过长（最大 5000 字符）")
        
        if image_b64:
            img_err = svc.validate_image(image_b64)
            if img_err:
                raise ImageError(f"图片无效: {img_err}")
            if not svc.check_vision_support():
                raise ValidationError("当前模型不支持图片问答，请使用文本提问")
        
        chat_history = svc.session_manager.get_session(session_id)
        
        # 语义缓存检查（仅对文本问题）
        question_embedding = None
        if question and not image_b64:
            _, question_embedding = svc.check_semantic_cache(question)
        
        return question, image_b64, course, session_id, chat_history, question_embedding

    def _record_metrics(self, endpoint: str, start_time: float):
        """记录请求指标。"""
        latency = _time.time() - start_time
        record_request("POST", endpoint, 200, latency)
        set_active_sessions(self.service.session_manager.session_count)

    # ============================================================
    # 路由处理
    # ============================================================

    def index(self):
        """前端页面。"""
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return send_file(str(template_path))
        return "<p>模板未加载。请将 index.html 放入 templates/ 目录。</p>", 200, {
            "Content-Type": "text/html; charset=utf-8"
        }

    def api_ask(self):
        """问答接口。"""
        start_time = _time.time()
        svc = self.service
        
        # 使用辅助方法处理公共逻辑
        question, image_b64, course, session_id, chat_history, question_embedding = self._prepare_request(allow_image=True)
        
        stream = request.get_json().get("stream", False)
        display_text = question or "[图片提问]"
        
        if svc.logger:
            svc.logger.debug(
                f"[{session_id[:8]}] 问题: {display_text}"
                + (f" | 课程: {course}" if course else "")
            )
        
        if stream and not image_b64:
            raise ValidationError("流式请求请使用 /api/ask/stream 接口")
        
        # 语义缓存检查（完整逻辑）
        if not image_b64:
            cached_result, question_embedding = svc.check_semantic_cache(question)
            if cached_result:
                if svc.logger:
                    svc.logger.info(f"[{session_id[:8]}] 语义缓存命中")
                chat_history.add_user(display_text)
                chat_history.add_ai(cached_result["answer"])
                svc.session_manager.mark_dirty()
                svc.session_manager.save_throttled()
                
                self._record_metrics("/api/ask", start_time)
                
                return jsonify({
                    "answer": cached_result["answer"],
                    "sources": cached_result["sources"],
                    "recall_count": 0,
                    "cached": True,
                    "similarity": cached_result.get("similarity", 0)
                })
        
        try:
            if image_b64:
                answer, source_docs = svc.ask_with_image(question, image_b64, chat_history)
                raw_recall_count = len(source_docs)
            else:
                chain = svc.get_chain_for_course(course) if course and course != "all" else svc.qa_chain
                
                retrieval_start = _time.time()
                result = svc.invoke_with_retry(chain, {
                    "question": question,
                    "chat_history": chat_history.get_history(),
                })
                retrieval_latency = _time.time() - retrieval_start
                record_retrieval(
                    course or "all", retrieval_latency,
                    svc.config.get("retrieval", {}).get("top_k", 8)
                )
                
                answer = result.get("answer", "（生成回答失败）")
                source_docs = result.get("source_documents", [])
                raw_recall_count = svc.config.get("retrieval", {}).get("top_k", 8)
                
                llm_model = svc.config.get("llm", {}).get("model_name", "unknown")
                record_llm_request(llm_model, "success", retrieval_latency)
        except Exception as e:
            record_error("LLM_ERROR")
            if svc.logger:
                svc.logger.error(f"问答失败: {e}", exc_info=True)
            raise svc.classify_llm_error(e)
        
        chat_history.add_user(display_text)
        chat_history.add_ai(answer)
        svc.session_manager.mark_dirty()
        svc.session_manager.save_throttled()
        
        if not image_b64:
            svc.write_semantic_cache(question, question_embedding, answer, source_docs)
        
        self._record_metrics("/api/ask", start_time)
        
        return jsonify({
            "answer": answer,
            "sources": svc.extract_sources(source_docs),
            "recall_count": raw_recall_count,
        })

    def api_ask_stream(self):
        """流式问答接口（SSE）。"""
        start_time = _time.time()
        svc = self.service

        # Use helper method for common request preparation
        question, image_b64, course, session_id, chat_history, question_embedding = self._prepare_request(allow_image=False)

        if svc.logger:
            svc.logger.debug(
                f"[{session_id[:8]}] 流式问题: {question}"
                + (f" | 课程: {course}" if course else "")
            )

        # Semantic cache check
        cached_result, question_embedding = svc.check_semantic_cache(question)
        if cached_result:
            if svc.logger:
                svc.logger.info(f"[{session_id[:8]}] 流式语义缓存命中")
            chat_history.add_user(question)
            chat_history.add_ai(cached_result["answer"])
            svc.session_manager.mark_dirty()
            svc.session_manager.save_throttled()

            self._record_metrics("/api/ask/stream", start_time)

            def generate_cached():
                yield f"data: {json.dumps({'sources': cached_result['sources'], 'recall_count': 0, 'cached': True}, ensure_ascii=False)}\n\n"
                answer = cached_result["answer"]
                for i in range(0, len(answer), 5):
                    yield f"data: {json.dumps({'token': answer[i:i+5]}, ensure_ascii=False)}\n\n"
                    _time.sleep(0.02)
                yield f"data: {json.dumps({'done': True, 'full_answer': answer, 'cached': True}, ensure_ascii=False)}\n\n"

            return Response(
                stream_with_context(generate_cached()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
            )

        def generate():
            try:
                chain = svc.get_chain_for_course(course) if course and course != "all" else svc.qa_chain

                top_k = svc.config.get("retrieval", {}).get("top_k", 8)

                # Use module-level StreamCallback class
                callback = StreamCallback(top_k=top_k)
                result_holder = {'answer': '', 'error': None}

                def run_chain():
                    try:
                        result = chain.invoke({
                            "question": question,
                            "chat_history": chat_history.get_history(),
                        }, config={"callbacks": [callback]})
                        result_holder['answer'] = result.get("answer", "")
                        result_holder['sources'] = result.get("source_documents", [])
                        callback.queue.put(None)
                    except Exception as e:
                        result_holder['error'] = str(e)
                        callback.queue.put(None)

                thread = threading.Thread(target=run_chain)
                thread.start()

                wait_start = _time.time()
                while not callback.retrieval_done and _time.time() - wait_start < 10:
                    _time.sleep(0.1)

                if callback.sources:
                    sources_data = svc.extract_sources(callback.sources)
                    recall_count = callback.raw_recall_count or len(callback.sources)
                    yield f"data: {json.dumps({'sources': sources_data, 'recall_count': recall_count}, ensure_ascii=False)}\n\n"

                full_answer = []
                timeout_occurred = False
                while True:
                    try:
                        token = callback.queue.get(timeout=60)
                    except queue.Empty:
                        if svc.logger:
                            svc.logger.warning(f"[{session_id[:8]}] SSE 超时（60s 无新 token）")
                        yield f"data: {json.dumps({'error': '生成超时，请重试'}, ensure_ascii=False)}\n\n"
                        timeout_occurred = True
                        break
                    if token is None:
                        break
                    full_answer.append(token)
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                thread.join()

                if result_holder.get('error'):
                    yield f"data: {json.dumps({'error': result_holder['error']}, ensure_ascii=False)}\n\n"
                elif timeout_occurred:
                    pass
                elif full_answer or result_holder.get('answer'):
                    answer = result_holder['answer'] or ''.join(full_answer)
                    chat_history.add_user(question)
                    chat_history.add_ai(answer)
                    svc.session_manager.mark_dirty()
                    svc.session_manager.save_throttled()

                    if question_embedding is not None:
                        sources_data = svc.extract_sources(result_holder.get('sources', []))
                        svc.write_semantic_cache(question, question_embedding, answer, sources_data)

                    latency = _time.time() - start_time
                    record_request("POST", "/api/ask/stream", 200, latency)
                    set_active_sessions(svc.session_manager.session_count)

                    llm_model = svc.config.get("llm", {}).get("model_name", "unknown")
                    record_llm_request(llm_model, "success", latency)

                    yield f"data: {json.dumps({'done': True, 'full_answer': answer, 'cached': False}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'error': '生成失败，请重试'}, ensure_ascii=False)}\n\n"

            except Exception as e:
                if svc.logger:
                    svc.logger.error(f"流式问答失败: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': '服务内部错误，请稍后重试'}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    def api_history(self):
        """获取当前 session 的对话历史。"""
        session_id = request.args.get("session_id") or "default"
        if not self._validate_token(session_id):
            raise AuthError("无权访问此会话")
        chat_history = self.service.session_manager.get_session(session_id)

        messages = []
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "bot", "content": msg.content, "sources": []})
        return jsonify({"messages": messages})

    def api_reset(self):
        """重置当前 session 的对话历史。"""
        data = request.get_json() or {}
        session_id = data.get("session_id") or "default"
        if not self._validate_token(session_id):
            raise AuthError("无权操作此会话")
        self.service.session_manager.get_session(session_id).clear()
        self.service.session_manager.mark_dirty()
        self.service.session_manager.save_throttled()
        if self.service.logger:
            self.service.logger.debug(f"[{session_id[:8]}] 对话历史已重置")
        return jsonify({"status": "ok", "message": "对话历史已清空"})

    def api_config(self):
        """获取或更新系统配置。"""
        if request.method == "GET":
            cfg = self.service.config
            # 脱敏：隐藏 API Key 中间部分
            def mask_key(v):
                if isinstance(v, str) and len(v) > 12:
                    return v[:4] + "****" + v[-4:]
                return v
            safe = {}
            for section, values in cfg.items():
                if isinstance(values, dict):
                    safe[section] = {k: mask_key(v) if "key" in k.lower() else v for k, v in values.items()}
                else:
                    safe[section] = values
            return jsonify({"config": safe})

        # PUT — 更新配置
        data = request.get_json() or {}
        new_config = data.get("config", {})
        if not new_config:
            raise ValidationError("配置数据为空")

        # 将新配置写入 YAML 文件
        import yaml as _yaml
        cfg_path = Path(self.config_path)
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                old_cfg = _yaml.safe_load(f) or {}
            # 递归合并
            def merge(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        merge(base[k], v)
                    else:
                        base[k] = v
            merge(old_cfg, new_config)
            with open(cfg_path, "w", encoding="utf-8") as f:
                _yaml.dump(old_cfg, f, allow_unicode=True, default_flow_style=False)
            # 热加载
            self.config_manager.reload()
            if self.service.logger:
                self.service.logger.info("配置已更新并重新加载")
            return jsonify({"status": "ok", "message": "配置已更新"})
        except Exception as e:
            if self.service.logger:
                self.service.logger.error(f"配置更新失败: {e}", exc_info=True)
            raise QAError(f"配置更新失败: {e}", status_code=500)

    def api_system_info(self):
        """获取系统信息。"""
        import platform as _platform
        svc = self.service
        cfg = svc.config

        # 尝试获取系统资源信息
        mem_info = {"total_mb": 0, "used_percent": 0}
        disk_info = {"total_gb": 0, "used_percent": 0}
        try:
            import psutil as _psutil
            mem = _psutil.virtual_memory()
            disk = _psutil.disk_usage("/")
            mem_info = {"total_mb": round(mem.total / 1048576), "used_percent": mem.percent}
            disk_info = {"total_gb": round(disk.total / 1073741824, 1), "used_percent": disk.percent}
        except ImportError:
            pass  # psutil 未安装，使用默认值
        except Exception:
            pass

        return jsonify({
            "version": "RAG v1.0",
            "python": _platform.python_version(),
            "os": f"{_platform.system()} {_platform.release()}",
            "memory": mem_info,
            "disk": disk_info,
            "model": cfg.get("llm", {}).get("model_name", "unknown"),
            "embedding_model": cfg.get("embedding", {}).get("model_name", "unknown"),
            "reranker_model": cfg.get("reranker", {}).get("model_name", "unknown"),
            "sessions": svc.session_manager.session_count,
            "vectorstore": "Chroma" if svc.vectorstore else "未初始化",
        })

    def health_check(self):
        """健康检查端点。"""
        svc = self.service
        components = {
            "qa_chain": svc.qa_chain is not None,
            "llm": svc.llm is not None,
            "embedder": svc.embedder is not None,
            "vectorstore": svc.vectorstore is not None,
            "retriever": svc.retriever is not None,
        }
        all_healthy = all(components.values())
        return jsonify({
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
            "sessions": svc.session_manager.session_count,
            "timestamp": int(_time.time())
        }), (200 if all_healthy else 503)


# ---- 应用工厂函数（供 Gunicorn 使用）----

def create_app(config_path: str = None) -> Flask:
    """创建并初始化 Flask 应用（供 WSGI 服务器调用）。"""
    if config_path is None:
        config_path = _os.environ.get("QA_CONFIG", "config.yaml")
    qa_app = QAApp(config_path=config_path)
    qa_app.init_system()
    return qa_app.app


# ---- 启动 ----

def main():
    parser = argparse.ArgumentParser(description="专业课程答疑 Web 服务器")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--port", "-p", type=int, default=5000, help="服务端口（默认: 5000）")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（默认: 127.0.0.1）")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 60)
    print("  专业课程答疑 Web 服务器")
    print("=" * 60)

    print("正在初始化系统组件...")
    try:
        qa_app = QAApp(config_path=args.config)
        qa_app.init_system()
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        sys.exit(1)

    print(f"  访问地址: http://{args.host}:{args.port}")
    print(f"  按 Ctrl+C 停止服务")
    print("=" * 60)

    qa_app.app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
