"""
性能监控和告警模块。

提供：
- 性能指标收集和聚合
- 告警规则定义和触发
- 告警通知（邮件、Webhook、日志）
- 性能趋势分析
"""

import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态。"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    """告警规则定义。"""
    name: str
    metric_name: str
    threshold: float
    operator: str  # ">", "<", ">=", "<=", "=="
    duration_seconds: int = 0  # 持续时间阈值
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """告警实例。"""
    rule: AlertRule
    current_value: float
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            "rule_name": self.rule.name,
            "metric_name": self.rule.metric_name,
            "current_value": self.current_value,
            "threshold": self.rule.threshold,
            "operator": self.rule.operator,
            "severity": self.rule.severity.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "message": self.message,
            "metadata": self.metadata,
        }


class AlertNotifier:
    """告警通知器接口。"""
    
    def notify(self, alert: Alert):
        """
        发送告警通知。
        
        Args:
            alert: 告警实例
        """
        raise NotImplementedError


class LogAlertNotifier(AlertNotifier):
    """日志告警通知器。"""
    
    def notify(self, alert: Alert):
        """记录告警到日志。"""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.rule.severity, logger.warning)
        
        message = (
            f"[告警] {alert.rule.name}: "
            f"{alert.rule.metric_name} = {alert.current_value:.2f} "
            f"{alert.rule.operator} {alert.rule.threshold} "
            f"(严重程度: {alert.rule.severity.value})"
        )
        log_func(message)


class WebhookAlertNotifier(AlertNotifier):
    """Webhook 告警通知器。"""
    
    def __init__(self, webhook_url: str, headers: Dict = None):
        """
        初始化 Webhook 通知器。
        
        Args:
            webhook_url: Webhook URL
            headers: 请求头
        """
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    def notify(self, alert: Alert):
        """发送告警到 Webhook。"""
        try:
            import requests
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(f"Webhook 通知失败: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook 通知异常: {e}", exc_info=True)


class MetricCollector:
    """
    指标收集器（桥接到 Prometheus）。

    所有指标操作同时写入 Prometheus 指标（metrics.py）和本地缓存，
    避免双重收集，保持单一数据源。
    """

    def __init__(self):
        """初始化指标收集器。"""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._prom_available = False
        self._init_prom()

    def _init_prom(self):
        """尝试导入 Prometheus 指标模块。"""
        try:
            from src import metrics as _prom
            self._prom = _prom
            self._prom_available = True
        except Exception:
            self._prom = None
            self._prom_available = False
    
    def inc_counter(self, name: str, value: float = 1.0, labels: Dict = None):
        """
        增加计数器。

        Args:
            name: 指标名称
            value: 增加的值
            labels: 标签字典
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

        # 桥接到 Prometheus
        if self._prom_available and self._prom:
            try:
                if "error" in name.lower():
                    self._prom.ERROR_COUNT.labels(error_type=name).inc(value)
                elif "llm" in name.lower():
                    model = labels.get("model", "unknown") if labels else "unknown"
                    status = labels.get("status", "success") if labels else "success"
                    self._prom.LLM_REQUEST_COUNT.labels(model=model, status=status).inc(value)
                elif "retrieval" in name.lower():
                    course = labels.get("course", "unknown") if labels else "unknown"
                    self._prom.RETRIEVAL_COUNT.labels(course=course).inc(value)
                else:
                    method = labels.get("method", "unknown") if labels else "unknown"
                    endpoint = labels.get("endpoint", "unknown") if labels else "unknown"
                    status = labels.get("status", 200) if labels else 200
                    self._prom.REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc(value)
            except Exception as e:
                logger.debug(f"Prometheus 桥接失败: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """
        设置仪表值。

        Args:
            name: 指标名称
            value: 仪表值
            labels: 标签字典
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

        # 桥接到 Prometheus
        if self._prom_available and self._prom:
            try:
                if "session" in name.lower():
                    self._prom.ACTIVE_SESSIONS.set(value)
            except Exception as e:
                logger.debug(f"Prometheus 桥接失败: {e}")
    
    def observe_histogram(self, name: str, value: float, labels: Dict = None):
        """
        观察直方图值。

        Args:
            name: 指标名称
            value: 观察值
            labels: 标签字典
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

        # 桥接到 Prometheus
        if self._prom_available and self._prom:
            try:
                if "request" in name.lower() and "duration" in name.lower():
                    method = labels.get("method", "unknown") if labels else "unknown"
                    endpoint = labels.get("endpoint", "unknown") if labels else "unknown"
                    self._prom.REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(value)
                elif "llm" in name.lower() and "duration" in name.lower():
                    model = labels.get("model", "unknown") if labels else "unknown"
                    self._prom.LLM_LATENCY.labels(model=model).observe(value)
                elif "retrieval" in name.lower() and "duration" in name.lower():
                    course = labels.get("course", "unknown") if labels else "unknown"
                    self._prom.RETRIEVAL_LATENCY.labels(course=course).observe(value)
                elif "retrieved" in name.lower() and "docs" in name.lower():
                    course = labels.get("course", "unknown") if labels else "unknown"
                    self._prom.RETRIEVED_DOCS.labels(course=course).observe(value)
            except Exception as e:
                logger.debug(f"Prometheus 桥接失败: {e}")
    
    def get_counter(self, name: str, labels: Dict = None) -> float:
        """
        获取计数器值。
        
        Args:
            name: 指标名称
            labels: 标签字典
            
        Returns:
            计数器值
        """
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Dict = None) -> float:
        """
        获取仪表值。
        
        Args:
            name: 指标名称
            labels: 标签字典
            
        Returns:
            仪表值
        """
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict = None) -> Dict:
        """
        获取直方图统计信息。
        
        Args:
            name: 指标名称
            labels: 标签字典
            
        Returns:
            统计信息字典（count, sum, avg, min, max）
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
            
            return {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        """
        生成指标键。
        
        Args:
            name: 指标名称
            labels: 标签字典
            
        Returns:
            指标键字符串
        """
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_all_metrics(self) -> Dict:
        """
        获取所有指标。
        
        Returns:
            所有指标字典
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k) for k in self._histograms
                },
            }


class AlertManager:
    """
    告警管理器。
    
    支持：
    - 告警规则注册
    - 告警评估和触发
    - 告警通知
    - 告警历史查询
    """
    
    def __init__(self, metric_collector: MetricCollector):
        """
        初始化告警管理器。
        
        Args:
            metric_collector: 指标收集器
        """
        self.metric_collector = metric_collector
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notifiers: List[AlertNotifier] = []
        self._lock = threading.Lock()
        self._running = False
        self._eval_thread: Optional[threading.Thread] = None
    
    def register_rule(self, rule: AlertRule):
        """
        注册告警规则。
        
        Args:
            rule: 告警规则
        """
        self._rules[rule.name] = rule
        logger.info(f"注册告警规则: {rule.name}")
    
    def unregister_rule(self, name: str):
        """
        注销告警规则。
        
        Args:
            name: 规则名称
        """
        if name in self._rules:
            del self._rules[name]
            logger.info(f"注销告警规则: {name}")
    
    def add_notifier(self, notifier: AlertNotifier):
        """
        添加告警通知器。
        
        Args:
            notifier: 告警通知器
        """
        self._notifiers.append(notifier)
    
    def start(self, eval_interval: int = 10):
        """
        启动告警评估。
        
        Args:
            eval_interval: 评估间隔（秒）
        """
        if self._running:
            logger.warning("告警管理器已在运行")
            return
        
        self._running = True
        self._eval_thread = threading.Thread(
            target=self._eval_loop,
            args=(eval_interval,),
            daemon=True,
        )
        self._eval_thread.start()
        logger.info(f"告警管理器已启动，评估间隔: {eval_interval}s")
    
    def stop(self):
        """停止告警评估。"""
        self._running = False
        if self._eval_thread:
            self._eval_thread.join(timeout=5)
        logger.info("告警管理器已停止")
    
    def _eval_loop(self, interval: int):
        """告警评估循环。"""
        while self._running:
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"告警评估异常: {e}", exc_info=True)
            time.sleep(interval)
    
    def _evaluate_rules(self):
        """评估所有告警规则。"""
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            try:
                # 获取当前指标值
                current_value = self._get_metric_value(rule.metric_name)
                
                # 检查是否触发告警
                if self._check_threshold(current_value, rule):
                    self._trigger_alert(rule, current_value)
                else:
                    self._resolve_alert(rule.name)
                    
            except Exception as e:
                logger.error(f"评估规则 {rule.name} 失败: {e}", exc_info=True)
    
    def _get_metric_value(self, metric_name: str) -> float:
        """
        获取指标值。
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标值
        """
        # 尝试从计数器获取
        value = self.metric_collector.get_counter(metric_name)
        if value != 0:
            return value
        
        # 尝试从仪表获取
        value = self.metric_collector.get_gauge(metric_name)
        if value != 0:
            return value
        
        # 尝试从直方图获取平均值
        stats = self.metric_collector.get_histogram_stats(metric_name)
        return stats.get("avg", 0.0)
    
    def _check_threshold(self, value: float, rule: AlertRule) -> bool:
        """
        检查是否超过阈值。
        
        Args:
            value: 当前值
            rule: 告警规则
            
        Returns:
            是否触发告警
        """
        op = rule.operator
        threshold = rule.threshold
        
        if op == ">":
            return value > threshold
        elif op == "<":
            return value < threshold
        elif op == ">=":
            return value >= threshold
        elif op == "<=":
            return value <= threshold
        elif op == "==":
            return value == threshold
        else:
            logger.error(f"未知操作符: {op}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """
        触发告警。
        
        Args:
            rule: 告警规则
            current_value: 当前值
        """
        if rule.name in self._active_alerts:
            return  # 已触发
        
        alert = Alert(
            rule=rule,
            current_value=current_value,
            timestamp=time.time(),
            message=f"{rule.metric_name} = {current_value:.2f} {rule.operator} {rule.threshold}",
        )
        
        with self._lock:
            self._active_alerts[rule.name] = alert
            self._alert_history.append(alert)
        
        logger.warning(f"告警触发: {rule.name} - {alert.message}")
        
        # 发送通知
        for notifier in self._notifiers:
            try:
                notifier.notify(alert)
            except Exception as e:
                logger.error(f"告警通知失败: {e}", exc_info=True)
    
    def _resolve_alert(self, rule_name: str):
        """
        解决告警。
        
        Args:
            rule_name: 规则名称
        """
        with self._lock:
            if rule_name in self._active_alerts:
                alert = self._active_alerts[rule_name]
                alert.status = AlertStatus.RESOLVED
                del self._active_alerts[rule_name]
                logger.info(f"告警解决: {rule_name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """
        获取活跃告警。
        
        Returns:
            活跃告警列表
        """
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """
        获取告警历史。
        
        Args:
            limit: 最大返回数量
            
        Returns:
            告警历史列表
        """
        with self._lock:
            return self._alert_history[-limit:]


# 全局指标收集器和告警管理器
_metric_collector: Optional[MetricCollector] = None
_alert_manager: Optional[AlertManager] = None


def get_metric_collector() -> MetricCollector:
    """获取全局指标收集器。"""
    global _metric_collector
    if _metric_collector is None:
        _metric_collector = MetricCollector()
    return _metric_collector


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器。"""
    global _alert_manager
    if _alert_manager is None:
        collector = get_metric_collector()
        _alert_manager = AlertManager(collector)
    return _alert_manager


def init_monitoring(
    webhook_url: Optional[str] = None,
    eval_interval: int = 10,
) -> None:
    """
    初始化监控系统。
    
    Args:
        webhook_url: Webhook 告警 URL
        eval_interval: 告警评估间隔（秒）
    """
    collector = get_metric_collector()
    manager = get_alert_manager()
    
    # 添加默认通知器
    manager.add_notifier(LogAlertNotifier())
    
    if webhook_url:
        manager.add_notifier(WebhookAlertNotifier(webhook_url))
    
    # 注册默认告警规则
    _register_default_rules(manager)
    
    # 启动告警评估
    manager.start(eval_interval)
    
    logger.info("监控系统初始化完成")


def _register_default_rules(manager: AlertManager):
    """注册默认告警规则。"""
    # 高错误率告警
    manager.register_rule(AlertRule(
        name="high_error_rate",
        metric_name="errors_total",
        threshold=100,
        operator=">",
        severity=AlertSeverity.CRITICAL,
        description="错误率过高",
    ))
    
    # 高延迟告警
    manager.register_rule(AlertRule(
        name="high_latency",
        metric_name="request_duration_seconds",
        threshold=5.0,
        operator=">",
        severity=AlertSeverity.WARNING,
        description="请求延迟过高",
    ))
    
    # 活跃会话过多
    manager.register_rule(AlertRule(
        name="too_many_sessions",
        metric_name="active_sessions",
        threshold=1000,
        operator=">",
        severity=AlertSeverity.WARNING,
        description="活跃会话过多",
    ))
