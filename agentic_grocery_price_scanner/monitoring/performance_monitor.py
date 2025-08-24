"""
Production-level performance monitoring and benchmarking system.
Provides comprehensive metrics, alerting, and performance analysis.
"""

import asyncio
import logging
import time
import json
import psutil
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        tags_str = ""
        if self.tags:
            tag_pairs = [f'{k}="{v}"' for k, v in self.tags.items()]
            tags_str = "{" + ",".join(tag_pairs) + "}"
        
        return f"{self.name}{tags_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition and results."""
    
    name: str
    target_value: float
    comparison: str = "less_than"  # less_than, greater_than, equal_to
    unit: str = "seconds"
    description: str = ""
    
    # Results
    current_value: Optional[float] = None
    last_measured: Optional[datetime] = None
    historical_values: List[float] = field(default_factory=list)
    
    @property
    def is_passing(self) -> bool:
        """Check if current value meets benchmark."""
        if self.current_value is None:
            return False
            
        if self.comparison == "less_than":
            return self.current_value < self.target_value
        elif self.comparison == "greater_than":
            return self.current_value > self.target_value
        else:  # equal_to
            return abs(self.current_value - self.target_value) < 0.01
    
    @property
    def performance_ratio(self) -> float:
        """Get performance ratio (1.0 = meeting target exactly)."""
        if self.current_value is None:
            return 0.0
        
        if self.comparison == "less_than":
            return self.target_value / max(self.current_value, 0.001)
        elif self.comparison == "greater_than":
            return self.current_value / max(self.target_value, 0.001)
        else:
            return 1.0 - abs(self.current_value - self.target_value) / max(self.target_value, 0.001)
    
    def update_value(self, value: float):
        """Update benchmark with new measurement."""
        self.current_value = value
        self.last_measured = datetime.now()
        self.historical_values.append(value)
        
        # Limit historical data
        if len(self.historical_values) > 100:
            self.historical_values = self.historical_values[-50:]


@dataclass 
class SystemAlert:
    """System performance alert."""
    
    alert_id: str
    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "metric": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # System resource tracking
        self.process = psutil.Process()
        self.system_metrics = [
            "cpu_percent",
            "memory_mb", 
            "memory_percent",
            "disk_io_read_mb",
            "disk_io_write_mb",
            "network_bytes_sent",
            "network_bytes_recv"
        ]
    
    async def start_collection(self):
        """Start continuous metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        timestamp = datetime.now()
        
        # CPU and Memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Disk I/O
        disk_io = self.process.io_counters()
        disk_read_mb = disk_io.read_bytes / 1024 / 1024
        disk_write_mb = disk_io.write_bytes / 1024 / 1024
        
        # Network I/O (system-wide)
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_recv = net_io.bytes_recv
        
        # Store metrics
        metrics_data = {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "disk_io_read_mb": disk_read_mb,
            "disk_io_write_mb": disk_write_mb,
            "network_bytes_sent": network_sent,
            "network_bytes_recv": network_recv
        }
        
        for metric_name, value in metrics_data.items():
            self.metrics[metric_name].append({
                "timestamp": timestamp,
                "value": value
            })
    
    def record_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a custom application metric."""
        metric_data = {
            "timestamp": datetime.now(),
            "value": value,
            "type": metric_type.value,
            "tags": tags or {}
        }
        
        self.metrics[name].append(metric_data)
    
    def get_metric_summary(self, name: str, duration_minutes: int = 10) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        if name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_values = [
            entry["value"] for entry in self.metrics[name]
            if entry["timestamp"] >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "p95": statistics.quantiles(recent_values, n=20)[18] if len(recent_values) >= 20 else max(recent_values),
            "p99": statistics.quantiles(recent_values, n=100)[98] if len(recent_values) >= 100 else max(recent_values)
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all collected metrics."""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = self.get_metric_summary(metric_name)
        return summary


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.alerts: List[SystemAlert] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.workflow_metrics: Dict[str, List[float]] = defaultdict(list)
        self.agent_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        self._initialize_default_benchmarks()
        self._initialize_alert_thresholds()
        
        logger.info("Initialized PerformanceMonitor")
    
    def _initialize_default_benchmarks(self):
        """Initialize default performance benchmarks."""
        self.benchmarks = {
            "single_workflow_time": PerformanceBenchmark(
                name="single_workflow_time",
                target_value=60.0,
                comparison="less_than",
                unit="seconds",
                description="Single workflow completion time"
            ),
            "workflow_memory_usage": PerformanceBenchmark(
                name="workflow_memory_usage",
                target_value=500.0,
                comparison="less_than", 
                unit="MB",
                description="Memory usage per workflow"
            ),
            "batch_processing_time": PerformanceBenchmark(
                name="batch_processing_time",
                target_value=180.0,
                comparison="less_than",
                unit="seconds", 
                description="5-recipe batch processing time"
            ),
            "success_rate": PerformanceBenchmark(
                name="success_rate",
                target_value=95.0,
                comparison="greater_than",
                unit="percent",
                description="Workflow success rate"
            ),
            "cache_hit_rate": PerformanceBenchmark(
                name="cache_hit_rate",
                target_value=80.0,
                comparison="greater_than",
                unit="percent",
                description="Cache hit rate"
            ),
            "data_quality_score": PerformanceBenchmark(
                name="data_quality_score", 
                target_value=90.0,
                comparison="greater_than",
                unit="percent",
                description="Overall data quality score"
            )
        }
    
    def _initialize_alert_thresholds(self):
        """Initialize alerting thresholds."""
        self.alert_thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_mb": {"warning": 1000.0, "critical": 1500.0},
            "memory_percent": {"warning": 80.0, "critical": 90.0},
            "workflow_failure_rate": {"warning": 10.0, "critical": 20.0},
            "response_time_p95": {"warning": 120.0, "critical": 180.0},
            "error_rate": {"warning": 5.0, "critical": 10.0}
        }
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        await self.metrics_collector.start_collection()
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        await self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")
    
    def record_workflow_execution(
        self,
        workflow_type: str,
        execution_time: float,
        success: bool,
        memory_used: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record workflow execution metrics."""
        # Record execution time
        self.workflow_metrics[f"{workflow_type}_execution_time"].append(execution_time)
        
        # Record success/failure
        success_value = 1.0 if success else 0.0
        self.workflow_metrics[f"{workflow_type}_success"].append(success_value)
        
        # Record memory usage if provided
        if memory_used is not None:
            self.workflow_metrics[f"{workflow_type}_memory"].append(memory_used)
        
        # Update benchmarks
        if workflow_type == "single" and "single_workflow_time" in self.benchmarks:
            self.benchmarks["single_workflow_time"].update_value(execution_time)
        
        if workflow_type == "batch" and "batch_processing_time" in self.benchmarks:
            self.benchmarks["batch_processing_time"].update_value(execution_time)
        
        if memory_used and "workflow_memory_usage" in self.benchmarks:
            self.benchmarks["workflow_memory_usage"].update_value(memory_used)
        
        # Update success rate benchmark
        recent_successes = self.workflow_metrics[f"{workflow_type}_success"][-100:]  # Last 100
        if recent_successes:
            success_rate = (sum(recent_successes) / len(recent_successes)) * 100
            self.benchmarks["success_rate"].update_value(success_rate)
        
        # Record custom metrics
        self.metrics_collector.record_custom_metric(
            f"workflow_{workflow_type}_duration",
            execution_time,
            MetricType.TIMER,
            {"workflow_type": workflow_type, "success": str(success)}
        )
        
        # Check for alerts
        self._check_performance_alerts()
    
    def record_agent_performance(
        self,
        agent_type: str,
        operation: str,
        execution_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record agent-specific performance metrics."""
        agent_key = f"{agent_type}_{operation}"
        
        # Record metrics
        self.agent_metrics[agent_type][f"{operation}_time"].append(execution_time)
        self.agent_metrics[agent_type][f"{operation}_success"].append(1.0 if success else 0.0)
        
        # Record as custom metric
        self.metrics_collector.record_custom_metric(
            f"agent_{agent_key}_duration",
            execution_time,
            MetricType.TIMER,
            {"agent": agent_type, "operation": operation, "success": str(success)}
        )
    
    def record_cache_metrics(self, hits: int, misses: int):
        """Record cache performance metrics."""
        total_requests = hits + misses
        if total_requests > 0:
            hit_rate = (hits / total_requests) * 100
            self.benchmarks["cache_hit_rate"].update_value(hit_rate)
            
            self.metrics_collector.record_custom_metric("cache_hit_rate", hit_rate, MetricType.GAUGE)
            self.metrics_collector.record_custom_metric("cache_requests_total", total_requests, MetricType.COUNTER)
    
    def record_quality_metrics(self, quality_score: float, issues_count: int):
        """Record data quality metrics."""
        self.benchmarks["data_quality_score"].update_value(quality_score)
        
        self.metrics_collector.record_custom_metric("data_quality_score", quality_score, MetricType.GAUGE)
        self.metrics_collector.record_custom_metric("quality_issues_count", issues_count, MetricType.COUNTER)
    
    def _check_performance_alerts(self):
        """Check metrics against alert thresholds."""
        current_time = datetime.now()
        
        # Check system metrics
        system_summary = self.metrics_collector.get_all_metrics_summary()
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in system_summary:
                current_value = system_summary[metric_name].get("mean", 0)
                
                # Check critical threshold
                if current_value > thresholds.get("critical", float("inf")):
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"Critical threshold exceeded for {metric_name}",
                        "system",
                        metric_name,
                        current_value,
                        thresholds["critical"]
                    )
                
                # Check warning threshold
                elif current_value > thresholds.get("warning", float("inf")):
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Warning threshold exceeded for {metric_name}",
                        "system", 
                        metric_name,
                        current_value,
                        thresholds["warning"]
                    )
    
    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        component: str,
        metric_name: str,
        current_value: float,
        threshold_value: float
    ):
        """Create a new performance alert."""
        alert_id = f"{component}_{metric_name}_{int(time.time())}"
        
        alert = SystemAlert(
            alert_id=alert_id,
            level=level,
            message=message,
            component=component,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        self.alerts.append(alert)
        
        # Limit alerts history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
        
        logger.log(
            logging.WARNING if level == AlertLevel.WARNING else logging.ERROR,
            f"Performance Alert: {message} (Current: {current_value:.2f}, Threshold: {threshold_value:.2f})"
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = datetime.now()
        
        # Benchmark status
        benchmark_status = {}
        for name, benchmark in self.benchmarks.items():
            benchmark_status[name] = {
                "current_value": benchmark.current_value,
                "target_value": benchmark.target_value,
                "is_passing": benchmark.is_passing,
                "performance_ratio": benchmark.performance_ratio,
                "last_measured": benchmark.last_measured.isoformat() if benchmark.last_measured else None,
                "unit": benchmark.unit,
                "description": benchmark.description
            }
        
        # System metrics summary
        system_metrics = self.metrics_collector.get_all_metrics_summary()
        
        # Recent alerts
        recent_alerts = [alert.to_dict() for alert in self.alerts[-50:]]
        unresolved_alerts = [alert.to_dict() for alert in self.alerts if not alert.resolved]
        
        # Workflow performance summary
        workflow_summary = {}
        for workflow_type in ["single", "batch", "concurrent"]:
            execution_times = self.workflow_metrics.get(f"{workflow_type}_execution_time", [])
            successes = self.workflow_metrics.get(f"{workflow_type}_success", [])
            
            if execution_times:
                workflow_summary[workflow_type] = {
                    "avg_execution_time": statistics.mean(execution_times[-50:]),  # Last 50
                    "p95_execution_time": statistics.quantiles(execution_times[-50:], n=20)[18] if len(execution_times) >= 20 else max(execution_times[-50:]),
                    "success_rate": (sum(successes[-50:]) / len(successes[-50:])) * 100 if successes else 0
                }
        
        # Agent performance summary
        agent_summary = {}
        for agent_type, agent_data in self.agent_metrics.items():
            agent_summary[agent_type] = {}
            for operation, values in agent_data.items():
                if values and "_time" in operation:
                    op_name = operation.replace("_time", "")
                    agent_summary[agent_type][op_name] = {
                        "avg_time": statistics.mean(values[-20:]),  # Last 20
                        "success_rate": (
                            sum(agent_data.get(f"{op_name}_success", [])[-20:]) / 
                            len(agent_data.get(f"{op_name}_success", [])[-20:]) * 100
                        ) if agent_data.get(f"{op_name}_success") else 0
                    }
        
        return {
            "timestamp": current_time.isoformat(),
            "benchmarks": benchmark_status,
            "system_metrics": system_metrics,
            "workflow_performance": workflow_summary,
            "agent_performance": agent_summary,
            "alerts": {
                "recent_alerts": recent_alerts,
                "unresolved_count": len(unresolved_alerts),
                "critical_count": len([a for a in unresolved_alerts if a["level"] == "critical"]),
                "warning_count": len([a for a in unresolved_alerts if a["level"] == "warning"])
            },
            "health_score": self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        # Benchmark scores
        for benchmark in self.benchmarks.values():
            if benchmark.current_value is not None:
                ratio = benchmark.performance_ratio
                score = min(100, ratio * 100) if ratio <= 1.0 else max(0, 100 - (ratio - 1) * 50)
                scores.append(score)
        
        # System resource scores
        system_metrics = self.metrics_collector.get_all_metrics_summary()
        
        # CPU score (inverted - lower is better)
        if "cpu_percent" in system_metrics:
            cpu_score = max(0, 100 - system_metrics["cpu_percent"].get("mean", 0))
            scores.append(cpu_score)
        
        # Memory score (inverted - lower is better) 
        if "memory_percent" in system_metrics:
            memory_score = max(0, 100 - system_metrics["memory_percent"].get("mean", 0))
            scores.append(memory_score)
        
        # Alert penalty
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        critical_penalty = len([a for a in unresolved_alerts if a.level == AlertLevel.CRITICAL]) * 20
        warning_penalty = len([a for a in unresolved_alerts if a.level == AlertLevel.WARNING]) * 5
        alert_penalty = min(50, critical_penalty + warning_penalty)
        
        base_score = statistics.mean(scores) if scores else 50
        final_score = max(0, base_score - alert_penalty)
        
        return final_score
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Add benchmark metrics
        for name, benchmark in self.benchmarks.items():
            if benchmark.current_value is not None:
                metric = PerformanceMetric(
                    name=f"grocery_scanner_benchmark_{name}",
                    metric_type=MetricType.GAUGE,
                    value=benchmark.current_value,
                    tags={"target": str(benchmark.target_value), "unit": benchmark.unit}
                )
                lines.append(metric.to_prometheus_format())
        
        # Add system metrics
        system_metrics = self.metrics_collector.get_all_metrics_summary()
        for metric_name, summary in system_metrics.items():
            if "mean" in summary:
                metric = PerformanceMetric(
                    name=f"grocery_scanner_system_{metric_name}",
                    metric_type=MetricType.GAUGE,
                    value=summary["mean"]
                )
                lines.append(metric.to_prometheus_format())
        
        # Add alert count
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        alert_metric = PerformanceMetric(
            name="grocery_scanner_alerts_total",
            metric_type=MetricType.GAUGE,
            value=len(unresolved_alerts)
        )
        lines.append(alert_metric.to_prometheus_format())
        
        return "\n".join(lines)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_score = self._calculate_health_score()
        
        status = "healthy"
        if health_score < 50:
            status = "unhealthy"
        elif health_score < 80:
            status = "degraded"
        
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in unresolved_alerts if a.level == AlertLevel.CRITICAL]
        
        return {
            "status": status,
            "health_score": health_score,
            "timestamp": datetime.now().isoformat(),
            "critical_issues": len(critical_alerts),
            "total_unresolved_alerts": len(unresolved_alerts),
            "benchmarks_passing": sum(1 for b in self.benchmarks.values() if b.is_passing),
            "total_benchmarks": len(self.benchmarks)
        }


# Global instance
performance_monitor = PerformanceMonitor()