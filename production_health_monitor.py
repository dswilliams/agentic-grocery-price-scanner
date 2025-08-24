"""
Production health monitoring and diagnostic system.
Comprehensive monitoring dashboard with real-time health checks and alerting.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path

# Import all production components
from agentic_grocery_price_scanner.config.store_profiles import store_profile_manager
from agentic_grocery_price_scanner.reliability import scraping_reliability_manager
from agentic_grocery_price_scanner.quality import data_quality_manager
from agentic_grocery_price_scanner.monitoring import performance_monitor
from agentic_grocery_price_scanner.caching import cache_manager
from agentic_grocery_price_scanner.recovery import error_recovery_manager

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Overall system health status."""
    
    status: str  # healthy, degraded, unhealthy
    score: float  # 0-100
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class ProductionHealthMonitor:
    """Comprehensive production health monitoring system."""
    
    def __init__(self):
        self.monitoring_interval = 30.0  # seconds
        self.is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Health check functions
        self.health_checks = {
            "store_profiles": self._check_store_profiles,
            "scraping_reliability": self._check_scraping_reliability,
            "data_quality": self._check_data_quality,
            "performance_monitor": self._check_performance_monitor,
            "cache_system": self._check_cache_system,
            "error_recovery": self._check_error_recovery
        }
        
        # Alerting thresholds
        self.alert_thresholds = {
            "overall_health_score": 70.0,
            "store_availability": 80.0,
            "cache_hit_rate": 70.0,
            "success_rate": 85.0,
            "response_time_p95": 120.0,
            "memory_usage_mb": 1000.0
        }
        
        logger.info("Initialized ProductionHealthMonitor")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        
        # Start performance monitor
        await performance_monitor.start_monitoring()
        
        # Start health check loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Started production health monitoring (interval: {self.monitoring_interval}s)")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        await performance_monitor.stop_monitoring()
        
        logger.info("Stopped production health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Perform comprehensive health check
                health_status = await self.comprehensive_health_check()
                
                # Log health status
                self._log_health_status(health_status)
                
                # Check for critical issues
                await self._handle_critical_issues(health_status)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def comprehensive_health_check(self) -> HealthStatus:
        """Perform comprehensive system health check."""
        logger.debug("Starting comprehensive health check")
        
        component_health = {}
        all_alerts = []
        recommendations = []
        component_scores = []
        
        # Execute all health checks concurrently
        health_check_tasks = {
            name: check_func() for name, check_func in self.health_checks.items()
        }
        
        results = await asyncio.gather(*health_check_tasks.values(), return_exceptions=True)
        
        # Process results
        for i, (component_name, result) in enumerate(zip(health_check_tasks.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {component_name}: {result}")
                component_health[component_name] = {
                    "status": "error",
                    "error": str(result),
                    "score": 0.0
                }
                component_scores.append(0.0)
            else:
                component_health[component_name] = result
                component_scores.append(result.get("score", 50.0))
                
                # Collect alerts and recommendations
                if "alerts" in result:
                    all_alerts.extend(result["alerts"])
                
                if "recommendations" in result:
                    recommendations.extend(result["recommendations"])
        
        # Calculate overall health score
        overall_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
        
        # Determine overall status
        if overall_score >= 85.0:
            overall_status = "healthy"
        elif overall_score >= 70.0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        health_status = HealthStatus(
            status=overall_status,
            score=overall_score,
            timestamp=datetime.now(),
            components=component_health,
            alerts=all_alerts,
            recommendations=list(set(recommendations))  # Remove duplicates
        )
        
        logger.debug(f"Health check completed: {overall_status} ({overall_score:.1f}/100)")
        
        return health_status
    
    async def _check_store_profiles(self) -> Dict[str, Any]:
        """Check store profiles and availability."""
        health_report = store_profile_manager.get_store_health_report()
        available_stores = sum(1 for store_data in health_report["stores"].values() if store_data["available"])
        total_stores = len(health_report["stores"])
        
        availability_rate = (available_stores / total_stores * 100) if total_stores > 0 else 0
        
        alerts = []
        recommendations = []
        
        # Check availability threshold
        if availability_rate < self.alert_thresholds["store_availability"]:
            alerts.append({
                "level": "warning" if availability_rate > 50 else "critical",
                "message": f"Store availability is {availability_rate:.1f}% (threshold: {self.alert_thresholds['store_availability']}%)",
                "component": "store_profiles"
            })
            recommendations.append("Check store connectivity and circuit breaker status")
        
        # Check for degraded stores
        degraded_stores = [
            store_id for store_id, data in health_report["stores"].items()
            if data["health_status"] == "degraded"
        ]
        
        if degraded_stores:
            recommendations.append(f"Monitor degraded stores: {', '.join(degraded_stores)}")
        
        score = min(100.0, availability_rate + (sum(
            store_data["success_rate"] for store_data in health_report["stores"].values()
        ) / len(health_report["stores"]) if health_report["stores"] else 0))
        
        return {
            "status": "healthy" if availability_rate >= 80 else "degraded" if availability_rate >= 50 else "unhealthy",
            "score": score,
            "available_stores": available_stores,
            "total_stores": total_stores,
            "availability_rate": availability_rate,
            "degraded_stores": degraded_stores,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    async def _check_scraping_reliability(self) -> Dict[str, Any]:
        """Check scraping reliability system."""
        reliability_report = scraping_reliability_manager.get_reliability_report()
        
        success_rate = reliability_report["overall_metrics"]["success_rate"]
        avg_response_time = reliability_report["overall_metrics"]["avg_response_time"]
        cache_hit_rate = reliability_report["overall_metrics"]["cache_hit_rate"]
        
        alerts = []
        recommendations = []
        
        # Check success rate
        if success_rate < self.alert_thresholds["success_rate"]:
            alerts.append({
                "level": "warning" if success_rate > 70 else "critical",
                "message": f"Scraping success rate is {success_rate:.1f}% (threshold: {self.alert_thresholds['success_rate']}%)",
                "component": "scraping_reliability"
            })
            recommendations.append("Review recent failures and adjust retry strategies")
        
        # Check response time
        if avg_response_time > self.alert_thresholds["response_time_p95"]:
            alerts.append({
                "level": "warning",
                "message": f"Average response time is {avg_response_time:.1f}s (threshold: {self.alert_thresholds['response_time_p95']}s)",
                "component": "scraping_reliability"
            })
            recommendations.append("Consider optimizing scraping strategies or increasing timeouts")
        
        # Check cache performance
        if cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            recommendations.append("Review cache warming strategies to improve hit rate")
        
        score = (success_rate + cache_hit_rate) / 2 - (avg_response_time / 10)  # Penalty for slow responses
        score = max(0, min(100, score))
        
        return {
            "status": "healthy" if success_rate >= 85 and avg_response_time < 30 else "degraded" if success_rate >= 70 else "unhealthy",
            "score": score,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    async def _check_data_quality(self) -> Dict[str, Any]:
        """Check data quality system."""
        quality_report = data_quality_manager.get_quality_report()
        
        if "latest_metrics" not in quality_report:
            return {
                "status": "unknown",
                "score": 50.0,
                "message": "No quality data available",
                "alerts": [],
                "recommendations": ["Run quality assessment on recent data"]
            }
        
        quality_score = quality_report["latest_metrics"]["overall_quality_score"]
        issue_rate = quality_report["latest_metrics"]["issue_rate"]
        
        alerts = []
        recommendations = []
        
        # Check quality score
        if quality_score < 80:
            alerts.append({
                "level": "warning" if quality_score > 60 else "critical",
                "message": f"Data quality score is {quality_score:.1f}% (target: >80%)",
                "component": "data_quality"
            })
            recommendations.append("Review data collection methods and validation rules")
        
        # Check issue rate
        if issue_rate > 15:
            recommendations.append("High issue rate detected - review data sources")
        
        return {
            "status": "healthy" if quality_score >= 80 else "degraded" if quality_score >= 60 else "unhealthy",
            "score": quality_score,
            "quality_score": quality_score,
            "issue_rate": issue_rate,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    async def _check_performance_monitor(self) -> Dict[str, Any]:
        """Check performance monitoring system."""
        health_check = await performance_monitor.health_check()
        performance_report = performance_monitor.get_performance_report()
        
        health_score = health_check["health_score"]
        critical_issues = health_check["critical_issues"]
        
        alerts = []
        recommendations = []
        
        # Check critical alerts
        if critical_issues > 0:
            alerts.append({
                "level": "critical",
                "message": f"{critical_issues} critical performance issues detected",
                "component": "performance_monitor"
            })
            recommendations.append("Review and resolve critical performance alerts")
        
        # Check health score
        if health_score < self.alert_thresholds["overall_health_score"]:
            alerts.append({
                "level": "warning",
                "message": f"Performance health score is {health_score:.1f}% (threshold: {self.alert_thresholds['overall_health_score']}%)",
                "component": "performance_monitor"
            })
        
        # Check system resources
        if "system_metrics" in performance_report:
            memory_usage = performance_report["system_metrics"].get("memory_mb", {}).get("mean", 0)
            if memory_usage > self.alert_thresholds["memory_usage_mb"]:
                alerts.append({
                    "level": "warning",
                    "message": f"High memory usage: {memory_usage:.1f}MB (threshold: {self.alert_thresholds['memory_usage_mb']}MB)",
                    "component": "performance_monitor"
                })
                recommendations.append("Monitor for memory leaks or optimize resource usage")
        
        return {
            "status": health_check["status"],
            "score": health_score,
            "health_score": health_score,
            "critical_issues": critical_issues,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    async def _check_cache_system(self) -> Dict[str, Any]:
        """Check caching system health."""
        cache_health = await cache_manager.health_check()
        cache_analysis = await cache_manager.analyze_cache_performance()
        
        overall_hit_rate = cache_analysis["performance_metrics"]["overall_hit_rate"]
        cache_efficiency = cache_analysis["performance_metrics"]["cache_efficiency_score"]
        
        alerts = []
        recommendations = cache_analysis.get("recommendations", [])
        
        # Check hit rate
        if overall_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alerts.append({
                "level": "warning",
                "message": f"Cache hit rate is {overall_hit_rate:.1f}% (threshold: {self.alert_thresholds['cache_hit_rate']}%)",
                "component": "cache_system"
            })
        
        # Check memory utilization
        memory_util = cache_analysis["memory_cache"]["utilization"]
        if memory_util > 90:
            alerts.append({
                "level": "warning",
                "message": f"Memory cache utilization is {memory_util:.1f}%",
                "component": "cache_system"
            })
        
        return {
            "status": cache_health["status"],
            "score": cache_efficiency,
            "hit_rate": overall_hit_rate,
            "cache_efficiency": cache_efficiency,
            "memory_utilization": memory_util,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    async def _check_error_recovery(self) -> Dict[str, Any]:
        """Check error recovery system."""
        recovery_health = await error_recovery_manager.health_check()
        recovery_report = await error_recovery_manager.get_recovery_report()
        
        manual_review_items = recovery_health["manual_review_items"]
        recent_failures = recovery_health["recent_failures_24h"]
        
        alerts = []
        recommendations = []
        
        # Check manual review queue
        if manual_review_items > 10:
            alerts.append({
                "level": "warning" if manual_review_items < 50 else "critical",
                "message": f"{manual_review_items} items require manual review",
                "component": "error_recovery"
            })
            recommendations.append("Review and process items in dead letter queue")
        
        # Check recent failure rate
        if recent_failures > 20:
            recommendations.append("High failure rate detected - review error patterns")
        
        score = max(0, 100 - (manual_review_items * 2) - (recent_failures / 2))
        
        return {
            "status": recovery_health["status"],
            "score": score,
            "manual_review_items": manual_review_items,
            "recent_failures_24h": recent_failures,
            "alerts": alerts,
            "recommendations": recommendations
        }
    
    def _log_health_status(self, health_status: HealthStatus):
        """Log health status information."""
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌"
        }
        
        emoji = status_emoji.get(health_status.status, "❓")
        
        logger.info(f"{emoji} System Health: {health_status.status.upper()} ({health_status.score:.1f}/100)")
        
        # Log component statuses
        for component, data in health_status.components.items():
            comp_status = data.get("status", "unknown")
            comp_score = data.get("score", 0)
            comp_emoji = status_emoji.get(comp_status, "❓")
            logger.debug(f"  {comp_emoji} {component}: {comp_status} ({comp_score:.1f})")
        
        # Log alerts
        if health_status.alerts:
            logger.warning(f"Active alerts: {len(health_status.alerts)}")
            for alert in health_status.alerts[:3]:  # Show first 3 alerts
                logger.warning(f"  - {alert['level'].upper()}: {alert['message']}")
    
    async def _handle_critical_issues(self, health_status: HealthStatus):
        """Handle critical health issues."""
        critical_alerts = [a for a in health_status.alerts if a.get("level") == "critical"]
        
        if critical_alerts:
            logger.critical(f"CRITICAL ISSUES DETECTED: {len(critical_alerts)} critical alerts")
            
            for alert in critical_alerts:
                logger.critical(f"CRITICAL: {alert['message']} ({alert.get('component', 'unknown')})")
            
            # Could trigger automated remediation actions here
            # For example: restart services, clear caches, etc.
    
    async def export_health_report(self, output_file: str = None) -> Dict[str, Any]:
        """Export detailed health report."""
        health_status = await self.comprehensive_health_check()
        
        # Create detailed report
        report = {
            "timestamp": health_status.timestamp.isoformat(),
            "overall_status": health_status.status,
            "overall_score": health_status.score,
            "components": health_status.components,
            "alerts": health_status.alerts,
            "recommendations": health_status.recommendations,
            "system_info": {
                "monitoring_interval": self.monitoring_interval,
                "alert_thresholds": self.alert_thresholds
            }
        }
        
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Health report exported to {output_path}")
        
        return report


async def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Production Health Monitor")
    parser.add_argument("--mode", choices=["check", "monitor", "export"], default="check",
                       help="Operation mode")
    parser.add_argument("--interval", type=float, default=30.0,
                       help="Monitoring interval in seconds")
    parser.add_argument("--export-file", type=str,
                       help="Export report to file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize health monitor
    health_monitor = ProductionHealthMonitor()
    health_monitor.monitoring_interval = args.interval
    
    try:
        if args.mode == "check":
            # Single health check
            print("🔍 Performing comprehensive health check...")
            health_status = await health_monitor.comprehensive_health_check()
            
            # Print summary
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
            emoji = status_emoji.get(health_status.status, "❓")
            
            print(f"\n{emoji} System Health: {health_status.status.upper()} ({health_status.score:.1f}/100)")
            print(f"📊 Components checked: {len(health_status.components)}")
            print(f"⚠️ Active alerts: {len(health_status.alerts)}")
            print(f"💡 Recommendations: {len(health_status.recommendations)}")
            
            # Show component breakdown
            print("\n📋 Component Status:")
            for component, data in health_status.components.items():
                comp_emoji = status_emoji.get(data.get("status", "unknown"), "❓")
                print(f"  {comp_emoji} {component}: {data.get('score', 0):.1f}/100")
            
            # Show alerts if any
            if health_status.alerts:
                print("\n🚨 Active Alerts:")
                for alert in health_status.alerts:
                    level_emoji = {"warning": "⚠️", "critical": "🔥"}
                    print(f"  {level_emoji.get(alert['level'], '❓')} {alert['message']}")
            
            # Show recommendations
            if health_status.recommendations:
                print("\n💡 Recommendations:")
                for rec in health_status.recommendations:
                    print(f"  • {rec}")
        
        elif args.mode == "monitor":
            # Continuous monitoring
            print(f"🔄 Starting continuous health monitoring (interval: {args.interval}s)")
            print("Press Ctrl+C to stop...")
            
            await health_monitor.start_monitoring()
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping health monitoring...")
            finally:
                await health_monitor.stop_monitoring()
        
        elif args.mode == "export":
            # Export detailed report
            output_file = args.export_file or f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            print(f"📊 Exporting health report to {output_file}...")
            
            report = await health_monitor.export_health_report(output_file)
            print(f"✅ Health report exported successfully")
            print(f"📈 Overall score: {report['overall_score']:.1f}/100")
    
    except Exception as e:
        logger.error(f"Error in health monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())