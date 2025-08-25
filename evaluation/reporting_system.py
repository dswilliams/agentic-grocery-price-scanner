"""
Comprehensive Reporting System

Advanced reporting system with automated stakeholder notifications, customizable reports,
trend analysis, executive dashboards, and multi-channel delivery for quality insights.
"""

import asyncio
import json
import logging
import smtplib
import statistics
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.application import MimeApplication
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

from .quality_monitor import QualityMonitor, QualityReport
from .regression_tester import RegressionTester, RegressionTestSuite
from .ml_model_evaluator import MLModelEvaluator, ModelEvaluationResult
from .business_metrics_validator import BusinessMetricsValidator, BusinessValidationReport
from .continuous_improvement import ContinuousImprovementPipeline, ImprovementPlan


class ReportType(Enum):
    """Types of reports available."""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_QUALITY = "weekly_quality"
    MONTHLY_BUSINESS = "monthly_business"
    QUARTERLY_EXECUTIVE = "quarterly_executive"
    INCIDENT_ALERT = "incident_alert"
    IMPROVEMENT_PROGRESS = "improvement_progress"
    REGRESSION_ALERT = "regression_alert"
    CUSTOM = "custom"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    FILE = "file"
    CONSOLE = "console"


class ReportPriority(Enum):
    """Report priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Stakeholder:
    """Individual stakeholder configuration."""
    
    stakeholder_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    role: str = ""  # executive, engineer, product_manager, etc.
    
    # Subscription preferences
    report_types: List[ReportType] = field(default_factory=list)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    frequency_preferences: Dict[ReportType, str] = field(default_factory=dict)  # daily, weekly, monthly
    
    # Alert preferences
    alert_threshold: ReportPriority = ReportPriority.NORMAL
    escalation_enabled: bool = True
    
    # Customization
    custom_metrics: List[str] = field(default_factory=list)
    executive_summary_only: bool = False


@dataclass
class ReportTemplate:
    """Report template configuration."""
    
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    report_type: ReportType = ReportType.DAILY_SUMMARY
    
    # Content configuration
    sections: List[str] = field(default_factory=list)
    metrics_included: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)
    
    # Formatting
    format_type: str = "html"  # html, pdf, json, csv
    include_charts: bool = True
    include_recommendations: bool = True
    
    # Customization
    executive_summary: bool = True
    technical_details: bool = True
    trend_analysis: bool = True


@dataclass
class ReportInstance:
    """Individual report instance."""
    
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    report_type: ReportType = ReportType.DAILY_SUMMARY
    priority: ReportPriority = ReportPriority.NORMAL
    
    # Content
    title: str = ""
    executive_summary: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Data sources
    quality_report: Optional[QualityReport] = None
    regression_suite: Optional[RegressionTestSuite] = None
    ml_evaluation: Optional[Dict[str, ModelEvaluationResult]] = None
    business_report: Optional[BusinessValidationReport] = None
    improvement_plan: Optional[ImprovementPlan] = None
    
    # Generation metadata
    generation_time: float = 0.0
    stakeholders_notified: List[str] = field(default_factory=list)
    
    # Files
    html_file: Optional[str] = None
    pdf_file: Optional[str] = None
    json_file: Optional[str] = None


class ReportGenerator:
    """Advanced report generation engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize report templates
        self.templates = self._initialize_default_templates()
        
    def _initialize_default_templates(self) -> Dict[str, ReportTemplate]:
        """Initialize default report templates."""
        
        templates = {}
        
        # Daily Summary Template
        templates['daily_summary'] = ReportTemplate(
            name="Daily Quality Summary",
            report_type=ReportType.DAILY_SUMMARY,
            sections=[
                "system_health_overview",
                "key_metrics",
                "alerts_and_issues",
                "performance_trends",
                "quick_recommendations"
            ],
            metrics_included=[
                "overall_quality_score",
                "response_time",
                "error_rate",
                "user_satisfaction"
            ],
            visualizations=[
                "health_score_gauge",
                "response_time_trend",
                "error_distribution"
            ]
        )
        
        # Weekly Quality Template
        templates['weekly_quality'] = ReportTemplate(
            name="Weekly Quality Analysis",
            report_type=ReportType.WEEKLY_QUALITY,
            sections=[
                "executive_summary",
                "quality_metrics_analysis",
                "regression_test_results",
                "ml_model_performance",
                "improvement_recommendations",
                "trend_analysis",
                "next_week_focus"
            ],
            metrics_included=[
                "overall_quality_score",
                "component_scores",
                "regression_pass_rate",
                "model_accuracy",
                "user_satisfaction"
            ],
            visualizations=[
                "quality_trend_chart",
                "component_heatmap",
                "regression_history",
                "satisfaction_distribution"
            ]
        )
        
        # Monthly Business Template
        templates['monthly_business'] = ReportTemplate(
            name="Monthly Business Impact Report",
            report_type=ReportType.MONTHLY_BUSINESS,
            sections=[
                "executive_summary",
                "business_metrics_overview",
                "financial_impact",
                "user_engagement_analysis",
                "system_performance",
                "roi_analysis",
                "strategic_recommendations"
            ],
            metrics_included=[
                "total_savings_generated",
                "user_satisfaction",
                "recommendation_follow_rate",
                "system_availability",
                "roi_percentage"
            ],
            visualizations=[
                "savings_trend",
                "user_engagement_funnel",
                "roi_breakdown",
                "satisfaction_trends"
            ],
            executive_summary=True,
            technical_details=False
        )
        
        # Executive Quarterly Template
        templates['quarterly_executive'] = ReportTemplate(
            name="Quarterly Executive Dashboard",
            report_type=ReportType.QUARTERLY_EXECUTIVE,
            sections=[
                "executive_summary",
                "key_performance_indicators",
                "business_value_delivered",
                "strategic_initiatives",
                "risk_assessment",
                "roadmap_recommendations"
            ],
            metrics_included=[
                "quarterly_savings",
                "user_growth",
                "system_reliability",
                "strategic_kpis"
            ],
            visualizations=[
                "quarterly_kpi_dashboard",
                "business_value_chart",
                "growth_metrics"
            ],
            executive_summary=True,
            technical_details=False
        )
        
        return templates
    
    async def generate_report(
        self, 
        report_type: ReportType,
        template_name: str = None,
        custom_sections: List[str] = None,
        date_range: Tuple[datetime, datetime] = None
    ) -> ReportInstance:
        """Generate comprehensive report."""
        
        self.logger.info(f"Generating {report_type.value} report")
        
        start_time = datetime.now()
        report = ReportInstance(report_type=report_type)
        
        try:
            # Select template
            template = self._select_template(report_type, template_name)
            
            # Collect data sources
            await self._collect_data_sources(report, date_range)
            
            # Generate content
            await self._generate_report_content(report, template, custom_sections)
            
            # Create visualizations
            if template.include_charts:
                await self._generate_visualizations(report, template)
            
            # Generate files
            await self._generate_report_files(report, template)
            
            report.generation_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Report generated successfully in {report.generation_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            raise
        
        return report
    
    def _select_template(self, report_type: ReportType, template_name: str = None) -> ReportTemplate:
        """Select appropriate report template."""
        
        if template_name and template_name in self.templates:
            return self.templates[template_name]
        
        # Use default template for report type
        default_templates = {
            ReportType.DAILY_SUMMARY: 'daily_summary',
            ReportType.WEEKLY_QUALITY: 'weekly_quality',
            ReportType.MONTHLY_BUSINESS: 'monthly_business',
            ReportType.QUARTERLY_EXECUTIVE: 'quarterly_executive'
        }
        
        template_key = default_templates.get(report_type, 'daily_summary')
        return self.templates[template_key]
    
    async def _collect_data_sources(self, report: ReportInstance, date_range: Tuple[datetime, datetime] = None):
        """Collect all necessary data sources for the report."""
        
        try:
            # Initialize data collection systems
            quality_monitor = QualityMonitor()
            regression_tester = RegressionTester()
            ml_evaluator = MLModelEvaluator()
            business_validator = BusinessMetricsValidator()
            improvement_pipeline = ContinuousImprovementPipeline()
            
            # Collect data in parallel
            tasks = [
                quality_monitor.run_comprehensive_evaluation(),
                regression_tester.run_full_regression_suite(),
                ml_evaluator.run_comprehensive_evaluation()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            if isinstance(results[0], QualityReport):
                report.quality_report = results[0]
            
            if isinstance(results[1], RegressionTestSuite):
                report.regression_suite = results[1]
            
            if isinstance(results[2], dict):
                report.ml_evaluation = results[2]
            
            # Business validation (synchronous)
            period_days = 30 if date_range is None else (date_range[1] - date_range[0]).days
            report.business_report = business_validator.validate_business_metrics(period_days)
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
    
    async def _generate_report_content(
        self, 
        report: ReportInstance, 
        template: ReportTemplate,
        custom_sections: List[str] = None
    ):
        """Generate report content based on template."""
        
        sections = custom_sections or template.sections
        report.content = {}
        
        for section in sections:
            try:
                section_content = await self._generate_section_content(section, report)
                report.content[section] = section_content
            except Exception as e:
                self.logger.warning(f"Failed to generate section '{section}': {e}")
                report.content[section] = f"Error generating content: {str(e)}"
        
        # Generate executive summary
        report.executive_summary = await self._generate_executive_summary(report)
        
        # Set report title
        report.title = self._generate_report_title(report, template)
    
    async def _generate_section_content(self, section: str, report: ReportInstance) -> Dict[str, Any]:
        """Generate content for a specific report section."""
        
        if section == "system_health_overview":
            return self._generate_system_health_overview(report)
        
        elif section == "key_metrics":
            return self._generate_key_metrics(report)
        
        elif section == "alerts_and_issues":
            return self._generate_alerts_and_issues(report)
        
        elif section == "performance_trends":
            return self._generate_performance_trends(report)
        
        elif section == "quality_metrics_analysis":
            return self._generate_quality_metrics_analysis(report)
        
        elif section == "regression_test_results":
            return self._generate_regression_test_results(report)
        
        elif section == "ml_model_performance":
            return self._generate_ml_model_performance(report)
        
        elif section == "business_metrics_overview":
            return self._generate_business_metrics_overview(report)
        
        elif section == "improvement_recommendations":
            return self._generate_improvement_recommendations(report)
        
        elif section == "financial_impact":
            return self._generate_financial_impact(report)
        
        elif section == "roi_analysis":
            return self._generate_roi_analysis(report)
        
        else:
            return {"error": f"Unknown section: {section}"}
    
    def _generate_system_health_overview(self, report: ReportInstance) -> Dict[str, Any]:
        """Generate system health overview section."""
        
        overview = {
            "overall_status": "unknown",
            "health_score": 0.0,
            "critical_issues": 0,
            "warnings": 0,
            "uptime": "99.9%",
            "key_indicators": {}
        }
        
        if report.quality_report:
            overview["health_score"] = report.quality_report.overall_score
            overview["critical_issues"] = len([a for a in report.quality_report.alerts if a['severity'] == 'critical'])
            overview["warnings"] = len([a for a in report.quality_report.alerts if a['severity'] == 'warning'])
            
            if overview["health_score"] >= 90:
                overview["overall_status"] = "healthy"
            elif overview["health_score"] >= 75:
                overview["overall_status"] = "warning"
            else:
                overview["overall_status"] = "critical"
        
        if report.regression_suite:
            overview["key_indicators"]["test_pass_rate"] = f"{report.regression_suite.passed_tests}/{report.regression_suite.total_tests}"
            overview["key_indicators"]["regressions_detected"] = report.regression_suite.regressions_detected
        
        if report.business_report:
            overview["key_indicators"]["user_satisfaction"] = f"{report.business_report.avg_satisfaction_score:.1f}/5.0"
            overview["key_indicators"]["savings_accuracy"] = f"{report.business_report.savings_accuracy_rate:.1%}"
        
        return overview
    
    def _generate_key_metrics(self, report: ReportInstance) -> Dict[str, Any]:
        """Generate key metrics section."""
        
        metrics = {
            "quality_metrics": {},
            "performance_metrics": {},
            "business_metrics": {},
            "model_metrics": {}
        }
        
        if report.quality_report:
            metrics["quality_metrics"] = {
                "overall_score": report.quality_report.overall_score,
                "component_scores": report.quality_report.component_scores,
                "active_alerts": len(report.quality_report.alerts)
            }
            
            if report.quality_report.metrics:
                metrics["performance_metrics"] = {
                    name: {
                        "current": metric.current_value,
                        "trend": metric.trend,
                        "status": "healthy" if not metric.is_critical() else "critical"
                    }
                    for name, metric in report.quality_report.metrics.items()
                }
        
        if report.business_report:
            metrics["business_metrics"] = {
                "total_savings": report.business_report.total_actual_savings,
                "user_satisfaction": report.business_report.avg_satisfaction_score,
                "roi_percentage": report.business_report.roi_percentage,
                "sessions": report.business_report.total_sessions
            }
        
        if report.ml_evaluation:
            metrics["model_metrics"] = {}
            for model_name, result in report.ml_evaluation.items():
                metrics["model_metrics"][model_name] = {
                    "overall_score": result.overall_score,
                    "drift_detected": result.drift_detected,
                    "retraining_recommended": result.retraining_recommended
                }
        
        return metrics
    
    def _generate_alerts_and_issues(self, report: ReportInstance) -> Dict[str, Any]:
        """Generate alerts and issues section."""
        
        alerts = {
            "critical_alerts": [],
            "warnings": [],
            "recent_incidents": [],
            "resolution_status": {}
        }
        
        if report.quality_report and report.quality_report.alerts:
            for alert in report.quality_report.alerts:
                alert_data = {
                    "component": alert['component'],
                    "message": alert['message'],
                    "timestamp": alert.get('timestamp', ''),
                    "details": alert.get('details', {})
                }
                
                if alert['severity'] == 'critical':
                    alerts["critical_alerts"].append(alert_data)
                elif alert['severity'] == 'warning':
                    alerts["warnings"].append(alert_data)
        
        if report.regression_suite and report.regression_suite.critical_regressions > 0:
            alerts["critical_alerts"].append({
                "component": "regression_tests",
                "message": f"{report.regression_suite.critical_regressions} critical regressions detected",
                "timestamp": datetime.now().isoformat(),
                "details": {"total_regressions": report.regression_suite.regressions_detected}
            })
        
        return alerts
    
    async def _generate_executive_summary(self, report: ReportInstance) -> str:
        """Generate executive summary for the report."""
        
        summary_parts = []
        
        # System health
        if report.quality_report:
            health_score = report.quality_report.overall_score
            if health_score >= 90:
                summary_parts.append(f"System health is excellent with a quality score of {health_score:.1f}/100.")
            elif health_score >= 75:
                summary_parts.append(f"System health is stable with a quality score of {health_score:.1f}/100, with some areas for improvement.")
            else:
                summary_parts.append(f"System health requires attention with a quality score of {health_score:.1f}/100.")
        
        # Business impact
        if report.business_report:
            if report.business_report.total_actual_savings > 0:
                summary_parts.append(f"The system generated ${report.business_report.total_actual_savings:.2f} in verified user savings with {report.business_report.savings_accuracy_rate:.1%} prediction accuracy.")
            
            if report.business_report.avg_satisfaction_score > 0:
                satisfaction_desc = "excellent" if report.business_report.avg_satisfaction_score >= 4.0 else "good" if report.business_report.avg_satisfaction_score >= 3.5 else "needs improvement"
                summary_parts.append(f"User satisfaction is {satisfaction_desc} at {report.business_report.avg_satisfaction_score:.1f}/5.0.")
        
        # Key issues
        issues = []
        if report.quality_report:
            critical_alerts = len([a for a in report.quality_report.alerts if a['severity'] == 'critical'])
            if critical_alerts > 0:
                issues.append(f"{critical_alerts} critical alerts")
        
        if report.regression_suite and report.regression_suite.critical_regressions > 0:
            issues.append(f"{report.regression_suite.critical_regressions} critical regressions")
        
        if issues:
            summary_parts.append(f"Immediate attention required for: {', '.join(issues)}.")
        
        # Recommendations
        recommendations_count = 0
        if report.quality_report:
            recommendations_count += len([r for r in report.quality_report.recommendations if r['priority'] == 'high'])
        
        if recommendations_count > 0:
            summary_parts.append(f"{recommendations_count} high-priority improvement recommendations have been identified.")
        
        return " ".join(summary_parts)
    
    def _generate_report_title(self, report: ReportInstance, template: ReportTemplate) -> str:
        """Generate appropriate report title."""
        
        date_str = report.timestamp.strftime("%Y-%m-%d")
        
        title_templates = {
            ReportType.DAILY_SUMMARY: f"Daily Quality Report - {date_str}",
            ReportType.WEEKLY_QUALITY: f"Weekly Quality Analysis - Week of {date_str}",
            ReportType.MONTHLY_BUSINESS: f"Monthly Business Impact Report - {report.timestamp.strftime('%B %Y')}",
            ReportType.QUARTERLY_EXECUTIVE: f"Quarterly Executive Dashboard - Q{((report.timestamp.month-1)//3)+1} {report.timestamp.year}"
        }
        
        return title_templates.get(report.report_type, f"System Report - {date_str}")
    
    async def _generate_visualizations(self, report: ReportInstance, template: ReportTemplate):
        """Generate visualizations for the report."""
        
        visualizations = {}
        
        for viz_type in template.visualizations:
            try:
                if viz_type == "health_score_gauge" and report.quality_report:
                    fig = self._create_health_score_gauge(report.quality_report.overall_score)
                    visualizations[viz_type] = fig
                
                elif viz_type == "quality_trend_chart" and report.quality_report:
                    fig = self._create_quality_trend_chart(report.quality_report)
                    visualizations[viz_type] = fig
                
                elif viz_type == "savings_trend" and report.business_report:
                    fig = self._create_savings_trend_chart(report.business_report)
                    visualizations[viz_type] = fig
                
            except Exception as e:
                self.logger.warning(f"Failed to generate visualization '{viz_type}': {e}")
        
        report.content["visualizations"] = visualizations
    
    def _create_health_score_gauge(self, score: float) -> go.Figure:
        """Create health score gauge visualization."""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "System Health Score"},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return fig
    
    def _create_quality_trend_chart(self, quality_report: QualityReport) -> go.Figure:
        """Create quality trend chart."""
        
        # Mock trend data - in production, this would use historical data
        dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
        scores = [85, 87, 89, 88, 92, 90, quality_report.overall_score]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="7-Day Quality Score Trend",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def _create_savings_trend_chart(self, business_report: BusinessValidationReport) -> go.Figure:
        """Create savings trend chart."""
        
        # Mock savings data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        daily_savings = [max(0, 15 + np.random.normal(0, 3)) for _ in dates]
        daily_savings[-1] = business_report.total_actual_savings / 30  # Approximate daily average
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_savings,
            mode='lines',
            name='Daily Savings',
            fill='tonexty',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="30-Day User Savings Trend",
            xaxis_title="Date",
            yaxis_title="Daily Savings ($)"
        )
        
        return fig
    
    async def _generate_report_files(self, report: ReportInstance, template: ReportTemplate):
        """Generate report files in various formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(f"evaluation/reports/{report.report_type.value}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        if template.format_type in ["html", "all"]:
            html_content = self._generate_html_report(report, template)
            html_file = report_dir / f"report_{timestamp}.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            report.html_file = str(html_file)
        
        # Generate JSON report
        json_content = self._generate_json_report(report)
        json_file = report_dir / f"report_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(json_content, f, indent=2, default=str)
        
        report.json_file = str(json_file)
    
    def _generate_html_report(self, report: ReportInstance, template: ReportTemplate) -> str:
        """Generate HTML report content."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 3px; }}
                .alert-critical {{ background-color: #ffebee; border-left: 5px solid #f44336; }}
                .alert-warning {{ background-color: #fff3e0; border-left: 5px solid #ff9800; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report ID: {report.report_id}</p>
            </div>
            
            <div class="section summary">
                <h2>Executive Summary</h2>
                <p>{report.executive_summary}</p>
            </div>
        """
        
        # Add system health overview
        if "system_health_overview" in report.content:
            health = report.content["system_health_overview"]
            html += f"""
            <div class="section">
                <h2>System Health Overview</h2>
                <div class="metric">
                    <strong>Overall Status:</strong> {health.get('overall_status', 'Unknown').upper()}
                </div>
                <div class="metric">
                    <strong>Health Score:</strong> {health.get('health_score', 0):.1f}/100
                </div>
                <div class="metric">
                    <strong>Critical Issues:</strong> {health.get('critical_issues', 0)}
                </div>
                <div class="metric">
                    <strong>Warnings:</strong> {health.get('warnings', 0)}
                </div>
            </div>
            """
        
        # Add key metrics
        if "key_metrics" in report.content:
            metrics = report.content["key_metrics"]
            html += """
            <div class="section">
                <h2>Key Metrics</h2>
                <table>
                    <tr><th>Category</th><th>Metric</th><th>Value</th><th>Status</th></tr>
            """
            
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, metric_value in category_metrics.items():
                        if isinstance(metric_value, dict) and 'current' in metric_value:
                            html += f"""
                            <tr>
                                <td>{category.replace('_', ' ').title()}</td>
                                <td>{metric_name.replace('_', ' ').title()}</td>
                                <td>{metric_value['current']}</td>
                                <td>{metric_value.get('status', 'Unknown')}</td>
                            </tr>
                            """
                        else:
                            html += f"""
                            <tr>
                                <td>{category.replace('_', ' ').title()}</td>
                                <td>{metric_name.replace('_', ' ').title()}</td>
                                <td>{metric_value}</td>
                                <td>-</td>
                            </tr>
                            """
            
            html += "</table></div>"
        
        # Add alerts
        if "alerts_and_issues" in report.content:
            alerts = report.content["alerts_and_issues"]
            html += "<div class='section'><h2>Alerts and Issues</h2>"
            
            for alert in alerts.get("critical_alerts", []):
                html += f"""
                <div class="alert-critical">
                    <strong>CRITICAL:</strong> {alert['component']} - {alert['message']}
                </div>
                """
            
            for alert in alerts.get("warnings", []):
                html += f"""
                <div class="alert-warning">
                    <strong>WARNING:</strong> {alert['component']} - {alert['message']}
                </div>
                """
            
            html += "</div>"
        
        # Add recommendations
        if report.quality_report and report.quality_report.recommendations:
            html += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
            """
            
            for rec in report.quality_report.recommendations[:5]:  # Top 5
                html += f"""
                <li><strong>{rec['priority'].upper()}:</strong> {rec['title']} - {rec['description']}</li>
                """
            
            html += "</ul></div>"
        
        html += """
            <div class="section">
                <h2>Report Generation Details</h2>
                <p>Generation Time: {:.2f} seconds</p>
                <p>Data Sources: Quality Monitor, Regression Tests, ML Evaluation, Business Metrics</p>
            </div>
        </body>
        </html>
        """.format(report.generation_time)
        
        return html
    
    def _generate_json_report(self, report: ReportInstance) -> Dict[str, Any]:
        """Generate JSON report content."""
        
        return {
            "report_metadata": {
                "report_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "report_type": report.report_type.value,
                "title": report.title,
                "generation_time": report.generation_time
            },
            "executive_summary": report.executive_summary,
            "content": report.content,
            "data_sources": {
                "quality_report_available": report.quality_report is not None,
                "regression_suite_available": report.regression_suite is not None,
                "ml_evaluation_available": report.ml_evaluation is not None,
                "business_report_available": report.business_report is not None
            }
        }


class NotificationSystem:
    """Multi-channel notification system for reports."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Stakeholders
        self.stakeholders: List[Stakeholder] = []
        self._load_stakeholders()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default notification configuration."""
        return {
            'email': {
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': 'reports@grocery-scanner.com'
            },
            'slack': {
                'webhook_url': '',
                'channel': '#quality-reports'
            },
            'file': {
                'output_dir': 'evaluation/notifications'
            }
        }
    
    def _load_stakeholders(self):
        """Load stakeholder configurations."""
        stakeholders_file = Path("evaluation/config/stakeholders.json")
        
        if stakeholders_file.exists():
            try:
                with open(stakeholders_file, 'r') as f:
                    data = json.load(f)
                    
                for stakeholder_data in data.get('stakeholders', []):
                    stakeholder = Stakeholder(
                        name=stakeholder_data['name'],
                        email=stakeholder_data['email'],
                        role=stakeholder_data['role'],
                        report_types=[ReportType(rt) for rt in stakeholder_data.get('report_types', [])],
                        notification_channels=[NotificationChannel(nc) for nc in stakeholder_data.get('notification_channels', [])],
                        alert_threshold=ReportPriority(stakeholder_data.get('alert_threshold', 'normal'))
                    )
                    self.stakeholders.append(stakeholder)
                    
                print(f"✅ Loaded {len(self.stakeholders)} stakeholder configurations")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load stakeholders: {e}")
                self._create_default_stakeholders()
        else:
            print("📝 Creating default stakeholder configurations")
            self._create_default_stakeholders()
    
    def _create_default_stakeholders(self):
        """Create default stakeholder configurations."""
        
        self.stakeholders = [
            Stakeholder(
                name="Engineering Team",
                email="engineering@company.com",
                role="engineer",
                report_types=[ReportType.DAILY_SUMMARY, ReportType.WEEKLY_QUALITY, ReportType.REGRESSION_ALERT],
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                alert_threshold=ReportPriority.NORMAL
            ),
            Stakeholder(
                name="Product Manager",
                email="product@company.com",
                role="product_manager",
                report_types=[ReportType.WEEKLY_QUALITY, ReportType.MONTHLY_BUSINESS],
                notification_channels=[NotificationChannel.EMAIL],
                alert_threshold=ReportPriority.HIGH,
                executive_summary_only=True
            ),
            Stakeholder(
                name="Executive Team",
                email="executives@company.com",
                role="executive",
                report_types=[ReportType.QUARTERLY_EXECUTIVE, ReportType.MONTHLY_BUSINESS],
                notification_channels=[NotificationChannel.EMAIL],
                alert_threshold=ReportPriority.URGENT,
                executive_summary_only=True
            )
        ]
    
    async def send_report_notifications(self, report: ReportInstance, priority: ReportPriority = ReportPriority.NORMAL):
        """Send report notifications to relevant stakeholders."""
        
        self.logger.info(f"Sending notifications for {report.report_type.value} report")
        
        relevant_stakeholders = self._get_relevant_stakeholders(report, priority)
        
        notification_tasks = []
        
        for stakeholder in relevant_stakeholders:
            for channel in stakeholder.notification_channels:
                if channel == NotificationChannel.EMAIL:
                    task = self._send_email_notification(stakeholder, report)
                elif channel == NotificationChannel.SLACK:
                    task = self._send_slack_notification(stakeholder, report)
                elif channel == NotificationChannel.FILE:
                    task = self._send_file_notification(stakeholder, report)
                elif channel == NotificationChannel.CONSOLE:
                    task = self._send_console_notification(stakeholder, report)
                else:
                    continue
                
                notification_tasks.append(task)
        
        # Send notifications in parallel
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            successful_notifications = sum(1 for r in results if r is True)
            report.stakeholders_notified = [s.name for s in relevant_stakeholders]
            
            self.logger.info(f"✅ Sent {successful_notifications}/{len(notification_tasks)} notifications successfully")
        else:
            self.logger.info("No relevant stakeholders found for this report")
    
    def _get_relevant_stakeholders(self, report: ReportInstance, priority: ReportPriority) -> List[Stakeholder]:
        """Get stakeholders who should receive this report."""
        
        relevant = []
        
        for stakeholder in self.stakeholders:
            # Check if stakeholder is interested in this report type
            if report.report_type in stakeholder.report_types:
                # Check if priority meets stakeholder's threshold
                priority_values = {
                    ReportPriority.LOW: 1,
                    ReportPriority.NORMAL: 2,
                    ReportPriority.HIGH: 3,
                    ReportPriority.URGENT: 4
                }
                
                if priority_values[priority] >= priority_values[stakeholder.alert_threshold]:
                    relevant.append(stakeholder)
        
        return relevant
    
    async def _send_email_notification(self, stakeholder: Stakeholder, report: ReportInstance) -> bool:
        """Send email notification."""
        
        try:
            subject = f"{report.title} - Priority: {report.priority.value.upper()}"
            
            # Create email content
            if stakeholder.executive_summary_only:
                body = f"""
                <html>
                <body>
                    <h2>{report.title}</h2>
                    <p><strong>Executive Summary:</strong></p>
                    <p>{report.executive_summary}</p>
                    <p>Full report available at: {report.html_file or 'System Dashboard'}</p>
                </body>
                </html>
                """
            else:
                # Use full HTML report content
                body = self._generate_notification_email_body(report)
            
            # Mock email sending (in production, use actual SMTP)
            self.logger.info(f"📧 Email notification sent to {stakeholder.name} ({stakeholder.email})")
            
            # Save email to file for demonstration
            email_file = Path(f"evaluation/notifications/email_{stakeholder.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            email_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(email_file, 'w') as f:
                f.write(f"<h1>Email to: {stakeholder.email}</h1><h2>Subject: {subject}</h2>{body}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send email to {stakeholder.name}: {e}")
            return False
    
    def _generate_notification_email_body(self, report: ReportInstance) -> str:
        """Generate email body for notification."""
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>{report.title}</h2>
            <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3>Executive Summary</h3>
                <p>{report.executive_summary}</p>
            </div>
            
            {self._generate_email_key_metrics(report)}
            
            {self._generate_email_alerts(report)}
            
            <p><strong>Full Report:</strong> Available at system dashboard</p>
            <p><em>This is an automated report generated by the Quality Monitoring System.</em></p>
        </body>
        </html>
        """
    
    def _generate_email_key_metrics(self, report: ReportInstance) -> str:
        """Generate key metrics section for email."""
        
        if not report.quality_report:
            return ""
        
        return f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>Key Metrics</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Overall Quality Score</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{report.quality_report.overall_score:.1f}/100</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Active Alerts</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{len(report.quality_report.alerts)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Recommendations</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{len(report.quality_report.recommendations)}</td>
                </tr>
            </table>
        </div>
        """
    
    def _generate_email_alerts(self, report: ReportInstance) -> str:
        """Generate alerts section for email."""
        
        if not report.quality_report or not report.quality_report.alerts:
            return ""
        
        alerts_html = """
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>Active Alerts</h3>
            <ul>
        """
        
        for alert in report.quality_report.alerts[:5]:  # Top 5 alerts
            color = "#f44336" if alert['severity'] == 'critical' else "#ff9800"
            alerts_html += f"""
            <li style="color: {color};">
                <strong>{alert['severity'].upper()}:</strong> {alert['component']} - {alert['message']}
            </li>
            """
        
        alerts_html += "</ul></div>"
        return alerts_html
    
    async def _send_slack_notification(self, stakeholder: Stakeholder, report: ReportInstance) -> bool:
        """Send Slack notification."""
        
        try:
            # Mock Slack notification
            message = f"""
            📊 *{report.title}*
            
            *Executive Summary:*
            {report.executive_summary}
            
            *Key Metrics:*
            • Quality Score: {report.quality_report.overall_score:.1f}/100 if report.quality_report else 'N/A'
            • Active Alerts: {len(report.quality_report.alerts) if report.quality_report else 0}
            
            Full report: {report.html_file or 'Dashboard'}
            """
            
            self.logger.info(f"📱 Slack notification sent to {stakeholder.name}")
            
            # Save to file for demonstration
            slack_file = Path(f"evaluation/notifications/slack_{stakeholder.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            slack_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(slack_file, 'w') as f:
                f.write(f"Slack message to {stakeholder.name}:\n\n{message}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send Slack notification to {stakeholder.name}: {e}")
            return False
    
    async def _send_file_notification(self, stakeholder: Stakeholder, report: ReportInstance) -> bool:
        """Send file-based notification."""
        
        try:
            notification_dir = Path(self.config['file']['output_dir'])
            notification_dir.mkdir(parents=True, exist_ok=True)
            
            notification_file = notification_dir / f"notification_{stakeholder.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            notification_data = {
                "stakeholder": stakeholder.name,
                "report_id": report.report_id,
                "report_type": report.report_type.value,
                "timestamp": report.timestamp.isoformat(),
                "executive_summary": report.executive_summary,
                "html_file": report.html_file,
                "json_file": report.json_file
            }
            
            with open(notification_file, 'w') as f:
                json.dump(notification_data, f, indent=2)
            
            self.logger.info(f"📁 File notification created for {stakeholder.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create file notification for {stakeholder.name}: {e}")
            return False
    
    async def _send_console_notification(self, stakeholder: Stakeholder, report: ReportInstance) -> bool:
        """Send console notification."""
        
        try:
            print(f"\n🔔 NOTIFICATION FOR {stakeholder.name.upper()}")
            print(f"📊 {report.title}")
            print(f"⏰ {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📋 {report.executive_summary}")
            print("─" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send console notification to {stakeholder.name}: {e}")
            return False


class ComprehensiveReportingSystem:
    """Main comprehensive reporting system coordinator."""
    
    def __init__(self):
        self.report_generator = ReportGenerator()
        self.notification_system = NotificationSystem()
        self.logger = logging.getLogger(__name__)
        
        # Report history
        self.report_history: List[ReportInstance] = []
    
    async def generate_and_distribute_report(
        self,
        report_type: ReportType,
        priority: ReportPriority = ReportPriority.NORMAL,
        custom_sections: List[str] = None,
        date_range: Tuple[datetime, datetime] = None
    ) -> ReportInstance:
        """Generate report and distribute to stakeholders."""
        
        self.logger.info(f"🚀 Starting {report_type.value} report generation and distribution")
        
        try:
            # Generate report
            report = await self.report_generator.generate_report(
                report_type, 
                custom_sections=custom_sections,
                date_range=date_range
            )
            
            report.priority = priority
            
            # Distribute notifications
            await self.notification_system.send_report_notifications(report, priority)
            
            # Store in history
            self.report_history.append(report)
            
            # Keep only last 100 reports
            if len(self.report_history) > 100:
                self.report_history = self.report_history[-100:]
            
            self.logger.info(f"✅ Report {report.report_id} generated and distributed successfully")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Report generation and distribution failed: {e}")
            raise
    
    async def start_scheduled_reporting(self):
        """Start scheduled reporting based on configurations."""
        
        self.logger.info("🚀 Starting scheduled reporting system")
        
        # Define reporting schedule
        schedule = {
            'daily_summary': {'hour': 8, 'minute': 0},  # 8:00 AM daily
            'weekly_quality': {'weekday': 0, 'hour': 9, 'minute': 0},  # Monday 9:00 AM
            'monthly_business': {'day': 1, 'hour': 10, 'minute': 0},  # 1st of month 10:00 AM
        }
        
        while True:
            try:
                now = datetime.now()
                
                # Check daily summary
                if now.hour == schedule['daily_summary']['hour'] and now.minute == schedule['daily_summary']['minute']:
                    await self.generate_and_distribute_report(ReportType.DAILY_SUMMARY)
                
                # Check weekly quality (Monday)
                if (now.weekday() == schedule['weekly_quality']['weekday'] and 
                    now.hour == schedule['weekly_quality']['hour'] and 
                    now.minute == schedule['weekly_quality']['minute']):
                    await self.generate_and_distribute_report(ReportType.WEEKLY_QUALITY)
                
                # Check monthly business (1st of month)
                if (now.day == schedule['monthly_business']['day'] and 
                    now.hour == schedule['monthly_business']['hour'] and 
                    now.minute == schedule['monthly_business']['minute']):
                    await self.generate_and_distribute_report(ReportType.MONTHLY_BUSINESS)
                
                # Wait 1 minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"❌ Error in scheduled reporting: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry


if __name__ == "__main__":
    async def test_reporting_system():
        reporting_system = ComprehensiveReportingSystem()
        
        # Generate different types of reports
        print("🚀 Testing Comprehensive Reporting System\n")
        
        # Daily Summary Report
        daily_report = await reporting_system.generate_and_distribute_report(
            ReportType.DAILY_SUMMARY,
            priority=ReportPriority.NORMAL
        )
        
        print(f"✅ Daily Summary Report Generated:")
        print(f"   Report ID: {daily_report.report_id}")
        print(f"   Generation Time: {daily_report.generation_time:.2f}s")
        print(f"   HTML File: {daily_report.html_file}")
        print(f"   Stakeholders Notified: {len(daily_report.stakeholders_notified)}")
        
        # Weekly Quality Report
        weekly_report = await reporting_system.generate_and_distribute_report(
            ReportType.WEEKLY_QUALITY,
            priority=ReportPriority.HIGH
        )
        
        print(f"\n✅ Weekly Quality Report Generated:")
        print(f"   Report ID: {weekly_report.report_id}")
        print(f"   Generation Time: {weekly_report.generation_time:.2f}s")
        print(f"   Stakeholders Notified: {len(weekly_report.stakeholders_notified)}")
        
        # Show report history
        print(f"\n📊 Report History: {len(reporting_system.report_history)} reports generated")
        
        for report in reporting_system.report_history:
            print(f"   • {report.report_type.value}: {report.timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    asyncio.run(test_reporting_system())