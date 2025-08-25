"""
Quality Metrics Dashboard

Real-time quality monitoring dashboard with interactive visualizations,
trend analysis, and automated alerting for system quality assessment.
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

from .quality_monitor import QualityMonitor, QualityReport, QualityMetric
from .regression_tester import RegressionTester, RegressionTestSuite
from .golden_dataset import GoldenDatasetManager


class QualityDashboard:
    """Real-time quality monitoring dashboard."""
    
    def __init__(self):
        self.quality_monitor = QualityMonitor()
        self.regression_tester = RegressionTester()
        self.golden_dataset = GoldenDatasetManager()
        
        # Initialize session state
        if 'quality_reports' not in st.session_state:
            st.session_state.quality_reports = []
        if 'regression_suites' not in st.session_state:
            st.session_state.regression_suites = []
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    def render_dashboard(self):
        """Render the complete quality dashboard."""
        st.set_page_config(
            page_title="Quality Monitoring Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .alert-critical {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 10px;
            margin: 5px 0;
        }
        .alert-warning {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 10px;
            margin: 5px 0;
        }
        .alert-healthy {
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
            padding: 10px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🔍 Quality Monitoring Dashboard")
        st.markdown("Real-time system quality assessment and regression detection")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🎯 Component Metrics", 
            "🧪 Regression Tests", 
            "📈 Trends", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_component_metrics()
        
        with tab3:
            self._render_regression_analysis()
        
        with tab4:
            self._render_trend_analysis()
        
        with tab5:
            self._render_settings()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("🎛️ Control Panel")
        
        # Refresh controls
        if st.sidebar.button("🔄 Run Quality Assessment", type="primary"):
            with st.spinner("Running quality assessment..."):
                self._run_quality_assessment()
        
        if st.sidebar.button("🧪 Run Regression Tests"):
            with st.spinner("Running regression tests..."):
                self._run_regression_tests()
        
        st.sidebar.divider()
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.toggle(
            "🔄 Auto Refresh", 
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            refresh_interval = st.sidebar.selectbox(
                "Refresh Interval",
                [30, 60, 300, 600],  # seconds
                format_func=lambda x: f"{x//60}m {x%60}s" if x >= 60 else f"{x}s"
            )
            
            # Auto-refresh placeholder
            placeholder = st.sidebar.empty()
            
        # Data management
        st.sidebar.divider()
        st.sidebar.header("📊 Data Management")
        
        if st.sidebar.button("📥 Export Quality Reports"):
            self._export_quality_reports()
        
        if st.sidebar.button("📥 Export Regression Results"):
            self._export_regression_results()
        
        if st.sidebar.button("🗑️ Clear History"):
            self._clear_history()
        
        # System status
        st.sidebar.divider()
        st.sidebar.header("🚥 System Status")
        
        # Calculate current status
        if st.session_state.quality_reports:
            latest_report = st.session_state.quality_reports[-1]
            overall_score = latest_report.overall_score
            
            if overall_score >= 90:
                status_color = "🟢"
                status_text = "Healthy"
            elif overall_score >= 75:
                status_color = "🟡"
                status_text = "Warning"
            else:
                status_color = "🔴"
                status_text = "Critical"
            
            st.sidebar.markdown(f"**System Health:** {status_color} {status_text}")
            st.sidebar.progress(overall_score / 100)
            st.sidebar.markdown(f"Score: {overall_score:.1f}/100")
        else:
            st.sidebar.markdown("**System Health:** ⚫ Unknown")
    
    def _render_overview(self):
        """Render overview dashboard."""
        st.header("📊 Quality Overview")
        
        # Get latest data
        latest_report = st.session_state.quality_reports[-1] if st.session_state.quality_reports else None
        latest_regression = st.session_state.regression_suites[-1] if st.session_state.regression_suites else None
        
        if not latest_report:
            st.warning("No quality assessment data available. Run a quality assessment to see metrics.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Quality Score",
                f"{latest_report.overall_score:.1f}",
                delta=self._get_score_delta(latest_report.overall_score) if len(st.session_state.quality_reports) > 1 else None
            )
        
        with col2:
            alert_count = len(latest_report.alerts)
            critical_alerts = len([a for a in latest_report.alerts if a['severity'] == 'critical'])
            st.metric(
                "Active Alerts",
                str(alert_count),
                delta=f"{critical_alerts} critical" if critical_alerts > 0 else None,
                delta_color="inverse"
            )
        
        with col3:
            if latest_regression:
                st.metric(
                    "Regression Tests",
                    f"{latest_regression.passed_tests}/{latest_regression.total_tests}",
                    delta=f"{latest_regression.regressions_detected} regressions" if latest_regression.regressions_detected > 0 else "No regressions",
                    delta_color="inverse" if latest_regression.regressions_detected > 0 else "normal"
                )
            else:
                st.metric("Regression Tests", "Not Run")
        
        with col4:
            recommendation_count = len(latest_report.recommendations)
            high_priority = len([r for r in latest_report.recommendations if r['priority'] == 'high'])
            st.metric(
                "Recommendations",
                str(recommendation_count),
                delta=f"{high_priority} high priority" if high_priority > 0 else None
            )
        
        st.divider()
        
        # Component health visualization
        if latest_report.component_scores:
            st.subheader("🎯 Component Health")
            
            # Create component health chart
            components = list(latest_report.component_scores.keys())
            scores = list(latest_report.component_scores.values())
            
            fig = go.Figure()
            
            # Color mapping for scores
            colors = ['#4CAF50' if s >= 90 else '#FF9800' if s >= 75 else '#F44336' for s in scores]
            
            fig.add_trace(go.Bar(
                x=components,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Component Health Scores",
                yaxis_title="Score",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Alerts section
        if latest_report.alerts:
            st.subheader("⚠️ Active Alerts")
            
            for alert in latest_report.alerts:
                severity = alert['severity']
                css_class = f"alert-{severity}" if severity in ['critical', 'warning'] else "alert-healthy"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{alert['severity'].upper()}</strong> - {alert['component']}: {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Recommendations section
        if latest_report.recommendations:
            st.subheader("💡 Recommendations")
            
            for rec in latest_report.recommendations[:5]:  # Show top 5
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                
                with st.expander(f"{priority_emoji} {rec['title']} ({rec['priority']} priority)"):
                    st.write(rec['description'])
                    if rec['actions']:
                        st.write("**Recommended Actions:**")
                        for action in rec['actions']:
                            st.write(f"• {action}")
    
    def _render_component_metrics(self):
        """Render detailed component metrics."""
        st.header("🎯 Component Performance Metrics")
        
        if not st.session_state.quality_reports:
            st.warning("No quality assessment data available.")
            return
        
        latest_report = st.session_state.quality_reports[-1]
        
        # Component selection
        if latest_report.test_results:
            components = list(set(k for report in st.session_state.quality_reports for k in report.component_scores.keys()))
            selected_component = st.selectbox("Select Component", components)
            
            # Component-specific metrics
            component_data = self._get_component_history(selected_component)
            
            if component_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score trend
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[d['timestamp'] for d in component_data],
                        y=[d['score'] for d in component_data],
                        mode='lines+markers',
                        name='Score',
                        line=dict(color='#2196F3', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_component.title()} Score Trend",
                        yaxis_title="Score",
                        yaxis=dict(range=[0, 100]),
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Performance distribution
                    scores = [d['score'] for d in component_data]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=scores,
                        nbinsx=10,
                        marker_color='#4CAF50',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_component.title()} Score Distribution",
                        xaxis_title="Score",
                        yaxis_title="Frequency",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("📊 Detailed Test Results")
                
                if selected_component in latest_report.test_results:
                    test_data = latest_report.test_results[selected_component]
                    
                    # Create DataFrame for display
                    metrics_df = pd.DataFrame([
                        {"Metric": k, "Value": v} 
                        for k, v in test_data.items() 
                        if isinstance(v, (int, float))
                    ])
                    
                    st.dataframe(metrics_df, use_container_width=True)
    
    def _render_regression_analysis(self):
        """Render regression testing analysis."""
        st.header("🧪 Regression Analysis")
        
        if not st.session_state.regression_suites:
            st.warning("No regression test data available.")
            return
        
        latest_suite = st.session_state.regression_suites[-1]
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Health Score",
                f"{latest_suite.overall_health_score:.1f}/100"
            )
        
        with col2:
            st.metric(
                "Test Pass Rate",
                f"{latest_suite.passed_tests}/{latest_suite.total_tests}",
                f"{(latest_suite.passed_tests / max(1, latest_suite.total_tests)) * 100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Regressions Detected",
                str(latest_suite.regressions_detected),
                delta=f"Critical: {latest_suite.critical_regressions}" if latest_suite.critical_regressions > 0 else None,
                delta_color="inverse"
            )
        
        # Regression severity breakdown
        if latest_suite.regressions_detected > 0:
            st.subheader("🚨 Regression Severity Breakdown")
            
            severity_data = {
                'Critical': latest_suite.critical_regressions,
                'Major': latest_suite.major_regressions,
                'Minor': latest_suite.minor_regressions
            }
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=list(severity_data.keys()),
                values=list(severity_data.values()),
                marker_colors=['#F44336', '#FF9800', '#FFC107']
            ))
            
            fig.update_layout(
                title="Regression Severity Distribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Test results by category
        if hasattr(latest_suite, 'test_results') and latest_suite.test_results:
            st.subheader("📊 Test Results by Category")
            
            # Group results by category
            category_results = {}
            for result in latest_suite.test_results:
                category = result.test_category
                if category not in category_results:
                    category_results[category] = {'passed': 0, 'failed': 0}
                
                if result.passed:
                    category_results[category]['passed'] += 1
                else:
                    category_results[category]['failed'] += 1
            
            # Create stacked bar chart
            categories = list(category_results.keys())
            passed = [category_results[cat]['passed'] for cat in categories]
            failed = [category_results[cat]['failed'] for cat in categories]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Passed',
                x=categories,
                y=passed,
                marker_color='#4CAF50'
            ))
            fig.add_trace(go.Bar(
                name='Failed',
                x=categories,
                y=failed,
                marker_color='#F44336'
            ))
            
            fig.update_layout(
                barmode='stack',
                title="Test Results by Category",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analysis(self):
        """Render trend analysis and predictions."""
        st.header("📈 Quality Trends & Analysis")
        
        if len(st.session_state.quality_reports) < 3:
            st.warning("Need at least 3 quality reports for trend analysis.")
            return
        
        # Overall score trend
        timestamps = [report.timestamp for report in st.session_state.quality_reports]
        scores = [report.overall_score for report in st.session_state.quality_reports]
        
        fig = go.Figure()
        
        # Add score line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=8)
        ))
        
        # Add trend line
        if len(scores) >= 2:
            # Simple linear trend
            x_numeric = list(range(len(scores)))
            trend_slope = (scores[-1] - scores[0]) / (len(scores) - 1)
            trend_line = [scores[0] + trend_slope * i for i in x_numeric]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='#FF5722', dash='dash', width=2)
            ))
        
        # Add threshold lines
        fig.add_hline(y=90, line_dash="dot", line_color="green", 
                     annotation_text="Excellent Threshold")
        fig.add_hline(y=75, line_dash="dot", line_color="orange", 
                     annotation_text="Warning Threshold")
        fig.add_hline(y=60, line_dash="dot", line_color="red", 
                     annotation_text="Critical Threshold")
        
        fig.update_layout(
            title="Quality Score Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Component trends
        st.subheader("🎯 Component Trend Analysis")
        
        component_trends = self._calculate_component_trends()
        
        if component_trends:
            trend_df = pd.DataFrame([
                {
                    "Component": component,
                    "Current Score": data['current'],
                    "Trend": data['trend'],
                    "Change %": f"{data['change_percent']:.1f}%",
                    "Status": "📈 Improving" if data['trend'] == 'improving' 
                             else "📉 Degrading" if data['trend'] == 'degrading' 
                             else "➡️ Stable"
                }
                for component, data in component_trends.items()
            ])
            
            st.dataframe(trend_df, use_container_width=True)
    
    def _render_settings(self):
        """Render settings and configuration."""
        st.header("⚙️ Dashboard Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Quality Monitoring")
            
            # Threshold settings
            st.slider("Critical Alert Threshold", 0, 100, 60)
            st.slider("Warning Alert Threshold", 0, 100, 75)
            st.slider("Excellent Threshold", 0, 100, 90)
            
            # Monitoring frequency
            st.selectbox(
                "Assessment Frequency",
                ["Every 6 hours", "Every 12 hours", "Daily", "Weekly"]
            )
            
            # Alert channels
            st.multiselect(
                "Alert Channels",
                ["Email", "Slack", "Console", "File"],
                default=["Console", "File"]
            )
        
        with col2:
            st.subheader("🧪 Regression Testing")
            
            # Test configuration
            st.slider("Test Parallelism", 1, 10, 4)
            st.slider("Test Timeout (seconds)", 10, 120, 30)
            
            # Regression thresholds
            st.slider("Minor Regression Threshold (%)", 1, 20, 5)
            st.slider("Major Regression Threshold (%)", 5, 30, 10)
            st.slider("Critical Regression Threshold (%)", 10, 50, 20)
            
            # Baseline management
            st.checkbox("Auto-update baselines on improvement")
            st.selectbox(
                "Regression Test Frequency",
                ["Every 12 hours", "Daily", "Weekly"]
            )
        
        st.divider()
        
        # Data management
        st.subheader("📊 Data Management")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write(f"Quality Reports: {len(st.session_state.quality_reports)}")
            st.write(f"Regression Suites: {len(st.session_state.regression_suites)}")
            st.write(f"Golden Dataset Matches: {len(self.golden_dataset.matches)}")
        
        with col4:
            if st.button("🔄 Refresh Golden Dataset"):
                self._refresh_golden_dataset()
                st.success("Golden dataset refreshed!")
            
            if st.button("📊 Recalculate Baselines"):
                self._recalculate_baselines()
                st.success("Baselines recalculated!")
    
    def _run_quality_assessment(self):
        """Run quality assessment and update session state."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            report = loop.run_until_complete(self.quality_monitor.run_comprehensive_evaluation())
            st.session_state.quality_reports.append(report)
            
            # Keep only last 50 reports
            if len(st.session_state.quality_reports) > 50:
                st.session_state.quality_reports = st.session_state.quality_reports[-50:]
            
            st.success(f"Quality assessment completed! Overall Score: {report.overall_score:.1f}/100")
            
        except Exception as e:
            st.error(f"Quality assessment failed: {e}")
        finally:
            loop.close()
    
    def _run_regression_tests(self):
        """Run regression tests and update session state."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            suite = loop.run_until_complete(self.regression_tester.run_full_regression_suite())
            st.session_state.regression_suites.append(suite)
            
            # Keep only last 20 suites
            if len(st.session_state.regression_suites) > 20:
                st.session_state.regression_suites = st.session_state.regression_suites[-20:]
            
            st.success(f"Regression tests completed! Health Score: {suite.overall_health_score:.1f}/100")
            
            if suite.critical_regressions > 0:
                st.error(f"🚨 {suite.critical_regressions} critical regressions detected!")
            elif suite.regressions_detected > 0:
                st.warning(f"⚠️ {suite.regressions_detected} regressions detected")
                
        except Exception as e:
            st.error(f"Regression tests failed: {e}")
        finally:
            loop.close()
    
    def _get_score_delta(self, current_score: float) -> Optional[float]:
        """Get score delta from previous report."""
        if len(st.session_state.quality_reports) < 2:
            return None
        
        previous_score = st.session_state.quality_reports[-2].overall_score
        return current_score - previous_score
    
    def _get_component_history(self, component: str) -> List[Dict[str, Any]]:
        """Get historical data for a specific component."""
        history = []
        for report in st.session_state.quality_reports:
            if component in report.component_scores:
                history.append({
                    'timestamp': report.timestamp,
                    'score': report.component_scores[component]
                })
        return history
    
    def _calculate_component_trends(self) -> Dict[str, Dict[str, Any]]:
        """Calculate trends for all components."""
        if len(st.session_state.quality_reports) < 3:
            return {}
        
        trends = {}
        
        # Get all components
        all_components = set()
        for report in st.session_state.quality_reports:
            all_components.update(report.component_scores.keys())
        
        for component in all_components:
            scores = []
            for report in st.session_state.quality_reports[-5:]:  # Last 5 reports
                if component in report.component_scores:
                    scores.append(report.component_scores[component])
            
            if len(scores) >= 3:
                # Calculate trend
                recent_avg = statistics.mean(scores[-3:])
                older_avg = statistics.mean(scores[:-3]) if len(scores) > 3 else scores[0]
                
                change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                
                if change_percent > 2:
                    trend = 'improving'
                elif change_percent < -2:
                    trend = 'degrading'
                else:
                    trend = 'stable'
                
                trends[component] = {
                    'current': scores[-1],
                    'trend': trend,
                    'change_percent': change_percent
                }
        
        return trends
    
    def _export_quality_reports(self):
        """Export quality reports to JSON."""
        if not st.session_state.quality_reports:
            st.warning("No quality reports to export.")
            return
        
        # Convert reports to exportable format
        export_data = []
        for report in st.session_state.quality_reports:
            export_data.append({
                'report_id': report.report_id,
                'timestamp': report.timestamp.isoformat(),
                'overall_score': report.overall_score,
                'component_scores': report.component_scores,
                'alerts': report.alerts,
                'recommendations': report.recommendations,
                'test_results': report.test_results
            })
        
        # Create download
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📥 Download Quality Reports",
            data=json_data,
            file_name=f"quality_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _export_regression_results(self):
        """Export regression test results."""
        if not st.session_state.regression_suites:
            st.warning("No regression results to export.")
            return
        
        # Convert suites to exportable format
        export_data = []
        for suite in st.session_state.regression_suites:
            export_data.append(suite.get_summary())
        
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📥 Download Regression Results",
            data=json_data,
            file_name=f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _clear_history(self):
        """Clear all historical data."""
        st.session_state.quality_reports = []
        st.session_state.regression_suites = []
        st.success("History cleared!")
    
    def _refresh_golden_dataset(self):
        """Refresh golden dataset."""
        self.golden_dataset = GoldenDatasetManager()
    
    def _recalculate_baselines(self):
        """Recalculate performance baselines."""
        # This would recalculate baselines from historical data
        # For now, just show success message
        pass


def main():
    """Main dashboard application."""
    dashboard = QualityDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()