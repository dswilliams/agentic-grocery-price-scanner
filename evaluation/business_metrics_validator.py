"""
Business Metrics Validation System

Comprehensive system for validating business impact metrics including actual vs predicted
savings, time efficiency, user satisfaction, and ROI measurement for the grocery
price comparison system.
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from .golden_dataset import GoldenDatasetManager
from ..data_models.product import Product
from ..data_models.ingredient import Ingredient
from ..workflow.grocery_workflow import GroceryWorkflow


class SatisfactionLevel(Enum):
    """User satisfaction levels."""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5


class BusinessImpactCategory(Enum):
    """Categories of business impact metrics."""
    FINANCIAL = "financial"
    TIME_EFFICIENCY = "time_efficiency"
    USER_EXPERIENCE = "user_experience"
    ACCURACY = "accuracy"
    COVERAGE = "coverage"
    RELIABILITY = "reliability"


@dataclass
class UserSession:
    """Represents a user shopping session for metric tracking."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Input data
    ingredients_requested: List[str] = field(default_factory=list)
    optimization_strategy: str = "balanced"
    budget_limit: Optional[float] = None
    preferred_stores: List[str] = field(default_factory=list)
    
    # System outputs
    predicted_savings: float = 0.0
    predicted_time_saved: float = 0.0  # minutes
    recommended_stores: List[str] = field(default_factory=list)
    total_products_found: int = 0
    match_confidence_avg: float = 0.0
    
    # Actual outcomes (collected via feedback/survey)
    actual_savings: Optional[float] = None
    actual_time_spent: Optional[float] = None  # minutes
    stores_actually_visited: List[str] = field(default_factory=list)
    products_actually_purchased: int = 0
    user_satisfaction: Optional[SatisfactionLevel] = None
    
    # Derived metrics
    savings_accuracy: Optional[float] = None
    time_efficiency_score: Optional[float] = None
    recommendation_follow_rate: float = 0.0
    
    # Feedback
    user_feedback: str = ""
    issues_encountered: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def calculate_derived_metrics(self):
        """Calculate derived business metrics."""
        # Savings accuracy
        if self.actual_savings is not None and self.predicted_savings > 0:
            # Calculate how close the prediction was to reality
            error = abs(self.actual_savings - self.predicted_savings)
            self.savings_accuracy = max(0.0, 1.0 - (error / max(self.predicted_savings, 1.0)))
        
        # Time efficiency score
        if self.actual_time_spent is not None and self.predicted_time_saved > 0:
            # Score based on whether time was actually saved
            if self.actual_time_spent <= self.predicted_time_saved:
                self.time_efficiency_score = 1.0
            else:
                # Penalty for taking longer than predicted
                time_overrun = self.actual_time_spent - self.predicted_time_saved
                self.time_efficiency_score = max(0.0, 1.0 - (time_overrun / 60.0))  # Penalty per hour
        
        # Recommendation follow rate
        if self.recommended_stores:
            followed_recommendations = len(set(self.stores_actually_visited) & set(self.recommended_stores))
            self.recommendation_follow_rate = followed_recommendations / len(self.recommended_stores)


@dataclass
class BusinessMetric:
    """Individual business performance metric."""
    
    metric_name: str
    category: BusinessImpactCategory
    current_value: float
    target_value: float
    unit: str
    
    # Historical tracking
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Trend analysis
    trend_direction: str = "stable"  # improving, degrading, stable
    trend_strength: float = 0.0  # -1 to 1, negative = degrading
    
    # Benchmarks
    industry_benchmark: Optional[float] = None
    internal_benchmark: Optional[float] = None
    
    def add_measurement(self, value: float):
        """Add new measurement and update trend."""
        self.historical_values.append((datetime.now(), value))
        previous_value = self.current_value
        self.current_value = value
        
        # Keep only last 50 measurements
        if len(self.historical_values) > 50:
            self.historical_values = self.historical_values[-50:]
        
        # Update trend if we have enough data
        if len(self.historical_values) >= 5:
            recent_values = [v[1] for v in self.historical_values[-5:]]
            
            # Calculate trend using linear regression slope
            x = np.arange(len(recent_values))
            slope, _ = np.polyfit(x, recent_values, 1)
            
            self.trend_strength = np.clip(slope / max(abs(np.mean(recent_values)), 1.0), -1.0, 1.0)
            
            if self.trend_strength > 0.1:
                self.trend_direction = "improving"
            elif self.trend_strength < -0.1:
                self.trend_direction = "degrading"
            else:
                self.trend_direction = "stable"
    
    def get_performance_vs_target(self) -> float:
        """Get performance as percentage of target."""
        if self.target_value == 0:
            return 0.0
        return (self.current_value / self.target_value) * 100
    
    def get_status(self) -> str:
        """Get metric status based on target achievement."""
        performance = self.get_performance_vs_target()
        
        if performance >= 100:
            return "exceeding"
        elif performance >= 90:
            return "meeting"
        elif performance >= 75:
            return "approaching"
        else:
            return "underperforming"


@dataclass
class BusinessValidationReport:
    """Comprehensive business metrics validation report."""
    
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    period_start: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    period_end: datetime = field(default_factory=datetime.now)
    
    # Core metrics
    metrics: Dict[str, BusinessMetric] = field(default_factory=dict)
    
    # User satisfaction analysis
    total_sessions: int = 0
    avg_satisfaction_score: float = 0.0
    satisfaction_distribution: Dict[SatisfactionLevel, int] = field(default_factory=dict)
    
    # Financial impact
    total_predicted_savings: float = 0.0
    total_actual_savings: float = 0.0
    savings_accuracy_rate: float = 0.0
    avg_savings_per_session: float = 0.0
    
    # Time efficiency
    avg_predicted_time_saved: float = 0.0  # minutes
    avg_actual_time_spent: float = 0.0  # minutes
    time_prediction_accuracy: float = 0.0
    
    # System performance
    avg_match_confidence: float = 0.0
    store_coverage_rate: float = 0.0
    recommendation_follow_rate: float = 0.0
    
    # ROI calculation
    estimated_user_value_created: float = 0.0
    system_operational_cost: float = 0.0  # Would be calculated from infrastructure costs
    roi_percentage: float = 0.0
    
    # Insights and recommendations
    key_insights: List[str] = field(default_factory=list)
    improvement_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


class BusinessMetricsValidator:
    """Comprehensive business metrics validation system."""
    
    def __init__(self):
        self.golden_dataset = GoldenDatasetManager()
        self.user_sessions: List[UserSession] = []
        self.logger = logging.getLogger(__name__)
        
        # Load historical session data
        self._load_session_data()
        
        # Initialize core business metrics
        self.business_metrics = self._initialize_business_metrics()
    
    def _load_session_data(self):
        """Load historical user session data."""
        session_file = Path("evaluation/data/user_sessions.json")
        
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    
                for session_data in data.get('sessions', []):
                    session = UserSession(
                        session_id=session_data['session_id'],
                        timestamp=datetime.fromisoformat(session_data['timestamp']),
                        ingredients_requested=session_data['ingredients_requested'],
                        optimization_strategy=session_data.get('optimization_strategy', 'balanced'),
                        predicted_savings=session_data.get('predicted_savings', 0.0),
                        actual_savings=session_data.get('actual_savings'),
                        user_satisfaction=SatisfactionLevel(session_data['user_satisfaction']) if session_data.get('user_satisfaction') else None
                    )
                    session.calculate_derived_metrics()
                    self.user_sessions.append(session)
                    
                print(f"✅ Loaded {len(self.user_sessions)} user sessions")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load session data: {e}")
                self._create_mock_session_data()
        else:
            print("📝 No session data found, creating mock data for demonstration")
            self._create_mock_session_data()
    
    def _create_mock_session_data(self):
        """Create mock user session data for demonstration."""
        # Create diverse mock sessions with realistic data
        mock_sessions = [
            {
                'ingredients': ['milk', 'bread', 'eggs'],
                'strategy': 'cost_only',
                'predicted_savings': 8.50,
                'actual_savings': 7.25,
                'satisfaction': SatisfactionLevel.SATISFIED
            },
            {
                'ingredients': ['chicken breast', 'rice', 'vegetables'],
                'strategy': 'balanced',
                'predicted_savings': 12.30,
                'actual_savings': 14.10,
                'satisfaction': SatisfactionLevel.VERY_SATISFIED
            },
            {
                'ingredients': ['organic milk', 'whole grain bread', 'free range eggs'],
                'strategy': 'quality_first',
                'predicted_savings': 5.75,
                'actual_savings': 4.90,
                'satisfaction': SatisfactionLevel.SATISFIED
            },
            {
                'ingredients': ['ground beef', 'pasta', 'sauce'],
                'strategy': 'convenience',
                'predicted_savings': 6.20,
                'actual_savings': 2.30,
                'satisfaction': SatisfactionLevel.DISSATISFIED
            },
            {
                'ingredients': ['bananas', 'apples', 'oranges'],
                'strategy': 'balanced',
                'predicted_savings': 3.40,
                'actual_savings': 3.80,
                'satisfaction': SatisfactionLevel.SATISFIED
            }
        ]
        
        for i, mock in enumerate(mock_sessions):
            # Create multiple sessions over different time periods
            for week_offset in range(4):  # 4 weeks of data
                session = UserSession(
                    timestamp=datetime.now() - timedelta(weeks=week_offset, days=i),
                    ingredients_requested=mock['ingredients'],
                    optimization_strategy=mock['strategy'],
                    predicted_savings=mock['predicted_savings'],
                    actual_savings=mock['actual_savings'],
                    predicted_time_saved=15.0 + (i * 5),  # 15-35 minutes
                    actual_time_spent=18.0 + (i * 4),  # 18-34 minutes
                    user_satisfaction=mock['satisfaction'],
                    recommended_stores=['metro_ca', 'walmart_ca'],
                    stores_actually_visited=['metro_ca'] if week_offset % 2 == 0 else ['metro_ca', 'walmart_ca']
                )
                session.calculate_derived_metrics()
                self.user_sessions.append(session)
        
        print(f"📝 Created {len(self.user_sessions)} mock user sessions for demonstration")
    
    def _initialize_business_metrics(self) -> Dict[str, BusinessMetric]:
        """Initialize core business metrics with targets."""
        return {
            'savings_accuracy': BusinessMetric(
                metric_name="Savings Prediction Accuracy",
                category=BusinessImpactCategory.FINANCIAL,
                current_value=0.0,
                target_value=0.85,  # 85% accuracy target
                unit="accuracy_rate"
            ),
            'user_satisfaction': BusinessMetric(
                metric_name="User Satisfaction Score",
                category=BusinessImpactCategory.USER_EXPERIENCE,
                current_value=0.0,
                target_value=4.2,  # 4.2/5.0 target
                unit="satisfaction_score"
            ),
            'time_efficiency': BusinessMetric(
                metric_name="Time Efficiency Score",
                category=BusinessImpactCategory.TIME_EFFICIENCY,
                current_value=0.0,
                target_value=0.80,  # 80% of predictions should be accurate
                unit="efficiency_score"
            ),
            'recommendation_follow_rate': BusinessMetric(
                metric_name="Recommendation Follow Rate",
                category=BusinessImpactCategory.USER_EXPERIENCE,
                current_value=0.0,
                target_value=0.70,  # 70% follow rate target
                unit="follow_rate"
            ),
            'avg_savings_per_session': BusinessMetric(
                metric_name="Average Savings per Session",
                category=BusinessImpactCategory.FINANCIAL,
                current_value=0.0,
                target_value=10.0,  # $10 average savings target
                unit="dollars"
            ),
            'match_confidence': BusinessMetric(
                metric_name="Average Match Confidence",
                category=BusinessImpactCategory.ACCURACY,
                current_value=0.0,
                target_value=0.90,  # 90% confidence target
                unit="confidence_score"
            )
        }
    
    def validate_business_metrics(self, period_days: int = 30) -> BusinessValidationReport:
        """Generate comprehensive business metrics validation report."""
        self.logger.info(f"Validating business metrics for {period_days}-day period")
        
        report = BusinessValidationReport()
        report.period_start = datetime.now() - timedelta(days=period_days)
        
        # Filter sessions to the specified period
        period_sessions = [
            session for session in self.user_sessions
            if report.period_start <= session.timestamp <= report.period_end
        ]
        
        report.total_sessions = len(period_sessions)
        
        if not period_sessions:
            self.logger.warning("No sessions found for the specified period")
            return report
        
        # Calculate financial metrics
        self._calculate_financial_metrics(report, period_sessions)
        
        # Calculate time efficiency metrics
        self._calculate_time_efficiency_metrics(report, period_sessions)
        
        # Calculate user satisfaction metrics
        self._calculate_satisfaction_metrics(report, period_sessions)
        
        # Calculate system performance metrics
        self._calculate_system_performance_metrics(report, period_sessions)
        
        # Calculate ROI
        self._calculate_roi_metrics(report, period_sessions)
        
        # Update business metrics
        self._update_business_metrics(report)
        
        # Generate insights and recommendations
        self._generate_business_insights(report, period_sessions)
        
        self.logger.info(f"Business validation completed for {len(period_sessions)} sessions")
        
        return report
    
    def _calculate_financial_metrics(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Calculate financial impact metrics."""
        predicted_savings = [s.predicted_savings for s in sessions if s.predicted_savings > 0]
        actual_savings = [s.actual_savings for s in sessions if s.actual_savings is not None]
        accuracy_scores = [s.savings_accuracy for s in sessions if s.savings_accuracy is not None]
        
        if predicted_savings:
            report.total_predicted_savings = sum(predicted_savings)
            report.avg_savings_per_session = statistics.mean(predicted_savings)
        
        if actual_savings:
            report.total_actual_savings = sum(actual_savings)
        
        if accuracy_scores:
            report.savings_accuracy_rate = statistics.mean(accuracy_scores)
        
        # Calculate savings distribution
        if actual_savings:
            savings_ranges = {
                '$0-5': len([s for s in actual_savings if 0 <= s <= 5]),
                '$5-10': len([s for s in actual_savings if 5 < s <= 10]),
                '$10-20': len([s for s in actual_savings if 10 < s <= 20]),
                '$20+': len([s for s in actual_savings if s > 20])
            }
            
            # Store as insight
            report.key_insights.append(
                f"Savings distribution: {max(savings_ranges.items(), key=lambda x: x[1])[0]} range most common"
            )
    
    def _calculate_time_efficiency_metrics(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Calculate time efficiency metrics."""
        predicted_times = [s.predicted_time_saved for s in sessions if s.predicted_time_saved > 0]
        actual_times = [s.actual_time_spent for s in sessions if s.actual_time_spent is not None]
        efficiency_scores = [s.time_efficiency_score for s in sessions if s.time_efficiency_score is not None]
        
        if predicted_times:
            report.avg_predicted_time_saved = statistics.mean(predicted_times)
        
        if actual_times:
            report.avg_actual_time_spent = statistics.mean(actual_times)
        
        if efficiency_scores:
            report.time_prediction_accuracy = statistics.mean(efficiency_scores)
        
        # Analyze time vs savings correlation
        if len(actual_times) >= 5 and len([s.actual_savings for s in sessions if s.actual_savings]) >= 5:
            # Calculate correlation (simplified)
            time_savings_pairs = [
                (s.actual_time_spent, s.actual_savings) 
                for s in sessions 
                if s.actual_time_spent is not None and s.actual_savings is not None
            ]
            
            if len(time_savings_pairs) >= 3:
                correlation = np.corrcoef(
                    [pair[0] for pair in time_savings_pairs],
                    [pair[1] for pair in time_savings_pairs]
                )[0, 1]
                
                if not np.isnan(correlation):
                    if correlation > 0.3:
                        report.key_insights.append("Higher time investment correlates with higher savings")
                    elif correlation < -0.3:
                        report.key_insights.append("Users achieve savings efficiently with less time")
    
    def _calculate_satisfaction_metrics(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Calculate user satisfaction metrics."""
        satisfaction_scores = [s.user_satisfaction.value for s in sessions if s.user_satisfaction is not None]
        
        if satisfaction_scores:
            report.avg_satisfaction_score = statistics.mean(satisfaction_scores)
            
            # Calculate distribution
            for level in SatisfactionLevel:
                count = len([s for s in sessions if s.user_satisfaction == level])
                report.satisfaction_distribution[level] = count
            
            # Analyze satisfaction drivers
            high_satisfaction_sessions = [
                s for s in sessions 
                if s.user_satisfaction and s.user_satisfaction.value >= 4
            ]
            
            if high_satisfaction_sessions:
                # Common characteristics of satisfied users
                avg_savings_satisfied = statistics.mean([
                    s.actual_savings for s in high_satisfaction_sessions 
                    if s.actual_savings is not None
                ]) if any(s.actual_savings for s in high_satisfaction_sessions) else 0
                
                if avg_savings_satisfied > report.avg_savings_per_session * 1.2:
                    report.success_factors.append("Higher actual savings drive satisfaction")
                
                # Follow rate analysis
                avg_follow_rate_satisfied = statistics.mean([
                    s.recommendation_follow_rate for s in high_satisfaction_sessions
                ])
                
                if avg_follow_rate_satisfied > 0.8:
                    report.success_factors.append("Users who follow recommendations are more satisfied")
    
    def _calculate_system_performance_metrics(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Calculate system performance metrics."""
        confidence_scores = [s.match_confidence_avg for s in sessions if s.match_confidence_avg > 0]
        follow_rates = [s.recommendation_follow_rate for s in sessions]
        
        if confidence_scores:
            report.avg_match_confidence = statistics.mean(confidence_scores)
        
        if follow_rates:
            report.recommendation_follow_rate = statistics.mean(follow_rates)
        
        # Store coverage analysis
        all_stores_recommended = set()
        all_stores_visited = set()
        
        for session in sessions:
            all_stores_recommended.update(session.recommended_stores)
            all_stores_visited.update(session.stores_actually_visited)
        
        if all_stores_recommended:
            store_coverage = len(all_stores_visited & all_stores_recommended) / len(all_stores_recommended)
            report.store_coverage_rate = store_coverage
        
        # Analyze recommendation effectiveness
        strategy_performance = {}
        for session in sessions:
            strategy = session.optimization_strategy
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            
            if session.user_satisfaction:
                strategy_performance[strategy].append(session.user_satisfaction.value)
        
        if strategy_performance:
            best_strategy = max(
                strategy_performance.items(),
                key=lambda x: statistics.mean(x[1])
            )[0]
            report.success_factors.append(f"'{best_strategy}' strategy shows highest satisfaction")
    
    def _calculate_roi_metrics(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Calculate return on investment metrics."""
        # Estimate value created for users
        total_actual_savings = sum([
            s.actual_savings for s in sessions 
            if s.actual_savings is not None
        ])
        
        # Estimate time value (assuming $20/hour value of user time)
        time_value_per_hour = 20.0
        total_time_saved_hours = sum([
            max(0, s.predicted_time_saved - (s.actual_time_spent or s.predicted_time_saved)) / 60.0
            for s in sessions
            if s.predicted_time_saved > 0
        ])
        
        time_value_created = total_time_saved_hours * time_value_per_hour
        
        report.estimated_user_value_created = total_actual_savings + time_value_created
        
        # Mock operational cost (in production, this would be calculated from actual infrastructure costs)
        monthly_operational_cost = 500.0  # $500/month for infrastructure, APIs, etc.
        days_in_period = (report.period_end - report.period_start).days
        report.system_operational_cost = (monthly_operational_cost / 30) * days_in_period
        
        # Calculate ROI
        if report.system_operational_cost > 0:
            report.roi_percentage = (
                (report.estimated_user_value_created - report.system_operational_cost) 
                / report.system_operational_cost
            ) * 100
        
        # ROI insights
        if report.roi_percentage > 200:
            report.success_factors.append("System generates strong positive ROI")
        elif report.roi_percentage < 50:
            report.risk_factors.append("ROI below target threshold")
    
    def _update_business_metrics(self, report: BusinessValidationReport):
        """Update business metrics with current values."""
        # Update each metric
        if 'savings_accuracy' in self.business_metrics:
            self.business_metrics['savings_accuracy'].add_measurement(report.savings_accuracy_rate)
        
        if 'user_satisfaction' in self.business_metrics:
            self.business_metrics['user_satisfaction'].add_measurement(report.avg_satisfaction_score)
        
        if 'time_efficiency' in self.business_metrics:
            self.business_metrics['time_efficiency'].add_measurement(report.time_prediction_accuracy)
        
        if 'recommendation_follow_rate' in self.business_metrics:
            self.business_metrics['recommendation_follow_rate'].add_measurement(report.recommendation_follow_rate)
        
        if 'avg_savings_per_session' in self.business_metrics:
            self.business_metrics['avg_savings_per_session'].add_measurement(report.avg_savings_per_session)
        
        if 'match_confidence' in self.business_metrics:
            self.business_metrics['match_confidence'].add_measurement(report.avg_match_confidence)
        
        # Store metrics in report
        report.metrics = self.business_metrics.copy()
    
    def _generate_business_insights(self, report: BusinessValidationReport, sessions: List[UserSession]):
        """Generate actionable business insights and recommendations."""
        
        # Performance insights
        if report.savings_accuracy_rate < 0.75:
            report.risk_factors.append("Savings prediction accuracy below acceptable threshold")
            report.improvement_recommendations.append({
                'priority': 'high',
                'category': 'accuracy',
                'title': 'Improve Savings Prediction Model',
                'description': 'Savings predictions are consistently off by more than 25%',
                'actions': [
                    'Collect more real-world pricing data',
                    'Improve price prediction algorithms',
                    'Account for sale cycles and seasonal pricing',
                    'Implement dynamic pricing updates'
                ]
            })
        
        if report.avg_satisfaction_score < 3.5:
            report.risk_factors.append("User satisfaction below target")
            report.improvement_recommendations.append({
                'priority': 'high',
                'category': 'user_experience',
                'title': 'Improve User Satisfaction',
                'description': 'Average satisfaction score is below acceptable threshold',
                'actions': [
                    'Conduct user experience research',
                    'Improve recommendation quality',
                    'Enhance user interface and workflow',
                    'Implement user feedback collection system'
                ]
            })
        
        if report.recommendation_follow_rate < 0.5:
            report.risk_factors.append("Low recommendation adoption rate")
            report.improvement_recommendations.append({
                'priority': 'medium',
                'category': 'recommendations',
                'title': 'Increase Recommendation Adoption',
                'description': 'Users are not following system recommendations',
                'actions': [
                    'Improve explanation of recommendations',
                    'Make recommendations more convenient to follow',
                    'Provide clearer value propositions',
                    'Personalize recommendations based on user history'
                ]
            })
        
        # Success insights
        if report.roi_percentage > 150:
            report.key_insights.append(f"System generates strong ROI of {report.roi_percentage:.1f}%")
        
        if report.avg_satisfaction_score > 4.0:
            report.key_insights.append("High user satisfaction indicates strong product-market fit")
        
        # Trend insights
        for metric_name, metric in self.business_metrics.items():
            if metric.trend_direction == "degrading" and metric.trend_strength < -0.2:
                report.risk_factors.append(f"{metric.metric_name} showing concerning downward trend")
            elif metric.trend_direction == "improving" and metric.trend_strength > 0.2:
                report.success_factors.append(f"{metric.metric_name} showing positive improvement trend")
        
        # Operational insights
        if report.total_sessions < 10:
            report.improvement_recommendations.append({
                'priority': 'medium',
                'category': 'adoption',
                'title': 'Increase User Adoption',
                'description': 'Low number of active user sessions',
                'actions': [
                    'Improve user onboarding experience',
                    'Implement referral programs',
                    'Enhance marketing and user acquisition',
                    'Add more compelling use cases'
                ]
            })
    
    def generate_business_dashboard_data(self, report: BusinessValidationReport) -> Dict[str, Any]:
        """Generate data for business metrics dashboard visualization."""
        
        dashboard_data = {
            'summary': {
                'total_sessions': report.total_sessions,
                'avg_satisfaction': report.avg_satisfaction_score,
                'total_savings': report.total_actual_savings,
                'roi_percentage': report.roi_percentage,
                'key_insights_count': len(report.key_insights),
                'risk_factors_count': len(report.risk_factors)
            },
            'financial_metrics': {
                'predicted_vs_actual_savings': {
                    'predicted': report.total_predicted_savings,
                    'actual': report.total_actual_savings,
                    'accuracy': report.savings_accuracy_rate
                },
                'savings_per_session': report.avg_savings_per_session,
                'roi': report.roi_percentage
            },
            'satisfaction_metrics': {
                'average_score': report.avg_satisfaction_score,
                'distribution': {level.name: count for level, count in report.satisfaction_distribution.items()},
                'target_achievement': (report.avg_satisfaction_score / 4.2) * 100  # Target is 4.2
            },
            'performance_metrics': {
                'recommendation_follow_rate': report.recommendation_follow_rate,
                'match_confidence': report.avg_match_confidence,
                'store_coverage': report.store_coverage_rate
            },
            'trends': {
                metric_name: {
                    'current_value': metric.current_value,
                    'target_value': metric.target_value,
                    'trend_direction': metric.trend_direction,
                    'performance_vs_target': metric.get_performance_vs_target(),
                    'status': metric.get_status()
                }
                for metric_name, metric in report.metrics.items()
            },
            'insights': {
                'success_factors': report.success_factors,
                'risk_factors': report.risk_factors,
                'key_insights': report.key_insights,
                'recommendations': report.improvement_recommendations
            }
        }
        
        return dashboard_data
    
    def save_validation_report(self, report: BusinessValidationReport, output_file: str = None):
        """Save business validation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation/results/business_validation_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to JSON-serializable format
        report_data = {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'summary': {
                'total_sessions': report.total_sessions,
                'avg_satisfaction_score': report.avg_satisfaction_score,
                'total_predicted_savings': report.total_predicted_savings,
                'total_actual_savings': report.total_actual_savings,
                'savings_accuracy_rate': report.savings_accuracy_rate,
                'roi_percentage': report.roi_percentage
            },
            'metrics': {
                name: {
                    'metric_name': metric.metric_name,
                    'category': metric.category.value,
                    'current_value': metric.current_value,
                    'target_value': metric.target_value,
                    'performance_vs_target': metric.get_performance_vs_target(),
                    'status': metric.get_status(),
                    'trend_direction': metric.trend_direction,
                    'trend_strength': metric.trend_strength,
                    'unit': metric.unit
                }
                for name, metric in report.metrics.items()
            },
            'insights': {
                'key_insights': report.key_insights,
                'success_factors': report.success_factors,
                'risk_factors': report.risk_factors,
                'improvement_recommendations': report.improvement_recommendations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✅ Business validation report saved to {output_path}")
        
        # Also generate dashboard data file
        dashboard_data = self.generate_business_dashboard_data(report)
        dashboard_file = output_path.parent / f"business_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"✅ Business dashboard data saved to {dashboard_file}")
    
    def add_user_session(self, session: UserSession):
        """Add a new user session for tracking."""
        session.calculate_derived_metrics()
        self.user_sessions.append(session)
        
        # Save updated session data
        self._save_session_data()
    
    def _save_session_data(self):
        """Save user session data to file."""
        session_file = Path("evaluation/data/user_sessions.json")
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sessions to JSON-serializable format
        sessions_data = {
            'metadata': {
                'total_sessions': len(self.user_sessions),
                'last_updated': datetime.now().isoformat()
            },
            'sessions': [
                {
                    'session_id': session.session_id,
                    'timestamp': session.timestamp.isoformat(),
                    'ingredients_requested': session.ingredients_requested,
                    'optimization_strategy': session.optimization_strategy,
                    'predicted_savings': session.predicted_savings,
                    'actual_savings': session.actual_savings,
                    'user_satisfaction': session.user_satisfaction.value if session.user_satisfaction else None
                }
                for session in self.user_sessions[-100:]  # Keep last 100 sessions
            ]
        }
        
        with open(session_file, 'w') as f:
            json.dump(sessions_data, f, indent=2, default=str)


if __name__ == "__main__":
    def test_business_validation():
        validator = BusinessMetricsValidator()
        report = validator.validate_business_metrics(period_days=30)
        
        print(f"\n💼 Business Metrics Validation Results:")
        print(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
        print(f"Total Sessions: {report.total_sessions}")
        print(f"Average Satisfaction: {report.avg_satisfaction_score:.2f}/5.0")
        print(f"Total Actual Savings: ${report.total_actual_savings:.2f}")
        print(f"Savings Accuracy: {report.savings_accuracy_rate:.1%}")
        print(f"ROI: {report.roi_percentage:.1f}%")
        
        print(f"\n📈 Key Insights ({len(report.key_insights)}):")
        for insight in report.key_insights:
            print(f"  • {insight}")
        
        print(f"\n✅ Success Factors ({len(report.success_factors)}):")
        for factor in report.success_factors:
            print(f"  • {factor}")
        
        print(f"\n⚠️ Risk Factors ({len(report.risk_factors)}):")
        for risk in report.risk_factors:
            print(f"  • {risk}")
        
        print(f"\n💡 High Priority Recommendations:")
        high_priority = [rec for rec in report.improvement_recommendations if rec['priority'] == 'high']
        for rec in high_priority:
            print(f"  • {rec['title']}: {rec['description']}")
        
        # Save report
        validator.save_validation_report(report)
    
    test_business_validation()