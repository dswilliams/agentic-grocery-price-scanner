"""
Advanced analytics and optimization for collection methods.
Tracks performance, learns patterns, and optimizes scraping strategies.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import os

from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics to track."""
    
    SUCCESS_RATE = "success_rate"
    RESPONSE_TIME = "response_time"
    PRODUCTS_PER_MINUTE = "products_per_minute"
    ERROR_RATE = "error_rate"
    CONFIDENCE_SCORE = "confidence_score"
    USER_INTERVENTION_RATE = "user_intervention_rate"


@dataclass
class CollectionSession:
    """Data class for tracking individual collection sessions."""
    
    session_id: str
    query: str
    store_id: str
    collection_method: DataCollectionMethod
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    products_collected: int
    error_messages: List[str]
    user_interventions: int
    confidence_scores: List[float]
    metadata: Dict[str, Any]
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def products_per_minute(self) -> Optional[float]:
        """Calculate products collected per minute."""
        duration = self.duration_seconds
        if duration and duration > 0:
            return (self.products_collected / duration) * 60
        return None
    
    @property
    def average_confidence(self) -> Optional[float]:
        """Calculate average confidence score."""
        if self.confidence_scores:
            return statistics.mean(self.confidence_scores)
        return None


@dataclass
class MethodPerformanceStats:
    """Performance statistics for a collection method."""
    
    method: DataCollectionMethod
    total_sessions: int
    successful_sessions: int
    total_products: int
    total_duration: float
    average_response_time: float
    error_count: int
    user_interventions: int
    confidence_scores: List[float]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_sessions == 0:
            return 0.0
        return (self.successful_sessions / self.total_sessions) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_sessions == 0:
            return 0.0
        return (self.error_count / self.total_sessions) * 100
    
    @property
    def average_products_per_session(self) -> float:
        """Calculate average products per session."""
        if self.successful_sessions == 0:
            return 0.0
        return self.total_products / self.successful_sessions
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return statistics.mean(self.confidence_scores)
    
    @property
    def products_per_minute(self) -> float:
        """Calculate products per minute."""
        if self.total_duration == 0:
            return 0.0
        return (self.total_products / self.total_duration) * 60


class CollectionAnalytics:
    """Advanced analytics for collection method performance."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize analytics tracker."""
        self.storage_path = storage_path or "logs/collection_analytics.pkl"
        self.sessions: List[CollectionSession] = []
        self.method_preferences: Dict[str, Dict[str, float]] = {}  # query_pattern -> method -> score
        self.optimization_rules: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_analytics_data()
    
    def record_session(
        self,
        session_id: str,
        query: str,
        store_id: str,
        collection_method: DataCollectionMethod,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        success: bool = True,
        products: Optional[List[Product]] = None,
        error_messages: Optional[List[str]] = None,
        user_interventions: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a collection session for analytics."""
        
        products = products or []
        error_messages = error_messages or []
        metadata = metadata or {}
        
        session = CollectionSession(
            session_id=session_id,
            query=query,
            store_id=store_id,
            collection_method=collection_method,
            start_time=start_time,
            end_time=end_time or datetime.now(),
            success=success,
            products_collected=len(products),
            error_messages=error_messages,
            user_interventions=user_interventions,
            confidence_scores=[p.confidence_score for p in products],
            metadata=metadata
        )
        
        self.sessions.append(session)
        self._update_method_preferences(query, collection_method, success, len(products))
        self._save_analytics_data()
        
        logger.info(f"Recorded analytics session: {session_id}")
    
    def get_method_performance(
        self,
        method: Optional[DataCollectionMethod] = None,
        days_back: int = 30
    ) -> Dict[DataCollectionMethod, MethodPerformanceStats]:
        """Get performance statistics for collection methods."""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = [s for s in self.sessions if s.start_time >= cutoff_date]
        
        if method:
            recent_sessions = [s for s in recent_sessions if s.collection_method == method]
        
        # Group sessions by method
        method_sessions = {}
        for session in recent_sessions:
            method = session.collection_method
            if method not in method_sessions:
                method_sessions[method] = []
            method_sessions[method].append(session)
        
        # Calculate statistics for each method
        performance_stats = {}
        for method, sessions in method_sessions.items():
            successful_sessions = [s for s in sessions if s.success]
            
            total_duration = sum(s.duration_seconds or 0 for s in sessions)
            all_confidence_scores = []
            for session in sessions:
                all_confidence_scores.extend(session.confidence_scores)
            
            stats = MethodPerformanceStats(
                method=method,
                total_sessions=len(sessions),
                successful_sessions=len(successful_sessions),
                total_products=sum(s.products_collected for s in sessions),
                total_duration=total_duration,
                average_response_time=total_duration / max(len(sessions), 1),
                error_count=sum(len(s.error_messages) for s in sessions),
                user_interventions=sum(s.user_interventions for s in sessions),
                confidence_scores=all_confidence_scores
            )
            
            performance_stats[method] = stats
        
        return performance_stats
    
    def get_optimization_recommendations(self, query: str, store_id: str) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on historical data."""
        recommendations = []
        
        # Analyze historical performance for similar queries
        similar_sessions = self._find_similar_sessions(query, store_id)
        
        if similar_sessions:
            method_performance = {}
            for session in similar_sessions:
                method = session.collection_method
                if method not in method_performance:
                    method_performance[method] = {"successes": 0, "total": 0, "avg_products": 0}
                
                method_performance[method]["total"] += 1
                if session.success:
                    method_performance[method]["successes"] += 1
                    method_performance[method]["avg_products"] += session.products_collected
            
            # Calculate success rates and recommend best method
            best_method = None
            best_score = 0
            
            for method, perf in method_performance.items():
                success_rate = perf["successes"] / perf["total"] if perf["total"] > 0 else 0
                avg_products = perf["avg_products"] / max(perf["successes"], 1)
                
                # Combined score: success rate + product yield
                score = success_rate * 0.7 + (avg_products / 10) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_method = method
            
            if best_method:
                recommendations.append({
                    "type": "method_preference",
                    "recommendation": f"Start with {best_method.value} for this query type",
                    "confidence": best_score,
                    "reason": f"Historical success rate: {(best_score * 100):.1f}%"
                })
        
        # Time-based recommendations
        time_recommendations = self._get_time_based_recommendations(store_id)
        recommendations.extend(time_recommendations)
        
        # Query pattern recommendations
        pattern_recommendations = self._get_query_pattern_recommendations(query)
        recommendations.extend(pattern_recommendations)
        
        return recommendations
    
    def generate_performance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        performance_stats = self.get_method_performance(days_back=days_back)
        
        report = {
            "report_period": f"Last {days_back} days",
            "generation_time": datetime.now().isoformat(),
            "summary": {
                "total_sessions": sum(len(self.sessions) for _ in performance_stats),
                "total_products": sum(stats.total_products for stats in performance_stats.values()),
                "overall_success_rate": 0,
                "most_effective_method": None,
                "least_effective_method": None
            },
            "method_performance": {},
            "trends": {},
            "recommendations": []
        }
        
        # Method performance details
        method_scores = {}
        for method, stats in performance_stats.items():
            method_data = {
                "success_rate": stats.success_rate,
                "products_per_minute": stats.products_per_minute,
                "average_confidence": stats.average_confidence,
                "error_rate": stats.error_rate,
                "user_intervention_rate": (stats.user_interventions / max(stats.total_sessions, 1)) * 100
            }
            
            report["method_performance"][method.value] = method_data
            
            # Calculate overall effectiveness score
            effectiveness_score = (
                stats.success_rate * 0.4 +
                min(stats.products_per_minute * 10, 100) * 0.3 +
                stats.average_confidence * 100 * 0.2 +
                (100 - stats.error_rate) * 0.1
            )
            method_scores[method] = effectiveness_score
        
        # Identify best and worst methods
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            worst_method = min(method_scores, key=method_scores.get)
            
            report["summary"]["most_effective_method"] = best_method.value
            report["summary"]["least_effective_method"] = worst_method.value
            
            overall_success_rate = sum(
                stats.success_rate * stats.total_sessions 
                for stats in performance_stats.values()
            ) / sum(stats.total_sessions for stats in performance_stats.values())
            report["summary"]["overall_success_rate"] = overall_success_rate
        
        # Trends analysis
        report["trends"] = self._analyze_trends(days_back)
        
        # Generate recommendations
        report["recommendations"] = self._generate_global_recommendations(performance_stats)
        
        return report
    
    def predict_optimal_strategy(
        self,
        query: str,
        stores: List[str],
        time_constraint: Optional[int] = None
    ) -> Dict[str, Any]:
        """Predict optimal scraping strategy based on analytics."""
        
        strategy = {
            "recommended_order": [],
            "estimated_success_probability": {},
            "estimated_duration": {},
            "confidence_intervals": {},
            "risk_assessment": {}
        }
        
        for store_id in stores:
            similar_sessions = self._find_similar_sessions(query, store_id)
            
            method_predictions = {}
            for method in DataCollectionMethod:
                method_sessions = [s for s in similar_sessions if s.collection_method == method]
                
                if method_sessions:
                    success_rate = sum(1 for s in method_sessions if s.success) / len(method_sessions)
                    avg_duration = statistics.mean(s.duration_seconds or 0 for s in method_sessions)
                    avg_products = statistics.mean(s.products_collected for s in method_sessions)
                    
                    method_predictions[method] = {
                        "success_probability": success_rate,
                        "estimated_duration": avg_duration,
                        "estimated_products": avg_products,
                        "confidence": min(len(method_sessions) / 5, 1.0)  # More sessions = higher confidence
                    }
            
            # Sort methods by predicted effectiveness
            sorted_methods = sorted(
                method_predictions.items(),
                key=lambda x: x[1]["success_probability"] * x[1]["estimated_products"],
                reverse=True
            )
            
            strategy["recommended_order"].append({
                "store_id": store_id,
                "method_order": [method.value for method, _ in sorted_methods],
                "predictions": {method.value: pred for method, pred in method_predictions.items()}
            })
        
        return strategy
    
    def _find_similar_sessions(self, query: str, store_id: str, days_back: int = 90) -> List[CollectionSession]:
        """Find sessions similar to the given query and store."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        similar_sessions = []
        query_words = set(query.lower().split())
        
        for session in self.sessions:
            if session.start_time < cutoff_date:
                continue
            
            # Match store exactly
            if session.store_id != store_id:
                continue
            
            # Calculate query similarity
            session_words = set(session.query.lower().split())
            similarity = len(query_words & session_words) / len(query_words | session_words)
            
            if similarity > 0.3:  # At least 30% word overlap
                similar_sessions.append(session)
        
        return similar_sessions
    
    def _update_method_preferences(
        self,
        query: str,
        method: DataCollectionMethod,
        success: bool,
        products_count: int
    ) -> None:
        """Update method preferences based on session results."""
        query_pattern = self._extract_query_pattern(query)
        
        if query_pattern not in self.method_preferences:
            self.method_preferences[query_pattern] = {}
        
        if method.value not in self.method_preferences[query_pattern]:
            self.method_preferences[query_pattern][method.value] = 0.5  # Neutral start
        
        # Update preference based on success and productivity
        score_delta = 0.1 if success else -0.1
        score_delta += (products_count / 20) * 0.05  # Bonus for productivity
        
        current_score = self.method_preferences[query_pattern][method.value]
        new_score = max(0.0, min(1.0, current_score + score_delta))
        self.method_preferences[query_pattern][method.value] = new_score
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract pattern from query for preference learning."""
        # Simple categorization - could be enhanced with NLP
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["milk", "dairy", "cheese", "yogurt"]):
            return "dairy_products"
        elif any(word in query_lower for word in ["bread", "bakery", "muffin", "cake"]):
            return "bakery_products"
        elif any(word in query_lower for word in ["meat", "chicken", "beef", "pork"]):
            return "meat_products"
        elif any(word in query_lower for word in ["fruit", "apple", "banana", "orange"]):
            return "fruits"
        elif any(word in query_lower for word in ["vegetable", "carrot", "lettuce", "tomato"]):
            return "vegetables"
        else:
            return "general_products"
    
    def _get_time_based_recommendations(self, store_id: str) -> List[Dict[str, Any]]:
        """Get recommendations based on time patterns."""
        recommendations = []
        
        # Analyze performance by hour of day
        current_hour = datetime.now().hour
        
        # This would analyze historical data by time - placeholder logic
        if 9 <= current_hour <= 11:
            recommendations.append({
                "type": "timing",
                "recommendation": "Good time for automated scraping (low server load)",
                "confidence": 0.8,
                "reason": "Historical data shows better success rates during morning hours"
            })
        elif 12 <= current_hour <= 14:
            recommendations.append({
                "type": "timing",
                "recommendation": "Consider human assistance (peak traffic)",
                "confidence": 0.7,
                "reason": "Server load typically higher during lunch hours"
            })
        
        return recommendations
    
    def _get_query_pattern_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """Get recommendations based on query patterns."""
        recommendations = []
        pattern = self._extract_query_pattern(query)
        
        if pattern in self.method_preferences:
            best_method = max(
                self.method_preferences[pattern].items(),
                key=lambda x: x[1]
            )
            
            if best_method[1] > 0.7:  # High confidence
                recommendations.append({
                    "type": "query_pattern",
                    "recommendation": f"Use {best_method[0]} for {pattern} queries",
                    "confidence": best_method[1],
                    "reason": f"Learned preference for {pattern} category"
                })
        
        return recommendations
    
    def _analyze_trends(self, days_back: int) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = [s for s in self.sessions if s.start_time >= cutoff_date]
        
        # Group by week
        weekly_stats = {}
        for session in recent_sessions:
            week = session.start_time.strftime("%Y-W%U")
            if week not in weekly_stats:
                weekly_stats[week] = {"total": 0, "successful": 0, "products": 0}
            
            weekly_stats[week]["total"] += 1
            if session.success:
                weekly_stats[week]["successful"] += 1
                weekly_stats[week]["products"] += session.products_collected
        
        # Calculate trends
        weeks = sorted(weekly_stats.keys())
        if len(weeks) >= 2:
            success_rates = [weekly_stats[w]["successful"] / max(weekly_stats[w]["total"], 1) for w in weeks]
            product_rates = [weekly_stats[w]["products"] / max(weekly_stats[w]["successful"], 1) for w in weeks]
            
            success_trend = "improving" if success_rates[-1] > success_rates[0] else "declining"
            product_trend = "improving" if product_rates[-1] > product_rates[0] else "declining"
        else:
            success_trend = "insufficient_data"
            product_trend = "insufficient_data"
        
        return {
            "success_rate_trend": success_trend,
            "productivity_trend": product_trend,
            "weekly_data": weekly_stats
        }
    
    def _generate_global_recommendations(
        self,
        performance_stats: Dict[DataCollectionMethod, MethodPerformanceStats]
    ) -> List[Dict[str, Any]]:
        """Generate global optimization recommendations."""
        recommendations = []
        
        if not performance_stats:
            return recommendations
        
        # Find methods that need improvement
        for method, stats in performance_stats.items():
            if stats.success_rate < 70:
                recommendations.append({
                    "type": "improvement",
                    "method": method.value,
                    "recommendation": f"Investigate {method.value} configuration - low success rate",
                    "priority": "high",
                    "current_rate": stats.success_rate
                })
            
            if stats.user_interventions > stats.total_sessions * 0.5:
                recommendations.append({
                    "type": "automation",
                    "method": method.value,
                    "recommendation": f"Improve {method.value} automation - high user intervention",
                    "priority": "medium",
                    "intervention_rate": (stats.user_interventions / stats.total_sessions) * 100
                })
        
        return recommendations
    
    def _save_analytics_data(self) -> None:
        """Save analytics data to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                "sessions": [asdict(session) for session in self.sessions],
                "method_preferences": self.method_preferences,
                "optimization_rules": self.optimization_rules
            }
            
            with open(self.storage_path, "wb") as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save analytics data: {e}")
    
    def _load_analytics_data(self) -> None:
        """Load analytics data from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                
                # Convert session dictionaries back to objects
                self.sessions = []
                for session_data in data.get("sessions", []):
                    # Convert datetime strings back to datetime objects
                    session_data["start_time"] = datetime.fromisoformat(session_data["start_time"])
                    if session_data["end_time"]:
                        session_data["end_time"] = datetime.fromisoformat(session_data["end_time"])
                    
                    # Convert method string back to enum
                    session_data["collection_method"] = DataCollectionMethod(
                        session_data["collection_method"]
                    )
                    
                    self.sessions.append(CollectionSession(**session_data))
                
                self.method_preferences = data.get("method_preferences", {})
                self.optimization_rules = data.get("optimization_rules", [])
                
                logger.info(f"Loaded {len(self.sessions)} analytics sessions")
                
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")
            # Initialize empty if load fails
            self.sessions = []
            self.method_preferences = {}
            self.optimization_rules = []


def export_analytics_report(analytics: CollectionAnalytics, output_path: str) -> None:
    """Export analytics report to JSON file."""
    report = analytics.generate_performance_report()
    
    try:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analytics report exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export analytics report: {e}")


def create_analytics_dashboard_data(analytics: CollectionAnalytics) -> Dict[str, Any]:
    """Create data structure for analytics dashboard."""
    performance_stats = analytics.get_method_performance()
    
    dashboard_data = {
        "overview": {
            "total_sessions": len(analytics.sessions),
            "success_rate": 0,
            "total_products": 0,
            "last_updated": datetime.now().isoformat()
        },
        "method_comparison": {},
        "trends": analytics._analyze_trends(30),
        "recent_sessions": []
    }
    
    # Calculate overview metrics
    if analytics.sessions:
        successful_sessions = sum(1 for s in analytics.sessions if s.success)
        dashboard_data["overview"]["success_rate"] = (successful_sessions / len(analytics.sessions)) * 100
        dashboard_data["overview"]["total_products"] = sum(s.products_collected for s in analytics.sessions)
    
    # Method comparison data
    for method, stats in performance_stats.items():
        dashboard_data["method_comparison"][method.value] = {
            "success_rate": stats.success_rate,
            "products_per_minute": stats.products_per_minute,
            "average_confidence": stats.average_confidence,
            "total_sessions": stats.total_sessions
        }
    
    # Recent sessions (last 10)
    recent_sessions = sorted(analytics.sessions, key=lambda s: s.start_time, reverse=True)[:10]
    for session in recent_sessions:
        dashboard_data["recent_sessions"].append({
            "session_id": session.session_id,
            "query": session.query,
            "store_id": session.store_id,
            "method": session.collection_method.value,
            "success": session.success,
            "products_collected": session.products_collected,
            "duration": session.duration_seconds,
            "start_time": session.start_time.isoformat()
        })
    
    return dashboard_data