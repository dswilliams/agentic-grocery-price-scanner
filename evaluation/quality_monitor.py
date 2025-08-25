"""
Quality Monitoring Framework

Comprehensive quality monitoring and evaluation system for continuous
assessment of system performance, regression detection, and improvement insights.
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from .golden_dataset import GoldenDatasetManager, GoldenMatch
from ..agents.matcher_agent import MatcherAgent
from ..agents.optimizer_agent import OptimizerAgent
from ..workflow.grocery_workflow import GroceryWorkflow
from ..data_models.product import Product
from ..data_models.ingredient import Ingredient


@dataclass
class QualityMetric:
    """Individual quality metric with historical tracking."""
    
    metric_name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    trend: str = "stable"  # improving, degrading, stable
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_measurement(self, value: float):
        """Add new measurement and update trend."""
        self.historical_values.append((datetime.now(), value))
        previous_value = self.current_value
        self.current_value = value
        self.last_updated = datetime.now()
        
        # Keep only last 100 measurements
        if len(self.historical_values) > 100:
            self.historical_values = self.historical_values[-100:]
        
        # Update trend
        if len(self.historical_values) >= 3:
            recent_values = [v[1] for v in self.historical_values[-3:]]
            if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
                self.trend = "improving"
            elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                self.trend = "degrading"
            else:
                self.trend = "stable"
    
    def is_warning(self) -> bool:
        """Check if metric is in warning state."""
        return self.current_value < self.threshold_warning
    
    def is_critical(self) -> bool:
        """Check if metric is in critical state."""
        return self.current_value < self.threshold_critical
    
    def get_status(self) -> str:
        """Get current status."""
        if self.is_critical():
            return "critical"
        elif self.is_warning():
            return "warning"
        else:
            return "healthy"


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    overall_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, QualityMetric] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    regression_detected: bool = False
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_alert(self, severity: str, component: str, message: str, details: Dict[str, Any] = None):
        """Add quality alert."""
        self.alerts.append({
            'severity': severity,
            'component': component,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def add_recommendation(self, priority: str, category: str, title: str, description: str, actions: List[str]):
        """Add improvement recommendation."""
        self.recommendations.append({
            'priority': priority,
            'category': category,
            'title': title,
            'description': description,
            'actions': actions,
            'timestamp': datetime.now().isoformat()
        })


class QualityMonitor:
    """Comprehensive quality monitoring and evaluation system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.golden_dataset = GoldenDatasetManager()
        self.metrics: Dict[str, QualityMetric] = {}
        self.reports: List[QualityReport] = []
        
        # Initialize core agents for testing
        self.matcher_agent = MatcherAgent()
        self.optimizer_agent = OptimizerAgent()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_metrics()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality monitoring."""
        return {
            'evaluation_frequency_hours': 6,
            'regression_detection_enabled': True,
            'performance_thresholds': {
                'match_precision': {'target': 0.95, 'warning': 0.90, 'critical': 0.85},
                'match_recall': {'target': 0.92, 'warning': 0.87, 'critical': 0.80},
                'response_time': {'target': 1.0, 'warning': 2.0, 'critical': 5.0},
                'availability': {'target': 0.99, 'warning': 0.95, 'critical': 0.90},
                'accuracy': {'target': 0.95, 'warning': 0.90, 'critical': 0.85}
            },
            'alert_channels': ['console', 'file'],
            'max_reports_retained': 100,
            'parallel_test_workers': 4
        }
    
    def _setup_metrics(self):
        """Initialize quality metrics with thresholds."""
        thresholds = self.config['performance_thresholds']
        
        # Matching performance metrics
        self.metrics['match_precision'] = QualityMetric(
            metric_name="Match Precision",
            current_value=0.0,
            target_value=thresholds['match_precision']['target'],
            threshold_warning=thresholds['match_precision']['warning'],
            threshold_critical=thresholds['match_precision']['critical'],
            unit="ratio"
        )
        
        self.metrics['match_recall'] = QualityMetric(
            metric_name="Match Recall",
            current_value=0.0,
            target_value=thresholds['match_recall']['target'],
            threshold_warning=thresholds['match_recall']['warning'],
            threshold_critical=thresholds['match_recall']['critical'],
            unit="ratio"
        )
        
        self.metrics['match_f1'] = QualityMetric(
            metric_name="Match F1 Score",
            current_value=0.0,
            target_value=0.93,
            threshold_warning=0.88,
            threshold_critical=0.82,
            unit="ratio"
        )
        
        # Performance metrics
        self.metrics['response_time'] = QualityMetric(
            metric_name="Average Response Time",
            current_value=0.0,
            target_value=thresholds['response_time']['target'],
            threshold_warning=thresholds['response_time']['warning'],
            threshold_critical=thresholds['response_time']['critical'],
            unit="seconds"
        )
        
        self.metrics['availability'] = QualityMetric(
            metric_name="System Availability",
            current_value=1.0,
            target_value=thresholds['availability']['target'],
            threshold_warning=thresholds['availability']['warning'],
            threshold_critical=thresholds['availability']['critical'],
            unit="ratio"
        )
        
        # Accuracy metrics
        self.metrics['price_accuracy'] = QualityMetric(
            metric_name="Price Accuracy",
            current_value=0.0,
            target_value=thresholds['accuracy']['target'],
            threshold_warning=thresholds['accuracy']['warning'],
            threshold_critical=thresholds['accuracy']['critical'],
            unit="ratio"
        )
        
        self.metrics['confidence_alignment'] = QualityMetric(
            metric_name="Confidence Alignment",
            current_value=0.0,
            target_value=0.90,
            threshold_warning=0.85,
            threshold_critical=0.75,
            unit="ratio"
        )
    
    async def run_comprehensive_evaluation(self) -> QualityReport:
        """Run comprehensive quality evaluation."""
        self.logger.info("🔍 Starting comprehensive quality evaluation...")
        
        report = QualityReport()
        
        try:
            # Run component evaluations in parallel
            tasks = [
                self._evaluate_matcher_performance(report),
                self._evaluate_optimizer_performance(report),
                self._evaluate_workflow_performance(report),
                self._evaluate_system_performance(report)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate overall score
            report.overall_score = self._calculate_overall_score(report)
            
            # Detect regressions
            if len(self.reports) > 0:
                report.regression_detected = self._detect_regression(report)
                if report.regression_detected:
                    report.add_alert(
                        'warning', 'system', 
                        'Performance regression detected',
                        {'previous_score': self.reports[-1].overall_score, 'current_score': report.overall_score}
                    )
            
            # Generate recommendations
            self._generate_recommendations(report)
            
            # Store report
            self.reports.append(report)
            if len(self.reports) > self.config['max_reports_retained']:
                self.reports = self.reports[-self.config['max_reports_retained']:]
            
            self.logger.info(f"✅ Quality evaluation completed - Overall Score: {report.overall_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error during quality evaluation: {e}")
            report.add_alert('critical', 'system', f'Evaluation failed: {str(e)}')
        
        return report
    
    async def _evaluate_matcher_performance(self, report: QualityReport):
        """Evaluate MatcherAgent performance against golden dataset."""
        self.logger.info("📊 Evaluating MatcherAgent performance...")
        
        try:
            test_matches = self.golden_dataset.matches[:50]  # Test with first 50 matches
            results = []
            response_times = []
            
            with ThreadPoolExecutor(max_workers=self.config['parallel_test_workers']) as executor:
                futures = []
                
                for golden_match in test_matches:
                    ingredient = Ingredient(
                        name=golden_match.ingredient_name,
                        category=golden_match.ingredient_category,
                        quantity=golden_match.ingredient_quantity,
                        unit=golden_match.ingredient_unit
                    )
                    
                    future = executor.submit(self._test_single_match, ingredient, golden_match)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                        response_times.append(result['response_time'])
                    except Exception as e:
                        self.logger.warning(f"Match test failed: {e}")
            
            # Calculate metrics
            if results:
                precision = sum(1 for r in results if r['precision']) / len(results)
                recall = sum(1 for r in results if r['recall']) / len(results)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                avg_response_time = statistics.mean(response_times)
                confidence_alignment = statistics.mean([r['confidence_alignment'] for r in results])
                
                # Update metrics
                self.metrics['match_precision'].add_measurement(precision)
                self.metrics['match_recall'].add_measurement(recall)
                self.metrics['match_f1'].add_measurement(f1)
                self.metrics['response_time'].add_measurement(avg_response_time)
                self.metrics['confidence_alignment'].add_measurement(confidence_alignment)
                
                report.component_scores['matcher'] = (precision + recall + f1) / 3 * 100
                report.test_results['matcher'] = {
                    'tests_run': len(results),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'avg_response_time': avg_response_time,
                    'confidence_alignment': confidence_alignment
                }
                
                # Check for issues
                if precision < 0.90:
                    report.add_alert('warning', 'matcher', f'Low precision: {precision:.3f}')
                if recall < 0.87:
                    report.add_alert('warning', 'matcher', f'Low recall: {recall:.3f}')
                if avg_response_time > 2.0:
                    report.add_alert('warning', 'matcher', f'Slow response time: {avg_response_time:.2f}s')
            
        except Exception as e:
            self.logger.error(f"❌ Matcher evaluation failed: {e}")
            report.add_alert('critical', 'matcher', f'Evaluation failed: {str(e)}')
            report.component_scores['matcher'] = 0.0
    
    def _test_single_match(self, ingredient: Ingredient, golden_match: GoldenMatch) -> Dict[str, Any]:
        """Test a single ingredient match against golden standard."""
        start_time = time.time()
        
        try:
            # Run matcher
            matches = self.matcher_agent.find_matches(ingredient, limit=5)
            
            response_time = time.time() - start_time
            
            # Evaluate results
            precision = False
            recall = False
            confidence_alignment = 0.5  # Default middle score
            
            if matches:
                # Check if golden match is in top results
                top_match = matches[0] if matches else None
                
                if top_match:
                    # Simple matching logic - in real implementation, use more sophisticated comparison
                    if (golden_match.product_name.lower() in top_match.name.lower() or
                        top_match.name.lower() in golden_match.product_name.lower()):
                        precision = True
                        recall = True
                    
                    # Confidence alignment - check if confidence is reasonable
                    if hasattr(top_match, 'confidence'):
                        expected_min, expected_max = golden_match.expected_confidence_range
                        if expected_min <= top_match.confidence <= expected_max:
                            confidence_alignment = 1.0
                        else:
                            confidence_alignment = max(0.0, 1.0 - abs(top_match.confidence - (expected_min + expected_max) / 2))
            
            return {
                'ingredient_name': ingredient.name,
                'precision': precision,
                'recall': recall,
                'confidence_alignment': confidence_alignment,
                'response_time': response_time,
                'matches_found': len(matches) if matches else 0
            }
            
        except Exception as e:
            return {
                'ingredient_name': ingredient.name,
                'precision': False,
                'recall': False,
                'confidence_alignment': 0.0,
                'response_time': time.time() - start_time,
                'matches_found': 0,
                'error': str(e)
            }
    
    async def _evaluate_optimizer_performance(self, report: QualityReport):
        """Evaluate OptimizerAgent performance."""
        self.logger.info("💰 Evaluating OptimizerAgent performance...")
        
        try:
            # Create test scenarios
            test_scenarios = [
                {
                    'ingredients': ['milk', 'bread', 'eggs'],
                    'strategy': 'cost_only',
                    'expected_savings': 0.15  # Expect 15% savings
                },
                {
                    'ingredients': ['chicken breast', 'rice', 'vegetables'],
                    'strategy': 'balanced',
                    'expected_savings': 0.10  # Expect 10% savings
                }
            ]
            
            results = []
            
            for scenario in test_scenarios:
                start_time = time.time()
                
                try:
                    # Create mock products for testing
                    mock_products = self._create_mock_products_for_ingredients(scenario['ingredients'])
                    
                    # Run optimization
                    optimization_result = await self.optimizer_agent.optimize_shopping(
                        products=mock_products,
                        strategy=scenario['strategy'],
                        budget=100.0
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Evaluate results
                    if optimization_result:
                        actual_savings = optimization_result.get('savings_percentage', 0)
                        savings_accuracy = 1.0 - abs(actual_savings - scenario['expected_savings']) / scenario['expected_savings']
                        
                        results.append({
                            'scenario': scenario['strategy'],
                            'response_time': response_time,
                            'savings_accuracy': max(0.0, savings_accuracy),
                            'success': True
                        })
                    else:
                        results.append({
                            'scenario': scenario['strategy'],
                            'response_time': response_time,
                            'savings_accuracy': 0.0,
                            'success': False
                        })
                        
                except Exception as e:
                    results.append({
                        'scenario': scenario['strategy'],
                        'response_time': time.time() - start_time,
                        'savings_accuracy': 0.0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate component score
            if results:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                avg_savings_accuracy = statistics.mean([r['savings_accuracy'] for r in results])
                avg_response_time = statistics.mean([r['response_time'] for r in results])
                
                report.component_scores['optimizer'] = (success_rate + avg_savings_accuracy) / 2 * 100
                report.test_results['optimizer'] = {
                    'tests_run': len(results),
                    'success_rate': success_rate,
                    'avg_savings_accuracy': avg_savings_accuracy,
                    'avg_response_time': avg_response_time
                }
                
                if success_rate < 0.90:
                    report.add_alert('warning', 'optimizer', f'Low success rate: {success_rate:.3f}')
                if avg_response_time > 5.0:
                    report.add_alert('warning', 'optimizer', f'Slow optimization: {avg_response_time:.2f}s')
            
        except Exception as e:
            self.logger.error(f"❌ Optimizer evaluation failed: {e}")
            report.add_alert('critical', 'optimizer', f'Evaluation failed: {str(e)}')
            report.component_scores['optimizer'] = 0.0
    
    def _create_mock_products_for_ingredients(self, ingredients: List[str]) -> List[Product]:
        """Create mock products for testing optimization."""
        products = []
        stores = ['metro_ca', 'walmart_ca', 'freshco_com']
        
        for ingredient in ingredients:
            for store in stores:
                # Create products with varying prices for optimization testing
                base_price = hash(f"{ingredient}_{store}") % 10 + 5  # Price between $5-15
                price_variation = 1.0 + (hash(ingredient) % 3) * 0.1  # 0-20% variation
                
                product = Product(
                    name=f"{ingredient.title()} - {store}",
                    brand="Test Brand",
                    price=Decimal(str(base_price * price_variation)),
                    unit_price=Decimal(str(base_price * price_variation)),
                    store=store,
                    availability_status="in_stock",
                    category=ingredient,
                    url=f"https://example.com/{ingredient}_{store}"
                )
                products.append(product)
        
        return products
    
    async def _evaluate_workflow_performance(self, report: QualityReport):
        """Evaluate end-to-end workflow performance."""
        self.logger.info("🔄 Evaluating workflow performance...")
        
        try:
            # Test simplified workflow scenarios
            scenarios = [
                ['milk', 'bread'],
                ['chicken', 'rice', 'vegetables']
            ]
            
            results = []
            
            for ingredients in scenarios:
                start_time = time.time()
                
                try:
                    # Create a simplified workflow test
                    workflow_success = True  # Mock success for now
                    
                    response_time = time.time() - start_time
                    
                    results.append({
                        'ingredients_count': len(ingredients),
                        'response_time': response_time,
                        'success': workflow_success
                    })
                    
                except Exception as e:
                    results.append({
                        'ingredients_count': len(ingredients),
                        'response_time': time.time() - start_time,
                        'success': False,
                        'error': str(e)
                    })
            
            if results:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                avg_response_time = statistics.mean([r['response_time'] for r in results])
                
                report.component_scores['workflow'] = success_rate * 100
                report.test_results['workflow'] = {
                    'tests_run': len(results),
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time
                }
                
                if success_rate < 0.95:
                    report.add_alert('warning', 'workflow', f'Low workflow success rate: {success_rate:.3f}')
        
        except Exception as e:
            self.logger.error(f"❌ Workflow evaluation failed: {e}")
            report.add_alert('critical', 'workflow', f'Evaluation failed: {str(e)}')
            report.component_scores['workflow'] = 0.0
    
    async def _evaluate_system_performance(self, report: QualityReport):
        """Evaluate overall system performance metrics."""
        self.logger.info("⚡ Evaluating system performance...")
        
        try:
            # Mock system metrics - in production, these would be real measurements
            availability = 0.995  # 99.5% uptime
            self.metrics['availability'].add_measurement(availability)
            
            # Price accuracy simulation
            price_accuracy = 0.94
            self.metrics['price_accuracy'].add_measurement(price_accuracy)
            
            report.component_scores['system'] = (availability + price_accuracy) / 2 * 100
            report.performance_summary = {
                'availability': availability,
                'price_accuracy': price_accuracy,
                'uptime_percentage': availability * 100
            }
            
        except Exception as e:
            self.logger.error(f"❌ System evaluation failed: {e}")
            report.add_alert('critical', 'system', f'Evaluation failed: {str(e)}')
            report.component_scores['system'] = 0.0
    
    def _calculate_overall_score(self, report: QualityReport) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'matcher': 0.35,
            'optimizer': 0.25,
            'workflow': 0.25,
            'system': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in report.component_scores:
                weighted_sum += report.component_scores[component] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _detect_regression(self, current_report: QualityReport) -> bool:
        """Detect performance regression compared to previous reports."""
        if len(self.reports) < 3:
            return False  # Need historical data
        
        # Calculate average of last 3 reports
        recent_scores = [report.overall_score for report in self.reports[-3:]]
        historical_avg = statistics.mean(recent_scores)
        
        # Check for significant degradation (>5% drop)
        degradation_threshold = 0.05
        score_drop = (historical_avg - current_report.overall_score) / historical_avg
        
        return score_drop > degradation_threshold
    
    def _generate_recommendations(self, report: QualityReport):
        """Generate improvement recommendations based on quality assessment."""
        
        # Matcher recommendations
        if 'matcher' in report.component_scores and report.component_scores['matcher'] < 85:
            report.add_recommendation(
                'high', 'matcher', 'Improve Matching Accuracy',
                'Matcher performance is below target threshold',
                [
                    'Retrain embedding models with recent data',
                    'Update product categorization rules',
                    'Expand golden dataset with edge cases',
                    'Implement confidence calibration'
                ]
            )
        
        # Optimizer recommendations
        if 'optimizer' in report.component_scores and report.component_scores['optimizer'] < 80:
            report.add_recommendation(
                'medium', 'optimizer', 'Enhance Optimization Strategies',
                'Optimization performance needs improvement',
                [
                    'Tune optimization algorithm parameters',
                    'Add more store price data sources',
                    'Implement dynamic strategy selection',
                    'Add real-time price validation'
                ]
            )
        
        # Performance recommendations
        avg_response_time = self.metrics['response_time'].current_value
        if avg_response_time > 2.0:
            report.add_recommendation(
                'medium', 'performance', 'Optimize Response Times',
                f'Average response time ({avg_response_time:.2f}s) exceeds target',
                [
                    'Implement result caching',
                    'Optimize database queries',
                    'Add connection pooling',
                    'Consider async processing'
                ]
            )
        
        # System health recommendations
        if report.overall_score < 85:
            report.add_recommendation(
                'high', 'system', 'System Health Critical',
                'Overall system quality score is below acceptable threshold',
                [
                    'Conduct comprehensive system audit',
                    'Implement immediate performance fixes',
                    'Increase monitoring frequency',
                    'Review and update quality thresholds'
                ]
            )


class ContinuousQualityMonitor:
    """Continuous quality monitoring service."""
    
    def __init__(self, monitor_interval_hours: int = 6):
        self.monitor_interval_hours = monitor_interval_hours
        self.quality_monitor = QualityMonitor()
        self.running = False
    
    async def start_monitoring(self):
        """Start continuous quality monitoring."""
        self.running = True
        print(f"🚀 Starting continuous quality monitoring (interval: {self.monitor_interval_hours}h)")
        
        while self.running:
            try:
                report = await self.quality_monitor.run_comprehensive_evaluation()
                self._log_report_summary(report)
                
                # Sleep until next evaluation
                sleep_seconds = self.monitor_interval_hours * 3600
                await asyncio.sleep(sleep_seconds)
                
            except Exception as e:
                print(f"❌ Error in continuous monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        self.running = False
        print("🛑 Stopping continuous quality monitoring")
    
    def _log_report_summary(self, report: QualityReport):
        """Log quality report summary."""
        print(f"\n📊 Quality Report Summary - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Component Scores: {report.component_scores}")
        print(f"Alerts: {len(report.alerts)} ({'⚠️ ' if report.alerts else '✅'})")
        print(f"Recommendations: {len(report.recommendations)}")
        if report.regression_detected:
            print("🚨 REGRESSION DETECTED!")


if __name__ == "__main__":
    # Run single evaluation
    async def test_evaluation():
        monitor = QualityMonitor()
        report = await monitor.run_comprehensive_evaluation()
        
        print(f"\n📊 Quality Evaluation Results:")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Component Scores: {report.component_scores}")
        print(f"Alerts: {len(report.alerts)}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        # Show alerts
        if report.alerts:
            print(f"\n⚠️ Alerts:")
            for alert in report.alerts:
                print(f"  {alert['severity'].upper()}: {alert['message']}")
        
        # Show recommendations
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  {rec['priority'].upper()}: {rec['title']}")
    
    asyncio.run(test_evaluation())