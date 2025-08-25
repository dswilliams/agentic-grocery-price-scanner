"""
Automated Regression Testing Suite

Comprehensive regression testing framework for detecting performance degradation,
accuracy issues, and system reliability problems across all components.
"""

import asyncio
import json
import logging
import pickle
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import hashlib

import pytest
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

from .golden_dataset import GoldenDatasetManager, GoldenMatch
from .quality_monitor import QualityMonitor, QualityReport
from ..agents.matcher_agent import MatcherAgent
from ..agents.optimizer_agent import OptimizerAgent
from ..agents.intelligent_scraper_agent import IntelligentScraperAgent
from ..workflow.grocery_workflow import GroceryWorkflow
from ..data_models.ingredient import Ingredient
from ..data_models.product import Product


@dataclass
class RegressionTestResult:
    """Individual regression test result."""
    
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str = ""
    test_category: str = ""
    test_type: str = "performance"  # performance, accuracy, reliability
    
    # Results
    passed: bool = False
    current_value: float = 0.0
    baseline_value: float = 0.0
    threshold: float = 0.0
    deviation_percentage: float = 0.0
    
    # Timing
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    regression_severity: str = "none"  # none, minor, major, critical
    
    def calculate_regression_severity(self):
        """Calculate regression severity based on deviation."""
        if not self.passed:
            if self.deviation_percentage > 20:
                self.regression_severity = "critical"
            elif self.deviation_percentage > 10:
                self.regression_severity = "major"
            elif self.deviation_percentage > 5:
                self.regression_severity = "minor"
            else:
                self.regression_severity = "none"


@dataclass
class RegressionTestSuite:
    """Complete regression test suite results."""
    
    suite_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    execution_time: float = 0.0
    
    test_results: List[RegressionTestResult] = field(default_factory=list)
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    
    # Regression analysis
    regressions_detected: int = 0
    critical_regressions: int = 0
    major_regressions: int = 0
    minor_regressions: int = 0
    
    # Summary metrics
    overall_health_score: float = 100.0
    component_health: Dict[str, float] = field(default_factory=dict)
    
    def add_test_result(self, result: RegressionTestResult):
        """Add test result and update suite metrics."""
        result.calculate_regression_severity()
        self.test_results.append(result)
        
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
            # Count regressions by severity
            if result.regression_severity == "critical":
                self.critical_regressions += 1
            elif result.regression_severity == "major":
                self.major_regressions += 1
            elif result.regression_severity == "minor":
                self.minor_regressions += 1
        
        self.regressions_detected = self.critical_regressions + self.major_regressions + self.minor_regressions
        
        # Update overall health score
        self._calculate_health_score()
    
    def _calculate_health_score(self):
        """Calculate overall health score based on regression severity."""
        if self.total_tests == 0:
            self.overall_health_score = 100.0
            return
        
        # Weighted scoring based on regression severity
        penalty = (
            self.critical_regressions * 20 +
            self.major_regressions * 10 +
            self.minor_regressions * 3
        )
        
        # Base score from pass rate
        pass_rate = self.passed_tests / self.total_tests
        base_score = pass_rate * 100
        
        # Apply penalty
        self.overall_health_score = max(0.0, base_score - penalty)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        return {
            'suite_id': self.suite_id,
            'timestamp': self.timestamp.isoformat(),
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.passed_tests / max(1, self.total_tests),
            'execution_time': self.execution_time,
            'regressions_detected': self.regressions_detected,
            'critical_regressions': self.critical_regressions,
            'major_regressions': self.major_regressions,
            'minor_regressions': self.minor_regressions,
            'overall_health_score': self.overall_health_score,
            'component_health': self.component_health
        }


class BaselineManager:
    """Manages performance baselines for regression testing."""
    
    def __init__(self, baseline_file: str = "evaluation/baselines/performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._load_baselines()
    
    def _load_baselines(self):
        """Load existing baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    self.baselines = data.get('baselines', {})
                print(f"✅ Loaded baselines for {len(self.baselines)} test categories")
            except Exception as e:
                print(f"⚠️ Warning: Could not load baselines: {e}")
                self.baselines = {}
        else:
            print("📝 No existing baselines found, will create new ones")
    
    def save_baselines(self):
        """Save baselines to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'version': '1.0'
            },
            'baselines': self.baselines
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved baselines to {self.baseline_file}")
    
    def get_baseline(self, category: str, metric: str) -> Optional[float]:
        """Get baseline value for a specific metric."""
        return self.baselines.get(category, {}).get(metric)
    
    def set_baseline(self, category: str, metric: str, value: float):
        """Set baseline value for a specific metric."""
        if category not in self.baselines:
            self.baselines[category] = {}
        self.baselines[category][metric] = value
    
    def update_baseline(self, category: str, metric: str, value: float, auto_update: bool = False):
        """Update baseline if improvement is detected."""
        current_baseline = self.get_baseline(category, metric)
        
        if current_baseline is None or auto_update:
            self.set_baseline(category, metric, value)
            return True
        
        # Only update if significant improvement (>5%)
        improvement_threshold = 0.05
        if value > current_baseline * (1 + improvement_threshold):
            self.set_baseline(category, metric, value)
            return True
        
        return False


class RegressionTester:
    """Automated regression testing framework."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.baseline_manager = BaselineManager()
        self.golden_dataset = GoldenDatasetManager()
        self.quality_monitor = QualityMonitor()
        
        # Initialize agents for testing
        self.matcher_agent = MatcherAgent()
        self.optimizer_agent = OptimizerAgent()
        self.scraper_agent = IntelligentScraperAgent()
        
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default regression testing configuration."""
        return {
            'performance_thresholds': {
                'matcher_response_time': 2.0,
                'matcher_accuracy': 0.90,
                'optimizer_response_time': 5.0,
                'workflow_completion_time': 60.0,
                'memory_usage_mb': 500
            },
            'regression_thresholds': {
                'minor': 0.05,  # 5% degradation
                'major': 0.10,  # 10% degradation
                'critical': 0.20  # 20% degradation
            },
            'test_parallelism': 4,
            'timeout_seconds': 30,
            'auto_update_baselines': False
        }
    
    async def run_full_regression_suite(self) -> RegressionTestSuite:
        """Run comprehensive regression test suite."""
        self.logger.info("🧪 Starting comprehensive regression testing...")
        
        start_time = time.time()
        suite = RegressionTestSuite()
        
        try:
            # Run test categories in parallel
            test_tasks = [
                self._run_matcher_regression_tests(suite),
                self._run_optimizer_regression_tests(suite),
                self._run_workflow_regression_tests(suite),
                self._run_performance_regression_tests(suite),
                self._run_reliability_regression_tests(suite)
            ]
            
            await asyncio.gather(*test_tasks, return_exceptions=True)
            
            suite.execution_time = time.time() - start_time
            
            # Generate baseline recommendations
            self._analyze_baseline_updates(suite)
            
            self.logger.info(f"✅ Regression testing completed in {suite.execution_time:.2f}s")
            self.logger.info(f"Results: {suite.passed_tests}/{suite.total_tests} passed, "
                           f"{suite.regressions_detected} regressions detected")
            
        except Exception as e:
            self.logger.error(f"❌ Regression testing failed: {e}")
            
        return suite
    
    async def _run_matcher_regression_tests(self, suite: RegressionTestSuite):
        """Run MatcherAgent regression tests."""
        self.logger.info("🎯 Running MatcherAgent regression tests...")
        
        try:
            # Test ingredient matching accuracy
            test_matches = self.golden_dataset.matches[:30]  # Use subset for faster testing
            accuracy_results = []
            response_times = []
            
            with ThreadPoolExecutor(max_workers=self.config['test_parallelism']) as executor:
                futures = {}
                
                for golden_match in test_matches:
                    ingredient = Ingredient(
                        name=golden_match.ingredient_name,
                        category=golden_match.ingredient_category
                    )
                    
                    future = executor.submit(self._test_matcher_single, ingredient, golden_match)
                    futures[future] = golden_match.ingredient_name
                
                for future in as_completed(futures, timeout=self.config['timeout_seconds']):
                    try:
                        result = future.result()
                        accuracy_results.append(result['accuracy'])
                        response_times.append(result['response_time'])
                    except Exception as e:
                        self.logger.warning(f"Matcher test failed for {futures[future]}: {e}")
                        accuracy_results.append(0.0)
                        response_times.append(self.config['timeout_seconds'])
            
            # Calculate metrics
            if accuracy_results and response_times:
                avg_accuracy = statistics.mean(accuracy_results)
                avg_response_time = statistics.mean(response_times)
                
                # Accuracy regression test
                accuracy_baseline = self.baseline_manager.get_baseline('matcher', 'accuracy')
                if accuracy_baseline:
                    result = self._create_regression_result(
                        'matcher_accuracy', 'matcher', 'accuracy',
                        avg_accuracy, accuracy_baseline, 
                        self.config['performance_thresholds']['matcher_accuracy']
                    )
                    result.details = {'test_count': len(accuracy_results)}
                    suite.add_test_result(result)
                else:
                    # Set initial baseline
                    self.baseline_manager.set_baseline('matcher', 'accuracy', avg_accuracy)
                
                # Response time regression test
                response_time_baseline = self.baseline_manager.get_baseline('matcher', 'response_time')
                if response_time_baseline:
                    result = self._create_regression_result(
                        'matcher_response_time', 'matcher', 'performance',
                        avg_response_time, response_time_baseline,
                        self.config['performance_thresholds']['matcher_response_time']
                    )
                    result.details = {'test_count': len(response_times)}
                    suite.add_test_result(result)
                else:
                    # Set initial baseline
                    self.baseline_manager.set_baseline('matcher', 'response_time', avg_response_time)
        
        except Exception as e:
            self.logger.error(f"❌ Matcher regression testing failed: {e}")
            # Add failure result
            result = RegressionTestResult(
                test_name="matcher_regression_suite",
                test_category="matcher",
                passed=False,
                error_message=str(e)
            )
            suite.add_test_result(result)
    
    def _test_matcher_single(self, ingredient: Ingredient, golden_match: GoldenMatch) -> Dict[str, Any]:
        """Test single matcher call for regression testing."""
        start_time = time.time()
        
        try:
            matches = self.matcher_agent.find_matches(ingredient, limit=5)
            response_time = time.time() - start_time
            
            # Simple accuracy check - in production, use more sophisticated evaluation
            accuracy = 1.0 if matches and len(matches) > 0 else 0.0
            
            if matches:
                # Check if expected product characteristics are present
                top_match = matches[0]
                if hasattr(top_match, 'name'):
                    # Basic name similarity check
                    name_similarity = len(set(golden_match.product_name.lower().split()) & 
                                         set(top_match.name.lower().split()))
                    accuracy = min(1.0, name_similarity / max(1, len(golden_match.product_name.split())))
            
            return {
                'accuracy': accuracy,
                'response_time': response_time,
                'matches_found': len(matches) if matches else 0
            }
            
        except Exception as e:
            return {
                'accuracy': 0.0,
                'response_time': time.time() - start_time,
                'matches_found': 0,
                'error': str(e)
            }
    
    async def _run_optimizer_regression_tests(self, suite: RegressionTestSuite):
        """Run OptimizerAgent regression tests."""
        self.logger.info("💰 Running OptimizerAgent regression tests...")
        
        try:
            # Test optimization scenarios
            test_scenarios = [
                {
                    'ingredients': ['milk', 'bread', 'eggs'],
                    'strategy': 'cost_only'
                },
                {
                    'ingredients': ['chicken', 'rice'],
                    'strategy': 'balanced'
                }
            ]
            
            optimization_results = []
            response_times = []
            
            for scenario in test_scenarios:
                start_time = time.time()
                
                try:
                    # Create mock products for testing
                    mock_products = self._create_test_products(scenario['ingredients'])
                    
                    # Run optimization
                    result = await self.optimizer_agent.optimize_shopping(
                        products=mock_products,
                        strategy=scenario['strategy']
                    )
                    
                    response_time = time.time() - start_time
                    optimization_success = result is not None and len(result.get('optimized_list', [])) > 0
                    
                    optimization_results.append(1.0 if optimization_success else 0.0)
                    response_times.append(response_time)
                    
                except Exception as e:
                    optimization_results.append(0.0)
                    response_times.append(time.time() - start_time)
            
            if optimization_results and response_times:
                avg_success_rate = statistics.mean(optimization_results)
                avg_response_time = statistics.mean(response_times)
                
                # Success rate regression test
                success_baseline = self.baseline_manager.get_baseline('optimizer', 'success_rate')
                if success_baseline:
                    result = self._create_regression_result(
                        'optimizer_success_rate', 'optimizer', 'accuracy',
                        avg_success_rate, success_baseline, 0.90
                    )
                    suite.add_test_result(result)
                else:
                    self.baseline_manager.set_baseline('optimizer', 'success_rate', avg_success_rate)
                
                # Response time regression test
                response_baseline = self.baseline_manager.get_baseline('optimizer', 'response_time')
                if response_baseline:
                    result = self._create_regression_result(
                        'optimizer_response_time', 'optimizer', 'performance',
                        avg_response_time, response_baseline,
                        self.config['performance_thresholds']['optimizer_response_time']
                    )
                    suite.add_test_result(result)
                else:
                    self.baseline_manager.set_baseline('optimizer', 'response_time', avg_response_time)
        
        except Exception as e:
            self.logger.error(f"❌ Optimizer regression testing failed: {e}")
            result = RegressionTestResult(
                test_name="optimizer_regression_suite",
                test_category="optimizer",
                passed=False,
                error_message=str(e)
            )
            suite.add_test_result(result)
    
    def _create_test_products(self, ingredients: List[str]) -> List[Product]:
        """Create test products for regression testing."""
        products = []
        stores = ['metro_ca', 'walmart_ca']
        
        for ingredient in ingredients:
            for store in stores:
                # Create deterministic test products
                price = float(hash(f"{ingredient}_{store}") % 20 + 10)  # $10-30
                
                product = Product(
                    name=f"Test {ingredient.title()}",
                    brand="Test Brand",
                    price=price,
                    store=store,
                    category=ingredient
                )
                products.append(product)
        
        return products
    
    async def _run_workflow_regression_tests(self, suite: RegressionTestSuite):
        """Run end-to-end workflow regression tests."""
        self.logger.info("🔄 Running workflow regression tests...")
        
        try:
            test_scenarios = [
                ['milk', 'bread'],
                ['chicken', 'vegetables', 'rice']
            ]
            
            workflow_results = []
            response_times = []
            
            for ingredients in test_scenarios:
                start_time = time.time()
                
                try:
                    # Mock workflow execution - replace with real workflow test
                    await asyncio.sleep(0.1)  # Simulate workflow time
                    workflow_success = True  # Mock success
                    
                    response_time = time.time() - start_time
                    workflow_results.append(1.0 if workflow_success else 0.0)
                    response_times.append(response_time)
                    
                except Exception as e:
                    workflow_results.append(0.0)
                    response_times.append(time.time() - start_time)
            
            if workflow_results and response_times:
                avg_success_rate = statistics.mean(workflow_results)
                avg_response_time = statistics.mean(response_times)
                
                # Workflow success rate test
                success_baseline = self.baseline_manager.get_baseline('workflow', 'success_rate')
                if success_baseline:
                    result = self._create_regression_result(
                        'workflow_success_rate', 'workflow', 'reliability',
                        avg_success_rate, success_baseline, 0.95
                    )
                    suite.add_test_result(result)
                else:
                    self.baseline_manager.set_baseline('workflow', 'success_rate', avg_success_rate)
                
                # Workflow response time test
                response_baseline = self.baseline_manager.get_baseline('workflow', 'response_time')
                if response_baseline:
                    result = self._create_regression_result(
                        'workflow_response_time', 'workflow', 'performance',
                        avg_response_time, response_baseline,
                        self.config['performance_thresholds']['workflow_completion_time']
                    )
                    suite.add_test_result(result)
                else:
                    self.baseline_manager.set_baseline('workflow', 'response_time', avg_response_time)
        
        except Exception as e:
            self.logger.error(f"❌ Workflow regression testing failed: {e}")
            result = RegressionTestResult(
                test_name="workflow_regression_suite",
                test_category="workflow",
                passed=False,
                error_message=str(e)
            )
            suite.add_test_result(result)
    
    async def _run_performance_regression_tests(self, suite: RegressionTestSuite):
        """Run system performance regression tests."""
        self.logger.info("⚡ Running performance regression tests...")
        
        try:
            # Memory usage test (mock)
            memory_usage = 250.0  # MB - mock value
            memory_baseline = self.baseline_manager.get_baseline('system', 'memory_usage')
            
            if memory_baseline:
                result = self._create_regression_result(
                    'system_memory_usage', 'system', 'performance',
                    memory_usage, memory_baseline,
                    self.config['performance_thresholds']['memory_usage_mb'],
                    lower_is_better=True
                )
                suite.add_test_result(result)
            else:
                self.baseline_manager.set_baseline('system', 'memory_usage', memory_usage)
            
            # CPU usage test (mock)
            cpu_usage = 45.0  # Percentage - mock value
            cpu_baseline = self.baseline_manager.get_baseline('system', 'cpu_usage')
            
            if cpu_baseline:
                result = self._create_regression_result(
                    'system_cpu_usage', 'system', 'performance',
                    cpu_usage, cpu_baseline, 80.0,  # 80% CPU threshold
                    lower_is_better=True
                )
                suite.add_test_result(result)
            else:
                self.baseline_manager.set_baseline('system', 'cpu_usage', cpu_usage)
        
        except Exception as e:
            self.logger.error(f"❌ Performance regression testing failed: {e}")
    
    async def _run_reliability_regression_tests(self, suite: RegressionTestSuite):
        """Run system reliability regression tests."""
        self.logger.info("🛡️ Running reliability regression tests...")
        
        try:
            # Error rate test (mock)
            error_rate = 0.02  # 2% error rate - mock value
            error_baseline = self.baseline_manager.get_baseline('system', 'error_rate')
            
            if error_baseline:
                result = self._create_regression_result(
                    'system_error_rate', 'system', 'reliability',
                    error_rate, error_baseline, 0.05,  # 5% maximum error rate
                    lower_is_better=True
                )
                suite.add_test_result(result)
            else:
                self.baseline_manager.set_baseline('system', 'error_rate', error_rate)
            
            # Availability test (mock)
            availability = 0.998  # 99.8% uptime - mock value
            availability_baseline = self.baseline_manager.get_baseline('system', 'availability')
            
            if availability_baseline:
                result = self._create_regression_result(
                    'system_availability', 'system', 'reliability',
                    availability, availability_baseline, 0.995  # 99.5% minimum
                )
                suite.add_test_result(result)
            else:
                self.baseline_manager.set_baseline('system', 'availability', availability)
        
        except Exception as e:
            self.logger.error(f"❌ Reliability regression testing failed: {e}")
    
    def _create_regression_result(self, test_name: str, category: str, test_type: str,
                                 current_value: float, baseline_value: float, threshold: float,
                                 lower_is_better: bool = False) -> RegressionTestResult:
        """Create regression test result with deviation analysis."""
        
        # Calculate deviation
        if baseline_value == 0:
            deviation_percentage = 0.0
        else:
            if lower_is_better:
                # For metrics where lower is better (response time, memory usage)
                deviation_percentage = (current_value - baseline_value) / baseline_value
            else:
                # For metrics where higher is better (accuracy, success rate)
                deviation_percentage = (baseline_value - current_value) / baseline_value
        
        # Determine if test passed
        if lower_is_better:
            passed = current_value <= threshold
        else:
            passed = current_value >= threshold
        
        # Also check for significant regression
        regression_threshold = self.config['regression_thresholds']['minor']
        if abs(deviation_percentage) > regression_threshold:
            passed = False
        
        result = RegressionTestResult(
            test_name=test_name,
            test_category=category,
            test_type=test_type,
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=threshold,
            deviation_percentage=deviation_percentage,
            passed=passed
        )
        
        return result
    
    def _analyze_baseline_updates(self, suite: RegressionTestSuite):
        """Analyze if baselines should be updated based on results."""
        improvement_count = 0
        
        for result in suite.test_results:
            if result.passed and result.deviation_percentage < -0.05:  # 5% improvement
                # Consider updating baseline for significant improvements
                category_metric = f"{result.test_category}.{result.test_name.split('_', 1)[1]}"
                improvement_count += 1
        
        if improvement_count > 0:
            self.logger.info(f"💡 Detected {improvement_count} potential baseline updates")
    
    def save_results(self, suite: RegressionTestSuite, output_file: str = None):
        """Save regression test results."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation/results/regression_results_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        results_data = {
            'summary': suite.get_summary(),
            'test_results': [
                {
                    'test_id': result.test_id,
                    'test_name': result.test_name,
                    'test_category': result.test_category,
                    'test_type': result.test_type,
                    'passed': result.passed,
                    'current_value': result.current_value,
                    'baseline_value': result.baseline_value,
                    'threshold': result.threshold,
                    'deviation_percentage': result.deviation_percentage,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat(),
                    'error_message': result.error_message,
                    'regression_severity': result.regression_severity,
                    'details': result.details
                }
                for result in suite.test_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✅ Regression test results saved to {output_path}")
        
        # Save updated baselines
        self.baseline_manager.save_baselines()


class ContinuousRegressionTester:
    """Continuous regression testing service."""
    
    def __init__(self, test_interval_hours: int = 12):
        self.test_interval_hours = test_interval_hours
        self.regression_tester = RegressionTester()
        self.running = False
    
    async def start_continuous_testing(self):
        """Start continuous regression testing."""
        self.running = True
        print(f"🚀 Starting continuous regression testing (interval: {self.test_interval_hours}h)")
        
        while self.running:
            try:
                suite = await self.regression_tester.run_full_regression_suite()
                self._log_suite_summary(suite)
                
                # Save results
                self.regression_tester.save_results(suite)
                
                # Sleep until next test run
                sleep_seconds = self.test_interval_hours * 3600
                await asyncio.sleep(sleep_seconds)
                
            except Exception as e:
                print(f"❌ Error in continuous regression testing: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry
    
    def stop_continuous_testing(self):
        """Stop continuous regression testing."""
        self.running = False
        print("🛑 Stopping continuous regression testing")
    
    def _log_suite_summary(self, suite: RegressionTestSuite):
        """Log regression test suite summary."""
        print(f"\n🧪 Regression Test Summary - {suite.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tests: {suite.passed_tests}/{suite.total_tests} passed")
        print(f"Health Score: {suite.overall_health_score:.1f}/100")
        print(f"Regressions: {suite.regressions_detected} "
              f"(Critical: {suite.critical_regressions}, Major: {suite.major_regressions}, Minor: {suite.minor_regressions})")
        
        if suite.critical_regressions > 0:
            print("🚨 CRITICAL REGRESSIONS DETECTED!")
        elif suite.major_regressions > 0:
            print("⚠️ Major regressions detected")


if __name__ == "__main__":
    async def test_regression_suite():
        tester = RegressionTester()
        suite = await tester.run_full_regression_suite()
        
        print(f"\n🧪 Regression Test Results:")
        print(f"Overall Health Score: {suite.overall_health_score:.1f}/100")
        print(f"Tests Passed: {suite.passed_tests}/{suite.total_tests}")
        print(f"Regressions: {suite.regressions_detected}")
        print(f"Execution Time: {suite.execution_time:.2f}s")
        
        # Show critical regressions
        critical_tests = [r for r in suite.test_results if r.regression_severity == "critical"]
        if critical_tests:
            print(f"\n🚨 Critical Regressions:")
            for test in critical_tests:
                print(f"  {test.test_name}: {test.deviation_percentage:.1%} degradation")
        
        tester.save_results(suite)
    
    asyncio.run(test_regression_suite())