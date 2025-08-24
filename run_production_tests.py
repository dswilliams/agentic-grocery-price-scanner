"""
Production-level test runner demonstrating real-world scenarios with high load,
store failures, and comprehensive system validation.
"""

import asyncio
import logging
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import sys
import psutil
import gc

# Import production monitoring
from production_health_monitor import ProductionHealthMonitor

# Import all production components
from agentic_grocery_price_scanner.workflow import GroceryWorkflow, WorkflowStatus
from agentic_grocery_price_scanner.data_models import Recipe, Ingredient, Product
from agentic_grocery_price_scanner.data_models.base import UnitType
from agentic_grocery_price_scanner.config.store_profiles import store_profile_manager
from agentic_grocery_price_scanner.reliability import scraping_reliability_manager
from agentic_grocery_price_scanner.quality import data_quality_manager
from agentic_grocery_price_scanner.monitoring import performance_monitor
from agentic_grocery_price_scanner.caching import cache_manager
from agentic_grocery_price_scanner.recovery import error_recovery_manager

logger = logging.getLogger(__name__)


class ProductionTestRunner:
    """Comprehensive production test runner with real-world scenarios."""
    
    def __init__(self):
        self.health_monitor = ProductionHealthMonitor()
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        
        # Test scenarios
        self.scenarios = {
            "basic_workflow": self._test_basic_workflow,
            "concurrent_load": self._test_concurrent_load,
            "store_failures": self._test_store_failures,
            "high_stress": self._test_high_stress_conditions,
            "error_recovery": self._test_error_recovery,
            "cache_performance": self._test_cache_performance,
            "data_quality": self._test_data_quality,
            "endurance": self._test_endurance_operation
        }
        
        # Real-world test data
        self.test_recipes = [
            Recipe(
                name="Family Breakfast",
                ingredients=[
                    Ingredient(name="eggs", quantity=12, unit=UnitType.UNIT),
                    Ingredient(name="milk", quantity=2, unit=UnitType.LITER),
                    Ingredient(name="bread", quantity=2, unit=UnitType.UNIT),
                    Ingredient(name="butter", quantity=500, unit=UnitType.GRAM),
                    Ingredient(name="orange juice", quantity=1, unit=UnitType.LITER)
                ]
            ),
            Recipe(
                name="Week Dinner Prep",
                ingredients=[
                    Ingredient(name="chicken breast", quantity=2, unit=UnitType.KILOGRAM),
                    Ingredient(name="ground beef", quantity=1, unit=UnitType.KILOGRAM),
                    Ingredient(name="salmon fillets", quantity=800, unit=UnitType.GRAM),
                    Ingredient(name="pasta", quantity=1, unit=UnitType.KILOGRAM),
                    Ingredient(name="rice", quantity=2, unit=UnitType.KILOGRAM),
                    Ingredient(name="mixed vegetables", quantity=1.5, unit=UnitType.KILOGRAM),
                    Ingredient(name="onions", quantity=2, unit=UnitType.KILOGRAM),
                    Ingredient(name="tomatoes", quantity=1, unit=UnitType.KILOGRAM)
                ]
            ),
            Recipe(
                name="Party Shopping",
                ingredients=[
                    Ingredient(name="chips", quantity=6, unit=UnitType.UNIT),
                    Ingredient(name="soda", quantity=12, unit=UnitType.UNIT),
                    Ingredient(name="cheese", quantity=1, unit=UnitType.KILOGRAM),
                    Ingredient(name="crackers", quantity=4, unit=UnitType.UNIT),
                    Ingredient(name="deli meat", quantity=500, unit=UnitType.GRAM),
                    Ingredient(name="vegetables for dip", quantity=1, unit=UnitType.UNIT),
                    Ingredient(name="ice cream", quantity=3, unit=UnitType.UNIT)
                ]
            )
        ]
        
        self.simple_ingredient_lists = [
            ["milk", "bread", "eggs"],
            ["chicken", "rice", "vegetables"],
            ["pasta", "tomatoes", "cheese"],
            ["yogurt", "bananas", "cereal"],
            ["ground beef", "onions", "potatoes"]
        ]
        
        logger.info("Initialized ProductionTestRunner")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all production tests."""
        logger.info("🚀 Starting comprehensive production tests")
        
        # Start monitoring
        await self.health_monitor.start_monitoring()
        await performance_monitor.start_monitoring()
        
        try:
            # Get baseline health
            baseline_health = await self.health_monitor.comprehensive_health_check()
            logger.info(f"📊 Baseline health score: {baseline_health.score:.1f}/100")
            
            # Run each test scenario
            for scenario_name, test_func in self.scenarios.items():
                logger.info(f"🧪 Running test: {scenario_name}")
                
                start_time = time.time()
                
                try:
                    result = await test_func()
                    execution_time = time.time() - start_time
                    
                    self.test_results[scenario_name] = {
                        "status": "passed",
                        "execution_time": execution_time,
                        "result": result
                    }
                    
                    logger.info(f"✅ {scenario_name} passed in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.test_results[scenario_name] = {
                        "status": "failed",
                        "execution_time": execution_time,
                        "error": str(e)
                    }
                    
                    logger.error(f"❌ {scenario_name} failed after {execution_time:.2f}s: {e}")
                
                # Brief pause between tests
                await asyncio.sleep(2)
                gc.collect()
            
            # Final health check
            final_health = await self.health_monitor.comprehensive_health_check()
            logger.info(f"📊 Final health score: {final_health.score:.1f}/100")
            
            # Compile final results
            final_results = await self._compile_final_results(baseline_health, final_health)
            
            return final_results
            
        finally:
            await performance_monitor.stop_monitoring()
            await self.health_monitor.stop_monitoring()
    
    async def run_specific_test(self, test_name: str) -> Dict[str, Any]:
        """Run a specific test scenario."""
        if test_name not in self.scenarios:
            raise ValueError(f"Unknown test scenario: {test_name}")
        
        logger.info(f"🧪 Running specific test: {test_name}")
        
        await performance_monitor.start_monitoring()
        
        try:
            start_time = time.time()
            result = await self.scenarios[test_name]()
            execution_time = time.time() - start_time
            
            return {
                "test_name": test_name,
                "status": "passed",
                "execution_time": execution_time,
                "result": result
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "test_name": test_name,
                "status": "failed",
                "execution_time": execution_time,
                "error": str(e)
            }
        
        finally:
            await performance_monitor.stop_monitoring()
    
    async def _test_basic_workflow(self) -> Dict[str, Any]:
        """Test basic workflow functionality."""
        workflow = GroceryWorkflow()
        
        # Test simple ingredient list
        ingredients = ["milk", "bread", "eggs", "chicken"]
        
        result = await workflow.execute(
            recipes=None,
            ingredients=ingredients,
            config={
                "optimization_strategy": "balanced",
                "target_stores": ["metro_ca", "walmart_ca"],
                "workflow_timeout": 60
            }
        )
        
        # Validate results
        assert result is not None, "Workflow should return results"
        assert result.get("status") == WorkflowStatus.COMPLETED.value, "Workflow should complete successfully"
        
        return {
            "ingredients_processed": len(ingredients),
            "status": result.get("status"),
            "stores_used": result.get("optimization_results", {}).get("stores_used", [])
        }
    
    async def _test_concurrent_load(self) -> Dict[str, Any]:
        """Test concurrent workflow execution."""
        concurrent_count = 8
        
        async def run_workflow(workflow_id: int):
            workflow = GroceryWorkflow()
            ingredients = random.choice(self.simple_ingredient_lists)
            
            start_time = time.time()
            
            try:
                result = await workflow.execute(
                    recipes=None,
                    ingredients=ingredients,
                    config={
                        "optimization_strategy": "balanced",
                        "target_stores": ["metro_ca", "walmart_ca"],
                        "workflow_timeout": 90
                    }
                )
                
                execution_time = time.time() - start_time
                success = result and result.get("status") == WorkflowStatus.COMPLETED.value
                
                return {
                    "workflow_id": workflow_id,
                    "success": success,
                    "execution_time": execution_time,
                    "ingredients": len(ingredients)
                }
            
            except Exception as e:
                execution_time = time.time() - start_time
                return {
                    "workflow_id": workflow_id,
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e)
                }
        
        # Execute workflows concurrently
        tasks = [run_workflow(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful_workflows = [r for r in results if r["success"]]
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        success_rate = (len(successful_workflows) / len(results)) * 100
        
        return {
            "total_workflows": concurrent_count,
            "successful_workflows": len(successful_workflows),
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max(r["execution_time"] for r in results)
        }
    
    async def _test_store_failures(self) -> Dict[str, Any]:
        """Test system behavior with store failures."""
        # Simulate store failures by triggering circuit breakers
        original_states = {}
        
        # Force circuit breaker open for one store
        test_store = "metro_ca"
        profile = store_profile_manager.get_profile(test_store)
        original_states[test_store] = {
            "failures": profile.consecutive_failures,
            "circuit_open": store_profile_manager.circuit_breakers[test_store].is_open
        }
        
        # Trigger circuit breaker
        profile.consecutive_failures = profile.circuit_breaker_threshold + 1
        store_profile_manager.circuit_breakers[test_store].is_open = True
        store_profile_manager.circuit_breakers[test_store].next_retry_time = datetime.now() + timedelta(minutes=5)
        
        try:
            workflow = GroceryWorkflow()
            
            result = await workflow.execute(
                recipes=None,
                ingredients=["milk", "bread", "eggs"],
                config={
                    "optimization_strategy": "balanced",
                    "target_stores": ["metro_ca", "walmart_ca"],  # Include the "failed" store
                    "fallback_enabled": True,
                    "workflow_timeout": 60
                }
            )
            
            # Should still complete despite store failure
            success = result and result.get("status") == WorkflowStatus.COMPLETED.value
            
            return {
                "workflow_completed": success,
                "failed_store": test_store,
                "fallback_used": True,
                "result_status": result.get("status") if result else None
            }
        
        finally:
            # Restore original state
            profile.consecutive_failures = original_states[test_store]["failures"]
            store_profile_manager.circuit_breakers[test_store].is_open = original_states[test_store]["circuit_open"]
            store_profile_manager.circuit_breakers[test_store].next_retry_time = None
    
    async def _test_high_stress_conditions(self) -> Dict[str, Any]:
        """Test system under high stress."""
        stress_workflows = 15
        large_ingredient_count = 20
        
        # Create large ingredient lists
        all_ingredients = [
            "milk", "bread", "eggs", "chicken", "beef", "pork", "fish", "salmon",
            "rice", "pasta", "potatoes", "onions", "tomatoes", "carrots", "broccoli",
            "cheese", "yogurt", "butter", "oil", "salt", "pepper", "garlic",
            "apples", "bananas", "oranges", "lettuce", "spinach", "bell peppers"
        ]
        
        async def stress_workflow(workflow_id: int):
            workflow = GroceryWorkflow()
            ingredients = random.sample(all_ingredients, min(large_ingredient_count, len(all_ingredients)))
            
            start_time = time.time()
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = await workflow.execute(
                    recipes=None,
                    ingredients=ingredients,
                    config={
                        "optimization_strategy": "adaptive",
                        "target_stores": ["metro_ca", "walmart_ca", "freshco_com"],
                        "enable_parallel_scraping": True,
                        "enable_parallel_matching": True,
                        "workflow_timeout": 120
                    }
                )
                
                execution_time = time.time() - start_time
                memory_end = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_end - memory_start
                
                success = result and result.get("status") == WorkflowStatus.COMPLETED.value
                
                return {
                    "workflow_id": workflow_id,
                    "success": success,
                    "execution_time": execution_time,
                    "memory_used_mb": memory_used,
                    "ingredients_count": len(ingredients)
                }
            
            except Exception as e:
                execution_time = time.time() - start_time
                return {
                    "workflow_id": workflow_id,
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e)
                }
        
        # Execute stress test
        tasks = [stress_workflow(i) for i in range(stress_workflows)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        success_rate = (len(successful) / len(results)) * 100
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        max_memory = max(r.get("memory_used_mb", 0) for r in results)
        
        return {
            "stress_workflows": stress_workflows,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "max_memory_used_mb": max_memory,
            "total_ingredients_processed": sum(r.get("ingredients_count", 0) for r in successful)
        }
    
    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms."""
        workflow_id = f"test_recovery_{int(time.time())}"
        
        # Create a checkpoint first
        checkpoint_id = await error_recovery_manager.create_workflow_checkpoint(
            workflow_id=workflow_id,
            stage="test_stage",
            state_data={"test": "data", "ingredients": ["milk", "bread"]},
            completed_stages=["initialization"],
            progress=0.5
        )
        
        # Test checkpoint retrieval
        retrieved_checkpoint = await error_recovery_manager.checkpoint_manager.get_latest_checkpoint(workflow_id)
        
        assert retrieved_checkpoint is not None, "Should be able to retrieve checkpoint"
        assert retrieved_checkpoint.workflow_id == workflow_id, "Checkpoint should match workflow ID"
        
        # Test error handling
        test_error = ValueError("Test error for recovery")
        recovery_action = await error_recovery_manager.handle_error(
            workflow_id=workflow_id,
            stage="test_stage",
            error=test_error,
            execution_context={"test": True},
            state_data={"test_data": True}
        )
        
        # Get recovery report
        recovery_report = await error_recovery_manager.get_recovery_report()
        
        return {
            "checkpoint_created": checkpoint_id is not None,
            "checkpoint_retrieved": retrieved_checkpoint is not None,
            "recovery_action": recovery_action.value,
            "dlq_items": recovery_report["dead_letter_queue"]["pending_items"]
        }
    
    async def _test_cache_performance(self) -> Dict[str, Any]:
        """Test cache system performance."""
        # Test cache operations
        test_keys = [f"test_key_{i}" for i in range(100)]
        test_data = [{"product": f"test_product_{i}", "price": 4.99 + i} for i in range(100)]
        
        # Populate cache
        for key, data in zip(test_keys, test_data):
            await cache_manager.put(key, data, ttl_seconds=3600)
        
        # Test cache retrieval performance
        cache_hits = 0
        retrieval_times = []
        
        for key in test_keys:
            start_time = time.time()
            cached_data = await cache_manager.get(key)
            retrieval_time = time.time() - start_time
            
            retrieval_times.append(retrieval_time)
            
            if cached_data is not None:
                cache_hits += 1
        
        # Test cache analysis
        cache_analysis = await cache_manager.analyze_cache_performance()
        
        hit_rate = (cache_hits / len(test_keys)) * 100
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        
        return {
            "cache_hit_rate": hit_rate,
            "avg_retrieval_time_ms": avg_retrieval_time * 1000,
            "cache_efficiency_score": cache_analysis["performance_metrics"]["cache_efficiency_score"],
            "memory_utilization": cache_analysis["memory_cache"]["utilization"]
        }
    
    async def _test_data_quality(self) -> Dict[str, Any]:
        """Test data quality framework."""
        # Create test products with various quality issues
        test_products = []
        
        # Good products
        for i in range(50):
            test_products.append(Product(
                name=f"Good Product {i}",
                price=4.99 + (i * 0.1),
                store_id="test_store",
                brand="TestBrand",
                image_url="https://example.com/image.jpg"
            ))
        
        # Products with issues
        test_products.extend([
            Product(name="", price=5.99, store_id="test_store"),  # Missing name
            Product(name="Bad Price Product", price=0, store_id="test_store"),  # Invalid price
            Product(name="Expensive Product", price=999.99, store_id="test_store"),  # Price anomaly
            Product(name="Duplicate Product", price=4.99, store_id="test_store"),  # Duplicate
            Product(name="Duplicate Product", price=4.99, store_id="test_store")   # Duplicate
        ])
        
        # Run quality assessment
        metrics, alerts = await data_quality_manager.assess_product_quality(test_products, "test_store")
        
        return {
            "total_products": metrics.total_products,
            "overall_quality_score": metrics.overall_quality_score,
            "valid_products": metrics.valid_products,
            "products_with_issues": metrics.products_with_issues,
            "alerts_generated": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity.value == "critical"])
        }
    
    async def _test_endurance_operation(self) -> Dict[str, Any]:
        """Test endurance operation (abbreviated for test suite)."""
        test_duration_seconds = 60  # Reduced for test suite
        workflow_interval = 10
        
        end_time = time.time() + test_duration_seconds
        workflow_count = 0
        successful_workflows = 0
        
        while time.time() < end_time:
            ingredients = random.choice(self.simple_ingredient_lists)
            
            try:
                workflow = GroceryWorkflow()
                result = await workflow.execute(
                    recipes=None,
                    ingredients=ingredients,
                    config={
                        "optimization_strategy": "balanced",
                        "target_stores": ["metro_ca", "walmart_ca"],
                        "workflow_timeout": 30
                    }
                )
                
                workflow_count += 1
                
                if result and result.get("status") == WorkflowStatus.COMPLETED.value:
                    successful_workflows += 1
            
            except Exception as e:
                workflow_count += 1
                logger.warning(f"Endurance workflow failed: {e}")
            
            await asyncio.sleep(workflow_interval)
        
        success_rate = (successful_workflows / max(workflow_count, 1)) * 100
        
        return {
            "duration_seconds": test_duration_seconds,
            "workflows_executed": workflow_count,
            "successful_workflows": successful_workflows,
            "success_rate": success_rate
        }
    
    async def _compile_final_results(
        self,
        baseline_health: Any,
        final_health: Any
    ) -> Dict[str, Any]:
        """Compile final test results."""
        
        total_execution_time = time.time() - self.start_time
        
        # Count results
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "passed"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "failed"])
        total_tests = len(self.test_results)
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Get performance metrics
        performance_report = performance_monitor.get_performance_report()
        
        # Get system resource usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_execution_time,
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "health_comparison": {
                "baseline_score": baseline_health.score,
                "final_score": final_health.score,
                "score_change": final_health.score - baseline_health.score
            },
            "resource_usage": {
                "peak_memory_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            },
            "performance_metrics": performance_report.get("overall_metrics", {}),
            "individual_tests": self.test_results,
            "recommendations": final_health.recommendations
        }
        
        return final_results


async def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Production Test Runner")
    parser.add_argument("--test", type=str, help="Run specific test scenario")
    parser.add_argument("--list-tests", action="store_true", help="List available tests")
    parser.add_argument("--export", type=str, help="Export results to file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize test runner
    test_runner = ProductionTestRunner()
    
    try:
        if args.list_tests:
            # List available tests
            print("📋 Available test scenarios:")
            for test_name in test_runner.scenarios.keys():
                print(f"  • {test_name}")
            return
        
        if args.test:
            # Run specific test
            print(f"🧪 Running test: {args.test}")
            result = await test_runner.run_specific_test(args.test)
            
            status_emoji = "✅" if result["status"] == "passed" else "❌"
            print(f"\n{status_emoji} Test Result: {result['status'].upper()}")
            print(f"⏱️ Execution time: {result['execution_time']:.2f}s")
            
            if result["status"] == "failed":
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📊 Results: {json.dumps(result['result'], indent=2, default=str)}")
        
        else:
            # Run all tests
            print("🚀 Running comprehensive production tests...")
            print("This may take several minutes...")
            
            results = await test_runner.run_all_tests()
            
            # Display summary
            summary = results["test_summary"]
            health = results["health_comparison"]
            
            print("\n" + "="*60)
            print("🏁 PRODUCTION TEST RESULTS SUMMARY")
            print("="*60)
            
            print(f"📊 Test Results: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['success_rate']:.1f}%)")
            print(f"⏱️ Total execution time: {results['total_execution_time']:.2f}s")
            print(f"🏥 Health score change: {health['baseline_score']:.1f} → {health['final_score']:.1f} ({health['score_change']:+.1f})")
            print(f"💾 Peak memory usage: {results['resource_usage']['peak_memory_mb']:.1f} MB")
            
            # Show individual test results
            print("\n📋 Individual Test Results:")
            for test_name, test_result in results["individual_tests"].items():
                status_emoji = "✅" if test_result["status"] == "passed" else "❌"
                print(f"  {status_emoji} {test_name}: {test_result['status']} ({test_result['execution_time']:.2f}s)")
                
                if test_result["status"] == "failed":
                    print(f"     ❌ {test_result.get('error', 'Unknown error')}")
            
            # Show recommendations
            if results["recommendations"]:
                print("\n💡 Recommendations:")
                for rec in results["recommendations"]:
                    print(f"  • {rec}")
            
            # Export results if requested
            if args.export:
                export_path = Path(args.export)
                with open(export_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results exported to {export_path}")
            
            # Exit with appropriate code
            if summary["success_rate"] < 80:
                print(f"\n⚠️ WARNING: Success rate {summary['success_rate']:.1f}% below 80% threshold")
                sys.exit(1)
            
            print(f"\n🎉 Production tests completed successfully!")
    
    except Exception as e:
        logger.error(f"Error running production tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())