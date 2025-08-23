"""
Comprehensive test suite for the OptimizerAgent.
Tests multi-store shopping optimization with various strategies and scenarios.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import List, Dict, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_ingredients() -> List[Dict[str, Any]]:
    """Create a comprehensive list of test ingredients."""
    return [
        {"name": "milk", "quantity": 2.0, "unit": "l", "category": "dairy"},
        {"name": "bread", "quantity": 1.0, "unit": "pieces", "category": "bakery"},
        {"name": "eggs", "quantity": 12.0, "unit": "pieces", "category": "dairy"},
        {"name": "chicken breast", "quantity": 1.0, "unit": "kg", "category": "meat"},
        {"name": "bananas", "quantity": 6.0, "unit": "pieces", "category": "produce"},
        {"name": "rice", "quantity": 2.0, "unit": "kg", "category": "grains"},
        {"name": "olive oil", "quantity": 500.0, "unit": "ml", "category": "cooking"},
        {"name": "tomatoes", "quantity": 4.0, "unit": "pieces", "category": "produce"},
        {"name": "cheese", "quantity": 200.0, "unit": "g", "category": "dairy"},
        {"name": "pasta", "quantity": 500.0, "unit": "g", "category": "grains"}
    ]

def create_large_test_ingredients() -> List[Dict[str, Any]]:
    """Create a large shopping list for stress testing."""
    base_ingredients = create_test_ingredients()
    
    # Add more ingredients for comprehensive testing
    additional_ingredients = [
        {"name": "yogurt", "quantity": 1.0, "unit": "pieces", "category": "dairy"},
        {"name": "apples", "quantity": 8.0, "unit": "pieces", "category": "produce"},
        {"name": "carrots", "quantity": 1.0, "unit": "kg", "category": "produce"},
        {"name": "onions", "quantity": 1.0, "unit": "kg", "category": "produce"},
        {"name": "potatoes", "quantity": 2.0, "unit": "kg", "category": "produce"},
        {"name": "ground beef", "quantity": 500.0, "unit": "g", "category": "meat"},
        {"name": "salmon fillet", "quantity": 400.0, "unit": "g", "category": "meat"},
        {"name": "butter", "quantity": 250.0, "unit": "g", "category": "dairy"},
        {"name": "flour", "quantity": 1.0, "unit": "kg", "category": "baking"},
        {"name": "sugar", "quantity": 500.0, "unit": "g", "category": "baking"},
        {"name": "salt", "quantity": 1.0, "unit": "pieces", "category": "cooking"},
        {"name": "black pepper", "quantity": 1.0, "unit": "pieces", "category": "cooking"},
        {"name": "garlic", "quantity": 3.0, "unit": "pieces", "category": "produce"},
        {"name": "bell peppers", "quantity": 3.0, "unit": "pieces", "category": "produce"},
        {"name": "broccoli", "quantity": 1.0, "unit": "pieces", "category": "produce"}
    ]
    
    return base_ingredients + additional_ingredients

class OptimizerAgentTester:
    """Comprehensive tester for OptimizerAgent functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.performance_metrics = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🧪 Starting OptimizerAgent Test Suite")
        print("=" * 60)
        
        test_methods = [
            ("basic_functionality", self.test_basic_functionality),
            ("strategy_comparison", self.test_strategy_comparison),
            ("budget_constraints", self.test_budget_constraints),
            ("store_preferences", self.test_store_preferences),
            ("large_shopping_list", self.test_large_shopping_list),
            ("quality_optimization", self.test_quality_optimization),
            ("time_efficiency", self.test_time_efficiency),
            ("savings_estimation", self.test_savings_estimation),
            ("error_handling", self.test_error_handling),
            ("performance_benchmarks", self.test_performance_benchmarks),
            ("integration_tests", self.test_integration_tests),
            ("workflow_validation", self.test_workflow_validation),
            ("multi_criteria_scoring", self.test_multi_criteria_scoring),
            ("real_world_scenarios", self.test_real_world_scenarios)
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n🔍 Running test: {test_name}")
            try:
                start_time = datetime.now()
                result = await test_method()
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                self.test_results[test_name] = {
                    "success": result.get("success", False),
                    "duration": duration,
                    "details": result,
                    "timestamp": start_time.isoformat()
                }
                
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                print(f"   {status} ({duration:.2f}s)")
                
                if not result.get("success", False):
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ FAILED - Exception: {e}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Generate summary report
        return self.generate_test_report()
    
    async def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic OptimizerAgent functionality."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            # Initialize optimizer
            optimizer = OptimizerAgent()
            
            # Test basic optimization
            result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                strategy="adaptive"
            )
            
            # Validate results
            success = (
                result.get("success", False) and
                "recommended_strategy" in result and
                "savings_analysis" in result and
                len(result["recommended_strategy"]) > 0
            )
            
            return {
                "success": success,
                "total_trips": len(result.get("recommended_strategy", [])),
                "total_cost": result.get("savings_analysis", {}).get("optimized_cost", 0),
                "coverage": result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "strategy_selected": result.get("optimization_summary", {}).get("selected_strategy", "unknown")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_strategy_comparison(self) -> Dict[str, Any]:
        """Test all optimization strategies."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.0, unit=UnitType.KG, category="meat"),
                Ingredient(name="bananas", quantity=6.0, unit=UnitType.PIECES, category="produce")
            ]
            
            optimizer = OptimizerAgent()
            strategies = ["cost_only", "convenience", "balanced", "quality_first", "time_efficient"]
            
            strategy_results = await optimizer.compare_strategies(
                ingredients=ingredients,
                strategies=strategies
            )
            
            # Validate that all strategies ran
            successful_strategies = sum(1 for result in strategy_results.values() if result.get("success", False))
            
            # Analyze results
            costs = []
            store_counts = []
            times = []
            
            for strategy, result in strategy_results.items():
                if result.get("success", False):
                    costs.append(result["savings_analysis"]["optimized_cost"])
                    store_counts.append(result["optimization_summary"]["total_stores"])
                    times.append(result["optimization_summary"]["total_time"])
            
            return {
                "success": successful_strategies >= 3,  # At least 3 strategies should work
                "successful_strategies": successful_strategies,
                "total_strategies_tested": len(strategies),
                "cost_range": f"${min(costs):.2f} - ${max(costs):.2f}" if costs else "N/A",
                "store_range": f"{min(store_counts)} - {max(store_counts)}" if store_counts else "N/A",
                "time_range": f"{min(times)} - {max(times)} min" if times else "N/A",
                "strategy_results": {k: v.get("success", False) for k, v in strategy_results.items()}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_budget_constraints(self) -> Dict[str, Any]:
        """Test budget constraint handling."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from decimal import Decimal
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.0, unit=UnitType.KG, category="meat")
            ]
            
            optimizer = OptimizerAgent()
            
            # Test 1: High budget (should not constrain)
            high_budget_criteria = OptimizationCriteria(max_budget=Decimal("200.00"))
            high_budget_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                criteria=high_budget_criteria,
                strategy="cost_only"
            )
            
            # Test 2: Low budget (should constrain)
            low_budget_criteria = OptimizationCriteria(max_budget=Decimal("20.00"))
            low_budget_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                criteria=low_budget_criteria,
                strategy="cost_only"
            )
            
            # Validate budget constraints
            high_budget_success = high_budget_result.get("success", False)
            low_budget_cost = low_budget_result.get("savings_analysis", {}).get("optimized_cost", 999)
            
            # Low budget should either succeed with cost <= budget or fail gracefully
            low_budget_valid = (
                not low_budget_result.get("success", True) or 
                float(low_budget_cost) <= 20.00
            )
            
            return {
                "success": high_budget_success and low_budget_valid,
                "high_budget_cost": high_budget_result.get("savings_analysis", {}).get("optimized_cost", 0),
                "low_budget_cost": low_budget_cost,
                "low_budget_constrained": float(low_budget_cost) <= 20.00 if low_budget_result.get("success") else "failed",
                "high_budget_trips": len(high_budget_result.get("recommended_strategy", [])),
                "low_budget_trips": len(low_budget_result.get("recommended_strategy", []))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_store_preferences(self) -> Dict[str, Any]:
        """Test store preference handling."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery")
            ]
            
            optimizer = OptimizerAgent()
            
            # Test with preferred stores
            preferred_criteria = OptimizationCriteria(
                preferred_stores=["metro_ca"],
                max_stores=2
            )
            
            preferred_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                criteria=preferred_criteria,
                strategy="balanced"
            )
            
            # Test with avoided stores
            avoid_criteria = OptimizationCriteria(
                avoid_stores=["walmart_ca"],
                max_stores=3
            )
            
            avoid_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                criteria=avoid_criteria,
                strategy="balanced"
            )
            
            # Validate preferences
            preferred_success = preferred_result.get("success", False)
            avoid_success = avoid_result.get("success", False)
            
            # Check if preferred store was used
            preferred_stores_used = [
                trip["store_id"] for trip in preferred_result.get("recommended_strategy", [])
            ]
            
            avoided_stores_used = [
                trip["store_id"] for trip in avoid_result.get("recommended_strategy", [])
            ]
            
            return {
                "success": preferred_success and avoid_success,
                "preferred_stores_used": preferred_stores_used,
                "avoided_stores_used": avoided_stores_used,
                "metro_used_when_preferred": "metro_ca" in preferred_stores_used,
                "walmart_avoided": "walmart_ca" not in avoided_stores_used,
                "preferred_total_cost": preferred_result.get("savings_analysis", {}).get("optimized_cost", 0),
                "avoid_total_cost": avoid_result.get("savings_analysis", {}).get("optimized_cost", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_large_shopping_list(self) -> Dict[str, Any]:
        """Test with a large shopping list (stress test)."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create large ingredient list
            large_ingredients = []
            test_data = create_large_test_ingredients()
            
            for item in test_data:
                large_ingredients.append(Ingredient(
                    name=item["name"],
                    quantity=item["quantity"],
                    unit=UnitType(item["unit"]),
                    category=item.get("category")
                ))
            
            optimizer = OptimizerAgent()
            
            # Test optimization with large list
            start_time = datetime.now()
            result = await optimizer.optimize_shopping_list(
                ingredients=large_ingredients,
                strategy="adaptive"
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Validate performance
            success = (
                result.get("success", False) and
                processing_time < 60.0 and  # Should complete within 60 seconds
                len(result.get("recommended_strategy", [])) <= 5  # Reasonable number of stores
            )
            
            return {
                "success": success,
                "ingredient_count": len(large_ingredients),
                "processing_time": processing_time,
                "total_trips": len(result.get("recommended_strategy", [])),
                "total_cost": result.get("savings_analysis", {}).get("optimized_cost", 0),
                "coverage": result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "items_matched": result.get("optimization_summary", {}).get("total_items", 0),
                "unmatched_count": len(result.get("unmatched_ingredients", []))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_quality_optimization(self) -> Dict[str, Any]:
        """Test quality-first optimization strategy."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="organic milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="whole grain bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="free range eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            optimizer = OptimizerAgent()
            
            # Test quality-first strategy
            quality_criteria = OptimizationCriteria(
                quality_threshold=0.8,  # High quality threshold
                max_stores=3
            )
            
            quality_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                criteria=quality_criteria,
                strategy="quality_first"
            )
            
            # Compare with cost-only strategy
            cost_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                strategy="cost_only"
            )
            
            # Validate quality optimization
            quality_success = quality_result.get("success", False)
            cost_success = cost_result.get("success", False)
            
            quality_cost = quality_result.get("savings_analysis", {}).get("optimized_cost", 0) if quality_success else 0
            cost_only_cost = cost_result.get("savings_analysis", {}).get("optimized_cost", 0) if cost_success else 0
            
            # Quality strategy should typically cost more than cost-only
            cost_difference = float(quality_cost) - float(cost_only_cost) if both_success := (quality_success and cost_success) else 0
            
            return {
                "success": quality_success and cost_success,
                "quality_cost": quality_cost,
                "cost_only_cost": cost_only_cost,
                "cost_difference": cost_difference,
                "quality_premium": cost_difference > 0 if both_success else False,
                "quality_coverage": quality_result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "quality_trips": len(quality_result.get("recommended_strategy", [])),
                "cost_trips": len(cost_result.get("recommended_strategy", []))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_time_efficiency(self) -> Dict[str, Any]:
        """Test time-efficient optimization strategy."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="bananas", quantity=6.0, unit=UnitType.PIECES, category="produce")
            ]
            
            optimizer = OptimizerAgent()
            
            # Test time-efficient strategy
            time_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                strategy="time_efficient"
            )
            
            # Compare with cost-only strategy
            cost_result = await optimizer.optimize_shopping_list(
                ingredients=ingredients,
                strategy="cost_only"
            )
            
            # Validate time optimization
            time_success = time_result.get("success", False)
            cost_success = cost_result.get("success", False)
            
            time_total = time_result.get("optimization_summary", {}).get("total_time", 999) if time_success else 999
            cost_total = cost_result.get("optimization_summary", {}).get("total_time", 999) if cost_success else 999
            
            # Time-efficient should generally be faster
            time_savings = cost_total - time_total if time_success and cost_success else 0
            
            return {
                "success": time_success and cost_success,
                "time_efficient_duration": time_total,
                "cost_only_duration": cost_total,
                "time_savings": time_savings,
                "time_is_faster": time_savings > 0,
                "time_stores": len(time_result.get("recommended_strategy", [])),
                "cost_stores": len(cost_result.get("recommended_strategy", [])),
                "time_cost": time_result.get("savings_analysis", {}).get("optimized_cost", 0),
                "convenience_achieved": len(time_result.get("recommended_strategy", [])) <= 2
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_savings_estimation(self) -> Dict[str, Any]:
        """Test savings estimation functionality."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.0, unit=UnitType.KG, category="meat")
            ]
            
            optimizer = OptimizerAgent()
            
            # Test savings estimation
            estimate = await optimizer.estimate_savings(
                ingredients=ingredients,
                current_shopping_method="convenience"
            )
            
            # Validate estimation
            success = (
                "current_cost" in estimate and
                "optimized_cost" in estimate and
                "potential_savings" in estimate and
                "savings_percentage" in estimate and
                "recommendation" in estimate
            )
            
            # Validate logical consistency
            current_cost = estimate.get("current_cost", 0)
            optimized_cost = estimate.get("optimized_cost", 0)
            potential_savings = estimate.get("potential_savings", 0)
            
            logical_consistency = abs(float(current_cost) - float(optimized_cost) - float(potential_savings)) < 0.01
            
            return {
                "success": success and logical_consistency,
                "current_cost": current_cost,
                "optimized_cost": optimized_cost,
                "potential_savings": potential_savings,
                "savings_percentage": estimate.get("savings_percentage", 0),
                "recommendation": estimate.get("recommendation", "unknown"),
                "logical_consistency": logical_consistency,
                "savings_positive": float(potential_savings) >= 0
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from decimal import Decimal
            
            optimizer = OptimizerAgent()
            error_tests = {}
            
            # Test 1: Empty ingredient list
            try:
                empty_result = await optimizer.optimize_shopping_list(ingredients=[])
                error_tests["empty_ingredients"] = not empty_result.get("success", True)
            except Exception:
                error_tests["empty_ingredients"] = True
            
            # Test 2: Invalid strategy
            try:
                ingredients = [Ingredient(name="milk", quantity=1.0, unit=UnitType.L)]
                invalid_result = await optimizer.optimize_shopping_list(
                    ingredients=ingredients,
                    strategy="invalid_strategy"
                )
                error_tests["invalid_strategy"] = not invalid_result.get("success", True)
            except Exception:
                error_tests["invalid_strategy"] = True
            
            # Test 3: Impossible budget constraint
            try:
                impossible_criteria = OptimizationCriteria(max_budget=Decimal("0.01"))
                impossible_result = await optimizer.optimize_shopping_list(
                    ingredients=ingredients,
                    criteria=impossible_criteria
                )
                error_tests["impossible_budget"] = not impossible_result.get("success", True)
            except Exception:
                error_tests["impossible_budget"] = True
            
            # Test 4: Non-existent ingredients
            try:
                fake_ingredients = [
                    Ingredient(name="unicorn_meat", quantity=1.0, unit=UnitType.KG),
                    Ingredient(name="dragon_eggs", quantity=3.0, unit=UnitType.PIECES)
                ]
                fake_result = await optimizer.optimize_shopping_list(ingredients=fake_ingredients)
                # Should succeed but with low coverage
                error_tests["fake_ingredients"] = (
                    fake_result.get("success", False) and
                    fake_result.get("optimization_summary", {}).get("coverage_percentage", 100) < 50
                )
            except Exception:
                error_tests["fake_ingredients"] = False
            
            success_count = sum(1 for passed in error_tests.values() if passed)
            
            return {
                "success": success_count >= 3,  # At least 3 error handling tests should pass
                "tests_passed": success_count,
                "total_tests": len(error_tests),
                "error_test_results": error_tests
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance with various scenarios."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            optimizer = OptimizerAgent()
            benchmarks = {}
            
            # Benchmark 1: Small list (3 ingredients)
            small_ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES)
            ]
            
            start_time = datetime.now()
            small_result = await optimizer.optimize_shopping_list(
                ingredients=small_ingredients,
                strategy="adaptive"
            )
            small_time = (datetime.now() - start_time).total_seconds()
            benchmarks["small_list"] = {
                "time": small_time,
                "success": small_result.get("success", False),
                "ingredient_count": len(small_ingredients)
            }
            
            # Benchmark 2: Medium list (10 ingredients)
            medium_ingredients = []
            test_data = create_test_ingredients()
            for item in test_data:
                medium_ingredients.append(Ingredient(
                    name=item["name"],
                    quantity=item["quantity"],
                    unit=UnitType(item["unit"])
                ))
            
            start_time = datetime.now()
            medium_result = await optimizer.optimize_shopping_list(
                ingredients=medium_ingredients,
                strategy="adaptive"
            )
            medium_time = (datetime.now() - start_time).total_seconds()
            benchmarks["medium_list"] = {
                "time": medium_time,
                "success": medium_result.get("success", False),
                "ingredient_count": len(medium_ingredients)
            }
            
            # Benchmark 3: Strategy comparison speed
            start_time = datetime.now()
            comparison_result = await optimizer.compare_strategies(
                ingredients=small_ingredients,
                strategies=["cost_only", "convenience", "balanced"]
            )
            comparison_time = (datetime.now() - start_time).total_seconds()
            benchmarks["strategy_comparison"] = {
                "time": comparison_time,
                "success": len([r for r in comparison_result.values() if r.get("success")]) >= 2,
                "strategies_tested": len(comparison_result)
            }
            
            # Validate performance criteria
            performance_acceptable = (
                small_time < 10.0 and  # Small list should complete in <10 seconds
                medium_time < 30.0 and  # Medium list should complete in <30 seconds
                comparison_time < 45.0  # Strategy comparison should complete in <45 seconds
            )
            
            return {
                "success": performance_acceptable,
                "benchmarks": benchmarks,
                "small_list_time": small_time,
                "medium_list_time": medium_time,
                "comparison_time": comparison_time,
                "performance_acceptable": performance_acceptable,
                "avg_time_per_ingredient": {
                    "small": small_time / len(small_ingredients),
                    "medium": medium_time / len(medium_ingredients)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_integration_tests(self) -> Dict[str, Any]:
        """Test integration with other system components."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.agents.matcher_agent import MatcherAgent
            from agentic_grocery_price_scanner.llm_client.ollama_client import OllamaClient
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Test with actual components (if available)
            integration_results = {}
            
            # Test 1: OptimizerAgent with MatcherAgent integration
            try:
                matcher_agent = MatcherAgent()
                optimizer = OptimizerAgent(matcher_agent=matcher_agent)
                
                ingredients = [
                    Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy")
                ]
                
                result = await optimizer.optimize_shopping_list(
                    ingredients=ingredients,
                    strategy="adaptive"
                )
                
                integration_results["matcher_integration"] = result.get("success", False)
            except Exception as e:
                integration_results["matcher_integration"] = False
                integration_results["matcher_error"] = str(e)
            
            # Test 2: LLM Client integration
            try:
                llm_client = OllamaClient()
                health_check = await llm_client.health_check()
                integration_results["llm_integration"] = health_check.get("service_available", False)
            except Exception as e:
                integration_results["llm_integration"] = False
                integration_results["llm_error"] = str(e)
            
            # Test 3: Analytics integration
            try:
                optimizer = OptimizerAgent()
                analytics = optimizer.get_optimization_analytics()
                integration_results["analytics_integration"] = "optimization_stats" in analytics
            except Exception as e:
                integration_results["analytics_integration"] = False
                integration_results["analytics_error"] = str(e)
            
            successful_integrations = sum(1 for result in integration_results.values() if result is True)
            
            return {
                "success": successful_integrations >= 2,  # At least 2 integrations should work
                "successful_integrations": successful_integrations,
                "integration_results": integration_results,
                "matcher_available": integration_results.get("matcher_integration", False),
                "llm_available": integration_results.get("llm_integration", False),
                "analytics_available": integration_results.get("analytics_integration", False)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_workflow_validation(self) -> Dict[str, Any]:
        """Test LangGraph workflow validation."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            optimizer = OptimizerAgent()
            
            # Test workflow components
            workflow_tests = {}
            
            # Test 1: Workflow compilation
            try:
                # Check if workflow is properly compiled
                workflow_tests["workflow_compiled"] = optimizer.workflow is not None
            except Exception as e:
                workflow_tests["workflow_compiled"] = False
            
            # Test 2: State management
            try:
                # Test with progress tracking
                progress_messages = []
                def progress_callback(message: str):
                    progress_messages.append(message)
                
                ingredients = [Ingredient(name="milk", quantity=1.0, unit=UnitType.L)]
                
                result = await optimizer.execute({
                    "ingredients": ingredients,
                    "strategy": "adaptive",
                    "progress_callback": progress_callback
                })
                
                workflow_tests["state_management"] = len(progress_messages) > 0 and result.get("success", False)
            except Exception as e:
                workflow_tests["state_management"] = False
            
            # Test 3: Checkpointing
            try:
                # Test workflow checkpointing functionality
                workflow_tests["checkpointing"] = optimizer.checkpointer is not None
            except Exception as e:
                workflow_tests["checkpointing"] = False
            
            # Test 4: Metadata tracking
            try:
                ingredients = [Ingredient(name="milk", quantity=1.0, unit=UnitType.L)]
                result = await optimizer.optimize_shopping_list(ingredients=ingredients)
                
                metadata = result.get("optimization_metadata", {})
                workflow_tests["metadata_tracking"] = (
                    "start_time" in metadata and
                    "end_time" in metadata and
                    "stages_completed" in metadata
                )
            except Exception as e:
                workflow_tests["metadata_tracking"] = False
            
            successful_tests = sum(1 for passed in workflow_tests.values() if passed)
            
            return {
                "success": successful_tests >= 3,  # At least 3 workflow tests should pass
                "successful_tests": successful_tests,
                "total_tests": len(workflow_tests),
                "workflow_tests": workflow_tests
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_multi_criteria_scoring(self) -> Dict[str, Any]:
        """Test multi-criteria scoring functionality."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create test ingredients
            ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            optimizer = OptimizerAgent()
            scoring_tests = {}
            
            # Test different optimization strategies and their scoring
            strategies = ["cost_only", "convenience", "quality_first", "time_efficient"]
            
            strategy_results = {}
            for strategy in strategies:
                try:
                    result = await optimizer.optimize_shopping_list(
                        ingredients=ingredients,
                        strategy=strategy
                    )
                    
                    if result.get("success", False):
                        strategy_results[strategy] = {
                            "cost": result["savings_analysis"]["optimized_cost"],
                            "stores": result["optimization_summary"]["total_stores"],
                            "time": result["optimization_summary"]["total_time"],
                            "coverage": result["optimization_summary"]["coverage_percentage"]
                        }
                except Exception:
                    continue
            
            # Validate scoring logic
            scoring_tests["strategy_differentiation"] = len(strategy_results) >= 3
            
            # Test criteria weights affect results
            if len(strategy_results) >= 2:
                costs = [data["cost"] for data in strategy_results.values()]
                stores = [data["stores"] for data in strategy_results.values()]
                times = [data["time"] for data in strategy_results.values()]
                
                # Different strategies should produce different results
                scoring_tests["cost_variation"] = len(set(costs)) > 1
                scoring_tests["store_variation"] = len(set(stores)) > 1
                scoring_tests["time_variation"] = len(set(times)) > 1
                
                # Cost-only should generally be cheapest
                if "cost_only" in strategy_results:
                    cost_only_cost = float(strategy_results["cost_only"]["cost"])
                    other_costs = [float(data["cost"]) for strategy, data in strategy_results.items() if strategy != "cost_only"]
                    scoring_tests["cost_optimization"] = all(cost_only_cost <= cost for cost in other_costs)
                
                # Convenience should generally use fewer stores
                if "convenience" in strategy_results:
                    convenience_stores = strategy_results["convenience"]["stores"]
                    other_stores = [data["stores"] for strategy, data in strategy_results.items() if strategy != "convenience"]
                    scoring_tests["convenience_optimization"] = all(convenience_stores <= stores for stores in other_stores)
            
            successful_scoring_tests = sum(1 for passed in scoring_tests.values() if passed)
            
            return {
                "success": successful_scoring_tests >= 4,
                "successful_tests": successful_scoring_tests,
                "total_tests": len(scoring_tests),
                "scoring_tests": scoring_tests,
                "strategy_results": strategy_results,
                "strategies_tested": len(strategy_results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_real_world_scenarios(self) -> Dict[str, Any]:
        """Test with realistic shopping scenarios."""
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from decimal import Decimal
            
            optimizer = OptimizerAgent()
            scenario_results = {}
            
            # Scenario 1: Weekly family shopping
            family_ingredients = [
                Ingredient(name="milk", quantity=4.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=2.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=24.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="chicken breast", quantity=2.0, unit=UnitType.KG, category="meat"),
                Ingredient(name="ground beef", quantity=1.0, unit=UnitType.KG, category="meat"),
                Ingredient(name="rice", quantity=2.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="bananas", quantity=12.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="apples", quantity=8.0, unit=UnitType.PIECES, category="produce")
            ]
            
            family_criteria = OptimizationCriteria(
                max_budget=Decimal("150.00"),
                max_stores=3,
                quality_threshold=0.7
            )
            
            family_result = await optimizer.optimize_shopping_list(
                ingredients=family_ingredients,
                criteria=family_criteria,
                strategy="balanced"
            )
            
            scenario_results["family_shopping"] = {
                "success": family_result.get("success", False),
                "total_cost": family_result.get("savings_analysis", {}).get("optimized_cost", 0),
                "stores_used": len(family_result.get("recommended_strategy", [])),
                "coverage": family_result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "within_budget": (
                    float(family_result.get("savings_analysis", {}).get("optimized_cost", 999)) <= 150.0
                    if family_result.get("success", False) else False
                )
            }
            
            # Scenario 2: Quick convenience shopping
            quick_ingredients = [
                Ingredient(name="milk", quantity=1.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=6.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            quick_criteria = OptimizationCriteria(
                max_stores=1,
                preferred_stores=["metro_ca"]
            )
            
            quick_result = await optimizer.optimize_shopping_list(
                ingredients=quick_ingredients,
                criteria=quick_criteria,
                strategy="convenience"
            )
            
            scenario_results["quick_shopping"] = {
                "success": quick_result.get("success", False),
                "single_store": len(quick_result.get("recommended_strategy", [])) <= 1,
                "total_time": quick_result.get("optimization_summary", {}).get("total_time", 999),
                "coverage": quick_result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "time_efficient": (
                    quick_result.get("optimization_summary", {}).get("total_time", 999) <= 30
                    if quick_result.get("success", False) else False
                )
            }
            
            # Scenario 3: Budget-conscious shopping
            budget_ingredients = [
                Ingredient(name="rice", quantity=2.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="pasta", quantity=1.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="canned tomatoes", quantity=4.0, unit=UnitType.PIECES, category="pantry"),
                Ingredient(name="onions", quantity=2.0, unit=UnitType.KG, category="produce"),
                Ingredient(name="carrots", quantity=1.0, unit=UnitType.KG, category="produce")
            ]
            
            budget_criteria = OptimizationCriteria(
                max_budget=Decimal("30.00"),
                max_stores=5  # Allow many stores for cost optimization
            )
            
            budget_result = await optimizer.optimize_shopping_list(
                ingredients=budget_ingredients,
                criteria=budget_criteria,
                strategy="cost_only"
            )
            
            scenario_results["budget_shopping"] = {
                "success": budget_result.get("success", False),
                "total_cost": budget_result.get("savings_analysis", {}).get("optimized_cost", 0),
                "within_budget": (
                    float(budget_result.get("savings_analysis", {}).get("optimized_cost", 999)) <= 30.0
                    if budget_result.get("success", False) else False
                ),
                "coverage": budget_result.get("optimization_summary", {}).get("coverage_percentage", 0),
                "cost_per_item": (
                    float(budget_result.get("savings_analysis", {}).get("optimized_cost", 0)) / len(budget_ingredients)
                    if budget_result.get("success", False) else 0
                )
            }
            
            # Overall scenario validation
            successful_scenarios = sum(
                1 for scenario in scenario_results.values() 
                if scenario.get("success", False)
            )
            
            return {
                "success": successful_scenarios >= 2,  # At least 2 scenarios should succeed
                "successful_scenarios": successful_scenarios,
                "total_scenarios": len(scenario_results),
                "scenario_results": scenario_results,
                "family_within_budget": scenario_results["family_shopping"].get("within_budget", False),
                "quick_single_store": scenario_results["quick_shopping"].get("single_store", False),
                "budget_affordable": scenario_results["budget_shopping"].get("within_budget", False)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        
        # Calculate overall metrics
        total_duration = sum(result.get("duration", 0) for result in self.test_results.values())
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        # Categorize test results
        passed_tests = [name for name, result in self.test_results.items() if result.get("success", False)]
        failed_tests = [name for name, result in self.test_results.items() if not result.get("success", False)]
        
        # Performance analysis
        performance_tests = {
            name: result for name, result in self.test_results.items()
            if name in ["performance_benchmarks", "large_shopping_list"]
        }
        
        # Generate recommendations
        recommendations = []
        
        if successful_tests / total_tests < 0.8:
            recommendations.append("Consider investigating failing tests for system stability")
        
        if avg_duration > 10.0:
            recommendations.append("Consider optimizing performance for faster execution")
        
        if "integration_tests" in failed_tests:
            recommendations.append("Check integration with external dependencies")
        
        if not recommendations:
            recommendations.append("All tests performing well - system is ready for production")
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "average_duration": avg_duration
            },
            "test_results": {
                "passed": passed_tests,
                "failed": failed_tests
            },
            "performance_analysis": {
                test_name: {
                    "duration": result.get("duration", 0),
                    "details": result.get("details", {})
                }
                for test_name, result in performance_tests.items()
            },
            "detailed_results": self.test_results,
            "recommendations": recommendations,
            "overall_status": "PASSED" if successful_tests / total_tests >= 0.8 else "FAILED",
            "generated_at": datetime.now().isoformat()
        }
        
        return report

async def main():
    """Run the comprehensive OptimizerAgent test suite."""
    print("🚀 OptimizerAgent Comprehensive Test Suite")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run all tests
        tester = OptimizerAgentTester()
        report = await tester.run_all_tests()
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        summary = report["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['successful_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Average Duration: {summary['average_duration']:.2f}s per test")
        
        print(f"\n🎯 Overall Status: {report['overall_status']}")
        
        # Show failed tests
        if report["test_results"]["failed"]:
            print("\n❌ Failed Tests:")
            for test_name in report["test_results"]["failed"]:
                error = report["detailed_results"][test_name].get("error", "Unknown error")
                print(f"   • {test_name}: {error}")
        
        # Show recommendations
        if report["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        # Performance highlights
        perf_analysis = report["performance_analysis"]
        if perf_analysis:
            print("\n⚡ Performance Highlights:")
            for test_name, perf_data in perf_analysis.items():
                duration = perf_data["duration"]
                details = perf_data.get("details", {})
                
                if test_name == "performance_benchmarks" and details.get("success", False):
                    benchmarks = details.get("benchmarks", {})
                    print(f"   • Small list: {benchmarks.get('small_list', {}).get('time', 0):.2f}s")
                    print(f"   • Medium list: {benchmarks.get('medium_list', {}).get('time', 0):.2f}s")
                    print(f"   • Strategy comparison: {benchmarks.get('strategy_comparison', {}).get('time', 0):.2f}s")
                
                elif test_name == "large_shopping_list" and details.get("success", False):
                    ingredient_count = details.get("ingredient_count", 0)
                    processing_time = details.get("processing_time", 0)
                    print(f"   • Large list ({ingredient_count} ingredients): {processing_time:.2f}s")
        
        # Save detailed report
        report_filename = f"optimizer_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Final status
        if report["overall_status"] == "PASSED":
            print("\n🎉 OptimizerAgent is ready for production use!")
        else:
            print("\n⚠️  OptimizerAgent needs attention before production deployment.")
        
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        logger.exception("Test suite failed")

if __name__ == "__main__":
    asyncio.run(main())