"""
OptimizerAgent Demonstration Script
Interactive demonstration of multi-store shopping optimization with real scenarios.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import List, Dict, Any
from datetime import datetime
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizerDemo:
    """Interactive demonstration of OptimizerAgent capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.scenarios = {
            "1": {
                "name": "🏠 Weekly Family Shopping",
                "description": "Large family shopping list with budget constraints",
                "method": self.demo_family_shopping
            },
            "2": {
                "name": "⚡ Quick Convenience Shopping", 
                "description": "Fast shopping for essential items at preferred store",
                "method": self.demo_quick_shopping
            },
            "3": {
                "name": "💰 Budget-Conscious Shopping",
                "description": "Cost optimization with strict budget limits",
                "method": self.demo_budget_shopping
            },
            "4": {
                "name": "⭐ Quality-First Shopping",
                "description": "Premium products with quality prioritization",
                "method": self.demo_quality_shopping
            },
            "5": {
                "name": "⚖️  Strategy Comparison",
                "description": "Compare all optimization strategies side-by-side",
                "method": self.demo_strategy_comparison
            },
            "6": {
                "name": "🎯 Custom Shopping List",
                "description": "Enter your own ingredients for optimization",
                "method": self.demo_custom_shopping
            },
            "7": {
                "name": "📊 Savings Estimation",
                "description": "Estimate potential savings from optimization",
                "method": self.demo_savings_estimation
            },
            "8": {
                "name": "🔧 Batch Recipe Processing",
                "description": "Process multiple recipes for meal planning",
                "method": self.demo_batch_recipes
            }
        }
    
    def display_banner(self):
        """Display the demo banner."""
        print("🛒" + "=" * 78 + "🛒")
        print("🎯                    OptimizerAgent Interactive Demo                    🎯")
        print("🛒" + "=" * 78 + "🛒")
        print("🧠 Intelligent Multi-Store Shopping Optimization with AI")
        print("⚡ LangGraph Workflow • 🤖 Local LLMs • 💡 Smart Decision Making")
        print("🛒" + "=" * 78 + "🛒")
        print()
    
    def display_menu(self):
        """Display the demo menu."""
        print("📋 Choose a demonstration scenario:")
        print()
        
        for key, scenario in self.scenarios.items():
            print(f"   {key}. {scenario['name']}")
            print(f"      {scenario['description']}")
            print()
        
        print("   0. Exit Demo")
        print()
    
    def display_progress(self, message: str):
        """Display progress message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    async def demo_family_shopping(self):
        """Demonstrate weekly family shopping optimization."""
        print("\n🏠 Weekly Family Shopping Demo")
        print("=" * 50)
        print("Scenario: Large family needs groceries for the week")
        print("Challenge: Balance cost, quality, and convenience")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create family shopping list
            family_ingredients = [
                Ingredient(name="milk", quantity=4.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="whole grain bread", quantity=2.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="free range eggs", quantity=24.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="chicken breast", quantity=2.0, unit=UnitType.KG, category="meat"),
                Ingredient(name="ground beef", quantity=1.5, unit=UnitType.KG, category="meat"),
                Ingredient(name="salmon fillet", quantity=800.0, unit=UnitType.G, category="meat"),
                Ingredient(name="bananas", quantity=12.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="apples", quantity=10.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="carrots", quantity=2.0, unit=UnitType.KG, category="produce"),
                Ingredient(name="broccoli", quantity=3.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="rice", quantity=2.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="pasta", quantity=1.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="olive oil", quantity=500.0, unit=UnitType.ML, category="cooking"),
                Ingredient(name="cheddar cheese", quantity=400.0, unit=UnitType.G, category="dairy"),
                Ingredient(name="yogurt", quantity=6.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            print(f"📋 Shopping List: {len(family_ingredients)} items")
            for ingredient in family_ingredients:
                print(f"   • {ingredient.name} - {ingredient.quantity} {ingredient.unit}")
            
            print("\n🎯 Optimization Criteria:")
            family_criteria = OptimizationCriteria(
                max_budget=Decimal("200.00"),
                max_stores=3,
                quality_threshold=0.7,
                preferred_stores=["metro_ca"],
                bulk_buying_ok=True
            )
            
            print(f"   • Budget: ${family_criteria.max_budget}")
            print(f"   • Max stores: {family_criteria.max_stores}")
            print(f"   • Quality threshold: {family_criteria.quality_threshold}")
            print(f"   • Preferred stores: {family_criteria.preferred_stores}")
            
            # Initialize optimizer with progress callback
            optimizer = OptimizerAgent()
            
            def progress_callback(message: str):
                self.display_progress(message)
            
            print("\n🔄 Running optimization...")
            start_time = datetime.now()
            
            result = await optimizer.optimize_shopping_list(
                ingredients=family_ingredients,
                criteria=family_criteria,
                strategy="balanced"
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False):
                self.display_optimization_results(result, processing_time, "Family Shopping")
                
                # Show detailed shopping plan
                print("\n🗺️  Detailed Shopping Plan:")
                for i, trip in enumerate(result["recommended_strategy"], 1):
                    print(f"\n   Trip {i}: {trip['store_name']} ({trip['store_id']})")
                    print(f"   📍 Travel time: {trip['travel_time']} minutes")
                    print(f"   🛍️  Items to buy: {trip['total_items']}")
                    print(f"   💰 Trip cost: ${trip['total_cost']:.2f}")
                    print(f"   ⏱️  Shopping time: {trip['estimated_time']} minutes")
                    
                    # Show top items for this trip
                    print("   📦 Key items:")
                    for j, product_info in enumerate(trip['products'][:5], 1):
                        product = product_info['product']
                        ingredient = product_info['ingredient']
                        price_display = f"${product['price']}"
                        if product['on_sale'] and product['sale_price']:
                            price_display = f"${product['sale_price']} (was ${product['price']}) 🏷️"
                        print(f"      {j}. {ingredient['name']} → {product['name']} - {price_display}")
                    
                    if len(trip['products']) > 5:
                        print(f"      ... and {len(trip['products']) - 5} more items")
                
                # Show savings analysis
                savings = result["savings_analysis"]
                print(f"\n💰 Savings Analysis:")
                print(f"   Single-store cost: ${savings['convenience_cost']:.2f}")
                print(f"   Optimized cost: ${savings['optimized_cost']:.2f}")
                print(f"   Total savings: ${savings['total_savings']:.2f} ({savings['savings_percentage']:.1f}%)")
                
                # Show budget performance
                budget_used = float(savings['optimized_cost'])
                budget_limit = float(family_criteria.max_budget)
                budget_remaining = budget_limit - budget_used
                
                print(f"\n📊 Budget Performance:")
                print(f"   Budget limit: ${budget_limit:.2f}")
                print(f"   Amount used: ${budget_used:.2f}")
                print(f"   Remaining: ${budget_remaining:.2f}")
                print(f"   Budget utilization: {budget_used/budget_limit:.1%}")
                
                if budget_remaining >= 0:
                    print("   ✅ Within budget!")
                else:
                    print("   ⚠️  Over budget - consider adjustments")
                    
            else:
                print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Family shopping demo failed")
    
    async def demo_quick_shopping(self):
        """Demonstrate quick convenience shopping."""
        print("\n⚡ Quick Convenience Shopping Demo")
        print("=" * 50)
        print("Scenario: Need essentials quickly from preferred store")
        print("Challenge: Minimize time while ensuring availability")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create quick shopping list
            quick_ingredients = [
                Ingredient(name="milk", quantity=1.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="eggs", quantity=6.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="bananas", quantity=4.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="yogurt", quantity=2.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            print(f"📋 Quick Shopping List: {len(quick_ingredients)} items")
            for ingredient in quick_ingredients:
                print(f"   • {ingredient.name}")
            
            # Quick shopping criteria
            quick_criteria = OptimizationCriteria(
                max_stores=1,
                preferred_stores=["metro_ca"],
                quality_threshold=0.6
            )
            
            print("\n🎯 Optimization Criteria:")
            print(f"   • Max stores: {quick_criteria.max_stores} (convenience priority)")
            print(f"   • Preferred store: {quick_criteria.preferred_stores[0]}")
            print(f"   • Quality threshold: {quick_criteria.quality_threshold} (flexible)")
            
            optimizer = OptimizerAgent()
            
            print("\n🔄 Finding fastest shopping option...")
            start_time = datetime.now()
            
            result = await optimizer.optimize_shopping_list(
                ingredients=quick_ingredients,
                criteria=quick_criteria,
                strategy="convenience"
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False) and result["recommended_strategy"]:
                trip = result["recommended_strategy"][0]
                
                print("\n✅ Quick Shopping Solution Found!")
                print(f"🏪 Store: {trip['store_name']}")
                print(f"📍 Travel time: {trip['travel_time']} minutes")
                print(f"🛍️  Items available: {trip['total_items']}/{len(quick_ingredients)}")
                print(f"💰 Total cost: ${trip['total_cost']:.2f}")
                print(f"⏱️  Total time: {trip['estimated_time']} minutes")
                
                # Show coverage analysis
                coverage = result["optimization_summary"]["coverage_percentage"]
                print(f"📊 Coverage: {coverage:.1f}%")
                
                if coverage == 100:
                    print("   ✅ All items available at preferred store!")
                else:
                    unmatched = result.get("unmatched_ingredients", [])
                    print(f"   ⚠️  Missing items: {', '.join(ing['name'] for ing in unmatched)}")
                    print("   💡 Consider adding another quick stop or substitutions")
                
                # Show time efficiency
                print(f"\n⚡ Time Efficiency Analysis:")
                print(f"   Travel: {trip['travel_time']} minutes")
                print(f"   Shopping: {trip['estimated_time'] - trip['travel_time']} minutes")
                print(f"   Total: {trip['estimated_time']} minutes")
                
                if trip['estimated_time'] <= 20:
                    print("   🚀 Super fast! Perfect for quick trips")
                elif trip['estimated_time'] <= 30:
                    print("   👍 Good speed for convenience shopping")
                else:
                    print("   🐌 Consider fewer items or closer store")
                
                # Show items to buy
                print(f"\n🛍️  Shopping List for {trip['store_name']}:")
                for i, product_info in enumerate(trip['products'], 1):
                    product = product_info['product']
                    ingredient = product_info['ingredient']
                    print(f"   {i}. {ingredient['name']} → {product['name']}")
                    print(f"      💰 ${product['price']} | 📍 {product['store_id']}")
                    if product['on_sale']:
                        print("      🏷️  ON SALE!")
                        
            else:
                print(f"❌ Quick shopping optimization failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Quick shopping demo failed")
    
    async def demo_budget_shopping(self):
        """Demonstrate budget-conscious shopping optimization."""
        print("\n💰 Budget-Conscious Shopping Demo")
        print("=" * 50)
        print("Scenario: Maximum value shopping with strict budget")
        print("Challenge: Get most items within budget constraints")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from decimal import Decimal
            
            # Create budget shopping list - focus on staples
            budget_ingredients = [
                Ingredient(name="rice", quantity=2.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="pasta", quantity=1.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="canned tomatoes", quantity=4.0, unit=UnitType.PIECES, category="pantry"),
                Ingredient(name="dried beans", quantity=1.0, unit=UnitType.KG, category="pantry"),
                Ingredient(name="onions", quantity=2.0, unit=UnitType.KG, category="produce"),
                Ingredient(name="carrots", quantity=2.0, unit=UnitType.KG, category="produce"),
                Ingredient(name="potatoes", quantity=3.0, unit=UnitType.KG, category="produce"),
                Ingredient(name="cooking oil", quantity=1.0, unit=UnitType.L, category="cooking"),
                Ingredient(name="flour", quantity=2.0, unit=UnitType.KG, category="baking"),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="peanut butter", quantity=1.0, unit=UnitType.PIECES, category="pantry")
            ]
            
            budget_limit = Decimal("50.00")
            
            print(f"📋 Budget Shopping List: {len(budget_ingredients)} staple items")
            print("Focus on nutritious, versatile ingredients for maximum value")
            print()
            
            for ingredient in budget_ingredients:
                print(f"   • {ingredient.name} - {ingredient.quantity} {ingredient.unit}")
            
            print(f"\n💰 Strict Budget: ${budget_limit}")
            print("Goal: Maximize items within budget using cost optimization")
            
            # Budget optimization criteria
            budget_criteria = OptimizationCriteria(
                max_budget=budget_limit,
                max_stores=5,  # Allow multiple stores for best prices
                quality_threshold=0.4,  # Lower quality threshold for budget shopping
                bulk_buying_ok=True,
                sale_priority=0.5  # High priority on sale items
            )
            
            optimizer = OptimizerAgent()
            
            print("\n🔄 Finding cheapest options across all stores...")
            start_time = datetime.now()
            
            result = await optimizer.optimize_shopping_list(
                ingredients=budget_ingredients,
                criteria=budget_criteria,
                strategy="cost_only"
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False):
                self.display_optimization_results(result, processing_time, "Budget Shopping")
                
                # Detailed budget analysis
                optimized_cost = float(result["savings_analysis"]["optimized_cost"])
                convenience_cost = float(result["savings_analysis"]["convenience_cost"])
                savings_amount = convenience_cost - optimized_cost
                
                print(f"\n💰 Detailed Budget Analysis:")
                print(f"   Budget limit: ${float(budget_limit):.2f}")
                print(f"   Optimized cost: ${optimized_cost:.2f}")
                print(f"   Budget remaining: ${float(budget_limit) - optimized_cost:.2f}")
                print(f"   Budget utilization: {optimized_cost / float(budget_limit):.1%}")
                print(f"   Savings vs convenience shopping: ${savings_amount:.2f} ({savings_amount/convenience_cost:.1%})")
                
                if optimized_cost <= float(budget_limit):
                    print("   ✅ Within budget!")
                    remaining = float(budget_limit) - optimized_cost
                    if remaining > 5:
                        print(f"   💡 Consider adding ${remaining:.2f} worth of extras")
                else:
                    overage = optimized_cost - float(budget_limit)
                    print(f"   ⚠️  Over budget by ${overage:.2f}")
                    print("   💡 Consider removing some items or finding cheaper alternatives")
                
                # Show cost per item analysis
                items_in_budget = sum(trip['total_items'] for trip in result["recommended_strategy"])
                cost_per_item = optimized_cost / items_in_budget if items_in_budget > 0 else 0
                
                print(f"\n📊 Value Analysis:")
                print(f"   Items in budget: {items_in_budget}/{len(budget_ingredients)}")
                print(f"   Average cost per item: ${cost_per_item:.2f}")
                print(f"   Coverage: {result['optimization_summary']['coverage_percentage']:.1f}%")
                
                # Show store breakdown by cost
                print(f"\n🏪 Store Cost Breakdown:")
                total_cost = 0
                for i, trip in enumerate(result["recommended_strategy"], 1):
                    trip_cost = float(trip['total_cost'])
                    total_cost += trip_cost
                    percentage = trip_cost / optimized_cost * 100 if optimized_cost > 0 else 0
                    
                    print(f"   {i}. {trip['store_name']}")
                    print(f"      Cost: ${trip_cost:.2f} ({percentage:.1f}% of total)")
                    print(f"      Items: {trip['total_items']}")
                    print(f"      Avg per item: ${trip_cost / trip['total_items']:.2f}")
                
                # Show deals and sales
                print(f"\n🏷️  Deals & Sales Found:")
                sale_count = 0
                total_sale_savings = 0
                
                for trip in result["recommended_strategy"]:
                    for product_info in trip['products']:
                        product = product_info['product']
                        if product['on_sale'] and product['sale_price']:
                            sale_savings = float(product['price']) - float(product['sale_price'])
                            total_sale_savings += sale_savings
                            sale_count += 1
                            print(f"   🏷️  {product['name']}: ${product['sale_price']} (save ${sale_savings:.2f})")
                
                if sale_count > 0:
                    print(f"   Total sale savings: ${total_sale_savings:.2f} from {sale_count} items")
                else:
                    print("   No special deals found - all regular prices")
                    
            else:
                print(f"❌ Budget optimization failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Budget shopping demo failed")
    
    async def demo_quality_shopping(self):
        """Demonstrate quality-first shopping optimization."""
        print("\n⭐ Quality-First Shopping Demo")
        print("=" * 50)
        print("Scenario: Premium shopping prioritizing product quality")
        print("Challenge: Find best quality products within reasonable cost")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create quality-focused shopping list
            quality_ingredients = [
                Ingredient(name="organic milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="sourdough bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="free range eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="grass fed beef", quantity=800.0, unit=UnitType.G, category="meat"),
                Ingredient(name="wild caught salmon", quantity=600.0, unit=UnitType.G, category="meat"),
                Ingredient(name="organic spinach", quantity=200.0, unit=UnitType.G, category="produce"),
                Ingredient(name="heirloom tomatoes", quantity=4.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="extra virgin olive oil", quantity=500.0, unit=UnitType.ML, category="cooking"),
                Ingredient(name="aged cheddar", quantity=300.0, unit=UnitType.G, category="dairy"),
                Ingredient(name="organic quinoa", quantity=1.0, unit=UnitType.KG, category="grains")
            ]
            
            print(f"📋 Premium Shopping List: {len(quality_ingredients)} items")
            print("Focus on organic, free-range, and artisanal products")
            print()
            
            for ingredient in quality_ingredients:
                print(f"   • {ingredient.name}")
            
            # Quality-first criteria
            quality_criteria = OptimizationCriteria(
                max_stores=3,
                quality_threshold=0.8,  # High quality requirement
                preferred_stores=["metro_ca"],  # Premium stores
                brand_preferences={"dairy": ["organic valley", "avalon"], "meat": ["grass fed", "free range"]}
            )
            
            print("\n🎯 Quality Optimization Criteria:")
            print(f"   • Quality threshold: {quality_criteria.quality_threshold} (high)")
            print(f"   • Max stores: {quality_criteria.max_stores}")
            print(f"   • Preferred stores: {quality_criteria.preferred_stores}")
            print(f"   • Brand preferences: Premium/organic brands")
            
            optimizer = OptimizerAgent()
            
            print("\n🔄 Finding highest quality options...")
            start_time = datetime.now()
            
            result = await optimizer.optimize_shopping_list(
                ingredients=quality_ingredients,
                criteria=quality_criteria,
                strategy="quality_first"
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False):
                self.display_optimization_results(result, processing_time, "Quality Shopping")
                
                # Compare with cost optimization
                print("\n📊 Quality vs Cost Comparison:")
                cost_result = await optimizer.optimize_shopping_list(
                    ingredients=quality_ingredients,
                    strategy="cost_only"
                )
                
                if cost_result.get("success", False):
                    quality_cost = float(result["savings_analysis"]["optimized_cost"])
                    budget_cost = float(cost_result["savings_analysis"]["optimized_cost"])
                    quality_premium = quality_cost - budget_cost
                    premium_percentage = (quality_premium / budget_cost * 100) if budget_cost > 0 else 0
                    
                    print(f"   Quality-first cost: ${quality_cost:.2f}")
                    print(f"   Cost-only alternative: ${budget_cost:.2f}")
                    print(f"   Quality premium: ${quality_premium:.2f} ({premium_percentage:.1f}%)")
                    
                    if premium_percentage <= 20:
                        print("   ✅ Excellent value for quality upgrade!")
                    elif premium_percentage <= 50:
                        print("   👍 Reasonable premium for higher quality")
                    else:
                        print("   💰 Significant premium - evaluate value carefully")
                
                # Show quality analysis
                print(f"\n⭐ Quality Analysis:")
                quality_trips = result["recommended_strategy"]
                
                high_quality_count = 0
                total_items = 0
                
                for trip in quality_trips:
                    for product_info in trip['products']:
                        total_items += 1
                        product = product_info['product']
                        # Simulate quality scoring (in real implementation, this would come from matching)
                        if any(quality_term in product['name'].lower() for quality_term in 
                               ['organic', 'free range', 'grass fed', 'wild caught', 'artisanal', 'premium']):
                            high_quality_count += 1
                
                quality_percentage = (high_quality_count / total_items * 100) if total_items > 0 else 0
                
                print(f"   Items meeting quality criteria: {high_quality_count}/{total_items}")
                print(f"   Quality achievement: {quality_percentage:.1f}%")
                
                if quality_percentage >= 80:
                    print("   🏆 Excellent quality selection!")
                elif quality_percentage >= 60:
                    print("   ⭐ Good quality mix")
                else:
                    print("   📈 Consider higher quality alternatives if available")
                
                # Show premium product highlights
                print(f"\n🏆 Premium Product Highlights:")
                premium_products = []
                
                for trip in quality_trips:
                    for product_info in trip['products']:
                        product = product_info['product']
                        ingredient = product_info['ingredient']
                        if any(quality_term in product['name'].lower() for quality_term in 
                               ['organic', 'free range', 'grass fed', 'wild caught', 'artisanal', 'premium']):
                            premium_products.append({
                                'ingredient': ingredient['name'],
                                'product': product['name'],
                                'price': product['price'],
                                'store': product['store_id']
                            })
                
                for i, item in enumerate(premium_products[:5], 1):
                    print(f"   {i}. {item['ingredient']} → {item['product']}")
                    print(f"      💰 ${item['price']} at {item['store']}")
                
                if len(premium_products) > 5:
                    print(f"      ... and {len(premium_products) - 5} more premium items")
                    
            else:
                print(f"❌ Quality optimization failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Quality shopping demo failed")
    
    async def demo_strategy_comparison(self):
        """Demonstrate strategy comparison across all optimization approaches."""
        print("\n⚖️  Strategy Comparison Demo")
        print("=" * 50)
        print("Scenario: Compare all optimization strategies with same ingredients")
        print("Challenge: Understand trade-offs between different approaches")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create balanced shopping list for comparison
            comparison_ingredients = [
                Ingredient(name="milk", quantity=2.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.0, unit=UnitType.KG, category="meat"),
                Ingredient(name="eggs", quantity=12.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="bananas", quantity=6.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="rice", quantity=1.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="tomatoes", quantity=4.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="cheese", quantity=200.0, unit=UnitType.G, category="dairy")
            ]
            
            print(f"📋 Test Shopping List: {len(comparison_ingredients)} items")
            for ingredient in comparison_ingredients:
                print(f"   • {ingredient.name}")
            
            strategies = ["cost_only", "convenience", "balanced", "quality_first", "time_efficient"]
            
            print(f"\n🔄 Running optimization with {len(strategies)} strategies...")
            
            optimizer = OptimizerAgent()
            start_time = datetime.now()
            
            results = await optimizer.compare_strategies(
                ingredients=comparison_ingredients,
                strategies=strategies
            )
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️  Comparison completed in {total_time:.2f} seconds")
            print()
            
            # Display comparison table
            print("📊 Strategy Comparison Results:")
            print("=" * 90)
            print(f"{'Strategy':<15} {'Cost':<10} {'Stores':<8} {'Time':<8} {'Coverage':<10} {'Savings':<10} {'Status':<8}")
            print("-" * 90)
            
            valid_results = {}
            
            for strategy, result in results.items():
                if result.get("success", False):
                    cost = result["savings_analysis"]["optimized_cost"]
                    stores = result["optimization_summary"]["total_stores"]
                    time_min = result["optimization_summary"]["total_time"]
                    coverage = result["optimization_summary"]["coverage_percentage"]
                    savings = result["savings_analysis"]["savings_percentage"]
                    
                    print(f"{strategy:<15} ${cost:<9.2f} {stores:<8} {time_min:<8} {coverage:<9.1f}% {savings:<9.1f}% {'✅':<8}")
                    valid_results[strategy] = result
                else:
                    error = result.get("error", "Unknown error")[:30]
                    print(f"{strategy:<15} {'FAILED':<60} {'❌':<8}")
                    print(f"{'':>15} Error: {error}")
            
            print("-" * 90)
            
            if valid_results:
                # Analyze best performers
                print("\n🏆 Best Performers:")
                
                # Lowest cost
                best_cost = min(valid_results.items(), key=lambda x: float(x[1]["savings_analysis"]["optimized_cost"]))
                print(f"   💰 Lowest cost: {best_cost[0]} (${best_cost[1]['savings_analysis']['optimized_cost']:.2f})")
                
                # Fewest stores
                best_convenience = min(valid_results.items(), key=lambda x: x[1]["optimization_summary"]["total_stores"])
                print(f"   🚗 Most convenient: {best_convenience[0]} ({best_convenience[1]['optimization_summary']['total_stores']} stores)")
                
                # Fastest time
                best_time = min(valid_results.items(), key=lambda x: x[1]["optimization_summary"]["total_time"])
                print(f"   ⚡ Fastest: {best_time[0]} ({best_time[1]['optimization_summary']['total_time']} minutes)")
                
                # Best coverage
                best_coverage = max(valid_results.items(), key=lambda x: x[1]["optimization_summary"]["coverage_percentage"])
                print(f"   📊 Best coverage: {best_coverage[0]} ({best_coverage[1]['optimization_summary']['coverage_percentage']:.1f}%)")
                
                # Highest savings
                best_savings = max(valid_results.items(), key=lambda x: x[1]["savings_analysis"]["savings_percentage"])
                print(f"   💎 Best savings: {best_savings[0]} ({best_savings[1]['savings_analysis']['savings_percentage']:.1f}%)")
                
                # Show detailed breakdown for top strategies
                print("\n📈 Detailed Analysis of Top Strategies:")
                
                # Analyze cost vs convenience trade-off
                cost_data = [(strategy, float(data["savings_analysis"]["optimized_cost"]), 
                             data["optimization_summary"]["total_stores"]) 
                            for strategy, data in valid_results.items()]
                cost_data.sort(key=lambda x: x[1])  # Sort by cost
                
                print("\n   💰 Cost vs Convenience Trade-off:")
                for strategy, cost, stores in cost_data:
                    convenience_score = 5 - stores if stores <= 5 else 1  # Simple convenience scoring
                    print(f"      {strategy:<15}: ${cost:<8.2f} | {stores} stores | Convenience: {'⭐' * convenience_score}")
                
                # Show strategy recommendations
                print(f"\n💡 Strategy Recommendations:")
                
                cheapest_strategy = cost_data[0][0]
                most_convenient = best_convenience[0]
                
                if cheapest_strategy == most_convenient:
                    print(f"   🎯 Clear winner: {cheapest_strategy} offers both best cost and convenience!")
                else:
                    print(f"   💰 For budget-conscious: Choose {cheapest_strategy}")
                    print(f"   🚗 For convenience: Choose {most_convenient}")
                    print(f"   ⚖️  For balance: Consider 'balanced' strategy")
                
                # Performance insights
                cost_range = max(float(data["savings_analysis"]["optimized_cost"]) for data in valid_results.values()) - \
                           min(float(data["savings_analysis"]["optimized_cost"]) for data in valid_results.values())
                
                store_range = max(data["optimization_summary"]["total_stores"] for data in valid_results.values()) - \
                            min(data["optimization_summary"]["total_stores"] for data in valid_results.values())
                
                print(f"\n📊 Strategy Performance Insights:")
                print(f"   Cost variation: ${cost_range:.2f} across strategies")
                print(f"   Store count variation: {store_range} stores")
                print(f"   Strategies successful: {len(valid_results)}/{len(strategies)}")
                
                if cost_range < 5.0:
                    print("   💡 Low cost variation - convenience may be deciding factor")
                elif cost_range > 20.0:
                    print("   💡 High cost variation - strategy choice significantly impacts budget")
                
                if store_range <= 1:
                    print("   💡 Similar convenience across strategies")
                else:
                    print("   💡 Convenience varies significantly - consider your time constraints")
                    
            else:
                print("❌ No strategies succeeded - check system configuration")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Strategy comparison demo failed")
    
    async def demo_custom_shopping(self):
        """Interactive demo where user can input their own shopping list."""
        print("\n🎯 Custom Shopping List Demo")
        print("=" * 50)
        print("Enter your own ingredients for personalized optimization!")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from decimal import Decimal
            
            # Collect ingredients from user
            print("📝 Enter ingredients one by one (press Enter with empty line to finish):")
            print("Format: 'ingredient name' or 'ingredient name, quantity, unit'")
            print("Example: 'milk' or 'milk, 2, l' or 'chicken breast, 1.5, kg'")
            print()
            
            custom_ingredients = []
            ingredient_count = 1
            
            while True:
                try:
                    user_input = input(f"   {ingredient_count}. ").strip()
                    
                    if not user_input:
                        break
                    
                    # Parse input
                    parts = [p.strip() for p in user_input.split(',')]
                    
                    if len(parts) == 1:
                        # Just ingredient name
                        ingredient = Ingredient(
                            name=parts[0],
                            quantity=1.0,
                            unit=UnitType.PIECES
                        )
                    elif len(parts) == 3:
                        # Name, quantity, unit
                        name = parts[0]
                        quantity = float(parts[1])
                        unit = UnitType(parts[2].lower())
                        
                        ingredient = Ingredient(
                            name=name,
                            quantity=quantity,
                            unit=unit
                        )
                    else:
                        print("   ⚠️  Invalid format. Use: 'name' or 'name, quantity, unit'")
                        continue
                    
                    custom_ingredients.append(ingredient)
                    print(f"      ✅ Added: {ingredient.name} ({ingredient.quantity} {ingredient.unit})")
                    ingredient_count += 1
                    
                except ValueError as e:
                    print(f"   ⚠️  Error parsing input: {e}")
                    continue
                except KeyboardInterrupt:
                    print("\n   Cancelled by user")
                    return
            
            if not custom_ingredients:
                print("No ingredients entered. Demo cancelled.")
                return
            
            print(f"\n📋 Your Shopping List ({len(custom_ingredients)} items):")
            for i, ingredient in enumerate(custom_ingredients, 1):
                print(f"   {i}. {ingredient.name} - {ingredient.quantity} {ingredient.unit}")
            
            # Get user preferences
            print("\n⚙️  Optimization Preferences:")
            
            # Budget
            budget_input = input("   Maximum budget (press Enter for no limit): $").strip()
            max_budget = Decimal(budget_input) if budget_input else None
            
            # Store preference
            print("   Store preference:")
            print("   1. No preference (all stores)")
            print("   2. Prefer Metro")
            print("   3. Prefer Walmart")
            print("   4. Prefer FreshCo")
            
            store_choice = input("   Choice (1-4): ").strip()
            preferred_stores = []
            if store_choice == "2":
                preferred_stores = ["metro_ca"]
            elif store_choice == "3":
                preferred_stores = ["walmart_ca"]
            elif store_choice == "4":
                preferred_stores = ["freshco_com"]
            
            # Max stores
            max_stores_input = input("   Maximum stores to visit (1-5, default 3): ").strip()
            max_stores = int(max_stores_input) if max_stores_input and max_stores_input.isdigit() else 3
            
            # Strategy
            print("   Optimization strategy:")
            print("   1. Cost only (cheapest)")
            print("   2. Convenience (fewer stores)")
            print("   3. Balanced (cost + convenience)")
            print("   4. Quality first")
            print("   5. Time efficient")
            print("   6. Adaptive (AI chooses)")
            
            strategy_choice = input("   Choice (1-6): ").strip()
            strategy_map = {
                "1": "cost_only",
                "2": "convenience", 
                "3": "balanced",
                "4": "quality_first",
                "5": "time_efficient",
                "6": "adaptive"
            }
            strategy = strategy_map.get(strategy_choice, "adaptive")
            
            # Create optimization criteria
            criteria = OptimizationCriteria(
                max_budget=max_budget,
                max_stores=max_stores,
                preferred_stores=preferred_stores,
                quality_threshold=0.7
            )
            
            print(f"\n🎯 Your Optimization Settings:")
            print(f"   Budget: ${max_budget}" if max_budget else "   Budget: No limit")
            print(f"   Max stores: {max_stores}")
            print(f"   Preferred stores: {preferred_stores if preferred_stores else 'None'}")
            print(f"   Strategy: {strategy}")
            
            # Run optimization
            optimizer = OptimizerAgent()
            
            print(f"\n🔄 Optimizing your shopping list...")
            start_time = datetime.now()
            
            result = await optimizer.optimize_shopping_list(
                ingredients=custom_ingredients,
                criteria=criteria,
                strategy=strategy
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False):
                self.display_optimization_results(result, processing_time, "Custom Shopping")
                
                # Show personalized recommendations
                print(f"\n💡 Personalized Recommendations:")
                
                savings = result["savings_analysis"]
                if float(savings["savings_percentage"]) > 10:
                    print(f"   💰 Great savings potential! You could save ${savings['total_savings']:.2f}")
                elif float(savings["savings_percentage"]) > 5:
                    print(f"   👍 Moderate savings of ${savings['total_savings']:.2f} available")
                else:
                    print(f"   ✅ Your current approach is already efficient!")
                
                stores_needed = len(result["recommended_strategy"])
                if stores_needed == 1:
                    print(f"   🚗 Convenient! Everything available at one store")
                elif stores_needed <= 2:
                    print(f"   👍 Reasonable shopping with {stores_needed} stores")
                else:
                    print(f"   🚶 {stores_needed} stores needed - consider if time vs savings is worth it")
                
                coverage = result["optimization_summary"]["coverage_percentage"]
                if coverage == 100:
                    print(f"   ✅ Perfect! All items found")
                elif coverage >= 90:
                    print(f"   👍 Excellent coverage at {coverage:.1f}%")
                else:
                    print(f"   ⚠️  Some items not found ({coverage:.1f}% coverage)")
                    unmatched = result.get("unmatched_ingredients", [])
                    if unmatched:
                        print(f"   Missing: {', '.join(ing['name'] for ing in unmatched)}")
                        print(f"   💡 Consider substitutions or additional stores")
                
                # Budget check
                if max_budget:
                    cost = float(result["savings_analysis"]["optimized_cost"])
                    budget_used = cost / float(max_budget)
                    
                    if budget_used <= 0.8:
                        remaining = float(max_budget) - cost
                        print(f"   💰 Under budget! ${remaining:.2f} remaining for extras")
                    elif budget_used <= 1.0:
                        print(f"   ✅ Within budget ({budget_used:.1%} used)")
                    else:
                        overage = cost - float(max_budget)
                        print(f"   ⚠️  Over budget by ${overage:.2f} - consider removing items")
                        
            else:
                print(f"❌ Custom optimization failed: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Custom shopping demo cancelled by user")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Custom shopping demo failed")
    
    async def demo_savings_estimation(self):
        """Demonstrate savings estimation functionality."""
        print("\n📊 Savings Estimation Demo")
        print("=" * 50)
        print("Scenario: Estimate how much you could save with optimization")
        print("Challenge: Compare your current shopping method vs optimal")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            
            # Create realistic shopping list
            estimation_ingredients = [
                Ingredient(name="milk", quantity=3.0, unit=UnitType.L, category="dairy"),
                Ingredient(name="bread", quantity=2.0, unit=UnitType.PIECES, category="bakery"),
                Ingredient(name="chicken breast", quantity=1.5, unit=UnitType.KG, category="meat"),
                Ingredient(name="eggs", quantity=18.0, unit=UnitType.PIECES, category="dairy"),
                Ingredient(name="bananas", quantity=8.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="rice", quantity=1.0, unit=UnitType.KG, category="grains"),
                Ingredient(name="pasta", quantity=500.0, unit=UnitType.G, category="grains"),
                Ingredient(name="tomatoes", quantity=6.0, unit=UnitType.PIECES, category="produce"),
                Ingredient(name="cheese", quantity=300.0, unit=UnitType.G, category="dairy"),
                Ingredient(name="yogurt", quantity=4.0, unit=UnitType.PIECES, category="dairy")
            ]
            
            print(f"📋 Sample Weekly Shopping List ({len(estimation_ingredients)} items):")
            for ingredient in estimation_ingredients:
                print(f"   • {ingredient.name}")
            
            shopping_methods = [
                ("convenience", "Single store convenience shopping"),
                ("balanced", "Balanced approach (some store comparison)"),
                ("cost_only", "Full cost optimization")
            ]
            
            print(f"\n🔄 Estimating savings across different approaches...")
            
            optimizer = OptimizerAgent()
            estimates = {}
            
            for method, description in shopping_methods:
                print(f"   Analyzing {description}...")
                
                try:
                    estimate = await optimizer.estimate_savings(
                        ingredients=estimation_ingredients,
                        current_shopping_method=method
                    )
                    estimates[method] = estimate
                except Exception as e:
                    print(f"     ⚠️  Failed: {e}")
                    estimates[method] = {"success": False, "error": str(e)}
            
            # Display savings comparison
            print(f"\n💰 Savings Estimation Results:")
            print("=" * 70)
            print(f"{'Method':<20} {'Current Cost':<15} {'Optimized':<15} {'Savings':<15} {'%':<10}")
            print("-" * 70)
            
            valid_estimates = {}
            
            for method, estimate in estimates.items():
                method_name = method.replace("_", " ").title()
                
                if estimate.get("success", True) and "current_cost" in estimate:
                    current = estimate["current_cost"]
                    optimized = estimate["optimized_cost"]
                    savings = estimate["potential_savings"]
                    percentage = estimate["savings_percentage"]
                    
                    print(f"{method_name:<20} ${current:<14.2f} ${optimized:<14.2f} ${savings:<14.2f} {percentage:<9.1f}%")
                    valid_estimates[method] = estimate
                else:
                    print(f"{method_name:<20} {'FAILED':<60}")
            
            print("-" * 70)
            
            if valid_estimates:
                # Analysis and insights
                print(f"\n📈 Savings Analysis:")
                
                costs = [(method, float(est["current_cost"])) for method, est in valid_estimates.items()]
                costs.sort(key=lambda x: x[1])
                
                cheapest_method, cheapest_cost = costs[0]
                most_expensive_method, most_expensive_cost = costs[-1]
                
                max_savings = most_expensive_cost - cheapest_cost
                max_savings_pct = (max_savings / most_expensive_cost * 100) if most_expensive_cost > 0 else 0
                
                print(f"   💡 Shopping method comparison:")
                print(f"      Most expensive: {most_expensive_method.replace('_', ' ').title()} (${most_expensive_cost:.2f})")
                print(f"      Least expensive: {cheapest_method.replace('_', ' ').title()} (${cheapest_cost:.2f})")
                print(f"      Maximum savings: ${max_savings:.2f} ({max_savings_pct:.1f}%)")
                
                # Recommendations
                print(f"\n💡 Personalized Recommendations:")
                
                if max_savings_pct > 20:
                    print(f"   🎯 High savings potential! Consider switching to {cheapest_method.replace('_', ' ')}")
                    print(f"   💰 You could save ${max_savings:.2f} per shopping trip")
                    
                    # Calculate annual savings
                    weekly_savings = max_savings
                    annual_savings = weekly_savings * 52
                    print(f"   📅 Annual potential savings: ${annual_savings:.2f}")
                    
                elif max_savings_pct > 10:
                    print(f"   👍 Moderate savings available with optimization")
                    print(f"   💡 Consider {cheapest_method.replace('_', ' ')} for better value")
                    
                else:
                    print(f"   ✅ Your shopping methods are already quite efficient!")
                    print(f"   💡 Minor optimizations could still save ${max_savings:.2f}")
                
                # Time vs money analysis
                print(f"\n⏰ Time vs Money Trade-offs:")
                
                if "convenience" in valid_estimates and "cost_only" in valid_estimates:
                    conv_cost = float(valid_estimates["convenience"]["current_cost"])
                    cost_only_cost = float(valid_estimates["cost_only"]["optimized_cost"])
                    time_savings_cost = conv_cost - cost_only_cost
                    
                    # Estimate time difference (simplified)
                    convenience_time = 30  # minutes
                    cost_optimized_time = 75  # minutes (multiple stores)
                    time_difference = cost_optimized_time - convenience_time
                    
                    hourly_value = (time_savings_cost / (time_difference / 60)) if time_difference > 0 else 0
                    
                    print(f"   🚗 Convenience shopping: ~{convenience_time} minutes")
                    print(f"   💰 Cost-optimized shopping: ~{cost_optimized_time} minutes")
                    print(f"   ⏱️  Extra time for savings: {time_difference} minutes")
                    print(f"   💡 Your time worth for savings: ${hourly_value:.2f}/hour")
                    
                    if hourly_value > 20:
                        print(f"   🎯 Excellent return on time investment!")
                    elif hourly_value > 10:
                        print(f"   👍 Good value for your time")
                    else:
                        print(f"   🤔 Consider if time savings are worth the cost")
                
                # Weekly vs monthly analysis
                print(f"\n📅 Budget Impact Analysis:")
                
                if valid_estimates:
                    avg_cost = sum(float(est["current_cost"]) for est in valid_estimates.values()) / len(valid_estimates)
                    
                    print(f"   Weekly shopping: ${avg_cost:.2f}")
                    print(f"   Monthly budget: ${avg_cost * 4:.2f}")
                    print(f"   Annual grocery spend: ${avg_cost * 52:.2f}")
                    
                    if max_savings > 0:
                        monthly_savings = max_savings * 4
                        annual_savings = max_savings * 52
                        
                        print(f"   💰 Monthly savings potential: ${monthly_savings:.2f}")
                        print(f"   💎 Annual savings potential: ${annual_savings:.2f}")
                        
                        # Savings suggestions
                        if annual_savings >= 500:
                            print(f"   🎯 Major savings! Could fund a vacation or emergency fund")
                        elif annual_savings >= 200:
                            print(f"   👍 Significant savings for other priorities")
                        else:
                            print(f"   ✅ Every bit helps - perfect for small treats")
                            
            else:
                print("❌ No valid savings estimates - check system configuration")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Savings estimation demo failed")
    
    async def demo_batch_recipes(self):
        """Demonstrate batch recipe processing for meal planning."""
        print("\n🔧 Batch Recipe Processing Demo")
        print("=" * 50)
        print("Scenario: Process multiple recipes for weekly meal planning")
        print("Challenge: Optimize consolidated ingredients across recipes")
        print()
        
        try:
            from agentic_grocery_price_scanner.agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
            from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
            from agentic_grocery_price_scanner.data_models.base import UnitType
            from collections import defaultdict
            from decimal import Decimal
            
            # Define sample recipes
            recipes = {
                "Chicken Stir Fry": [
                    Ingredient(name="chicken breast", quantity=600.0, unit=UnitType.G, category="meat"),
                    Ingredient(name="bell peppers", quantity=2.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="onions", quantity=1.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="soy sauce", quantity=1.0, unit=UnitType.PIECES, category="cooking"),
                    Ingredient(name="rice", quantity=400.0, unit=UnitType.G, category="grains")
                ],
                "Spaghetti Carbonara": [
                    Ingredient(name="pasta", quantity=400.0, unit=UnitType.G, category="grains"),
                    Ingredient(name="eggs", quantity=4.0, unit=UnitType.PIECES, category="dairy"),
                    Ingredient(name="bacon", quantity=200.0, unit=UnitType.G, category="meat"),
                    Ingredient(name="parmesan cheese", quantity=100.0, unit=UnitType.G, category="dairy"),
                    Ingredient(name="black pepper", quantity=1.0, unit=UnitType.PIECES, category="cooking")
                ],
                "Chicken Curry": [
                    Ingredient(name="chicken breast", quantity=800.0, unit=UnitType.G, category="meat"),
                    Ingredient(name="coconut milk", quantity=400.0, unit=UnitType.ML, category="pantry"),
                    Ingredient(name="onions", quantity=2.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="curry powder", quantity=1.0, unit=UnitType.PIECES, category="cooking"),
                    Ingredient(name="rice", quantity=300.0, unit=UnitType.G, category="grains")
                ],
                "Caesar Salad": [
                    Ingredient(name="romaine lettuce", quantity=2.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="parmesan cheese", quantity=50.0, unit=UnitType.G, category="dairy"),
                    Ingredient(name="bread", quantity=1.0, unit=UnitType.PIECES, category="bakery"),
                    Ingredient(name="caesar dressing", quantity=1.0, unit=UnitType.PIECES, category="condiments"),
                    Ingredient(name="eggs", quantity=2.0, unit=UnitType.PIECES, category="dairy")
                ],
                "Beef Tacos": [
                    Ingredient(name="ground beef", quantity=500.0, unit=UnitType.G, category="meat"),
                    Ingredient(name="taco shells", quantity=8.0, unit=UnitType.PIECES, category="pantry"),
                    Ingredient(name="tomatoes", quantity=3.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="lettuce", quantity=1.0, unit=UnitType.PIECES, category="produce"),
                    Ingredient(name="cheddar cheese", quantity=200.0, unit=UnitType.G, category="dairy"),
                    Ingredient(name="onions", quantity=1.0, unit=UnitType.PIECES, category="produce")
                ]
            }
            
            print(f"🍽️  Sample Recipe Collection ({len(recipes)} recipes):")
            for recipe_name, ingredients in recipes.items():
                print(f"   📋 {recipe_name}: {len(ingredients)} ingredients")
            
            print(f"\n🔧 Consolidating ingredients across all recipes...")
            
            # Consolidate ingredients
            consolidated = defaultdict(float)
            ingredient_details = {}
            
            for recipe_name, ingredients in recipes.items():
                for ingredient in ingredients:
                    key = (ingredient.name, ingredient.unit)
                    consolidated[key] += ingredient.quantity
                    ingredient_details[key] = ingredient
            
            # Create consolidated shopping list
            consolidated_ingredients = []
            for (name, unit), total_quantity in consolidated.items():
                original_ingredient = ingredient_details[(name, unit)]
                
                consolidated_ingredient = Ingredient(
                    name=name,
                    quantity=total_quantity,
                    unit=unit,
                    category=original_ingredient.category
                )
                consolidated_ingredients.append(consolidated_ingredient)
            
            print(f"📦 Consolidated Shopping List ({len(consolidated_ingredients)} unique items):")
            for ingredient in sorted(consolidated_ingredients, key=lambda x: x.category or ""):
                print(f"   • {ingredient.name}: {ingredient.quantity} {ingredient.unit}")
            
            print(f"\n💰 Recipe-by-recipe vs Consolidated Optimization:")
            
            optimizer = OptimizerAgent()
            
            # Optimize each recipe individually
            print(f"\n🔄 Optimizing recipes individually...")
            individual_results = []
            individual_total_cost = 0
            individual_total_time = 0
            individual_stores = set()
            
            for recipe_name, ingredients in recipes.items():
                print(f"   Optimizing {recipe_name}...")
                
                try:
                    result = await optimizer.optimize_shopping_list(
                        ingredients=ingredients,
                        strategy="balanced"
                    )
                    
                    if result.get("success", False):
                        cost = float(result["savings_analysis"]["optimized_cost"])
                        time_min = result["optimization_summary"]["total_time"]
                        stores = [trip["store_id"] for trip in result["recommended_strategy"]]
                        
                        individual_results.append({
                            "recipe": recipe_name,
                            "cost": cost,
                            "time": time_min,
                            "stores": stores
                        })
                        
                        individual_total_cost += cost
                        individual_total_time += time_min
                        individual_stores.update(stores)
                        
                        print(f"      ✅ ${cost:.2f}, {time_min} min, {len(stores)} stores")
                    else:
                        print(f"      ❌ Failed")
                        
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            # Optimize consolidated list
            print(f"\n🔄 Optimizing consolidated shopping list...")
            
            consolidated_criteria = OptimizationCriteria(
                max_budget=Decimal("300.00"),
                max_stores=4,
                bulk_buying_ok=True
            )
            
            consolidated_result = await optimizer.optimize_shopping_list(
                ingredients=consolidated_ingredients,
                criteria=consolidated_criteria,
                strategy="balanced"
            )
            
            # Compare results
            if consolidated_result.get("success", False) and individual_results:
                consolidated_cost = float(consolidated_result["savings_analysis"]["optimized_cost"])
                consolidated_time = consolidated_result["optimization_summary"]["total_time"]
                consolidated_stores = len(consolidated_result["recommended_strategy"])
                
                print(f"\n📊 Optimization Comparison:")
                print("=" * 60)
                print(f"{'Method':<20} {'Total Cost':<12} {'Total Time':<12} {'Stores':<10}")
                print("-" * 60)
                print(f"{'Individual recipes':<20} ${individual_total_cost:<11.2f} {individual_total_time:<11} {len(individual_stores):<10}")
                print(f"{'Consolidated':<20} ${consolidated_cost:<11.2f} {consolidated_time:<11} {consolidated_stores:<10}")
                print("-" * 60)
                
                # Calculate savings
                cost_savings = individual_total_cost - consolidated_cost
                time_savings = individual_total_time - consolidated_time
                store_reduction = len(individual_stores) - consolidated_stores
                
                print(f"{'Savings':<20} ${cost_savings:<11.2f} {time_savings:<11} {store_reduction:<10}")
                
                savings_percentage = (cost_savings / individual_total_cost * 100) if individual_total_cost > 0 else 0
                time_efficiency = (time_savings / individual_total_time * 100) if individual_total_time > 0 else 0
                
                print(f"\n💡 Consolidation Benefits:")
                print(f"   💰 Cost savings: ${cost_savings:.2f} ({savings_percentage:.1f}%)")
                print(f"   ⏰ Time savings: {time_savings} minutes ({time_efficiency:.1f}%)")
                print(f"   🏪 Store reduction: {store_reduction} fewer stores")
                
                # Efficiency analysis
                if cost_savings > 20:
                    print(f"   🎯 Excellent cost efficiency from bulk buying!")
                elif cost_savings > 10:
                    print(f"   👍 Good cost efficiency from consolidation")
                else:
                    print(f"   ✅ Minimal cost difference - convenience is main benefit")
                
                if time_savings > 60:
                    print(f"   🚀 Major time savings from consolidated shopping!")
                elif time_savings > 30:
                    print(f"   ⏰ Significant time savings")
                else:
                    print(f"   ⏱️  Minor time impact")
                
                # Meal planning insights
                print(f"\n🍽️  Meal Planning Insights:")
                
                # Identify shared ingredients
                shared_ingredients = {}
                for ingredient in consolidated_ingredients:
                    recipe_count = 0
                    for recipe_ingredients in recipes.values():
                        if any(ing.name == ingredient.name for ing in recipe_ingredients):
                            recipe_count += 1
                    
                    if recipe_count > 1:
                        shared_ingredients[ingredient.name] = recipe_count
                
                if shared_ingredients:
                    print(f"   🔄 Shared ingredients across recipes:")
                    for ingredient_name, count in sorted(shared_ingredients.items(), key=lambda x: x[1], reverse=True):
                        print(f"      • {ingredient_name}: used in {count} recipes")
                
                # Shopping trip organization
                print(f"\n🛒 Optimized Shopping Plan:")
                for i, trip in enumerate(consolidated_result["recommended_strategy"], 1):
                    print(f"\n   Trip {i}: {trip['store_name']}")
                    print(f"   💰 Cost: ${trip['total_cost']:.2f}")
                    print(f"   📦 Items: {trip['total_items']}")
                    
                    # Show which recipes benefit from this trip
                    trip_ingredients = [p["ingredient"]["name"] for p in trip["products"]]
                    benefiting_recipes = []
                    
                    for recipe_name, recipe_ingredients in recipes.items():
                        if any(ing.name in trip_ingredients for ing in recipe_ingredients):
                            benefiting_recipes.append(recipe_name)
                    
                    if benefiting_recipes:
                        print(f"   🍽️  Supports recipes: {', '.join(benefiting_recipes)}")
                
                # Weekly meal prep recommendations
                print(f"\n📅 Weekly Meal Prep Recommendations:")
                print(f"   🛒 Single shopping trip saves {time_savings} minutes")
                print(f"   💰 Bulk buying saves ${cost_savings:.2f}")
                print(f"   📋 {len(recipes)} meals planned with {len(consolidated_ingredients)} ingredients")
                print(f"   ⭐ Efficiency score: {100 - (consolidated_cost/individual_total_cost * 100):.1f}% improvement")
                
                if consolidated_cost < individual_total_cost and consolidated_time < individual_total_time:
                    print(f"   🏆 Consolidated shopping is clearly superior!")
                elif cost_savings > 0:
                    print(f"   👍 Consolidated shopping provides good value")
                else:
                    print(f"   ⚖️  Similar costs - choose based on convenience preference")
                    
            else:
                print("❌ Consolidated optimization failed or no individual results to compare")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Batch recipe processing demo failed")
    
    def display_optimization_results(self, result: Dict[str, Any], processing_time: float, scenario_name: str):
        """Display standardized optimization results."""
        print(f"\n✅ {scenario_name} Optimization Complete!")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        summary = result["optimization_summary"]
        savings = result["savings_analysis"]
        
        print(f"\n📊 Optimization Results:")
        print(f"   Strategy selected: {summary['selected_strategy']}")
        print(f"   Total cost: ${savings['optimized_cost']:.2f}")
        print(f"   Stores needed: {summary['total_stores']}")
        print(f"   Total time: {summary['total_time']} minutes")
        print(f"   Coverage: {summary['coverage_percentage']:.1f}%")
        
        if float(savings['total_savings']) > 0:
            print(f"   💰 Savings: ${savings['total_savings']:.2f} ({savings['savings_percentage']:.1f}%)")
        
        # Show unmatched items if any
        unmatched = result.get("unmatched_ingredients", [])
        if unmatched:
            print(f"   ⚠️  Unmatched items: {', '.join(ing['name'] for ing in unmatched)}")
    
    async def run_interactive_demo(self):
        """Run the interactive demo menu."""
        self.display_banner()
        
        try:
            while True:
                self.display_menu()
                
                try:
                    choice = input("🎯 Select a demo (0-8): ").strip()
                    
                    if choice == "0":
                        print("\n👋 Thank you for exploring OptimizerAgent!")
                        print("🎉 Ready to revolutionize your grocery shopping!")
                        break
                    
                    if choice in self.scenarios:
                        scenario = self.scenarios[choice]
                        print(f"\n🚀 Starting: {scenario['name']}")
                        
                        start_time = datetime.now()
                        await scenario["method"]()
                        end_time = datetime.now()
                        
                        duration = (end_time - start_time).total_seconds()
                        print(f"\n✅ Demo completed in {duration:.1f} seconds")
                        
                        # Ask if user wants to continue
                        print("\n" + "="*50)
                        continue_choice = input("Continue with another demo? (y/N): ").strip().lower()
                        if continue_choice not in ['y', 'yes']:
                            print("\n👋 Thanks for exploring OptimizerAgent!")
                            break
                    else:
                        print("❌ Invalid choice. Please select 0-8.")
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️  Demo interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Demo error: {e}")
                    logger.exception("Demo error")
                    
                    continue_choice = input("\nTry another demo? (y/N): ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break
                        
        except KeyboardInterrupt:
            print("\n\n👋 Demo session ended by user")

async def main():
    """Main demo entry point."""
    demo = OptimizerDemo()
    await demo.run_interactive_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo ended by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo failed")