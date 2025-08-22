#!/usr/bin/env python3
"""
Demonstration of LLM integration with grocery-specific tasks.
Shows both Qwen 2.5 1.5B and Phi-3.5 Mini working on real grocery price scanner scenarios.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic_grocery_price_scanner.llm_client import OllamaClient, PromptTemplates
from agentic_grocery_price_scanner.llm_client.ollama_client import ModelType


async def demo_ingredient_processing():
    """Demonstrate Qwen 2.5 1.5B for fast ingredient processing."""
    print("🥕 Ingredient Processing with Qwen 2.5 1.5B")
    print("=" * 60)
    
    client = OllamaClient()
    
    # Complex ingredient list from a real recipe
    shopping_list = [
        "1 lb fresh organic ground beef (80/20 lean)",
        "2 cups unbleached all-purpose flour - King Arthur brand",
        "1 dozen large free-range brown eggs",
        "32 fl oz Horizon Organic Whole Milk",
        "1 bag (5 lbs) russet potatoes",
        "2 lbs fresh broccoli crowns",
        "1 container (32 oz) Greek yogurt - Fage Total 0%",
        "1 loaf whole grain bread - Dave's Killer Bread",
        "2 lbs boneless skinless chicken breasts",
        "1 jar (24 oz) marinara sauce - Rao's Homemade"
    ]
    
    print("Original Shopping List:")
    for i, item in enumerate(shopping_list, 1):
        print(f"  {i:2}. {item}")
    
    print(f"\nProcessing {len(shopping_list)} items with Qwen 2.5 1.5B...")
    print("\nNormalized Ingredients:")
    
    normalized_items = []
    for i, item in enumerate(shopping_list, 1):
        # Normalize ingredient
        normalize_prompt = PromptTemplates.format_template(
            "NORMALIZE_INGREDIENT",
            ingredient=item
        )
        
        normalized = await client.generate(
            normalize_prompt,
            model=ModelType.QWEN_1_5B,
            temperature=0.1,
            max_tokens=15
        )
        
        # Extract brand
        brand_prompt = PromptTemplates.format_template(
            "NORMALIZE_BRAND",
            product_text=item
        )
        
        brand = await client.generate(
            brand_prompt,
            model=ModelType.QWEN_1_5B,
            temperature=0.1,
            max_tokens=10
        )
        
        # Classify category
        category_prompt = PromptTemplates.format_template(
            "CLASSIFY_PRODUCT_CATEGORY",
            product_name=normalized.strip()
        )
        
        category = await client.generate(
            category_prompt,
            model=ModelType.QWEN_1_5B,
            temperature=0.1,
            max_tokens=10
        )
        
        normalized_items.append({
            "original": item,
            "normalized": normalized.strip(),
            "brand": brand.strip(),
            "category": category.strip()
        })
        
        print(f"  {i:2}. {normalized.strip():<25} | {brand.strip():<15} | {category.strip()}")
    
    return normalized_items


async def demo_shopping_optimization():
    """Demonstrate Phi-3.5 Mini for complex shopping optimization."""
    print("\n🧠 Shopping Optimization with Phi-3.5 Mini")
    print("=" * 60)
    
    client = OllamaClient()
    
    # Simulate store price data
    price_comparison = """
    Metro Canada:
    - Ground beef (80/20): $6.99/lb
    - King Arthur flour: $4.49
    - Free-range eggs: $5.99/dozen
    - Horizon milk: $4.99
    - Russet potatoes: $3.99/5lb
    
    Walmart Canada:
    - Ground beef (80/20): $5.99/lb  
    - King Arthur flour: $3.99
    - Free-range eggs: $4.99/dozen
    - Horizon milk: $4.79
    - Russet potatoes: $2.99/5lb
    
    FreshCo:
    - Ground beef (80/20): $6.49/lb
    - Generic flour: $2.99
    - Regular eggs: $3.99/dozen
    - Generic milk: $3.99
    - Russet potatoes: $3.49/5lb
    """
    
    optimization_prompt = PromptTemplates.format_template(
        "OPTIMIZATION_ADVICE",
        total_items=10,
        estimated_cost=65.00,
        store_list=["Metro Canada", "Walmart Canada", "FreshCo"],
        preferences="organic when possible, save money, single store preferred",
        time_available="45 minutes",
        product_data="See normalized shopping list above",
        price_comparison=price_comparison
    )
    
    print("Analyzing shopping optimization with Phi-3.5 Mini...")
    print("Store price comparison:")
    print(price_comparison)
    
    optimization_result = await client.structured_output(
        optimization_prompt,
        response_schema={
            "type": "object",
            "properties": {
                "cost_savings": {
                    "type": "object",
                    "properties": {
                        "potential_savings": {"type": "string"},
                        "percentage": {"type": "string"},
                        "best_store_combination": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "time_optimization": {
                    "type": "object",
                    "properties": {
                        "single_store_option": {"type": "string"},
                        "multi_store_route": {"type": "array", "items": {"type": "string"}},
                        "estimated_time": {"type": "string"}
                    }
                },
                "substitution_suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "substitute": {"type": "string"},
                            "savings": {"type": "string"},
                            "reason": {"type": "string"}
                        }
                    }
                },
                "priority_recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        model=ModelType.PHI3_5_MINI
    )
    
    print("\n📊 Optimization Results:")
    print("\n💰 Cost Savings:")
    cost_savings = optimization_result.get("cost_savings", {})
    print(f"  Potential savings: {cost_savings.get('potential_savings', 'N/A')}")
    print(f"  Percentage: {cost_savings.get('percentage', 'N/A')}")
    print(f"  Best stores: {', '.join(cost_savings.get('best_store_combination', []))}")
    
    print("\n⏰ Time Optimization:")
    time_opt = optimization_result.get("time_optimization", {})
    print(f"  Single store: {time_opt.get('single_store_option', 'N/A')}")
    print(f"  Multi-store route: {' → '.join(time_opt.get('multi_store_route', []))}")
    print(f"  Estimated time: {time_opt.get('estimated_time', 'N/A')}")
    
    print("\n🔄 Substitution Suggestions:")
    substitutions = optimization_result.get("substitution_suggestions", [])
    for sub in substitutions[:3]:  # Show top 3
        print(f"  '{sub.get('original', '')}' → '{sub.get('substitute', '')}'")
        print(f"    Savings: {sub.get('savings', 'N/A')} | Reason: {sub.get('reason', 'N/A')}")
    
    print("\n⭐ Priority Recommendations:")
    recommendations = optimization_result.get("priority_recommendations", [])
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    return optimization_result


async def demo_intelligent_strategy_selection():
    """Demonstrate intelligent scraping strategy selection."""
    print("\n🎯 Intelligent Strategy Selection")
    print("=" * 60)
    
    client = OllamaClient()
    
    # Simulate different scraping scenarios
    scenarios = [
        {
            "name": "High Success Rate Store",
            "store": "Metro Canada",
            "query": "organic milk",
            "layer1_success": 85,
            "layer2_success": 98,
            "time_limit": 15,
            "user_pref": "automated"
        },
        {
            "name": "Difficult Store (Heavy Protection)",
            "store": "Walmart Canada", 
            "query": "bread",
            "layer1_success": 20,
            "layer2_success": 95,
            "time_limit": 5,
            "user_pref": "fast"
        },
        {
            "name": "Time-Critical Search",
            "store": "FreshCo",
            "query": "chicken breast",
            "layer1_success": 70,
            "layer2_success": 90,
            "time_limit": 3,
            "user_pref": "reliable"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"Store: {scenario['store']} | Query: '{scenario['query']}'")
        print(f"Success rates: L1={scenario['layer1_success']}%, L2={scenario['layer2_success']}%")
        print(f"Time limit: {scenario['time_limit']} min | Preference: {scenario['user_pref']}")
        
        strategy_prompt = PromptTemplates.format_template(
            "SCRAPING_STRATEGY",
            store_name=scenario['store'],
            query=scenario['query'],
            previous_attempts=3,
            layer1_success=scenario['layer1_success'],
            layer2_success=scenario['layer2_success'],
            layer3_success=100,
            time_limit=scenario['time_limit'],
            user_preference=scenario['user_pref']
        )
        
        strategy = await client.structured_output(
            strategy_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "recommended_strategy": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "estimated_time": {"type": "string"},
                    "success_probability": {"type": "number"},
                    "fallback_plan": {"type": "string"}
                }
            },
            model=ModelType.PHI3_5_MINI
        )
        
        print(f"🤖 Recommended Strategy: Layer {strategy.get('recommended_strategy', 'N/A')}")
        print(f"   Confidence: {strategy.get('confidence', 0):.1%}")
        print(f"   Success Probability: {strategy.get('success_probability', 0):.1%}")
        print(f"   Estimated Time: {strategy.get('estimated_time', 'N/A')}")
        print(f"   Reasoning: {strategy.get('reasoning', 'N/A')[:100]}...")


async def demo_product_matching():
    """Demonstrate intelligent product matching."""
    print("\n🔍 Product Matching & Search Optimization")
    print("=" * 60)
    
    client = OllamaClient()
    
    # User searches for specific items
    user_queries = ["almond milk", "sourdough bread", "greek yogurt"]
    
    # Simulated available products from store
    available_products = [
        "Silk Original Almondmilk 64oz",
        "Blue Diamond Almond Breeze Unsweetened 32oz", 
        "Pacific Foods Organic Unsweetened Almond Milk",
        "Wonder Bread Classic White Sandwich Bread",
        "Dave's Killer Bread 21 Whole Grains",
        "Artisan Sourdough Loaf - Bakery Fresh",
        "Chobani Greek Yogurt Plain Nonfat 32oz",
        "Fage Total 0% Greek Yogurt",
        "Dannon Oikos Triple Zero Greek Yogurt"
    ]
    
    for query in user_queries:
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate search variations
        variations_prompt = PromptTemplates.format_template(
            "GENERATE_SEARCH_VARIATIONS",
            original_query=query
        )
        
        variations_result = await client.structured_output(
            variations_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "variations": {"type": "array", "items": {"type": "string"}}
                }
            },
            model=ModelType.QWEN_1_5B
        )
        
        variations = variations_result.get("variations", [])
        print(f"   Search variations: {', '.join(variations[:3])}...")
        
        # Match products
        product_list = '\n'.join([f"- {p}" for p in available_products])
        match_prompt = PromptTemplates.format_template(
            "MATCH_PRODUCTS",
            ingredient=query,
            product_list=product_list
        )
        
        matches = await client.structured_output(
            match_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_name": {"type": "string"},
                                "confidence": {"type": "number"},
                                "reason": {"type": "string"}
                            }
                        }
                    }
                }
            },
            model=ModelType.QWEN_1_5B
        )
        
        print("   📦 Best Matches:")
        for i, match in enumerate(matches.get("matches", [])[:3], 1):
            conf = match.get("confidence", 0)
            print(f"     {i}. {match.get('product_name', 'N/A')} ({conf:.1%})")
            print(f"        {match.get('reason', 'N/A')}")


async def main():
    """Run comprehensive LLM grocery task demonstration."""
    print("🚀 LLM Grocery Integration Demonstration")
    print("=" * 80)
    print("Showcasing Qwen 2.5 1.5B and Phi-3.5 Mini working together")
    print("for real grocery price scanner tasks.\n")
    
    try:
        # Test 1: Fast ingredient processing with Qwen
        normalized_items = await demo_ingredient_processing()
        
        # Test 2: Complex optimization with Phi-3.5
        optimization = await demo_shopping_optimization()
        
        # Test 3: Intelligent strategy selection
        await demo_intelligent_strategy_selection()
        
        # Test 4: Product matching and search
        await demo_product_matching()
        
        print("\n" + "=" * 80)
        print("✅ LLM Integration Demonstration Complete!")
        print("\n🎯 Key Achievements:")
        print("   ✓ Qwen 2.5 1.5B: Fast ingredient normalization, brand extraction, classification")
        print("   ✓ Phi-3.5 Mini: Complex reasoning, optimization, strategy decisions")
        print("   ✓ Intelligent model routing based on task complexity")
        print("   ✓ Structured JSON output for consistent data processing")
        print("   ✓ Performance optimization with caching and batching")
        print("   ✓ Error handling and fallback mechanisms")
        print("\n🚀 Ready for integration with the grocery price scanner!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())