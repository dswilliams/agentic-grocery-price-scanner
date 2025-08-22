#!/usr/bin/env python3
"""
Example integration of LLM client with the grocery price scanner system.
Shows how to enhance the existing agents with local LLM capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic_grocery_price_scanner.llm_client import OllamaClient, PromptTemplates
from agentic_grocery_price_scanner.llm_client.ollama_client import ModelType


class LLMEnhancedGroceryAgent:
    """
    Enhanced grocery price scanner agent with local LLM capabilities.
    """
    
    def __init__(self):
        self.llm_client = OllamaClient(enable_caching=True)
        
    async def initialize(self):
        """Initialize the LLM client and check health."""
        health = await self.llm_client.health_check()
        if not health['service_available']:
            raise Exception("LLM service is not available. Please start Ollama.")
        
        print(f"✅ LLM Agent initialized with {health['models_available']} models")
        return health
    
    async def normalize_shopping_list(self, shopping_list):
        """
        Normalize a shopping list using Qwen for fast processing.
        """
        print(f"🔄 Normalizing {len(shopping_list)} items...")
        
        normalized_items = []
        for item in shopping_list:
            # Normalize ingredient name
            normalized = await self.llm_client.generate(
                PromptTemplates.format_template("NORMALIZE_INGREDIENT", ingredient=item),
                model=ModelType.QWEN_1_5B,
                temperature=0.1,
                max_tokens=20
            )
            
            # Extract brand
            brand = await self.llm_client.generate(
                PromptTemplates.format_template("NORMALIZE_BRAND", product_text=item),
                model=ModelType.QWEN_1_5B,
                temperature=0.1,
                max_tokens=10
            )
            
            # Classify category
            category = await self.llm_client.generate(
                PromptTemplates.format_template("CLASSIFY_PRODUCT_CATEGORY", product_name=normalized.strip()),
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
        
        return normalized_items
    
    async def select_scraping_strategy(self, store_name, query, context):
        """
        Use Phi-3.5 to intelligently select the best scraping strategy.
        """
        print(f"🎯 Selecting strategy for {store_name}: '{query}'")
        
        strategy = await self.llm_client.structured_output(
            PromptTemplates.format_template(
                "SCRAPING_STRATEGY",
                store_name=store_name,
                query=query,
                previous_attempts=context.get('previous_attempts', 0),
                layer1_success=context.get('layer1_success', 70),
                layer2_success=context.get('layer2_success', 95),
                layer3_success=100,
                time_limit=context.get('time_limit', 10),
                user_preference=context.get('user_preference', 'balanced')
            ),
            response_schema={
                "type": "object",
                "properties": {
                    "recommended_strategy": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "estimated_time": {"type": "string"},
                    "success_probability": {"type": "number"}
                }
            },
            model=ModelType.PHI3_5_MINI
        )
        
        return strategy
    
    async def match_products(self, ingredient, available_products):
        """
        Match an ingredient to available products using intelligent matching.
        """
        product_list = '\n'.join([f"- {p}" for p in available_products])
        
        matches = await self.llm_client.structured_output(
            PromptTemplates.format_template(
                "MATCH_PRODUCTS",
                ingredient=ingredient,
                product_list=product_list
            ),
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
        
        return matches.get("matches", [])
    
    async def optimize_shopping_plan(self, items, store_data, preferences):
        """
        Use Phi-3.5 to create an optimized shopping plan.
        """
        print("🧠 Optimizing shopping plan...")
        
        # Format store data for the prompt
        price_comparison = "\n".join([
            f"{store}: {', '.join([f'{item}: {price}' for item, price in prices.items()])}"
            for store, prices in store_data.items()
        ])
        
        optimization = await self.llm_client.structured_output(
            PromptTemplates.format_template(
                "OPTIMIZATION_ADVICE",
                total_items=len(items),
                estimated_cost=sum([item.get('price', 0) for item in items]),
                store_list=list(store_data.keys()),
                preferences=preferences,
                time_available=preferences.get('time_available', '30 minutes'),
                product_data=str(items),
                price_comparison=price_comparison
            ),
            response_schema={
                "type": "object",
                "properties": {
                    "cost_savings": {
                        "type": "object",
                        "properties": {
                            "potential_savings": {"type": "string"},
                            "percentage": {"type": "string"},
                            "best_store_combination": {"type": "array"}
                        }
                    },
                    "time_optimization": {
                        "type": "object",
                        "properties": {
                            "single_store_option": {"type": "string"},
                            "estimated_time": {"type": "string"}
                        }
                    },
                    "priority_recommendations": {"type": "array"}
                }
            },
            model=ModelType.PHI3_5_MINI
        )
        
        return optimization
    
    async def generate_search_variations(self, query):
        """
        Generate search query variations for better product matching.
        """
        variations = await self.llm_client.structured_output(
            PromptTemplates.format_template("GENERATE_SEARCH_VARIATIONS", original_query=query),
            response_schema={
                "type": "object", 
                "properties": {
                    "variations": {"type": "array", "items": {"type": "string"}}
                }
            },
            model=ModelType.QWEN_1_5B
        )
        
        return variations.get("variations", [])
    
    def get_performance_stats(self):
        """Get LLM performance statistics."""
        return self.llm_client.get_model_stats()


async def demo_enhanced_agent():
    """Demonstrate the LLM-enhanced grocery agent."""
    print("🚀 LLM-Enhanced Grocery Price Scanner Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = LLMEnhancedGroceryAgent()
    await agent.initialize()
    
    # Example shopping list
    shopping_list = [
        "2 lbs organic chicken breast",
        "1 gallon whole milk",
        "dozen large eggs",
        "sourdough bread loaf",
        "greek yogurt container"
    ]
    
    print(f"\n📝 Original Shopping List:")
    for i, item in enumerate(shopping_list, 1):
        print(f"   {i}. {item}")
    
    # Step 1: Normalize shopping list
    normalized_items = await agent.normalize_shopping_list(shopping_list)
    
    print(f"\n✨ Normalized Items:")
    for item in normalized_items:
        print(f"   {item['normalized']} | {item['brand']} | {item['category']}")
    
    # Step 2: Strategy selection for different stores
    stores = ["Metro Canada", "Walmart Canada", "FreshCo"]
    
    print(f"\n🎯 Strategy Selection:")
    for store in stores:
        context = {
            'previous_attempts': 2,
            'layer1_success': 75,
            'layer2_success': 95,
            'time_limit': 8,
            'user_preference': 'balanced'
        }
        
        strategy = await agent.select_scraping_strategy(store, "chicken breast", context)
        strategy_num = strategy.get('recommended_strategy', '1')
        confidence = strategy.get('confidence', 0)
        
        print(f"   {store}: Layer {strategy_num} ({confidence:.1%} confidence)")
    
    # Step 3: Product matching example
    query = "greek yogurt"
    available_products = [
        "Chobani Greek Yogurt Plain 32oz",
        "Fage Total 0% Greek Yogurt",
        "Dannon Oikos Triple Zero",
        "Organic Valley Whole Milk",
        "Wonder Bread White"
    ]
    
    print(f"\n🔍 Product Matching for '{query}':")
    matches = await agent.match_products(query, available_products)
    for i, match in enumerate(matches[:3], 1):
        conf = match.get('confidence', 0)
        print(f"   {i}. {match.get('product_name', 'N/A')} ({conf:.1%})")
    
    # Step 4: Shopping optimization
    mock_store_data = {
        "Metro Canada": {"chicken": "$8.99/lb", "milk": "$4.99", "eggs": "$5.99"},
        "Walmart Canada": {"chicken": "$7.99/lb", "milk": "$4.79", "eggs": "$4.99"},
        "FreshCo": {"chicken": "$8.49/lb", "milk": "$4.29", "eggs": "$4.49"}
    }
    
    preferences = {
        "budget_conscious": True,
        "time_available": "30 minutes",
        "organic_preferred": False
    }
    
    optimization = await agent.optimize_shopping_plan(normalized_items, mock_store_data, preferences)
    
    print(f"\n💰 Shopping Optimization:")
    cost_savings = optimization.get("cost_savings", {})
    print(f"   Potential savings: {cost_savings.get('potential_savings', 'N/A')}")
    print(f"   Best stores: {', '.join(cost_savings.get('best_store_combination', []))}")
    
    time_opt = optimization.get("time_optimization", {})
    print(f"   Recommended store: {time_opt.get('single_store_option', 'N/A')}")
    
    # Step 5: Performance statistics
    print(f"\n📊 Performance Statistics:")
    stats = agent.get_performance_stats()
    print(f"   Cache size: {stats['cache_size']} entries")
    print(f"   Available models: {len(stats['available_models'])}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🚀 The LLM-enhanced agent is ready for integration!")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_agent())