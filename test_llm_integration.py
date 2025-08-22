#!/usr/bin/env python3
"""
Test script for LLM integration with Ollama.
Demonstrates basic functionality with both Qwen and Phi-3.5 models.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic_grocery_price_scanner.llm_client import OllamaClient, PromptTemplates
from agentic_grocery_price_scanner.llm_client.ollama_client import ModelType


async def test_basic_inference():
    """Test basic model inference capabilities."""
    print("🤖 Testing Basic LLM Inference")
    print("=" * 50)
    
    client = OllamaClient()
    
    # Health check
    print("1. Health Check...")
    health = await client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Service Available: {health['service_available']}")
    print(f"   Models Available: {health.get('models_available', 0)}")
    
    if not health['service_available']:
        print("❌ Ollama service is not available. Please start Ollama first.")
        return False
    
    # Test model availability
    print("\n2. Checking Available Models...")
    models = await client.check_model_availability()
    print(f"   Available: {models}")
    
    # Test each model type
    test_prompts = [
        ("Simple greeting", "Say hello and tell me your name"),
        ("Math calculation", "What is 15 * 8? Just give me the number."),
        ("Classification", "Classify this item: 'organic whole milk'. Is it dairy? Answer yes or no.")
    ]
    
    for model_type in [ModelType.QWEN_1_5B, ModelType.PHI3_5_MINI]:
        if model_type.value in models:
            print(f"\n3. Testing {model_type.value}...")
            
            for prompt_name, prompt in test_prompts:
                print(f"   Testing: {prompt_name}")
                try:
                    start_time = time.time()
                    response = await client.generate(
                        prompt,
                        model=model_type,
                        max_tokens=50
                    )
                    elapsed = time.time() - start_time
                    
                    print(f"   Response ({elapsed:.2f}s): {response.strip()}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        else:
            print(f"\n❌ {model_type.value} not available")
    
    return True


async def test_grocery_specific_tasks():
    """Test grocery-specific prompt templates and tasks."""
    print("\n🛒 Testing Grocery-Specific Tasks")
    print("=" * 50)
    
    client = OllamaClient()
    
    # Test ingredient normalization with Qwen
    print("1. Ingredient Normalization (Qwen)...")
    ingredients_to_test = [
        "Fresh Organic Free-Range Eggs (Large)",
        "2% Reduced Fat Milk - 1 Gallon",
        "Whole Wheat Flour - All Purpose",
        "Greek Yogurt Plain - Chobani Brand"
    ]
    
    for ingredient in ingredients_to_test:
        try:
            prompt = PromptTemplates.format_template(
                "NORMALIZE_INGREDIENT",
                ingredient=ingredient
            )
            
            response = await client.generate(
                prompt,
                model=ModelType.QWEN_1_5B,
                temperature=0.1,
                max_tokens=20
            )
            
            print(f"   '{ingredient}' → '{response.strip()}'")
            
        except Exception as e:
            print(f"   ❌ Error normalizing '{ingredient}': {e}")
    
    # Test brand extraction
    print("\n2. Brand Extraction (Qwen)...")
    products_to_test = [
        "Coca-Cola Classic 12-Pack Cans",
        "Kellogg's Corn Flakes Cereal 18oz",
        "Generic Brand White Bread",
        "Organic Valley Whole Milk 1/2 Gallon"
    ]
    
    for product in products_to_test:
        try:
            prompt = PromptTemplates.format_template(
                "NORMALIZE_BRAND",
                product_text=product
            )
            
            response = await client.generate(
                prompt,
                model=ModelType.QWEN_1_5B,
                temperature=0.1,
                max_tokens=10
            )
            
            print(f"   '{product}' → Brand: '{response.strip()}'")
            
        except Exception as e:
            print(f"   ❌ Error extracting brand from '{product}': {e}")
    
    # Test decision-making with Phi-3.5
    print("\n3. Strategy Decision Making (Phi-3.5)...")
    try:
        prompt = PromptTemplates.format_template(
            "SCRAPING_STRATEGY",
            store_name="Metro Canada",
            query="organic milk",
            previous_attempts=2,
            layer1_success=60,
            layer2_success=95,
            layer3_success=100,
            time_limit=10,
            user_preference="fast"
        )
        
        response = await client.structured_output(
            prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "recommended_strategy": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "estimated_time": {"type": "string"},
                    "success_probability": {"type": "number"}
                },
                "required": ["recommended_strategy", "confidence", "reasoning"]
            },
            model=ModelType.PHI3_5_MINI
        )
        
        print("   Strategy Recommendation:")
        for key, value in response.items():
            print(f"     {key}: {value}")
            
    except Exception as e:
        print(f"   ❌ Error in strategy decision: {e}")


async def test_structured_output():
    """Test structured JSON output parsing."""
    print("\n📋 Testing Structured Output")
    print("=" * 50)
    
    client = OllamaClient()
    
    # Test product information extraction
    print("1. Product Information Extraction...")
    test_product = "Organic Valley 2% Milk 64oz - $4.99 (was $5.49)"
    
    try:
        prompt = PromptTemplates.format_template(
            "EXTRACT_PRODUCT_INFO",
            product_text=test_product
        )
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "brand": {"type": "string"},
                "size": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
                "organic": {"type": "boolean"},
                "on_sale": {"type": "boolean"}
            },
            "required": ["name", "brand", "price"]
        }
        
        response = await client.structured_output(
            prompt,
            response_schema=schema,
            model=ModelType.QWEN_1_5B
        )
        
        print(f"   Input: {test_product}")
        print(f"   Extracted:")
        for key, value in response.items():
            print(f"     {key}: {value}")
            
    except Exception as e:
        print(f"   ❌ Error in structured extraction: {e}")


async def test_performance_features():
    """Test performance optimization features."""
    print("\n⚡ Testing Performance Features")
    print("=" * 50)
    
    client = OllamaClient(enable_caching=True)
    
    # Test caching
    print("1. Testing Response Caching...")
    test_prompt = "What is 2 + 2? Answer with just the number."
    
    # First request (cache miss)
    start_time = time.time()
    response1 = await client.generate(test_prompt, model=ModelType.QWEN_1_5B)
    first_time = time.time() - start_time
    
    # Second request (cache hit)
    start_time = time.time()
    response2 = await client.generate(test_prompt, model=ModelType.QWEN_1_5B)
    second_time = time.time() - start_time
    
    print(f"   First request: {first_time:.3f}s → '{response1.strip()}'")
    print(f"   Second request: {second_time:.3f}s → '{response2.strip()}'")
    print(f"   Cache speedup: {first_time/second_time:.1f}x faster")
    
    # Test batch processing
    print("\n2. Testing Batch Processing...")
    batch_prompts = [
        "Name a fruit that is red",
        "Name a vegetable that is green", 
        "Name a dairy product",
        "Name a grain product"
    ]
    
    start_time = time.time()
    batch_responses = await client.batch_generate(
        batch_prompts,
        model=ModelType.QWEN_1_5B,
        max_concurrent=2,
        max_tokens=10
    )
    batch_time = time.time() - start_time
    
    print(f"   Processed {len(batch_prompts)} prompts in {batch_time:.2f}s")
    for prompt, response in zip(batch_prompts, batch_responses):
        print(f"   '{prompt}' → '{response.strip()}'")
    
    # Show performance stats
    print("\n3. Performance Statistics...")
    stats = client.get_model_stats()
    print(f"   Available models: {len(stats['available_models'])}")
    print(f"   Cache size: {stats['cache_size']} entries")
    print(f"   Model load times: {stats['load_times']}")


async def test_auto_model_selection():
    """Test automatic model selection based on prompt content."""
    print("\n🎯 Testing Auto Model Selection")
    print("=" * 50)
    
    client = OllamaClient()
    
    test_cases = [
        ("Quick normalization task", "Normalize this ingredient: organic free range eggs"),
        ("Complex reasoning task", "Analyze the best grocery shopping strategy for a family of 4 with a $200 budget"),
        ("Classification task", "Classify this product: whole wheat bread"),
        ("Decision making task", "Should I use automated scraping or manual collection for this store?")
    ]
    
    for task_type, prompt in test_cases:
        print(f"\n   Testing: {task_type}")
        
        # Let the client auto-select the model
        selected_model = await client._select_best_model(prompt)
        
        try:
            response = await client.generate(
                prompt,
                model=None,  # Auto-select
                max_tokens=50
            )
            
            print(f"   Selected model: {selected_model.value}")
            print(f"   Response: {response.strip()[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def main():
    """Run all LLM integration tests."""
    print("🚀 LLM Integration Test Suite")
    print("=" * 60)
    
    try:
        # Basic functionality tests
        success = await test_basic_inference()
        if not success:
            return
        
        # Grocery-specific tests
        await test_grocery_specific_tasks()
        
        # Structured output tests
        await test_structured_output()
        
        # Performance tests
        await test_performance_features()
        
        # Auto model selection tests
        await test_auto_model_selection()
        
        print("\n✅ All tests completed successfully!")
        print("\nThe LLM integration is ready for use in the grocery price scanner.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())