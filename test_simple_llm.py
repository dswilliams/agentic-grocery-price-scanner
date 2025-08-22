#!/usr/bin/env python3
"""
Simple test to debug LLM connection issues.
"""

import asyncio
import aiohttp
import json


async def test_direct_api():
    """Test Ollama API directly with aiohttp."""
    print("Testing direct Ollama API connection...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test /api/tags endpoint
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    print(f"✅ Connected! Available models: {models}")
                else:
                    print(f"❌ HTTP {response.status}: {await response.text()}")
                    return False
            
            # Test generation
            print("\nTesting generation...")
            generation_data = {
                "model": "qwen2.5:1.5b",
                "prompt": "Say hello in one word",
                "stream": False
            }
            
            async with session.post(
                "http://localhost:11434/api/generate", 
                json=generation_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Generation successful: {result.get('response', '')}")
                    return True
                else:
                    print(f"❌ Generation failed HTTP {response.status}: {await response.text()}")
                    return False
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


async def test_ollama_client():
    """Test our OllamaClient."""
    print("\nTesting OllamaClient...")
    
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from agentic_grocery_price_scanner.llm_client.ollama_client import OllamaClient, ModelType
        
        client = OllamaClient(timeout=60)  # Increase timeout
        
        # Test model availability
        models = await client.check_model_availability()
        print(f"Available models: {models}")
        
        # Test simple generation
        response = await client.generate(
            "Say hello",
            model=ModelType.QWEN_1_5B,
            max_tokens=10
        )
        print(f"Generated response: {response}")
        
        return True
    
    except Exception as e:
        print(f"❌ OllamaClient error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    success1 = await test_direct_api()
    if success1:
        success2 = await test_ollama_client()
        if success2:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ OllamaClient test failed")
    else:
        print("\n❌ Direct API test failed")


if __name__ == "__main__":
    asyncio.run(main())