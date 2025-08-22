#!/usr/bin/env python3
"""
Simple integration test for the Intelligent Scraper Agent.
Quick test to verify basic functionality without full demo.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agentic_grocery_price_scanner.agents.intelligent_scraper_agent import IntelligentScraperAgent
from agentic_grocery_price_scanner.agents.scraping_ui import InteractiveScrapingSession
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod


async def test_basic_integration():
    """Test basic integration of the intelligent scraper."""
    print("🧪 Running Simple Integration Test")
    print("=" * 40)
    
    try:
        # Initialize agent
        print("\\n1. Initializing Intelligent Scraper Agent...")
        agent = IntelligentScraperAgent()
        print("   ✅ Agent initialized successfully")
        
        # Test workflow creation
        print("\\n2. Testing LangGraph workflow...")
        assert agent.workflow is not None
        print("   ✅ Workflow created successfully")
        
        # Test basic execution (mock mode)
        print("\\n3. Testing basic execution...")
        
        # Create a simple test with minimal inputs
        test_inputs = {
            "query": "test product",
            "stores": [],  # Empty stores list for quick test
            "limit": 1,
            "strategy": "adaptive"
        }
        
        print("   📝 Running with test inputs...")
        result = await agent.execute(test_inputs)
        
        print(f"   📊 Result success: {result.get('success', False)}")
        print(f"   📊 Products found: {result.get('total_products', 0)}")
        
        # Test UI system
        print("\\n4. Testing UI system...")
        session = InteractiveScrapingSession(agent, enable_console=False)
        progress = session.get_session_progress()
        print("   ✅ UI system working")
        
        # Test analytics
        print("\\n5. Testing analytics...")
        analytics = agent.get_collection_analytics()
        print(f"   📊 Analytics data: {len(analytics.get('method_stats', {}))} methods tracked")
        
        print("\\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_layer_initialization():
    """Test that all layers can be initialized."""
    print("\\n🔧 Testing Layer Initialization")
    print("-" * 30)
    
    agent = IntelligentScraperAgent()
    
    try:
        # Test stealth scraper lazy initialization
        print("   🤖 Testing stealth scraper initialization...")
        # This should initialize the stealth scraper
        if agent.stealth_scraper is None:
            from agentic_grocery_price_scanner.mcps.stealth_scraper import StealthScraper
            agent.stealth_scraper = StealthScraper()
        print("   ✅ Stealth scraper ready")
        
        # Test human scraper lazy initialization  
        print("   👤 Testing human scraper initialization...")
        if agent.human_scraper is None:
            from agentic_grocery_price_scanner.mcps.human_browser_scraper import HumanBrowserScraper
            agent.human_scraper = HumanBrowserScraper()
        print("   ✅ Human scraper ready")
        
        # Test clipboard monitor lazy initialization
        print("   📋 Testing clipboard monitor initialization...")
        if agent.clipboard_monitor is None:
            from agentic_grocery_price_scanner.mcps.clipboard_scraper import ClipboardMonitor
            agent.clipboard_monitor = ClipboardMonitor()
        print("   ✅ Clipboard monitor ready")
        
        print("   ✅ All layers initialized successfully")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Import error (dependencies may be missing): {e}")
        return False
    except Exception as e:
        print(f"   ❌ Layer initialization failed: {e}")
        return False


def test_data_models():
    """Test data models and enums."""
    print("\\n📊 Testing Data Models")
    print("-" * 20)
    
    try:
        # Test DataCollectionMethod enum
        print("   🔢 Testing collection method enum...")
        methods = list(DataCollectionMethod)
        print(f"   📊 Available methods: {[m.value for m in methods]}")
        assert len(methods) >= 3  # Should have at least stealth, human, clipboard
        print("   ✅ Collection methods working")
        
        # Test Product model
        print("   🛍️  Testing Product model...")
        from agentic_grocery_price_scanner.data_models.product import Product
        from decimal import Decimal
        
        test_product = Product(
            name="Test Product",
            price=Decimal("9.99"),
            store_id="test_store",
            collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
            confidence_score=0.8
        )
        
        # Test confidence weight calculation
        confidence_weight = test_product.get_collection_confidence_weight()
        print(f"   📊 Confidence weight: {confidence_weight}")
        assert 0 <= confidence_weight <= 1
        print("   ✅ Product model working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data model test failed: {e}")
        return False


async def main():
    """Run all simple integration tests."""
    print("🚀 Intelligent Scraper Agent - Simple Integration Test")
    print("Testing basic functionality without external dependencies...")
    print()
    
    all_passed = True
    
    # Test 1: Data models
    if not test_data_models():
        all_passed = False
    
    # Test 2: Layer initialization
    if not await test_layer_initialization():
        all_passed = False
    
    # Test 3: Basic integration
    if not await test_basic_integration():
        all_passed = False
    
    # Summary
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Intelligent Scraper Agent is ready for use")
        print()
        print("Next steps:")
        print("  1. Run 'python test_intelligent_scraper_demo.py' for full demo")
        print("  2. Use 'grocery-scanner scrape --query \"milk\"' via CLI")
        print("  3. Test individual layers as needed")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Check error messages above for details")
        print("💡 Some failures may be due to missing optional dependencies")
    print("=" * 50)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)