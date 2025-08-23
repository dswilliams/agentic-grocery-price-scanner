"""
Simple test for the master workflow to validate basic functionality.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test_basic_workflow():
    """Test basic workflow with string ingredients."""
    from agentic_grocery_price_scanner.workflow import GroceryWorkflow
    
    print("🧪 Testing basic master workflow...")
    
    try:
        workflow = GroceryWorkflow(enable_checkpointing=False)  # Disable checkpointing to avoid serialization issues
        
        ingredients = ["milk", "bread", "eggs"]
        
        config = {
            "scraping_strategy": "adaptive",
            "matching_strategy": "adaptive",
            "optimization_strategy": "balanced",
            "target_stores": ["metro_ca", "walmart_ca"],
            "max_stores": 2,
            "workflow_timeout": 60,
            "enable_parallel_scraping": False,  # Sequential for simplicity
            "enable_parallel_matching": False
        }
        
        def progress_callback(info):
            stage = info.get("stage", "unknown")
            message = info.get("message", "")
            print(f"🔄 [{stage}] {message}")
        
        print("🚀 Starting workflow...")
        
        result = await workflow.execute(
            recipes=None,
            ingredients=ingredients,
            config=config,
            progress_callback=progress_callback
        )
        
        print("✅ Workflow completed!")
        print(f"   Execution ID: {result.get('execution_id', 'unknown')}")
        print(f"   Status: {result.get('workflow_status', 'unknown')}")
        
        # Check execution metrics
        metrics = result.get("execution_metrics")
        if metrics:
            print(f"   Total time: {metrics.total_execution_time:.2f}s")
            print(f"   Ingredients: {metrics.total_ingredients}")
            print(f"   Status: {metrics.status}")
        
        # Check workflow summary
        summary = result.get("workflow_summary", {})
        if summary:
            print(f"   Summary:")
            print(f"     Ingredients processed: {summary.get('ingredients_processed', 0)}")
            print(f"     Products collected: {summary.get('products_collected', 0)}")
            print(f"     Matches found: {summary.get('matches_found', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_basic_workflow()
    
    if success:
        print("\n🎉 Basic workflow test passed!")
    else:
        print("\n💥 Basic workflow test failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())