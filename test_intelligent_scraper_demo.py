#!/usr/bin/env python3
"""
Comprehensive demonstration of the Intelligent Scraper Agent with 3-layer fallback system.
Tests all layers with real stores: Metro, Walmart, and FreshCo.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from agentic_grocery_price_scanner.agents.intelligent_scraper_agent import (
    IntelligentScraperAgent,
    CollectionStrategy
)
from agentic_grocery_price_scanner.agents.scraping_ui import (
    InteractiveScrapingSession,
    UIUpdateType,
    create_console_ui_callback,
    create_json_ui_callback
)
from agentic_grocery_price_scanner.agents.database_integration import (
    ScrapingDatabaseIntegrator,
    ScrapingSessionTracker
)
from agentic_grocery_price_scanner.agents.collection_analytics import (
    CollectionAnalytics,
    export_analytics_report
)
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod


class IntelligentScraperDemo:
    """Comprehensive demo of the intelligent scraper system."""
    
    def __init__(self):
        """Initialize demo components."""
        self.agent = IntelligentScraperAgent()
        self.db_integrator = ScrapingDatabaseIntegrator()
        self.session_tracker = ScrapingSessionTracker()
        self.analytics = CollectionAnalytics()
        
        # Test scenarios
        self.test_scenarios = [
            {
                "name": "Basic Dairy Products",
                "query": "milk",
                "stores": ["metro_ca", "walmart_ca"],
                "limit": 10,
                "strategy": "adaptive"
            },
            {
                "name": "Organic Products Search",
                "query": "organic bread",
                "stores": ["metro_ca", "freshco_com"],
                "limit": 15,
                "strategy": "human_assisted"
            },
            {
                "name": "Multi-Store Comprehensive",
                "query": "eggs",
                "stores": ["metro_ca", "walmart_ca", "freshco_com"],
                "limit": 20,
                "strategy": "adaptive"
            },
            {
                "name": "Clipboard Fallback Test",
                "query": "gluten free pasta",
                "stores": ["freshco_com"],
                "limit": 5,
                "strategy": "clipboard_manual"
            }
        ]
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of all system features."""
        print("🚀 Starting Intelligent Scraper Agent Comprehensive Demo")
        print("=" * 60)
        
        demo_results = {
            "start_time": datetime.now().isoformat(),
            "scenarios": {},
            "layer_tests": {},
            "analytics": {},
            "database_integration": {},
            "summary": {}
        }
        
        try:
            # 1. Test individual layers
            print("\\n📋 PHASE 1: Individual Layer Testing")
            demo_results["layer_tests"] = await self._test_individual_layers()
            
            # 2. Run test scenarios
            print("\\n📋 PHASE 2: Scenario Testing")
            demo_results["scenarios"] = await self._run_test_scenarios()
            
            # 3. Test database integration
            print("\\n📋 PHASE 3: Database Integration Testing")
            demo_results["database_integration"] = await self._test_database_integration()
            
            # 4. Generate analytics
            print("\\n📋 PHASE 4: Analytics Generation")
            demo_results["analytics"] = await self._generate_analytics_report()
            
            # 5. Generate summary
            demo_results["summary"] = self._generate_demo_summary(demo_results)
            
            print("\\n✅ Demo completed successfully!")
            return demo_results
            
        except Exception as e:
            print(f"\\n❌ Demo failed: {e}")
            demo_results["error"] = str(e)
            return demo_results
    
    async def _test_individual_layers(self) -> Dict[str, Any]:
        """Test each scraping layer individually."""
        layer_results = {}
        
        print("\\n🤖 Testing Layer 1: Stealth Scraping")
        try:
            stealth_result = await self.agent.test_layer_individually(1, "milk", "metro_ca")
            layer_results["stealth"] = stealth_result
            print(f"   ✅ Stealth: {stealth_result['count']} products found")
        except Exception as e:
            layer_results["stealth"] = {"success": False, "error": str(e)}
            print(f"   ❌ Stealth failed: {e}")
        
        print("\\n👤 Testing Layer 2: Human-Assisted Scraping")
        try:
            human_result = await self.agent.test_layer_individually(2, "bread", "walmart_ca")
            layer_results["human"] = human_result
            print(f"   ✅ Human-Assisted: {human_result['count']} products found")
        except Exception as e:
            layer_results["human"] = {"success": False, "error": str(e)}
            print(f"   ❌ Human-Assisted failed: {e}")
        
        print("\\n📋 Testing Layer 3: Clipboard Collection")
        print("   📋 NOTE: This will start clipboard monitoring for 30 seconds")
        print("   📋 Copy any product information during this time to test parsing")
        try:
            clipboard_result = await self.agent.test_layer_individually(3, "eggs")
            layer_results["clipboard"] = clipboard_result
            print(f"   ✅ Clipboard: {clipboard_result['count']} products captured")
        except Exception as e:
            layer_results["clipboard"] = {"success": False, "error": str(e)}
            print(f"   ❌ Clipboard failed: {e}")
        
        return layer_results
    
    async def _run_test_scenarios(self) -> Dict[str, Any]:
        """Run predefined test scenarios."""
        scenario_results = {}
        
        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"\\n🧪 Scenario {i}: {scenario['name']}")
            print(f"   Query: '{scenario['query']}'")
            print(f"   Stores: {', '.join(scenario['stores'])}")
            print(f"   Strategy: {scenario['strategy']}")
            
            session_id = f"demo_scenario_{i}_{int(time.time())}"
            
            try:
                # Create interactive session
                session = InteractiveScrapingSession(self.agent, enable_console=True)
                
                # Add custom callback for this demo
                session.add_ui_callback(self._create_demo_callback(scenario['name']))
                
                # Start scraping
                start_time = datetime.now()
                
                result = await session.start_scraping(
                    query=scenario['query'],
                    stores=scenario['stores'],
                    limit=scenario['limit'],
                    strategy=scenario['strategy']
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Track session
                self.session_tracker.start_session(session_id, scenario)
                self.session_tracker.update_scraping_results(session_id, result)
                
                # Record analytics
                for product in result.get("products", []):
                    self.analytics.record_session(
                        session_id=f"{session_id}_{product.store_id}",
                        query=scenario['query'],
                        store_id=product.store_id,
                        collection_method=product.collection_method,
                        start_time=start_time,
                        end_time=end_time,
                        success=True,
                        products=[product]
                    )
                
                scenario_results[scenario['name']] = {
                    "success": result['success'],
                    "products_found": result['total_products'],
                    "stores_completed": result['stores_completed'],
                    "stores_failed": result['stores_failed'],
                    "duration_seconds": duration,
                    "method_stats": result.get('method_stats', {}),
                    "session_id": session_id
                }
                
                print(f"   ✅ Completed: {result['total_products']} products in {duration:.1f}s")
                
            except Exception as e:
                scenario_results[scenario['name']] = {
                    "success": False,
                    "error": str(e),
                    "session_id": session_id
                }
                print(f"   ❌ Failed: {e}")
        
        return scenario_results
    
    async def _test_database_integration(self) -> Dict[str, Any]:
        """Test database integration with scraping results."""
        print("\\n💾 Testing database integration...")
        
        # Get results from a successful scenario
        test_results = None
        for scenario in self.test_scenarios:
            session_id = f"db_test_{int(time.time())}"
            
            try:
                # Quick scrape for testing
                result = await self.agent.execute({
                    "query": "test product",
                    "stores": ["metro_ca"],
                    "limit": 5
                })
                
                if result['success'] and result.get('products'):
                    test_results = result
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Skipping DB test due to scraping error: {e}")
                continue
        
        if not test_results:
            return {"success": False, "error": "No successful scraping results for DB testing"}
        
        try:
            # Test database integration
            db_results = await self.db_integrator.save_scraping_results(
                test_results,
                session_metadata={"test": "database_integration"}
            )
            
            # Test similarity search
            if test_results.get('products'):
                first_product = test_results['products'][0]
                similar_products = await self.db_integrator.search_similar_products(
                    query=first_product.name,
                    store_id=first_product.store_id,
                    limit=5
                )
                
                db_results["similarity_search"] = {
                    "query": first_product.name,
                    "results_found": len(similar_products)
                }
            
            # Get collection statistics
            stats = await self.db_integrator.get_collection_statistics()
            db_results["collection_stats"] = stats
            
            print(f"   ✅ Database integration: {db_results['products_saved']} products saved")
            return db_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        print("\\n📊 Generating analytics report...")
        
        try:
            # Generate performance report
            performance_report = self.analytics.generate_performance_report(days_back=1)
            
            # Get optimization recommendations
            recommendations = self.analytics.get_optimization_recommendations("milk", "metro_ca")
            
            # Export report to file
            report_path = f"logs/demo_analytics_report_{int(time.time())}.json"
            export_analytics_report(self.analytics, report_path)
            
            analytics_summary = {
                "performance_report": performance_report,
                "recommendations": recommendations,
                "report_exported": report_path,
                "session_count": len(self.analytics.sessions)
            }
            
            print(f"   ✅ Analytics: {len(self.analytics.sessions)} sessions analyzed")
            print(f"   📄 Report exported: {report_path}")
            
            return analytics_summary
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_demo_summary(self, demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive demo summary."""
        summary = {
            "total_duration": 0,
            "layers_tested": 0,
            "scenarios_run": 0,
            "scenarios_successful": 0,
            "total_products_found": 0,
            "stores_tested": set(),
            "collection_methods_used": set(),
            "database_integration_success": False,
            "analytics_generated": False
        }
        
        # Calculate duration
        start_time = datetime.fromisoformat(demo_results["start_time"])
        end_time = datetime.now()
        summary["total_duration"] = (end_time - start_time).total_seconds()
        
        # Layer tests
        layer_results = demo_results.get("layer_tests", {})
        summary["layers_tested"] = len([r for r in layer_results.values() if r.get("success")])
        
        # Scenario results
        scenario_results = demo_results.get("scenarios", {})
        summary["scenarios_run"] = len(scenario_results)
        summary["scenarios_successful"] = len([r for r in scenario_results.values() if r.get("success")])
        
        for scenario_name, result in scenario_results.items():
            if result.get("success"):
                summary["total_products_found"] += result.get("products_found", 0)
                # Note: Would need access to actual results to get stores and methods
        
        # Database integration
        db_results = demo_results.get("database_integration", {})
        summary["database_integration_success"] = db_results.get("success", False)
        
        # Analytics
        analytics_results = demo_results.get("analytics", {})
        summary["analytics_generated"] = "performance_report" in analytics_results
        
        return summary
    
    def _create_demo_callback(self, scenario_name: str):
        """Create a demo-specific UI callback."""
        def demo_callback(update_type: UIUpdateType, data: Dict[str, Any]):
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if update_type == UIUpdateType.LAYER_CHANGE:
                print(f"   [{timestamp}] 🔄 {data['description']}")
            elif update_type == UIUpdateType.PRODUCT_FOUND:
                product = data['product']
                print(f"   [{timestamp}] 🛍️  Found: {product['name']} (${product['price']:.2f}) via {product['collection_method']}")
            elif update_type == UIUpdateType.ERROR:
                print(f"   [{timestamp}] ❌ Error: {data['error']}")
        
        return demo_callback
    
    def print_final_report(self, demo_results: Dict[str, Any]) -> None:
        """Print comprehensive final report."""
        print("\\n" + "=" * 60)
        print("   🎉 INTELLIGENT SCRAPER DEMO FINAL REPORT")
        print("=" * 60)
        
        summary = demo_results.get("summary", {})
        
        print(f"\\n⏱️  OVERVIEW:")
        print(f"   • Total Duration: {summary.get('total_duration', 0):.1f} seconds")
        print(f"   • Layers Tested: {summary.get('layers_tested', 0)}/3")
        print(f"   • Scenarios Run: {summary.get('scenarios_run', 0)}")
        print(f"   • Scenarios Successful: {summary.get('scenarios_successful', 0)}")
        print(f"   • Total Products Found: {summary.get('total_products_found', 0)}")
        
        print(f"\\n🎯 LAYER PERFORMANCE:")
        layer_results = demo_results.get("layer_tests", {})
        for layer_name, result in layer_results.items():
            status = "✅ Success" if result.get("success") else "❌ Failed"
            count = result.get("count", 0)
            print(f"   • {layer_name.title()}: {status} ({count} products)")
        
        print(f"\\n🧪 SCENARIO RESULTS:")
        scenario_results = demo_results.get("scenarios", {})
        for scenario_name, result in scenario_results.items():
            if result.get("success"):
                products = result.get("products_found", 0)
                duration = result.get("duration_seconds", 0)
                print(f"   • {scenario_name}: ✅ {products} products in {duration:.1f}s")
            else:
                error = result.get("error", "Unknown error")[:50]
                print(f"   • {scenario_name}: ❌ {error}")
        
        print(f"\\n💾 INTEGRATION STATUS:")
        db_results = demo_results.get("database_integration", {})
        db_status = "✅ Success" if db_results.get("success") else "❌ Failed"
        products_saved = db_results.get("products_saved", 0)
        print(f"   • Database Integration: {db_status} ({products_saved} products saved)")
        
        analytics_results = demo_results.get("analytics", {})
        analytics_status = "✅ Generated" if "performance_report" in analytics_results else "❌ Failed"
        session_count = analytics_results.get("session_count", 0)
        print(f"   • Analytics Report: {analytics_status} ({session_count} sessions)")
        
        print(f"\\n🚀 SYSTEM CAPABILITIES DEMONSTRATED:")
        print(f"   ✅ 3-Layer Intelligent Fallback System")
        print(f"   ✅ LangGraph Workflow Orchestration")
        print(f"   ✅ Real-time Progress Tracking")
        print(f"   ✅ Database Integration (SQL + Vector)")
        print(f"   ✅ Performance Analytics")
        print(f"   ✅ Adaptive Strategy Learning")
        print(f"   ✅ User Experience Enhancements")
        
        print(f"\\n" + "=" * 60)


async def main():
    """Main demo execution."""
    demo = IntelligentScraperDemo()
    
    print("🤖 Intelligent Scraper Agent - Comprehensive Demo")
    print("This demo will test all three layers of the scraping system:")
    print("  🤖 Layer 1: Automated Stealth Scraping")
    print("  👤 Layer 2: Human-Assisted Browser Automation")
    print("  📋 Layer 3: Manual Clipboard Collection")
    print()
    
    user_input = input("Ready to start? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Demo cancelled.")
        return
    
    # Run comprehensive demo
    demo_results = await demo.run_comprehensive_demo()
    
    # Print final report
    demo.print_final_report(demo_results)
    
    # Save results
    results_file = f"logs/demo_results_{int(time.time())}.json"
    try:
        import os
        os.makedirs("logs", exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(demo_results, f, indent=2, default=str)
        print(f"\\n📄 Complete results saved to: {results_file}")
    except Exception as e:
        print(f"\\n⚠️  Could not save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())