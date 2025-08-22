"""
LangGraph-based intelligent scraper agent with 3-layer fallback system.
This agent implements smart routing through stealth → human assistance → clipboard collection.
"""

import asyncio
import json
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable, TypedDict
from enum import Enum
from datetime import datetime
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..data_models import Product, StoreConfig
from ..data_models.base import DataCollectionMethod
from ..config import load_store_configs
from ..mcps.stealth_scraper import StealthScraper, StealthConfig
from ..mcps.human_browser_scraper import HumanBrowserScraper, BrowserProfile
from ..mcps.clipboard_scraper import ClipboardMonitor, ClipboardProduct
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CollectionStrategy(Enum):
    """Strategy for product data collection."""
    
    AUTO_STEALTH = "auto_stealth"
    HUMAN_ASSISTED = "human_assisted"
    CLIPBOARD_MANUAL = "clipboard_manual"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ScrapingState(TypedDict):
    """State structure for LangGraph scraping workflow."""
    
    query: str
    stores: List[str]
    limit: int
    strategy: CollectionStrategy
    current_layer: int
    products: List[Product]
    errors: Dict[str, str]
    success_rates: Dict[str, float]
    user_assistance_active: bool
    clipboard_monitoring: bool
    completed_stores: List[str]
    failed_stores: List[str]
    collection_metadata: Dict[str, Any]
    progress_callback: Optional[Callable[[str], None]]


class IntelligentScraperAgent(BaseAgent):
    """LangGraph-based intelligent scraper with 3-layer fallback system."""
    
    def __init__(self):
        """Initialize the intelligent scraper agent."""
        super().__init__("intelligent_scraper")
        
        # Initialize scraping components
        self.stealth_scraper = None  # Lazy initialization
        self.human_scraper = None    # Lazy initialization
        self.clipboard_monitor = None  # Lazy initialization
        
        # Collection method statistics
        self.method_stats = {
            DataCollectionMethod.AUTOMATED_STEALTH: {"attempts": 0, "successes": 0},
            DataCollectionMethod.HUMAN_BROWSER: {"attempts": 0, "successes": 0},
            DataCollectionMethod.CLIPBOARD_MANUAL: {"attempts": 0, "successes": 0}
        }
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute intelligent scraping with LangGraph workflow.
        
        Args:
            inputs: Dictionary containing:
                - query: Search query string
                - stores: List of store IDs (optional)
                - limit: Maximum products per store (optional)
                - strategy: Collection strategy (optional)
                - progress_callback: Callback for progress updates (optional)
        
        Returns:
            Dictionary with scraped products and collection metadata
        """
        query = inputs.get("query")
        if not query:
            raise ValueError("Query is required for scraping")
        
        # Initialize state
        initial_state: ScrapingState = {
            "query": query,
            "stores": inputs.get("stores", []),
            "limit": inputs.get("limit", 50),
            "strategy": CollectionStrategy(inputs.get("strategy", "adaptive")),
            "current_layer": 1,
            "products": [],
            "errors": {},
            "success_rates": {},
            "user_assistance_active": False,
            "clipboard_monitoring": False,
            "completed_stores": [],
            "failed_stores": [],
            "collection_metadata": {
                "start_time": datetime.now().isoformat(),
                "layers_attempted": [],
                "user_interactions": []
            },
            "progress_callback": inputs.get("progress_callback")
        }
        
        self.log_info(f"Starting intelligent scraping for query: '{query}'")
        
        # Execute workflow
        try:
            config = {"configurable": {"thread_id": f"scrape_{int(time.time())}"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Compile results
            result = {
                "success": True,
                "query": query,
                "products": final_state["products"],
                "total_products": len(final_state["products"]),
                "stores_completed": len(final_state["completed_stores"]),
                "stores_failed": len(final_state["failed_stores"]),
                "errors": final_state["errors"],
                "collection_metadata": final_state["collection_metadata"],
                "method_stats": self._calculate_method_stats(final_state["products"]),
                "success_rates": final_state["success_rates"]
            }
            
            self.log_info(f"Scraping completed: {len(final_state['products'])} total products")
            return result
            
        except Exception as e:
            self.log_error(f"Workflow execution failed: {e}", e)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "products": [],
                "total_products": 0
            }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for intelligent scraping."""
        workflow = StateGraph(ScrapingState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_scraping)
        workflow.add_node("try_stealth", self._try_stealth_scraping)
        workflow.add_node("evaluate_stealth", self._evaluate_stealth_results)
        workflow.add_node("escalate_to_human", self._escalate_to_human)
        workflow.add_node("try_human", self._try_human_scraping)
        workflow.add_node("evaluate_human", self._evaluate_human_results)
        workflow.add_node("enable_clipboard", self._enable_clipboard_mode)
        workflow.add_node("try_clipboard", self._try_clipboard_collection)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("adaptive_learning", self._adaptive_learning)
        workflow.add_node("finalize", self._finalize_scraping)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "try_stealth")
        workflow.add_edge("try_stealth", "evaluate_stealth")
        
        # Conditional routing from stealth evaluation
        workflow.add_conditional_edges(
            "evaluate_stealth",
            self._should_escalate_from_stealth,
            {
                "continue_stealth": "aggregate_results",
                "escalate_human": "escalate_to_human",
                "skip_to_clipboard": "enable_clipboard"
            }
        )
        
        workflow.add_edge("escalate_to_human", "try_human")
        workflow.add_edge("try_human", "evaluate_human")
        
        # Conditional routing from human evaluation
        workflow.add_conditional_edges(
            "evaluate_human",
            self._should_escalate_from_human,
            {
                "continue_human": "aggregate_results",
                "escalate_clipboard": "enable_clipboard"
            }
        )
        
        workflow.add_edge("enable_clipboard", "try_clipboard")
        workflow.add_edge("try_clipboard", "aggregate_results")
        workflow.add_edge("aggregate_results", "adaptive_learning")
        workflow.add_edge("adaptive_learning", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    # Workflow Node Functions
    
    async def _initialize_scraping(self, state: ScrapingState) -> ScrapingState:
        """Initialize scraping session and load store configurations."""
        self.log_info("Initializing scraping session")
        
        if state["progress_callback"]:
            state["progress_callback"]("🚀 Initializing intelligent scraping system...")
        
        try:
            # Load store configurations
            all_stores = load_store_configs()
            
            # Filter to requested stores or use all active
            if state["stores"]:
                stores = [s for s in state["stores"] if s in all_stores]
            else:
                stores = [k for k, v in all_stores.items() if getattr(v, 'active', True)]
            
            state["stores"] = stores
            state["collection_metadata"]["stores_to_scrape"] = stores
            state["collection_metadata"]["initialization_time"] = datetime.now().isoformat()
            
            self.log_info(f"Initialized for {len(stores)} stores: {stores}")
            
        except Exception as e:
            self.log_error(f"Initialization failed: {e}", e)
            state["errors"]["initialization"] = str(e)
        
        return state
    
    async def _try_stealth_scraping(self, state: ScrapingState) -> ScrapingState:
        """Attempt Layer 1: Automated stealth scraping."""
        self.log_info("Attempting Layer 1: Stealth scraping")
        
        if state["progress_callback"]:
            state["progress_callback"]("🤖 Layer 1: Attempting stealth scraping...")
        
        state["current_layer"] = 1
        state["collection_metadata"]["layers_attempted"].append("stealth")
        
        # Lazy initialize stealth scraper
        if not self.stealth_scraper:
            self.stealth_scraper = StealthScraper()
        
        try:
            config = StealthConfig(
                headless=True,
                enable_stealth=True,
                navigation_timeout=30000
            )
            
            stealth_products = []
            
            for store_id in state["stores"]:
                try:
                    self.method_stats[DataCollectionMethod.AUTOMATED_STEALTH]["attempts"] += 1
                    
                    self.log_debug(f"Stealth scraping {store_id}")
                    products = await self.stealth_scraper.scrape_store(
                        store_id=store_id,
                        query=state["query"],
                        limit=state["limit"],
                        config=config
                    )
                    
                    # Mark products with collection method
                    for product in products:
                        product.collection_method = DataCollectionMethod.AUTOMATED_STEALTH
                        product.confidence_score = 0.8
                    
                    stealth_products.extend(products)
                    state["completed_stores"].append(store_id)
                    self.method_stats[DataCollectionMethod.AUTOMATED_STEALTH]["successes"] += 1
                    
                    self.log_info(f"Stealth scraped {len(products)} products from {store_id}")
                    
                except Exception as e:
                    self.log_error(f"Stealth scraping failed for {store_id}: {e}")
                    state["failed_stores"].append(store_id)
                    state["errors"][f"stealth_{store_id}"] = str(e)
            
            state["products"].extend(stealth_products)
            
        except Exception as e:
            self.log_error(f"Stealth scraping layer failed: {e}", e)
            state["errors"]["stealth_layer"] = str(e)
        
        return state
    
    async def _evaluate_stealth_results(self, state: ScrapingState) -> ScrapingState:
        """Evaluate stealth scraping results and decide next action."""
        stealth_success_rate = len(state["completed_stores"]) / max(len(state["stores"]), 1)
        state["success_rates"]["stealth"] = stealth_success_rate
        
        self.log_info(f"Stealth success rate: {stealth_success_rate:.2%}")
        
        if state["progress_callback"]:
            state["progress_callback"](
                f"🤖 Layer 1 complete: {stealth_success_rate:.0%} success rate "
                f"({len(state['products'])} products collected)"
            )
        
        return state
    
    def _should_escalate_from_stealth(self, state: ScrapingState) -> str:
        """Decide whether to escalate from stealth layer."""
        success_rate = state["success_rates"].get("stealth", 0)
        products_count = len(state["products"])
        
        # Continue with stealth if success rate > 80% or enough products collected
        if success_rate > 0.8 or products_count >= state["limit"] * 0.7:
            return "continue_stealth"
        
        # Skip to clipboard if completely failed
        if success_rate == 0 and len(state["failed_stores"]) == len(state["stores"]):
            return "skip_to_clipboard"
        
        # Otherwise escalate to human assistance
        return "escalate_human"
    
    async def _escalate_to_human(self, state: ScrapingState) -> ScrapingState:
        """Prepare for human-assisted scraping."""
        self.log_info("Escalating to Layer 2: Human-assisted scraping")
        
        if state["progress_callback"]:
            state["progress_callback"]("👤 Layer 2: Preparing human-assisted scraping...")
        
        state["user_assistance_active"] = True
        state["collection_metadata"]["user_interactions"].append({
            "type": "escalation_to_human",
            "timestamp": datetime.now().isoformat(),
            "reason": "stealth_insufficient"
        })
        
        return state
    
    async def _try_human_scraping(self, state: ScrapingState) -> ScrapingState:
        """Attempt Layer 2: Human-assisted browser scraping."""
        self.log_info("Attempting Layer 2: Human-assisted scraping")
        
        if state["progress_callback"]:
            state["progress_callback"]("👤 Layer 2: Starting human-assisted collection...")
        
        state["current_layer"] = 2
        state["collection_metadata"]["layers_attempted"].append("human_browser")
        
        # Lazy initialize human scraper
        if not self.human_scraper:
            self.human_scraper = HumanBrowserScraper()
        
        try:
            profile = BrowserProfile(auto_detect=True)
            human_products = []
            
            # Only scrape failed stores from previous layer
            failed_stores = [s for s in state["stores"] if s in state["failed_stores"]]
            
            for store_id in failed_stores:
                try:
                    self.method_stats[DataCollectionMethod.HUMAN_BROWSER]["attempts"] += 1
                    
                    self.log_info(f"Human-assisted scraping {store_id}")
                    
                    # Provide user guidance
                    if state["progress_callback"]:
                        state["progress_callback"](
                            f"👤 Please assist with {store_id} - opening browser with your profile..."
                        )
                    
                    products = await self.human_scraper.scrape_with_assistance(
                        store_id=store_id,
                        query=state["query"],
                        limit=state["limit"],
                        profile=profile
                    )
                    
                    # Mark products with collection method
                    for product in products:
                        product.collection_method = DataCollectionMethod.HUMAN_BROWSER
                        product.confidence_score = 1.0  # Highest confidence for human-verified
                    
                    human_products.extend(products)
                    state["completed_stores"].append(store_id)
                    state["failed_stores"].remove(store_id)
                    self.method_stats[DataCollectionMethod.HUMAN_BROWSER]["successes"] += 1
                    
                    self.log_info(f"Human-assisted scraped {len(products)} products from {store_id}")
                    
                except Exception as e:
                    self.log_error(f"Human-assisted scraping failed for {store_id}: {e}")
                    state["errors"][f"human_{store_id}"] = str(e)
            
            state["products"].extend(human_products)
            
        except Exception as e:
            self.log_error(f"Human-assisted scraping layer failed: {e}", e)
            state["errors"]["human_layer"] = str(e)
        
        return state
    
    async def _evaluate_human_results(self, state: ScrapingState) -> ScrapingState:
        """Evaluate human-assisted scraping results."""
        completed_rate = len(state["completed_stores"]) / max(len(state["stores"]), 1)
        state["success_rates"]["human_assisted"] = completed_rate
        
        self.log_info(f"Human-assisted completion rate: {completed_rate:.2%}")
        
        if state["progress_callback"]:
            state["progress_callback"](
                f"👤 Layer 2 complete: {completed_rate:.0%} stores completed "
                f"({len(state['products'])} total products)"
            )
        
        return state
    
    def _should_escalate_from_human(self, state: ScrapingState) -> str:
        """Decide whether to escalate from human layer."""
        remaining_failed = len(state["failed_stores"])
        
        # Continue if all stores completed
        if remaining_failed == 0:
            return "continue_human"
        
        # Escalate to clipboard for remaining failed stores
        return "escalate_clipboard"
    
    async def _enable_clipboard_mode(self, state: ScrapingState) -> ScrapingState:
        """Enable Layer 3: Clipboard monitoring mode."""
        self.log_info("Enabling Layer 3: Clipboard monitoring")
        
        if state["progress_callback"]:
            state["progress_callback"]("📋 Layer 3: Enabling clipboard monitoring mode...")
        
        state["current_layer"] = 3
        state["clipboard_monitoring"] = True
        state["collection_metadata"]["layers_attempted"].append("clipboard")
        
        # Lazy initialize clipboard monitor
        if not self.clipboard_monitor:
            self.clipboard_monitor = ClipboardMonitor()
        
        return state
    
    async def _try_clipboard_collection(self, state: ScrapingState) -> ScrapingState:
        """Attempt Layer 3: Clipboard-based manual collection."""
        self.log_info("Attempting Layer 3: Clipboard collection")
        
        if state["progress_callback"]:
            state["progress_callback"]("📋 Layer 3: Manual clipboard collection active...")
        
        state["collection_metadata"]["layers_attempted"].append("clipboard_manual")
        
        try:
            # Start clipboard monitoring
            self.clipboard_monitor.start_monitoring()
            
            # Provide user instructions
            remaining_stores = [s for s in state["stores"] if s not in state["completed_stores"]]
            
            if state["progress_callback"]:
                state["progress_callback"](
                    f"📋 Please manually browse and copy product info for: {', '.join(remaining_stores)}"
                )
                state["progress_callback"]("📋 Copy any product text and it will be automatically parsed!")
            
            # Monitor for a reasonable time or until user indicates completion
            monitoring_duration = 300  # 5 minutes
            clipboard_products = []
            
            start_time = time.time()
            while time.time() - start_time < monitoring_duration:
                try:
                    # Check for new clipboard products
                    new_products = self.clipboard_monitor.get_recent_products()
                    
                    for clipboard_product in new_products:
                        if clipboard_product.suggested_product:
                            product = clipboard_product.suggested_product
                            product.collection_method = DataCollectionMethod.CLIPBOARD_MANUAL
                            product.confidence_score = 0.95  # High confidence for manual entry
                            clipboard_products.append(product)
                            
                            self.method_stats[DataCollectionMethod.CLIPBOARD_MANUAL]["successes"] += 1
                            
                            if state["progress_callback"]:
                                state["progress_callback"](
                                    f"📋 Captured: {product.name} - ${product.price}"
                                )
                    
                    # Check if user wants to continue or has enough products
                    if len(clipboard_products) >= state["limit"]:
                        break
                    
                    await asyncio.sleep(2)  # Check every 2 seconds
                    
                except KeyboardInterrupt:
                    self.log_info("User interrupted clipboard monitoring")
                    break
                except Exception as e:
                    self.log_error(f"Clipboard monitoring error: {e}")
                    break
            
            state["products"].extend(clipboard_products)
            
            # Mark remaining stores as completed through manual method
            for store_id in remaining_stores:
                if store_id not in state["completed_stores"]:
                    state["completed_stores"].append(store_id)
                    if store_id in state["failed_stores"]:
                        state["failed_stores"].remove(store_id)
            
            self.log_info(f"Clipboard collected {len(clipboard_products)} products")
            
        except Exception as e:
            self.log_error(f"Clipboard collection failed: {e}", e)
            state["errors"]["clipboard_layer"] = str(e)
        finally:
            # Stop monitoring
            if self.clipboard_monitor:
                self.clipboard_monitor.stop_monitoring()
                state["clipboard_monitoring"] = False
        
        return state
    
    async def _aggregate_results(self, state: ScrapingState) -> ScrapingState:
        """Aggregate and deduplicate results from all collection methods."""
        self.log_info("Aggregating results from all collection layers")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔄 Aggregating and deduplicating products...")
        
        # Deduplicate products based on name and store
        seen_products = set()
        deduplicated_products = []
        
        for product in state["products"]:
            product_key = (product.name.lower().strip(), product.store_id)
            if product_key not in seen_products:
                seen_products.add(product_key)
                deduplicated_products.append(product)
            else:
                # Keep the product with higher confidence
                existing_idx = next(
                    i for i, p in enumerate(deduplicated_products)
                    if (p.name.lower().strip(), p.store_id) == product_key
                )
                existing_product = deduplicated_products[existing_idx]
                
                if product.get_collection_confidence_weight() > existing_product.get_collection_confidence_weight():
                    deduplicated_products[existing_idx] = product
        
        state["products"] = deduplicated_products
        
        # Update collection metadata
        state["collection_metadata"]["aggregation_time"] = datetime.now().isoformat()
        state["collection_metadata"]["deduplicated_count"] = len(deduplicated_products)
        
        self.log_info(f"Aggregated {len(deduplicated_products)} unique products")
        
        return state
    
    async def _adaptive_learning(self, state: ScrapingState) -> ScrapingState:
        """Learn from collection results to improve future strategies."""
        self.log_info("Performing adaptive learning from collection results")
        
        # Analyze success rates by method
        method_performance = {}
        for method in DataCollectionMethod:
            stats = self.method_stats.get(method, {"attempts": 0, "successes": 0})
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                method_performance[method.value] = success_rate
        
        state["collection_metadata"]["method_performance"] = method_performance
        
        # Store learning insights
        insights = []
        
        if method_performance.get("automated_stealth", 0) > 0.8:
            insights.append("Stealth scraping is highly effective for this query type")
        elif method_performance.get("human_browser", 0) > 0.9:
            insights.append("Human assistance significantly improves success rates")
        
        if len(state["products"]) < state["limit"] * 0.5:
            insights.append("Consider increasing clipboard collection time for better coverage")
        
        state["collection_metadata"]["insights"] = insights
        
        self.log_info(f"Generated {len(insights)} adaptive learning insights")
        
        return state
    
    async def _finalize_scraping(self, state: ScrapingState) -> ScrapingState:
        """Finalize scraping session with summary and cleanup."""
        self.log_info("Finalizing scraping session")
        
        state["collection_metadata"]["end_time"] = datetime.now().isoformat()
        state["collection_metadata"]["total_duration"] = (
            datetime.fromisoformat(state["collection_metadata"]["end_time"]) -
            datetime.fromisoformat(state["collection_metadata"]["start_time"])
        ).total_seconds()
        
        if state["progress_callback"]:
            state["progress_callback"](
                f"✅ Scraping complete! Collected {len(state['products'])} products "
                f"across {len(state['completed_stores'])} stores"
            )
        
        self.log_info(
            f"Scraping session completed: {len(state['products'])} products, "
            f"{len(state['completed_stores'])} stores, "
            f"{len(state['collection_metadata']['layers_attempted'])} layers used"
        )
        
        return state
    
    # Utility Methods
    
    def _calculate_method_stats(self, products: List[Product]) -> Dict[str, Any]:
        """Calculate statistics about collection methods used."""
        method_counts = {}
        total_confidence = 0
        
        for product in products:
            method = product.collection_method.value
            method_counts[method] = method_counts.get(method, 0) + 1
            total_confidence += product.get_collection_confidence_weight()
        
        return {
            "method_distribution": method_counts,
            "average_confidence": total_confidence / max(len(products), 1),
            "total_products": len(products),
            "method_stats": self.method_stats
        }
    
    def get_collection_analytics(self) -> Dict[str, Any]:
        """Get analytics about collection method performance."""
        return {
            "method_stats": self.method_stats,
            "recommendations": self._generate_strategy_recommendations()
        }
    
    def _generate_strategy_recommendations(self) -> List[str]:
        """Generate recommendations for future scraping strategies."""
        recommendations = []
        
        for method, stats in self.method_stats.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                
                if success_rate > 0.8:
                    recommendations.append(f"Prioritize {method.value} for similar queries (high success rate)")
                elif success_rate < 0.3:
                    recommendations.append(f"Consider skipping {method.value} for similar queries (low success rate)")
        
        return recommendations
    
    # Public API Methods for Testing and Integration
    
    async def test_layer_individually(self, layer: int, query: str, store_id: str) -> Dict[str, Any]:
        """Test individual scraping layers for debugging."""
        if layer == 1:
            return await self._test_stealth_layer(query, store_id)
        elif layer == 2:
            return await self._test_human_layer(query, store_id)
        elif layer == 3:
            return await self._test_clipboard_layer(query)
        else:
            raise ValueError("Layer must be 1, 2, or 3")
    
    async def _test_stealth_layer(self, query: str, store_id: str) -> Dict[str, Any]:
        """Test stealth scraping layer."""
        if not self.stealth_scraper:
            self.stealth_scraper = StealthScraper()
        
        try:
            config = StealthConfig(headless=True, enable_stealth=True)
            products = await self.stealth_scraper.scrape_store(store_id, query, 10, config)
            return {"success": True, "products": products, "count": len(products)}
        except Exception as e:
            return {"success": False, "error": str(e), "products": []}
    
    async def _test_human_layer(self, query: str, store_id: str) -> Dict[str, Any]:
        """Test human-assisted scraping layer."""
        if not self.human_scraper:
            self.human_scraper = HumanBrowserScraper()
        
        try:
            profile = BrowserProfile(auto_detect=True)
            products = await self.human_scraper.scrape_with_assistance(store_id, query, 10, profile)
            return {"success": True, "products": products, "count": len(products)}
        except Exception as e:
            return {"success": False, "error": str(e), "products": []}
    
    async def _test_clipboard_layer(self, query: str) -> Dict[str, Any]:
        """Test clipboard collection layer."""
        if not self.clipboard_monitor:
            self.clipboard_monitor = ClipboardMonitor()
        
        try:
            self.clipboard_monitor.start_monitoring()
            print(f"📋 Clipboard monitoring active for query: {query}")
            print("📋 Copy product information and it will be automatically parsed!")
            print("📋 Press Ctrl+C to stop monitoring...")
            
            products = []
            try:
                while True:
                    new_products = self.clipboard_monitor.get_recent_products()
                    for cp in new_products:
                        if cp.suggested_product:
                            products.append(cp.suggested_product)
                            print(f"📋 Captured: {cp.suggested_product.name}")
                    
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                pass
            finally:
                self.clipboard_monitor.stop_monitoring()
            
            return {"success": True, "products": products, "count": len(products)}
            
        except Exception as e:
            return {"success": False, "error": str(e), "products": []}


# For backward compatibility, create an alias
ScraperAgent = IntelligentScraperAgent