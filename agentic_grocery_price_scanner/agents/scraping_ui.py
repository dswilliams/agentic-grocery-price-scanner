"""
User experience enhancements for the intelligent scraping system.
Provides real-time feedback, progress tracking, and user guidance.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import threading
import sys

from ..data_models import Product
from ..data_models.base import DataCollectionMethod


class UIUpdateType(Enum):
    """Types of UI updates."""
    
    PROGRESS = "progress"
    STATUS = "status"
    PRODUCT_FOUND = "product_found"
    USER_INPUT_REQUIRED = "user_input_required"
    ERROR = "error"
    LAYER_CHANGE = "layer_change"
    COMPLETION = "completion"


class ScrapingProgress:
    """Track scraping progress across all layers."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.current_layer = 1
        self.layers_attempted = []
        self.products_collected = 0
        self.stores_completed = []
        self.stores_failed = []
        self.current_status = "Initializing..."
        self.user_interactions = []
        self.error_messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "current_layer": self.current_layer,
            "layers_attempted": self.layers_attempted,
            "products_collected": self.products_collected,
            "stores_completed": self.stores_completed,
            "stores_failed": self.stores_failed,
            "current_status": self.current_status,
            "user_interactions": self.user_interactions,
            "error_messages": self.error_messages,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds()
        }


class ScrapingUIManager:
    """Manages user interface and experience for scraping operations."""
    
    def __init__(self, enable_console_output: bool = True):
        """Initialize UI manager."""
        self.enable_console_output = enable_console_output
        self.progress = ScrapingProgress()
        self.callbacks: List[Callable[[UIUpdateType, Dict[str, Any]], None]] = []
        self.user_input_queue = asyncio.Queue()
        self._ui_lock = threading.Lock()
    
    def add_callback(self, callback: Callable[[UIUpdateType, Dict[str, Any]], None]) -> None:
        """Add a callback for UI updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[UIUpdateType, Dict[str, Any]], None]) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def update_progress(self, message: str, layer: Optional[int] = None) -> None:
        """Update progress with a message."""
        with self._ui_lock:
            self.progress.current_status = message
            if layer:
                self.progress.current_layer = layer
            
            self._emit_update(UIUpdateType.PROGRESS, {
                "message": message,
                "layer": layer,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {message}")
    
    def layer_changed(self, new_layer: int, layer_name: str) -> None:
        """Notify that scraping layer has changed."""
        with self._ui_lock:
            self.progress.current_layer = new_layer
            if layer_name not in self.progress.layers_attempted:
                self.progress.layers_attempted.append(layer_name)
            
            layer_descriptions = {
                1: "🤖 Automated Stealth Scraping",
                2: "👤 Human-Assisted Browser Automation",
                3: "📋 Manual Clipboard Collection"
            }
            
            description = layer_descriptions.get(new_layer, f"Layer {new_layer}")
            
            self._emit_update(UIUpdateType.LAYER_CHANGE, {
                "layer": new_layer,
                "layer_name": layer_name,
                "description": description,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                print(f"\\n{'='*60}")
                print(f"   {description}")
                print(f"{'='*60}")
    
    def product_found(self, product: Product, source: str) -> None:
        """Notify that a product was found."""
        with self._ui_lock:
            self.progress.products_collected += 1
            
            self._emit_update(UIUpdateType.PRODUCT_FOUND, {
                "product": {
                    "name": product.name,
                    "brand": product.brand,
                    "price": float(product.price),
                    "store_id": product.store_id,
                    "collection_method": product.collection_method.value,
                    "confidence_score": product.confidence_score
                },
                "source": source,
                "total_count": self.progress.products_collected,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                price_str = f"${product.price:.2f}"
                method_emoji = {
                    DataCollectionMethod.AUTOMATED_STEALTH: "🤖",
                    DataCollectionMethod.HUMAN_BROWSER: "👤",
                    DataCollectionMethod.CLIPBOARD_MANUAL: "📋"
                }.get(product.collection_method, "❓")
                
                print(f"  {method_emoji} Found: {product.name} - {price_str} at {product.store_id}")
    
    def store_completed(self, store_id: str, product_count: int) -> None:
        """Notify that a store was completed."""
        with self._ui_lock:
            if store_id not in self.progress.stores_completed:
                self.progress.stores_completed.append(store_id)
            
            if self.enable_console_output:
                print(f"  ✅ Completed {store_id}: {product_count} products")
    
    def store_failed(self, store_id: str, error: str) -> None:
        """Notify that a store failed."""
        with self._ui_lock:
            if store_id not in self.progress.stores_failed:
                self.progress.stores_failed.append(store_id)
            
            self.progress.error_messages.append(f"{store_id}: {error}")
            
            self._emit_update(UIUpdateType.ERROR, {
                "store_id": store_id,
                "error": error,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                print(f"  ❌ Failed {store_id}: {error}")
    
    def require_user_input(self, prompt: str, input_type: str = "text") -> str:
        """Request user input with a prompt."""
        with self._ui_lock:
            self.progress.user_interactions.append({
                "type": "input_request",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            self._emit_update(UIUpdateType.USER_INPUT_REQUIRED, {
                "prompt": prompt,
                "input_type": input_type,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                print(f"\\n⚠️  {prompt}")
                if input_type == "confirmation":
                    return input("Press Enter to continue or 'q' to quit: ").strip()
                else:
                    return input("> ").strip()
    
    def show_layer_instructions(self, layer: int, instructions: List[str]) -> None:
        """Show instructions for a specific layer."""
        if self.enable_console_output:
            print(f"\\n📋 Instructions for Layer {layer}:")
            for i, instruction in enumerate(instructions, 1):
                print(f"  {i}. {instruction}")
            print()
    
    def show_completion_summary(self, final_results: Dict[str, Any]) -> None:
        """Show final completion summary."""
        with self._ui_lock:
            self._emit_update(UIUpdateType.COMPLETION, {
                "results": final_results,
                "progress": self.progress.to_dict()
            })
            
            if self.enable_console_output:
                self._print_completion_summary(final_results)
    
    def _print_completion_summary(self, results: Dict[str, Any]) -> None:
        """Print completion summary to console."""
        print(f"\\n{'='*60}")
        print("   🎉 SCRAPING COMPLETED!")
        print(f"{'='*60}")
        
        # Basic stats
        total_products = results.get("total_products", 0)
        stores_completed = results.get("stores_completed", 0)
        stores_failed = results.get("stores_failed", 0)
        
        print(f"\\n📊 SUMMARY:")
        print(f"  • Total Products: {total_products}")
        print(f"  • Stores Completed: {stores_completed}")
        print(f"  • Stores Failed: {stores_failed}")
        print(f"  • Duration: {self.progress.to_dict()['elapsed_time']:.1f}s")
        
        # Method breakdown
        method_stats = results.get("method_stats", {})
        if method_stats:
            print(f"\\n🔍 COLLECTION METHODS:")
            method_distribution = method_stats.get("method_distribution", {})
            for method, count in method_distribution.items():
                emoji = {
                    "automated_stealth": "🤖",
                    "human_browser": "👤", 
                    "clipboard_manual": "📋"
                }.get(method, "❓")
                print(f"  • {emoji} {method.replace('_', ' ').title()}: {count} products")
        
        # Layer effectiveness
        layers_used = self.progress.layers_attempted
        print(f"\\n🎯 LAYERS USED: {' → '.join(layers_used)}")
        
        # Top products by confidence
        products = results.get("products", [])
        if products:
            print(f"\\n⭐ TOP PRODUCTS (by confidence):")
            sorted_products = sorted(products, key=lambda p: p.get_collection_confidence_weight(), reverse=True)
            for i, product in enumerate(sorted_products[:5], 1):
                confidence = product.get_collection_confidence_weight()
                print(f"  {i}. {product.name} - ${product.price:.2f} ({confidence:.1%} confidence)")
        
        print(f"\\n{'='*60}")
    
    def _emit_update(self, update_type: UIUpdateType, data: Dict[str, Any]) -> None:
        """Emit update to all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(update_type, data)
            except Exception as e:
                if self.enable_console_output:
                    print(f"Warning: UI callback error: {e}")
    
    def create_progress_callback(self) -> Callable[[str], None]:
        """Create a progress callback function for the scraper agent."""
        def progress_callback(message: str):
            # Parse layer information from message
            layer = None
            if "Layer 1" in message or "🤖" in message:
                layer = 1
            elif "Layer 2" in message or "👤" in message:
                layer = 2
            elif "Layer 3" in message or "📋" in message:
                layer = 3
            
            self.update_progress(message, layer)
        
        return progress_callback


class InteractiveScrapingSession:
    """Manages an interactive scraping session with rich UI feedback."""
    
    def __init__(self, agent, enable_console: bool = True):
        """Initialize interactive session."""
        self.agent = agent
        self.ui_manager = ScrapingUIManager(enable_console_output=enable_console)
        self.session_data = {}
    
    async def start_scraping(
        self,
        query: str,
        stores: Optional[List[str]] = None,
        limit: int = 50,
        strategy: str = "adaptive"
    ) -> Dict[str, Any]:
        """Start an interactive scraping session."""
        
        # Setup UI callbacks
        progress_callback = self.ui_manager.create_progress_callback()
        
        # Prepare inputs
        inputs = {
            "query": query,
            "stores": stores or [],
            "limit": limit,
            "strategy": strategy,
            "progress_callback": progress_callback
        }
        
        # Show initial instructions
        self.ui_manager.update_progress(f"🚀 Starting intelligent scraping for: '{query}'")
        
        try:
            # Execute scraping with UI feedback
            results = await self.agent.execute(inputs)
            
            # Show completion summary
            self.ui_manager.show_completion_summary(results)
            
            return results
            
        except Exception as e:
            self.ui_manager.update_progress(f"❌ Scraping failed: {e}")
            raise
    
    def add_ui_callback(self, callback: Callable[[UIUpdateType, Dict[str, Any]], None]) -> None:
        """Add a custom UI callback for external integrations."""
        self.ui_manager.add_callback(callback)
    
    def get_session_progress(self) -> Dict[str, Any]:
        """Get current session progress."""
        return self.ui_manager.progress.to_dict()


def create_console_ui_callback() -> Callable[[UIUpdateType, Dict[str, Any]], None]:
    """Create a simple console UI callback for debugging."""
    def console_callback(update_type: UIUpdateType, data: Dict[str, Any]):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if update_type == UIUpdateType.PROGRESS:
            print(f"[{timestamp}] 📊 {data['message']}")
        elif update_type == UIUpdateType.PRODUCT_FOUND:
            product = data['product']
            print(f"[{timestamp}] 🛍️  Found: {product['name']} (${product['price']:.2f})")
        elif update_type == UIUpdateType.LAYER_CHANGE:
            print(f"[{timestamp}] 🔄 {data['description']}")
        elif update_type == UIUpdateType.ERROR:
            print(f"[{timestamp}] ❌ Error: {data['error']}")
        elif update_type == UIUpdateType.COMPLETION:
            total = data['results'].get('total_products', 0)
            print(f"[{timestamp}] ✅ Completed! {total} products collected")
    
    return console_callback


def create_json_ui_callback(output_file: str) -> Callable[[UIUpdateType, Dict[str, Any]], None]:
    """Create a JSON file UI callback for external monitoring."""
    def json_callback(update_type: UIUpdateType, data: Dict[str, Any]):
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "type": update_type.value,
            "data": data
        }
        
        try:
            with open(output_file, "a") as f:
                json.dump(update_record, f)
                f.write("\\n")
        except Exception as e:
            print(f"Warning: Failed to write JSON update: {e}")
    
    return json_callback