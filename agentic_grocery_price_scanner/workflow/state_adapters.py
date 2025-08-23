"""
State adapters for transforming between master workflow state and agent-specific states.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from ..data_models import Recipe, Ingredient, Product
from ..agents.intelligent_scraper_agent import ScrapingState, CollectionStrategy
from ..agents.matcher_agent import MatchingState, MatchingStrategy
from ..agents.optimizer_agent import OptimizationState, OptimizationStrategy, OptimizationCriteria

logger = logging.getLogger(__name__)


class StateAdapter:
    """Adapter for transforming between master workflow state and agent-specific states."""
    
    @staticmethod
    def to_scraping_state(
        master_state: Dict[str, Any], 
        ingredient: Ingredient,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ScrapingState:
        """Transform master state to scraping state for a specific ingredient."""
        return ScrapingState(
            query=ingredient.name,
            stores=master_state.get("target_stores", ["metro_ca", "walmart_ca", "freshco_com"]),
            limit=master_state.get("products_per_store", 20),
            strategy=CollectionStrategy(master_state.get("scraping_strategy", "adaptive")),
            current_layer=1,
            products=[],
            errors={},
            success_rates={},
            user_assistance_active=False,
            clipboard_monitoring=False,
            completed_stores=[],
            failed_stores=[],
            collection_metadata={
                "ingredient_name": ingredient.name,
                "ingredient_category": ingredient.category,
                "workflow_execution_id": master_state.get("execution_id"),
                "start_time": datetime.now().isoformat()
            },
            progress_callback=progress_callback
        )
    
    @staticmethod
    def to_matching_state(
        master_state: Dict[str, Any],
        ingredient: Ingredient,
        available_products: List[Product],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> MatchingState:
        """Transform master state to matching state for a specific ingredient."""
        return MatchingState(
            ingredient=ingredient,
            search_query=ingredient.name,
            strategy=MatchingStrategy(master_state.get("matching_strategy", "adaptive")),
            vector_candidates=[],
            brand_normalized_candidates=[],
            llm_analysis={},
            final_matches=[],
            confidence_threshold=master_state.get("confidence_threshold", 0.5),
            max_results=master_state.get("max_matches_per_ingredient", 5),
            require_human_review=False,
            substitution_suggestions=[],
            category_analysis={},
            matching_metadata={
                "ingredient_name": ingredient.name,
                "available_products_count": len(available_products),
                "workflow_execution_id": master_state.get("execution_id"),
                "start_time": datetime.now().isoformat()
            },
            progress_callback=progress_callback
        )
    
    @staticmethod
    def to_optimization_state(
        master_state: Dict[str, Any],
        ingredients: List[Ingredient],
        matched_products: Dict[str, List[Dict[str, Any]]],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> OptimizationState:
        """Transform master state to optimization state."""
        # Create optimization criteria from master state preferences
        criteria = OptimizationCriteria(
            max_budget=master_state.get("max_budget"),
            max_stores=master_state.get("max_stores", 3),
            preferred_stores=master_state.get("preferred_stores", []),
            avoid_stores=master_state.get("avoid_stores", []),
            max_travel_time=master_state.get("max_travel_time"),
            quality_threshold=master_state.get("quality_threshold", 0.7),
            brand_preferences=master_state.get("brand_preferences", {}),
            dietary_restrictions=master_state.get("dietary_restrictions", []),
            bulk_buying_ok=master_state.get("bulk_buying_ok", True),
            sale_priority=master_state.get("sale_priority", 0.3)
        )
        
        return OptimizationState(
            ingredients=ingredients,
            criteria=criteria,
            strategy=OptimizationStrategy(master_state.get("optimization_strategy", "adaptive")),
            matched_products=matched_products,
            store_analysis={},
            cost_analysis={},
            convenience_analysis={},
            quality_analysis={},
            shopping_strategies={},
            recommended_strategy=[],
            optimization_metadata={
                "total_ingredients": len(ingredients),
                "workflow_execution_id": master_state.get("execution_id"),
                "start_time": datetime.now().isoformat()
            },
            progress_callback=progress_callback
        )
    
    @staticmethod
    def merge_scraping_results(
        master_state: Dict[str, Any],
        ingredient: Ingredient,
        scraping_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge scraping results back into master state."""
        updated_state = master_state.copy()
        
        # Update product collections
        if "collected_products" not in updated_state:
            updated_state["collected_products"] = {}
        
        updated_state["collected_products"][ingredient.name] = {
            "products": scraping_results.get("products", []),
            "collection_metadata": scraping_results.get("collection_metadata", {}),
            "success_rates": scraping_results.get("success_rates", {}),
            "errors": scraping_results.get("errors", {})
        }
        
        # Update workflow progress
        if "completed_ingredients" not in updated_state:
            updated_state["completed_ingredients"] = []
        
        updated_state["completed_ingredients"].append(ingredient.name)
        updated_state["workflow_progress"]["scraping"] = len(updated_state["completed_ingredients"])
        
        # Accumulate performance metrics
        if "performance_metrics" not in updated_state:
            updated_state["performance_metrics"] = {}
        
        if "scraping" not in updated_state["performance_metrics"]:
            updated_state["performance_metrics"]["scraping"] = {
                "total_products_collected": 0,
                "total_execution_time": 0,
                "success_rate_by_store": {},
                "collection_method_distribution": {}
            }
        
        # Update metrics
        products = scraping_results.get("products", [])
        updated_state["performance_metrics"]["scraping"]["total_products_collected"] += len(products)
        
        # Update collection method distribution
        for product in products:
            method = getattr(product, 'collection_method', 'unknown')
            if method not in updated_state["performance_metrics"]["scraping"]["collection_method_distribution"]:
                updated_state["performance_metrics"]["scraping"]["collection_method_distribution"][method] = 0
            updated_state["performance_metrics"]["scraping"]["collection_method_distribution"][method] += 1
        
        return updated_state
    
    @staticmethod
    def merge_matching_results(
        master_state: Dict[str, Any],
        ingredient: Ingredient,
        matching_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge matching results back into master state."""
        updated_state = master_state.copy()
        
        # Update matched products
        if "matched_products" not in updated_state:
            updated_state["matched_products"] = {}
        
        updated_state["matched_products"][ingredient.name] = matching_results.get("final_matches", [])
        
        # Update workflow progress
        updated_state["workflow_progress"]["matching"] += 1
        
        # Update performance metrics
        if "matching" not in updated_state["performance_metrics"]:
            updated_state["performance_metrics"]["matching"] = {
                "total_matches": 0,
                "avg_confidence": 0.0,
                "quality_distribution": {},
                "substitution_rate": 0.0
            }
        
        matches = matching_results.get("final_matches", [])
        updated_state["performance_metrics"]["matching"]["total_matches"] += len(matches)
        
        # Calculate average confidence
        if matches:
            total_confidence = sum(match.get("confidence", 0) for match in matches)
            avg_confidence = total_confidence / len(matches)
            
            # Update running average
            current_avg = updated_state["performance_metrics"]["matching"]["avg_confidence"]
            current_count = updated_state["workflow_progress"]["matching"]
            updated_avg = ((current_avg * (current_count - 1)) + avg_confidence) / current_count
            updated_state["performance_metrics"]["matching"]["avg_confidence"] = updated_avg
        
        return updated_state
    
    @staticmethod
    def merge_optimization_results(
        master_state: Dict[str, Any],
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge optimization results back into master state."""
        updated_state = master_state.copy()
        
        # Update final results
        updated_state["optimization_results"] = optimization_results
        updated_state["workflow_progress"]["optimization"] = 1
        
        # Update performance metrics
        updated_state["performance_metrics"]["optimization"] = {
            "recommended_strategy": optimization_results.get("recommended_strategy", []),
            "total_savings": float(optimization_results.get("total_savings", 0)),
            "savings_percentage": optimization_results.get("savings_percentage", 0),
            "store_distribution": {}
        }
        
        # Calculate store distribution
        recommended_trips = optimization_results.get("recommended_strategy", [])
        for trip in recommended_trips:
            store_id = trip.get("store_id", "unknown")
            if store_id not in updated_state["performance_metrics"]["optimization"]["store_distribution"]:
                updated_state["performance_metrics"]["optimization"]["store_distribution"][store_id] = {
                    "items": 0,
                    "cost": 0
                }
            
            updated_state["performance_metrics"]["optimization"]["store_distribution"][store_id]["items"] += trip.get("total_items", 0)
            updated_state["performance_metrics"]["optimization"]["store_distribution"][store_id]["cost"] += float(trip.get("total_cost", 0))
        
        return updated_state