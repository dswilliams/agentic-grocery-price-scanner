"""
OptimizerAgent for multi-store shopping optimization using LangGraph and Phi-3.5 Mini.
This agent uses complex reasoning to balance cost, convenience, and quality across all stores.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..data_models.ingredient import Ingredient
from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod
from ..llm_client.ollama_client import OllamaClient, ModelType
from ..llm_client.prompt_templates import PromptTemplates
from .base_agent import BaseAgent
from .matcher_agent import MatcherAgent

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for shopping decisions."""
    
    COST_ONLY = "cost_only"              # Pure cost minimization
    CONVENIENCE = "convenience"          # Minimize shopping trips
    BALANCED = "balanced"                # Balance cost vs convenience
    QUALITY_FIRST = "quality_first"      # Prioritize product quality
    TIME_EFFICIENT = "time_efficient"    # Minimize total shopping time
    ADAPTIVE = "adaptive"                # AI-selected strategy


class StorePreference(Enum):
    """Store preference levels."""
    
    PREFERRED = "preferred"              # User's preferred stores
    ACCEPTABLE = "acceptable"            # Stores user will visit
    AVOID = "avoid"                      # Stores to avoid if possible
    EMERGENCY_ONLY = "emergency_only"    # Last resort stores


@dataclass
class OptimizationCriteria:
    """User preferences and constraints for optimization."""
    
    max_budget: Optional[Decimal] = None
    max_stores: int = 3
    preferred_stores: List[str] = None
    avoid_stores: List[str] = None
    max_travel_time: Optional[int] = None  # minutes
    quality_threshold: float = 0.7
    brand_preferences: Dict[str, List[str]] = None  # category: [preferred_brands]
    dietary_restrictions: List[str] = None
    bulk_buying_ok: bool = True
    sale_priority: float = 0.3  # Weight for sale items (0-1)
    
    def __post_init__(self):
        if self.preferred_stores is None:
            self.preferred_stores = []
        if self.avoid_stores is None:
            self.avoid_stores = []
        if self.brand_preferences is None:
            self.brand_preferences = {}
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []


@dataclass
class ShoppingTrip:
    """Represents a shopping trip to a specific store."""
    
    store_id: str
    store_name: str
    products: List[Tuple[Product, Ingredient]]  # (product, ingredient) pairs
    total_cost: Decimal
    total_items: int
    estimated_time: int  # minutes
    priority_score: float
    travel_time: int = 15  # Default travel time in minutes
    convenience_score: float = 0.0
    quality_score: float = 0.0


@dataclass
class OptimizationResult:
    """Complete optimization result with multiple strategies."""
    
    recommended_strategy: List[ShoppingTrip]
    alternative_strategies: Dict[str, List[ShoppingTrip]]
    total_savings: Decimal
    savings_percentage: float
    optimization_metadata: Dict[str, Any]
    unmatched_ingredients: List[Ingredient]
    substitution_recommendations: List[Dict[str, Any]]


class OptimizationState(TypedDict):
    """State structure for LangGraph optimization workflow."""
    
    ingredients: List[Ingredient]
    criteria: OptimizationCriteria
    strategy: OptimizationStrategy
    matched_products: Dict[str, List[Dict[str, Any]]]  # ingredient_name -> matches
    store_analysis: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    convenience_analysis: Dict[str, Any]
    quality_analysis: Dict[str, Any]
    shopping_strategies: Dict[str, List[ShoppingTrip]]
    recommended_strategy: List[ShoppingTrip]
    optimization_metadata: Dict[str, Any]
    progress_callback: Optional[Callable[[str], None]]


class OptimizerAgent(BaseAgent):
    """LangGraph-based multi-store shopping optimizer."""
    
    def __init__(
        self,
        matcher_agent: Optional[MatcherAgent] = None,
        llm_client: Optional[OllamaClient] = None
    ):
        """Initialize the optimizer agent."""
        super().__init__("optimizer")
        
        # Initialize components
        self.matcher_agent = matcher_agent or MatcherAgent()
        self.llm_client = llm_client or OllamaClient()
        
        # Store information (would be loaded from config in real implementation)
        self.store_info = {
            "metro_ca": {
                "name": "Metro",
                "travel_time": 15,
                "preference": StorePreference.PREFERRED,
                "specialties": ["fresh_produce", "deli", "bakery"],
                "avg_price_factor": 1.1
            },
            "walmart_ca": {
                "name": "Walmart",
                "travel_time": 20,
                "preference": StorePreference.ACCEPTABLE,
                "specialties": ["bulk_items", "packaged_goods", "household"],
                "avg_price_factor": 0.9
            },
            "freshco_com": {
                "name": "FreshCo",
                "travel_time": 25,
                "preference": StorePreference.ACCEPTABLE,
                "specialties": ["budget_friendly", "ethnic_foods"],
                "avg_price_factor": 0.85
            }
        }
        
        # Optimization statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "avg_savings_percentage": 0.0,
            "avg_stores_reduced": 0.0,
            "strategy_performance": {
                strategy.value: {"uses": 0, "avg_satisfaction": 0.0}
                for strategy in OptimizationStrategy
            }
        }
        
        # Build LangGraph workflow
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multi-store shopping optimization.
        
        Args:
            inputs: Dictionary containing:
                - ingredients: List of Ingredient objects or dicts
                - criteria: OptimizationCriteria object or dict (optional)
                - strategy: Optimization strategy (optional)
                - progress_callback: Progress callback function (optional)
        
        Returns:
            Dictionary with optimization results and metadata
        """
        # Parse ingredients input
        ingredients_data = inputs.get("ingredients", [])
        ingredients = []
        for ingredient_data in ingredients_data:
            if isinstance(ingredient_data, dict):
                ingredients.append(Ingredient(**ingredient_data))
            elif isinstance(ingredient_data, Ingredient):
                ingredients.append(ingredient_data)
            else:
                raise ValueError("Ingredients must be provided as Ingredient objects or dicts")
        
        if not ingredients:
            raise ValueError("At least one ingredient must be provided")
        
        # Parse criteria
        criteria_data = inputs.get("criteria", {})
        if isinstance(criteria_data, dict):
            criteria = OptimizationCriteria(**criteria_data)
        elif isinstance(criteria_data, OptimizationCriteria):
            criteria = criteria_data
        else:
            criteria = OptimizationCriteria()
        
        # Initialize state
        initial_state: OptimizationState = {
            "ingredients": ingredients,
            "criteria": criteria,
            "strategy": OptimizationStrategy(inputs.get("strategy", "adaptive")),
            "matched_products": {},
            "store_analysis": {},
            "cost_analysis": {},
            "convenience_analysis": {},
            "quality_analysis": {},
            "shopping_strategies": {},
            "recommended_strategy": [],
            "optimization_metadata": {
                "start_time": datetime.now().isoformat(),
                "stages_completed": [],
                "performance_metrics": {}
            },
            "progress_callback": inputs.get("progress_callback")
        }
        
        self.log_info(f"Starting optimization for {len(ingredients)} ingredients")
        
        # Execute workflow
        try:
            config = {"configurable": {"thread_id": f"optimize_{int(datetime.now().timestamp())}"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Update statistics
            self._update_optimization_stats(final_state)
            
            # Compile results
            result = self._compile_optimization_results(final_state)
            
            self.log_info(f"Optimization completed: {len(final_state['recommended_strategy'])} shopping trips")
            return result
            
        except Exception as e:
            self.log_error(f"Optimization workflow failed: {e}", e)
            return {
                "success": False,
                "error": str(e),
                "ingredients": [ing.name for ing in ingredients],
                "shopping_trips": [],
                "total_cost": 0
            }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for shopping optimization."""
        workflow = StateGraph(OptimizationState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_optimization)
        workflow.add_node("match_ingredients", self._match_ingredients_to_products)
        workflow.add_node("analyze_stores", self._analyze_store_coverage)
        workflow.add_node("cost_optimization", self._optimize_for_cost)
        workflow.add_node("convenience_analysis", self._analyze_convenience_factors)
        workflow.add_node("quality_assessment", self._assess_product_quality)
        workflow.add_node("strategy_generation", self._generate_shopping_strategies)
        workflow.add_node("multi_criteria_scoring", self._apply_multi_criteria_scoring)
        workflow.add_node("route_optimization", self._optimize_shopping_routes)
        workflow.add_node("substitution_analysis", self._analyze_substitution_opportunities)
        workflow.add_node("budget_validation", self._validate_budget_constraints)
        workflow.add_node("recommendation_selection", self._select_best_recommendation)
        workflow.add_node("finalize", self._finalize_optimization)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "match_ingredients")
        workflow.add_edge("match_ingredients", "analyze_stores")
        workflow.add_edge("analyze_stores", "cost_optimization")
        workflow.add_edge("cost_optimization", "convenience_analysis")
        workflow.add_edge("convenience_analysis", "quality_assessment")
        workflow.add_edge("quality_assessment", "strategy_generation")
        workflow.add_edge("strategy_generation", "multi_criteria_scoring")
        workflow.add_edge("multi_criteria_scoring", "route_optimization")
        workflow.add_edge("route_optimization", "substitution_analysis")
        workflow.add_edge("substitution_analysis", "budget_validation")
        workflow.add_edge("budget_validation", "recommendation_selection")
        workflow.add_edge("recommendation_selection", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    # Workflow Node Functions
    
    async def _initialize_optimization(self, state: OptimizationState) -> OptimizationState:
        """Initialize optimization session."""
        self.log_info("Initializing optimization session")
        
        if state["progress_callback"]:
            state["progress_callback"](
                f"🛒 Initializing optimization for {len(state['ingredients'])} ingredients..."
            )
        
        state["optimization_metadata"]["stages_completed"].append("initialize")
        state["optimization_metadata"]["ingredient_count"] = len(state["ingredients"])
        state["optimization_metadata"]["criteria"] = {
            "max_budget": str(state["criteria"].max_budget) if state["criteria"].max_budget else None,
            "max_stores": state["criteria"].max_stores,
            "preferred_stores": state["criteria"].preferred_stores,
            "quality_threshold": state["criteria"].quality_threshold
        }
        
        return state
    
    async def _match_ingredients_to_products(self, state: OptimizationState) -> OptimizationState:
        """Match all ingredients to available products using MatcherAgent."""
        self.log_info("Matching ingredients to products")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔍 Finding products for all ingredients...")
        
        matched_products = {}
        
        for ingredient in state["ingredients"]:
            try:
                # Use MatcherAgent for intelligent product matching
                match_result = await self.matcher_agent.match_ingredient(
                    ingredient=ingredient,
                    strategy="adaptive",
                    confidence_threshold=0.3,  # Lower threshold for more options
                    max_results=10  # Get more options for optimization
                )
                
                if match_result["success"]:
                    matched_products[ingredient.name] = match_result["matches"]
                    self.log_info(f"Found {len(match_result['matches'])} matches for '{ingredient.name}'")
                else:
                    matched_products[ingredient.name] = []
                    self.log_info(f"No matches found for '{ingredient.name}'")
                    
            except Exception as e:
                self.log_error(f"Failed to match ingredient '{ingredient.name}': {e}")
                matched_products[ingredient.name] = []
        
        state["matched_products"] = matched_products
        state["optimization_metadata"]["stages_completed"].append("match_ingredients")
        state["optimization_metadata"]["total_product_matches"] = sum(
            len(matches) for matches in matched_products.values()
        )
        
        return state
    
    async def _analyze_store_coverage(self, state: OptimizationState) -> OptimizationState:
        """Analyze which stores can fulfill the shopping list."""
        self.log_info("Analyzing store coverage")
        
        if state["progress_callback"]:
            state["progress_callback"]("🏪 Analyzing store coverage...")
        
        store_analysis = {}
        
        for store_id, store_info in self.store_info.items():
            coverage_data = {
                "store_id": store_id,
                "store_name": store_info["name"],
                "available_ingredients": [],
                "missing_ingredients": [],
                "total_products": 0,
                "avg_confidence": 0.0,
                "total_cost": Decimal("0.00"),
                "coverage_percentage": 0.0
            }
            
            total_confidence = 0.0
            confidence_count = 0
            
            for ingredient_name, matches in state["matched_products"].items():
                store_matches = [
                    match for match in matches 
                    if match["product"].store_id == store_id
                ]
                
                if store_matches:
                    # Use the best match for this store
                    best_match = max(store_matches, key=lambda x: x["confidence"])
                    coverage_data["available_ingredients"].append({
                        "ingredient": ingredient_name,
                        "product": best_match["product"],
                        "confidence": best_match["confidence"]
                    })
                    coverage_data["total_cost"] += best_match["product"].price
                    total_confidence += best_match["confidence"]
                    confidence_count += 1
                else:
                    coverage_data["missing_ingredients"].append(ingredient_name)
                
                coverage_data["total_products"] += len(store_matches)
            
            if confidence_count > 0:
                coverage_data["avg_confidence"] = total_confidence / confidence_count
            
            coverage_data["coverage_percentage"] = (
                len(coverage_data["available_ingredients"]) / len(state["ingredients"]) * 100
                if state["ingredients"] else 0
            )
            
            store_analysis[store_id] = coverage_data
        
        state["store_analysis"] = store_analysis
        state["optimization_metadata"]["stages_completed"].append("analyze_stores")
        
        return state
    
    async def _optimize_for_cost(self, state: OptimizationState) -> OptimizationState:
        """Analyze cost optimization opportunities."""
        self.log_info("Optimizing for cost")
        
        if state["progress_callback"]:
            state["progress_callback"]("💰 Analyzing cost optimization...")
        
        cost_analysis = {
            "cheapest_per_ingredient": {},
            "bulk_opportunities": [],
            "sale_items": [],
            "cross_store_savings": {},
            "total_min_cost": Decimal("0.00"),
            "total_convenience_cost": Decimal("0.00")
        }
        
        # Find cheapest option for each ingredient across all stores
        for ingredient_name, matches in state["matched_products"].items():
            if not matches:
                continue
                
            # Sort by effective price (considering sales)
            sorted_matches = sorted(
                matches,
                key=lambda x: x["product"].sale_price if x["product"].on_sale 
                else x["product"].price
            )
            
            cheapest = sorted_matches[0]
            cost_analysis["cheapest_per_ingredient"][ingredient_name] = {
                "product": cheapest["product"],
                "confidence": cheapest["confidence"],
                "savings_vs_avg": self._calculate_savings_vs_average(cheapest, sorted_matches)
            }
            
            cost_analysis["total_min_cost"] += (
                cheapest["product"].sale_price if cheapest["product"].on_sale 
                else cheapest["product"].price
            )
            
            # Identify sale items
            if cheapest["product"].on_sale:
                cost_analysis["sale_items"].append({
                    "ingredient": ingredient_name,
                    "product": cheapest["product"],
                    "savings": cheapest["product"].price - cheapest["product"].sale_price
                })
            
            # Identify bulk opportunities
            if (cheapest["product"].size and 
                cheapest["product"].size > 1.5 and  # Assume bulk if >1.5x standard size
                cheapest["product"].price_per_unit):
                cost_analysis["bulk_opportunities"].append({
                    "ingredient": ingredient_name,
                    "product": cheapest["product"],
                    "unit_savings": self._calculate_unit_savings(cheapest, sorted_matches)
                })
        
        # Calculate convenience shopping cost (single store)
        best_single_store = max(
            state["store_analysis"].values(),
            key=lambda x: x["coverage_percentage"]
        )
        cost_analysis["total_convenience_cost"] = best_single_store["total_cost"]
        
        # Calculate cross-store savings potential
        cost_analysis["cross_store_savings"] = {
            "potential_savings": cost_analysis["total_convenience_cost"] - cost_analysis["total_min_cost"],
            "savings_percentage": float(
                (cost_analysis["total_convenience_cost"] - cost_analysis["total_min_cost"]) /
                cost_analysis["total_convenience_cost"] * 100
                if cost_analysis["total_convenience_cost"] > 0 else 0
            )
        }
        
        state["cost_analysis"] = cost_analysis
        state["optimization_metadata"]["stages_completed"].append("cost_optimization")
        
        return state
    
    def _calculate_savings_vs_average(
        self, 
        cheapest_match: Dict[str, Any], 
        all_matches: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate savings compared to average price."""
        if len(all_matches) <= 1:
            return Decimal("0.00")
        
        cheapest_price = (
            cheapest_match["product"].sale_price if cheapest_match["product"].on_sale
            else cheapest_match["product"].price
        )
        
        total_price = sum(
            match["product"].sale_price if match["product"].on_sale 
            else match["product"].price
            for match in all_matches
        )
        
        avg_price = total_price / len(all_matches)
        return avg_price - cheapest_price
    
    def _calculate_unit_savings(
        self, 
        bulk_match: Dict[str, Any], 
        all_matches: List[Dict[str, Any]]
    ) -> Optional[Decimal]:
        """Calculate unit price savings for bulk items."""
        bulk_product = bulk_match["product"]
        if not bulk_product.price_per_unit:
            return None
        
        # Find regular sized items
        regular_items = [
            match for match in all_matches 
            if (match["product"].size and 
                match["product"].size <= 1.5 and 
                match["product"].price_per_unit)
        ]
        
        if not regular_items:
            return None
        
        avg_unit_price = sum(
            match["product"].price_per_unit for match in regular_items
        ) / len(regular_items)
        
        return avg_unit_price - bulk_product.price_per_unit
    
    async def _analyze_convenience_factors(self, state: OptimizationState) -> OptimizationState:
        """Analyze convenience factors like travel time and store consolidation."""
        self.log_info("Analyzing convenience factors")
        
        if state["progress_callback"]:
            state["progress_callback"]("🚗 Analyzing convenience factors...")
        
        convenience_analysis = {
            "single_store_options": {},
            "travel_time_analysis": {},
            "store_consolidation_opportunities": []
        }
        
        # Analyze single-store shopping options
        for store_id, analysis in state["store_analysis"].items():
            if analysis["coverage_percentage"] >= 80:  # Can fulfill 80%+ of list
                convenience_score = self._calculate_convenience_score(store_id, analysis)
                convenience_analysis["single_store_options"][store_id] = {
                    "coverage_percentage": analysis["coverage_percentage"],
                    "total_cost": analysis["total_cost"],
                    "travel_time": self.store_info[store_id]["travel_time"],
                    "convenience_score": convenience_score,
                    "missing_ingredients": analysis["missing_ingredients"]
                }
        
        # Analyze travel time optimization
        for store_id, store_info in self.store_info.items():
            convenience_analysis["travel_time_analysis"][store_id] = {
                "travel_time": store_info["travel_time"],
                "preference": store_info["preference"].value,
                "efficiency_score": self._calculate_efficiency_score(store_id, state)
            }
        
        # Find store consolidation opportunities
        consolidation_opportunities = self._find_consolidation_opportunities(state)
        convenience_analysis["store_consolidation_opportunities"] = consolidation_opportunities
        
        state["convenience_analysis"] = convenience_analysis
        state["optimization_metadata"]["stages_completed"].append("convenience_analysis")
        
        return state
    
    def _calculate_convenience_score(self, store_id: str, analysis: Dict[str, Any]) -> float:
        """Calculate convenience score for a store."""
        coverage_weight = 0.4
        travel_weight = 0.3
        preference_weight = 0.3
        
        coverage_score = analysis["coverage_percentage"] / 100.0
        
        # Invert travel time (shorter is better)
        max_travel = 60  # Assume max reasonable travel time
        travel_score = 1.0 - (self.store_info[store_id]["travel_time"] / max_travel)
        
        # Store preference score
        preference_scores = {
            StorePreference.PREFERRED: 1.0,
            StorePreference.ACCEPTABLE: 0.7,
            StorePreference.AVOID: 0.3,
            StorePreference.EMERGENCY_ONLY: 0.1
        }
        preference_score = preference_scores[self.store_info[store_id]["preference"]]
        
        return (
            coverage_score * coverage_weight +
            travel_score * travel_weight +
            preference_score * preference_weight
        )
    
    def _calculate_efficiency_score(self, store_id: str, state: OptimizationState) -> float:
        """Calculate shopping efficiency score for a store."""
        analysis = state["store_analysis"][store_id]
        
        if analysis["coverage_percentage"] == 0:
            return 0.0
        
        # Items per minute efficiency
        travel_time = self.store_info[store_id]["travel_time"]
        shopping_time = len(analysis["available_ingredients"]) * 3  # 3 min per item
        total_time = travel_time + shopping_time
        
        efficiency = len(analysis["available_ingredients"]) / total_time
        return min(efficiency, 1.0)  # Cap at 1.0
    
    def _find_consolidation_opportunities(self, state: OptimizationState) -> List[Dict[str, Any]]:
        """Find opportunities to consolidate shopping across fewer stores."""
        opportunities = []
        
        # Try all combinations of 2 stores
        store_ids = list(self.store_info.keys())
        for i in range(len(store_ids)):
            for j in range(i + 1, len(store_ids)):
                store1, store2 = store_ids[i], store_ids[j]
                
                combined_coverage = self._calculate_combined_coverage(
                    store1, store2, state
                )
                
                if combined_coverage["coverage_percentage"] >= 90:
                    opportunities.append({
                        "stores": [store1, store2],
                        "store_names": [
                            self.store_info[store1]["name"],
                            self.store_info[store2]["name"]
                        ],
                        "coverage_percentage": combined_coverage["coverage_percentage"],
                        "total_cost": combined_coverage["total_cost"],
                        "total_travel_time": (
                            self.store_info[store1]["travel_time"] +
                            self.store_info[store2]["travel_time"]
                        ),
                        "efficiency_score": combined_coverage["efficiency_score"]
                    })
        
        # Sort by efficiency score
        return sorted(opportunities, key=lambda x: x["efficiency_score"], reverse=True)
    
    def _calculate_combined_coverage(
        self, 
        store1: str, 
        store2: str, 
        state: OptimizationState
    ) -> Dict[str, Any]:
        """Calculate coverage when combining two stores."""
        covered_ingredients = set()
        total_cost = Decimal("0.00")
        
        for ingredient_name, matches in state["matched_products"].items():
            # Find best match across both stores
            store_matches = [
                match for match in matches 
                if match["product"].store_id in [store1, store2]
            ]
            
            if store_matches:
                best_match = max(store_matches, key=lambda x: x["confidence"])
                covered_ingredients.add(ingredient_name)
                total_cost += best_match["product"].price
        
        coverage_percentage = len(covered_ingredients) / len(state["ingredients"]) * 100
        
        # Calculate efficiency (items per total time)
        total_travel_time = (
            self.store_info[store1]["travel_time"] +
            self.store_info[store2]["travel_time"]
        )
        shopping_time = len(covered_ingredients) * 3
        efficiency_score = len(covered_ingredients) / (total_travel_time + shopping_time)
        
        return {
            "coverage_percentage": coverage_percentage,
            "total_cost": total_cost,
            "efficiency_score": efficiency_score,
            "covered_ingredients": list(covered_ingredients)
        }
    
    async def _assess_product_quality(self, state: OptimizationState) -> OptimizationState:
        """Assess product quality across different options."""
        self.log_info("Assessing product quality")
        
        if state["progress_callback"]:
            state["progress_callback"]("⭐ Assessing product quality...")
        
        quality_analysis = {
            "quality_scores": {},
            "high_quality_options": [],
            "quality_vs_cost_tradeoffs": []
        }
        
        for ingredient_name, matches in state["matched_products"].items():
            if not matches:
                continue
            
            quality_scores = []
            
            for match in matches:
                product = match["product"]
                quality_score = self._calculate_product_quality_score(product, match)
                quality_scores.append({
                    "product": product,
                    "quality_score": quality_score,
                    "confidence": match["confidence"],
                    "combined_score": quality_score * match["confidence"]
                })
            
            # Sort by combined quality and confidence
            quality_scores.sort(key=lambda x: x["combined_score"], reverse=True)
            quality_analysis["quality_scores"][ingredient_name] = quality_scores
            
            # Identify high-quality options
            high_quality = [
                score for score in quality_scores 
                if score["quality_score"] >= state["criteria"].quality_threshold
            ]
            
            if high_quality:
                quality_analysis["high_quality_options"].append({
                    "ingredient": ingredient_name,
                    "options": high_quality[:3]  # Top 3 high-quality options
                })
            
            # Analyze quality vs cost tradeoffs
            if len(quality_scores) >= 2:
                cheapest = min(quality_scores, key=lambda x: x["product"].price)
                highest_quality = max(quality_scores, key=lambda x: x["quality_score"])
                
                if cheapest != highest_quality:
                    cost_difference = highest_quality["product"].price - cheapest["product"].price
                    quality_difference = highest_quality["quality_score"] - cheapest["quality_score"]
                    
                    quality_analysis["quality_vs_cost_tradeoffs"].append({
                        "ingredient": ingredient_name,
                        "cheapest": cheapest,
                        "highest_quality": highest_quality,
                        "cost_difference": cost_difference,
                        "quality_difference": quality_difference,
                        "value_ratio": float(quality_difference / cost_difference) if cost_difference > 0 else 0
                    })
        
        state["quality_analysis"] = quality_analysis
        state["optimization_metadata"]["stages_completed"].append("quality_assessment")
        
        return state
    
    def _calculate_product_quality_score(
        self, 
        product: Product, 
        match: Dict[str, Any]
    ) -> float:
        """Calculate quality score for a product."""
        score = 0.0
        
        # Collection method quality (higher confidence methods = higher quality)
        collection_weights = {
            DataCollectionMethod.HUMAN_BROWSER: 1.0,
            DataCollectionMethod.CLIPBOARD_MANUAL: 0.95,
            DataCollectionMethod.API_DIRECT: 0.9,
            DataCollectionMethod.AUTOMATED_STEALTH: 0.8,
            DataCollectionMethod.MOCK_DATA: 0.1
        }
        score += collection_weights.get(product.collection_method, 0.5) * 0.3
        
        # Brand recognition (if known brands are preferred)
        if product.brand:
            score += 0.2
        
        # Product information completeness
        info_completeness = 0.0
        if product.category:
            info_completeness += 0.25
        if product.description:
            info_completeness += 0.25
        if product.nutrition_info:
            info_completeness += 0.25
        if product.image_url:
            info_completeness += 0.25
        
        score += info_completeness * 0.2
        
        # Sale status (good deals can indicate quality retailers)
        if product.on_sale and product.sale_price:
            score += 0.1
        
        # Stock availability
        if product.in_stock:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _generate_shopping_strategies(self, state: OptimizationState) -> OptimizationState:
        """Generate different shopping strategies."""
        self.log_info("Generating shopping strategies")
        
        if state["progress_callback"]:
            state["progress_callback"]("📋 Generating shopping strategies...")
        
        strategies = {}
        
        # Strategy 1: Pure Cost Optimization
        strategies["cost_only"] = await self._generate_cost_only_strategy(state)
        
        # Strategy 2: Convenience (Single Store)
        strategies["convenience"] = await self._generate_convenience_strategy(state)
        
        # Strategy 3: Balanced (Cost + Convenience)
        strategies["balanced"] = await self._generate_balanced_strategy(state)
        
        # Strategy 4: Quality First
        strategies["quality_first"] = await self._generate_quality_first_strategy(state)
        
        # Strategy 5: Time Efficient
        strategies["time_efficient"] = await self._generate_time_efficient_strategy(state)
        
        state["shopping_strategies"] = strategies
        state["optimization_metadata"]["stages_completed"].append("strategy_generation")
        
        return state
    
    async def _generate_cost_only_strategy(self, state: OptimizationState) -> List[ShoppingTrip]:
        """Generate pure cost minimization strategy."""
        trips = {}
        
        for ingredient_name, cheapest_info in state["cost_analysis"]["cheapest_per_ingredient"].items():
            product = cheapest_info["product"]
            store_id = product.store_id
            
            if store_id not in trips:
                trips[store_id] = ShoppingTrip(
                    store_id=store_id,
                    store_name=self.store_info[store_id]["name"],
                    products=[],
                    total_cost=Decimal("0.00"),
                    total_items=0,
                    estimated_time=self.store_info[store_id]["travel_time"],
                    priority_score=1.0,
                    travel_time=self.store_info[store_id]["travel_time"]
                )
            
            # Find the ingredient object
            ingredient = next(
                ing for ing in state["ingredients"] 
                if ing.name == ingredient_name
            )
            
            trips[store_id].products.append((product, ingredient))
            trips[store_id].total_cost += product.sale_price if product.on_sale else product.price
            trips[store_id].total_items += 1
            trips[store_id].estimated_time += 3  # 3 minutes per item
        
        return list(trips.values())
    
    async def _generate_convenience_strategy(self, state: OptimizationState) -> List[ShoppingTrip]:
        """Generate single-store convenience strategy."""
        # Find the best single store option
        best_store = None
        best_score = 0.0
        
        for store_id, option in state["convenience_analysis"]["single_store_options"].items():
            if option["convenience_score"] > best_score:
                best_score = option["convenience_score"]
                best_store = store_id
        
        if not best_store:
            # Fallback to store with highest coverage
            best_store = max(
                state["store_analysis"].keys(),
                key=lambda x: state["store_analysis"][x]["coverage_percentage"]
            )
        
        # Create single shopping trip
        trip = ShoppingTrip(
            store_id=best_store,
            store_name=self.store_info[best_store]["name"],
            products=[],
            total_cost=Decimal("0.00"),
            total_items=0,
            estimated_time=self.store_info[best_store]["travel_time"],
            priority_score=1.0,
            travel_time=self.store_info[best_store]["travel_time"],
            convenience_score=1.0
        )
        
        # Add all available products from this store
        for ingredient in state["ingredients"]:
            matches = state["matched_products"].get(ingredient.name, [])
            store_matches = [
                match for match in matches 
                if match["product"].store_id == best_store
            ]
            
            if store_matches:
                best_match = max(store_matches, key=lambda x: x["confidence"])
                product = best_match["product"]
                trip.products.append((product, ingredient))
                trip.total_cost += product.sale_price if product.on_sale else product.price
                trip.total_items += 1
                trip.estimated_time += 3
        
        return [trip]
    
    async def _generate_balanced_strategy(self, state: OptimizationState) -> List[ShoppingTrip]:
        """Generate balanced cost/convenience strategy."""
        # Use LLM for complex decision making
        prompt = self._create_balanced_strategy_prompt(state)
        
        try:
            # Define response schema
            schema = {
                "type": "object",
                "properties": {
                    "recommended_stores": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["recommended_stores"]
            }
            
            llm_response = await self.llm_client.structured_output(
                prompt=prompt,
                response_schema=schema,
                model=ModelType.PHI3_5_MINI
            )
            
            recommended_stores = llm_response.get("recommended_stores", [])
            
        except Exception as e:
            self.log_error(f"LLM strategy generation failed: {e}")
            # Fallback to top 2 stores by coverage
            recommended_stores = sorted(
                state["store_analysis"].keys(),
                key=lambda x: state["store_analysis"][x]["coverage_percentage"],
                reverse=True
            )[:2]
        
        # Create trips for recommended stores
        trips = self._create_trips_for_stores(recommended_stores, state)
        return trips
    
    def _create_balanced_strategy_prompt(self, state: OptimizationState) -> str:
        """Create prompt for LLM-based balanced strategy generation."""
        store_summaries = []
        for store_id, analysis in state["store_analysis"].items():
            store_info = self.store_info[store_id]
            summary = {
                "store": store_info["name"],
                "coverage": f"{analysis['coverage_percentage']:.1f}%",
                "cost": f"${analysis['total_cost']:.2f}",
                "travel_time": f"{store_info['travel_time']} min",
                "missing": analysis["missing_ingredients"]
            }
            store_summaries.append(json.dumps(summary))
        
        cost_savings = state["cost_analysis"]["cross_store_savings"]
        
        prompt = f"""
You are optimizing a grocery shopping strategy balancing cost and convenience.

INGREDIENTS TO BUY: {len(state['ingredients'])} items
BUDGET: {state['criteria'].max_budget or 'No limit'}
MAX STORES: {state['criteria'].max_stores}

STORE OPTIONS:
{chr(10).join(store_summaries)}

COST ANALYSIS:
- Single store shopping: ${cost_savings['potential_savings']:.2f} more expensive
- Multi-store savings: {cost_savings['savings_percentage']:.1f}%

CONSOLIDATION OPPORTUNITIES:
{json.dumps([{
    'stores': opp['store_names'],
    'coverage': f"{opp['coverage_percentage']:.1f}%",
    'cost': f"${opp['total_cost']:.2f}",
    'travel_time': f"{opp['total_travel_time']} min"
} for opp in state['convenience_analysis']['store_consolidation_opportunities'][:3]])}

Recommend 1-{state['criteria'].max_stores} stores that best balance cost savings with shopping convenience.
Consider travel time, coverage, and total cost.
"""
        return prompt
    
    def _create_trips_for_stores(
        self, 
        store_ids: List[str], 
        state: OptimizationState
    ) -> List[ShoppingTrip]:
        """Create optimized shopping trips for specified stores."""
        trips = {}
        assigned_ingredients = set()
        
        # First pass: assign each ingredient to the best store among selected ones
        for ingredient in state["ingredients"]:
            matches = state["matched_products"].get(ingredient.name, [])
            store_matches = [
                match for match in matches 
                if match["product"].store_id in store_ids
            ]
            
            if not store_matches:
                continue
            
            # Choose best match (considering confidence and price)
            best_match = max(
                store_matches,
                key=lambda x: x["confidence"] * 0.7 + (1.0 / float(x["product"].price)) * 0.3
            )
            
            product = best_match["product"]
            store_id = product.store_id
            
            if store_id not in trips:
                trips[store_id] = ShoppingTrip(
                    store_id=store_id,
                    store_name=self.store_info[store_id]["name"],
                    products=[],
                    total_cost=Decimal("0.00"),
                    total_items=0,
                    estimated_time=self.store_info[store_id]["travel_time"],
                    priority_score=1.0,
                    travel_time=self.store_info[store_id]["travel_time"]
                )
            
            trips[store_id].products.append((product, ingredient))
            trips[store_id].total_cost += product.sale_price if product.on_sale else product.price
            trips[store_id].total_items += 1
            trips[store_id].estimated_time += 3
            assigned_ingredients.add(ingredient.name)
        
        return list(trips.values())
    
    async def _generate_quality_first_strategy(self, state: OptimizationState) -> List[ShoppingTrip]:
        """Generate quality-first strategy."""
        trips = {}
        
        for ingredient_name, quality_scores in state["quality_analysis"]["quality_scores"].items():
            if not quality_scores:
                continue
            
            # Choose highest quality option above threshold
            quality_options = [
                score for score in quality_scores
                if score["quality_score"] >= state["criteria"].quality_threshold
            ]
            
            if not quality_options:
                # Fallback to highest quality available
                quality_options = quality_scores[:1]
            
            best_quality = quality_options[0]
            product = best_quality["product"]
            store_id = product.store_id
            
            if store_id not in trips:
                trips[store_id] = ShoppingTrip(
                    store_id=store_id,
                    store_name=self.store_info[store_id]["name"],
                    products=[],
                    total_cost=Decimal("0.00"),
                    total_items=0,
                    estimated_time=self.store_info[store_id]["travel_time"],
                    priority_score=1.0,
                    travel_time=self.store_info[store_id]["travel_time"],
                    quality_score=best_quality["quality_score"]
                )
            
            ingredient = next(
                ing for ing in state["ingredients"] 
                if ing.name == ingredient_name
            )
            
            trips[store_id].products.append((product, ingredient))
            trips[store_id].total_cost += product.sale_price if product.on_sale else product.price
            trips[store_id].total_items += 1
            trips[store_id].estimated_time += 3
        
        return list(trips.values())
    
    async def _generate_time_efficient_strategy(self, state: OptimizationState) -> List[ShoppingTrip]:
        """Generate time-efficient strategy."""
        # Find most efficient store combinations
        efficiency_options = []
        
        # Single store options
        for store_id, analysis in state["convenience_analysis"]["travel_time_analysis"].items():
            if state["store_analysis"][store_id]["coverage_percentage"] >= 70:
                efficiency_options.append({
                    "stores": [store_id],
                    "efficiency_score": analysis["efficiency_score"],
                    "total_time": analysis["travel_time"] + (
                        len(state["store_analysis"][store_id]["available_ingredients"]) * 3
                    )
                })
        
        # Two-store combinations
        for opp in state["convenience_analysis"]["store_consolidation_opportunities"]:
            if opp["coverage_percentage"] >= 85:
                efficiency_options.append({
                    "stores": opp["stores"],
                    "efficiency_score": opp["efficiency_score"],
                    "total_time": opp["total_travel_time"] + (
                        len(opp.get("covered_ingredients", [])) * 3
                    )
                })
        
        # Choose most time-efficient option
        if efficiency_options:
            best_option = max(efficiency_options, key=lambda x: x["efficiency_score"])
            return self._create_trips_for_stores(best_option["stores"], state)
        else:
            # Fallback to convenience strategy
            return await self._generate_convenience_strategy(state)
    
    async def _apply_multi_criteria_scoring(self, state: OptimizationState) -> OptimizationState:
        """Apply multi-criteria decision framework to score strategies."""
        self.log_info("Applying multi-criteria scoring")
        
        if state["progress_callback"]:
            state["progress_callback"]("⚖️ Evaluating strategy trade-offs...")
        
        criteria_weights = self._determine_criteria_weights(state["strategy"])
        
        scored_strategies = {}
        
        for strategy_name, trips in state["shopping_strategies"].items():
            if not trips:
                continue
                
            scores = {
                "cost_score": self._calculate_cost_score(trips, state),
                "convenience_score": self._calculate_convenience_score_for_trips(trips, state),
                "quality_score": self._calculate_quality_score_for_trips(trips, state),
                "time_score": self._calculate_time_score(trips, state),
                "coverage_score": self._calculate_coverage_score(trips, state)
            }
            
            # Calculate weighted total score
            total_score = (
                scores["cost_score"] * criteria_weights["cost"] +
                scores["convenience_score"] * criteria_weights["convenience"] +
                scores["quality_score"] * criteria_weights["quality"] +
                scores["time_score"] * criteria_weights["time"] +
                scores["coverage_score"] * criteria_weights["coverage"]
            )
            
            scored_strategies[strategy_name] = {
                "trips": trips,
                "scores": scores,
                "total_score": total_score,
                "total_cost": sum(trip.total_cost for trip in trips),
                "total_stores": len(trips),
                "total_time": sum(trip.estimated_time for trip in trips)
            }
        
        state["shopping_strategies"] = scored_strategies
        state["optimization_metadata"]["stages_completed"].append("multi_criteria_scoring")
        
        return state
    
    def _determine_criteria_weights(self, strategy: OptimizationStrategy) -> Dict[str, float]:
        """Determine criteria weights based on optimization strategy."""
        weights = {
            OptimizationStrategy.COST_ONLY: {
                "cost": 0.7, "convenience": 0.1, "quality": 0.1, "time": 0.05, "coverage": 0.05
            },
            OptimizationStrategy.CONVENIENCE: {
                "cost": 0.1, "convenience": 0.5, "quality": 0.2, "time": 0.1, "coverage": 0.1
            },
            OptimizationStrategy.BALANCED: {
                "cost": 0.3, "convenience": 0.3, "quality": 0.2, "time": 0.1, "coverage": 0.1
            },
            OptimizationStrategy.QUALITY_FIRST: {
                "cost": 0.1, "convenience": 0.2, "quality": 0.5, "time": 0.1, "coverage": 0.1
            },
            OptimizationStrategy.TIME_EFFICIENT: {
                "cost": 0.2, "convenience": 0.2, "quality": 0.1, "time": 0.4, "coverage": 0.1
            },
            OptimizationStrategy.ADAPTIVE: {
                "cost": 0.25, "convenience": 0.25, "quality": 0.25, "time": 0.15, "coverage": 0.1
            }
        }
        
        return weights.get(strategy, weights[OptimizationStrategy.BALANCED])
    
    def _calculate_cost_score(self, trips: List[ShoppingTrip], state: OptimizationState) -> float:
        """Calculate cost score (lower cost = higher score)."""
        total_cost = sum(trip.total_cost for trip in trips)
        min_possible_cost = state["cost_analysis"]["total_min_cost"]
        max_cost = state["cost_analysis"]["total_convenience_cost"]
        
        if max_cost <= min_possible_cost:
            return 1.0
        
        # Normalize: 1.0 for min cost, 0.0 for max cost
        normalized_score = 1.0 - float(total_cost - min_possible_cost) / float(max_cost - min_possible_cost)
        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_convenience_score_for_trips(
        self, 
        trips: List[ShoppingTrip], 
        state: OptimizationState
    ) -> float:
        """Calculate convenience score (fewer stores = higher score)."""
        num_stores = len(trips)
        max_stores = state["criteria"].max_stores
        
        # Base score from store count
        store_score = 1.0 - (num_stores - 1) / max(max_stores - 1, 1)
        
        # Adjust for store preferences
        preference_bonus = 0.0
        for trip in trips:
            store_pref = self.store_info[trip.store_id]["preference"]
            if store_pref == StorePreference.PREFERRED:
                preference_bonus += 0.1
            elif store_pref == StorePreference.AVOID:
                preference_bonus -= 0.2
        
        return max(0.0, min(1.0, store_score + preference_bonus))
    
    def _calculate_quality_score_for_trips(
        self, 
        trips: List[ShoppingTrip], 
        state: OptimizationState
    ) -> float:
        """Calculate quality score based on product quality."""
        total_quality = 0.0
        total_items = 0
        
        for trip in trips:
            for product, ingredient in trip.products:
                # Find quality score for this product
                ingredient_quality = state["quality_analysis"]["quality_scores"].get(ingredient.name, [])
                product_quality = next(
                    (score["quality_score"] for score in ingredient_quality 
                     if score["product"] == product),
                    0.5  # Default quality score
                )
                total_quality += product_quality
                total_items += 1
        
        return total_quality / total_items if total_items > 0 else 0.0
    
    def _calculate_time_score(self, trips: List[ShoppingTrip], state: OptimizationState) -> float:
        """Calculate time efficiency score (less time = higher score)."""
        total_time = sum(trip.estimated_time for trip in trips)
        
        # Estimate best possible time (single store with shortest travel time)
        min_travel_time = min(info["travel_time"] for info in self.store_info.values())
        min_shopping_time = len(state["ingredients"]) * 3
        min_possible_time = min_travel_time + min_shopping_time
        
        # Estimate worst case (all stores)
        max_time = sum(info["travel_time"] for info in self.store_info.values()) + min_shopping_time
        
        if max_time <= min_possible_time:
            return 1.0
        
        # Normalize: 1.0 for min time, 0.0 for max time
        normalized_score = 1.0 - (total_time - min_possible_time) / (max_time - min_possible_time)
        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_coverage_score(self, trips: List[ShoppingTrip], state: OptimizationState) -> float:
        """Calculate ingredient coverage score."""
        covered_ingredients = set()
        
        for trip in trips:
            for product, ingredient in trip.products:
                covered_ingredients.add(ingredient.name)
        
        coverage_percentage = len(covered_ingredients) / len(state["ingredients"])
        return coverage_percentage
    
    async def _optimize_shopping_routes(self, state: OptimizationState) -> OptimizationState:
        """Optimize shopping routes and timing."""
        self.log_info("Optimizing shopping routes")
        
        if state["progress_callback"]:
            state["progress_callback"]("🗺️ Optimizing shopping routes...")
        
        # Add route optimization to each strategy
        for strategy_name, strategy_data in state["shopping_strategies"].items():
            trips = strategy_data["trips"]
            
            if len(trips) > 1:
                # Optimize trip order based on travel efficiency
                optimized_trips = self._optimize_trip_order(trips)
                
                # Add timing recommendations
                for i, trip in enumerate(optimized_trips):
                    trip.priority_score = len(trips) - i  # Earlier trips have higher priority
                    
                    # Add time-of-day recommendations
                    if len(trip.products) > 10:  # Large shopping trip
                        strategy_data["timing_recommendation"] = "Morning (less crowded)"
                    elif any(p.category == "bakery" for p, _ in trip.products):
                        strategy_data["timing_recommendation"] = "Morning (fresh baked goods)"
                    else:
                        strategy_data["timing_recommendation"] = "Flexible"
                
                strategy_data["trips"] = optimized_trips
        
        state["optimization_metadata"]["stages_completed"].append("route_optimization")
        return state
    
    def _optimize_trip_order(self, trips: List[ShoppingTrip]) -> List[ShoppingTrip]:
        """Optimize the order of shopping trips."""
        if len(trips) <= 1:
            return trips
        
        # Simple optimization: prioritize by efficiency and perishables
        def trip_priority(trip):
            # Higher priority for stores with fresh items
            fresh_items = sum(
                1 for product, _ in trip.products
                if product.category in ["produce", "dairy", "meat", "bakery"]
            )
            
            # Higher priority for stores with many items (efficiency)
            efficiency = trip.total_items / max(trip.travel_time, 1)
            
            return fresh_items * 2 + efficiency
        
        return sorted(trips, key=trip_priority, reverse=True)
    
    async def _analyze_substitution_opportunities(self, state: OptimizationState) -> OptimizationState:
        """Analyze substitution opportunities to improve optimization."""
        self.log_info("Analyzing substitution opportunities")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔄 Finding substitution opportunities...")
        
        substitution_opportunities = []
        
        # For each strategy, check if substitutions could improve it
        for strategy_name, strategy_data in state["shopping_strategies"].items():
            trips = strategy_data["trips"]
            strategy_substitutions = []
            
            # Check each ingredient for better alternatives
            covered_ingredients = set()
            for trip in trips:
                for product, ingredient in trip.products:
                    covered_ingredients.add(ingredient.name)
            
            # Find missing ingredients and suggest substitutions
            missing_ingredients = [
                ing for ing in state["ingredients"]
                if ing.name not in covered_ingredients
            ]
            
            for missing_ingredient in missing_ingredients:
                # Try to find substitutions using MatcherAgent
                try:
                    substitution_result = await self.matcher_agent.suggest_substitutions(
                        ingredient_name=missing_ingredient.name,
                        max_suggestions=3
                    )
                    
                    if substitution_result:
                        strategy_substitutions.extend(substitution_result)
                        
                except Exception as e:
                    self.log_error(f"Substitution analysis failed for {missing_ingredient.name}: {e}")
            
            if strategy_substitutions:
                strategy_data["substitution_suggestions"] = strategy_substitutions[:5]  # Top 5
        
        state["optimization_metadata"]["stages_completed"].append("substitution_analysis")
        return state
    
    async def _validate_budget_constraints(self, state: OptimizationState) -> OptimizationState:
        """Validate budget constraints and adjust strategies if needed."""
        self.log_info("Validating budget constraints")
        
        if state["progress_callback"]:
            state["progress_callback"]("💵 Validating budget constraints...")
        
        max_budget = state["criteria"].max_budget
        
        if max_budget:
            # Check each strategy against budget
            for strategy_name, strategy_data in state["shopping_strategies"].items():
                total_cost = strategy_data["total_cost"]
                
                if total_cost > max_budget:
                    # Strategy exceeds budget - try to adjust
                    adjusted_trips = self._adjust_for_budget(
                        strategy_data["trips"], 
                        max_budget,
                        state
                    )
                    
                    if adjusted_trips:
                        strategy_data["trips"] = adjusted_trips
                        strategy_data["total_cost"] = sum(trip.total_cost for trip in adjusted_trips)
                        strategy_data["budget_adjusted"] = True
                        strategy_data["budget_savings"] = float(total_cost - strategy_data["total_cost"])
                    else:
                        # Cannot adjust to fit budget
                        strategy_data["budget_violation"] = True
                        strategy_data["budget_excess"] = float(total_cost - max_budget)
        
        state["optimization_metadata"]["stages_completed"].append("budget_validation")
        return state
    
    def _adjust_for_budget(
        self, 
        trips: List[ShoppingTrip], 
        max_budget: Decimal,
        state: OptimizationState
    ) -> Optional[List[ShoppingTrip]]:
        """Adjust shopping trips to fit within budget."""
        current_total = sum(trip.total_cost for trip in trips)
        
        if current_total <= max_budget:
            return trips
        
        # Try to reduce costs by substituting with cheaper alternatives
        adjusted_trips = []
        
        for trip in trips:
            adjusted_trip = ShoppingTrip(
                store_id=trip.store_id,
                store_name=trip.store_name,
                products=[],
                total_cost=Decimal("0.00"),
                total_items=0,
                estimated_time=trip.travel_time,
                priority_score=trip.priority_score,
                travel_time=trip.travel_time
            )
            
            for product, ingredient in trip.products:
                # Look for cheaper alternatives
                matches = state["matched_products"].get(ingredient.name, [])
                store_matches = [
                    match for match in matches
                    if match["product"].store_id == trip.store_id
                ]
                
                if store_matches:
                    # Sort by price (cheapest first)
                    cheapest_matches = sorted(
                        store_matches,
                        key=lambda x: x["product"].sale_price if x["product"].on_sale 
                        else x["product"].price
                    )
                    
                    # Use cheapest option that meets minimum quality
                    suitable_match = None
                    for match in cheapest_matches:
                        if match["confidence"] >= 0.5:  # Minimum confidence threshold
                            suitable_match = match
                            break
                    
                    if suitable_match:
                        adjusted_product = suitable_match["product"]
                        adjusted_trip.products.append((adjusted_product, ingredient))
                        adjusted_trip.total_cost += (
                            adjusted_product.sale_price if adjusted_product.on_sale 
                            else adjusted_product.price
                        )
                        adjusted_trip.total_items += 1
                        adjusted_trip.estimated_time += 3
            
            if adjusted_trip.products:  # Only add if we found products
                adjusted_trips.append(adjusted_trip)
        
        # Check if adjusted trips fit budget
        adjusted_total = sum(trip.total_cost for trip in adjusted_trips)
        
        if adjusted_total <= max_budget:
            return adjusted_trips
        else:
            return None  # Cannot fit within budget
    
    async def _select_best_recommendation(self, state: OptimizationState) -> OptimizationState:
        """Select the best recommendation based on strategy and scoring."""
        self.log_info("Selecting best recommendation")
        
        if state["progress_callback"]:
            state["progress_callback"]("🎯 Selecting optimal strategy...")
        
        # Filter out strategies that violate constraints
        valid_strategies = {}
        
        for strategy_name, strategy_data in state["shopping_strategies"].items():
            # Check budget constraint
            if state["criteria"].max_budget and strategy_data.get("budget_violation"):
                continue
            
            # Check store count constraint
            if len(strategy_data["trips"]) > state["criteria"].max_stores:
                continue
            
            valid_strategies[strategy_name] = strategy_data
        
        if not valid_strategies:
            # No valid strategies - relax constraints
            self.log_info("No strategies meet all constraints, selecting best available")
            valid_strategies = state["shopping_strategies"]
        
        # Select best strategy based on total score
        best_strategy_name = max(
            valid_strategies.keys(),
            key=lambda name: valid_strategies[name]["total_score"]
        )
        
        best_strategy = valid_strategies[best_strategy_name]
        state["recommended_strategy"] = best_strategy["trips"]
        
        # Store alternative strategies (excluding the selected one)
        alternatives = {
            name: data for name, data in valid_strategies.items()
            if name != best_strategy_name
        }
        state["optimization_metadata"]["alternative_strategies"] = alternatives
        state["optimization_metadata"]["selected_strategy"] = best_strategy_name
        state["optimization_metadata"]["selection_reasoning"] = self._generate_selection_reasoning(
            best_strategy_name, best_strategy, alternatives
        )
        
        state["optimization_metadata"]["stages_completed"].append("recommendation_selection")
        return state
    
    def _generate_selection_reasoning(
        self,
        selected_strategy: str,
        selected_data: Dict[str, Any],
        alternatives: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate reasoning for strategy selection."""
        reasons = []
        
        reasons.append(f"Selected '{selected_strategy}' strategy")
        reasons.append(f"Total score: {selected_data['total_score']:.3f}")
        reasons.append(f"Total cost: ${selected_data['total_cost']:.2f}")
        reasons.append(f"Number of stores: {selected_data['total_stores']}")
        reasons.append(f"Estimated time: {selected_data['total_time']} minutes")
        
        # Compare with alternatives
        if alternatives:
            cost_comparison = min(alt["total_cost"] for alt in alternatives.values())
            if selected_data["total_cost"] <= cost_comparison:
                reasons.append("Offers the lowest total cost")
            
            store_comparison = min(alt["total_stores"] for alt in alternatives.values())
            if selected_data["total_stores"] <= store_comparison:
                reasons.append("Requires the fewest store visits")
            
            time_comparison = min(alt["total_time"] for alt in alternatives.values())
            if selected_data["total_time"] <= time_comparison:
                reasons.append("Has the shortest total shopping time")
        
        return "; ".join(reasons)
    
    async def _finalize_optimization(self, state: OptimizationState) -> OptimizationState:
        """Finalize the optimization session."""
        self.log_info("Finalizing optimization")
        
        state["optimization_metadata"]["end_time"] = datetime.now().isoformat()
        state["optimization_metadata"]["total_duration"] = (
            datetime.fromisoformat(state["optimization_metadata"]["end_time"]) -
            datetime.fromisoformat(state["optimization_metadata"]["start_time"])
        ).total_seconds()
        
        if state["progress_callback"]:
            total_trips = len(state["recommended_strategy"])
            total_cost = sum(trip.total_cost for trip in state["recommended_strategy"])
            state["progress_callback"](
                f"✅ Optimization complete! {total_trips} shopping trips, "
                f"total cost: ${total_cost:.2f}"
            )
        
        self.log_info(
            f"Optimization completed: {len(state['recommended_strategy'])} trips, "
            f"strategy: {state['optimization_metadata']['selected_strategy']}"
        )
        
        state["optimization_metadata"]["stages_completed"].append("finalize")
        return state
    
    # Utility Methods
    
    def _compile_optimization_results(self, state: OptimizationState) -> Dict[str, Any]:
        """Compile final optimization results."""
        recommended_trips = state["recommended_strategy"]
        
        # Calculate savings
        convenience_cost = state["cost_analysis"]["total_convenience_cost"]
        optimized_cost = sum(trip.total_cost for trip in recommended_trips)
        total_savings = convenience_cost - optimized_cost
        savings_percentage = float(total_savings / convenience_cost * 100) if convenience_cost > 0 else 0
        
        # Find unmatched ingredients
        covered_ingredients = set()
        for trip in recommended_trips:
            for product, ingredient in trip.products:
                covered_ingredients.add(ingredient.name)
        
        unmatched_ingredients = [
            ing for ing in state["ingredients"]
            if ing.name not in covered_ingredients
        ]
        
        # Compile alternative strategies
        alternative_strategies = {}
        for name, data in state["optimization_metadata"].get("alternative_strategies", {}).items():
            alternative_strategies[name] = {
                "trips": data["trips"],
                "total_cost": float(data["total_cost"]),
                "total_stores": data["total_stores"],
                "total_time": data["total_time"],
                "total_score": data["total_score"]
            }
        
        return {
            "success": True,
            "recommended_strategy": [
                {
                    "store_id": trip.store_id,
                    "store_name": trip.store_name,
                    "products": [
                        {
                            "product": {
                                "name": product.name,
                                "brand": product.brand,
                                "price": float(product.price),
                                "sale_price": float(product.sale_price) if product.sale_price else None,
                                "on_sale": product.on_sale,
                                "store_id": product.store_id
                            },
                            "ingredient": {
                                "name": ingredient.name,
                                "quantity": ingredient.quantity,
                                "unit": ingredient.unit
                            }
                        }
                        for product, ingredient in trip.products
                    ],
                    "total_cost": float(trip.total_cost),
                    "total_items": trip.total_items,
                    "estimated_time": trip.estimated_time,
                    "travel_time": trip.travel_time,
                    "priority_score": trip.priority_score
                }
                for trip in recommended_trips
            ],
            "alternative_strategies": alternative_strategies,
            "savings_analysis": {
                "total_savings": float(total_savings),
                "savings_percentage": savings_percentage,
                "convenience_cost": float(convenience_cost),
                "optimized_cost": float(optimized_cost)
            },
            "optimization_summary": {
                "selected_strategy": state["optimization_metadata"]["selected_strategy"],
                "selection_reasoning": state["optimization_metadata"]["selection_reasoning"],
                "total_stores": len(recommended_trips),
                "total_items": sum(trip.total_items for trip in recommended_trips),
                "total_time": sum(trip.estimated_time for trip in recommended_trips),
                "coverage_percentage": (
                    len(covered_ingredients) / len(state["ingredients"]) * 100
                    if state["ingredients"] else 0
                )
            },
            "unmatched_ingredients": [
                {
                    "name": ing.name,
                    "quantity": ing.quantity,
                    "unit": ing.unit,
                    "category": ing.category
                }
                for ing in unmatched_ingredients
            ],
            "optimization_metadata": state["optimization_metadata"]
        }
    
    def _update_optimization_stats(self, state: OptimizationState) -> None:
        """Update optimization statistics."""
        self.optimization_stats["total_optimizations"] += 1
        
        # Calculate savings percentage
        convenience_cost = state["cost_analysis"]["total_convenience_cost"]
        optimized_cost = sum(trip.total_cost for trip in state["recommended_strategy"])
        
        if convenience_cost > 0:
            savings_percentage = float((convenience_cost - optimized_cost) / convenience_cost * 100)
            
            # Update running average
            total_opts = self.optimization_stats["total_optimizations"]
            current_avg = self.optimization_stats["avg_savings_percentage"]
            self.optimization_stats["avg_savings_percentage"] = (
                (current_avg * (total_opts - 1) + savings_percentage) / total_opts
            )
        
        # Update stores reduced
        max_possible_stores = len(self.store_info)
        stores_used = len(state["recommended_strategy"])
        stores_reduced = max_possible_stores - stores_used
        
        total_opts = self.optimization_stats["total_optimizations"]
        current_avg_reduced = self.optimization_stats["avg_stores_reduced"]
        self.optimization_stats["avg_stores_reduced"] = (
            (current_avg_reduced * (total_opts - 1) + stores_reduced) / total_opts
        )
        
        # Update strategy performance
        selected_strategy = state["optimization_metadata"]["selected_strategy"]
        if selected_strategy in self.optimization_stats["strategy_performance"]:
            self.optimization_stats["strategy_performance"][selected_strategy]["uses"] += 1
    
    def get_optimization_analytics(self) -> Dict[str, Any]:
        """Get analytics about optimization performance."""
        return {
            "optimization_stats": self.optimization_stats,
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for improving optimization performance."""
        recommendations = []
        
        stats = self.optimization_stats
        
        if stats["total_optimizations"] > 0:
            if stats["avg_savings_percentage"] < 10:
                recommendations.append("Consider expanding product database for better cost optimization")
            
            if stats["avg_stores_reduced"] < 1:
                recommendations.append("Review store consolidation logic to reduce shopping trips")
            
            # Analyze strategy performance
            strategy_perf = stats["strategy_performance"]
            most_used = max(strategy_perf.keys(), key=lambda k: strategy_perf[k]["uses"])
            
            if strategy_perf[most_used]["uses"] / stats["total_optimizations"] > 0.7:
                recommendations.append(f"Strategy '{most_used}' is overused - consider improving other strategies")
        
        return recommendations
    
    # Public API Methods
    
    async def optimize_shopping_list(
        self,
        ingredients: List[Ingredient],
        criteria: Optional[OptimizationCriteria] = None,
        strategy: str = "adaptive"
    ) -> Dict[str, Any]:
        """Public API for optimizing a shopping list."""
        return await self.execute({
            "ingredients": ingredients,
            "criteria": criteria or OptimizationCriteria(),
            "strategy": strategy
        })
    
    async def compare_strategies(
        self,
        ingredients: List[Ingredient],
        strategies: List[str],
        criteria: Optional[OptimizationCriteria] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple optimization strategies."""
        results = {}
        
        for strategy in strategies:
            try:
                result = await self.optimize_shopping_list(
                    ingredients=ingredients,
                    criteria=criteria,
                    strategy=strategy
                )
                results[strategy] = result
            except Exception as e:
                self.log_error(f"Strategy comparison failed for {strategy}: {e}")
                results[strategy] = {"success": False, "error": str(e)}
        
        return results
    
    async def estimate_savings(
        self,
        ingredients: List[Ingredient],
        current_shopping_method: str = "convenience"
    ) -> Dict[str, Any]:
        """Estimate potential savings from optimization."""
        # Get current method cost
        current_result = await self.optimize_shopping_list(
            ingredients=ingredients,
            strategy=current_shopping_method
        )
        
        # Get optimized cost
        optimized_result = await self.optimize_shopping_list(
            ingredients=ingredients,
            strategy="cost_only"
        )
        
        if current_result["success"] and optimized_result["success"]:
            current_cost = current_result["savings_analysis"]["optimized_cost"]
            optimized_cost = optimized_result["savings_analysis"]["optimized_cost"]
            potential_savings = current_cost - optimized_cost
            savings_percentage = (potential_savings / current_cost * 100) if current_cost > 0 else 0
            
            return {
                "current_cost": current_cost,
                "optimized_cost": optimized_cost,
                "potential_savings": potential_savings,
                "savings_percentage": savings_percentage,
                "recommendation": "optimization_worthwhile" if savings_percentage > 10 else "minimal_benefit"
            }
        else:
            return {
                "success": False,
                "error": "Failed to calculate savings estimate"
            }