"""
MatcherAgent for intelligently matching ingredients to products using vector search and local LLMs.
This agent uses LangGraph for workflow orchestration and combines semantic search with LLM reasoning.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Callable
from enum import Enum
from datetime import datetime
from difflib import SequenceMatcher

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..data_models.ingredient import Ingredient
from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod
from ..vector_db.qdrant_client import QdrantVectorDB
from ..llm_client.ollama_client import OllamaClient, ModelType
from ..llm_client.prompt_templates import PromptTemplates
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MatchingStrategy(Enum):
    """Strategy for ingredient-to-product matching."""
    
    VECTOR_ONLY = "vector_only"
    LLM_ONLY = "llm_only" 
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class MatchingQuality(Enum):
    """Quality levels for matching results."""
    
    EXCELLENT = "excellent"  # 0.9+ confidence
    GOOD = "good"           # 0.7-0.89 confidence
    FAIR = "fair"           # 0.5-0.69 confidence
    POOR = "poor"           # 0.3-0.49 confidence
    REJECTED = "rejected"    # <0.3 confidence


class MatchingState(TypedDict):
    """State structure for LangGraph matching workflow."""
    
    ingredient: Ingredient
    search_query: str
    strategy: MatchingStrategy
    vector_candidates: List[Tuple[Product, float]]
    brand_normalized_candidates: List[Tuple[Product, float]]
    llm_analysis: Dict[str, Any]
    final_matches: List[Dict[str, Any]]
    confidence_threshold: float
    max_results: int
    require_human_review: bool
    substitution_suggestions: List[Dict[str, Any]]
    category_analysis: Dict[str, Any]
    matching_metadata: Dict[str, Any]
    progress_callback: Optional[Callable[[str], None]]


class MatcherAgent(BaseAgent):
    """LangGraph-based intelligent ingredient-to-product matcher."""
    
    def __init__(
        self,
        vector_db: Optional[QdrantVectorDB] = None,
        llm_client: Optional[OllamaClient] = None
    ):
        """Initialize the matcher agent."""
        super().__init__("matcher")
        
        # Initialize components
        self.vector_db = vector_db or QdrantVectorDB(in_memory=True)
        self.llm_client = llm_client or OllamaClient()
        
        # Matching statistics
        self.matching_stats = {
            "total_matches": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "avg_confidence": 0.0,
            "quality_distribution": {
                quality.value: 0 for quality in MatchingQuality
            },
            "strategy_performance": {
                strategy.value: {"attempts": 0, "successes": 0}
                for strategy in MatchingStrategy
            }
        }
        
        # Build LangGraph workflow
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute intelligent ingredient-to-product matching.
        
        Args:
            inputs: Dictionary containing:
                - ingredient: Ingredient object or dict
                - strategy: Matching strategy (optional)
                - confidence_threshold: Minimum confidence (optional)
                - max_results: Maximum matches to return (optional)
                - progress_callback: Progress callback function (optional)
        
        Returns:
            Dictionary with matching results and metadata
        """
        # Parse ingredient input
        ingredient_data = inputs.get("ingredient")
        if isinstance(ingredient_data, dict):
            ingredient = Ingredient(**ingredient_data)
        elif isinstance(ingredient_data, Ingredient):
            ingredient = ingredient_data
        else:
            raise ValueError("Ingredient must be provided as Ingredient object or dict")
        
        # Initialize state
        initial_state: MatchingState = {
            "ingredient": ingredient,
            "search_query": ingredient.name,
            "strategy": MatchingStrategy(inputs.get("strategy", "adaptive")),
            "vector_candidates": [],
            "brand_normalized_candidates": [],
            "llm_analysis": {},
            "final_matches": [],
            "confidence_threshold": inputs.get("confidence_threshold", 0.5),
            "max_results": inputs.get("max_results", 5),
            "require_human_review": False,
            "substitution_suggestions": [],
            "category_analysis": {},
            "matching_metadata": {
                "start_time": datetime.now().isoformat(),
                "stages_completed": [],
                "performance_metrics": {}
            },
            "progress_callback": inputs.get("progress_callback")
        }
        
        self.log_info(f"Starting ingredient matching for: '{ingredient.name}'")
        
        # Execute workflow
        try:
            config = {"configurable": {"thread_id": f"match_{int(datetime.now().timestamp())}"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Update statistics
            self._update_matching_stats(final_state)
            
            # Compile results
            result = {
                "success": True,
                "ingredient": ingredient.name,
                "matches": final_state["final_matches"],
                "total_matches": len(final_state["final_matches"]),
                "substitution_suggestions": final_state["substitution_suggestions"],
                "category_analysis": final_state["category_analysis"],
                "require_human_review": final_state["require_human_review"],
                "matching_metadata": final_state["matching_metadata"],
                "quality_distribution": self._analyze_match_quality(final_state["final_matches"])
            }
            
            self.log_info(f"Matching completed: {len(final_state['final_matches'])} matches found")
            return result
            
        except Exception as e:
            self.log_error(f"Matching workflow failed: {e}", e)
            return {
                "success": False,
                "error": str(e),
                "ingredient": ingredient.name,
                "matches": [],
                "total_matches": 0
            }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for intelligent matching."""
        workflow = StateGraph(MatchingState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_matching)
        workflow.add_node("normalize_query", self._normalize_search_query)
        workflow.add_node("vector_search", self._perform_vector_search)
        workflow.add_node("brand_normalization", self._normalize_brands)
        workflow.add_node("llm_analysis", self._perform_llm_analysis)
        workflow.add_node("fuzzy_matching", self._perform_fuzzy_matching)
        workflow.add_node("confidence_scoring", self._calculate_confidence_scores)
        workflow.add_node("quality_control", self._apply_quality_control)
        workflow.add_node("substitution_analysis", self._analyze_substitutions)
        workflow.add_node("category_validation", self._validate_categories)
        workflow.add_node("final_ranking", self._rank_final_matches)
        workflow.add_node("finalize", self._finalize_matching)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "normalize_query")
        workflow.add_edge("normalize_query", "vector_search")
        workflow.add_edge("vector_search", "brand_normalization")
        workflow.add_edge("brand_normalization", "llm_analysis")
        workflow.add_edge("llm_analysis", "fuzzy_matching")
        workflow.add_edge("fuzzy_matching", "confidence_scoring")
        workflow.add_edge("confidence_scoring", "quality_control")
        workflow.add_edge("quality_control", "substitution_analysis")
        workflow.add_edge("substitution_analysis", "category_validation")
        workflow.add_edge("category_validation", "final_ranking")
        workflow.add_edge("final_ranking", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    # Workflow Node Functions
    
    async def _initialize_matching(self, state: MatchingState) -> MatchingState:
        """Initialize matching session."""
        self.log_info("Initializing ingredient matching session")
        
        if state["progress_callback"]:
            state["progress_callback"](f"🔍 Initializing matching for '{state['ingredient'].name}'...")
        
        state["matching_metadata"]["stages_completed"].append("initialize")
        state["matching_metadata"]["ingredient_category"] = state["ingredient"].category
        state["matching_metadata"]["ingredient_alternatives"] = state["ingredient"].alternatives
        
        return state
    
    async def _normalize_search_query(self, state: MatchingState) -> MatchingState:
        """Normalize the search query using LLM."""
        self.log_info("Normalizing search query")
        
        if state["progress_callback"]:
            state["progress_callback"]("🧠 Normalizing ingredient name...")
        
        try:
            # Use LLM to normalize the ingredient name
            prompt = PromptTemplates.format_template(
                "NORMALIZE_INGREDIENT",
                ingredient=state["ingredient"].name
            )
            
            normalized_query = await self.llm_client.generate(
                prompt=prompt,
                model=ModelType.QWEN_1_5B,
                temperature=0.1
            )
            
            state["search_query"] = normalized_query.strip()
            state["matching_metadata"]["normalized_query"] = normalized_query.strip()
            
            self.log_info(f"Normalized query: '{state['search_query']}'")
            
        except Exception as e:
            self.log_error(f"Query normalization failed: {e}")
            # Fallback to original name
            state["search_query"] = state["ingredient"].name
        
        state["matching_metadata"]["stages_completed"].append("normalize_query")
        return state
    
    async def _perform_vector_search(self, state: MatchingState) -> MatchingState:
        """Perform semantic vector search."""
        self.log_info("Performing vector search")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔍 Searching product database...")
        
        try:
            # Search with normalized query
            vector_results = self.vector_db.search_similar_products(
                query=state["search_query"],
                limit=state["max_results"] * 3,  # Get more candidates for filtering
                min_confidence=0.0,  # No filtering at this stage
                in_stock_only=True,
                use_confidence_weighting=True
            )
            
            state["vector_candidates"] = vector_results
            
            # Initialize seen_products set
            seen_products = set()
            for product, score in vector_results:
                product_key = (product.name, product.store_id)
                seen_products.add(product_key)
            
            # Also search with original ingredient name if different
            if state["search_query"] != state["ingredient"].name:
                original_results = self.vector_db.search_similar_products(
                    query=state["ingredient"].name,
                    limit=state["max_results"] * 2,
                    min_confidence=0.0,
                    in_stock_only=True,
                    use_confidence_weighting=True
                )
                
                # Merge results and deduplicate
                all_candidates = vector_results + original_results
                deduplicated = []
                
                for product, score in all_candidates:
                    product_key = (product.name, product.store_id)
                    if product_key not in seen_products:
                        seen_products.add(product_key)
                        deduplicated.append((product, score))
                
                state["vector_candidates"] = sorted(deduplicated, key=lambda x: x[1], reverse=True)
            
            # Search with alternatives if provided
            if state["ingredient"].alternatives:
                for alternative in state["ingredient"].alternatives:
                    alt_results = self.vector_db.search_similar_products(
                        query=alternative,
                        limit=state["max_results"],
                        min_confidence=0.0,
                        in_stock_only=True,
                        use_confidence_weighting=True
                    )
                    
                    # Add to candidates
                    for product, score in alt_results:
                        product_key = (product.name, product.store_id)
                        if product_key not in seen_products:
                            seen_products.add(product_key)
                            state["vector_candidates"].append((product, score * 0.9))  # Slightly lower score for alternatives
            
            # Re-sort all candidates
            state["vector_candidates"] = sorted(state["vector_candidates"], key=lambda x: x[1], reverse=True)
            
            self.log_info(f"Found {len(state['vector_candidates'])} vector candidates")
            
        except Exception as e:
            self.log_error(f"Vector search failed: {e}")
            state["vector_candidates"] = []
        
        state["matching_metadata"]["stages_completed"].append("vector_search")
        state["matching_metadata"]["vector_candidates_count"] = len(state["vector_candidates"])
        
        return state
    
    async def _normalize_brands(self, state: MatchingState) -> MatchingState:
        """Normalize brand names for better matching."""
        self.log_info("Normalizing brand names")
        
        if state["progress_callback"]:
            state["progress_callback"]("🏷️ Analyzing brands...")
        
        try:
            brand_normalized = []
            
            for product, vector_score in state["vector_candidates"]:
                # Calculate brand similarity
                brand_score = self._calculate_brand_similarity(
                    state["ingredient"].name,
                    product.name,
                    product.brand
                )
                
                # Combine vector and brand scores
                combined_score = (vector_score * 0.7) + (brand_score * 0.3)
                brand_normalized.append((product, combined_score))
            
            state["brand_normalized_candidates"] = sorted(
                brand_normalized, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            self.log_info(f"Brand normalization completed for {len(brand_normalized)} candidates")
            
        except Exception as e:
            self.log_error(f"Brand normalization failed: {e}")
            state["brand_normalized_candidates"] = state["vector_candidates"]
        
        state["matching_metadata"]["stages_completed"].append("brand_normalization")
        return state
    
    def _calculate_brand_similarity(self, ingredient_name: str, product_name: str, brand: Optional[str]) -> float:
        """Calculate brand similarity score."""
        if not brand:
            return 0.5  # Neutral score for generic products
        
        ingredient_lower = ingredient_name.lower()
        product_lower = product_name.lower()
        brand_lower = brand.lower()
        
        # Check if brand is mentioned in ingredient
        if brand_lower in ingredient_lower:
            return 1.0
        
        # Check for common brand variations
        brand_variations = self._get_brand_variations(brand_lower)
        for variation in brand_variations:
            if variation in ingredient_lower:
                return 0.9
        
        # Check if ingredient specifies a different brand
        known_brands = ["kraft", "nestle", "kellogs", "general mills", "heinz", "pepsi", "coca cola"]
        for known_brand in known_brands:
            if known_brand in ingredient_lower and known_brand != brand_lower:
                return 0.3  # Lower score if different brand specified
        
        return 0.6  # Default score when no specific brand preference
    
    def _get_brand_variations(self, brand: str) -> List[str]:
        """Get common variations of a brand name."""
        variations = [brand]
        
        # Common brand abbreviations and variations
        brand_mappings = {
            "kellogg": ["kellogs", "kelloggs"],
            "kellogs": ["kellogg", "kelloggs"],
            "coca cola": ["coke", "coca-cola"],
            "pepsi": ["pepsi-cola"],
            "general mills": ["gm"],
            "kraft": ["kraft-heinz"],
        }
        
        for canonical, variants in brand_mappings.items():
            if brand in canonical or canonical in brand:
                variations.extend(variants)
        
        return list(set(variations))
    
    async def _perform_llm_analysis(self, state: MatchingState) -> MatchingState:
        """Use LLM to analyze and rank product matches."""
        self.log_info("Performing LLM analysis")
        
        if state["progress_callback"]:
            state["progress_callback"]("🧠 Analyzing matches with AI...")
        
        try:
            # Prepare product list for LLM analysis
            top_candidates = state["brand_normalized_candidates"][:10]  # Limit for LLM context
            
            product_list = []
            for i, (product, score) in enumerate(top_candidates):
                product_info = {
                    "index": i,
                    "name": product.name,
                    "brand": product.brand or "Generic",
                    "price": f"${product.price}",
                    "store": product.store_id,
                    "category": product.category or "Unknown",
                    "vector_score": round(score, 3)
                }
                product_list.append(json.dumps(product_info))
            
            # Use LLM to analyze matches
            prompt = PromptTemplates.format_template(
                "MATCH_PRODUCTS",
                ingredient=state["ingredient"].name,
                product_list="\n".join(product_list)
            )
            
            # Define response schema
            schema = {
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
                            },
                            "required": ["product_name", "confidence", "reason"]
                        }
                    }
                },
                "required": ["matches"]
            }
            
            llm_response = await self.llm_client.structured_output(
                prompt=prompt,
                response_schema=schema,
                model=ModelType.QWEN_1_5B
            )
            
            state["llm_analysis"] = llm_response
            
            self.log_info(f"LLM analyzed {len(llm_response.get('matches', []))} matches")
            
        except Exception as e:
            self.log_error(f"LLM analysis failed: {e}")
            state["llm_analysis"] = {"matches": []}
        
        state["matching_metadata"]["stages_completed"].append("llm_analysis")
        return state
    
    async def _perform_fuzzy_matching(self, state: MatchingState) -> MatchingState:
        """Perform fuzzy string matching for exact name matches."""
        self.log_info("Performing fuzzy matching")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔤 Checking name similarities...")
        
        # Add fuzzy scores to candidates
        enhanced_candidates = []
        
        for product, score in state["brand_normalized_candidates"]:
            # Calculate fuzzy similarity
            fuzzy_score = SequenceMatcher(
                None,
                state["ingredient"].name.lower(),
                product.name.lower()
            ).ratio()
            
            # Boost score for very similar names
            if fuzzy_score > 0.8:
                enhanced_score = score * (1 + fuzzy_score * 0.2)
            else:
                enhanced_score = score
            
            enhanced_candidates.append((product, enhanced_score, fuzzy_score))
        
        state["brand_normalized_candidates"] = [
            (product, score) for product, score, _ in enhanced_candidates
        ]
        
        state["matching_metadata"]["stages_completed"].append("fuzzy_matching")
        return state
    
    async def _calculate_confidence_scores(self, state: MatchingState) -> MatchingState:
        """Calculate final confidence scores combining all signals."""
        self.log_info("Calculating confidence scores")
        
        if state["progress_callback"]:
            state["progress_callback"]("📊 Calculating confidence scores...")
        
        # Combine vector scores, LLM analysis, and fuzzy matching
        llm_matches = {match["product_name"]: match for match in state["llm_analysis"].get("matches", [])}
        
        scored_matches = []
        
        for product, vector_score in state["brand_normalized_candidates"]:
            # Get LLM confidence if available
            llm_match = llm_matches.get(product.name)
            llm_confidence = llm_match["confidence"] if llm_match else 0.5
            llm_reason = llm_match["reason"] if llm_match else "No LLM analysis"
            
            # Calculate combined confidence
            combined_confidence = (
                vector_score * 0.4 +
                llm_confidence * 0.4 +
                product.get_collection_confidence_weight() * 0.2
            )
            
            # Quality boost for exact category matches
            if (state["ingredient"].category and 
                product.category and 
                state["ingredient"].category.lower() in product.category.lower()):
                combined_confidence *= 1.1
            
            # Create match object
            match = {
                "product": product,
                "confidence": min(combined_confidence, 1.0),  # Cap at 1.0
                "vector_score": vector_score,
                "llm_confidence": llm_confidence,
                "llm_reason": llm_reason,
                "quality": self._determine_match_quality(combined_confidence)
            }
            
            scored_matches.append(match)
        
        # Sort by confidence
        scored_matches = sorted(scored_matches, key=lambda x: x["confidence"], reverse=True)
        
        # Apply confidence threshold
        filtered_matches = [
            match for match in scored_matches 
            if match["confidence"] >= state["confidence_threshold"]
        ]
        
        # Limit to max results
        state["final_matches"] = filtered_matches[:state["max_results"]]
        
        self.log_info(f"Calculated confidence for {len(scored_matches)} matches, {len(state['final_matches'])} above threshold")
        
        state["matching_metadata"]["stages_completed"].append("confidence_scoring")
        state["matching_metadata"]["total_scored_matches"] = len(scored_matches)
        state["matching_metadata"]["matches_above_threshold"] = len(state["final_matches"])
        
        return state
    
    def _determine_match_quality(self, confidence: float) -> MatchingQuality:
        """Determine match quality based on confidence score."""
        if confidence >= 0.9:
            return MatchingQuality.EXCELLENT
        elif confidence >= 0.7:
            return MatchingQuality.GOOD
        elif confidence >= 0.5:
            return MatchingQuality.FAIR
        elif confidence >= 0.3:
            return MatchingQuality.POOR
        else:
            return MatchingQuality.REJECTED
    
    async def _apply_quality_control(self, state: MatchingState) -> MatchingState:
        """Apply quality control checks and flag for human review."""
        self.log_info("Applying quality control")
        
        if state["progress_callback"]:
            state["progress_callback"]("✅ Applying quality control...")
        
        # Check if human review is needed
        if not state["final_matches"]:
            state["require_human_review"] = True
            state["matching_metadata"]["human_review_reason"] = "No matches found above confidence threshold"
        
        elif all(match["quality"] in [MatchingQuality.POOR, MatchingQuality.FAIR] for match in state["final_matches"]):
            state["require_human_review"] = True
            state["matching_metadata"]["human_review_reason"] = "All matches have low confidence"
        
        elif len(state["final_matches"]) == 1 and state["final_matches"][0]["confidence"] < 0.8:
            state["require_human_review"] = True
            state["matching_metadata"]["human_review_reason"] = "Single match with moderate confidence"
        
        # Additional validation checks
        validated_matches = []
        for match in state["final_matches"]:
            # Skip products with suspicious pricing
            if match["product"].price <= 0:
                continue
            
            # Skip products with very generic names for specific ingredients
            if len(state["ingredient"].name.split()) > 2 and len(match["product"].name.split()) <= 1:
                match["confidence"] *= 0.8  # Reduce confidence
            
            validated_matches.append(match)
        
        state["final_matches"] = validated_matches
        
        state["matching_metadata"]["stages_completed"].append("quality_control")
        return state
    
    async def _analyze_substitutions(self, state: MatchingState) -> MatchingState:
        """Analyze potential substitutions for the ingredient."""
        self.log_info("Analyzing substitutions")
        
        if state["progress_callback"]:
            state["progress_callback"]("🔄 Finding substitutions...")
        
        try:
            # Generate substitution suggestions
            substitutions = []
            
            # Category-based substitutions
            if state["ingredient"].category:
                category_products = self.vector_db.search_similar_products(
                    query=state["ingredient"].category,
                    limit=10,
                    min_confidence=0.3,
                    in_stock_only=True
                )
                
                for product, score in category_products:
                    if not any(match["product"].name == product.name for match in state["final_matches"]):
                        substitutions.append({
                            "product": product,
                            "type": "category_alternative",
                            "reason": f"Alternative {state['ingredient'].category} product",
                            "confidence": score * 0.7
                        })
            
            # Alternative name substitutions
            for alternative in state["ingredient"].alternatives:
                alt_products = self.vector_db.search_similar_products(
                    query=alternative,
                    limit=5,
                    min_confidence=0.4,
                    in_stock_only=True
                )
                
                for product, score in alt_products:
                    if not any(match["product"].name == product.name for match in state["final_matches"]):
                        substitutions.append({
                            "product": product,
                            "type": "name_alternative",
                            "reason": f"Match for alternative name: {alternative}",
                            "confidence": score * 0.8
                        })
            
            # Sort and limit substitutions
            substitutions = sorted(substitutions, key=lambda x: x["confidence"], reverse=True)
            state["substitution_suggestions"] = substitutions[:3]
            
            self.log_info(f"Found {len(state['substitution_suggestions'])} substitution suggestions")
            
        except Exception as e:
            self.log_error(f"Substitution analysis failed: {e}")
            state["substitution_suggestions"] = []
        
        state["matching_metadata"]["stages_completed"].append("substitution_analysis")
        return state
    
    async def _validate_categories(self, state: MatchingState) -> MatchingState:
        """Validate product categories against ingredient expectations."""
        self.log_info("Validating categories")
        
        if state["progress_callback"]:
            state["progress_callback"]("📂 Validating categories...")
        
        try:
            # Analyze category consistency
            ingredient_category = state["ingredient"].category
            
            category_stats = {}
            for match in state["final_matches"]:
                product_category = match["product"].category or "Unknown"
                category_stats[product_category] = category_stats.get(product_category, 0) + 1
            
            # Flag category mismatches
            for match in state["final_matches"]:
                product_category = match["product"].category
                
                if ingredient_category and product_category:
                    if ingredient_category.lower() not in product_category.lower():
                        match["category_warning"] = f"Expected {ingredient_category}, found {product_category}"
                        match["confidence"] *= 0.9  # Slight confidence reduction
            
            state["category_analysis"] = {
                "ingredient_category": ingredient_category,
                "product_categories": category_stats,
                "category_consistency": len(category_stats) <= 2  # Good if products are in 1-2 categories
            }
            
        except Exception as e:
            self.log_error(f"Category validation failed: {e}")
            state["category_analysis"] = {}
        
        state["matching_metadata"]["stages_completed"].append("category_validation")
        return state
    
    async def _rank_final_matches(self, state: MatchingState) -> MatchingState:
        """Final ranking of matches with all factors considered."""
        self.log_info("Ranking final matches")
        
        if state["progress_callback"]:
            state["progress_callback"]("🏆 Finalizing match rankings...")
        
        # Apply final ranking adjustments
        for match in state["final_matches"]:
            product = match["product"]
            
            # Boost for exact brand matches
            if (hasattr(state["ingredient"], "brand") and 
                state["ingredient"].brand and 
                product.brand and 
                state["ingredient"].brand.lower() == product.brand.lower()):
                match["confidence"] *= 1.1
            
            # Boost for sale items (value consideration)
            if product.on_sale:
                match["confidence"] *= 1.05
                match["sale_advantage"] = True
            
            # Boost for higher data collection confidence
            if product.collection_method == DataCollectionMethod.HUMAN_BROWSER:
                match["confidence"] *= 1.02
            
            # Cap confidence at 1.0
            match["confidence"] = min(match["confidence"], 1.0)
        
        # Final sort
        state["final_matches"] = sorted(
            state["final_matches"], 
            key=lambda x: x["confidence"], 
            reverse=True
        )
        
        state["matching_metadata"]["stages_completed"].append("final_ranking")
        return state
    
    async def _finalize_matching(self, state: MatchingState) -> MatchingState:
        """Finalize the matching session."""
        self.log_info("Finalizing matching")
        
        state["matching_metadata"]["end_time"] = datetime.now().isoformat()
        state["matching_metadata"]["total_duration"] = (
            datetime.fromisoformat(state["matching_metadata"]["end_time"]) -
            datetime.fromisoformat(state["matching_metadata"]["start_time"])
        ).total_seconds()
        
        if state["progress_callback"]:
            state["progress_callback"](
                f"✅ Matching complete! Found {len(state['final_matches'])} matches "
                f"for '{state['ingredient'].name}'"
            )
        
        self.log_info(
            f"Matching completed: {len(state['final_matches'])} matches, "
            f"require_human_review: {state['require_human_review']}"
        )
        
        state["matching_metadata"]["stages_completed"].append("finalize")
        return state
    
    # Utility Methods
    
    def _update_matching_stats(self, state: MatchingState) -> None:
        """Update matching statistics."""
        self.matching_stats["total_matches"] += 1
        
        if state["final_matches"]:
            self.matching_stats["successful_matches"] += 1
            
            # Update average confidence
            total_confidence = sum(match["confidence"] for match in state["final_matches"])
            avg_confidence = total_confidence / len(state["final_matches"])
            
            current_avg = self.matching_stats["avg_confidence"]
            total_successful = self.matching_stats["successful_matches"]
            
            self.matching_stats["avg_confidence"] = (
                (current_avg * (total_successful - 1) + avg_confidence) / total_successful
            )
            
            # Update quality distribution
            for match in state["final_matches"]:
                quality = match["quality"].value
                self.matching_stats["quality_distribution"][quality] += 1
        else:
            self.matching_stats["failed_matches"] += 1
        
        # Update strategy performance
        strategy = state["strategy"].value
        self.matching_stats["strategy_performance"][strategy]["attempts"] += 1
        if state["final_matches"]:
            self.matching_stats["strategy_performance"][strategy]["successes"] += 1
    
    def _analyze_match_quality(self, matches: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze the quality distribution of matches."""
        quality_dist = {quality.value: 0 for quality in MatchingQuality}
        
        for match in matches:
            quality = match["quality"].value
            quality_dist[quality] += 1
        
        return quality_dist
    
    def get_matching_analytics(self) -> Dict[str, Any]:
        """Get analytics about matching performance."""
        return {
            "matching_stats": self.matching_stats,
            "recommendations": self._generate_matching_recommendations()
        }
    
    def _generate_matching_recommendations(self) -> List[str]:
        """Generate recommendations for improving matching performance."""
        recommendations = []
        
        stats = self.matching_stats
        
        if stats["total_matches"] > 0:
            success_rate = stats["successful_matches"] / stats["total_matches"]
            
            if success_rate < 0.7:
                recommendations.append("Consider lowering confidence thresholds or expanding product database")
            
            if stats["avg_confidence"] < 0.6:
                recommendations.append("Review ingredient normalization and vector embeddings quality")
            
            poor_quality = stats["quality_distribution"]["poor"] + stats["quality_distribution"]["rejected"]
            total_quality = sum(stats["quality_distribution"].values())
            
            if total_quality > 0 and poor_quality / total_quality > 0.3:
                recommendations.append("Improve product data quality and matching algorithms")
        
        return recommendations
    
    # Public API Methods
    
    async def match_ingredient(
        self,
        ingredient: Ingredient,
        strategy: str = "adaptive",
        confidence_threshold: float = 0.5,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Public API for matching a single ingredient."""
        return await self.execute({
            "ingredient": ingredient,
            "strategy": strategy,
            "confidence_threshold": confidence_threshold,
            "max_results": max_results
        })
    
    async def match_ingredients_batch(
        self,
        ingredients: List[Ingredient],
        strategy: str = "adaptive",
        confidence_threshold: float = 0.5,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Match multiple ingredients in batch."""
        results = []
        
        for ingredient in ingredients:
            result = await self.match_ingredient(
                ingredient=ingredient,
                strategy=strategy,
                confidence_threshold=confidence_threshold,
                max_results=max_results
            )
            results.append(result)
        
        return results
    
    async def suggest_substitutions(
        self,
        ingredient_name: str,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Get substitution suggestions for an ingredient."""
        # Create temporary ingredient for analysis
        temp_ingredient = Ingredient(
            name=ingredient_name,
            quantity=1.0,
            unit="pieces"
        )
        
        result = await self.match_ingredient(
            ingredient=temp_ingredient,
            confidence_threshold=0.3,
            max_results=1
        )
        
        return result.get("substitution_suggestions", [])[:max_suggestions]