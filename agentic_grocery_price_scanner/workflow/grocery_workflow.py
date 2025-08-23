"""
Master LangGraph workflow that coordinates ScraperAgent, MatcherAgent, and OptimizerAgent.
This workflow provides end-to-end grocery shopping optimization with intelligent orchestration.
"""

import asyncio
import json
import logging
import time
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable, TypedDict, Union
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from ..data_models import Recipe, Ingredient, Product
from ..data_models.base import UnitType
from ..agents.intelligent_scraper_agent import IntelligentScraperAgent
from ..agents.matcher_agent import MatcherAgent
from ..agents.optimizer_agent import OptimizerAgent, OptimizationCriteria
from .state_adapters import StateAdapter

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages of the grocery workflow."""
    
    INITIALIZE = "initialize"
    EXTRACT_INGREDIENTS = "extract_ingredients"
    PARALLEL_SCRAPING = "parallel_scraping"
    AGGREGATE_PRODUCTS = "aggregate_products"
    PARALLEL_MATCHING = "parallel_matching"
    AGGREGATE_MATCHES = "aggregate_matches"
    OPTIMIZE_SHOPPING = "optimize_shopping"
    FINALIZE_RESULTS = "finalize_results"
    ERROR_RECOVERY = "error_recovery"


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


@dataclass
class WorkflowExecutionMetrics:
    """Comprehensive execution metrics for workflow monitoring."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    
    # Stage timing
    stage_timings: Dict[str, float] = field(default_factory=dict)
    current_stage: Optional[WorkflowStage] = None
    
    # Agent performance
    scraping_time: float = 0.0
    matching_time: float = 0.0
    optimization_time: float = 0.0
    
    # Data metrics
    total_ingredients: int = 0
    total_products_collected: int = 0
    total_matches_found: int = 0
    avg_matching_confidence: float = 0.0
    
    # Success rates
    scraping_success_rate: float = 0.0
    matching_success_rate: float = 0.0
    optimization_success_rate: float = 0.0
    
    # Memory usage (MB)
    peak_memory_mb: float = 0.0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]


class GroceryWorkflowState(TypedDict):
    """Unified state structure for the master grocery workflow."""
    
    # Input data
    recipes: List[Recipe]
    raw_ingredients: Optional[List[str]]  # Alternative to recipes
    
    # Configuration
    execution_id: str
    target_stores: List[str]
    max_budget: Optional[Decimal]
    max_stores: int
    preferred_stores: List[str]
    avoid_stores: List[str]
    quality_threshold: float
    confidence_threshold: float
    
    # Execution control
    scraping_strategy: str
    matching_strategy: str
    optimization_strategy: str
    enable_parallel_scraping: bool
    enable_parallel_matching: bool
    max_concurrent_agents: int
    agent_timeout_seconds: int
    
    # Workflow state
    workflow_status: WorkflowStatus
    current_stage: Optional[WorkflowStage]
    extracted_ingredients: List[Ingredient]
    
    # Data collections
    collected_products: Dict[str, Dict[str, Any]]  # ingredient_name -> collection_data
    matched_products: Dict[str, List[Dict[str, Any]]]  # ingredient_name -> matches
    optimization_results: Optional[Dict[str, Any]]
    
    # Progress tracking
    workflow_progress: Dict[str, int]  # stage -> count
    ingredients_scraped: int
    ingredients_matched: int
    
    # Performance metrics
    execution_metrics: WorkflowExecutionMetrics
    performance_metrics: Dict[str, Any]
    
    # Error handling
    agent_errors: Dict[str, List[Dict[str, Any]]]  # agent_name -> errors
    failed_ingredients: List[str]
    recovery_attempts: Dict[str, int]
    
    # Callbacks (excluded from serialization)
    # Note: These are stored separately to avoid serialization issues
    
    # Caching
    enable_caching: bool
    cache_key: Optional[str]
    cached_results: Dict[str, Any]


class GroceryWorkflow:
    """Master LangGraph workflow coordinating all grocery agents."""
    
    def __init__(
        self,
        scraper_agent: Optional[IntelligentScraperAgent] = None,
        matcher_agent: Optional[MatcherAgent] = None,
        optimizer_agent: Optional[OptimizerAgent] = None,
        enable_checkpointing: bool = True
    ):
        """Initialize the master grocery workflow."""
        self.name = "grocery_workflow"
        
        # Initialize agents (lazy loading for performance)
        self._scraper_agent = scraper_agent
        self._matcher_agent = matcher_agent  
        self._optimizer_agent = optimizer_agent
        
        # Workflow configuration
        self.enable_checkpointing = enable_checkpointing
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        
        # Performance monitoring
        self.active_executions: Dict[str, GroceryWorkflowState] = {}
        self.execution_history: List[WorkflowExecutionMetrics] = []
        
        # Callback storage (separate from state to avoid serialization issues)
        self.execution_callbacks: Dict[str, Dict[str, Optional[Callable]]] = {}
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info(f"Initialized GroceryWorkflow with checkpointing: {enable_checkpointing}")
    
    @property
    def scraper_agent(self) -> IntelligentScraperAgent:
        """Lazy-loaded scraper agent."""
        if self._scraper_agent is None:
            self._scraper_agent = IntelligentScraperAgent()
        return self._scraper_agent
    
    @property
    def matcher_agent(self) -> MatcherAgent:
        """Lazy-loaded matcher agent."""
        if self._matcher_agent is None:
            self._matcher_agent = MatcherAgent()
        return self._matcher_agent
    
    @property
    def optimizer_agent(self) -> OptimizerAgent:
        """Lazy-loaded optimizer agent."""
        if self._optimizer_agent is None:
            self._optimizer_agent = OptimizerAgent(matcher_agent=self.matcher_agent)
        return self._optimizer_agent
    
    def _build_workflow(self) -> StateGraph:
        """Build the master LangGraph workflow."""
        workflow = StateGraph(GroceryWorkflowState)
        
        # Add all workflow nodes
        workflow.add_node("initialize_workflow", self._initialize_workflow)
        workflow.add_node("extract_ingredients", self._extract_ingredients)
        workflow.add_node("validate_ingredients", self._validate_ingredients)
        workflow.add_node("parallel_scraping", self._parallel_scraping)
        workflow.add_node("aggregate_products", self._aggregate_products)
        workflow.add_node("parallel_matching", self._parallel_matching)
        workflow.add_node("aggregate_matches", self._aggregate_matches)
        workflow.add_node("optimize_shopping", self._optimize_shopping)
        workflow.add_node("finalize_results", self._finalize_results)
        workflow.add_node("handle_errors", self._handle_errors)
        workflow.add_node("cleanup_resources", self._cleanup_resources)
        
        # Define workflow edges with conditional routing
        workflow.add_edge(START, "initialize_workflow")
        workflow.add_edge("initialize_workflow", "extract_ingredients")
        
        workflow.add_conditional_edges(
            "extract_ingredients",
            self._route_after_ingredient_extraction,
            {
                "validate": "validate_ingredients",
                "scraping": "parallel_scraping", 
                "error": "handle_errors"
            }
        )
        
        workflow.add_edge("validate_ingredients", "parallel_scraping")
        
        workflow.add_conditional_edges(
            "parallel_scraping",
            self._route_after_scraping,
            {
                "aggregate": "aggregate_products",
                "retry": "parallel_scraping",
                "error": "handle_errors"
            }
        )
        
        workflow.add_edge("aggregate_products", "parallel_matching")
        
        workflow.add_conditional_edges(
            "parallel_matching", 
            self._route_after_matching,
            {
                "aggregate": "aggregate_matches",
                "retry": "parallel_matching",
                "error": "handle_errors"
            }
        )
        
        workflow.add_edge("aggregate_matches", "optimize_shopping")
        
        workflow.add_conditional_edges(
            "optimize_shopping",
            self._route_after_optimization,
            {
                "finalize": "finalize_results",
                "retry": "optimize_shopping",
                "error": "handle_errors"
            }
        )
        
        workflow.add_edge("finalize_results", "cleanup_resources")
        workflow.add_edge("cleanup_resources", END)
        
        # Error recovery edges
        workflow.add_conditional_edges(
            "handle_errors",
            self._route_error_recovery,
            {
                "retry_scraping": "parallel_scraping",
                "retry_matching": "parallel_matching", 
                "retry_optimization": "optimize_shopping",
                "finalize_partial": "finalize_results",
                "abort": END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def execute(
        self,
        recipes: Optional[List[Recipe]] = None,
        ingredients: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete grocery workflow.
        
        Args:
            recipes: List of recipes to process
            ingredients: Alternative raw ingredient list
            config: Workflow configuration options
            progress_callback: Callback for progress updates
        
        Returns:
            Complete workflow results with optimization recommendations
        """
        if not recipes and not ingredients:
            raise ValueError("Either recipes or ingredients must be provided")
        
        # Initialize state
        state = self._create_initial_state(recipes, ingredients, config, progress_callback)
        
        try:
            # Store active execution and callbacks separately
            execution_id = state["execution_id"]
            self.active_executions[execution_id] = state
            
            # Store callbacks separately to avoid serialization issues
            self.execution_callbacks[execution_id] = {
                "progress_callback": progress_callback,
                "stage_callback": None,  # Not used yet
                "error_callback": None   # Not used yet
            }
            
            # Execute workflow with timeout
            timeout = config.get("workflow_timeout", 300) if config else 300  # 5 minute default
            
            result = await asyncio.wait_for(
                self.workflow.ainvoke(
                    state,
                    config={"configurable": {"thread_id": state["execution_id"]}}
                ),
                timeout=timeout
            )
            
            # Update execution metrics
            result["execution_metrics"].status = WorkflowStatus.COMPLETED
            result["execution_metrics"].end_time = datetime.now()
            
            # Store execution history
            self.execution_history.append(result["execution_metrics"])
            
            logger.info(f"Workflow {state['execution_id']} completed successfully")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Workflow {state['execution_id']} timed out after {timeout} seconds")
            state["execution_metrics"].status = WorkflowStatus.FAILED
            state["execution_metrics"].errors.append({
                "type": "timeout",
                "message": f"Workflow exceeded {timeout} second timeout",
                "timestamp": datetime.now().isoformat()
            })
            raise
            
        except GraphRecursionError as e:
            logger.error(f"Workflow {state['execution_id']} exceeded recursion limit: {e}")
            state["execution_metrics"].status = WorkflowStatus.FAILED
            raise
            
        except Exception as e:
            logger.error(f"Workflow {state['execution_id']} failed: {e}")
            state["execution_metrics"].status = WorkflowStatus.FAILED
            state["execution_metrics"].errors.append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
            
        finally:
            # Cleanup
            execution_id = state["execution_id"]
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            if execution_id in self.execution_callbacks:
                del self.execution_callbacks[execution_id]
    
    def _create_initial_state(
        self,
        recipes: Optional[List[Recipe]],
        ingredients: Optional[List[str]],
        config: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> GroceryWorkflowState:
        """Create initial workflow state."""
        execution_id = str(uuid.uuid4())
        
        # Set defaults from config
        config = config or {}
        
        return GroceryWorkflowState(
            # Input data
            recipes=recipes or [],
            raw_ingredients=ingredients,
            
            # Configuration  
            execution_id=execution_id,
            target_stores=config.get("target_stores", ["metro_ca", "walmart_ca", "freshco_com"]),
            max_budget=Decimal(str(config["max_budget"])) if config.get("max_budget") else None,
            max_stores=config.get("max_stores", 3),
            preferred_stores=config.get("preferred_stores", []),
            avoid_stores=config.get("avoid_stores", []),
            quality_threshold=config.get("quality_threshold", 0.7),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            
            # Execution control
            scraping_strategy=config.get("scraping_strategy", "adaptive"),
            matching_strategy=config.get("matching_strategy", "adaptive"),
            optimization_strategy=config.get("optimization_strategy", "adaptive"),
            enable_parallel_scraping=config.get("enable_parallel_scraping", True),
            enable_parallel_matching=config.get("enable_parallel_matching", True),
            max_concurrent_agents=config.get("max_concurrent_agents", 5),
            agent_timeout_seconds=config.get("agent_timeout_seconds", 120),
            
            # Workflow state
            workflow_status=WorkflowStatus.PENDING,
            current_stage=None,
            extracted_ingredients=[],
            
            # Data collections
            collected_products={},
            matched_products={},
            optimization_results=None,
            
            # Progress tracking
            workflow_progress={
                "ingredients_extracted": 0,
                "scraping": 0,
                "matching": 0,
                "optimization": 0
            },
            ingredients_scraped=0,
            ingredients_matched=0,
            
            # Performance metrics
            execution_metrics=WorkflowExecutionMetrics(
                execution_id=execution_id,
                status=WorkflowStatus.PENDING
            ),
            performance_metrics={},
            
            # Error handling
            agent_errors={},
            failed_ingredients=[],
            recovery_attempts={},
            
            # Callbacks are stored separately to avoid serialization issues
            
            # Caching
            enable_caching=config.get("enable_caching", True),
            cache_key=None,
            cached_results={}
        )
    
    # Workflow node implementations
    
    async def _initialize_workflow(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Initialize workflow execution."""
        logger.info(f"Initializing workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.INITIALIZE
        state["workflow_status"] = WorkflowStatus.RUNNING
        state["execution_metrics"].current_stage = WorkflowStage.INITIALIZE
        
        # Generate cache key if caching enabled
        if state["enable_caching"]:
            cache_components = []
            if state["recipes"]:
                cache_components.extend([r.name for r in state["recipes"]])
            if state["raw_ingredients"]:
                cache_components.extend(state["raw_ingredients"])
            
            state["cache_key"] = f"grocery_workflow_{hash(tuple(sorted(cache_components)))}"
        
        # Initialize performance metrics
        state["performance_metrics"] = {
            "scraping": {
                "total_products_collected": 0,
                "avg_execution_time": 0.0,
                "success_rate_by_store": {},
                "collection_method_distribution": {}
            },
            "matching": {
                "total_matches": 0,
                "avg_confidence": 0.0,
                "quality_distribution": {},
                "substitution_rate": 0.0
            },
            "optimization": {
                "recommended_strategy": [],
                "total_savings": 0.0,
                "savings_percentage": 0.0,
                "store_distribution": {}
            }
        }
        
        # Notify progress (use separate callback storage)
        execution_id = state["execution_id"]
        callbacks = self.execution_callbacks.get(execution_id, {})
        progress_callback = callbacks.get("progress_callback")
        if progress_callback:
            progress_callback({
                "stage": "initialize", 
                "progress": 0,
                "message": "Initializing workflow..."
            })
        
        return state
    
    async def _extract_ingredients(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Extract ingredients from recipes or process raw ingredient list."""
        logger.info(f"Extracting ingredients for workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.EXTRACT_INGREDIENTS
        stage_start = time.time()
        
        extracted_ingredients = []
        
        # Process recipes
        if state["recipes"]:
            for recipe in state["recipes"]:
                extracted_ingredients.extend(recipe.ingredients)
        
        # Process raw ingredients 
        if state["raw_ingredients"]:
            for ingredient_name in state["raw_ingredients"]:
                ingredient = Ingredient(
                    name=ingredient_name.strip(),
                    quantity=1.0,  # Default quantity
                    unit=UnitType.PIECES,   # Default unit (valid enum value)
                    category=None  # Will be inferred later
                )
                extracted_ingredients.append(ingredient)
        
        # Remove duplicates (same name and unit)
        unique_ingredients = []
        seen = set()
        
        for ingredient in extracted_ingredients:
            key = (ingredient.name.lower(), ingredient.unit)
            if key not in seen:
                unique_ingredients.append(ingredient)
                seen.add(key)
            else:
                # Combine quantities for duplicate ingredients
                for existing in unique_ingredients:
                    if (existing.name.lower(), existing.unit) == key:
                        existing.quantity += ingredient.quantity
                        break
        
        state["extracted_ingredients"] = unique_ingredients
        state["workflow_progress"]["ingredients_extracted"] = len(unique_ingredients)
        state["execution_metrics"].total_ingredients = len(unique_ingredients)
        
        # Update timing
        stage_time = time.time() - stage_start
        state["execution_metrics"].stage_timings["extract_ingredients"] = stage_time
        
        logger.info(f"Extracted {len(unique_ingredients)} unique ingredients")
        
        # Notify progress (use separate callback storage)
        execution_id = state["execution_id"]
        callbacks = self.execution_callbacks.get(execution_id, {})
        progress_callback = callbacks.get("progress_callback")
        if progress_callback:
            progress_callback({
                "stage": "extract_ingredients",
                "progress": 100,  # This stage is complete
                "message": f"Extracted {len(unique_ingredients)} ingredients",
                "ingredients": [ing.name for ing in unique_ingredients]
            })
        
        return state
    
    async def _validate_ingredients(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Validate extracted ingredients and categorize them."""
        logger.info(f"Validating ingredients for workflow {state['execution_id']}")
        
        validated_ingredients = []
        
        for ingredient in state["extracted_ingredients"]:
            # Basic validation
            if not ingredient.name or len(ingredient.name.strip()) < 2:
                state["execution_metrics"].warnings.append({
                    "type": "invalid_ingredient",
                    "message": f"Skipping invalid ingredient: '{ingredient.name}'",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Normalize ingredient name
            ingredient.name = ingredient.name.strip().lower()
            
            # Infer category if not set
            if not ingredient.category:
                ingredient.category = self._infer_category(ingredient.name)
            
            validated_ingredients.append(ingredient)
        
        state["extracted_ingredients"] = validated_ingredients
        
        logger.info(f"Validated {len(validated_ingredients)} ingredients")
        return state
    
    def _infer_category(self, ingredient_name: str) -> str:
        """Infer ingredient category from name."""
        # Simple categorization - could be enhanced with ML
        name_lower = ingredient_name.lower()
        
        if any(word in name_lower for word in ["milk", "cheese", "yogurt", "butter", "cream"]):
            return "dairy"
        elif any(word in name_lower for word in ["chicken", "beef", "pork", "fish", "turkey", "meat"]):
            return "meat"  
        elif any(word in name_lower for word in ["apple", "banana", "orange", "berry", "fruit"]):
            return "produce"
        elif any(word in name_lower for word in ["bread", "rice", "pasta", "cereal", "flour"]):
            return "grains"
        elif any(word in name_lower for word in ["oil", "salt", "pepper", "spice", "sauce"]):
            return "condiments"
        else:
            return "other"
    
    async def _parallel_scraping(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Execute parallel scraping across ingredients and stores."""
        logger.info(f"Starting parallel scraping for workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.PARALLEL_SCRAPING
        stage_start = time.time()
        
        if not state["enable_parallel_scraping"]:
            return await self._sequential_scraping(state)
        
        # Create scraping tasks
        semaphore = asyncio.Semaphore(state["max_concurrent_agents"])
        scraping_tasks = []
        
        for ingredient in state["extracted_ingredients"]:
            if ingredient.name not in state["failed_ingredients"]:
                task = self._scrape_single_ingredient(ingredient, state, semaphore)
                scraping_tasks.append(task)
        
        # Execute scraping with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*scraping_tasks, return_exceptions=True),
                timeout=state["agent_timeout_seconds"] * 2  # Allow extra time for parallel execution
            )
            
            # Process results
            for ingredient, result in zip(state["extracted_ingredients"], results):
                if isinstance(result, Exception):
                    logger.error(f"Scraping failed for {ingredient.name}: {result}")
                    state["failed_ingredients"].append(ingredient.name)
                    
                    if "scraper" not in state["agent_errors"]:
                        state["agent_errors"]["scraper"] = []
                    
                    state["agent_errors"]["scraper"].append({
                        "ingredient": ingredient.name,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # Merge successful result
                    state = StateAdapter.merge_scraping_results(state, ingredient, result)
                    state["ingredients_scraped"] += 1
            
        except asyncio.TimeoutError:
            logger.error("Parallel scraping timed out")
            state["execution_metrics"].errors.append({
                "type": "scraping_timeout", 
                "message": "Parallel scraping exceeded timeout",
                "timestamp": datetime.now().isoformat()
            })
        
        # Update timing
        stage_time = time.time() - stage_start
        state["execution_metrics"].stage_timings["parallel_scraping"] = stage_time
        state["execution_metrics"].scraping_time = stage_time
        
        logger.info(f"Completed scraping for {state['ingredients_scraped']} ingredients")
        return state
    
    async def _scrape_single_ingredient(
        self,
        ingredient: Ingredient,
        state: GroceryWorkflowState,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Scrape a single ingredient with concurrency control."""
        async with semaphore:
            try:
                # Create progress callback for this ingredient
                def ingredient_progress(message: str):
                    if state["progress_callback"]:
                        state["progress_callback"]({
                            "stage": "scraping",
                            "ingredient": ingredient.name,
                            "message": message
                        })
                
                # Convert to scraping state
                scraping_state = StateAdapter.to_scraping_state(
                    state, ingredient, ingredient_progress
                )
                
                # Execute scraping
                result = await self.scraper_agent.execute({
                    "query": ingredient.name,
                    "stores": state["target_stores"],
                    "strategy": state["scraping_strategy"],
                    "limit": 20
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to scrape {ingredient.name}: {e}")
                raise
    
    async def _sequential_scraping(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Execute sequential scraping as fallback."""
        logger.info("Executing sequential scraping")
        
        for ingredient in state["extracted_ingredients"]:
            if ingredient.name in state["failed_ingredients"]:
                continue
            
            try:
                result = await self._scrape_single_ingredient(ingredient, state, asyncio.Semaphore(1))
                state = StateAdapter.merge_scraping_results(state, ingredient, result)
                state["ingredients_scraped"] += 1
                
            except Exception as e:
                state["failed_ingredients"].append(ingredient.name)
                logger.error(f"Sequential scraping failed for {ingredient.name}: {e}")
        
        return state
    
    async def _aggregate_products(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Aggregate all collected products and compute statistics."""
        logger.info(f"Aggregating products for workflow {state['execution_id']}")
        
        total_products = 0
        store_distribution = {}
        method_distribution = {}
        
        for ingredient_name, collection_data in state["collected_products"].items():
            products = collection_data.get("products", [])
            total_products += len(products)
            
            for product in products:
                # Store distribution
                store_id = getattr(product, 'store_id', 'unknown')
                store_distribution[store_id] = store_distribution.get(store_id, 0) + 1
                
                # Collection method distribution
                method = getattr(product, 'collection_method', 'unknown')
                method_distribution[method] = method_distribution.get(method, 0) + 1
        
        # Update performance metrics
        state["performance_metrics"]["scraping"]["total_products_collected"] = total_products
        state["performance_metrics"]["scraping"]["store_distribution"] = store_distribution
        state["performance_metrics"]["scraping"]["collection_method_distribution"] = method_distribution
        
        # Calculate success rate
        total_ingredients = len(state["extracted_ingredients"])
        successful_ingredients = len([ing for ing in state["extracted_ingredients"] 
                                    if ing.name not in state["failed_ingredients"]])
        
        state["execution_metrics"].scraping_success_rate = successful_ingredients / total_ingredients if total_ingredients > 0 else 0
        
        logger.info(f"Aggregated {total_products} products from {successful_ingredients}/{total_ingredients} ingredients")
        
        # Notify progress
        if state["progress_callback"]:
            state["progress_callback"]({
                "stage": "aggregate_products",
                "progress": 100,
                "total_products": total_products,
                "successful_ingredients": successful_ingredients,
                "total_ingredients": total_ingredients
            })
        
        return state
    
    async def _parallel_matching(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Execute parallel matching of ingredients to products."""
        logger.info(f"Starting parallel matching for workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.PARALLEL_MATCHING
        stage_start = time.time()
        
        if not state["enable_parallel_matching"]:
            return await self._sequential_matching(state)
        
        # Create matching tasks
        semaphore = asyncio.Semaphore(state["max_concurrent_agents"])
        matching_tasks = []
        
        for ingredient in state["extracted_ingredients"]:
            if ingredient.name not in state["failed_ingredients"] and ingredient.name in state["collected_products"]:
                products = [Product.parse_obj(p) if isinstance(p, dict) else p 
                           for p in state["collected_products"][ingredient.name].get("products", [])]
                
                task = self._match_single_ingredient(ingredient, products, state, semaphore)
                matching_tasks.append(task)
        
        # Execute matching with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*matching_tasks, return_exceptions=True),
                timeout=state["agent_timeout_seconds"]
            )
            
            # Process results
            matched_count = 0
            total_confidence = 0.0
            
            for ingredient, result in zip(state["extracted_ingredients"], results):
                if isinstance(result, Exception):
                    logger.error(f"Matching failed for {ingredient.name}: {result}")
                    continue
                
                # Merge successful result
                state = StateAdapter.merge_matching_results(state, ingredient, result)
                
                matches = result.get("final_matches", [])
                if matches:
                    matched_count += 1
                    avg_conf = sum(m.get("confidence", 0) for m in matches) / len(matches)
                    total_confidence += avg_conf
            
            # Calculate average confidence
            state["execution_metrics"].avg_matching_confidence = (
                total_confidence / matched_count if matched_count > 0 else 0
            )
            state["execution_metrics"].matching_success_rate = (
                matched_count / len(state["extracted_ingredients"]) if state["extracted_ingredients"] else 0
            )
            
        except asyncio.TimeoutError:
            logger.error("Parallel matching timed out")
            state["execution_metrics"].errors.append({
                "type": "matching_timeout",
                "message": "Parallel matching exceeded timeout", 
                "timestamp": datetime.now().isoformat()
            })
        
        # Update timing
        stage_time = time.time() - stage_start
        state["execution_metrics"].stage_timings["parallel_matching"] = stage_time
        state["execution_metrics"].matching_time = stage_time
        
        logger.info(f"Completed matching for {state['workflow_progress']['matching']} ingredients")
        return state
    
    async def _match_single_ingredient(
        self,
        ingredient: Ingredient,
        products: List[Product], 
        state: GroceryWorkflowState,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Match a single ingredient with concurrency control."""
        async with semaphore:
            try:
                # Create progress callback for this ingredient
                def ingredient_progress(message: str):
                    if state["progress_callback"]:
                        state["progress_callback"]({
                            "stage": "matching",
                            "ingredient": ingredient.name,
                            "message": message
                        })
                
                # Convert to matching state
                matching_state = StateAdapter.to_matching_state(
                    state, ingredient, products, ingredient_progress
                )
                
                # Execute matching
                result = await self.matcher_agent.execute({
                    "ingredient": ingredient,
                    "strategy": state["matching_strategy"],
                    "confidence_threshold": state["confidence_threshold"],
                    "max_results": 5
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to match {ingredient.name}: {e}")
                raise
    
    async def _sequential_matching(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Execute sequential matching as fallback."""
        logger.info("Executing sequential matching")
        
        for ingredient in state["extracted_ingredients"]:
            if (ingredient.name not in state["failed_ingredients"] and 
                ingredient.name in state["collected_products"]):
                
                try:
                    products = [Product.parse_obj(p) if isinstance(p, dict) else p 
                               for p in state["collected_products"][ingredient.name].get("products", [])]
                    
                    result = await self._match_single_ingredient(ingredient, products, state, asyncio.Semaphore(1))
                    state = StateAdapter.merge_matching_results(state, ingredient, result)
                    
                except Exception as e:
                    logger.error(f"Sequential matching failed for {ingredient.name}: {e}")
        
        return state
    
    async def _aggregate_matches(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Aggregate matching results and compute statistics."""
        logger.info(f"Aggregating matches for workflow {state['execution_id']}")
        
        total_matches = 0
        confidence_scores = []
        quality_distribution = {}
        
        for ingredient_name, matches in state["matched_products"].items():
            total_matches += len(matches)
            
            for match in matches:
                confidence = match.get("confidence", 0)
                confidence_scores.append(confidence)
                
                # Quality distribution
                if confidence >= 0.9:
                    quality = "excellent"
                elif confidence >= 0.7:
                    quality = "good" 
                elif confidence >= 0.5:
                    quality = "fair"
                else:
                    quality = "poor"
                
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        # Update performance metrics
        state["performance_metrics"]["matching"]["total_matches"] = total_matches
        state["performance_metrics"]["matching"]["quality_distribution"] = quality_distribution
        
        if confidence_scores:
            state["performance_metrics"]["matching"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        state["execution_metrics"].total_matches_found = total_matches
        
        logger.info(f"Aggregated {total_matches} matches with avg confidence: {state['performance_metrics']['matching']['avg_confidence']:.3f}")
        
        # Notify progress
        if state["progress_callback"]:
            state["progress_callback"]({
                "stage": "aggregate_matches",
                "progress": 100,
                "total_matches": total_matches,
                "avg_confidence": state["performance_metrics"]["matching"]["avg_confidence"]
            })
        
        return state
    
    async def _optimize_shopping(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Execute shopping optimization across stores."""
        logger.info(f"Starting optimization for workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.OPTIMIZE_SHOPPING
        stage_start = time.time()
        
        try:
            # Create progress callback
            def optimization_progress(message: str):
                if state["progress_callback"]:
                    state["progress_callback"]({
                        "stage": "optimization",
                        "message": message
                    })
            
            # Convert to optimization state
            optimization_state = StateAdapter.to_optimization_state(
                state, state["extracted_ingredients"], state["matched_products"], optimization_progress
            )
            
            # Execute optimization
            result = await self.optimizer_agent.execute({
                "ingredients": state["extracted_ingredients"],
                "matched_products": state["matched_products"], 
                "strategy": state["optimization_strategy"],
                "max_budget": state["max_budget"],
                "max_stores": state["max_stores"],
                "preferred_stores": state["preferred_stores"]
            })
            
            # Merge results
            state = StateAdapter.merge_optimization_results(state, result)
            
            state["execution_metrics"].optimization_success_rate = 1.0
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            state["execution_metrics"].optimization_success_rate = 0.0
            state["execution_metrics"].errors.append({
                "type": "optimization_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        # Update timing
        stage_time = time.time() - stage_start
        state["execution_metrics"].stage_timings["optimize_shopping"] = stage_time
        state["execution_metrics"].optimization_time = stage_time
        
        return state
    
    async def _finalize_results(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Finalize workflow results and prepare output."""
        logger.info(f"Finalizing results for workflow {state['execution_id']}")
        
        state["current_stage"] = WorkflowStage.FINALIZE_RESULTS
        state["workflow_status"] = WorkflowStatus.COMPLETED
        
        # Update final metrics
        state["execution_metrics"].status = WorkflowStatus.COMPLETED
        state["execution_metrics"].end_time = datetime.now()
        state["execution_metrics"].total_products_collected = state["performance_metrics"]["scraping"]["total_products_collected"]
        state["execution_metrics"].total_matches_found = state["performance_metrics"]["matching"]["total_matches"]
        
        # Generate summary
        summary = {
            "execution_id": state["execution_id"],
            "total_execution_time": state["execution_metrics"].total_execution_time,
            "ingredients_processed": len(state["extracted_ingredients"]),
            "products_collected": state["execution_metrics"].total_products_collected,
            "matches_found": state["execution_metrics"].total_matches_found,
            "optimization_completed": state["optimization_results"] is not None,
            "success_rates": {
                "scraping": state["execution_metrics"].scraping_success_rate,
                "matching": state["execution_metrics"].matching_success_rate,
                "optimization": state["execution_metrics"].optimization_success_rate
            }
        }
        
        state["workflow_summary"] = summary
        
        # Final progress callback
        if state["progress_callback"]:
            state["progress_callback"]({
                "stage": "completed",
                "progress": 100,
                "summary": summary
            })
        
        logger.info(f"Workflow {state['execution_id']} finalized successfully")
        return state
    
    async def _cleanup_resources(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Cleanup workflow resources."""
        logger.info(f"Cleaning up resources for workflow {state['execution_id']}")
        
        # Clear large data structures to free memory
        if not state["enable_caching"]:
            state["collected_products"] = {}
            state["cached_results"] = {}
        
        # Store execution in history
        self.execution_history.append(state["execution_metrics"])
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return state
    
    async def _handle_errors(self, state: GroceryWorkflowState) -> GroceryWorkflowState:
        """Handle workflow errors with recovery strategies."""
        logger.info(f"Handling errors for workflow {state['execution_id']}")
        
        state["workflow_status"] = WorkflowStatus.RECOVERING
        
        # Implement error recovery logic based on current stage and error type
        current_stage = state["current_stage"]
        
        if current_stage == WorkflowStage.PARALLEL_SCRAPING:
            # Retry failed ingredients with sequential scraping
            failed_count = len(state["failed_ingredients"])
            if failed_count < len(state["extracted_ingredients"]) * 0.5:  # Less than 50% failed
                logger.info(f"Retrying {failed_count} failed ingredients with sequential scraping")
                # Clear failed ingredients to retry
                state["failed_ingredients"] = []
                return state
        
        elif current_stage == WorkflowStage.PARALLEL_MATCHING:
            # Continue with available matches if we have some successful results
            successful_matches = len([ing for ing in state["extracted_ingredients"] 
                                    if ing.name in state["matched_products"]])
            if successful_matches > 0:
                logger.info(f"Proceeding with {successful_matches} successful matches")
                return state
        
        # If recovery is not possible, finalize with partial results
        logger.warning("Recovery not possible, finalizing with partial results")
        state["workflow_status"] = WorkflowStatus.FAILED
        return state
    
    # Routing functions for conditional edges
    
    def _route_after_ingredient_extraction(self, state: GroceryWorkflowState) -> str:
        """Route after ingredient extraction."""
        if not state["extracted_ingredients"]:
            return "error"
        elif len(state["extracted_ingredients"]) > 50:  # Large batches need validation
            return "validate"
        else:
            return "scraping"
    
    def _route_after_scraping(self, state: GroceryWorkflowState) -> str:
        """Route after scraping completion."""
        failed_count = len(state["failed_ingredients"])
        total_count = len(state["extracted_ingredients"])
        
        # If more than 75% failed, try error recovery
        if failed_count > total_count * 0.75:
            return "error"
        
        # If some ingredients failed but we have retry attempts left
        if failed_count > 0 and state["recovery_attempts"].get("scraping", 0) < 2:
            state["recovery_attempts"]["scraping"] = state["recovery_attempts"].get("scraping", 0) + 1
            return "retry"
        
        return "aggregate"
    
    def _route_after_matching(self, state: GroceryWorkflowState) -> str:
        """Route after matching completion."""
        successful_matches = len([ing for ing in state["extracted_ingredients"] 
                                if ing.name in state["matched_products"]])
        
        # If no matches found, try error recovery
        if successful_matches == 0:
            return "error"
        
        # If less than 50% matched and we have retry attempts left
        if (successful_matches < len(state["extracted_ingredients"]) * 0.5 and 
            state["recovery_attempts"].get("matching", 0) < 1):
            state["recovery_attempts"]["matching"] = state["recovery_attempts"].get("matching", 0) + 1
            return "retry"
        
        return "aggregate"
    
    def _route_after_optimization(self, state: GroceryWorkflowState) -> str:
        """Route after optimization completion."""
        if state["optimization_results"] is None:
            # Try once more if we have successful matches
            if (state["matched_products"] and 
                state["recovery_attempts"].get("optimization", 0) < 1):
                state["recovery_attempts"]["optimization"] = state["recovery_attempts"].get("optimization", 0) + 1
                return "retry"
            else:
                return "error"
        
        return "finalize"
    
    def _route_error_recovery(self, state: GroceryWorkflowState) -> str:
        """Route error recovery based on current stage."""
        current_stage = state["current_stage"]
        
        if current_stage == WorkflowStage.PARALLEL_SCRAPING:
            return "retry_scraping"
        elif current_stage == WorkflowStage.PARALLEL_MATCHING:
            return "retry_matching"
        elif current_stage == WorkflowStage.OPTIMIZE_SHOPPING:
            return "retry_optimization"
        elif len(state["matched_products"]) > 0:
            return "finalize_partial"  # Have some data to work with
        else:
            return "abort"
    
    # Utility methods
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running workflow execution."""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "status": state["workflow_status"],
                "current_stage": state["current_stage"],
                "progress": state["workflow_progress"],
                "execution_time": state["execution_metrics"].total_execution_time,
                "errors": len(state["execution_metrics"].errors)
            }
        
        # Check history
        for metrics in self.execution_history:
            if metrics.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "status": metrics.status,
                    "execution_time": metrics.total_execution_time,
                    "completed": metrics.is_complete
                }
        
        return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            state["workflow_status"] = WorkflowStatus.CANCELLED
            state["execution_metrics"].status = WorkflowStatus.CANCELLED
            state["execution_metrics"].end_time = datetime.now()
            
            logger.info(f"Cancelled workflow execution {execution_id}")
            return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        completed_executions = [m for m in self.execution_history if m.status == WorkflowStatus.COMPLETED]
        
        if not completed_executions:
            return {"message": "No completed executions available"}
        
        return {
            "total_executions": len(self.execution_history),
            "completed_executions": len(completed_executions),
            "avg_execution_time": sum(m.total_execution_time for m in completed_executions) / len(completed_executions),
            "avg_ingredients_processed": sum(m.total_ingredients for m in completed_executions) / len(completed_executions),
            "avg_products_collected": sum(m.total_products_collected for m in completed_executions) / len(completed_executions),
            "avg_success_rates": {
                "scraping": sum(m.scraping_success_rate for m in completed_executions) / len(completed_executions),
                "matching": sum(m.matching_success_rate for m in completed_executions) / len(completed_executions),
                "optimization": sum(m.optimization_success_rate for m in completed_executions) / len(completed_executions)
            }
        }