"""
Real-time workflow executor for the Streamlit dashboard.
This module bridges the dashboard with the actual GroceryWorkflow execution.
"""

import asyncio
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agentic_grocery_price_scanner.workflow import GroceryWorkflow, WorkflowStatus
    from agentic_grocery_price_scanner.data_models import Recipe, Ingredient
except ImportError:
    # Fallback for testing - define WorkflowStatus enum
    from enum import Enum
    
    class WorkflowStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        RECOVERING = "recovering"
    
    GroceryWorkflow = None
    Recipe = None
    Ingredient = None

logger = logging.getLogger(__name__)

class RealTimeExecutor:
    """Manages real-time workflow execution for the dashboard."""
    
    def __init__(self):
        self.workflow: Optional[GroceryWorkflow] = None
        self.current_execution: Optional[Dict[str, Any]] = None
        self.execution_thread: Optional[threading.Thread] = None
        self.progress_callback: Optional[Callable] = None
        self.status = WorkflowStatus.PENDING
        self.metrics = {}
        self.results = None
        self.logs = []
        
    def initialize_workflow(self):
        """Initialize the GroceryWorkflow."""
        if self.workflow is None:
            try:
                self.workflow = GroceryWorkflow()
                logger.info("GroceryWorkflow initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GroceryWorkflow: {e}")
                raise
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def execute_workflow_async(self, execution_params: Dict[str, Any]):
        """Execute workflow asynchronously."""
        if self.execution_thread and self.execution_thread.is_alive():
            logger.warning("Workflow already running")
            return
        
        self.execution_thread = threading.Thread(
            target=self._run_workflow,
            args=(execution_params,),
            daemon=True
        )
        self.execution_thread.start()
    
    def _run_workflow(self, execution_params: Dict[str, Any]):
        """Run the workflow in a separate thread."""
        try:
            self.status = WorkflowStatus.RUNNING
            self.results = None
            self.logs = []
            
            # Initialize workflow if needed
            self.initialize_workflow()
            
            # Convert parameters to proper format
            ingredients_list = self._convert_ingredients(
                execution_params.get('ingredients_list', [])
            )
            
            recipes_list = self._convert_recipes(
                execution_params.get('recipes_list', [])
            )
            
            # Prepare workflow input
            workflow_input = {
                'ingredients': ingredients_list,
                'recipes': recipes_list,
                'budget_limit': execution_params.get('budget', 100.0),
                'optimization_strategy': execution_params.get('optimization_strategy', 'balanced'),
                'target_stores': execution_params.get('selected_stores', ['metro_ca', 'walmart_ca']),
                'max_stores': execution_params.get('max_stores', 2),
                'user_preferences': {
                    'quality_threshold': 0.8,
                    'max_travel_time': 60
                }
            }
            
            # Execute workflow with progress tracking
            self._execute_with_progress_tracking(workflow_input)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.status = WorkflowStatus.FAILED
            self.logs.append({
                'timestamp': datetime.now(),
                'level': 'ERROR',
                'message': f"Execution failed: {str(e)}"
            })
            
            if self.progress_callback:
                self.progress_callback({
                    'status': self.status,
                    'error': str(e)
                })
    
    def _execute_with_progress_tracking(self, workflow_input: Dict[str, Any]):
        """Execute workflow with detailed progress tracking."""
        
        # Mock execution for demo purposes
        # In production, this would integrate with the actual GroceryWorkflow
        stages = [
            "initialize",
            "extract_ingredients", 
            "parallel_scraping",
            "aggregate_products",
            "parallel_matching",
            "aggregate_matches",
            "optimize_shopping",
            "strategy_analysis",
            "finalize_results",
            "analytics",
            "complete"
        ]
        
        total_stages = len(stages)
        
        for i, stage in enumerate(stages):
            
            # Check if cancelled
            if self.status == WorkflowStatus.CANCELLED:
                return
            
            # Update progress
            progress = (i + 1) / total_stages
            
            self.logs.append({
                'timestamp': datetime.now(),
                'level': 'INFO',
                'message': f"Starting stage: {stage}"
            })
            
            # Simulate stage execution time
            stage_duration = self._get_stage_duration(stage)
            
            # Update metrics
            self.metrics.update({
                'current_stage': stage,
                'progress': progress,
                'stages_completed': i + 1,
                'total_stages': total_stages,
                'execution_time': time.time() - self.metrics.get('start_time', time.time())
            })
            
            # Call progress callback
            if self.progress_callback:
                self.progress_callback({
                    'status': self.status,
                    'metrics': self.metrics,
                    'current_stage': stage,
                    'progress': progress
                })
            
            # Simulate work
            time.sleep(stage_duration)
            
            self.logs.append({
                'timestamp': datetime.now(),
                'level': 'INFO', 
                'message': f"Completed stage: {stage}"
            })
        
        # Generate mock results
        self.results = self._generate_mock_results(workflow_input)
        self.status = WorkflowStatus.COMPLETED
        
        if self.progress_callback:
            self.progress_callback({
                'status': self.status,
                'results': self.results,
                'metrics': self.metrics
            })
    
    def _get_stage_duration(self, stage: str) -> float:
        """Get simulated duration for each stage."""
        durations = {
            "initialize": 1.0,
            "extract_ingredients": 2.0,
            "parallel_scraping": 8.0,
            "aggregate_products": 2.0, 
            "parallel_matching": 6.0,
            "aggregate_matches": 2.0,
            "optimize_shopping": 4.0,
            "strategy_analysis": 3.0,
            "finalize_results": 1.0,
            "analytics": 1.0,
            "complete": 0.5
        }
        return durations.get(stage, 2.0)
    
    def _convert_ingredients(self, ingredients_list: List[Dict[str, Any]]) -> List[Ingredient]:
        """Convert ingredient dictionaries to Ingredient objects."""
        converted = []
        
        for ing_data in ingredients_list:
            try:
                ingredient = Ingredient(
                    name=ing_data.get('name', ''),
                    quantity=float(ing_data.get('quantity', 1.0)),
                    unit=ing_data.get('unit', 'pieces')
                )
                converted.append(ingredient)
            except Exception as e:
                logger.warning(f"Failed to convert ingredient {ing_data}: {e}")
                
        return converted
    
    def _convert_recipes(self, recipes_list: List[Dict[str, Any]]) -> List[Recipe]:
        """Convert recipe dictionaries to Recipe objects."""
        converted = []
        
        for recipe_data in recipes_list:
            try:
                ingredients = self._convert_ingredients(
                    recipe_data.get('ingredients', [])
                )
                
                recipe = Recipe(
                    name=recipe_data.get('name', ''),
                    ingredients=ingredients,
                    servings=recipe_data.get('servings', 4)
                )
                converted.append(recipe)
            except Exception as e:
                logger.warning(f"Failed to convert recipe {recipe_data}: {e}")
                
        return converted
    
    def _generate_mock_results(self, workflow_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock results for demonstration."""
        
        total_ingredients = len(workflow_input['ingredients']) + \
                          sum(len(recipe.ingredients) for recipe in workflow_input['recipes'])
        
        return {
            'execution_summary': {
                'total_ingredients': total_ingredients,
                'products_collected': total_ingredients * 4,
                'matches_found': int(total_ingredients * 0.95),
                'stores_visited': min(len(workflow_input['target_stores']), workflow_input['max_stores']),
                'total_cost': round(total_ingredients * 3.5 + 15.0, 2),
                'estimated_savings': round(total_ingredients * 0.8, 2),
                'execution_time': self.metrics.get('execution_time', 30.0)
            },
            'shopping_lists': self._generate_shopping_lists(workflow_input),
            'optimization_results': {
                'selected_strategy': workflow_input['optimization_strategy'],
                'savings_vs_baseline': 18.7,
                'convenience_score': 8.2,
                'quality_score': 7.9
            },
            'performance_metrics': {
                'scraping_success_rate': 0.985,
                'matching_success_rate': 0.958,
                'optimization_success_rate': 0.992,
                'memory_usage_mb': 387.5,
                'total_execution_time': self.metrics.get('execution_time', 30.0)
            }
        }
    
    def _generate_shopping_lists(self, workflow_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock shopping lists by store."""
        
        stores = workflow_input['target_stores'][:workflow_input['max_stores']]
        shopping_lists = {}
        
        for i, store in enumerate(stores):
            store_name = {
                'metro_ca': 'Metro',
                'walmart_ca': 'Walmart',
                'freshco_com': 'FreshCo'
            }.get(store, store)
            
            # Mock items for each store
            base_items = [
                {'name': 'Organic Milk 2L', 'price': 6.99, 'confidence': 0.95},
                {'name': 'Whole Wheat Bread', 'price': 3.49, 'confidence': 0.88},
                {'name': 'Free Range Eggs', 'price': 4.99, 'confidence': 0.92},
                {'name': 'Chicken Breast 1kg', 'price': 12.99, 'confidence': 0.90},
                {'name': 'Basmati Rice 2kg', 'price': 8.99, 'confidence': 0.95}
            ]
            
            # Distribute items across stores
            items_per_store = len(base_items) // len(stores)
            start_idx = i * items_per_store
            end_idx = start_idx + items_per_store if i < len(stores) - 1 else len(base_items)
            
            store_items = base_items[start_idx:end_idx]
            store_total = sum(item['price'] for item in store_items)
            
            shopping_lists[store_name] = {
                'items': store_items,
                'total': round(store_total, 2),
                'savings': round(store_total * 0.12, 2)
            }
        
        return shopping_lists
    
    def cancel_execution(self):
        """Cancel the current workflow execution."""
        self.status = WorkflowStatus.CANCELLED
        logger.info("Workflow execution cancelled")
    
    def get_status(self) -> WorkflowStatus:
        """Get current execution status."""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        return self.metrics.copy()
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get execution results if available."""
        return self.results
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get execution logs."""
        return self.logs.copy()