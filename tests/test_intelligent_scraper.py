"""
Comprehensive tests for the intelligent scraper agent with 3-layer fallback system.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from agentic_grocery_price_scanner.agents.intelligent_scraper_agent import (
    IntelligentScraperAgent,
    CollectionStrategy,
    ScrapingState
)
from agentic_grocery_price_scanner.agents.scraping_ui import (
    ScrapingUIManager,
    InteractiveScrapingSession,
    UIUpdateType
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod, Currency


class TestIntelligentScraperAgent:
    """Test the intelligent scraper agent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return IntelligentScraperAgent()
    
    @pytest.fixture
    def sample_products(self):
        """Sample products for testing."""
        return [
            Product(
                name="Organic Milk 1L",
                brand="Organic Valley",
                price=Decimal("5.99"),
                currency=Currency.CAD,
                store_id="metro_ca",
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8
            ),
            Product(
                name="Whole Wheat Bread",
                brand="Wonder",
                price=Decimal("3.49"),
                currency=Currency.CAD,
                store_id="walmart_ca",
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=1.0
            ),
            Product(
                name="Free Range Eggs 12pk",
                brand="Happy Hen",
                price=Decimal("6.99"),
                currency=Currency.CAD,
                store_id="freshco_com",
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=0.95
            )
        ]
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "intelligent_scraper"
        assert agent.workflow is not None
        assert agent.checkpointer is not None
        assert len(agent.method_stats) == 3
    
    @pytest.mark.asyncio
    async def test_execute_basic_workflow(self, agent, sample_products):
        """Test basic workflow execution."""
        # Mock the workflow execution
        with patch.object(agent.workflow, 'ainvoke') as mock_workflow:
            mock_final_state = {
                "products": sample_products,
                "completed_stores": ["metro_ca", "walmart_ca"],
                "failed_stores": [],
                "errors": {},
                "collection_metadata": {"start_time": "2024-01-01T12:00:00"},
                "success_rates": {"stealth": 1.0}
            }
            mock_workflow.return_value = mock_final_state
            
            result = await agent.execute({
                "query": "milk",
                "stores": ["metro_ca", "walmart_ca"],
                "limit": 20
            })
            
            assert result["success"] is True
            assert result["query"] == "milk"
            assert len(result["products"]) == 3
            assert result["total_products"] == 3
            assert result["stores_completed"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_missing_query(self, agent):
        """Test execution fails with missing query."""
        with pytest.raises(ValueError, match="Query is required"):
            await agent.execute({})
    
    @pytest.mark.asyncio
    async def test_stealth_scraping_layer(self, agent, sample_products):
        """Test stealth scraping layer individually."""
        # Mock stealth scraper
        mock_stealth_scraper = AsyncMock()
        mock_stealth_scraper.scrape_store.return_value = sample_products[:1]
        agent.stealth_scraper = mock_stealth_scraper
        
        # Test stealth layer
        state = {
            "query": "milk",
            "stores": ["metro_ca"],
            "limit": 10,
            "products": [],
            "completed_stores": [],
            "failed_stores": [],
            "errors": {},
            "collection_metadata": {"layers_attempted": []},
            "progress_callback": None
        }
        
        result_state = await agent._try_stealth_scraping(state)
        
        assert len(result_state["products"]) == 1
        assert "metro_ca" in result_state["completed_stores"]
        assert result_state["products"][0].collection_method == DataCollectionMethod.AUTOMATED_STEALTH
        mock_stealth_scraper.scrape_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_human_scraping_layer(self, agent, sample_products):
        """Test human-assisted scraping layer."""
        # Mock human scraper
        mock_human_scraper = AsyncMock()
        mock_human_scraper.scrape_with_assistance.return_value = sample_products[1:2]
        agent.human_scraper = mock_human_scraper
        
        # Test human layer
        state = {
            "query": "bread",
            "stores": ["walmart_ca"],
            "limit": 10,
            "products": [],
            "completed_stores": [],
            "failed_stores": ["walmart_ca"],  # Failed in previous layer
            "errors": {},
            "collection_metadata": {"layers_attempted": []},
            "progress_callback": None
        }
        
        result_state = await agent._try_human_scraping(state)
        
        assert len(result_state["products"]) == 1
        assert "walmart_ca" in result_state["completed_stores"]
        assert "walmart_ca" not in result_state["failed_stores"]
        assert result_state["products"][0].collection_method == DataCollectionMethod.HUMAN_BROWSER
        mock_human_scraper.scrape_with_assistance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clipboard_collection_layer(self, agent, sample_products):
        """Test clipboard collection layer."""
        # Mock clipboard monitor
        mock_clipboard_monitor = MagicMock()
        mock_clipboard_product = MagicMock()
        mock_clipboard_product.suggested_product = sample_products[2]
        mock_clipboard_monitor.get_recent_products.return_value = [mock_clipboard_product]
        agent.clipboard_monitor = mock_clipboard_monitor
        
        # Test clipboard layer with short duration
        state = {
            "query": "eggs",
            "stores": ["freshco_com"],
            "limit": 10,
            "products": [],
            "completed_stores": [],
            "failed_stores": ["freshco_com"],
            "errors": {},
            "collection_metadata": {"layers_attempted": []},
            "progress_callback": None
        }
        
        # Patch time.time to simulate quick completion
        start_time = time.time()
        with patch('time.time', side_effect=lambda: start_time + 10):  # 10 seconds elapsed
            result_state = await agent._try_clipboard_collection(state)
        
        assert len(result_state["products"]) == 1
        assert "freshco_com" in result_state["completed_stores"]
        assert result_state["products"][0].collection_method == DataCollectionMethod.CLIPBOARD_MANUAL
    
    def test_decision_logic_stealth_success(self, agent):
        """Test decision logic when stealth scraping succeeds."""
        state = {
            "success_rates": {"stealth": 0.9},
            "products": [MagicMock() for _ in range(10)],
            "limit": 10,
            "stores": ["metro_ca"],
            "failed_stores": []
        }
        
        decision = agent._should_escalate_from_stealth(state)
        assert decision == "continue_stealth"
    
    def test_decision_logic_stealth_failure(self, agent):
        """Test decision logic when stealth scraping fails."""
        state = {
            "success_rates": {"stealth": 0.0},
            "products": [],
            "limit": 10,
            "stores": ["metro_ca"],
            "failed_stores": ["metro_ca"]
        }
        
        decision = agent._should_escalate_from_stealth(state)
        assert decision == "skip_to_clipboard"
    
    def test_decision_logic_stealth_partial(self, agent):
        """Test decision logic when stealth scraping partially succeeds."""
        state = {
            "success_rates": {"stealth": 0.5},
            "products": [MagicMock() for _ in range(3)],
            "limit": 10,
            "stores": ["metro_ca", "walmart_ca"],
            "failed_stores": ["walmart_ca"]
        }
        
        decision = agent._should_escalate_from_stealth(state)
        assert decision == "escalate_human"
    
    def test_decision_logic_human_success(self, agent):
        """Test decision logic when human scraping succeeds."""
        state = {
            "failed_stores": []
        }
        
        decision = agent._should_escalate_from_human(state)
        assert decision == "continue_human"
    
    def test_decision_logic_human_partial(self, agent):
        """Test decision logic when human scraping has remaining failures."""
        state = {
            "failed_stores": ["freshco_com"]
        }
        
        decision = agent._should_escalate_from_human(state)
        assert decision == "escalate_clipboard"
    
    @pytest.mark.asyncio
    async def test_aggregation_deduplication(self, agent):
        """Test product aggregation and deduplication."""
        # Create duplicate products with different confidence scores
        products = [
            Product(
                name="Organic Milk 1L",
                price=Decimal("5.99"),
                store_id="metro_ca",
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8
            ),
            Product(
                name="Organic Milk 1L",  # Same product
                price=Decimal("5.99"),
                store_id="metro_ca",
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=1.0  # Higher confidence
            ),
            Product(
                name="Different Product",
                price=Decimal("3.99"),
                store_id="metro_ca",
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=0.95
            )
        ]
        
        state = {
            "products": products,
            "collection_metadata": {},
            "progress_callback": None
        }
        
        result_state = await agent._aggregate_results(state)
        
        # Should have 2 unique products, with higher confidence one kept
        assert len(result_state["products"]) == 2
        
        # Find the organic milk product
        milk_product = next(p for p in result_state["products"] if "Organic Milk" in p.name)
        assert milk_product.collection_method == DataCollectionMethod.HUMAN_BROWSER  # Higher confidence
    
    def test_method_stats_calculation(self, agent, sample_products):
        """Test calculation of collection method statistics."""
        stats = agent._calculate_method_stats(sample_products)
        
        assert "method_distribution" in stats
        assert "average_confidence" in stats
        assert "total_products" in stats
        
        # Check method distribution
        distribution = stats["method_distribution"]
        assert distribution["automated_stealth"] == 1
        assert distribution["human_browser"] == 1
        assert distribution["clipboard_manual"] == 1
        
        assert stats["total_products"] == 3
    
    def test_strategy_recommendations(self, agent):
        """Test generation of strategy recommendations."""
        # Set up method stats
        agent.method_stats = {
            DataCollectionMethod.AUTOMATED_STEALTH: {"attempts": 10, "successes": 9},
            DataCollectionMethod.HUMAN_BROWSER: {"attempts": 5, "successes": 5},
            DataCollectionMethod.CLIPBOARD_MANUAL: {"attempts": 3, "successes": 1}
        }
        
        recommendations = agent._generate_strategy_recommendations()
        
        assert len(recommendations) >= 2
        assert any("Prioritize automated_stealth" in rec for rec in recommendations)
        assert any("Prioritize human_browser" in rec for rec in recommendations)
        assert any("Consider skipping clipboard_manual" in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_individual_layer_testing(self, agent):
        """Test individual layer testing functionality."""
        # Mock components
        agent.stealth_scraper = AsyncMock()
        agent.stealth_scraper.scrape_store.return_value = [MagicMock()]
        
        # Test stealth layer
        result = await agent.test_layer_individually(1, "milk", "metro_ca")
        assert result["success"] is True
        assert len(result["products"]) == 1
        
        # Test invalid layer
        with pytest.raises(ValueError, match="Layer must be 1, 2, or 3"):
            await agent.test_layer_individually(4, "milk", "metro_ca")
    
    def test_collection_analytics(self, agent):
        """Test collection analytics functionality."""
        # Set up some stats
        agent.method_stats[DataCollectionMethod.AUTOMATED_STEALTH] = {"attempts": 5, "successes": 4}
        
        analytics = agent.get_collection_analytics()
        
        assert "method_stats" in analytics
        assert "recommendations" in analytics
        assert isinstance(analytics["recommendations"], list)


class TestScrapingUIManager:
    """Test the scraping UI manager."""
    
    @pytest.fixture
    def ui_manager(self):
        """Create UI manager for testing."""
        return ScrapingUIManager(enable_console_output=False)  # Disable console for tests
    
    def test_ui_manager_initialization(self, ui_manager):
        """Test UI manager initializes correctly."""
        assert ui_manager.progress is not None
        assert ui_manager.callbacks == []
        assert ui_manager.enable_console_output is False
    
    def test_callback_management(self, ui_manager):
        """Test adding and removing callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Add callbacks
        ui_manager.add_callback(callback1)
        ui_manager.add_callback(callback2)
        assert len(ui_manager.callbacks) == 2
        
        # Remove callback
        ui_manager.remove_callback(callback1)
        assert len(ui_manager.callbacks) == 1
        assert callback2 in ui_manager.callbacks
    
    def test_progress_updates(self, ui_manager):
        """Test progress update functionality."""
        callback = MagicMock()
        ui_manager.add_callback(callback)
        
        # Update progress
        ui_manager.update_progress("Test message", layer=1)
        
        # Check callback was called
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == UIUpdateType.PROGRESS
        assert args[1]["message"] == "Test message"
        assert args[1]["layer"] == 1
        
        # Check progress state
        assert ui_manager.progress.current_status == "Test message"
        assert ui_manager.progress.current_layer == 1
    
    def test_layer_change_notification(self, ui_manager):
        """Test layer change notifications."""
        callback = MagicMock()
        ui_manager.add_callback(callback)
        
        ui_manager.layer_changed(2, "human_browser")
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == UIUpdateType.LAYER_CHANGE
        assert args[1]["layer"] == 2
        assert args[1]["layer_name"] == "human_browser"
        
        assert ui_manager.progress.current_layer == 2
        assert "human_browser" in ui_manager.progress.layers_attempted
    
    def test_product_found_notification(self, ui_manager):
        """Test product found notifications."""
        callback = MagicMock()
        ui_manager.add_callback(callback)
        
        product = Product(
            name="Test Product",
            price=Decimal("9.99"),
            store_id="test_store",
            collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
            confidence_score=0.8
        )
        
        ui_manager.product_found(product, "stealth_scraper")
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == UIUpdateType.PRODUCT_FOUND
        assert args[1]["product"]["name"] == "Test Product"
        assert args[1]["source"] == "stealth_scraper"
        
        assert ui_manager.progress.products_collected == 1
    
    def test_store_completion_tracking(self, ui_manager):
        """Test store completion tracking."""
        ui_manager.store_completed("metro_ca", 5)
        assert "metro_ca" in ui_manager.progress.stores_completed
        
        ui_manager.store_failed("walmart_ca", "Connection timeout")
        assert "walmart_ca" in ui_manager.progress.stores_failed
        assert len(ui_manager.progress.error_messages) == 1
    
    def test_progress_callback_creation(self, ui_manager):
        """Test creation of progress callback."""
        callback = ui_manager.create_progress_callback()
        
        # Test callback function
        callback("🤖 Layer 1: Starting stealth scraping...")
        
        assert ui_manager.progress.current_layer == 1
        assert "Layer 1" in ui_manager.progress.current_status
    
    def test_progress_serialization(self, ui_manager):
        """Test progress serialization to dict."""
        ui_manager.update_progress("Test message")
        ui_manager.layer_changed(2, "human")
        
        progress_dict = ui_manager.progress.to_dict()
        
        assert "start_time" in progress_dict
        assert "current_layer" in progress_dict
        assert "current_status" in progress_dict
        assert "elapsed_time" in progress_dict
        assert progress_dict["current_layer"] == 2
        assert progress_dict["current_status"] == "Test message"


class TestInteractiveScrapingSession:
    """Test the interactive scraping session."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        agent = MagicMock()
        agent.execute = AsyncMock()
        return agent
    
    @pytest.fixture
    def session(self, mock_agent):
        """Create session for testing."""
        return InteractiveScrapingSession(mock_agent, enable_console=False)
    
    @pytest.mark.asyncio
    async def test_session_start_scraping(self, session, mock_agent, sample_products):
        """Test starting a scraping session."""
        # Mock agent response
        mock_result = {
            "success": True,
            "products": sample_products,
            "total_products": len(sample_products)
        }
        mock_agent.execute.return_value = mock_result
        
        result = await session.start_scraping(
            query="milk",
            stores=["metro_ca"],
            limit=20,
            strategy="adaptive"
        )
        
        assert result["success"] is True
        assert len(result["products"]) == 3
        
        # Check agent was called with correct inputs
        mock_agent.execute.assert_called_once()
        call_args = mock_agent.execute.call_args[0][0]
        assert call_args["query"] == "milk"
        assert call_args["stores"] == ["metro_ca"]
        assert call_args["limit"] == 20
        assert call_args["strategy"] == "adaptive"
        assert call_args["progress_callback"] is not None
    
    def test_session_ui_callback_management(self, session):
        """Test UI callback management in session."""
        callback = MagicMock()
        session.add_ui_callback(callback)
        
        assert callback in session.ui_manager.callbacks
    
    def test_session_progress_tracking(self, session):
        """Test session progress tracking."""
        progress = session.get_session_progress()
        
        assert "start_time" in progress
        assert "current_layer" in progress
        assert "products_collected" in progress


@pytest.mark.integration
class TestFullSystemIntegration:
    """Integration tests for the complete scraping system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_scraping_workflow(self):
        """Test complete end-to-end workflow."""
        # This test would require actual store configurations
        # and mock network responses - placeholder for future implementation
        pass
    
    @pytest.mark.asyncio
    async def test_fallback_chain_execution(self):
        """Test complete fallback chain execution."""
        # This test would simulate failures in each layer
        # and verify proper escalation - placeholder for future implementation
        pass
    
    @pytest.mark.performance
    async def test_concurrent_store_scraping(self):
        """Test concurrent scraping of multiple stores."""
        # This test would verify performance with multiple stores
        # - placeholder for future implementation
        pass


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-m", "not integration",  # Skip integration tests by default
        "--tb=short"
    ])