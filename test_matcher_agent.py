#!/usr/bin/env python3
"""
Comprehensive test suite for the MatcherAgent.
Tests ingredient-to-product matching with various scenarios.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import List, Dict, Any

from agentic_grocery_price_scanner.agents.matcher_agent import MatcherAgent, MatchingStrategy, MatchingQuality
from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod, UnitType
from agentic_grocery_price_scanner.vector_db.qdrant_client import QdrantVectorDB
from agentic_grocery_price_scanner.llm_client.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatcherAgentTester:
    """Comprehensive tester for MatcherAgent functionality."""
    
    def __init__(self):
        """Initialize the tester with mock data."""
        self.vector_db = QdrantVectorDB(in_memory=True)
        self.llm_client = OllamaClient()
        self.matcher_agent = MatcherAgent(
            vector_db=self.vector_db,
            llm_client=self.llm_client
        )
        
        # Test ingredients
        self.test_ingredients = self._create_test_ingredients()
        
        # Mock products
        self.mock_products = self._create_mock_products()
    
    def _create_test_ingredients(self) -> List[Ingredient]:
        """Create test ingredients covering various scenarios."""
        return [
            # Basic ingredients
            Ingredient(
                name="milk",
                quantity=2.0,
                unit=UnitType.CUPS,
                category="dairy",
                alternatives=["whole milk", "2% milk"]
            ),
            
            # Brand-specific ingredient
            Ingredient(
                name="Kellogg's corn flakes",
                quantity=1.0,
                unit=UnitType.PACKAGES,
                category="cereal",
                alternatives=["corn flakes", "breakfast cereal"]
            ),
            
            # Complex ingredient
            Ingredient(
                name="organic free-range chicken breast",
                quantity=2.0,
                unit=UnitType.POUNDS,
                category="meat",
                alternatives=["chicken breast", "organic chicken"]
            ),
            
            # Specific size ingredient
            Ingredient(
                name="bread loaf whole wheat",
                quantity=1.0,
                unit=UnitType.PIECES,
                category="bakery",
                alternatives=["whole wheat bread", "bread"]
            ),
            
            # Ambiguous ingredient
            Ingredient(
                name="flour",
                quantity=2.0,
                unit=UnitType.CUPS,
                category="baking",
                alternatives=["all-purpose flour", "wheat flour"]
            ),
            
            # Uncommon ingredient
            Ingredient(
                name="tahini",
                quantity=0.25,
                unit=UnitType.CUPS,
                category="condiment",
                alternatives=["sesame paste", "sesame butter"]
            )
        ]
    
    def _create_mock_products(self) -> List[Product]:
        """Create mock products for testing."""
        return [
            # Dairy products
            Product(
                name="Whole Milk 4L",
                brand="Beatrice",
                price=Decimal("5.99"),
                currency="CAD",
                store_id="metro_ca",
                category="Dairy & Eggs",
                in_stock=True,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8,
                size=4.0,
                size_unit="l"
            ),
            
            Product(
                name="2% Milk 2L",
                brand="Natrel",
                price=Decimal("4.49"),
                currency="CAD", 
                store_id="walmart_ca",
                category="Dairy & Eggs",
                in_stock=True,
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=1.0,
                size=2.0,
                size_unit="l"
            ),
            
            # Cereals
            Product(
                name="Kellogg's Corn Flakes 525g",
                brand="Kellogg's",
                price=Decimal("6.49"),
                currency="CAD",
                store_id="metro_ca",
                category="Breakfast & Cereal",
                in_stock=True,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.85,
                keywords=["breakfast", "cereal", "corn"]
            ),
            
            Product(
                name="Corn Flakes Generic 500g",
                brand="No Name",
                price=Decimal("3.99"),
                currency="CAD",
                store_id="walmart_ca",
                category="Breakfast & Cereal",
                in_stock=True,
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=0.9,
                keywords=["breakfast", "cereal", "corn", "generic"]
            ),
            
            # Meat products
            Product(
                name="Organic Chicken Breast",
                brand="Maple Leaf",
                price=Decimal("12.99"),
                currency="CAD",
                store_id="metro_ca",
                category="Meat & Seafood",
                subcategory="Poultry",
                in_stock=True,
                on_sale=True,
                sale_price=Decimal("10.99"),
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=1.0,
                keywords=["organic", "chicken", "breast", "free-range"]
            ),
            
            Product(
                name="Chicken Breast Family Pack",
                brand="Fresh from Farm",
                price=Decimal("8.99"),
                currency="CAD",
                store_id="walmart_ca",
                category="Meat & Seafood",
                in_stock=True,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.7,
                size=2.5,
                size_unit="lbs"
            ),
            
            # Bakery products
            Product(
                name="Whole Wheat Bread 675g",
                brand="Wonder",
                price=Decimal("3.49"),
                currency="CAD",
                store_id="metro_ca",
                category="Bakery",
                in_stock=True,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8,
                keywords=["bread", "whole wheat", "loaf"]
            ),
            
            Product(
                name="Artisan Whole Grain Bread",
                brand="ACE Bakery",
                price=Decimal("4.99"),
                currency="CAD",
                store_id="walmart_ca", 
                category="Bakery",
                in_stock=True,
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=0.95,
                keywords=["bread", "whole grain", "artisan"]
            ),
            
            # Baking ingredients
            Product(
                name="All Purpose Flour 2.5kg",
                brand="Robin Hood",
                price=Decimal("4.99"),
                currency="CAD",
                store_id="metro_ca",
                category="Pantry",
                subcategory="Baking",
                in_stock=True,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8,
                keywords=["flour", "all-purpose", "baking"]
            ),
            
            Product(
                name="Organic Wheat Flour 1kg",
                brand="Bulk Barn",
                price=Decimal("3.99"),
                currency="CAD",
                store_id="walmart_ca",
                category="Pantry",
                in_stock=True,
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=0.9,
                keywords=["flour", "wheat", "organic"]
            ),
            
            # Specialty products
            Product(
                name="Tahini Sesame Paste 454g",
                brand="Joyva",
                price=Decimal("8.99"),
                currency="CAD",
                store_id="metro_ca",
                category="Condiments",
                in_stock=True,
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=1.0,
                keywords=["tahini", "sesame", "paste", "middle eastern"]
            ),
            
            Product(
                name="Organic Sesame Butter 250g",
                brand="Nuts to You",
                price=Decimal("12.49"),
                currency="CAD",
                store_id="walmart_ca",
                category="Health Food",
                in_stock=False,  # Out of stock
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.75,
                keywords=["sesame", "butter", "organic", "tahini"]
            )
        ]
    
    async def setup_test_data(self):
        """Setup test data in vector database."""
        logger.info("Setting up test data...")
        
        # Add mock products to vector database
        self.vector_db.add_products(self.mock_products)
        
        logger.info(f"Added {len(self.mock_products)} products to vector database")
    
    async def test_basic_matching(self):
        """Test basic ingredient matching functionality."""
        logger.info("\n=== Testing Basic Matching ===")
        
        test_ingredient = self.test_ingredients[0]  # milk
        
        result = await self.matcher_agent.match_ingredient(
            ingredient=test_ingredient,
            strategy="adaptive",
            confidence_threshold=0.3,
            max_results=3
        )
        
        logger.info(f"Ingredient: {test_ingredient.name}")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Total matches: {result['total_matches']}")
        
        for i, match in enumerate(result['matches'][:3]):
            product = match['product']
            logger.info(f"  {i+1}. {product.name} - {product.brand}")
            logger.info(f"     Confidence: {match['confidence']:.3f}")
            logger.info(f"     Price: ${product.price} at {product.store_id}")
            logger.info(f"     Quality: {match['quality'].value}")
            if 'llm_reason' in match:
                logger.info(f"     LLM Reason: {match['llm_reason']}")
        
        return result
    
    async def test_brand_matching(self):
        """Test brand-specific matching."""
        logger.info("\n=== Testing Brand Matching ===")
        
        test_ingredient = self.test_ingredients[1]  # Kellogg's corn flakes
        
        result = await self.matcher_agent.match_ingredient(
            ingredient=test_ingredient,
            strategy="hybrid",
            confidence_threshold=0.4,
            max_results=5
        )
        
        logger.info(f"Ingredient: {test_ingredient.name}")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Total matches: {result['total_matches']}")
        
        for i, match in enumerate(result['matches']):
            product = match['product']
            logger.info(f"  {i+1}. {product.name} - {product.brand}")
            logger.info(f"     Confidence: {match['confidence']:.3f}")
            kelloggs_brand = "Kellogg's"
            logger.info(f"     Brand match: {product.brand == kelloggs_brand}")
            if 'sale_advantage' in match:
                logger.info(f"     On sale: {product.on_sale}")
        
        return result
    
    async def test_complex_matching(self):
        """Test complex ingredient matching."""
        logger.info("\n=== Testing Complex Matching ===")
        
        test_ingredient = self.test_ingredients[2]  # organic free-range chicken breast
        
        result = await self.matcher_agent.match_ingredient(
            ingredient=test_ingredient,
            strategy="adaptive",
            confidence_threshold=0.3,
            max_results=3
        )
        
        logger.info(f"Ingredient: {test_ingredient.name}")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Total matches: {result['total_matches']}")
        logger.info(f"Require human review: {result['require_human_review']}")
        
        for i, match in enumerate(result['matches']):
            product = match['product']
            logger.info(f"  {i+1}. {product.name} - {product.brand}")
            logger.info(f"     Confidence: {match['confidence']:.3f}")
            logger.info(f"     Category: {product.category}")
            if 'category_warning' in match:
                logger.info(f"     Category warning: {match['category_warning']}")
        
        # Test substitutions
        logger.info(f"Substitution suggestions: {len(result['substitution_suggestions'])}")
        for i, sub in enumerate(result['substitution_suggestions'][:2]):
            logger.info(f"  Sub {i+1}: {sub['product'].name} ({sub['type']})")
            logger.info(f"           Reason: {sub['reason']}")
        
        return result
    
    async def test_fuzzy_matching(self):
        """Test fuzzy string matching capabilities."""
        logger.info("\n=== Testing Fuzzy Matching ===")
        
        # Test with slight misspelling
        misspelled_ingredient = Ingredient(
            name="bred",  # misspelled "bread"
            quantity=1.0,
            unit=UnitType.PIECES,
            category="bakery"
        )
        
        result = await self.matcher_agent.match_ingredient(
            ingredient=misspelled_ingredient,
            strategy="adaptive",
            confidence_threshold=0.2,
            max_results=3
        )
        
        logger.info(f"Ingredient: {misspelled_ingredient.name} (misspelled)")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Total matches: {result['total_matches']}")
        
        for i, match in enumerate(result['matches']):
            product = match['product']
            logger.info(f"  {i+1}. {product.name}")
            logger.info(f"     Confidence: {match['confidence']:.3f}")
            logger.info(f"     Vector score: {match.get('vector_score', 'N/A'):.3f}")
        
        return result
    
    async def test_substitution_suggestions(self):
        """Test substitution suggestion system."""
        logger.info("\n=== Testing Substitution Suggestions ===")
        
        # Test with uncommon ingredient
        test_ingredient = self.test_ingredients[5]  # tahini
        
        substitutions = await self.matcher_agent.suggest_substitutions(
            ingredient_name=test_ingredient.name,
            max_suggestions=3
        )
        
        logger.info(f"Ingredient: {test_ingredient.name}")
        logger.info(f"Substitution suggestions: {len(substitutions)}")
        
        for i, sub in enumerate(substitutions):
            logger.info(f"  {i+1}. {sub['product'].name}")
            logger.info(f"     Type: {sub['type']}")
            logger.info(f"     Reason: {sub['reason']}")
            logger.info(f"     Confidence: {sub['confidence']:.3f}")
        
        return substitutions
    
    async def test_batch_matching(self):
        """Test batch matching functionality."""
        logger.info("\n=== Testing Batch Matching ===")
        
        # Test with first 3 ingredients
        test_ingredients = self.test_ingredients[:3]
        
        results = await self.matcher_agent.match_ingredients_batch(
            ingredients=test_ingredients,
            strategy="adaptive",
            confidence_threshold=0.4,
            max_results=2
        )
        
        logger.info(f"Batch matched {len(test_ingredients)} ingredients")
        
        for i, result in enumerate(results):
            ingredient_name = test_ingredients[i].name
            logger.info(f"  {i+1}. {ingredient_name}: {result['total_matches']} matches")
            
            if result['matches']:
                best_match = result['matches'][0]
                logger.info(f"     Best: {best_match['product'].name} (conf: {best_match['confidence']:.3f})")
        
        return results
    
    async def test_quality_control(self):
        """Test quality control and human review flagging."""
        logger.info("\n=== Testing Quality Control ===")
        
        # Test with very high confidence threshold
        test_ingredient = self.test_ingredients[0]  # milk
        
        result = await self.matcher_agent.match_ingredient(
            ingredient=test_ingredient,
            strategy="adaptive",
            confidence_threshold=0.95,  # Very high threshold
            max_results=5
        )
        
        logger.info(f"Ingredient: {test_ingredient.name}")
        logger.info(f"High threshold (0.95) - Matches: {result['total_matches']}")
        logger.info(f"Require human review: {result['require_human_review']}")
        
        if 'matching_metadata' in result and 'human_review_reason' in result['matching_metadata']:
            logger.info(f"Review reason: {result['matching_metadata']['human_review_reason']}")
        
        # Test quality distribution
        quality_dist = result['quality_distribution']
        logger.info("Quality distribution:")
        for quality, count in quality_dist.items():
            if count > 0:
                logger.info(f"  {quality}: {count}")
        
        return result
    
    async def test_performance_analytics(self):
        """Test performance analytics and recommendations."""
        logger.info("\n=== Testing Performance Analytics ===")
        
        # Run several matches to generate statistics
        for ingredient in self.test_ingredients[:4]:
            await self.matcher_agent.match_ingredient(
                ingredient=ingredient,
                strategy="adaptive",
                confidence_threshold=0.3
            )
        
        analytics = self.matcher_agent.get_matching_analytics()
        
        logger.info("Matching Analytics:")
        stats = analytics['matching_stats']
        logger.info(f"  Total matches: {stats['total_matches']}")
        logger.info(f"  Successful: {stats['successful_matches']}")
        logger.info(f"  Failed: {stats['failed_matches']}")
        logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
        
        logger.info("Quality distribution:")
        for quality, count in stats['quality_distribution'].items():
            if count > 0:
                logger.info(f"  {quality}: {count}")
        
        logger.info("Strategy performance:")
        for strategy, perf in stats['strategy_performance'].items():
            if perf['attempts'] > 0:
                success_rate = perf['successes'] / perf['attempts']
                logger.info(f"  {strategy}: {success_rate:.1%} ({perf['successes']}/{perf['attempts']})")
        
        recommendations = analytics['recommendations']
        if recommendations:
            logger.info("Recommendations:")
            for rec in recommendations:
                logger.info(f"  - {rec}")
        
        return analytics
    
    async def test_llm_integration(self):
        """Test LLM integration and health check."""
        logger.info("\n=== Testing LLM Integration ===")
        
        # Test LLM health
        health = await self.llm_client.health_check()
        logger.info(f"LLM Health: {health['status']}")
        logger.info(f"Service available: {health['service_available']}")
        logger.info(f"Models available: {health.get('models_available', 0)}")
        
        if health['status'] == 'healthy':
            logger.info("✅ LLM integration is working correctly")
        else:
            logger.warning(f"⚠️ LLM health issue: {health.get('error', 'Unknown error')}")
        
        # Test model stats
        model_stats = self.llm_client.get_model_stats()
        logger.info(f"Available models: {model_stats['available_models']}")
        logger.info(f"Cache size: {model_stats['cache_size']}")
        
        return health
    
    async def run_comprehensive_test(self):
        """Run all tests in sequence."""
        logger.info("🚀 Starting MatcherAgent Comprehensive Test Suite")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_test_data()
            
            # Test LLM first
            await self.test_llm_integration()
            
            # Core functionality tests
            await self.test_basic_matching()
            await self.test_brand_matching()
            await self.test_complex_matching()
            await self.test_fuzzy_matching()
            
            # Advanced features
            await self.test_substitution_suggestions()
            await self.test_batch_matching()
            await self.test_quality_control()
            
            # Analytics and performance
            await self.test_performance_analytics()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ All tests completed successfully!")
            logger.info("MatcherAgent is functioning correctly.")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise


async def main():
    """Main function to run the test suite."""
    tester = MatcherAgentTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())