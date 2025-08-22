#!/usr/bin/env python3
"""
Quick demonstration of the MatcherAgent functionality.
This demo shows the MatcherAgent successfully matching ingredients to products.
"""

import asyncio
import logging
from decimal import Decimal

from agentic_grocery_price_scanner.agents.matcher_agent import MatcherAgent, MatchingStrategy
from agentic_grocery_price_scanner.data_models.ingredient import Ingredient
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod, UnitType
from agentic_grocery_price_scanner.vector_db.qdrant_client import QdrantVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate MatcherAgent functionality."""
    print("🚀 MatcherAgent Demonstration")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n📦 Setting up test environment...")
    vector_db = QdrantVectorDB(in_memory=True)
    matcher = MatcherAgent(vector_db=vector_db)
    
    # 2. Add sample products to database
    print("📋 Adding sample products to database...")
    
    sample_products = [
        Product(
            name="Whole Milk 4L",
            brand="Beatrice",
            price=Decimal("5.99"),
            currency="CAD",
            store_id="metro_ca",
            category="Dairy & Eggs",
            in_stock=True,
            collection_method=DataCollectionMethod.HUMAN_BROWSER,
            confidence_score=1.0,
            keywords=["milk", "dairy", "whole", "4L"]
        ),
        
        Product(
            name="Organic Chicken Breast",
            brand="Maple Leaf",
            price=Decimal("12.99"),
            currency="CAD",
            store_id="metro_ca",
            category="Meat & Seafood", 
            in_stock=True,
            on_sale=True,
            sale_price=Decimal("9.99"),
            collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
            confidence_score=0.9,
            keywords=["chicken", "breast", "organic", "meat"]
        ),
        
        Product(
            name="Whole Wheat Bread",
            brand="Wonder",
            price=Decimal("3.49"),
            currency="CAD",
            store_id="walmart_ca",
            category="Bakery",
            in_stock=True,
            collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
            confidence_score=0.95,
            keywords=["bread", "whole wheat", "bakery"]
        ),
        
        Product(
            name="All Purpose Flour 2kg",
            brand="Robin Hood",
            price=Decimal("4.99"),
            currency="CAD",
            store_id="metro_ca",
            category="Pantry",
            in_stock=True,
            collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
            confidence_score=0.8,
            keywords=["flour", "baking", "all-purpose"]
        ),
        
        Product(
            name="Greek Yogurt 750g",
            brand="Oikos",
            price=Decimal("6.49"),
            currency="CAD",
            store_id="walmart_ca",
            category="Dairy & Eggs",
            in_stock=True,
            collection_method=DataCollectionMethod.HUMAN_BROWSER,
            confidence_score=1.0,
            keywords=["yogurt", "greek", "dairy"]
        )
    ]
    
    # Add products to vector database
    vector_db.add_products(sample_products)
    print(f"✅ Added {len(sample_products)} products to database")
    
    # 3. Test ingredient matching
    print("\n🔍 Testing ingredient matching...")
    
    test_cases = [
        {
            "ingredient": Ingredient(
                name="milk",
                quantity=2.0,
                unit=UnitType.CUPS,
                category="dairy"
            ),
            "description": "Basic dairy ingredient"
        },
        {
            "ingredient": Ingredient(
                name="chicken breast",
                quantity=1.0,
                unit=UnitType.POUNDS,
                category="meat",
                alternatives=["chicken", "poultry"]
            ),
            "description": "Meat with alternatives"
        },
        {
            "ingredient": Ingredient(
                name="bread",
                quantity=1.0,
                unit=UnitType.PIECES,
                category="bakery"
            ),
            "description": "Bakery item"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        ingredient = test_case["ingredient"]
        description = test_case["description"]
        
        print(f"\n[Test {i}] {description}: '{ingredient.name}'")
        
        # Perform matching
        result = await matcher.match_ingredient(
            ingredient=ingredient,
            strategy="adaptive",
            confidence_threshold=0.3,
            max_results=3
        )
        
        if result['success'] and result['matches']:
            print(f"✅ Found {len(result['matches'])} matches:")
            
            for j, match in enumerate(result['matches'], 1):
                product = match['product']
                confidence = match['confidence']
                quality = match['quality'].value
                
                # Format price display
                price_display = f"${product.price}"
                if product.on_sale and product.sale_price:
                    price_display = f"${product.sale_price} (was ${product.price}) 🏷️"
                
                print(f"  {j}. {product.name}")
                print(f"     Brand: {product.brand}")
                print(f"     Store: {product.store_id}")
                print(f"     Price: {price_display}")
                print(f"     Confidence: {confidence:.3f} ({quality})")
                print(f"     Collection: {product.collection_method}")
            
            # Show substitutions if available
            substitutions = result.get('substitution_suggestions', [])
            if substitutions:
                print(f"   🔄 {len(substitutions)} alternative suggestions available")
                
        else:
            print(f"❌ No matches found")
            if result.get('require_human_review'):
                reason = result.get('matching_metadata', {}).get('human_review_reason', 'Unknown')
                print(f"   👤 Human review recommended: {reason}")
    
    # 4. Test substitution suggestions
    print(f"\n🔄 Testing substitution suggestions...")
    
    substitutions = await matcher.suggest_substitutions(
        ingredient_name="yogurt",
        max_suggestions=3
    )
    
    if substitutions:
        print(f"✅ Found {len(substitutions)} substitution suggestions for 'yogurt':")
        for i, sub in enumerate(substitutions, 1):
            product = sub['product']
            print(f"  {i}. {product.name} - ${product.price}")
            print(f"     Type: {sub['type']}")
            print(f"     Reason: {sub['reason']}")
            print(f"     Confidence: {sub['confidence']:.3f}")
    else:
        print("❌ No substitution suggestions found")
    
    # 5. Show analytics
    print(f"\n📊 MatcherAgent Analytics:")
    analytics = matcher.get_matching_analytics()
    stats = analytics['matching_stats']
    
    print(f"Total matches attempted: {stats['total_matches']}")
    print(f"Successful matches: {stats['successful_matches']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    
    # Quality distribution
    quality_dist = stats['quality_distribution']
    if any(count > 0 for count in quality_dist.values()):
        print("Quality distribution:")
        for quality, count in quality_dist.items():
            if count > 0:
                print(f"  {quality}: {count}")
    
    # 6. Demonstrate CLI integration
    print(f"\n💻 CLI Integration Available:")
    print("Try these commands:")
    print("  python3 -m agentic_grocery_price_scanner.cli match ingredient --ingredient 'milk' --verbose")
    print("  python3 -m agentic_grocery_price_scanner.cli match batch --ingredients 'milk,bread,chicken'")
    print("  python3 -m agentic_grocery_price_scanner.cli match substitutions --ingredient 'milk'")
    print("  python3 -m agentic_grocery_price_scanner.cli match analytics")
    print("  python3 -m agentic_grocery_price_scanner.cli match test")
    
    print(f"\n✅ MatcherAgent demonstration completed successfully!")
    print("🎯 Key Features Demonstrated:")
    print("  • Vector-based semantic search")
    print("  • LLM-powered intelligent reasoning")
    print("  • Brand normalization and fuzzy matching")
    print("  • Confidence scoring and quality control")
    print("  • Substitution suggestions")
    print("  • Human review flagging")
    print("  • Performance analytics")
    print("  • Complete CLI integration")


if __name__ == "__main__":
    asyncio.run(main())