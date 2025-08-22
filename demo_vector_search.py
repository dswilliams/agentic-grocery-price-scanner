#!/usr/bin/env python3
"""
Demonstration of vector database integration with real grocery data.
Shows all 3 scraping layers working with Qdrant vector search.
"""

import asyncio
import logging
from decimal import Decimal
from typing import List

from agentic_grocery_price_scanner.vector_db import (
    QdrantVectorDB, 
    ScraperVectorIntegration,
    ProductNormalizer
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod, Currency, UnitType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_grocery_data() -> tuple[List[dict], List[dict], List[str]]:
    """Create realistic sample data from different collection methods."""
    
    # Layer 1: Automated Stealth Scraper Data (Metro.ca)
    stealth_data = [
        {
            "name": "Neilson Organic Whole Milk",
            "brand": "Neilson",
            "price": "5.99",
            "size": "1L",
            "category": "Dairy & Eggs",
            "subcategory": "Milk",
            "description": "Organic whole milk from pasture-raised cows",
            "image_url": "https://metro.ca/images/neilson-organic-milk.jpg",
            "product_url": "https://metro.ca/en/online-grocery/aisles/dairy-eggs/milk/p/059749961479",
            "in_stock": True,
            "on_sale": False,
            "sku": "059749961479"
        },
        {
            "name": "PC Blue Menu Whole Wheat Bread",
            "brand": "President's Choice",
            "price": "3.49",
            "size": "675g",
            "category": "Bakery",
            "subcategory": "Bread",
            "description": "100% whole wheat bread, no artificial colours or flavours",
            "image_url": "https://metro.ca/images/pc-bread.jpg",
            "product_url": "https://metro.ca/en/online-grocery/aisles/bakery/bread/p/060383001234",
            "in_stock": True,
            "on_sale": True,
            "sale_price": "2.99",
            "sku": "060383001234"
        },
        {
            "name": "Bertolli Extra Virgin Olive Oil",
            "brand": "Bertolli",
            "price": "12.99",
            "size": "500ml",
            "category": "Pantry",
            "subcategory": "Oils & Vinegars",
            "description": "Premium extra virgin olive oil from Italy",
            "image_url": "https://metro.ca/images/bertolli-olive-oil.jpg",
            "product_url": "https://metro.ca/en/online-grocery/aisles/pantry/oils-vinegars/p/076808501234",
            "in_stock": True,
            "on_sale": False,
            "sku": "076808501234"
        }
    ]
    
    # Layer 2: Human-Assisted Browser Data (Walmart.ca)
    human_browser_data = [
        {
            "name": "Great Value 2% Milk",
            "brand": "Great Value",
            "price": "4.97",
            "size": "4L",
            "category": "Dairy",
            "subcategory": "Milk",
            "description": "Great Value 2% partly skimmed milk, excellent source of calcium",
            "image_url": "https://i5.walmartimages.ca/images/Enlarge/234/567/234567.jpg",
            "product_url": "https://www.walmart.ca/en/ip/great-value-2-milk/6000191234567",
            "in_stock": True,
            "on_sale": False,
            "nutrition_info": {
                "calories_per_250ml": 130,
                "protein_g": 8,
                "calcium_mg": 300
            },
            "sku": "6000191234567"
        },
        {
            "name": "Wonder Bread White",
            "brand": "Wonder",
            "price": "2.88",
            "size": "675g",
            "category": "Bakery",
            "subcategory": "Bread",
            "description": "Classic white bread, soft and fresh",
            "image_url": "https://i5.walmartimages.ca/images/Enlarge/345/678/345678.jpg",
            "product_url": "https://www.walmart.ca/en/ip/wonder-bread-white/6000191345678",
            "in_stock": True,
            "on_sale": True,
            "sale_price": "2.47",
            "sku": "6000191345678"
        },
        {
            "name": "Mazola Corn Oil",
            "brand": "Mazola",
            "price": "6.97",
            "size": "946ml",
            "category": "Pantry",
            "subcategory": "Cooking Oil",
            "description": "Pure corn oil for cooking and baking",
            "image_url": "https://i5.walmartimages.ca/images/Enlarge/456/789/456789.jpg",
            "product_url": "https://www.walmart.ca/en/ip/mazola-corn-oil/6000191456789",
            "in_stock": True,
            "on_sale": False,
            "sku": "6000191456789"
        }
    ]
    
    # Layer 3: Clipboard Manual Data (Mixed sources)
    clipboard_data = [
        """Fresh Organic Bananas
        Price: $1.99/lb
        FreshCo
        Organic, Fair Trade
        Sweet and ripe
        https://www.freshco.com/aisles/produce/fruits/bananas""",
        
        """Lactantia Whole Milk 1L
        Brand: Lactantia
        $4.49
        Vitamin D added
        Metro store
        In stock""",
        
        """Olive Oil Extra Virgin
        Colavita Brand
        $9.99 for 750ml
        Cold pressed
        First extraction
        Available at Walmart"""
    ]
    
    return stealth_data, human_browser_data, clipboard_data


async def demonstrate_vector_integration():
    """Demonstrate complete vector database integration."""
    
    print("🚀 Starting Vector Database Integration Demonstration")
    print("=" * 60)
    
    # Initialize the integration service
    integration = ScraperVectorIntegration(auto_save=True)
    
    # Get sample data
    stealth_data, human_data, clipboard_data = create_sample_grocery_data()
    
    print("\n📥 Processing data from all 3 collection layers...")
    
    # Layer 1: Process stealth scraper data
    print("\n🤖 Layer 1: Processing Automated Stealth Scraper Data (Metro.ca)")
    stealth_products = await integration.process_stealth_scraper_results(
        stealth_data, "metro_ca", "demo_session_stealth"
    )
    print(f"   ✅ Processed {len(stealth_products)} products from stealth scraper")
    for product in stealth_products:
        print(f"      - {product.name} (${product.price}, confidence: {product.confidence_score:.2f})")
    
    # Layer 2: Process human browser data
    print("\n👤 Layer 2: Processing Human-Assisted Browser Data (Walmart.ca)")
    human_products = await integration.process_human_browser_results(
        human_data, "walmart_ca", "demo_session_human"
    )
    print(f"   ✅ Processed {len(human_products)} products from human browser")
    for product in human_products:
        print(f"      - {product.name} (${product.price}, confidence: {product.confidence_score:.2f})")
    
    # Layer 3: Process clipboard data
    print("\n📋 Layer 3: Processing Clipboard Manual Data (Mixed sources)")
    clipboard_products = await integration.process_clipboard_data(
        clipboard_data, "demo_session_clipboard"
    )
    print(f"   ✅ Processed {len(clipboard_products)} products from clipboard")
    for product in clipboard_products:
        confidence_weight = product.get_collection_confidence_weight()
        print(f"      - {product.name} (${product.price}, confidence: {product.confidence_score:.2f}, weighted: {confidence_weight:.2f})")
    
    # Display collection statistics
    print("\n📊 Collection Statistics:")
    stats = integration.get_collection_statistics()
    print(f"   Total vectors: {stats.get('total_vectors', 0)}")
    
    method_dist = stats.get('method_distribution', {})
    if method_dist:
        print("   Collection method distribution:")
        for method, count in method_dist.items():
            print(f"      {method}: {count}")
    
    # Test various similarity searches
    print("\n🔍 Testing Vector Similarity Search:")
    print("-" * 40)
    
    search_queries = [
        ("milk products", "Finding all milk products across stores"),
        ("bread", "Finding bread products with different brands"),
        ("cooking oil", "Finding various cooking oils"),
        ("organic food", "Finding organic products"),
        ("dairy", "Finding all dairy products")
    ]
    
    for query, description in search_queries:
        print(f"\n🔎 {description}")
        print(f"   Query: '{query}'")
        
        results = await integration.find_similar_products(
            query, 
            limit=5, 
            min_confidence=0.3
        )
        
        if results:
            print(f"   Found {len(results)} similar products:")
            for i, (product, score) in enumerate(results, 1):
                confidence_weight = product.get_collection_confidence_weight()
                price_display = f"${product.sale_price}" if product.on_sale and product.sale_price else f"${product.price}"
                print(f"      {i}. {product.name}")
                print(f"         Store: {product.store_id} | Price: {price_display}")
                print(f"         Similarity: {score:.3f} | Confidence: {confidence_weight:.2f}")
                print(f"         Method: {product.collection_method}")
                print(f"         Keywords: {', '.join(product.keywords[:3])}")
        else:
            print("   No similar products found")
    
    # Test method-specific searches
    print("\n🎯 Testing Method-Specific Searches:")
    print("-" * 40)
    
    methods_to_test = [
        (DataCollectionMethod.AUTOMATED_STEALTH, "Automated Stealth Scraping"),
        (DataCollectionMethod.HUMAN_BROWSER, "Human-Assisted Browser"),
        (DataCollectionMethod.CLIPBOARD_MANUAL, "Clipboard Manual Entry")
    ]
    
    for method, method_name in methods_to_test:
        print(f"\n📋 Products from {method_name}:")
        
        results = await integration.find_similar_products(
            "*",  # Match all
            limit=10,
            prefer_method=method,
            min_confidence=0.0
        )
        
        method_products = [p for p, _ in results if p.collection_method == method]
        
        if method_products:
            print(f"   Found {len(method_products)} products:")
            for product, _ in [(p, s) for p, s in results if p.collection_method == method]:
                confidence_weight = product.get_collection_confidence_weight()
                print(f"      - {product.name} (confidence: {confidence_weight:.2f})")
        else:
            print("   No products found from this method")
    
    # Test confidence-based filtering
    print("\n⭐ Testing Confidence-Based Filtering:")
    print("-" * 40)
    
    confidence_thresholds = [0.95, 0.85, 0.70, 0.50]
    
    for threshold in confidence_thresholds:
        results = await integration.find_similar_products(
            "milk",
            limit=10,
            min_confidence=threshold
        )
        
        print(f"   Products with confidence >= {threshold}: {len(results)}")
    
    # Test store-specific searches
    print("\n🏪 Testing Store-Specific Searches:")
    print("-" * 40)
    
    # Get all unique stores
    all_results = await integration.find_similar_products("*", limit=100, min_confidence=0.0)
    stores = set(product.store_id for product, _ in all_results)
    
    for store in stores:
        store_results = await integration.find_similar_products(
            "*",
            limit=10,
            min_confidence=0.0
        )
        store_products = [p for p, _ in store_results if p.store_id == store]
        print(f"   {store}: {len(store_products)} products")
    
    print("\n✅ Vector Database Integration Demonstration Complete!")
    print("=" * 60)
    
    # Final summary
    final_stats = integration.get_collection_statistics()
    total_products = sum(len(products) for products in [stealth_products, human_products, clipboard_products])
    
    print(f"\n📈 Final Summary:")
    print(f"   Products processed: {total_products}")
    print(f"   Vector database size: {final_stats.get('total_vectors', 0)}")
    print(f"   Collection methods used: {len(final_stats.get('method_distribution', {}))}")
    print(f"   Search capabilities: ✅ Similarity, ✅ Filtering, ✅ Confidence weighting")
    print(f"   Integration status: ✅ All 3 layers connected successfully")


async def quick_clipboard_demo():
    """Quick demonstration of clipboard functionality."""
    
    print("\n🔧 Quick Clipboard Integration Test:")
    print("-" * 30)
    
    normalizer = ProductNormalizer()
    vector_db = QdrantVectorDB(in_memory=True, collection_name="clipboard_demo")
    
    # Test clipboard texts
    test_clipboards = [
        "Organic Free Range Eggs\nPrice: $6.99\nMetro.ca\nSize: 12 eggs\nBrand: Naturegg",
        "Whole Wheat Pasta 500g\n$2.49\nBarilla brand\nWalmart Canada",
        "Greek Yogurt Plain\n$4.99 for 750g\nLiberte\nNo added sugar"
    ]
    
    products = []
    for i, clipboard_text in enumerate(test_clipboards, 1):
        print(f"\n📋 Processing clipboard entry {i}:")
        print(f"   Raw text: {clipboard_text[:50]}...")
        
        product = normalizer.normalize_clipboard_data(clipboard_text)
        
        if product:
            products.append(product)
            vector_db.add_product(product)
            
            print(f"   ✅ Parsed: {product.name}")
            print(f"      Price: ${product.price}")
            print(f"      Store: {product.store_id}")
            print(f"      Confidence: {product.confidence_score:.2f}")
        else:
            print("   ❌ Could not parse clipboard data")
    
    # Test similarity search on clipboard products
    if products:
        print(f"\n🔍 Testing search on {len(products)} clipboard products:")
        
        results = vector_db.search_similar_products("eggs", limit=5)
        if results:
            for product, score in results:
                print(f"   - {product.name} (similarity: {score:.3f})")
        
        results = vector_db.search_similar_products("pasta", limit=5)
        if results:
            for product, score in results:
                print(f"   - {product.name} (similarity: {score:.3f})")


def main():
    """Main demonstration function."""
    
    print("🛒 Agentic Grocery Price Scanner")
    print("Vector Database Integration Demo")
    print("=" * 60)
    
    try:
        # Run main demonstration
        asyncio.run(demonstrate_vector_integration())
        
        # Run quick clipboard demo
        asyncio.run(quick_clipboard_demo())
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())