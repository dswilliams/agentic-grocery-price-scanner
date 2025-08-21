"""
Mock scraper agent for testing and demonstration purposes.
"""

import random
from decimal import Decimal
from typing import Any, Dict, List
from uuid import uuid4

from ..data_models import Product, UnitType, Currency
from .base_agent import BaseAgent


class MockScraperAgent(BaseAgent):
    """Mock scraper agent that generates fake product data for testing."""
    
    def __init__(self):
        """Initialize the mock scraper agent."""
        super().__init__("mock_scraper")
        
        # Mock product templates
        self.product_templates = {
            "milk": [
                {"name": "Organic Whole Milk", "brand": "Organic Valley", "category": "dairy"},
                {"name": "2% Milk", "brand": "Beatrice", "category": "dairy"},
                {"name": "Lactose-Free Milk", "brand": "Lactaid", "category": "dairy"},
                {"name": "Almond Milk", "brand": "Silk", "category": "plant-based"},
                {"name": "Oat Milk", "brand": "Oatly", "category": "plant-based"},
            ],
            "bread": [
                {"name": "Whole Wheat Bread", "brand": "Wonder", "category": "bakery"},
                {"name": "Sourdough Loaf", "brand": "Artisan Bakery", "category": "bakery"},
                {"name": "White Bread", "brand": "Dempster's", "category": "bakery"},
                {"name": "Multigrain Bread", "brand": "Silver Hills", "category": "bakery"},
            ],
            "chicken": [
                {"name": "Chicken Breast", "brand": "Fresh from Farm", "category": "meat"},
                {"name": "Whole Chicken", "brand": "Maple Leaf", "category": "meat"},
                {"name": "Chicken Thighs", "brand": "President's Choice", "category": "meat"},
            ],
            "apple": [
                {"name": "Gala Apples", "brand": "BC Tree Fruits", "category": "produce"},
                {"name": "Honeycrisp Apples", "brand": "Ontario Grown", "category": "produce"},
                {"name": "Red Delicious Apples", "brand": "Fresh Pick", "category": "produce"},
            ]
        }
        
        # Store-specific pricing variations
        self.store_pricing = {
            "metro_ca": {"multiplier": 1.1, "sale_chance": 0.3},
            "walmart_ca": {"multiplier": 0.9, "sale_chance": 0.2},
            "freshco_com": {"multiplier": 0.95, "sale_chance": 0.4},
        }
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mock scraping for specified stores and query.
        
        Args:
            inputs: Dictionary containing:
                - query: Search query string
                - stores: List of store IDs (optional, defaults to all)
                - limit: Maximum products per store (optional, defaults to 10)
        
        Returns:
            Dictionary with mock scraped products and metadata
        """
        query = inputs.get("query", "").lower()
        store_ids = inputs.get("stores", list(self.store_pricing.keys()))
        limit = inputs.get("limit", 10)
        
        self.log_info(f"Mock scraping for query: '{query}'")
        
        # Find matching product templates
        templates = []
        for keyword, products in self.product_templates.items():
            if keyword in query or any(keyword in word for word in query.split()):
                templates.extend(products)
        
        # If no specific matches, use random selection from all templates
        if not templates:
            templates = [p for products in self.product_templates.values() for p in products]
        
        # Generate products for each store
        all_products = []
        for store_id in store_ids:
            if store_id not in self.store_pricing:
                continue
                
            store_products = self._generate_store_products(
                store_id, templates, min(limit, len(templates))
            )
            all_products.extend(store_products)
            self.log_info(f"Generated {len(store_products)} products for {store_id}")
        
        result = {
            "success": True,
            "query": query,
            "products": all_products,
            "total_products": len(all_products),
            "stores_scraped": len(store_ids),
            "stores_failed": 0,
            "errors": {}
        }
        
        self.log_info(f"Mock scraping completed: {len(all_products)} total products")
        return result
    
    def _generate_store_products(
        self, 
        store_id: str, 
        templates: List[Dict], 
        count: int
    ) -> List[Product]:
        """Generate mock products for a specific store."""
        products = []
        store_config = self.store_pricing[store_id]
        
        # Randomly sample templates
        selected = random.sample(templates, min(count, len(templates)))
        
        for template in selected:
            # Generate base price
            base_price = round(random.uniform(2.0, 25.0), 2)
            price = round(base_price * store_config["multiplier"], 2)
            
            # Determine if on sale
            on_sale = random.random() < store_config["sale_chance"]
            sale_price = None
            if on_sale:
                sale_price = round(price * random.uniform(0.7, 0.9), 2)
            
            # Generate size info randomly
            size = round(random.uniform(0.5, 5.0), 2)
            size_units = [UnitType.LITERS, UnitType.KILOGRAMS, UnitType.GRAMS, UnitType.PIECES]
            size_unit = random.choice(size_units)
            
            # Create product
            product = Product(
                name=template["name"],
                brand=template["brand"],
                price=Decimal(str(price)),
                currency=Currency.CAD,
                size=size,
                size_unit=size_unit,
                store_id=store_id,
                category=template["category"],
                in_stock=random.random() > 0.05,  # 95% chance in stock
                on_sale=on_sale,
                sale_price=Decimal(str(sale_price)) if sale_price else None,
                keywords=self._generate_keywords(template["name"], template["brand"])
            )
            
            products.append(product)
        
        return products
    
    def _generate_keywords(self, name: str, brand: str) -> List[str]:
        """Generate search keywords from product name and brand."""
        keywords = []
        
        # Add name words
        name_words = [w.lower() for w in name.split() if len(w) > 2]
        keywords.extend(name_words)
        
        # Add brand
        if brand:
            keywords.append(brand.lower())
        
        # Remove duplicates
        return list(set(keywords))