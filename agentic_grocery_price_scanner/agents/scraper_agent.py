"""
Web scraper agent for extracting product data from grocery stores.
"""

import asyncio
import re
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..data_models import Product, StoreConfig
from ..config import load_store_configs
from .base_agent import BaseAgent


class ScraperAgent(BaseAgent):
    """Agent responsible for scraping product data from grocery stores."""
    
    def __init__(self):
        """Initialize the scraper agent."""
        super().__init__("scraper")
        self.session = requests.Session()
        self.scraped_products = []
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scraping for specified stores and query.
        
        Args:
            inputs: Dictionary containing:
                - query: Search query string
                - stores: List of store IDs (optional, defaults to all active)
                - limit: Maximum products per store (optional, defaults to 50)
        
        Returns:
            Dictionary with scraped products and metadata
        """
        query = inputs.get("query")
        if not query:
            raise ValueError("Query is required for scraping")
        
        store_ids = inputs.get("stores", [])
        limit = inputs.get("limit", 50)
        
        self.log_info(f"Starting scrape for query: '{query}'")
        
        # Load store configurations
        try:
            all_stores = load_store_configs()
        except Exception as e:
            self.log_error(f"Failed to load store configs: {e}", e)
            return {"success": False, "error": str(e)}
        
        # Filter to requested stores or use all active
        if store_ids:
            stores = {k: v for k, v in all_stores.items() if k in store_ids}
        else:
            stores = {k: v for k, v in all_stores.items() if getattr(v, 'active', True)}
        
        if not stores:
            self.log_error("No valid stores found for scraping")
            return {"success": False, "error": "No valid stores"}
        
        self.log_info(f"Scraping {len(stores)} stores: {list(stores.keys())}")
        
        # Scrape each store
        all_products = []
        errors = {}
        
        for store_id, store_config in stores.items():
            try:
                self.log_info(f"Scraping {store_config.name} ({store_id})")
                products = await self._scrape_store(store_config, query, limit)
                all_products.extend(products)
                self.log_info(f"Found {len(products)} products from {store_config.name}")
            except Exception as e:
                error_msg = f"Failed to scrape {store_id}: {e}"
                self.log_error(error_msg, e)
                errors[store_id] = str(e)
        
        self.scraped_products = all_products
        
        result = {
            "success": True,
            "query": query,
            "products": all_products,
            "total_products": len(all_products),
            "stores_scraped": len(stores) - len(errors),
            "stores_failed": len(errors),
            "errors": errors
        }
        
        self.log_info(f"Scraping completed: {len(all_products)} total products")
        return result
    
    async def _scrape_store(
        self, 
        store_config: StoreConfig, 
        query: str, 
        limit: int
    ) -> List[Product]:
        """Scrape products from a specific store."""
        # Prepare search URL
        search_url = store_config.search_url_template.format(query=query)
        self.log_debug(f"Scraping URL: {search_url}")
        
        # Set up headers
        headers = store_config.headers or {}
        headers.setdefault("User-Agent", self.settings.scraping.user_agent)
        
        # Make request with rate limiting
        try:
            time.sleep(store_config.rate_limit_seconds)
            
            response = self.session.get(
                search_url,
                headers=headers,
                timeout=store_config.timeout_seconds
            )
            response.raise_for_status()
            
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract products using CSS selectors
        products = []
        selectors = store_config.selectors
        
        if not selectors.get('product_container'):
            raise Exception("No product container selector configured")
        
        product_containers = soup.select(selectors['product_container'])
        self.log_debug(f"Found {len(product_containers)} product containers")
        
        for i, container in enumerate(product_containers[:limit]):
            try:
                product = self._extract_product_data(
                    container, store_config, selectors
                )
                if product:
                    products.append(product)
            except Exception as e:
                self.log_debug(f"Failed to extract product {i}: {e}")
                continue
        
        return products
    
    def _extract_product_data(
        self, 
        container, 
        store_config: StoreConfig,
        selectors: Dict[str, str]
    ) -> Optional[Product]:
        """Extract product data from a container element."""
        try:
            # Extract name
            name_element = container.select_one(selectors.get('product_name', ''))
            if not name_element:
                return None
            name = self._clean_text(name_element.get_text())
            
            # Extract price
            price_element = container.select_one(selectors.get('price', ''))
            if not price_element:
                return None
            price_text = self._clean_text(price_element.get_text())
            price = self._extract_price(price_text)
            
            if not price:
                return None
            
            # Extract optional fields
            brand = None
            if selectors.get('brand'):
                brand_element = container.select_one(selectors['brand'])
                if brand_element:
                    brand = self._clean_text(brand_element.get_text())
            
            image_url = None
            if selectors.get('image'):
                img_element = container.select_one(selectors['image'])
                if img_element:
                    image_url = img_element.get('src') or img_element.get('data-src')
                    if image_url and not image_url.startswith('http'):
                        image_url = urljoin(str(store_config.base_url), image_url)
            
            product_url = None
            if selectors.get('product_link'):
                link_element = container.select_one(selectors['product_link'])
                if link_element:
                    product_url = link_element.get('href')
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(str(store_config.base_url), product_url)
            
            # Check stock status
            in_stock = True
            if selectors.get('out_of_stock'):
                out_of_stock_element = container.select_one(selectors['out_of_stock'])
                if out_of_stock_element:
                    in_stock = False
            
            # Check sale status
            on_sale = False
            if selectors.get('on_sale'):
                sale_element = container.select_one(selectors['on_sale'])
                if sale_element:
                    on_sale = True
            
            # Create product
            product = Product(
                name=name,
                brand=brand,
                price=price,
                currency=store_config.currency,
                store_id=store_config.store_id,
                image_url=image_url,
                product_url=product_url,
                in_stock=in_stock,
                on_sale=on_sale,
                keywords=self._generate_keywords(name, brand)
            )
            
            return product
            
        except Exception as e:
            self.log_debug(f"Error extracting product data: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        return ' '.join(text.strip().split())
    
    def _extract_price(self, price_text: str) -> Optional[Decimal]:
        """Extract price from text using regex."""
        if not price_text:
            return None
        
        # Remove currency symbols and clean
        cleaned = re.sub(r'[^\d.,]', '', price_text)
        
        # Match decimal price patterns
        price_patterns = [
            r'(\d+\.\d{2})',  # 5.99
            r'(\d+,\d{2})',   # 5,99 (European format)
            r'(\d+)',         # 5 (whole number)
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, cleaned)
            if match:
                price_str = match.group(1).replace(',', '.')
                try:
                    return Decimal(price_str)
                except:
                    continue
        
        return None
    
    def _generate_keywords(self, name: str, brand: Optional[str] = None) -> List[str]:
        """Generate search keywords from product name and brand."""
        keywords = []
        
        if name:
            # Add full name
            keywords.append(name.lower())
            
            # Add individual words (3+ characters)
            words = re.findall(r'\b\w{3,}\b', name.lower())
            keywords.extend(words)
        
        if brand:
            keywords.append(brand.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Limit to 10 keywords