"""
Web scraping client for grocery store data using requests and BeautifulSoup.
Compatible with Python 3.9+.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any
import re
import random
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ..data_models.product import Product
from ..data_models.base import Currency, UnitType


logger = logging.getLogger(__name__)


class ScrapingConfig(BaseModel):
    """Configuration for scraping a grocery store."""
    
    store_id: str = Field(..., description="Unique store identifier")
    base_url: str = Field(..., description="Base URL for the store")
    search_url_template: str = Field(..., description="URL template for product search")
    product_selectors: Dict[str, str] = Field(..., description="CSS selectors for product data")
    rate_limit_delay: float = Field(default=1.0, description="Delay between requests in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    user_agent: Optional[str] = Field(None, description="Custom user agent")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")


class WebScrapingClient:
    """Async web scraping client using aiohttp and BeautifulSoup."""
    
    def __init__(self, configs: Optional[Dict[str, ScrapingConfig]] = None):
        """Initialize the web scraping client with store configurations."""
        self.configs = configs or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_active = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
        
    async def start_session(self) -> None:
        """Start the aiohttp session."""
        if self.session_active:
            return
            
        try:
            # Create session with reasonable timeout and user agent
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            self.session_active = True
            logger.info("Web scraping session started successfully")
        except Exception as e:
            logger.error(f"Failed to start web scraping session: {e}")
            raise
            
    async def close_session(self) -> None:
        """Close the aiohttp session."""
        if self.session and self.session_active:
            await self.session.close()
            self.session_active = False
            logger.info("Web scraping session closed")
            
    async def scrape_url(
        self, 
        url: str, 
        wait_time: float = 2.0
    ) -> Dict[str, Any]:
        """Scrape a URL and return the result."""
        if not self.session_active:
            await self.start_session()
            
        try:
            # Add delay to be respectful
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    return {
                        "success": True,
                        "html": html,
                        "status": response.status,
                        "url": str(response.url)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "status": response.status,
                        "html": None
                    }
                    
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "html": None
            }
            
    async def scrape_products(
        self, 
        store_id: str, 
        search_term: str = "groceries",
        max_products: int = 5
    ) -> List[Product]:
        """Scrape products from a specific store."""
        if store_id not in self.configs:
            raise ValueError(f"No configuration found for store: {store_id}")
            
        config = self.configs[store_id]
        products = []
        
        try:
            # Build search URL
            search_url = config.search_url_template.format(query=search_term)
            logger.info(f"Scraping products from: {search_url}")
            
            # Apply rate limiting
            await asyncio.sleep(config.rate_limit_delay)
            
            # Scrape the search results page
            result = await self.scrape_url(search_url)
            
            if not result.get("success"):
                logger.error(f"Failed to scrape {store_id}: {result.get('error', 'Unknown error')}")
                return products
                
            # Parse HTML with BeautifulSoup
            html = result.get("html")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to find product containers using multiple selectors
                product_containers = []
                container_selectors = [
                    '.product-tile', '.product-item', '.product-card', 
                    '.product', '[data-product]', '.search-result-item'
                ]
                
                for selector in container_selectors:
                    containers = soup.select(selector)
                    if containers:
                        product_containers = containers[:max_products]
                        logger.info(f"Found {len(containers)} products using selector: {selector}")
                        break
                
                if not product_containers:
                    logger.warning(f"No product containers found for {store_id}")
                    return products
                
                # Extract product data from each container
                for container in product_containers:
                    try:
                        raw_product = self._extract_product_data(container, config)
                        if raw_product:
                            product = self._parse_product(raw_product, store_id)
                            if product:
                                products.append(product)
                    except Exception as e:
                        logger.warning(f"Failed to extract product data: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to scrape products from {store_id}: {e}")
            
        logger.info(f"Successfully scraped {len(products)} products from {store_id}")
        return products
        
    def _extract_product_data(self, container: BeautifulSoup, config: ScrapingConfig) -> Dict[str, str]:
        """Extract product data from a BeautifulSoup container."""
        data = {}
        
        selectors = config.product_selectors
        
        # Extract name
        name_elem = container.select_one(selectors.get("name", ""))
        data["name"] = name_elem.get_text(strip=True) if name_elem else ""
        
        # Extract price
        price_elem = container.select_one(selectors.get("price", ""))
        data["price"] = price_elem.get_text(strip=True) if price_elem else "0"
        
        # Extract URL
        url_elem = container.select_one(selectors.get("url", ""))
        if url_elem:
            if url_elem.name == "a":
                data["url"] = url_elem.get("href", "")
            else:
                # Look for an anchor tag within the element
                link = url_elem.find("a")
                data["url"] = link.get("href", "") if link else ""
        else:
            data["url"] = ""
        
        # Extract image URL
        img_elem = container.select_one(selectors.get("image", ""))
        if img_elem:
            data["image"] = img_elem.get("src") or img_elem.get("data-src", "")
        else:
            data["image"] = ""
        
        # Extract brand
        brand_elem = container.select_one(selectors.get("brand", ""))
        data["brand"] = brand_elem.get_text(strip=True) if brand_elem else ""
        
        # Extract description
        desc_elem = container.select_one(selectors.get("description", ""))
        data["description"] = desc_elem.get_text(strip=True) if desc_elem else ""
        
        return data
        
    def _parse_product(self, raw_product: Dict[str, Any], store_id: str) -> Optional[Product]:
        """Parse raw product data into a Product model."""
        try:
            # Extract and clean price
            price_str = raw_product.get("price", "0")
            price = self._extract_price(price_str)
            
            if price <= 0:
                return None  # Skip products with invalid prices
                
            # Extract product name
            name = raw_product.get("name", "").strip()
            if not name:
                return None  # Skip products without names
                
            # Build product URL
            product_url = raw_product.get("url", "")
            if product_url and not product_url.startswith("http"):
                config = self.configs.get(store_id)
                if config:
                    product_url = config.base_url.rstrip("/") + "/" + product_url.lstrip("/")
                    
            product = Product(
                name=name,
                brand=raw_product.get("brand", "").strip() or None,
                price=price,
                currency=Currency.CAD,
                store_id=store_id,
                description=raw_product.get("description", "").strip() or None,
                product_url=product_url or None,
                image_url=raw_product.get("image") or None,
                in_stock=True,  # Assume in stock if listed
                keywords=self._generate_keywords(name, raw_product.get("brand", ""))
            )
            
            return product
            
        except Exception as e:
            logger.error(f"Failed to parse product: {e}")
            return None
            
    def _extract_price(self, price_str: str) -> Decimal:
        """Extract price from various string formats."""
        try:
            # Remove currency symbols and extra whitespace
            cleaned = re.sub(r'[^\d.,]', '', str(price_str))
            
            # Handle different decimal separators
            if ',' in cleaned and '.' in cleaned:
                # Assume comma is thousands separator if both present
                cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # Check if comma is decimal separator
                parts = cleaned.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
                    
            return Decimal(cleaned) if cleaned else Decimal('0')
            
        except (ValueError, ArithmeticError):
            return Decimal('0')
            
    def _generate_keywords(self, name: str, brand: str = "") -> List[str]:
        """Generate search keywords from product name and brand."""
        keywords = []
        
        # Add name words
        name_words = re.findall(r'\w+', name.lower())
        keywords.extend(name_words)
        
        # Add brand words
        if brand:
            brand_words = re.findall(r'\w+', brand.lower())
            keywords.extend(brand_words)
            
        # Add full name and brand as keywords
        keywords.append(name.lower())
        if brand:
            keywords.append(brand.lower())
            
        # Remove duplicates and short words
        keywords = list(set(word for word in keywords if len(word) > 2))
        
        return keywords
        
    def add_store_config(self, store_id: str, config: ScrapingConfig) -> None:
        """Add or update a store configuration."""
        self.configs[store_id] = config
        logger.info(f"Added configuration for store: {store_id}")
        
    def get_available_stores(self) -> List[str]:
        """Get list of configured store IDs."""
        return list(self.configs.keys())


# Default store configurations
DEFAULT_STORE_CONFIGS = {
    "metro_ca": ScrapingConfig(
        store_id="metro_ca",
        base_url="https://www.metro.ca",
        search_url_template="https://www.metro.ca/en/online-grocery/search?filter={query}",
        product_selectors={
            "name": "h3, .product-name, [data-testid='product-name'], .tile-product-name",
            "price": ".price, .product-price, [data-testid='product-price'], .tile-product-price",
            "url": "a[href*='/product/'], .product-link, [data-testid='product-link']",
            "image": "img[src*='product'], .product-image img, [data-testid='product-image'] img",
            "brand": ".product-brand, [data-testid='product-brand'], .tile-product-brand",
            "description": ".product-description, [data-testid='product-description']"
        },
        rate_limit_delay=2.0,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-CA,en;q=0.5",
        }
    ),
    "walmart_ca": ScrapingConfig(
        store_id="walmart_ca",
        base_url="https://www.walmart.ca",
        search_url_template="https://www.walmart.ca/search?q={query}",
        product_selectors={
            "name": "[data-automation-id='product-title'], .product-title",
            "price": "[data-automation-id='product-price'] .price, .price-current",
            "url": "[data-automation-id='product-title']",
            "image": ".product-image img, [data-automation-id='product-image'] img",
            "brand": ".product-brand, [data-automation-id='product-brand']",
            "description": ".product-description"
        },
        rate_limit_delay=2.0,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-CA,en;q=0.5",
        }
    ),
    "freshco_ca": ScrapingConfig(
        store_id="freshco_ca",
        base_url="https://www.freshco.com",
        search_url_template="https://www.freshco.com/search?q={query}",
        product_selectors={
            "name": ".product-name, .product-title",
            "price": ".product-price .price, .price-current",
            "url": ".product-link, .product-name",
            "image": ".product-image img",
            "brand": ".product-brand",
            "description": ".product-description"
        },
        rate_limit_delay=1.5,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-CA,en;q=0.5",
        }
    )
}


async def create_crawl4ai_client(
    store_configs: Optional[Dict[str, ScrapingConfig]] = None
) -> WebScrapingClient:
    """Factory function to create a web scraping client with default or custom configurations."""
    configs = store_configs or DEFAULT_STORE_CONFIGS
    client = WebScrapingClient(configs)
    return client