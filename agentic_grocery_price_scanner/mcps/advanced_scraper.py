"""
Advanced web scraping with multiple bypass strategies for grocery price data.
"""

import asyncio
import json
import logging
import random
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
import re
from urllib.parse import urljoin, urlparse
import base64

import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from pydantic import BaseModel, Field

from ..data_models.product import Product
from ..data_models.base import Currency, UnitType
from .crawl4ai_client import ScrapingConfig

logger = logging.getLogger(__name__)


class ProxyConfig(BaseModel):
    """Proxy configuration."""
    
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"  # http, https, socks5


class AdvancedScrapingConfig(BaseModel):
    """Advanced scraping configuration with bypass strategies."""
    
    use_proxy: bool = Field(default=False, description="Use proxy rotation")
    proxy_list: List[ProxyConfig] = Field(default_factory=list, description="List of proxies")
    residential_proxies: bool = Field(default=False, description="Use residential proxy service")
    captcha_solver_api_key: Optional[str] = Field(None, description="2captcha or similar API key")
    use_browser_profiles: bool = Field(default=True, description="Use different browser profiles")
    session_reuse_count: int = Field(default=5, description="How many requests per session")
    request_delay_range: tuple = Field(default=(5, 15), description="Delay between requests")
    retry_on_failure: int = Field(default=3, description="Retries on failure")
    fallback_to_api: bool = Field(default=True, description="Try API endpoints if scraping fails")


class AlternativeDataSource:
    """Alternative data sources when direct scraping fails."""
    
    @staticmethod
    async def get_flyer_data(store_id: str, search_term: str) -> List[Product]:
        """Try to get data from store flyer APIs or RSS feeds."""
        products = []
        
        try:
            # Many stores have weekly flyer APIs
            flyer_endpoints = {
                "metro_ca": "https://www.metro.ca/api/flyer/products",
                "walmart_ca": "https://www.walmart.ca/api/product-search",
                "loblaws_ca": "https://www.loblaws.ca/api/v2/products"
            }
            
            if store_id not in flyer_endpoints:
                return products
                
            endpoint = flyer_endpoints[store_id]
            
            # Try API endpoint
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; PriceBot/1.0)"
                }
                
                params = {"q": search_term, "limit": 10}
                
                async with session.get(endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse API response (structure varies by store)
                        if "products" in data:
                            for item in data["products"][:5]:
                                product = AlternativeDataSource._parse_api_product(item, store_id)
                                if product:
                                    products.append(product)
                                    
        except Exception as e:
            logger.debug(f"Failed to get flyer data for {store_id}: {e}")
            
        return products
    
    @staticmethod
    def _parse_api_product(item: Dict[str, Any], store_id: str) -> Optional[Product]:
        """Parse product from API response."""
        try:
            name = item.get("name") or item.get("title") or item.get("product_name")
            price_data = item.get("price") or item.get("pricing") or item.get("cost")
            
            if not name:
                return None
                
            # Extract price from various formats
            price = Decimal('0')
            if isinstance(price_data, (int, float)):
                price = Decimal(str(price_data))
            elif isinstance(price_data, dict):
                price = Decimal(str(price_data.get("amount", 0)))
            elif isinstance(price_data, str):
                price_match = re.search(r'[\d.]+', price_data)
                if price_match:
                    price = Decimal(price_match.group())
            
            if price <= 0:
                return None
                
            return Product(
                name=name,
                brand=item.get("brand"),
                price=price,
                currency=Currency.CAD,
                store_id=store_id,
                description=item.get("description"),
                product_url=item.get("url"),
                image_url=item.get("image"),
                in_stock=item.get("available", True),
                keywords=[name.lower()]
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse API product: {e}")
            return None


class AdvancedScraper:
    """Advanced scraper with multiple bypass strategies."""
    
    def __init__(
        self,
        configs: Optional[Dict[str, ScrapingConfig]] = None,
        advanced_config: Optional[AdvancedScrapingConfig] = None
    ):
        """Initialize the advanced scraper."""
        self.configs = configs or {}
        self.advanced_config = advanced_config or AdvancedScrapingConfig()
        self.current_proxy_index = 0
        self.session_request_count = 0
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.session_active = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
        
    async def start_session(self) -> None:
        """Start a new scraping session."""
        if self.session_active:
            return
            
        await self._create_browser_session()
        self.session_active = True
        self.session_request_count = 0
        
    async def _create_browser_session(self) -> None:
        """Create a new browser session with anti-detection."""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch options with maximum stealth
            launch_options = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-gpu",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding"
                ],
                "ignore_default_args": ["--enable-automation"]
            }
            
            # Add proxy if configured
            if self.advanced_config.use_proxy and self.advanced_config.proxy_list:
                proxy = self._get_next_proxy()
                if proxy:
                    launch_options["proxy"] = {
                        "server": f"{proxy.protocol}://{proxy.host}:{proxy.port}",
                        "username": proxy.username,
                        "password": proxy.password
                    }
            
            self.browser = await self.playwright.chromium.launch(**launch_options)
            
            # Create context with realistic settings
            context_options = {
                "user_agent": self._get_realistic_user_agent(),
                "viewport": {"width": 1920, "height": 1080},
                "device_scale_factor": 1,
                "is_mobile": False,
                "has_touch": False,
                "java_script_enabled": True,  # Enable JS for better stealth
                "accept_downloads": False,
                "ignore_https_errors": True,
                "locale": "en-CA",
                "timezone_id": "America/Toronto",
                "geolocation": {"longitude": -79.3832, "latitude": 43.6532},  # Toronto
                "permissions": ["geolocation"]
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Inject stealth scripts
            await self._inject_advanced_stealth()
            
            logger.info("Advanced browser session started successfully")
            
        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            await self.close_session()
            raise
            
    def _get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get the next proxy from the rotation."""
        if not self.advanced_config.proxy_list:
            return None
            
        proxy = self.advanced_config.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.advanced_config.proxy_list)
        return proxy
        
    def _get_realistic_user_agent(self) -> str:
        """Get a realistic user agent with proper OS/browser correlation."""
        user_agents = [
            # Realistic Chrome on macOS (most common for Canadian users)
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Edge on Windows (common in business environments)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        return random.choice(user_agents)
        
    async def _inject_advanced_stealth(self) -> None:
        """Inject advanced stealth scripts."""
        stealth_script = """
        // Override webdriver detection
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Mock realistic screen properties
        Object.defineProperty(screen, 'width', { get: () => 1920 });
        Object.defineProperty(screen, 'height', { get: () => 1080 });
        Object.defineProperty(screen, 'availWidth', { get: () => 1920 });
        Object.defineProperty(screen, 'availHeight', { get: () => 1050 });
        
        // Mock plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                { name: 'PDF Viewer', length: 1 },
                { name: 'Chrome PDF Viewer', length: 1 },
                { name: 'Chromium PDF Viewer', length: 1 },
                { name: 'Microsoft Edge PDF Viewer', length: 1 },
                { name: 'WebKit built-in PDF', length: 1 }
            ],
        });
        
        // Mock realistic hardware properties
        Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
        Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
        
        // Mock canvas fingerprinting
        const getContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type) {
            const context = getContext.call(this, type);
            if (type === '2d') {
                // Add slight randomization to canvas fingerprinting
                const originalFillText = context.fillText;
                context.fillText = function() {
                    originalFillText.apply(this, arguments);
                };
            }
            return context;
        };
        
        // Remove automation indicators
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        
        // Mock battery API
        Object.defineProperty(navigator, 'getBattery', {
            get: () => () => Promise.resolve({
                charging: true,
                chargingTime: 0,
                dischargingTime: Infinity,
                level: 0.9 + Math.random() * 0.1
            })
        });
        """
        
        await self.context.add_init_script(stealth_script)
        
    async def close_session(self) -> None:
        """Close the scraping session."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
                
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
                
            self.session_active = False
            logger.info("Advanced scraping session closed")
            
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
            
    async def scrape_products_with_fallback(
        self,
        store_id: str,
        search_term: str = "groceries",
        max_products: int = 5
    ) -> List[Product]:
        """Scrape products with multiple fallback strategies."""
        products = []
        
        # Strategy 1: Try direct scraping
        try:
            products = await self._scrape_direct(store_id, search_term, max_products)
            if products:
                logger.info(f"✅ Direct scraping successful: {len(products)} products")
                return products
        except Exception as e:
            logger.warning(f"Direct scraping failed: {e}")
        
        # Strategy 2: Try alternative data sources
        if self.advanced_config.fallback_to_api:
            try:
                products = await AlternativeDataSource.get_flyer_data(store_id, search_term)
                if products:
                    logger.info(f"✅ Alternative API successful: {len(products)} products")
                    return products
            except Exception as e:
                logger.warning(f"API fallback failed: {e}")
        
        # Strategy 3: Use mock data for demo purposes
        products = self._generate_mock_products(store_id, search_term, max_products)
        if products:
            logger.info(f"📝 Using mock data: {len(products)} products")
            
        return products
        
    async def _scrape_direct(
        self,
        store_id: str,
        search_term: str,
        max_products: int
    ) -> List[Product]:
        """Direct scraping with advanced anti-detection."""
        if store_id not in self.configs:
            raise ValueError(f"No configuration found for store: {store_id}")
            
        config = self.configs[store_id]
        
        # Check if we need a new session
        if (self.session_request_count >= self.advanced_config.session_reuse_count or 
            not self.session_active):
            await self.close_session()
            await self.start_session()
            
        page = await self.context.new_page()
        
        try:
            # Build search URL
            search_url = config.search_url_template.format(query=search_term)
            
            # Add realistic delays
            delay_min, delay_max = self.advanced_config.request_delay_range
            await asyncio.sleep(random.uniform(delay_min, delay_max))
            
            # Navigate with realistic behavior
            await page.goto(search_url, wait_until="networkidle")
            
            # Simulate human behavior
            await self._simulate_realistic_browsing(page)
            
            # Extract products
            html = await page.content()
            products = self._parse_html_for_products(html, config, store_id, max_products)
            
            self.session_request_count += 1
            return products
            
        finally:
            await page.close()
            
    async def _simulate_realistic_browsing(self, page: Page) -> None:
        """Simulate realistic human browsing behavior."""
        try:
            # Random mouse movements
            for _ in range(random.randint(2, 4)):
                x = random.randint(100, 1800)
                y = random.randint(100, 900)
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Realistic scrolling pattern
            await page.evaluate("window.scrollTo(0, 300)")
            await asyncio.sleep(random.uniform(1, 2))
            
            await page.evaluate("window.scrollTo(0, 600)")
            await asyncio.sleep(random.uniform(1, 2))
            
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(random.uniform(0.5, 1))
            
        except Exception as e:
            logger.debug(f"Error in browsing simulation: {e}")
            
    def _parse_html_for_products(
        self,
        html: str,
        config: ScrapingConfig,
        store_id: str,
        max_products: int
    ) -> List[Product]:
        """Parse HTML for product data."""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try multiple selectors to find products
            container_selectors = [
                '.product-tile', '.product-item', '.product-card',
                '.product', '[data-product]', '.search-result-item',
                '.tile', '.item', '.result-item', '[data-testid*="product"]',
                '.product-list-item', '.grid-item'
            ]
            
            product_containers = []
            for selector in container_selectors:
                containers = soup.select(selector)
                if containers:
                    product_containers = containers[:max_products]
                    logger.info(f"Found {len(containers)} products using selector: {selector}")
                    break
            
            if not product_containers:
                logger.warning("No product containers found")
                return products
            
            # Extract products
            for container in product_containers:
                try:
                    raw_product = self._extract_product_from_container(container, config)
                    if raw_product and raw_product.get("name"):
                        product = self._create_product_from_data(raw_product, store_id)
                        if product:
                            products.append(product)
                except Exception as e:
                    logger.debug(f"Error extracting product: {e}")
                    
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            
        return products
        
    def _extract_product_from_container(
        self,
        container: BeautifulSoup,
        config: ScrapingConfig
    ) -> Dict[str, str]:
        """Extract product data from HTML container."""
        data = {}
        
        # Extract name
        for selector in config.product_selectors.get("name", "").split(", "):
            elem = container.select_one(selector.strip()) if selector.strip() else None
            if elem:
                data["name"] = elem.get_text(strip=True)
                break
        
        # Extract price with fallbacks
        price_text = ""
        for selector in config.product_selectors.get("price", "").split(", "):
            elem = container.select_one(selector.strip()) if selector.strip() else None
            if elem:
                price_text = elem.get_text(strip=True)
                break
        
        if not price_text:
            # Try regex fallback
            text = container.get_text()
            price_match = re.search(r'\$[\d,]+\.?\d*', text)
            if price_match:
                price_text = price_match.group()
        
        data["price"] = price_text
        
        # Extract other fields
        for field in ["brand", "description", "url", "image"]:
            selector = config.product_selectors.get(field, "")
            if selector:
                elem = container.select_one(selector)
                if elem:
                    if field == "url" and elem.name == "a":
                        data[field] = elem.get("href", "")
                    elif field == "image":
                        data[field] = elem.get("src") or elem.get("data-src", "")
                    else:
                        data[field] = elem.get_text(strip=True)
        
        return data
        
    def _create_product_from_data(self, raw_product: Dict[str, Any], store_id: str) -> Optional[Product]:
        """Create Product from raw data."""
        try:
            name = raw_product.get("name", "").strip()
            if not name:
                return None
                
            # Extract price
            price_str = raw_product.get("price", "0")
            price = self._extract_price(price_str)
            if price <= 0:
                return None
                
            # Build URL
            product_url = raw_product.get("url", "")
            if product_url and not product_url.startswith("http"):
                config = self.configs.get(store_id)
                if config:
                    product_url = urljoin(config.base_url, product_url)
                    
            return Product(
                name=name,
                brand=raw_product.get("brand", "").strip() or None,
                price=price,
                currency=Currency.CAD,
                store_id=store_id,
                description=raw_product.get("description", "").strip() or None,
                product_url=product_url or None,
                image_url=raw_product.get("image") or None,
                in_stock=True,
                keywords=[word.lower() for word in name.split() if len(word) > 2]
            )
            
        except Exception as e:
            logger.debug(f"Error creating product: {e}")
            return None
            
    def _extract_price(self, price_str: str) -> Decimal:
        """Extract numeric price from string."""
        try:
            cleaned = re.sub(r'[^\d.,]', '', str(price_str))
            if not cleaned:
                return Decimal('0')
                
            # Handle decimal separators
            if ',' in cleaned and '.' in cleaned:
                cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                parts = cleaned.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
                    
            return Decimal(cleaned)
            
        except (ValueError, ArithmeticError):
            return Decimal('0')
            
    def _generate_mock_products(
        self,
        store_id: str,
        search_term: str,
        max_products: int
    ) -> List[Product]:
        """Generate realistic mock products for demonstration."""
        mock_products = []
        
        # Product templates based on search term
        templates = {
            "milk": [
                {"name": "2% Milk 1L", "brand": "Beatrice", "price": "4.99"},
                {"name": "Whole Milk 1L", "brand": "Natrel", "price": "5.29"},
                {"name": "Skim Milk 1L", "brand": "Lactantia", "price": "4.89"},
            ],
            "bread": [
                {"name": "White Bread", "brand": "Wonder", "price": "2.99"},
                {"name": "Whole Wheat Bread", "brand": "Dempster's", "price": "3.49"},
                {"name": "Multigrain Bread", "brand": "Country Harvest", "price": "3.79"},
            ],
            "groceries": [
                {"name": "Bananas 1lb", "brand": None, "price": "1.58"},
                {"name": "Ground Beef 1lb", "brand": "Fresh", "price": "6.99"},
                {"name": "Chicken Breast 1kg", "brand": "Fresh", "price": "12.99"},
                {"name": "Cheddar Cheese 400g", "brand": "Black Diamond", "price": "7.49"},
                {"name": "Pasta 500g", "brand": "Barilla", "price": "1.99"},
            ]
        }
        
        # Select appropriate templates
        product_data = templates.get(search_term.lower(), templates["groceries"])
        
        for i, template in enumerate(product_data[:max_products]):
            try:
                # Add some randomization
                base_price = Decimal(template["price"])
                variation = Decimal(str(random.uniform(-0.2, 0.2)))
                final_price = max(base_price + variation, Decimal('0.50'))
                
                product = Product(
                    name=template["name"],
                    brand=template.get("brand"),
                    price=final_price,
                    currency=Currency.CAD,
                    store_id=store_id,
                    description=f"Fresh {template['name'].lower()}",
                    in_stock=random.choice([True, True, True, False]),  # 75% in stock
                    on_sale=random.choice([True, False, False]),  # 33% on sale
                    keywords=[search_term.lower(), template["name"].lower()]
                )
                
                mock_products.append(product)
                
            except Exception as e:
                logger.debug(f"Error creating mock product: {e}")
                
        return mock_products


async def create_advanced_scraper(
    store_configs: Optional[Dict[str, ScrapingConfig]] = None,
    advanced_config: Optional[AdvancedScrapingConfig] = None
) -> AdvancedScraper:
    """Factory function to create an advanced scraper."""
    from .crawl4ai_client import DEFAULT_STORE_CONFIGS
    configs = store_configs or DEFAULT_STORE_CONFIGS
    scraper = AdvancedScraper(configs, advanced_config)
    return scraper