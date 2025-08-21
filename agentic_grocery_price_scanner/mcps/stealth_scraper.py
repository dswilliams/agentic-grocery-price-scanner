"""
Advanced stealth web scraping client with anti-bot detection bypass.
"""

import asyncio
import json
import logging
import random
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any
import re
from urllib.parse import urljoin
import os

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ..data_models.product import Product
from ..data_models.base import Currency, UnitType
from .crawl4ai_client import ScrapingConfig

logger = logging.getLogger(__name__)


class StealthConfig(BaseModel):
    """Configuration for stealth scraping."""
    
    headless: bool = Field(default=True, description="Run browser in headless mode")
    browser_type: str = Field(default="chromium", description="Browser type to use")
    viewport_width: int = Field(default=1920, description="Browser viewport width")
    viewport_height: int = Field(default=1080, description="Browser viewport height")
    user_data_dir: Optional[str] = Field(None, description="Browser user data directory")
    enable_stealth: bool = Field(default=True, description="Enable stealth mode")
    navigation_timeout: int = Field(default=30000, description="Navigation timeout in ms")
    page_load_delay: tuple = Field(default=(2, 5), description="Random delay range after page load")
    scroll_delay: tuple = Field(default=(1, 3), description="Random delay range between scrolls")
    typing_delay: tuple = Field(default=(50, 150), description="Random delay range between keystrokes")


class UserAgentRotator:
    """Rotates user agents to avoid detection."""
    
    def __init__(self):
        self.user_agents = [
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            
            # Firefox
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.user_agents)
    
    def get_matching_headers(self, user_agent: str) -> Dict[str, str]:
        """Get headers that match the user agent."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-CA,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Add browser-specific headers
        if "Chrome" in user_agent:
            headers["sec-ch-ua"] = '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'
            headers["sec-ch-ua-mobile"] = "?0"
            headers["sec-ch-ua-platform"] = '"macOS"' if "Mac" in user_agent else '"Windows"'
        
        return headers


class StealthScraper:
    """Advanced stealth web scraping client using Playwright."""
    
    def __init__(
        self, 
        configs: Optional[Dict[str, ScrapingConfig]] = None,
        stealth_config: Optional[StealthConfig] = None
    ):
        """Initialize the stealth scraper."""
        self.configs = configs or {}
        self.stealth_config = stealth_config or StealthConfig()
        self.user_agent_rotator = UserAgentRotator()
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
        """Start the Playwright browser session with stealth mode."""
        if self.session_active:
            return
            
        try:
            self.playwright = await async_playwright().start()
            
            # Browser launch options
            launch_options = {
                "headless": self.stealth_config.headless,
                "args": [
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",  # Faster loading
                    "--disable-javascript",  # We don't need JS for scraping
                ],
                "ignore_default_args": ["--enable-automation"]
            }
            
            # Add user data directory if specified
            if self.stealth_config.user_data_dir:
                launch_options["user_data_dir"] = self.stealth_config.user_data_dir
            
            # Launch browser
            if self.stealth_config.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif self.stealth_config.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(**launch_options)
            else:
                self.browser = await self.playwright.chromium.launch(**launch_options)
            
            # Create context with stealth settings
            user_agent = self.user_agent_rotator.get_random_user_agent()
            headers = self.user_agent_rotator.get_matching_headers(user_agent)
            
            context_options = {
                "user_agent": user_agent,
                "viewport": {
                    "width": self.stealth_config.viewport_width,
                    "height": self.stealth_config.viewport_height
                },
                "extra_http_headers": headers,
                "ignore_https_errors": True,
                "java_script_enabled": False,  # Disable JS for faster scraping
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Add stealth scripts to bypass detection
            if self.stealth_config.enable_stealth:
                await self._inject_stealth_scripts()
            
            self.session_active = True
            logger.info(f"Stealth browser session started with {self.stealth_config.browser_type}")
            logger.info(f"User Agent: {user_agent[:80]}...")
            
        except Exception as e:
            logger.error(f"Failed to start stealth browser session: {e}")
            await self.close_session()
            raise
            
    async def _inject_stealth_scripts(self) -> None:
        """Inject stealth scripts to bypass bot detection."""
        stealth_script = """
        // Remove webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Mock plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Mock languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-CA', 'en-US', 'en'],
        });
        
        // Mock permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Hide automation indicators
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        """
        
        await self.context.add_init_script(stealth_script)
        
    async def close_session(self) -> None:
        """Close the browser session."""
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
            logger.info("Stealth browser session closed")
            
        except Exception as e:
            logger.warning(f"Error closing browser session: {e}")
            
    async def scrape_url(self, url: str, wait_for_selector: str = "body") -> Dict[str, Any]:
        """Scrape a URL with stealth techniques."""
        if not self.session_active:
            await self.start_session()
            
        page: Page = None
        try:
            page = await self.context.new_page()
            
            # Set navigation timeout
            page.set_default_navigation_timeout(self.stealth_config.navigation_timeout)
            
            # Navigate with realistic timing
            await page.goto(url, wait_until="domcontentloaded")
            
            # Wait for page to load
            try:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            except:
                logger.warning(f"Selector '{wait_for_selector}' not found, continuing anyway")
            
            # Add realistic delays
            delay_min, delay_max = self.stealth_config.page_load_delay
            await asyncio.sleep(random.uniform(delay_min, delay_max))
            
            # Simulate human behavior - scroll and move mouse
            await self._simulate_human_behavior(page)
            
            # Get page content
            html = await page.content()
            
            return {
                "success": True,
                "html": html,
                "url": page.url,
                "title": await page.title()
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "html": None
            }
        finally:
            if page:
                await page.close()
                
    async def _simulate_human_behavior(self, page: Page) -> None:
        """Simulate human-like behavior on the page."""
        try:
            # Get page dimensions
            viewport = page.viewport_size
            if not viewport:
                return
                
            # Random mouse movements
            for _ in range(random.randint(1, 3)):
                x = random.randint(0, viewport["width"])
                y = random.randint(0, viewport["height"])
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Random scrolling
            scroll_count = random.randint(1, 3)
            for _ in range(scroll_count):
                scroll_y = random.randint(100, 500)
                await page.evaluate(f"window.scrollBy(0, {scroll_y})")
                
                delay_min, delay_max = self.stealth_config.scroll_delay
                await asyncio.sleep(random.uniform(delay_min, delay_max))
            
            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            logger.debug(f"Error simulating human behavior: {e}")
            
    async def scrape_products(
        self,
        store_id: str,
        search_term: str = "groceries", 
        max_products: int = 5
    ) -> List[Product]:
        """Scrape products from a store using stealth techniques."""
        if store_id not in self.configs:
            raise ValueError(f"No configuration found for store: {store_id}")
            
        config = self.configs[store_id]
        products = []
        
        try:
            # Build search URL
            search_url = config.search_url_template.format(query=search_term)
            logger.info(f"Stealth scraping products from: {search_url}")
            
            # Apply rate limiting
            await asyncio.sleep(config.rate_limit_delay)
            
            # Scrape the search results page
            result = await self.scrape_url(search_url)
            
            if not result.get("success"):
                logger.error(f"Failed to scrape {store_id}: {result.get('error')}")
                return products
                
            # Parse HTML with BeautifulSoup
            html = result.get("html")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to find product containers using multiple selectors
                product_containers = []
                container_selectors = [
                    '.product-tile', '.product-item', '.product-card',
                    '.product', '[data-product]', '.search-result-item',
                    '.tile', '.item', '.result-item', '[data-testid*="product"]'
                ]
                
                for selector in container_selectors:
                    containers = soup.select(selector)
                    if containers:
                        product_containers = containers[:max_products]
                        logger.info(f"Found {len(containers)} products using selector: {selector}")
                        break
                
                if not product_containers:
                    logger.warning(f"No product containers found for {store_id}")
                    # Log page title and some content for debugging
                    title = result.get("title", "No title")
                    logger.info(f"Page title: {title}")
                    
                    # Check if we hit a CAPTCHA or login page
                    if any(keyword in html.lower() for keyword in ['captcha', 'verify', 'robot', 'blocked']):
                        logger.warning("Detected CAPTCHA or bot detection page")
                    
                    return products
                
                # Extract product data from each container
                for i, container in enumerate(product_containers):
                    try:
                        raw_product = self._extract_product_data(container, config)
                        if raw_product and raw_product.get("name"):
                            product = self._parse_product(raw_product, store_id)
                            if product:
                                products.append(product)
                                logger.debug(f"Extracted product {i+1}: {product.name}")
                    except Exception as e:
                        logger.warning(f"Failed to extract product data from container {i+1}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to scrape products from {store_id}: {e}")
            
        logger.info(f"Successfully scraped {len(products)} products from {store_id}")
        return products
        
    def _extract_product_data(self, container, config: ScrapingConfig) -> Dict[str, str]:
        """Extract product data from a BeautifulSoup container."""
        data = {}
        selectors = config.product_selectors
        
        # Extract name with multiple fallbacks
        name_selectors = selectors.get("name", "").split(", ")
        for selector in name_selectors:
            if selector:
                name_elem = container.select_one(selector.strip())
                if name_elem:
                    data["name"] = name_elem.get_text(strip=True)
                    break
        
        if not data.get("name"):
            # Try common fallbacks
            for fallback in ["h1", "h2", "h3", "h4", ".title", "[data-name]", ".name"]:
                elem = container.select_one(fallback)
                if elem and elem.get_text(strip=True):
                    data["name"] = elem.get_text(strip=True)
                    break
        
        # Extract price with multiple fallbacks
        price_selectors = selectors.get("price", "").split(", ")
        for selector in price_selectors:
            if selector:
                price_elem = container.select_one(selector.strip())
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    if price_text and any(char.isdigit() for char in price_text):
                        data["price"] = price_text
                        break
        
        if not data.get("price"):
            # Try common price patterns
            for pattern in [r'\$[\d,]+\.?\d*', r'[\d,]+\.?\d*']:
                price_match = re.search(pattern, container.get_text())
                if price_match:
                    data["price"] = price_match.group()
                    break
        
        # Extract other fields
        for field, selector in selectors.items():
            if field not in ["name", "price"] and selector:
                elem = container.select_one(selector)
                if elem:
                    if field == "url" and elem.name == "a":
                        data[field] = elem.get("href", "")
                    elif field == "image":
                        data[field] = elem.get("src") or elem.get("data-src", "")
                    else:
                        data[field] = elem.get_text(strip=True)
        
        return data
        
    def _parse_product(self, raw_product: Dict[str, Any], store_id: str) -> Optional[Product]:
        """Parse raw product data into a Product model."""
        try:
            # Extract and clean price
            price_str = raw_product.get("price", "0")
            price = self._extract_price(price_str)
            
            if price <= 0:
                logger.debug(f"Skipping product with invalid price: {price_str}")
                return None
                
            # Extract product name
            name = raw_product.get("name", "").strip()
            if not name:
                logger.debug("Skipping product without name")
                return None
                
            # Build product URL
            product_url = raw_product.get("url", "")
            if product_url and not product_url.startswith("http"):
                config = self.configs.get(store_id)
                if config:
                    product_url = urljoin(config.base_url, product_url)
                    
            product = Product(
                name=name,
                brand=raw_product.get("brand", "").strip() or None,
                price=price,
                currency=Currency.CAD,
                store_id=store_id,
                description=raw_product.get("description", "").strip() or None,
                product_url=product_url or None,
                image_url=raw_product.get("image") or None,
                in_stock=True,
                keywords=self._generate_keywords(name, raw_product.get("brand", ""))
            )
            
            return product
            
        except Exception as e:
            logger.debug(f"Failed to parse product: {e}")
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


async def create_stealth_scraper(
    store_configs: Optional[Dict[str, ScrapingConfig]] = None,
    stealth_config: Optional[StealthConfig] = None
) -> StealthScraper:
    """Factory function to create a stealth scraper."""
    from .crawl4ai_client import DEFAULT_STORE_CONFIGS
    configs = store_configs or DEFAULT_STORE_CONFIGS
    scraper = StealthScraper(configs, stealth_config)
    return scraper