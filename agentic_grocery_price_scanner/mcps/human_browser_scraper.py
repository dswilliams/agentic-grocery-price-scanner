"""
Human-assisted browser scraping using your actual browser profile and manual interaction.
This leverages your existing browser sessions, cookies, and login state.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import re

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ..data_models.product import Product
from ..data_models.base import Currency, UnitType
from .crawl4ai_client import ScrapingConfig

logger = logging.getLogger(__name__)


class BrowserProfile(BaseModel):
    """Configuration for using existing browser profiles."""
    
    browser_type: str = Field(default="chrome", description="chrome, firefox, edge, safari")
    profile_path: Optional[str] = Field(None, description="Path to browser profile directory")
    user_data_dir: Optional[str] = Field(None, description="Chrome user data directory")
    auto_detect: bool = Field(default=True, description="Auto-detect default browser profile")


class ManualScrapingStep(BaseModel):
    """A step in the manual scraping workflow."""
    
    step_id: str
    instruction: str
    wait_for: str  # "user_input", "element", "navigation", "time"
    selector: Optional[str] = None
    timeout: int = 60
    required: bool = True


class HumanBrowserScraper:
    """Browser scraper that uses your existing browser profile and provides manual assistance."""
    
    def __init__(
        self, 
        configs: Optional[Dict[str, ScrapingConfig]] = None,
        browser_profile: Optional[BrowserProfile] = None
    ):
        """Initialize the human-assisted browser scraper."""
        self.configs = configs or {}
        self.browser_profile = browser_profile or BrowserProfile()
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.session_active = False
        self.user_data_dir = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
        
    def _detect_browser_profile(self) -> Dict[str, str]:
        """Auto-detect your default browser profile paths."""
        system = platform.system()
        home = Path.home()
        
        profiles = {}
        
        if system == "Darwin":  # macOS
            profiles.update({
                "chrome": str(home / "Library/Application Support/Google/Chrome/Default"),
                "chrome_user_data": str(home / "Library/Application Support/Google/Chrome"),
                "firefox": str(home / "Library/Application Support/Firefox/Profiles"),
                "edge": str(home / "Library/Application Support/Microsoft Edge/Default"),
                "safari": str(home / "Library/Safari")
            })
        elif system == "Windows":
            profiles.update({
                "chrome": str(home / "AppData/Local/Google/Chrome/User Data/Default"),
                "chrome_user_data": str(home / "AppData/Local/Google/Chrome/User Data"),
                "firefox": str(home / "AppData/Roaming/Mozilla/Firefox/Profiles"),
                "edge": str(home / "AppData/Local/Microsoft/Edge/User Data/Default")
            })
        elif system == "Linux":
            profiles.update({
                "chrome": str(home / ".config/google-chrome/Default"),
                "chrome_user_data": str(home / ".config/google-chrome"),
                "firefox": str(home / ".mozilla/firefox"),
                "edge": str(home / ".config/microsoft-edge/Default")
            })
            
        # Filter to only existing paths
        return {k: v for k, v in profiles.items() if Path(v).exists()}
        
    async def start_session(self) -> None:
        """Start a browser session using your actual browser profile."""
        if self.session_active:
            return
            
        try:
            self.playwright = await async_playwright().start()
            
            # Detect available browser profiles
            available_profiles = self._detect_browser_profile()
            logger.info(f"Available browser profiles: {list(available_profiles.keys())}")
            
            # Determine browser and profile to use
            browser_type = self.browser_profile.browser_type.lower()
            
            if self.browser_profile.user_data_dir:
                user_data_dir = self.browser_profile.user_data_dir
            elif f"{browser_type}_user_data" in available_profiles:
                user_data_dir = available_profiles[f"{browser_type}_user_data"]
            elif browser_type in available_profiles:
                user_data_dir = str(Path(available_profiles[browser_type]).parent)
            else:
                logger.warning(f"No profile found for {browser_type}, using temporary profile")
                user_data_dir = None
                
            self.user_data_dir = user_data_dir
            
            # Launch options with your profile
            launch_options = {
                "headless": False,  # Always visible for manual interaction
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                "ignore_default_args": ["--enable-automation"]
            }
            
            # Add user data directory
            if user_data_dir and Path(user_data_dir).exists():
                launch_options["user_data_dir"] = user_data_dir
                logger.info(f"Using browser profile: {user_data_dir}")
            
            # Launch the appropriate browser
            if browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif browser_type == "edge":
                self.browser = await self.playwright.chromium.launch(
                    channel="msedge", **launch_options
                )
            else:  # Default to Chrome/Chromium
                self.browser = await self.playwright.chromium.launch(**launch_options)
            
            # Create context (this inherits your existing sessions, cookies, etc.)
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-CA",
                timezone_id="America/Toronto"
            )
            
            self.session_active = True
            logger.info("✅ Human-assisted browser session started using your profile!")
            logger.info("🔑 Your existing cookies, sessions, and login state are available")
            
        except Exception as e:
            logger.error(f"Failed to start human browser session: {e}")
            await self.close_session()
            raise
            
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
            logger.info("Human browser session closed")
            
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
            
    async def scrape_with_human_assistance(
        self,
        store_id: str,
        search_term: str = "groceries",
        max_products: int = 5
    ) -> List[Product]:
        """Scrape products with human assistance and guidance."""
        if not self.session_active:
            await self.start_session()
            
        if store_id not in self.configs:
            raise ValueError(f"No configuration found for store: {store_id}")
            
        config = self.configs[store_id]
        
        logger.info(f"\n🤖 Starting human-assisted scraping for {store_id}")
        logger.info(f"🎯 Search term: '{search_term}' (max {max_products} products)")
        logger.info("=" * 60)
        
        page = await self.context.new_page()
        
        try:
            # Define the manual workflow
            workflow = self._create_scraping_workflow(config, search_term)
            
            products = []
            
            for step in workflow:
                logger.info(f"\n📍 STEP: {step.instruction}")
                
                success = await self._execute_manual_step(page, step)
                
                if not success and step.required:
                    logger.error(f"❌ Required step failed: {step.step_id}")
                    break
                    
            # Extract products after workflow completion
            logger.info(f"\n🔍 Analyzing page for products...")
            html = await page.content()
            products = await self._extract_products_with_guidance(
                page, html, config, store_id, max_products
            )
            
            if products:
                logger.info(f"\n✅ SUCCESS! Found {len(products)} products:")
                for i, product in enumerate(products, 1):
                    logger.info(f"   {i}. {product.name} - ${product.price} ({product.brand or 'No brand'})")
            else:
                logger.warning(f"\n⚠️  No products extracted. Let's try manual selection...")
                products = await self._manual_product_selection(page, store_id, max_products)
            
            return products
            
        finally:
            # Keep the page open for manual review
            logger.info(f"\n🔧 Browser page left open for manual review")
            logger.info("💡 You can manually adjust the data or close when done")
            
    def _create_scraping_workflow(self, config: ScrapingConfig, search_term: str) -> List[ManualScrapingStep]:
        """Create a workflow for manual scraping assistance."""
        search_url = config.search_url_template.format(query=search_term)
        
        return [
            ManualScrapingStep(
                step_id="navigate",
                instruction=f"🌐 Navigating to: {search_url}",
                wait_for="navigation",
                timeout=30,
                required=True
            ),
            ManualScrapingStep(
                step_id="wait_load",
                instruction="⏳ Waiting for page to fully load...",
                wait_for="time",
                timeout=5,
                required=True
            ),
            ManualScrapingStep(
                step_id="handle_protection",
                instruction="🛡️ If you see CAPTCHA or login prompts, please solve them manually.\n   The browser will wait for you to complete any required actions.",
                wait_for="user_input",
                timeout=120,
                required=False
            ),
            ManualScrapingStep(
                step_id="verify_results",
                instruction="👀 Please verify the search results are loaded.\n   You should see product listings on the page.",
                wait_for="user_input",
                timeout=60,
                required=True
            )
        ]
        
    async def _execute_manual_step(self, page: Page, step: ManualScrapingStep) -> bool:
        """Execute a manual step with appropriate waiting."""
        try:
            if step.wait_for == "navigation":
                search_url = step.instruction.split(": ")[-1]
                await page.goto(search_url, timeout=30000)
                return True
                
            elif step.wait_for == "time":
                await asyncio.sleep(step.timeout)
                return True
                
            elif step.wait_for == "user_input":
                logger.info("⏸️  Waiting for your input...")
                logger.info("   Press ENTER in this terminal when ready to continue")
                
                # Wait for user input with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, input),
                        timeout=step.timeout
                    )
                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout after {step.timeout}s, continuing...")
                    return not step.required
                    
            elif step.wait_for == "element" and step.selector:
                try:
                    await page.wait_for_selector(step.selector, timeout=step.timeout * 1000)
                    return True
                except:
                    logger.warning(f"Element '{step.selector}' not found")
                    return not step.required
                    
            return True
            
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return not step.required
            
    async def _extract_products_with_guidance(
        self,
        page: Page,
        html: str,
        config: ScrapingConfig,
        store_id: str,
        max_products: int
    ) -> List[Product]:
        """Extract products with user guidance."""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try automatic extraction first
            container_selectors = [
                '.product-tile', '.product-item', '.product-card',
                '.product', '[data-product]', '.search-result-item',
                '.tile', '.item', '.result-item', '[data-testid*="product"]'
            ]
            
            product_containers = []
            used_selector = None
            
            for selector in container_selectors:
                containers = soup.select(selector)
                if containers:
                    product_containers = containers[:max_products]
                    used_selector = selector
                    logger.info(f"🎯 Found {len(containers)} products using selector: {selector}")
                    break
            
            if product_containers:
                logger.info("🔄 Attempting automatic product extraction...")
                
                for i, container in enumerate(product_containers):
                    try:
                        raw_product = self._extract_product_data(container, config)
                        if raw_product and raw_product.get("name"):
                            product = self._parse_product(raw_product, store_id)
                            if product:
                                products.append(product)
                                logger.info(f"   ✅ Extracted: {product.name} - ${product.price}")
                            else:
                                logger.debug(f"   ⚠️  Could not parse product {i+1}")
                        else:
                            logger.debug(f"   ⚠️  No name found for product {i+1}")
                    except Exception as e:
                        logger.debug(f"   ❌ Error extracting product {i+1}: {e}")
                        
                if products:
                    logger.info(f"🎉 Automatic extraction successful: {len(products)} products")
                else:
                    logger.warning("⚠️  Automatic extraction found containers but no valid products")
            else:
                logger.warning("⚠️  No product containers found with standard selectors")
                
        except Exception as e:
            logger.error(f"Error in guided extraction: {e}")
            
        return products
        
    async def _manual_product_selection(
        self,
        page: Page,
        store_id: str,
        max_products: int
    ) -> List[Product]:
        """Allow user to manually select and input product data."""
        products = []
        
        logger.info("\n🎯 MANUAL PRODUCT SELECTION")
        logger.info("=" * 40)
        logger.info("Since automatic extraction failed, let's collect product data manually.")
        logger.info("For each product you want to add:")
        logger.info("1. Look at the product in the browser")
        logger.info("2. Enter the details when prompted")
        logger.info("3. Press ENTER to add more products, or 'done' to finish")
        
        for i in range(max_products):
            logger.info(f"\n📦 Product {i+1}/{max_products}:")
            
            try:
                name = await self._get_user_input("Product name (or 'done' to finish): ")
                if name.lower() in ['done', 'quit', 'exit', '']:
                    break
                    
                price_str = await self._get_user_input("Price (e.g., 4.99): ")
                price = self._extract_price(price_str)
                
                if price <= 0:
                    logger.warning("⚠️  Invalid price, skipping product")
                    continue
                    
                brand = await self._get_user_input("Brand (optional): ")
                description = await self._get_user_input("Description (optional): ")
                
                product = Product(
                    name=name,
                    brand=brand.strip() or None,
                    price=price,
                    currency=Currency.CAD,
                    store_id=store_id,
                    description=description.strip() or None,
                    in_stock=True,
                    keywords=[word.lower() for word in name.split() if len(word) > 2]
                )
                
                products.append(product)
                logger.info(f"   ✅ Added: {product.name} - ${product.price}")
                
            except KeyboardInterrupt:
                logger.info("\n⏹️  Manual entry cancelled")
                break
            except Exception as e:
                logger.error(f"   ❌ Error adding product: {e}")
                
        return products
        
    async def _get_user_input(self, prompt: str) -> str:
        """Get input from user with async support."""
        logger.info(prompt, end="")
        try:
            return await asyncio.get_event_loop().run_in_executor(None, input)
        except Exception:
            return ""
            
    def _extract_product_data(self, container, config: ScrapingConfig) -> Dict[str, str]:
        """Extract product data from HTML container."""
        data = {}
        
        # Extract name
        name_selectors = config.product_selectors.get("name", "").split(", ")
        for selector in name_selectors:
            if selector.strip():
                elem = container.select_one(selector.strip())
                if elem:
                    data["name"] = elem.get_text(strip=True)
                    break
        
        # Fallback name extraction
        if not data.get("name"):
            for tag in ["h1", "h2", "h3", "h4", ".title", ".name", "[data-name]"]:
                elem = container.select_one(tag)
                if elem and elem.get_text(strip=True):
                    data["name"] = elem.get_text(strip=True)
                    break
        
        # Extract price
        price_selectors = config.product_selectors.get("price", "").split(", ")
        for selector in price_selectors:
            if selector.strip():
                elem = container.select_one(selector.strip())
                if elem:
                    price_text = elem.get_text(strip=True)
                    if price_text and any(char.isdigit() for char in price_text):
                        data["price"] = price_text
                        break
        
        # Fallback price extraction
        if not data.get("price"):
            text = container.get_text()
            price_match = re.search(r'\$[\d,]+\.?\d*', text)
            if price_match:
                data["price"] = price_match.group()
        
        # Extract other fields
        for field in ["brand", "description"]:
            selector = config.product_selectors.get(field, "")
            if selector:
                elem = container.select_one(selector)
                if elem:
                    data[field] = elem.get_text(strip=True)
        
        return data
        
    def _parse_product(self, raw_product: Dict[str, Any], store_id: str) -> Optional[Product]:
        """Parse raw product data into Product model."""
        try:
            name = raw_product.get("name", "").strip()
            if not name:
                return None
                
            price = self._extract_price(raw_product.get("price", "0"))
            if price <= 0:
                return None
                
            return Product(
                name=name,
                brand=raw_product.get("brand", "").strip() or None,
                price=price,
                currency=Currency.CAD,
                store_id=store_id,
                description=raw_product.get("description", "").strip() or None,
                in_stock=True,
                keywords=[word.lower() for word in name.split() if len(word) > 2]
            )
            
        except Exception as e:
            logger.debug(f"Error parsing product: {e}")
            return None
            
    def _extract_price(self, price_str: str) -> Decimal:
        """Extract numeric price from string."""
        try:
            cleaned = re.sub(r'[^\d.,]', '', str(price_str))
            if not cleaned:
                return Decimal('0')
                
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
            
    async def quick_price_check(self, urls: List[str]) -> List[Product]:
        """Quick price check for specific product URLs."""
        if not self.session_active:
            await self.start_session()
            
        products = []
        
        logger.info(f"🚀 Quick price check for {len(urls)} URLs")
        logger.info("Using your browser profile with existing sessions")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n📍 Checking URL {i}/{len(urls)}: {url}")
            
            page = await self.context.new_page()
            
            try:
                await page.goto(url, timeout=30000)
                
                # Wait for page load
                await asyncio.sleep(2)
                
                # Try to extract product info
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract product name
                name = ""
                for selector in ["h1", ".product-name", "[data-testid*='name']", ".title"]:
                    elem = soup.select_one(selector)
                    if elem:
                        name = elem.get_text(strip=True)
                        break
                
                # Extract price
                price_text = ""
                for selector in [".price", ".product-price", "[data-testid*='price']", ".cost"]:
                    elem = soup.select_one(selector)
                    if elem:
                        price_text = elem.get_text(strip=True)
                        break
                
                if not price_text:
                    price_match = re.search(r'\$[\d,]+\.?\d*', soup.get_text())
                    if price_match:
                        price_text = price_match.group()
                
                price = self._extract_price(price_text)
                
                if name and price > 0:
                    product = Product(
                        name=name,
                        price=price,
                        currency=Currency.CAD,
                        store_id="manual_check",
                        product_url=url,
                        in_stock=True,
                        keywords=[word.lower() for word in name.split() if len(word) > 2]
                    )
                    products.append(product)
                    logger.info(f"   ✅ {name} - ${price}")
                else:
                    logger.warning(f"   ⚠️  Could not extract product data from {url}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error checking {url}: {e}")
            finally:
                await page.close()
                
        logger.info(f"\n🎉 Price check complete: {len(products)} products found")
        return products


async def create_human_browser_scraper(
    store_configs: Optional[Dict[str, ScrapingConfig]] = None,
    browser_profile: Optional[BrowserProfile] = None
) -> HumanBrowserScraper:
    """Factory function to create a human-assisted browser scraper."""
    from .crawl4ai_client import DEFAULT_STORE_CONFIGS
    configs = store_configs or DEFAULT_STORE_CONFIGS
    scraper = HumanBrowserScraper(configs, browser_profile)
    return scraper