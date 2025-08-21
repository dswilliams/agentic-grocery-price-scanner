"""
Unit tests for human-assisted browser scraping functionality.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from agentic_grocery_price_scanner.mcps.human_browser_scraper import (
    HumanBrowserScraper,
    BrowserProfile,
    ManualScrapingStep,
    create_human_browser_scraper
)
from agentic_grocery_price_scanner.mcps.crawl4ai_client import ScrapingConfig
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import Currency


class TestBrowserProfile:
    """Test browser profile configuration."""
    
    def test_browser_profile_defaults(self):
        """Test default browser profile configuration."""
        profile = BrowserProfile()
        assert profile.browser_type == "chrome"
        assert profile.auto_detect is True
        assert profile.profile_path is None
        
    def test_browser_profile_custom(self):
        """Test custom browser profile configuration."""
        profile = BrowserProfile(
            browser_type="firefox",
            profile_path="/custom/path",
            auto_detect=False
        )
        assert profile.browser_type == "firefox"
        assert profile.profile_path == "/custom/path"
        assert profile.auto_detect is False


class TestManualScrapingStep:
    """Test manual scraping step configuration."""
    
    def test_manual_step_creation(self):
        """Test creating manual scraping steps."""
        step = ManualScrapingStep(
            step_id="test_step",
            instruction="Test instruction",
            wait_for="user_input",
            timeout=60
        )
        assert step.step_id == "test_step"
        assert step.instruction == "Test instruction"
        assert step.wait_for == "user_input"
        assert step.timeout == 60
        assert step.required is True


class TestHumanBrowserScraper:
    """Test human browser scraper functionality."""
    
    @pytest.fixture
    def mock_configs(self):
        """Create mock store configurations."""
        return {
            "test_store": ScrapingConfig(
                store_id="test_store",
                base_url="https://test.com",
                search_url_template="https://test.com/search?q={query}",
                product_selectors={
                    "name": ".product-name",
                    "price": ".product-price",
                    "brand": ".product-brand"
                },
                rate_limit_delay=1.0
            )
        }
    
    def test_scraper_initialization(self, mock_configs):
        """Test scraper initialization."""
        profile = BrowserProfile(browser_type="firefox")
        scraper = HumanBrowserScraper(mock_configs, profile)
        
        assert scraper.configs == mock_configs
        assert scraper.browser_profile.browser_type == "firefox"
        assert scraper.session_active is False
        
    def test_browser_profile_detection(self):
        """Test browser profile detection."""
        scraper = HumanBrowserScraper()
        
        with patch('pathlib.Path.exists', return_value=True):
            profiles = scraper._detect_browser_profile()
            
            # Should find at least some profiles on any system
            assert isinstance(profiles, dict)
            
    def test_extract_price(self):
        """Test price extraction from various formats."""
        scraper = HumanBrowserScraper()
        
        test_cases = [
            ("$4.99", Decimal("4.99")),
            ("4.99", Decimal("4.99")),
            ("CAD $12.50", Decimal("12.50")),
            ("Price: $7.29", Decimal("7.29")),
            ("1,234.56", Decimal("1234.56")),
            ("€15.99", Decimal("15.99")),
            ("invalid", Decimal("0")),
            ("", Decimal("0"))
        ]
        
        for price_str, expected in test_cases:
            result = scraper._extract_price(price_str)
            assert result == expected, f"Failed for {price_str}: got {result}, expected {expected}"
            
    def test_parse_product(self, mock_configs):
        """Test product parsing from raw data."""
        scraper = HumanBrowserScraper(mock_configs)
        
        raw_product = {
            "name": "Test Milk 1L",
            "price": "$4.99",
            "brand": "Test Brand",
            "description": "Fresh milk"
        }
        
        product = scraper._parse_product(raw_product, "test_store")
        
        assert product is not None
        assert product.name == "Test Milk 1L"
        assert product.price == Decimal("4.99")
        assert product.brand == "Test Brand"
        assert product.store_id == "test_store"
        assert product.currency == Currency.CAD
        
    def test_parse_product_invalid_price(self, mock_configs):
        """Test product parsing with invalid price."""
        scraper = HumanBrowserScraper(mock_configs)
        
        raw_product = {
            "name": "Test Product",
            "price": "invalid",
            "brand": "Test Brand"
        }
        
        product = scraper._parse_product(raw_product, "test_store")
        assert product is None  # Should return None for invalid price
        
    def test_parse_product_missing_name(self, mock_configs):
        """Test product parsing with missing name."""
        scraper = HumanBrowserScraper(mock_configs)
        
        raw_product = {
            "name": "",
            "price": "$4.99",
            "brand": "Test Brand"
        }
        
        product = scraper._parse_product(raw_product, "test_store")
        assert product is None  # Should return None for missing name
        
    def test_extract_product_data(self, mock_configs):
        """Test product data extraction from HTML."""
        from bs4 import BeautifulSoup
        
        scraper = HumanBrowserScraper(mock_configs)
        config = mock_configs["test_store"]
        
        html = """
        <div class="product">
            <span class="product-name">Test Product</span>
            <span class="product-price">$9.99</span>
            <span class="product-brand">Test Brand</span>
        </div>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        container = soup.find("div", class_="product")
        
        data = scraper._extract_product_data(container, config)
        
        assert data["name"] == "Test Product"
        assert data["price"] == "$9.99"
        assert data["brand"] == "Test Brand"
        
    def test_create_scraping_workflow(self, mock_configs):
        """Test scraping workflow creation."""
        scraper = HumanBrowserScraper(mock_configs)
        config = mock_configs["test_store"]
        
        workflow = scraper._create_scraping_workflow(config, "milk")
        
        assert len(workflow) > 0
        assert all(isinstance(step, ManualScrapingStep) for step in workflow)
        assert any("navigate" in step.step_id.lower() for step in workflow)
        
    @pytest.mark.asyncio
    async def test_get_user_input_mock(self, mock_configs):
        """Test user input handling with mock."""
        scraper = HumanBrowserScraper(mock_configs)
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value="test input")
            mock_loop.return_value.run_in_executor = mock_executor
            
            result = await scraper._get_user_input("Test prompt: ")
            assert result == "test input"


class TestFactoryFunction:
    """Test factory function for creating scraper."""
    
    @pytest.mark.asyncio
    async def test_create_human_browser_scraper(self):
        """Test factory function."""
        scraper = await create_human_browser_scraper()
        
        assert isinstance(scraper, HumanBrowserScraper)
        assert len(scraper.configs) > 0  # Should have default configs
        
    @pytest.mark.asyncio
    async def test_create_human_browser_scraper_custom(self):
        """Test factory function with custom config."""
        custom_config = {
            "custom_store": ScrapingConfig(
                store_id="custom_store",
                base_url="https://custom.com",
                search_url_template="https://custom.com/search?q={query}",
                product_selectors={"name": ".name", "price": ".price"}
            )
        }
        
        profile = BrowserProfile(browser_type="firefox")
        scraper = await create_human_browser_scraper(custom_config, profile)
        
        assert scraper.configs == custom_config
        assert scraper.browser_profile.browser_type == "firefox"


class TestErrorHandling:
    """Test error handling in scraper."""
    
    def test_extract_price_edge_cases(self):
        """Test price extraction edge cases."""
        scraper = HumanBrowserScraper()
        
        edge_cases = [
            (None, Decimal("0")),
            (123, Decimal("123")),
            ("$1,234.56", Decimal("1234.56")),
            ("€1.234,56", Decimal("1234.56")),  # European format
            ("Free", Decimal("0")),
            ("N/A", Decimal("0")),
        ]
        
        for input_val, expected in edge_cases:
            result = scraper._extract_price(input_val)
            assert result == expected, f"Failed for {input_val}: got {result}, expected {expected}"
            
    def test_parse_product_edge_cases(self, mock_configs):
        """Test product parsing edge cases."""
        scraper = HumanBrowserScraper(mock_configs)
        
        # Test with None values
        raw_product = {
            "name": None,
            "price": None,
            "brand": None
        }
        
        product = scraper._parse_product(raw_product, "test_store")
        assert product is None
        
        # Test with empty strings
        raw_product = {
            "name": "",
            "price": "",
            "brand": ""
        }
        
        product = scraper._parse_product(raw_product, "test_store")
        assert product is None


@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual system resources)."""
    
    def test_browser_profile_detection_real(self):
        """Test real browser profile detection on current system."""
        scraper = HumanBrowserScraper()
        profiles = scraper._detect_browser_profile()
        
        # Should find at least one browser profile on most systems
        assert isinstance(profiles, dict)
        
        # Check that detected paths actually exist
        for browser, path in profiles.items():
            assert Path(path).exists(), f"Profile path doesn't exist: {path}"
            
    @pytest.mark.skipif(
        not any(Path(p).exists() for p in [
            Path.home() / "Library/Application Support/Google/Chrome",
            Path.home() / "AppData/Local/Google/Chrome/User Data",
            Path.home() / ".config/google-chrome"
        ]),
        reason="No Chrome profile found"
    )
    def test_chrome_profile_detection(self):
        """Test Chrome profile detection specifically."""
        scraper = HumanBrowserScraper()
        profiles = scraper._detect_browser_profile()
        
        chrome_profiles = [k for k in profiles.keys() if "chrome" in k.lower()]
        assert len(chrome_profiles) > 0, "No Chrome profiles detected"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])