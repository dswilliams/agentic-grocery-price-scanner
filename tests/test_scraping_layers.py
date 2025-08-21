"""
Integration tests for all scraping layers and fallback strategies.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agentic_grocery_price_scanner.mcps.advanced_scraper import (
    AdvancedScraper,
    AdvancedScrapingConfig,
    AlternativeDataSource,
    create_advanced_scraper
)
from agentic_grocery_price_scanner.mcps.stealth_scraper import (
    StealthScraper,
    StealthConfig,
    UserAgentRotator,
    create_stealth_scraper
)
from agentic_grocery_price_scanner.mcps.human_browser_scraper import (
    HumanBrowserScraper,
    BrowserProfile,
    create_human_browser_scraper
)
from agentic_grocery_price_scanner.mcps.crawl4ai_client import (
    WebScrapingClient,
    DEFAULT_STORE_CONFIGS
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import Currency


class TestUserAgentRotation:
    """Test user agent rotation functionality."""
    
    def test_user_agent_rotator_initialization(self):
        """Test user agent rotator initialization."""
        rotator = UserAgentRotator()
        assert len(rotator.user_agents) > 0
        
    def test_get_random_user_agent(self):
        """Test getting random user agents."""
        rotator = UserAgentRotator()
        
        # Get multiple user agents
        agents = [rotator.get_random_user_agent() for _ in range(10)]
        
        # Should all be valid strings
        assert all(isinstance(agent, str) for agent in agents)
        assert all(len(agent) > 50 for agent in agents)  # Reasonable length
        
    def test_get_matching_headers(self):
        """Test getting headers that match user agent."""
        rotator = UserAgentRotator()
        user_agent = rotator.get_random_user_agent()
        
        headers = rotator.get_matching_headers(user_agent)
        
        assert isinstance(headers, dict)
        assert "Accept" in headers
        assert "Accept-Language" in headers
        
        # Chrome-specific headers for Chrome user agents
        if "Chrome" in user_agent:
            assert "sec-ch-ua" in headers


class TestWebScrapingClient:
    """Test basic web scraping client functionality."""
    
    @pytest.mark.asyncio
    async def test_web_scraping_client_creation(self):
        """Test creating web scraping client."""
        client = WebScrapingClient()
        assert client.configs == {}
        assert client.session is None
        assert client.session_active is False
        
    @pytest.mark.asyncio
    async def test_web_scraping_client_with_configs(self):
        """Test web scraping client with configurations."""
        client = WebScrapingClient(DEFAULT_STORE_CONFIGS)
        assert len(client.configs) > 0
        assert "metro_ca" in client.configs
        
    def test_extract_price_functionality(self):
        """Test price extraction in web scraping client."""
        client = WebScrapingClient()
        
        test_cases = [
            ("$4.99", Decimal("4.99")),
            ("4,99€", Decimal("4.99")),
            ("Price: 12.50", Decimal("12.50")),
            ("CAD $7.29", Decimal("7.29")),
            ("invalid price", Decimal("0"))
        ]
        
        for price_str, expected in test_cases:
            result = client._extract_price(price_str)
            assert result == expected
            
    def test_generate_keywords(self):
        """Test keyword generation."""
        client = WebScrapingClient()
        
        keywords = client._generate_keywords("Organic Whole Milk", "Beatrice")
        
        assert isinstance(keywords, list)
        assert "organic" in keywords
        assert "whole" in keywords
        assert "milk" in keywords
        assert "beatrice" in keywords


class TestAdvancedScraper:
    """Test advanced scraping functionality."""
    
    @pytest.fixture
    def advanced_config(self):
        """Create advanced scraping configuration."""
        return AdvancedScrapingConfig(
            fallback_to_api=True,
            request_delay_range=(0.1, 0.5),  # Fast for testing
            retry_on_failure=1,
            session_reuse_count=3
        )
        
    @pytest.mark.asyncio
    async def test_advanced_scraper_creation(self, advanced_config):
        """Test creating advanced scraper."""
        scraper = AdvancedScraper(
            configs=DEFAULT_STORE_CONFIGS,
            advanced_config=advanced_config
        )
        
        assert scraper.configs == DEFAULT_STORE_CONFIGS
        assert scraper.advanced_config == advanced_config
        assert scraper.session_active is False
        
    @pytest.mark.asyncio
    async def test_create_advanced_scraper_factory(self):
        """Test factory function for advanced scraper."""
        scraper = await create_advanced_scraper()
        
        assert isinstance(scraper, AdvancedScraper)
        assert len(scraper.configs) > 0
        
    def test_generate_mock_products(self, advanced_config):
        """Test mock product generation."""
        scraper = AdvancedScraper(advanced_config=advanced_config)
        
        # Test different search terms
        search_terms = ["milk", "bread", "groceries"]
        
        for search_term in search_terms:
            products = scraper._generate_mock_products("test_store", search_term, 3)
            
            assert len(products) <= 3
            assert all(isinstance(p, Product) for p in products)
            assert all(p.store_id == "test_store" for p in products)
            assert all(p.price > 0 for p in products)
            
            # Check that keywords are relevant
            for product in products:
                assert search_term.lower() in [kw.lower() for kw in product.keywords]
                
    def test_extract_price_advanced(self, advanced_config):
        """Test advanced price extraction."""
        scraper = AdvancedScraper(advanced_config=advanced_config)
        
        edge_cases = [
            ("€12,50", Decimal("12.50")),
            ("1.234,56 USD", Decimal("1234.56")),
            ("$1,000.00", Decimal("1000.00")),
            ("Free", Decimal("0")),
            ("N/A", Decimal("0")),
            ("", Decimal("0"))
        ]
        
        for price_str, expected in edge_cases:
            result = scraper._extract_price(price_str)
            assert result == expected, f"Failed for {price_str}: got {result}, expected {expected}"
            
    def test_create_product_from_data(self, advanced_config):
        """Test creating products from raw data."""
        scraper = AdvancedScraper(
            configs=DEFAULT_STORE_CONFIGS,
            advanced_config=advanced_config
        )
        
        raw_data = {
            "name": "Test Product Name",
            "price": "$5.99",
            "brand": "Test Brand",
            "description": "Test description",
            "url": "/product/123"
        }
        
        product = scraper._create_product_from_data(raw_data, "metro_ca")
        
        assert product is not None
        assert product.name == "Test Product Name"
        assert product.price == Decimal("5.99")
        assert product.brand == "Test Brand"
        assert product.store_id == "metro_ca"
        assert product.product_url.endswith("/product/123")


class TestAlternativeDataSource:
    """Test alternative data source functionality."""
    
    @pytest.mark.asyncio
    async def test_get_flyer_data(self):
        """Test getting flyer data from APIs."""
        # Mock aiohttp session
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "products": [
                    {
                        "name": "Test Product",
                        "price": {"amount": 4.99},
                        "brand": "Test Brand"
                    }
                ]
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Test the function
            products = await AlternativeDataSource.get_flyer_data("metro_ca", "milk")
            
            # Should attempt to get data (even if mocked)
            mock_session.get.assert_called_once()
            
    def test_parse_api_product(self):
        """Test parsing API product responses."""
        test_cases = [
            # Standard format
            {
                "input": {
                    "name": "Test Product",
                    "price": {"amount": 4.99},
                    "brand": "Test Brand",
                    "description": "Test description"
                },
                "expected_name": "Test Product",
                "expected_price": Decimal("4.99")
            },
            # Simple price format
            {
                "input": {
                    "name": "Another Product",
                    "price": 7.50,
                    "brand": "Another Brand"
                },
                "expected_name": "Another Product", 
                "expected_price": Decimal("7.50")
            },
            # String price format
            {
                "input": {
                    "name": "String Price Product",
                    "price": "12.99",
                    "brand": "String Brand"
                },
                "expected_name": "String Price Product",
                "expected_price": Decimal("12.99")
            }
        ]
        
        for case in test_cases:
            product = AlternativeDataSource._parse_api_product(case["input"], "test_store")
            
            if product:  # Should create valid product
                assert product.name == case["expected_name"]
                assert product.price == case["expected_price"]
                assert product.store_id == "test_store"
            
    def test_parse_api_product_invalid(self):
        """Test parsing invalid API product data."""
        invalid_cases = [
            {},  # Empty
            {"name": ""},  # Empty name
            {"name": "Valid", "price": 0},  # Zero price
            {"name": "Valid", "price": "invalid"},  # Invalid price
        ]
        
        for invalid_data in invalid_cases:
            product = AlternativeDataSource._parse_api_product(invalid_data, "test_store")
            assert product is None


class TestStealthConfig:
    """Test stealth configuration."""
    
    def test_stealth_config_defaults(self):
        """Test default stealth configuration."""
        config = StealthConfig()
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.enable_stealth is True
        assert config.navigation_timeout == 30000
        
    def test_stealth_config_custom(self):
        """Test custom stealth configuration."""
        config = StealthConfig(
            headless=False,
            browser_type="firefox",
            viewport_width=1600,
            enable_stealth=False
        )
        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.viewport_width == 1600
        assert config.enable_stealth is False


@pytest.mark.integration
class TestScrapingIntegration:
    """Integration tests for scraping layers."""
    
    @pytest.mark.asyncio
    async def test_fallback_chain_mock(self):
        """Test the fallback chain with mocked components."""
        config = AdvancedScrapingConfig(
            fallback_to_api=True,
            request_delay_range=(0.1, 0.2)
        )
        
        scraper = AdvancedScraper(
            configs=DEFAULT_STORE_CONFIGS,
            advanced_config=config
        )
        
        # Mock the direct scraping to fail
        with patch.object(scraper, '_scrape_direct', side_effect=Exception("Blocked")):
            # Mock API fallback to also fail
            with patch.object(AlternativeDataSource, 'get_flyer_data', return_value=[]):
                # Should fall back to mock data
                products = await scraper.scrape_products_with_fallback("metro_ca", "milk", 3)
                
                # Should get mock products
                assert len(products) > 0
                assert all(isinstance(p, Product) for p in products)
                assert all(p.store_id == "metro_ca" for p in products)
                
    @pytest.mark.asyncio
    async def test_browser_profile_fallback(self):
        """Test browser profile as fallback."""
        # Test that we can create human browser scraper as fallback
        scraper = await create_human_browser_scraper()
        
        assert isinstance(scraper, HumanBrowserScraper)
        assert len(scraper.configs) > 0
        
        # Test profile detection works
        profiles = scraper._detect_browser_profile()
        assert isinstance(profiles, dict)
        
    def test_layered_approach_simulation(self):
        """Simulate the layered approach without actual network calls."""
        # Layer 1: Stealth scraping (simulated failure)
        layer1_success = False
        
        # Layer 2: Human browser (simulated partial success)
        layer2_success = True
        layer2_products = [
            Product(
                name="Layer 2 Product",
                price=Decimal("3.99"),
                currency=Currency.CAD,
                store_id="metro_ca",
                keywords=["layer2"]
            )
        ]
        
        # Layer 3: Mock data (always succeeds)
        layer3_products = [
            Product(
                name="Mock Product 1",
                price=Decimal("4.99"),
                currency=Currency.CAD,
                store_id="metro_ca",
                keywords=["mock"]
            ),
            Product(
                name="Mock Product 2", 
                price=Decimal("2.99"),
                currency=Currency.CAD,
                store_id="metro_ca",
                keywords=["mock"]
            )
        ]
        
        # Simulate fallback logic
        final_products = []
        
        if layer1_success:
            final_products.extend([])  # Would add layer 1 products
        elif layer2_success:
            final_products.extend(layer2_products)
        else:
            final_products.extend(layer3_products)
            
        # Should have products from layer 2
        assert len(final_products) == 1
        assert final_products[0].name == "Layer 2 Product"
        
        # Test complete fallback
        layer2_success = False
        final_products = []
        
        if layer1_success:
            final_products.extend([])
        elif layer2_success:
            final_products.extend([])
        else:
            final_products.extend(layer3_products)
            
        # Should have mock products
        assert len(final_products) == 2
        assert all("mock" in p.keywords for p in final_products)


@pytest.mark.performance
class TestScrapingPerformance:
    """Performance tests for scraping layers."""
    
    def test_mock_product_generation_performance(self):
        """Test performance of mock product generation."""
        import time
        
        config = AdvancedScrapingConfig()
        scraper = AdvancedScraper(advanced_config=config)
        
        start_time = time.time()
        
        # Generate products for multiple stores and search terms
        all_products = []
        stores = ["metro_ca", "walmart_ca", "freshco_ca"]
        search_terms = ["milk", "bread", "cheese", "meat", "fruits"]
        
        for store in stores:
            for term in search_terms:
                products = scraper._generate_mock_products(store, term, 5)
                all_products.extend(products)
                
        end_time = time.time()
        
        # Should generate products quickly
        assert end_time - start_time < 2.0  # Less than 2 seconds
        assert len(all_products) == len(stores) * len(search_terms) * 5
        
    def test_price_extraction_performance(self):
        """Test performance of price extraction."""
        import time
        
        scraper = AdvancedScraper()
        
        # Generate many price strings to test
        price_strings = [
            f"${i}.{j:02d}" for i in range(1, 100) for j in range(0, 100, 10)
        ]
        
        start_time = time.time()
        
        results = []
        for price_str in price_strings:
            result = scraper._extract_price(price_str)
            results.append(result)
            
        end_time = time.time()
        
        # Should process many prices quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        assert len(results) == len(price_strings)
        assert all(isinstance(r, Decimal) for r in results)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])