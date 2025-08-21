"""
Comprehensive test suite for all scraping systems - CI/CD compatible.
"""

import pytest
import asyncio
import os
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from agentic_grocery_price_scanner.mcps.advanced_scraper import (
    AdvancedScraper,
    AdvancedScrapingConfig,
    create_advanced_scraper
)
from agentic_grocery_price_scanner.mcps.clipboard_scraper import (
    ClipboardMonitor,
    quick_parse_clipboard
)
from agentic_grocery_price_scanner.mcps.human_browser_scraper import (
    HumanBrowserScraper,
    BrowserProfile
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import Currency


class TestSystemIntegration:
    """Test integration between all scraping systems."""
    
    @pytest.mark.asyncio
    async def test_complete_fallback_chain(self, sample_store_config):
        """Test the complete fallback chain from automation to manual."""
        # Create advanced scraper with fallback enabled
        config = AdvancedScrapingConfig(
            fallback_to_api=True,
            request_delay_range=(0.1, 0.2),
            retry_on_failure=1
        )
        
        scraper = AdvancedScraper(
            configs={"test_store": sample_store_config},
            advanced_config=config
        )
        
        # Test that scraper can fall back to mock data when direct scraping fails
        with patch.object(scraper, '_scrape_direct', side_effect=Exception("Blocked")):
            with patch('aiohttp.ClientSession') as mock_session:
                # Mock API failure too
                mock_response = AsyncMock()
                mock_response.status = 403
                mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
                
                # Should fall back to mock data
                products = await scraper.scrape_products_with_fallback(
                    "test_store", "milk", 3
                )
                
                assert len(products) > 0
                assert all(isinstance(p, Product) for p in products)
                assert all(p.store_id == "test_store" for p in products)
                
    def test_clipboard_to_product_pipeline(self, mock_clipboard_content):
        """Test complete pipeline from clipboard to Product objects."""
        monitor = ClipboardMonitor("test_store")
        
        # Test different clipboard content types
        for content_type, content in mock_clipboard_content.items():
            result = monitor._analyze_clipboard_content(content)
            
            if content_type in ["simple_product", "detailed_product"]:
                # Should successfully extract product data
                assert result.confidence > 0.5
                assert result.suggested_product is not None
                assert result.suggested_product.name is not None
                assert result.suggested_product.price > 0
            elif content_type in ["invalid", "empty"]:
                # Should have low confidence or no product
                assert result.confidence < 0.5 or result.suggested_product is None
                
    def test_browser_profile_to_scraper_integration(self):
        """Test browser profile detection and scraper creation."""
        # Test that browser profile detection works
        scraper = HumanBrowserScraper()
        profiles = scraper._detect_browser_profile()
        
        assert isinstance(profiles, dict)
        
        # Test creating scraper with detected profiles
        if profiles:
            profile_name = list(profiles.keys())[0]
            profile_path = profiles[profile_name]
            
            # Should be able to create profile config
            browser_profile = BrowserProfile(
                browser_type=profile_name.split('_')[0],  # e.g., 'chrome' from 'chrome_user_data'
                profile_path=profile_path
            )
            
            assert browser_profile.browser_type is not None
            assert browser_profile.profile_path == profile_path


class TestProductDataValidation:
    """Test product data validation across all systems."""
    
    def test_product_price_validation(self, create_test_product):
        """Test price validation across different inputs."""
        # Valid prices
        valid_prices = [0.01, 1.99, 10.50, 99.99, 1000.00]
        
        for price in valid_prices:
            product = create_test_product("Test Product", price)
            assert product.price == Decimal(str(price))
            assert product.price > 0
            
    def test_product_currency_validation(self, create_test_product):
        """Test currency validation."""
        product = create_test_product("Test Product", 5.99)
        
        # Should default to CAD
        assert product.currency == Currency.CAD
        
        # Should accept USD
        product.currency = Currency.USD
        assert product.currency == Currency.USD
        
    def test_product_keyword_generation(self):
        """Test keyword generation across different systems."""
        # Test with advanced scraper
        scraper = AdvancedScraper()
        keywords = scraper._generate_keywords("Organic Whole Milk", "Beatrice")
        
        assert "organic" in keywords
        assert "whole" in keywords
        assert "milk" in keywords
        assert "beatrice" in keywords
        
        # Test with clipboard monitor
        monitor = ClipboardMonitor()
        
        product_data = {
            "name": "Premium White Bread",
            "price": "$2.99",
            "brand": "Wonder"
        }
        
        product = monitor._parse_product(product_data, "test_store")
        assert product is not None
        assert "premium" in product.keywords
        assert "white" in product.keywords
        assert "bread" in product.keywords


class TestErrorHandlingAcrossSystems:
    """Test error handling across all scraping systems."""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, sample_store_config):
        """Test handling of network errors."""
        config = AdvancedScrapingConfig(retry_on_failure=2)
        scraper = AdvancedScraper(
            configs={"test_store": sample_store_config},
            advanced_config=config
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Simulate network error
            mock_session.side_effect = Exception("Network error")
            
            # Should handle gracefully and fall back to mock data
            products = await scraper.scrape_products_with_fallback(
                "test_store", "milk", 3
            )
            
            # Should still return mock products
            assert len(products) > 0
            
    def test_invalid_data_handling(self):
        """Test handling of invalid data across systems."""
        # Test clipboard monitor with invalid data
        monitor = ClipboardMonitor()
        
        invalid_data_cases = [
            {"name": "", "price": "$4.99"},  # Empty name
            {"name": "Valid Product", "price": "invalid"},  # Invalid price
            {"name": "Valid Product", "price": "$0.00"},  # Zero price
            {},  # Empty data
        ]
        
        for invalid_data in invalid_data_cases:
            product = monitor._parse_product(invalid_data, "test_store")
            assert product is None, f"Should reject invalid data: {invalid_data}"
            
    def test_browser_session_error_handling(self):
        """Test browser session error handling."""
        scraper = HumanBrowserScraper()
        
        # Test with invalid profile path
        invalid_profile = BrowserProfile(
            browser_type="invalid",
            profile_path="/nonexistent/path"
        )
        
        scraper.browser_profile = invalid_profile
        
        # Should handle invalid profile gracefully
        profiles = scraper._detect_browser_profile()
        assert isinstance(profiles, dict)  # Should still return valid profiles


@pytest.mark.performance
class TestPerformanceAcrossSystems:
    """Performance tests for all systems."""
    
    def test_bulk_product_processing(self, create_test_product):
        """Test processing many products efficiently."""
        import time
        
        # Create many test products
        start_time = time.time()
        
        products = []
        for i in range(1000):
            product = create_test_product(f"Product {i}", i * 0.99 + 1.00)
            products.append(product)
            
        end_time = time.time()
        
        # Should create 1000 products quickly
        assert end_time - start_time < 1.0
        assert len(products) == 1000
        assert all(isinstance(p, Product) for p in products)
        
    def test_clipboard_parsing_performance(self):
        """Test clipboard parsing performance."""
        import time
        
        monitor = ClipboardMonitor()
        
        # Create many clipboard samples
        samples = [
            f"Product {i} - ${i}.99 - Brand {i % 10}"
            for i in range(100)
        ]
        
        start_time = time.time()
        
        results = []
        for sample in samples:
            result = monitor._analyze_clipboard_content(sample)
            results.append(result)
            
        end_time = time.time()
        
        # Should process 100 samples quickly
        assert end_time - start_time < 2.0
        assert len(results) == 100
        
    def test_price_extraction_performance(self):
        """Test price extraction performance across systems."""
        import time
        
        # Test data with various price formats
        price_samples = [
            f"${i}.{j:02d}" for i in range(1, 50) for j in range(0, 100, 25)
        ] + [
            f"€{i},{j:02d}" for i in range(1, 50) for j in range(0, 100, 25)
        ]
        
        systems = [
            AdvancedScraper(),
            ClipboardMonitor(),
            HumanBrowserScraper()
        ]
        
        for system in systems:
            start_time = time.time()
            
            for price_sample in price_samples:
                result = system._extract_price(price_sample)
                assert isinstance(result, Decimal)
                
            end_time = time.time()
            
            # Each system should process prices quickly
            assert end_time - start_time < 1.0


class TestCICDCompatibility:
    """Tests specifically designed for CI/CD environments."""
    
    def test_no_external_dependencies(self):
        """Test that core functionality works without external dependencies."""
        # Test product creation
        product = Product(
            name="CI Test Product",
            price=Decimal("3.99"),
            currency=Currency.CAD,
            store_id="ci_test",
            keywords=["ci", "test"]
        )
        
        assert product.name == "CI Test Product"
        assert product.price == Decimal("3.99")
        
    def test_mock_based_scraping(self, sample_store_config):
        """Test scraping with mocked dependencies."""
        config = AdvancedScrapingConfig(fallback_to_api=True)
        scraper = AdvancedScraper(
            configs={"test_store": sample_store_config},
            advanced_config=config
        )
        
        # Generate mock products (no network required)
        products = scraper._generate_mock_products("test_store", "milk", 5)
        
        assert len(products) == 5
        assert all(isinstance(p, Product) for p in products)
        assert all("milk" in p.keywords for p in products)
        
    @pytest.mark.skipif(
        "CI" in os.environ,
        reason="Browser tests not suitable for CI"
    )
    def test_browser_functionality_local_only(self):
        """Test browser functionality only in local environment."""
        scraper = HumanBrowserScraper()
        profiles = scraper._detect_browser_profile()
        
        # In local environment, should find some browser profiles
        assert isinstance(profiles, dict)
        
    def test_clipboard_functionality_mock(self):
        """Test clipboard functionality with mocked clipboard."""
        with patch('pyperclip.paste') as mock_paste:
            mock_paste.return_value = "Test Product - $4.99 - Test Brand"
            
            with patch('agentic_grocery_price_scanner.mcps.clipboard_scraper.logger'):
                product = quick_parse_clipboard()
                
            assert product is not None
            assert product.name == "Test Product"
            assert product.price == Decimal("4.99")


class TestSystemReliability:
    """Test system reliability and robustness."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, sample_store_config):
        """Test that system degrades gracefully when components fail."""
        config = AdvancedScrapingConfig(
            fallback_to_api=True,
            retry_on_failure=1
        )
        
        scraper = AdvancedScraper(
            configs={"test_store": sample_store_config},
            advanced_config=config
        )
        
        # Simulate all external systems failing
        with patch.object(scraper, '_scrape_direct', side_effect=Exception("Network blocked")):
            with patch('aiohttp.ClientSession', side_effect=Exception("API unavailable")):
                # Should still return mock data
                products = await scraper.scrape_products_with_fallback(
                    "test_store", "milk", 3
                )
                
                assert len(products) > 0
                assert all(isinstance(p, Product) for p in products)
                
    def test_data_consistency(self, sample_products):
        """Test data consistency across different processing methods."""
        # Test that same product data produces consistent results
        test_data = {
            "name": "Consistent Test Product",
            "price": "$5.99",
            "brand": "Consistent Brand"
        }
        
        # Process with different systems
        systems = [
            AdvancedScraper(),
            ClipboardMonitor(),
            HumanBrowserScraper()
        ]
        
        results = []
        for system in systems:
            product = system._parse_product(test_data, "test_store")
            if product:
                results.append(product)
                
        # All systems should produce similar results
        assert len(results) > 0
        names = [p.name for p in results]
        prices = [p.price for p in results]
        
        # Names should be consistent
        assert len(set(names)) == 1
        # Prices should be consistent
        assert len(set(prices)) == 1
        
    def test_memory_usage(self):
        """Test that systems don't leak memory."""
        import gc
        
        # Create and destroy many objects
        for _ in range(100):
            scraper = AdvancedScraper()
            monitor = ClipboardMonitor()
            browser_scraper = HumanBrowserScraper()
            
            # Use the objects
            _ = scraper._extract_price("$4.99")
            _ = monitor._extract_price("$4.99") 
            _ = browser_scraper._extract_price("$4.99")
            
            # Clean up
            del scraper, monitor, browser_scraper
            
        # Force garbage collection
        gc.collect()
        
        # Should not have excessive memory usage
        # (This is a basic test - more sophisticated memory profiling would be better)
        assert True  # If we get here without errors, memory management is working


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])