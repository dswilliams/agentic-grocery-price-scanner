"""
Unit tests for clipboard-based product data extraction.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agentic_grocery_price_scanner.mcps.clipboard_scraper import (
    ClipboardMonitor,
    ClipboardProduct,
    start_clipboard_collection,
    quick_parse_clipboard
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import Currency


class TestClipboardProduct:
    """Test ClipboardProduct model."""
    
    def test_clipboard_product_creation(self):
        """Test creating a ClipboardProduct."""
        product = ClipboardProduct(
            raw_text="Test product text",
            confidence=0.8,
            extracted_fields={"name": "Test Product", "price": "$4.99"}
        )
        
        assert product.raw_text == "Test product text"
        assert product.confidence == 0.8
        assert product.extracted_fields["name"] == "Test Product"
        assert product.suggested_product is None
        
    def test_clipboard_product_with_suggested(self):
        """Test ClipboardProduct with suggested product."""
        suggested = Product(
            name="Test Product",
            price=Decimal("4.99"),
            currency=Currency.CAD,
            store_id="test_store",
            keywords=["test", "product"]
        )
        
        product = ClipboardProduct(
            raw_text="Test product text",
            confidence=0.9,
            extracted_fields={"name": "Test Product", "price": "$4.99"},
            suggested_product=suggested
        )
        
        assert product.suggested_product is not None
        assert product.suggested_product.name == "Test Product"
        assert product.suggested_product.price == Decimal("4.99")


class TestClipboardMonitor:
    """Test clipboard monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create a clipboard monitor for testing."""
        return ClipboardMonitor("test_store")
        
    def test_monitor_initialization(self, monitor):
        """Test clipboard monitor initialization."""
        assert monitor.store_id == "test_store"
        assert monitor.last_clipboard == ""
        assert monitor.is_monitoring is False
        assert len(monitor.products_collected) == 0
        
    def test_extract_price_patterns(self, monitor):
        """Test price extraction patterns."""
        test_cases = [
            ("Product costs $4.99", "4.99"),
            ("Price: $12.50", "12.50"),
            ("Total: 7.29 CAD", "7.29"),
            ("$1,234.56", "1234.56"),
            ("€15.99", "15.99"),
            ("19.99$", "19.99"),
        ]
        
        for text, expected_price in test_cases:
            result = monitor._extract_with_patterns(text, monitor.patterns["price"])
            assert result is not None, f"No price found in: {text}"
            extracted_price = monitor._extract_price(result)
            assert extracted_price == Decimal(expected_price), f"Wrong price for {text}: got {extracted_price}"
            
    def test_extract_product_name_patterns(self, monitor):
        """Test product name extraction."""
        test_cases = [
            (["Organic Whole Milk 1L", "$4.99", "Beatrice"], "Organic Whole Milk 1L"),
            (["Product: Premium Bread", "Price: $2.99"], "Premium Bread"),
            (["$5.99", "Fresh Ground Beef", "Metro"], "Fresh Ground Beef"),
        ]
        
        for lines, expected_name in test_cases:
            content = "\n".join(lines)
            result = monitor._extract_product_name(lines, content)
            assert result == expected_name, f"Wrong name extracted from {lines}: got {result}"
            
    def test_extract_size_patterns(self, monitor):
        """Test size/unit extraction."""
        test_cases = [
            ("Milk 1L bottle", ("1", "l")),
            ("Size: 454g package", ("454", "g")),
            ("Weight: 2.5 kg", ("2.5", "kg")),
            ("Volume: 500ml", ("500", "ml")),
            ("Count: 12 pieces", ("12", "pieces")),
        ]
        
        for text, expected in test_cases:
            result = monitor._extract_size(text)
            assert result == expected, f"Wrong size for {text}: got {result}, expected {expected}"
            
    def test_extract_brand_patterns(self, monitor):
        """Test brand extraction."""
        test_cases = [
            ("by Beatrice Dairy", "Beatrice Dairy"),
            ("from Wonder Bread Co.", "Wonder Bread Co."),
            ("Brand: Black Diamond", "Black Diamond"),
            ("Loblaws Brand Product", "Loblaws"),
        ]
        
        for text, expected_brand in test_cases:
            result = monitor._extract_with_patterns(text, monitor.patterns["brand"])
            if expected_brand:
                assert result is not None, f"No brand found in: {text}"
                assert expected_brand.lower() in result.lower(), f"Wrong brand for {text}: got {result}"
                
    def test_extract_store_patterns(self, monitor):
        """Test store extraction."""
        test_cases = [
            ("Available at Metro", "Metro"),
            ("from Walmart Canada", "Walmart"),
            ("@ Loblaws store", "Loblaws"),
            ("Sobeys exclusive", "Sobeys"),
        ]
        
        for text, expected_store in test_cases:
            result = monitor._extract_with_patterns(text, monitor.patterns["store"])
            if expected_store:
                assert result is not None, f"No store found in: {text}"
                assert expected_store.lower() in result.lower(), f"Wrong store for {text}: got {result}"
                
    def test_analyze_clipboard_content_simple(self, monitor):
        """Test analyzing simple clipboard content."""
        content = """
        Beatrice 2% Milk 1L
        $4.99
        Available at Metro
        """
        
        result = monitor._analyze_clipboard_content(content)
        
        assert result.confidence > 0.5  # Should have decent confidence
        assert "name" in result.extracted_fields
        assert "price" in result.extracted_fields
        assert result.extracted_fields["name"] == "Beatrice 2% Milk 1L"
        assert "$4.99" in result.extracted_fields["price"]
        assert result.suggested_product is not None
        
    def test_analyze_clipboard_content_complex(self, monitor):
        """Test analyzing complex clipboard content."""
        content = """
        Wonder Bread White Sandwich Loaf 675g
        Brand: Wonder
        Price: $2.49 CAD
        Size: 675g
        Available at Metro stores
        Fresh baked daily
        """
        
        result = monitor._analyze_clipboard_content(content)
        
        assert result.confidence > 0.6  # Should have high confidence
        assert result.suggested_product is not None
        assert result.suggested_product.name == "Wonder Bread White Sandwich Loaf 675g"
        assert result.suggested_product.price == Decimal("2.49")
        
    def test_analyze_clipboard_content_insufficient(self, monitor):
        """Test analyzing insufficient clipboard content."""
        content = "Just some random text without product info"
        
        result = monitor._analyze_clipboard_content(content)
        
        assert result.confidence < 0.3  # Should have low confidence
        assert result.suggested_product is None
        
    def test_collect_products(self, monitor):
        """Test collecting products."""
        # Simulate product collection
        test_product = Product(
            name="Test Product",
            price=Decimal("5.99"),
            currency=Currency.CAD,
            store_id="test_store",
            keywords=["test"]
        )
        
        monitor.products_collected.append(test_product)
        
        collected = monitor.get_collected_products()
        assert len(collected) == 1
        assert collected[0].name == "Test Product"
        assert collected[0].price == Decimal("5.99")
        
    def test_clear_collected_products(self, monitor):
        """Test clearing collected products."""
        # Add some products
        test_product = Product(
            name="Test Product",
            price=Decimal("5.99"),
            currency=Currency.CAD,
            store_id="test_store",
            keywords=["test"]
        )
        
        monitor.products_collected.append(test_product)
        assert len(monitor.get_collected_products()) == 1
        
        monitor.clear_collected_products()
        assert len(monitor.get_collected_products()) == 0
        
    def test_monitoring_state(self, monitor):
        """Test monitoring state management."""
        assert monitor.is_monitoring is False
        
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False


class TestClipboardParsing:
    """Test clipboard parsing edge cases."""
    
    def test_price_extraction_edge_cases(self):
        """Test price extraction with edge cases."""
        monitor = ClipboardMonitor()
        
        test_cases = [
            ("$0.99", Decimal("0.99")),
            ("€15,99", Decimal("15.99")),
            ("1.234,56 EUR", Decimal("1234.56")),
            ("Free shipping", Decimal("0")),
            ("No price", Decimal("0")),
            ("$1,000.00", Decimal("1000.00")),
            ("Price varies", Decimal("0")),
        ]
        
        for price_str, expected in test_cases:
            result = monitor._extract_price(price_str)
            assert result == expected, f"Failed for {price_str}: got {result}, expected {expected}"
            
    def test_name_extraction_edge_cases(self):
        """Test name extraction edge cases."""
        monitor = ClipboardMonitor()
        
        test_cases = [
            (["", "$4.99", "Brand"], None),  # Empty name
            (["$4.99", "Valid Product Name", "Extra info"], "Valid Product Name"),
            (["http://example.com", "Product Name", "$4.99"], "Product Name"),
            (["Price: $4.99", "Product Name Here"], "Product Name Here"),
        ]
        
        for lines, expected in test_cases:
            content = "\n".join(lines)
            result = monitor._extract_product_name(lines, content)
            assert result == expected, f"Wrong name for {lines}: got {result}, expected {expected}"
            
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        monitor = ClipboardMonitor()
        
        # High confidence case
        high_conf_content = """
        Organic Milk 2% 1L
        Beatrice Brand
        $5.49 CAD
        Size: 1L
        Available at Metro
        """
        
        result = monitor._analyze_clipboard_content(high_conf_content)
        assert result.confidence > 0.8
        
        # Medium confidence case
        med_conf_content = """
        Some Product Name
        $3.99
        """
        
        result = monitor._analyze_clipboard_content(med_conf_content)
        assert 0.3 < result.confidence < 0.8
        
        # Low confidence case
        low_conf_content = "Just some random text"
        
        result = monitor._analyze_clipboard_content(low_conf_content)
        assert result.confidence < 0.3


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('pyperclip.paste')
    def test_quick_parse_clipboard(self, mock_paste):
        """Test quick clipboard parsing."""
        mock_paste.return_value = """
        Test Product Name
        $7.99
        Test Brand
        """
        
        with patch('agentic_grocery_price_scanner.mcps.clipboard_scraper.logger'):
            product = quick_parse_clipboard()
            
        mock_paste.assert_called_once()
        assert product is not None
        assert product.name == "Test Product Name"
        assert product.price == Decimal("7.99")
        
    @patch('pyperclip.paste')
    def test_quick_parse_clipboard_empty(self, mock_paste):
        """Test quick parsing with empty clipboard."""
        mock_paste.return_value = ""
        
        product = quick_parse_clipboard()
        assert product is None
        
    @patch('pyperclip.paste')
    def test_quick_parse_clipboard_low_confidence(self, mock_paste):
        """Test quick parsing with low confidence content."""
        mock_paste.return_value = "random text without product info"
        
        with patch('agentic_grocery_price_scanner.mcps.clipboard_scraper.logger'):
            product = quick_parse_clipboard()
            
        assert product is None  # Should return None for low confidence
        
    @pytest.mark.asyncio
    async def test_start_clipboard_collection(self):
        """Test starting clipboard collection session."""
        with patch('agentic_grocery_price_scanner.mcps.clipboard_scraper.ClipboardMonitor') as MockMonitor:
            mock_instance = MockMonitor.return_value
            mock_instance.manual_clipboard_session = AsyncMock(return_value=[])
            
            result = await start_clipboard_collection("test_store", 5)
            
            MockMonitor.assert_called_once_with("test_store")
            mock_instance.manual_clipboard_session.assert_called_once_with(5)
            assert result == []


@pytest.mark.performance
class TestPerformance:
    """Performance tests for clipboard operations."""
    
    def test_pattern_matching_performance(self):
        """Test pattern matching performance."""
        import time
        
        monitor = ClipboardMonitor()
        test_content = """
        Large Product Name With Many Words Here
        Brand: Some Long Brand Name Company Inc.
        Price: $123.45 CAD
        Size: 2.5kg package
        Available at Metro Grocery Store Location
        Description: This is a long product description with many details
        """ * 100  # Repeat to make it larger
        
        start_time = time.time()
        result = monitor._analyze_clipboard_content(test_content)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert result.confidence > 0.5
        
    def test_multiple_extractions_performance(self):
        """Test performance with multiple extractions."""
        import time
        
        monitor = ClipboardMonitor()
        test_cases = [
            f"Product {i} - ${i}.99 - Brand {i}" 
            for i in range(100)
        ]
        
        start_time = time.time()
        results = []
        for content in test_cases:
            result = monitor._analyze_clipboard_content(content)
            results.append(result)
        end_time = time.time()
        
        # Should process 100 items in reasonable time
        assert end_time - start_time < 5.0
        assert len(results) == 100
        assert all(r.confidence > 0.3 for r in results)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])