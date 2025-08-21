"""
Pytest configuration and fixtures for grocery price scanner tests.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from agentic_grocery_price_scanner.mcps.crawl4ai_client import ScrapingConfig
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import Currency


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_store_config():
    """Create a sample store configuration for testing."""
    return ScrapingConfig(
        store_id="test_store",
        base_url="https://test-store.com",
        search_url_template="https://test-store.com/search?q={query}",
        product_selectors={
            "name": ".product-name, .title",
            "price": ".product-price, .price",
            "brand": ".product-brand, .brand",
            "description": ".product-description, .desc",
            "url": ".product-link, .link",
            "image": ".product-image img, .image img"
        },
        rate_limit_delay=1.0,
        max_retries=3
    )


@pytest.fixture
def sample_product():
    """Create a sample product for testing."""
    return Product(
        name="Test Organic Milk 1L",
        brand="Test Brand",
        price=Decimal("4.99"),
        currency=Currency.CAD,
        store_id="test_store",
        description="Fresh organic whole milk",
        in_stock=True,
        keywords=["milk", "organic", "dairy"]
    )


@pytest.fixture
def sample_products():
    """Create multiple sample products for testing."""
    return [
        Product(
            name="Organic Milk 1L",
            brand="Beatrice",
            price=Decimal("4.99"),
            currency=Currency.CAD,
            store_id="metro_ca",
            keywords=["milk", "organic"]
        ),
        Product(
            name="White Bread",
            brand="Wonder",
            price=Decimal("2.49"),
            currency=Currency.CAD,
            store_id="metro_ca",
            on_sale=True,
            sale_price=Decimal("1.99"),
            keywords=["bread", "white"]
        ),
        Product(
            name="Ground Beef 1lb",
            price=Decimal("6.99"),
            currency=Currency.CAD,
            store_id="walmart_ca",
            keywords=["beef", "meat"]
        )
    ]


@pytest.fixture
def mock_clipboard_content():
    """Sample clipboard content for testing."""
    return {
        "simple_product": """
        Organic Whole Milk 1L
        $4.99
        Beatrice
        """,
        "detailed_product": """
        Wonder Bread White Sandwich Loaf 675g
        Brand: Wonder
        Price: $2.49 CAD
        Size: 675g
        Available at Metro stores
        Fresh baked daily
        """,
        "price_only": "$5.99",
        "name_only": "Test Product Name",
        "invalid": "Random text without product info",
        "empty": ""
    }


@pytest.fixture
def mock_html_product_page():
    """Sample HTML content with product information."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Product Page</title></head>
    <body>
        <div class="product-container">
            <div class="product-tile">
                <h2 class="product-name">Organic Whole Milk</h2>
                <span class="product-price">$4.99</span>
                <span class="product-brand">Beatrice</span>
                <p class="product-description">Fresh organic whole milk, 3.25% M.F.</p>
                <img class="product-image" src="/images/milk.jpg" alt="Milk">
                <a class="product-link" href="/product/123">View Details</a>
            </div>
            <div class="product-tile">
                <h2 class="product-name">White Bread</h2>
                <span class="product-price">$2.49</span>
                <span class="product-brand">Wonder</span>
                <p class="product-description">Soft white sandwich bread</p>
                <img class="product-image" src="/images/bread.jpg" alt="Bread">
                <a class="product-link" href="/product/456">View Details</a>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def mock_search_results_html():
    """Sample HTML search results page."""
    return """
    <div class="search-results">
        <div class="product-item" data-product-id="1">
            <h3 class="title">Test Product 1</h3>
            <span class="price">$3.99</span>
            <span class="brand">Brand A</span>
        </div>
        <div class="product-item" data-product-id="2">
            <h3 class="title">Test Product 2</h3>
            <span class="price">$7.50</span>
            <span class="brand">Brand B</span>
        </div>
        <div class="product-item" data-product-id="3">
            <h3 class="title">Test Product 3</h3>
            <span class="price">$12.99</span>
            <span class="brand">Brand C</span>
        </div>
    </div>
    """


@pytest.fixture
def mock_api_response():
    """Sample API response for alternative data sources."""
    return {
        "products": [
            {
                "name": "API Product 1",
                "price": {"amount": 5.99, "currency": "CAD"},
                "brand": "API Brand",
                "description": "Product from API",
                "available": True,
                "url": "/api-product-1"
            },
            {
                "name": "API Product 2", 
                "price": 8.50,
                "brand": "Another Brand",
                "description": "Second API product",
                "available": True,
                "image": "/images/api-product-2.jpg"
            }
        ],
        "total": 2,
        "page": 1
    }


@pytest.fixture
def mock_browser_profiles():
    """Mock browser profile paths."""
    return {
        "chrome": "/Users/test/Library/Application Support/Google/Chrome/Default",
        "chrome_user_data": "/Users/test/Library/Application Support/Google/Chrome",
        "firefox": "/Users/test/Library/Application Support/Firefox/Profiles",
        "safari": "/Users/test/Library/Safari"
    }


@pytest.fixture
def mock_playwright_browser():
    """Mock Playwright browser for testing."""
    browser = AsyncMock()
    browser.new_context = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_playwright_page():
    """Mock Playwright page for testing."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    page.title = AsyncMock(return_value="Test Page")
    page.close = AsyncMock()
    page.mouse = AsyncMock()
    page.evaluate = AsyncMock()
    page.wait_for_selector = AsyncMock()
    return page


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    session = AsyncMock()
    
    # Mock response
    response = AsyncMock()
    response.status = 200
    response.text = AsyncMock(return_value="<html><body>Mock response</body></html>")
    response.json = AsyncMock(return_value={"test": "data"})
    response.url = "https://test.com"
    
    session.get.return_value.__aenter__.return_value = response
    session.close = AsyncMock()
    
    return session


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "browser: marks tests that require browser automation"
    )


# Utility functions for tests
@pytest.fixture
def assert_valid_product():
    """Helper function to assert valid product properties."""
    def _assert_valid_product(product: Product):
        assert isinstance(product, Product)
        assert product.name is not None and len(product.name) > 0
        assert product.price > 0
        assert product.currency in [Currency.CAD, Currency.USD]
        assert product.store_id is not None and len(product.store_id) > 0
        assert isinstance(product.keywords, list)
        assert product.in_stock is not None
        
    return _assert_valid_product


@pytest.fixture
def create_test_product():
    """Helper function to create test products."""
    def _create_test_product(name: str, price: float, store_id: str = "test_store", **kwargs):
        return Product(
            name=name,
            price=Decimal(str(price)),
            currency=Currency.CAD,
            store_id=store_id,
            keywords=[name.lower()],
            **kwargs
        )
        
    return _create_test_product


# Skip conditions for environment-specific tests
skip_if_no_browser = pytest.mark.skipif(
    not any([
        # Check for browser installations
        # These would be more complex checks in practice
        True  # Simplified for now
    ]),
    reason="No browser installation found"
)

skip_if_no_clipboard = pytest.mark.skipif(
    False,  # pyperclip should work on most systems
    reason="Clipboard functionality not available"
)

skip_if_ci = pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="Skipping in CI environment"
) if 'os' in locals() else lambda x: x