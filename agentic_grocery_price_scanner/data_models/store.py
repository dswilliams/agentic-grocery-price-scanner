"""
Store data model.
"""

from typing import Dict, List, Optional

from pydantic import Field, HttpUrl

from .base import BaseEntity, Currency


class StoreConfig(BaseEntity):
    """Configuration for a specific store."""
    
    name: str = Field(..., description="Store name")
    store_id: str = Field(..., description="Unique store identifier")
    base_url: HttpUrl = Field(..., description="Store website base URL")
    currency: Currency = Field(default=Currency.CAD, description="Store currency")
    
    # Scraping configuration
    search_url_template: str = Field(..., description="URL template for search queries")
    rate_limit_seconds: float = Field(default=1.0, ge=0.1, description="Rate limit between requests")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=30, ge=5, description="Request timeout")
    
    # CSS selectors for scraping
    selectors: Dict[str, str] = Field(default_factory=dict, description="CSS selectors for data extraction")
    
    # Headers and user agent
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for requests")
    
    # Location/delivery settings
    postal_code: Optional[str] = Field(None, description="Postal code for location-based pricing")
    delivery_available: bool = Field(default=False, description="Whether delivery is available")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Metro",
                "store_id": "metro_ca",
                "base_url": "https://www.metro.ca",
                "search_url_template": "https://www.metro.ca/en/online-grocery/search?filter={query}",
                "rate_limit_seconds": 2.0,
                "selectors": {
                    "product_name": ".product-name",
                    "price": ".price",
                    "brand": ".brand-name"
                },
                "headers": {
                    "User-Agent": "Mozilla/5.0 (compatible; GroceryScanner/1.0)"
                }
            }
        }
    }


class Store(BaseEntity):
    """Represents a grocery store with current status."""
    
    config: StoreConfig = Field(..., description="Store configuration")
    is_active: bool = Field(default=True, description="Whether store is currently active")
    last_scraped: Optional[str] = Field(None, description="Last successful scrape timestamp")
    error_count: int = Field(default=0, ge=0, description="Current error count")
    total_products: int = Field(default=0, ge=0, description="Total products scraped")
    
    @property
    def store_id(self) -> str:
        """Get store ID from config."""
        return self.config.store_id
    
    @property
    def name(self) -> str:
        """Get store name from config."""
        return self.config.name