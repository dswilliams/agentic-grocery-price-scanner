"""
Product data model.
"""

from decimal import Decimal
from typing import List, Optional

from pydantic import Field, HttpUrl

from .base import BaseEntity, Currency, UnitType, DataCollectionMethod


class Product(BaseEntity):
    """Represents a grocery product from a store."""
    
    name: str = Field(..., description="Product name")
    brand: Optional[str] = Field(None, description="Product brand")
    price: Decimal = Field(..., ge=0, description="Current price")
    currency: Currency = Field(default=Currency.CAD, description="Price currency")
    size: Optional[float] = Field(None, gt=0, description="Product size/weight")
    size_unit: Optional[UnitType] = Field(None, description="Unit of size/weight")
    price_per_unit: Optional[Decimal] = Field(None, ge=0, description="Price per unit (calculated)")
    store_id: str = Field(..., description="Store identifier")
    sku: Optional[str] = Field(None, description="Store SKU/product code")
    barcode: Optional[str] = Field(None, description="Product barcode")
    category: Optional[str] = Field(None, description="Product category")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    description: Optional[str] = Field(None, description="Product description")
    image_url: Optional[HttpUrl] = Field(None, description="Product image URL")
    product_url: Optional[HttpUrl] = Field(None, description="Product page URL")
    in_stock: bool = Field(default=True, description="Whether product is in stock")
    on_sale: bool = Field(default=False, description="Whether product is on sale")
    sale_price: Optional[Decimal] = Field(None, ge=0, description="Sale price if on sale")
    nutrition_info: Optional[dict] = Field(None, description="Nutritional information")
    keywords: List[str] = Field(default_factory=list, description="Search keywords for matching")
    
    # Data collection metadata
    collection_method: DataCollectionMethod = Field(
        default=DataCollectionMethod.AUTOMATED_STEALTH, 
        description="Method used to collect this product data"
    )
    confidence_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence in data accuracy (0.0-1.0)"
    )
    source_metadata: Optional[dict] = Field(
        None, 
        description="Additional metadata about data source and collection"
    )
    
    def calculate_price_per_unit(self) -> Optional[Decimal]:
        """Calculate price per unit if size information is available."""
        if self.size and self.size > 0:
            current_price = self.sale_price if self.on_sale and self.sale_price else self.price
            return current_price / Decimal(str(self.size))
        return None
    
    def get_collection_confidence_weight(self) -> float:
        """Get confidence weight based on collection method and score.
        
        Returns:
            Weighted confidence score considering collection method reliability.
        """
        method_weights = {
            DataCollectionMethod.HUMAN_BROWSER: 1.0,      # Highest reliability
            DataCollectionMethod.CLIPBOARD_MANUAL: 0.95,  # High reliability (human verified)
            DataCollectionMethod.API_DIRECT: 0.9,         # High reliability (official API)
            DataCollectionMethod.AUTOMATED_STEALTH: 0.8,  # Good reliability but automated
            DataCollectionMethod.MOCK_DATA: 0.1,          # Low reliability (test data)
        }
        
        method_weight = method_weights.get(self.collection_method, 0.5)
        return self.confidence_score * method_weight
    
    def __post_init__(self):
        """Calculate price per unit after initialization."""
        if self.size and self.price_per_unit is None:
            self.price_per_unit = self.calculate_price_per_unit()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Organic Whole Milk",
                "brand": "Organic Valley",
                "price": 5.99,
                "currency": "CAD",
                "size": 1.0,
                "size_unit": "l",
                "store_id": "metro_ca",
                "category": "dairy",
                "in_stock": True,
                "keywords": ["milk", "organic", "whole milk", "dairy"],
                "collection_method": "automated_stealth",
                "confidence_score": 0.95,
                "source_metadata": {"scraper": "stealth_scraper", "timestamp": "2024-01-01T12:00:00Z"}
            }
        }
    }