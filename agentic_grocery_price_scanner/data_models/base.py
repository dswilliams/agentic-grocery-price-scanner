"""
Base data models and shared types.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    """Base model with common fields."""
    
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    }


class UnitType(str, Enum):
    """Unit types for ingredients and products."""
    
    GRAMS = "g"
    KILOGRAMS = "kg"
    POUNDS = "lbs"
    OUNCES = "oz"
    MILLILITERS = "ml"
    LITERS = "l"
    CUPS = "cups"
    TABLESPOONS = "tbsp"
    TEASPOONS = "tsp"
    PIECES = "pieces"
    PACKAGES = "packages"


class Currency(str, Enum):
    """Supported currencies."""
    
    CAD = "CAD"
    USD = "USD"


class DataCollectionMethod(str, Enum):
    """Methods used to collect product data."""
    
    AUTOMATED_STEALTH = "automated_stealth"  # Layer 1: stealth_scraper, crawl4ai_client, advanced_scraper
    HUMAN_BROWSER = "human_browser"          # Layer 2: human_browser_scraper
    CLIPBOARD_MANUAL = "clipboard_manual"    # Layer 3: clipboard_scraper
    API_DIRECT = "api_direct"               # Direct API calls
    MOCK_DATA = "mock_data"                 # Mock data for testing