"""
Ingredient data model.
"""

from typing import List, Optional

from pydantic import Field

from .base import BaseEntity, UnitType


class Ingredient(BaseEntity):
    """Represents an ingredient in a recipe."""
    
    name: str = Field(..., description="Name of the ingredient")
    quantity: float = Field(..., gt=0, description="Quantity needed")
    unit: UnitType = Field(..., description="Unit of measurement")
    category: Optional[str] = Field(None, description="Ingredient category (e.g., 'dairy', 'produce')")
    notes: Optional[str] = Field(None, description="Additional notes about the ingredient")
    alternatives: List[str] = Field(default_factory=list, description="Alternative ingredient names")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "whole milk",
                "quantity": 2.0,
                "unit": "cups",
                "category": "dairy",
                "alternatives": ["2% milk", "milk"]
            }
        }
    }