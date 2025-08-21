"""
Recipe data model.
"""

from typing import List, Optional

from pydantic import Field, HttpUrl

from .base import BaseEntity
from .ingredient import Ingredient


class Recipe(BaseEntity):
    """Represents a recipe with ingredients."""
    
    name: str = Field(..., description="Name of the recipe")
    description: Optional[str] = Field(None, description="Recipe description")
    servings: int = Field(..., gt=0, description="Number of servings")
    prep_time_minutes: Optional[int] = Field(None, ge=0, description="Preparation time in minutes")
    cook_time_minutes: Optional[int] = Field(None, ge=0, description="Cooking time in minutes")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients")
    instructions: List[str] = Field(default_factory=list, description="Cooking instructions")
    tags: List[str] = Field(default_factory=list, description="Recipe tags (e.g., 'vegetarian', 'quick')")
    source_url: Optional[HttpUrl] = Field(None, description="URL source of the recipe")
    
    @property
    def total_time_minutes(self) -> Optional[int]:
        """Calculate total time if both prep and cook times are available."""
        if self.prep_time_minutes is not None and self.cook_time_minutes is not None:
            return self.prep_time_minutes + self.cook_time_minutes
        return None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Chocolate Chip Cookies",
                "description": "Classic homemade chocolate chip cookies",
                "servings": 24,
                "prep_time_minutes": 15,
                "cook_time_minutes": 12,
                "tags": ["dessert", "baking", "cookies"],
                "ingredients": [
                    {
                        "name": "all-purpose flour",
                        "quantity": 2.25,
                        "unit": "cups",
                        "category": "baking"
                    }
                ]
            }
        }
    }