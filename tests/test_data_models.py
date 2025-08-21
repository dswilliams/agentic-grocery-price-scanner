"""
Tests for data models.
"""

import pytest
from decimal import Decimal
from uuid import UUID

from agentic_grocery_price_scanner.data_models import (
    Ingredient, Recipe, Product, Store, StoreConfig, UnitType, Currency
)


class TestIngredient:
    """Test Ingredient model."""
    
    def test_ingredient_creation(self):
        """Test basic ingredient creation."""
        ingredient = Ingredient(
            name="whole milk",
            quantity=2.0,
            unit=UnitType.CUPS,
            category="dairy"
        )
        
        assert ingredient.name == "whole milk"
        assert ingredient.quantity == 2.0
        assert ingredient.unit == UnitType.CUPS
        assert ingredient.category == "dairy"
        assert isinstance(ingredient.id, UUID)
    
    def test_ingredient_alternatives(self):
        """Test ingredient with alternatives."""
        ingredient = Ingredient(
            name="whole milk",
            quantity=1.0,
            unit=UnitType.CUPS,
            alternatives=["2% milk", "milk"]
        )
        
        assert len(ingredient.alternatives) == 2
        assert "2% milk" in ingredient.alternatives


class TestRecipe:
    """Test Recipe model."""
    
    def test_recipe_creation(self):
        """Test basic recipe creation."""
        ingredients = [
            Ingredient(name="flour", quantity=2.0, unit=UnitType.CUPS),
            Ingredient(name="sugar", quantity=1.0, unit=UnitType.CUPS)
        ]
        
        recipe = Recipe(
            name="Test Recipe",
            servings=4,
            ingredients=ingredients,
            prep_time_minutes=15,
            cook_time_minutes=30
        )
        
        assert recipe.name == "Test Recipe"
        assert recipe.servings == 4
        assert len(recipe.ingredients) == 2
        assert recipe.total_time_minutes == 45
    
    def test_recipe_total_time_none(self):
        """Test recipe with missing time information."""
        recipe = Recipe(
            name="Test Recipe",
            servings=4,
            ingredients=[],
            prep_time_minutes=15
            # cook_time_minutes not provided
        )
        
        assert recipe.total_time_minutes is None


class TestProduct:
    """Test Product model."""
    
    def test_product_creation(self):
        """Test basic product creation."""
        product = Product(
            name="Organic Milk",
            brand="Test Brand",
            price=Decimal("5.99"),
            store_id="test_store",
            size=1.0,
            size_unit=UnitType.LITERS
        )
        
        assert product.name == "Organic Milk"
        assert product.price == Decimal("5.99")
        assert product.currency == Currency.CAD
        assert product.store_id == "test_store"
        assert product.in_stock is True
    
    def test_product_sale_price(self):
        """Test product with sale pricing."""
        product = Product(
            name="Sale Item",
            price=Decimal("10.00"),
            store_id="test_store",
            on_sale=True,
            sale_price=Decimal("7.99")
        )
        
        assert product.on_sale is True
        assert product.sale_price == Decimal("7.99")


class TestStoreConfig:
    """Test StoreConfig model."""
    
    def test_store_config_creation(self):
        """Test basic store config creation."""
        config = StoreConfig(
            name="Test Store",
            store_id="test_store",
            base_url="https://example.com",
            search_url_template="https://example.com/search?q={query}"
        )
        
        assert config.name == "Test Store"
        assert config.store_id == "test_store"
        assert config.currency == Currency.CAD
        assert config.rate_limit_seconds == 1.0  # default value
    
    def test_store_config_with_selectors(self):
        """Test store config with CSS selectors."""
        selectors = {
            "product_name": ".product-name",
            "price": ".price"
        }
        
        config = StoreConfig(
            name="Test Store",
            store_id="test_store", 
            base_url="https://example.com",
            search_url_template="https://example.com/search?q={query}",
            selectors=selectors
        )
        
        assert config.selectors["product_name"] == ".product-name"
        assert config.selectors["price"] == ".price"


class TestStore:
    """Test Store model."""
    
    def test_store_creation(self):
        """Test basic store creation."""
        config = StoreConfig(
            name="Test Store",
            store_id="test_store",
            base_url="https://example.com",
            search_url_template="https://example.com/search?q={query}"
        )
        
        store = Store(config=config)
        
        assert store.config.name == "Test Store"
        assert store.is_active is True
        assert store.error_count == 0
        assert store.store_id == "test_store"
        assert store.name == "Test Store"