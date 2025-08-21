"""
Tests for database utilities.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from agentic_grocery_price_scanner.utils.database import DatabaseManager


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_database_initialization(self):
        """Test database initialization with tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            db_manager = DatabaseManager(str(db_path))
            
            # Check that database file was created
            assert db_path.exists()
            
            # Check that tables were created
            tables = db_manager.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            table_names = [table["name"] for table in tables]
            
            expected_tables = [
                "recipes", "ingredients", "stores", "products",
                "ingredient_product_matches", "shopping_lists", "shopping_list_items"
            ]
            
            for table in expected_tables:
                assert table in table_names
    
    def test_database_operations(self):
        """Test basic database operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db_manager = DatabaseManager(str(db_path))
            
            # Test insert and fetch
            db_manager.execute_query(
                "INSERT INTO stores (id, store_id, name, base_url) VALUES (?, ?, ?, ?)",
                ("test-id", "test_store", "Test Store", "https://example.com")
            )
            
            result = db_manager.fetch_one(
                "SELECT * FROM stores WHERE store_id = ?",
                ("test_store",)
            )
            
            assert result is not None
            assert result["name"] == "Test Store"
            assert result["base_url"] == "https://example.com"
    
    def test_table_info(self):
        """Test getting table schema information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db_manager = DatabaseManager(str(db_path))
            
            info = db_manager.get_table_info("products")
            
            # Check that we got column information
            assert len(info) > 0
            column_names = [col["name"] for col in info]
            
            expected_columns = ["id", "name", "price", "store_id"]
            for col in expected_columns:
                assert col in column_names
    
    def test_table_count(self):
        """Test getting table row count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db_manager = DatabaseManager(str(db_path))
            
            # Initially empty
            count = db_manager.get_table_count("stores")
            assert count == 0
            
            # Add a record
            db_manager.execute_query(
                "INSERT INTO stores (id, store_id, name, base_url) VALUES (?, ?, ?, ?)",
                ("test-id", "test_store", "Test Store", "https://example.com")
            )
            
            # Count should be 1
            count = db_manager.get_table_count("stores")
            assert count == 1