"""
Database utilities and schema management.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from ..config import get_settings


logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for the grocery scanner."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager."""
        settings = get_settings()
        self.db_path = db_path or settings.database.db_path
        self.create_tables = settings.database.create_tables
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.create_tables:
            self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self) -> None:
        """Initialize database with required tables."""
        logger.info(f"Initializing database: {self.db_path}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create recipes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recipes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    servings INTEGER NOT NULL,
                    prep_time_minutes INTEGER,
                    cook_time_minutes INTEGER,
                    instructions TEXT,  -- JSON array
                    tags TEXT,  -- JSON array
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create ingredients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingredients (
                    id TEXT PRIMARY KEY,
                    recipe_id TEXT,
                    name TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    unit TEXT NOT NULL,
                    category TEXT,
                    notes TEXT,
                    alternatives TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recipe_id) REFERENCES recipes (id) ON DELETE CASCADE
                )
            """)
            
            # Create stores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stores (
                    id TEXT PRIMARY KEY,
                    store_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    currency TEXT DEFAULT 'CAD',
                    is_active BOOLEAN DEFAULT 1,
                    config TEXT,  -- JSON configuration
                    last_scraped TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    total_products INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create products table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    brand TEXT,
                    price DECIMAL(10, 2) NOT NULL,
                    currency TEXT DEFAULT 'CAD',
                    size REAL,
                    size_unit TEXT,
                    price_per_unit DECIMAL(10, 2),
                    store_id TEXT NOT NULL,
                    sku TEXT,
                    barcode TEXT,
                    category TEXT,
                    subcategory TEXT,
                    description TEXT,
                    image_url TEXT,
                    product_url TEXT,
                    in_stock BOOLEAN DEFAULT 1,
                    on_sale BOOLEAN DEFAULT 0,
                    sale_price DECIMAL(10, 2),
                    nutrition_info TEXT,  -- JSON
                    keywords TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (store_id) REFERENCES stores (store_id)
                )
            """)
            
            # Create ingredient_product_matches table for ML matching results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingredient_product_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingredient_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    match_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ingredient_id) REFERENCES ingredients (id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE,
                    UNIQUE(ingredient_id, product_id)
                )
            """)
            
            # Create shopping_lists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shopping_lists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    recipe_ids TEXT,  -- JSON array of recipe IDs
                    total_estimated_cost DECIMAL(10, 2),
                    optimization_strategy TEXT DEFAULT 'cheapest',
                    store_constraints TEXT,  -- JSON array of allowed store IDs
                    budget_limit DECIMAL(10, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create shopping_list_items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shopping_list_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shopping_list_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    quantity INTEGER DEFAULT 1,
                    notes TEXT,
                    FOREIGN KEY (shopping_list_id) REFERENCES shopping_lists (id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_store_id ON products (store_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products (name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_price ON products (price)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingredients_recipe_id ON ingredients (recipe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_ingredient ON ingredient_product_matches (ingredient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_product ON ingredient_product_matches (product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_confidence ON ingredient_product_matches (confidence_score)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def execute_query(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch one row as dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def fetch_all(self, query: str, params: tuple = ()) -> list[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_table_info(self, table_name: str) -> list[Dict[str, Any]]:
        """Get table schema information."""
        return self.fetch_all(f"PRAGMA table_info({table_name})")
    
    def get_table_count(self, table_name: str) -> int:
        """Get number of rows in table."""
        result = self.fetch_one(f"SELECT COUNT(*) as count FROM {table_name}")
        return result["count"] if result else 0


def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager()