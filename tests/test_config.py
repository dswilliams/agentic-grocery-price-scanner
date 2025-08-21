"""
Tests for configuration system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from agentic_grocery_price_scanner.config import get_settings, load_store_configs


class TestSettings:
    """Test settings configuration."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = get_settings()
        
        assert settings.app_name == "Agentic Grocery Price Scanner"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.database.db_path == "db/grocery_scanner.db"
        assert settings.vector_db.host == "localhost"
        assert settings.llm.default_model == "llama2"


class TestStoreLoader:
    """Test store configuration loader."""
    
    def test_load_store_configs(self):
        """Test loading store configurations from YAML."""
        # Create temporary YAML config
        config_data = {
            "stores": {
                "test_store": {
                    "name": "Test Store",
                    "base_url": "https://example.com",
                    "search_url_template": "https://example.com/search?q={query}",
                    "active": True,
                    "selectors": {
                        "product_name": ".product-name"
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            stores = load_store_configs(config_file)
            
            assert len(stores) == 1
            assert "test_store" in stores
            
            store = stores["test_store"]
            assert store.name == "Test Store"
            assert store.store_id == "test_store"
            assert store.selectors["product_name"] == ".product-name"
            
        finally:
            Path(config_file).unlink()
    
    def test_load_missing_config_file(self):
        """Test loading from non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_store_configs("non_existent_file.yaml")