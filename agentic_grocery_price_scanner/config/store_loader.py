"""
Store configuration loader.
"""

import yaml
from pathlib import Path
from typing import Dict, List

from ..data_models import StoreConfig


def load_store_configs(config_file: str = "config/stores.yaml") -> Dict[str, StoreConfig]:
    """
    Load store configurations from YAML file.
    
    Args:
        config_file: Path to the store configuration file
        
    Returns:
        Dictionary mapping store IDs to StoreConfig objects
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Store configuration file not found: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
    
    stores = {}
    for store_id, store_data in config_data.get('stores', {}).items():
        # Ensure store_id is set
        store_data['store_id'] = store_id
        stores[store_id] = StoreConfig(**store_data)
    
    return stores


def get_active_stores(config_file: str = "config/stores.yaml") -> List[StoreConfig]:
    """
    Get list of active store configurations.
    
    Args:
        config_file: Path to the store configuration file
        
    Returns:
        List of active StoreConfig objects
    """
    stores = load_store_configs(config_file)
    return [store for store in stores.values() if getattr(store, 'active', True)]