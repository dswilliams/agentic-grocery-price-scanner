"""
Application settings and configuration management.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    db_path: str = Field(default="db/grocery_scanner.db", description="SQLite database path")
    create_tables: bool = Field(default=True, description="Create tables on startup")
    
    model_config = {"env_prefix": "DB_"}


class VectorDBSettings(BaseSettings):
    """Vector database configuration for Qdrant."""
    
    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(default="products", description="Collection name for products")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    model_config = {"env_prefix": "QDRANT_"}


class LLMSettings(BaseSettings):
    """Language model configuration."""
    
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    default_model: str = Field(default="llama2", description="Default model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=1000, ge=1, description="Maximum tokens per response")
    
    model_config = {"env_prefix": "LLM_"}


class ScrapingSettings(BaseSettings):
    """Web scraping configuration."""
    
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; GroceryScanner/1.0)",
        description="User agent for web scraping"
    )
    default_timeout: int = Field(default=30, ge=1, description="Default request timeout")
    default_rate_limit: float = Field(default=1.0, ge=0.1, description="Default rate limit between requests")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    use_selenium: bool = Field(default=False, description="Whether to use Selenium for JavaScript rendering")
    headless_browser: bool = Field(default=True, description="Run browser in headless mode")
    
    model_config = {"env_prefix": "SCRAPING_"}


class Settings(BaseSettings):
    """Main application settings."""
    
    app_name: str = Field(default="Agentic Grocery Price Scanner", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Log file path")
    config_file: str = Field(default="config/stores.yaml", description="Store configuration file path")
    
    # Database settings (flattened)
    db_path: str = Field(default="db/grocery_scanner.db", description="SQLite database path")
    db_create_tables: bool = Field(default=True, description="Create tables on startup")
    
    # Vector DB settings (flattened)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection_name: str = Field(default="products", description="Collection name for products")
    qdrant_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    # LLM settings (flattened)
    llm_ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    llm_default_model: str = Field(default="llama2", description="Default model name")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    llm_max_tokens: int = Field(default=1000, ge=1, description="Maximum tokens per response")
    
    # Scraping settings (flattened)
    scraping_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; GroceryScanner/1.0)",
        description="User agent for web scraping"
    )
    scraping_default_timeout: int = Field(default=30, ge=1, description="Default request timeout")
    scraping_default_rate_limit: float = Field(default=1.0, ge=0.1, description="Default rate limit between requests")
    scraping_max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    scraping_use_selenium: bool = Field(default=False, description="Whether to use Selenium for JavaScript rendering")
    scraping_headless_browser: bool = Field(default=True, description="Run browser in headless mode")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
    
    @property
    def database(self) -> DatabaseSettings:
        """Get database settings."""
        return DatabaseSettings(
            db_path=self.db_path,
            create_tables=self.db_create_tables
        )
    
    @property
    def vector_db(self) -> VectorDBSettings:
        """Get vector database settings."""
        return VectorDBSettings(
            host=self.qdrant_host,
            port=self.qdrant_port,
            collection_name=self.qdrant_collection_name,
            embedding_model=self.qdrant_embedding_model
        )
    
    @property
    def llm(self) -> LLMSettings:
        """Get LLM settings."""
        return LLMSettings(
            ollama_base_url=self.llm_ollama_base_url,
            default_model=self.llm_default_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )
    
    @property
    def scraping(self) -> ScrapingSettings:
        """Get scraping settings."""
        return ScrapingSettings(
            user_agent=self.scraping_user_agent,
            default_timeout=self.scraping_default_timeout,
            default_rate_limit=self.scraping_default_rate_limit,
            max_retries=self.scraping_max_retries,
            use_selenium=self.scraping_use_selenium,
            headless_browser=self.scraping_headless_browser
        )


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()