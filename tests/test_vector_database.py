"""
Comprehensive tests for vector database integration with real grocery data.
"""

import pytest
import asyncio
from decimal import Decimal
from typing import List
from uuid import uuid4

from agentic_grocery_price_scanner.vector_db import (
    QdrantVectorDB, 
    EmbeddingService, 
    ProductNormalizer,
    ScraperVectorIntegration
)
from agentic_grocery_price_scanner.data_models.product import Product
from agentic_grocery_price_scanner.data_models.base import DataCollectionMethod, Currency, UnitType


class TestVectorDatabase:
    """Test suite for vector database functionality."""
    
    @pytest.fixture
    def vector_db(self):
        """Create in-memory vector database for testing."""
        return QdrantVectorDB(in_memory=True, collection_name="test_collection")
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing."""
        return EmbeddingService()
    
    @pytest.fixture
    def product_normalizer(self):
        """Create product normalizer for testing."""
        return ProductNormalizer()
    
    @pytest.fixture
    def sample_metro_products(self):
        """Real Metro.ca grocery products for testing."""
        return [
            Product(
                id=uuid4(),
                name="Organic Whole Milk",
                brand="Neilson",
                price=Decimal("5.99"),
                currency=Currency.CAD,
                size=1.0,
                size_unit=UnitType.LITERS,
                store_id="metro_ca",
                category="dairy",
                subcategory="milk",
                description="Fresh organic whole milk",
                in_stock=True,
                keywords=["milk", "organic", "whole", "dairy", "neilson"],
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.95,
                source_metadata={"scraper": "stealth", "page_url": "https://metro.ca/milk"}
            ),
            Product(
                id=uuid4(),
                name="Extra Virgin Olive Oil",
                brand="Bertolli",
                price=Decimal("12.99"),
                currency=Currency.CAD,
                size=500.0,
                size_unit=UnitType.MILLILITERS,
                store_id="metro_ca",
                category="pantry",
                subcategory="oils",
                description="Premium extra virgin olive oil",
                in_stock=True,
                on_sale=True,
                sale_price=Decimal("9.99"),
                keywords=["olive oil", "extra virgin", "bertolli", "cooking", "pantry"],
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=0.98,
                source_metadata={"scraper": "human_browser", "verified": True}
            ),
            Product(
                id=uuid4(),
                name="Fresh Bananas",
                brand=None,
                price=Decimal("1.99"),
                currency=Currency.CAD,
                size=1.0,
                size_unit=UnitType.POUNDS,
                store_id="metro_ca",
                category="produce",
                subcategory="fruits",
                description="Fresh yellow bananas",
                in_stock=True,
                keywords=["bananas", "fresh", "fruit", "produce"],
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=0.85,
                source_metadata={"scraper": "clipboard", "manual_entry": True}
            )
        ]
    
    @pytest.fixture
    def sample_walmart_products(self):
        """Real Walmart.ca grocery products for testing."""
        return [
            Product(
                id=uuid4(),
                name="Great Value 2% Milk",
                brand="Great Value",
                price=Decimal("4.97"),
                currency=Currency.CAD,
                size=4.0,
                size_unit=UnitType.LITERS,
                store_id="walmart_ca",
                category="dairy",
                subcategory="milk",
                description="Great Value 2% partly skimmed milk",
                in_stock=True,
                keywords=["milk", "2%", "great value", "dairy", "partly skimmed"],
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.90,
                source_metadata={"scraper": "stealth", "page_url": "https://walmart.ca/milk"}
            ),
            Product(
                id=uuid4(),
                name="President's Choice Olive Oil",
                brand="President's Choice",
                price=Decimal("8.98"),
                currency=Currency.CAD,
                size=750.0,
                size_unit=UnitType.MILLILITERS,
                store_id="walmart_ca",
                category="pantry",
                subcategory="oils",
                description="PC Extra Virgin Olive Oil",
                in_stock=True,
                keywords=["olive oil", "presidents choice", "extra virgin", "pc", "cooking"],
                collection_method=DataCollectionMethod.API_DIRECT,
                confidence_score=0.99,
                source_metadata={"scraper": "api", "api_endpoint": "products/search"}
            )
        ]
    
    @pytest.mark.unit
    def test_embedding_service_initialization(self, embedding_service):
        """Test embedding service loads model correctly."""
        assert embedding_service.model is not None
        assert embedding_service.embedding_dimension > 0
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
    
    @pytest.mark.unit
    def test_product_text_creation(self, embedding_service, sample_metro_products):
        """Test product text representation for embeddings."""
        product = sample_metro_products[0]  # Organic Whole Milk
        text = embedding_service.create_product_text(product)
        
        assert "Organic Whole Milk" in text
        assert "Neilson" in text
        assert "dairy" in text
        assert "1.0 l" in text
        assert "milk" in text
        assert "organic" in text
    
    @pytest.mark.unit
    def test_product_embedding_generation(self, embedding_service, sample_metro_products):
        """Test embedding generation for products."""
        product = sample_metro_products[0]
        embedding = embedding_service.encode_product(product)
        
        assert embedding is not None
        assert len(embedding) == embedding_service.embedding_dimension
        assert embedding.dtype.name.startswith('float')
    
    @pytest.mark.unit
    def test_batch_embedding_generation(self, embedding_service, sample_metro_products):
        """Test batch embedding generation."""
        embeddings = embedding_service.encode_products(sample_metro_products)
        
        assert len(embeddings) == len(sample_metro_products)
        for embedding in embeddings:
            assert len(embedding) == embedding_service.embedding_dimension
    
    @pytest.mark.unit
    def test_vector_db_initialization(self, vector_db):
        """Test vector database initialization."""
        assert vector_db.client is not None
        assert vector_db.collection_name == "test_collection"
        
        stats = vector_db.get_collection_stats()
        assert "total_vectors" in stats
    
    @pytest.mark.integration
    def test_add_single_product(self, vector_db, sample_metro_products):
        """Test adding a single product to vector database."""
        product = sample_metro_products[0]
        vector_db.add_product(product)
        
        # Retrieve the product
        retrieved = vector_db.get_product_by_id(product.id)
        assert retrieved is not None
        assert retrieved.name == product.name
        assert retrieved.collection_method == product.collection_method
    
    @pytest.mark.integration
    def test_add_multiple_products(self, vector_db, sample_metro_products, sample_walmart_products):
        """Test adding multiple products to vector database."""
        all_products = sample_metro_products + sample_walmart_products
        vector_db.add_products(all_products)
        
        stats = vector_db.get_collection_stats()
        assert stats["total_vectors"] == len(all_products)
    
    @pytest.mark.integration
    def test_similarity_search_basic(self, vector_db, sample_metro_products, sample_walmart_products):
        """Test basic similarity search functionality."""
        all_products = sample_metro_products + sample_walmart_products
        vector_db.add_products(all_products)
        
        # Search for milk products
        results = vector_db.search_similar_products("milk", limit=5)
        
        assert len(results) > 0
        
        # Should find both milk products
        milk_products = [p for p, _ in results if "milk" in p.name.lower()]
        assert len(milk_products) >= 2
        
        # Check confidence weighting
        for product, score in results:
            weighted_confidence = product.get_collection_confidence_weight()
            assert 0.0 <= weighted_confidence <= 1.0
    
    @pytest.mark.integration
    def test_similarity_search_with_filters(self, vector_db, sample_metro_products, sample_walmart_products):
        """Test similarity search with various filters."""
        all_products = sample_metro_products + sample_walmart_products
        vector_db.add_products(all_products)
        
        # Filter by store
        metro_results = vector_db.search_similar_products(
            "milk", 
            store_filter=["metro_ca"]
        )
        
        for product, _ in metro_results:
            assert product.store_id == "metro_ca"
        
        # Filter by collection method
        stealth_results = vector_db.search_similar_products(
            "milk",
            collection_method_filter=[DataCollectionMethod.AUTOMATED_STEALTH]
        )
        
        for product, _ in stealth_results:
            assert product.collection_method == DataCollectionMethod.AUTOMATED_STEALTH
        
        # Filter by confidence
        high_confidence_results = vector_db.search_similar_products(
            "oil",
            min_confidence=0.95
        )
        
        for product, _ in high_confidence_results:
            assert product.confidence_score >= 0.95
    
    @pytest.mark.unit
    def test_product_normalizer_stealth_data(self, product_normalizer):
        """Test normalizing data from stealth scraper."""
        raw_data = {
            "name": "Organic Whole Milk 1L",
            "brand": "Neilson",
            "price": "$5.99",
            "size": "1L",
            "category": "Dairy",
            "in_stock": True,
            "image_url": "https://metro.ca/images/milk.jpg"
        }
        
        product = product_normalizer.normalize_stealth_scraper_data(raw_data, "metro_ca")
        
        assert product.name == "Organic Whole Milk 1L"
        assert product.brand == "Neilson"
        assert product.price == Decimal("5.99")
        assert product.size == 1.0
        assert product.size_unit == UnitType.LITERS
        assert product.collection_method == DataCollectionMethod.AUTOMATED_STEALTH
        assert 0.8 <= product.confidence_score <= 1.0  # Good data completeness
    
    @pytest.mark.unit
    def test_product_normalizer_clipboard_data(self, product_normalizer):
        """Test normalizing clipboard data."""
        clipboard_text = """Fresh Bananas
        Price: $1.99/lb
        Metro.ca
        In Stock
        https://metro.ca/en/online-grocery/aisles/fruits-vegetables"""
        
        product = product_normalizer.normalize_clipboard_data(clipboard_text, "metro_ca")
        
        assert product is not None
        assert product.name == "Fresh Bananas"
        assert product.price == Decimal("1.99")
        assert product.store_id == "metro_ca"
        assert product.collection_method == DataCollectionMethod.CLIPBOARD_MANUAL
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scraper_integration_service(self, sample_metro_products):
        """Test scraper integration service with all layers."""
        integration = ScraperVectorIntegration(auto_save=False)
        
        # Test stealth scraper integration
        stealth_data = [{
            "name": "Test Milk Product",
            "brand": "Test Brand",
            "price": "4.99",
            "category": "dairy",
            "in_stock": True
        }]
        
        products = await integration.process_stealth_scraper_results(
            stealth_data, "test_store", "test_session"
        )
        
        assert len(products) == 1
        assert products[0].collection_method == DataCollectionMethod.AUTOMATED_STEALTH
        
        # Test clipboard integration
        clipboard_texts = [
            "Premium Milk\nPrice: $6.99\nBrand: Premium Brand"
        ]
        
        clipboard_products = await integration.process_clipboard_data(
            clipboard_texts, "test_session"
        )
        
        assert len(clipboard_products) == 1
        assert clipboard_products[0].collection_method == DataCollectionMethod.CLIPBOARD_MANUAL
        
        # Test statistics
        stats = integration.get_collection_statistics()
        assert "sessions" in stats
        assert "test_session" in stats["sessions"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_layer_similarity_search(self):
        """Test similarity search across all collection methods."""
        integration = ScraperVectorIntegration(auto_save=True)
        
        # Add products from different layers
        stealth_data = [{
            "name": "Stealth Organic Milk",
            "brand": "Organic Valley",
            "price": "5.99",
            "category": "dairy"
        }]
        
        human_data = [{
            "name": "Human Browser Milk",
            "brand": "Local Farm",
            "price": "7.99",
            "category": "dairy",
            "description": "Fresh from local farm"
        }]
        
        clipboard_text = ["Premium Milk\nPrice: $8.99\nOrganic"]
        
        # Process all data
        await integration.process_stealth_scraper_results(stealth_data, "store1")
        await integration.process_human_browser_results(human_data, "store2")
        await integration.process_clipboard_data(clipboard_text)
        
        # Search across all methods
        results = await integration.find_similar_products("organic milk", limit=10)
        
        assert len(results) > 0
        
        # Should find products from multiple collection methods
        methods_found = set(product.collection_method for product, _ in results)
        assert len(methods_found) >= 2
        
        # Verify confidence weighting affects results
        for product, score in results:
            weighted_confidence = product.get_collection_confidence_weight()
            assert score > 0  # Similarity score should be positive
            assert weighted_confidence > 0  # Weighted confidence should be positive
    
    @pytest.mark.performance
    def test_large_scale_vector_operations(self, vector_db):
        """Test vector database performance with larger dataset."""
        import time
        
        # Generate larger dataset
        products = []
        for i in range(100):
            product = Product(
                id=uuid4(),
                name=f"Test Product {i}",
                brand=f"Brand {i % 10}",
                price=Decimal(f"{i % 50 + 1}.99"),
                store_id=f"store_{i % 5}",
                category=f"category_{i % 10}",
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=0.8 + (i % 20) / 100,  # Vary confidence
                keywords=[f"keyword{i}", f"test{i % 10}", "product"]
            )
            products.append(product)
        
        # Time the batch insertion
        start_time = time.time()
        vector_db.add_products(products)
        insert_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert insert_time < 30.0  # 30 seconds for 100 products
        
        # Time similarity search
        start_time = time.time()
        results = vector_db.search_similar_products("test product", limit=20)
        search_time = time.time() - start_time
        
        assert search_time < 5.0  # 5 seconds for search
        assert len(results) > 0
    
    @pytest.mark.integration
    def test_confidence_weighted_ranking(self, vector_db):
        """Test that confidence weighting affects search ranking."""
        # Create products with different confidence scores
        high_confidence_product = Product(
            id=uuid4(),
            name="High Confidence Milk",
            brand="Premium Brand",
            price=Decimal("5.99"),
            store_id="premium_store",
            category="dairy",
            collection_method=DataCollectionMethod.HUMAN_BROWSER,
            confidence_score=0.98,
            keywords=["milk", "premium", "dairy"]
        )
        
        low_confidence_product = Product(
            id=uuid4(),
            name="Low Confidence Milk",
            brand="Generic Brand",
            price=Decimal("4.99"),
            store_id="generic_store",
            category="dairy",
            collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
            confidence_score=0.60,
            keywords=["milk", "generic", "dairy"]
        )
        
        vector_db.add_products([high_confidence_product, low_confidence_product])
        
        # Search with confidence weighting enabled
        results_weighted = vector_db.search_similar_products(
            "premium milk", 
            use_confidence_weighting=True
        )
        
        # Search without confidence weighting
        results_unweighted = vector_db.search_similar_products(
            "premium milk", 
            use_confidence_weighting=False
        )
        
        # Both should return results
        assert len(results_weighted) >= 2
        assert len(results_unweighted) >= 2
        
        # High confidence product should be weighted higher
        high_conf_score_weighted = next(
            score for product, score in results_weighted 
            if product.id == high_confidence_product.id
        )
        
        low_conf_score_weighted = next(
            score for product, score in results_weighted 
            if product.id == low_confidence_product.id
        )
        
        # With weighting, high confidence should score higher
        assert high_conf_score_weighted > low_conf_score_weighted
    
    @pytest.mark.cleanup
    def test_vector_database_cleanup(self, vector_db, sample_metro_products):
        """Test vector database cleanup operations."""
        # Add some products
        vector_db.add_products(sample_metro_products)
        
        # Verify they exist
        stats_before = vector_db.get_collection_stats()
        assert stats_before["total_vectors"] > 0
        
        # Test individual product deletion
        product_to_delete = sample_metro_products[0]
        success = vector_db.delete_product(product_to_delete.id)
        assert success
        
        # Verify deletion
        retrieved = vector_db.get_product_by_id(product_to_delete.id)
        assert retrieved is None
        
        # Test full collection clearing
        success = vector_db.clear_collection()
        assert success
        
        # Verify collection is empty
        stats_after = vector_db.get_collection_stats()
        assert stats_after["total_vectors"] == 0