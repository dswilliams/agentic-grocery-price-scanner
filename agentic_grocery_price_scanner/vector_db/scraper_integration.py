"""
Integration service connecting all 3 scraping layers with Qdrant vector database.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import asyncio

from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod
from .qdrant_client import QdrantVectorDB
from .product_normalizer import ProductNormalizer

logger = logging.getLogger(__name__)


class ScraperVectorIntegration:
    """Unified service for integrating all scraping layers with vector database."""

    def __init__(
        self,
        vector_db: Optional[QdrantVectorDB] = None,
        normalizer: Optional[ProductNormalizer] = None,
        auto_save: bool = True,
    ):
        """Initialize the integration service.

        Args:
            vector_db: Qdrant vector database client
            normalizer: Product normalization service
            auto_save: Automatically save products to vector database
        """
        self.vector_db = vector_db or QdrantVectorDB()
        self.normalizer = normalizer or ProductNormalizer()
        self.auto_save = auto_save

        # Track products by collection session
        self.session_products: Dict[str, List[Product]] = {}

        logger.info("Scraper-Vector integration service initialized")

    # Layer 1: Automated Stealth Scraping Integration

    async def process_stealth_scraper_results(
        self,
        scraper_results: List[Dict[str, Any]],
        store_id: str,
        session_id: Optional[str] = None,
    ) -> List[Product]:
        """Process results from stealth scraper with vector storage.

        Args:
            scraper_results: Raw data from stealth scraper
            store_id: Store identifier
            session_id: Optional session identifier for grouping

        Returns:
            List of normalized Product objects
        """
        logger.info(
            f"Processing {len(scraper_results)} stealth scraper results for {store_id}"
        )

        products = []
        for raw_data in scraper_results:
            try:
                # Normalize the data
                product = self.normalizer.normalize_stealth_scraper_data(
                    raw_data, store_id
                )
                products.append(product)

                # Auto-save to vector database
                if self.auto_save:
                    self.vector_db.add_product(product)

                logger.debug(f"Processed stealth product: {product.name}")

            except Exception as e:
                logger.error(f"Failed to process stealth scraper result: {e}")
                continue

        # Track in session
        if session_id:
            self.session_products.setdefault(session_id, []).extend(products)

        logger.info(
            f"Successfully processed {len(products)} products from stealth scraper"
        )
        return products

    async def process_advanced_scraper_results(
        self,
        scraper_results: List[Dict[str, Any]],
        store_id: str,
        session_id: Optional[str] = None,
    ) -> List[Product]:
        """Process results from advanced scraper (API + stealth fallback).

        Args:
            scraper_results: Raw data from advanced scraper
            store_id: Store identifier
            session_id: Optional session identifier

        Returns:
            List of normalized Product objects
        """
        logger.info(
            f"Processing {len(scraper_results)} advanced scraper results for {store_id}"
        )

        products = []
        for raw_data in scraper_results:
            try:
                # Determine collection method based on data source
                method = (
                    DataCollectionMethod.API_DIRECT
                    if raw_data.get("source") == "api"
                    else DataCollectionMethod.AUTOMATED_STEALTH
                )

                # Override the collection method in raw data
                raw_data["_collection_method"] = method

                product = self.normalizer.normalize_stealth_scraper_data(
                    raw_data, store_id
                )
                product.collection_method = method

                products.append(product)

                if self.auto_save:
                    self.vector_db.add_product(product)

                logger.debug(
                    f"Processed advanced product: {product.name} (method: {method})"
                )

            except Exception as e:
                logger.error(f"Failed to process advanced scraper result: {e}")
                continue

        if session_id:
            self.session_products.setdefault(session_id, []).extend(products)

        logger.info(
            f"Successfully processed {len(products)} products from advanced scraper"
        )
        return products

    # Layer 2: Human-Assisted Browser Integration

    async def process_human_browser_results(
        self,
        browser_results: List[Dict[str, Any]],
        store_id: str,
        session_id: Optional[str] = None,
    ) -> List[Product]:
        """Process results from human-assisted browser scraping.

        Args:
            browser_results: Raw data from human browser interaction
            store_id: Store identifier
            session_id: Optional session identifier

        Returns:
            List of normalized Product objects
        """
        logger.info(
            f"Processing {len(browser_results)} human browser results for {store_id}"
        )

        products = []
        for raw_data in browser_results:
            try:
                # Human browser data gets highest confidence
                product = self.normalizer.normalize_human_browser_data(
                    raw_data, store_id
                )
                products.append(product)

                if self.auto_save:
                    self.vector_db.add_product(product)

                logger.debug(f"Processed human browser product: {product.name}")

            except Exception as e:
                logger.error(f"Failed to process human browser result: {e}")
                continue

        if session_id:
            self.session_products.setdefault(session_id, []).extend(products)

        logger.info(
            f"Successfully processed {len(products)} products from human browser"
        )
        return products

    # Layer 3: Clipboard Manual Collection Integration

    async def process_clipboard_data(
        self, clipboard_texts: List[str], session_id: Optional[str] = None
    ) -> List[Product]:
        """Process clipboard data with intelligent parsing.

        Args:
            clipboard_texts: List of clipboard text entries
            session_id: Optional session identifier

        Returns:
            List of successfully parsed Product objects
        """
        logger.info(f"Processing {len(clipboard_texts)} clipboard entries")

        products = []
        for text in clipboard_texts:
            try:
                # Try to parse clipboard data
                product = self.normalizer.normalize_clipboard_data(text)

                if product:
                    products.append(product)

                    if self.auto_save:
                        self.vector_db.add_product(product)

                    logger.debug(f"Processed clipboard product: {product.name}")
                else:
                    logger.debug(f"Could not parse clipboard text: {text[:50]}...")

            except Exception as e:
                logger.error(f"Failed to process clipboard data: {e}")
                continue

        if session_id:
            self.session_products.setdefault(session_id, []).extend(products)

        logger.info(f"Successfully processed {len(products)} products from clipboard")
        return products

    def process_single_clipboard_text(self, text: str) -> Optional[Product]:
        """Process a single clipboard text entry immediately.

        Args:
            text: Clipboard text to parse

        Returns:
            Parsed Product object or None
        """
        try:
            product = self.normalizer.normalize_clipboard_data(text)

            if product and self.auto_save:
                self.vector_db.add_product(product)
                logger.info(f"Added clipboard product: {product.name}")

            return product

        except Exception as e:
            logger.error(f"Failed to process clipboard text: {e}")
            return None

    # Unified Search and Similarity Operations

    async def find_similar_products(
        self,
        query: str,
        limit: int = 10,
        prefer_method: Optional[DataCollectionMethod] = None,
        min_confidence: float = 0.5,
    ) -> List[Tuple[Product, float]]:
        """Find similar products across all collection methods.

        Args:
            query: Search query
            limit: Maximum results
            prefer_method: Prefer products from specific collection method
            min_confidence: Minimum confidence threshold

        Returns:
            List of (product, similarity_score) tuples
        """
        # Set up method filter if specified
        method_filter = [prefer_method] if prefer_method else None

        results = self.vector_db.search_similar_products(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            collection_method_filter=method_filter,
            use_confidence_weighting=True,
        )

        logger.info(f"Found {len(results)} similar products for '{query}'")
        return results

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about collected products.

        Returns:
            Statistics dictionary with method breakdown
        """
        stats = self.vector_db.get_collection_stats()

        # Add session statistics
        session_stats = {}
        for session_id, products in self.session_products.items():
            session_stats[session_id] = {
                "total_products": len(products),
                "methods": {
                    method.value: sum(
                        1 for p in products if p.collection_method == method
                    )
                    for method in DataCollectionMethod
                },
                "avg_confidence": (
                    sum(p.confidence_score for p in products) / len(products)
                    if products
                    else 0
                ),
            }

        stats["sessions"] = session_stats
        return stats

    async def batch_add_products(self, products: List[Product]) -> None:
        """Add multiple products to vector database in batch.

        Args:
            products: List of products to add
        """
        if products:
            self.vector_db.add_products(products)
            logger.info(f"Batch added {len(products)} products to vector database")

    def find_duplicate_products(
        self, similarity_threshold: float = 0.95, same_store_only: bool = True
    ) -> List[List[Product]]:
        """Find potential duplicate products across collection methods.

        Args:
            similarity_threshold: Similarity threshold for duplicates
            same_store_only: Only check within same store

        Returns:
            List of product groups that are likely duplicates
        """
        # This would require more complex vector similarity analysis
        # For now, return empty list - can be implemented as needed
        logger.info("Duplicate detection not yet implemented")
        return []

    def get_products_by_method(self, method: DataCollectionMethod) -> List[Product]:
        """Get all products collected by a specific method.

        Args:
            method: Collection method to filter by

        Returns:
            List of products from that method
        """
        results = self.vector_db.search_similar_products(
            query="*",  # Match all
            limit=10000,  # Large limit
            collection_method_filter=[method],
            min_confidence=0.0,
        )

        return [product for product, _ in results]

    def validate_product_quality(self, product: Product) -> Dict[str, Any]:
        """Validate product data quality and suggest improvements.

        Args:
            product: Product to validate

        Returns:
            Validation report
        """
        issues = []
        suggestions = []

        # Check required fields
        if not product.name or len(product.name) < 3:
            issues.append("Product name too short or missing")
            suggestions.append("Provide a more descriptive product name")

        if product.price <= 0:
            issues.append("Invalid price")
            suggestions.append("Ensure price is a positive value")

        # Check confidence score
        if product.confidence_score < 0.7:
            issues.append("Low confidence score")
            suggestions.append("Consider manual verification of product data")

        # Check collection method appropriateness
        if (
            product.collection_method == DataCollectionMethod.CLIPBOARD_MANUAL
            and product.confidence_score > 0.9
        ):
            suggestions.append(
                "High-quality clipboard data - consider upgrading to human browser method"
            )

        return {
            "overall_quality": "good" if not issues else "needs_improvement",
            "issues": issues,
            "suggestions": suggestions,
            "confidence_score": product.confidence_score,
            "weighted_confidence": product.get_collection_confidence_weight(),
        }
