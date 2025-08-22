"""
Qdrant vector database client for product similarity search.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class QdrantVectorDB:
    """Vector database client for product similarity search."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "grocery_products",
        embedding_service: Optional[EmbeddingService] = None,
        in_memory: bool = True,
    ):
        """Initialize the Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the vector collection
            embedding_service: Service for generating embeddings
            in_memory: Use in-memory storage (for development/testing)
        """
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()

        try:
            if in_memory:
                # Use in-memory storage for development
                self.client = QdrantClient(":memory:")
                logger.info("Initialized Qdrant with in-memory storage")
            else:
                # Use persistent storage
                self.client = QdrantClient(host=host, port=port)
                logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the product collection exists with proper schema."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")

                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

                # Create payload indexes for efficient filtering
                self._create_payload_indexes()
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create indexes on frequently queried payload fields."""
        indexes = [
            ("store_id", models.PayloadSchemaType.KEYWORD),
            ("collection_method", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("brand", models.PayloadSchemaType.KEYWORD),
            ("in_stock", models.PayloadSchemaType.BOOL),
            ("confidence_score", models.PayloadSchemaType.FLOAT),
        ]

        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index for field: {field_name}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Could not create index for {field_name}: {e}")

    def add_product(self, product: Product) -> None:
        """Add a single product to the vector database.

        Args:
            product: The product to add
        """
        self.add_products([product])

    def add_products(self, products: List[Product]) -> None:
        """Add multiple products to the vector database.

        Args:
            products: List of products to add
        """
        if not products:
            return

        logger.info(f"Adding {len(products)} products to vector database")

        try:
            # Generate embeddings
            embeddings = self.embedding_service.encode_products(products)

            # Prepare points for Qdrant
            points = []
            for product, embedding in zip(products, embeddings):
                payload = self._product_to_payload(product)

                point = models.PointStruct(
                    id=str(product.id), vector=embedding.tolist(), payload=payload
                )
                points.append(point)

            # Upload to Qdrant
            self.client.upsert(collection_name=self.collection_name, points=points)

            logger.info(f"Successfully added {len(products)} products")

        except Exception as e:
            logger.error(f"Failed to add products: {e}")
            raise

    def _product_to_payload(self, product: Product) -> Dict[str, Any]:
        """Convert product to Qdrant payload."""
        payload = {
            "id": str(product.id),
            "name": product.name,
            "brand": product.brand,
            "price": float(product.price),
            "currency": product.currency,
            "store_id": product.store_id,
            "category": product.category,
            "subcategory": product.subcategory,
            "in_stock": product.in_stock,
            "on_sale": product.on_sale,
            "keywords": product.keywords,
            "collection_method": product.collection_method,
            "confidence_score": product.confidence_score,
            "weighted_confidence": product.get_collection_confidence_weight(),
            "created_at": product.created_at.isoformat(),
        }

        # Add optional fields if they exist
        if product.size:
            payload["size"] = product.size
            payload["size_unit"] = product.size_unit

        if product.price_per_unit:
            payload["price_per_unit"] = float(product.price_per_unit)

        if product.sale_price:
            payload["sale_price"] = float(product.sale_price)

        if product.description:
            payload["description"] = product.description

        if product.source_metadata:
            payload["source_metadata"] = product.source_metadata

        return payload

    def search_similar_products(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        store_filter: Optional[List[str]] = None,
        collection_method_filter: Optional[List[DataCollectionMethod]] = None,
        in_stock_only: bool = True,
        use_confidence_weighting: bool = True,
    ) -> List[Tuple[Product, float]]:
        """Search for similar products using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            min_confidence: Minimum confidence score filter
            store_filter: Filter by specific stores
            collection_method_filter: Filter by collection methods
            in_stock_only: Only return in-stock products
            use_confidence_weighting: Apply confidence-based scoring

        Returns:
            List of (product, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_text(query)

            # Build filter conditions
            filter_conditions = []

            if min_confidence > 0:
                filter_conditions.append(
                    models.FieldCondition(
                        key="confidence_score", range=models.Range(gte=min_confidence)
                    )
                )

            if store_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="store_id", match=models.MatchAny(any=store_filter)
                    )
                )

            if collection_method_filter:
                method_values = [method.value for method in collection_method_filter]
                filter_conditions.append(
                    models.FieldCondition(
                        key="collection_method",
                        match=models.MatchAny(any=method_values),
                    )
                )

            if in_stock_only:
                filter_conditions.append(
                    models.FieldCondition(
                        key="in_stock", match=models.MatchValue(value=True)
                    )
                )

            # Combine filters
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(must=filter_conditions)

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )

            # Convert results to products
            results = []
            for result in search_results:
                product = self._payload_to_product(result.payload)

                # Apply confidence weighting to similarity score
                similarity_score = result.score
                if use_confidence_weighting:
                    weighted_confidence = result.payload.get("weighted_confidence", 1.0)
                    similarity_score *= weighted_confidence

                results.append((product, similarity_score))

            logger.info(f"Found {len(results)} similar products for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar products: {e}")
            raise

    def _payload_to_product(self, payload: Dict[str, Any]) -> Product:
        """Convert Qdrant payload back to Product object."""
        # Handle optional fields
        size = payload.get("size")
        size_unit = payload.get("size_unit")
        price_per_unit = payload.get("price_per_unit")
        sale_price = payload.get("sale_price")
        description = payload.get("description")
        source_metadata = payload.get("source_metadata")

        product_data = {
            "id": UUID(payload["id"]),
            "name": payload["name"],
            "brand": payload.get("brand"),
            "price": payload["price"],
            "currency": payload["currency"],
            "store_id": payload["store_id"],
            "category": payload.get("category"),
            "subcategory": payload.get("subcategory"),
            "in_stock": payload["in_stock"],
            "on_sale": payload.get("on_sale", False),
            "keywords": payload.get("keywords", []),
            "collection_method": payload["collection_method"],
            "confidence_score": payload["confidence_score"],
        }

        # Add optional fields if they exist
        if size is not None:
            product_data["size"] = size
        if size_unit is not None:
            product_data["size_unit"] = size_unit
        if price_per_unit is not None:
            product_data["price_per_unit"] = price_per_unit
        if sale_price is not None:
            product_data["sale_price"] = sale_price
        if description is not None:
            product_data["description"] = description
        if source_metadata is not None:
            product_data["source_metadata"] = source_metadata

        return Product(**product_data)

    def get_product_by_id(self, product_id: UUID) -> Optional[Product]:
        """Retrieve a product by its ID.

        Args:
            product_id: The product ID to search for

        Returns:
            The product if found, None otherwise
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(product_id)],
                with_payload=True,
            )

            if result:
                return self._payload_to_product(result[0].payload)
            return None

        except Exception as e:
            logger.error(f"Failed to get product by ID {product_id}: {e}")
            return None

    def delete_product(self, product_id: UUID) -> bool:
        """Delete a product from the vector database.

        Args:
            product_id: The product ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[str(product_id)]),
            )
            logger.info(f"Deleted product {product_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete product {product_id}: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # Get method distribution
            method_counts = {}
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True,
            )

            for point in scroll_result[0]:
                method = point.payload.get("collection_method", "unknown")
                method_counts[method] = method_counts.get(method, 0) + 1

            return {
                "total_vectors": info.vectors_count,
                "collection_status": info.status,
                "optimizer_status": info.optimizer_status,
                "method_distribution": method_counts,
                "indexed_fields": [
                    "store_id",
                    "collection_method",
                    "category",
                    "brand",
                    "in_stock",
                    "confidence_score",
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all vectors from the collection.

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=models.Filter()),
            )
            logger.info(f"Cleared collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
