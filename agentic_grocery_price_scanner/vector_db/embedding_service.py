"""
Embedding service for converting product data to vectors.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from ..data_models.product import Product

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from product data."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_product_text(self, product: Product) -> str:
        """Create a text representation of a product for embedding.

        Args:
            product: The product to create text for.

        Returns:
            A text string representing the product.
        """
        parts = []

        # Add product name (most important)
        if product.name:
            parts.append(product.name)

        # Add brand
        if product.brand:
            parts.append(f"brand: {product.brand}")

        # Add category information
        if product.category:
            parts.append(f"category: {product.category}")
        if product.subcategory:
            parts.append(f"subcategory: {product.subcategory}")

        # Add size information
        if product.size and product.size_unit:
            parts.append(f"size: {product.size} {product.size_unit}")

        # Add description
        if product.description:
            parts.append(product.description)

        # Add keywords
        if product.keywords:
            parts.append(f"keywords: {' '.join(product.keywords)}")

        return " ".join(parts)

    def encode_product(self, product: Product) -> np.ndarray:
        """Generate an embedding for a single product.

        Args:
            product: The product to encode.

        Returns:
            The embedding vector as a numpy array.
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        text = self.create_product_text(product)
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding

    def encode_products(self, products: List[Product]) -> List[np.ndarray]:
        """Generate embeddings for multiple products.

        Args:
            products: List of products to encode.

        Returns:
            List of embedding vectors.
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        texts = [self.create_product_text(product) for product in products]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        return [embedding for embedding in embeddings]

    def encode_text(self, text: str) -> np.ndarray:
        """Generate an embedding for arbitrary text (e.g., search queries).

        Args:
            text: The text to encode.

        Returns:
            The embedding vector as a numpy array.
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        return self.model.encode(text, convert_to_numpy=True)

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()
