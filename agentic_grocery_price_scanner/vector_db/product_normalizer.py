"""
Product normalization service for handling data from different collection methods.
"""

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, List
from uuid import uuid4

from ..data_models.product import Product
from ..data_models.base import DataCollectionMethod, Currency, UnitType

logger = logging.getLogger(__name__)


class ProductNormalizer:
    """Service for normalizing product data from different collection sources."""

    # Confidence scores based on data source characteristics
    METHOD_BASE_CONFIDENCE = {
        DataCollectionMethod.HUMAN_BROWSER: 0.95,  # Human verified, full page access
        DataCollectionMethod.CLIPBOARD_MANUAL: 0.90,  # Human input, manual verification
        DataCollectionMethod.API_DIRECT: 0.98,  # Official API, structured data
        DataCollectionMethod.AUTOMATED_STEALTH: 0.85,  # Automated but comprehensive
        DataCollectionMethod.MOCK_DATA: 0.10,  # Test data only
    }

    # Common size unit mappings
    UNIT_MAPPINGS = {
        "gram": UnitType.GRAMS,
        "grams": UnitType.GRAMS,
        "g": UnitType.GRAMS,
        "kilogram": UnitType.KILOGRAMS,
        "kilograms": UnitType.KILOGRAMS,
        "kg": UnitType.KILOGRAMS,
        "pound": UnitType.POUNDS,
        "pounds": UnitType.POUNDS,
        "lb": UnitType.POUNDS,
        "lbs": UnitType.POUNDS,
        "ounce": UnitType.OUNCES,
        "ounces": UnitType.OUNCES,
        "oz": UnitType.OUNCES,
        "milliliter": UnitType.MILLILITERS,
        "milliliters": UnitType.MILLILITERS,
        "ml": UnitType.MILLILITERS,
        "liter": UnitType.LITERS,
        "liters": UnitType.LITERS,
        "l": UnitType.LITERS,
        "cup": UnitType.CUPS,
        "cups": UnitType.CUPS,
        "piece": UnitType.PIECES,
        "pieces": UnitType.PIECES,
        "each": UnitType.PIECES,
        "package": UnitType.PACKAGES,
        "packages": UnitType.PACKAGES,
        "pkg": UnitType.PACKAGES,
    }

    def normalize_stealth_scraper_data(
        self, raw_data: Dict[str, Any], store_id: str
    ) -> Product:
        """Normalize data from automated stealth scraper.

        Args:
            raw_data: Raw scraped data
            store_id: Store identifier

        Returns:
            Normalized Product object
        """
        try:
            # Extract basic fields
            name = self._clean_text(raw_data.get("name", "Unknown Product"))
            brand = self._clean_text(raw_data.get("brand"))
            price = self._parse_price(raw_data.get("price", "0"))

            # Parse size information
            size, size_unit = self._parse_size(raw_data.get("size"))

            # Generate keywords from available data
            keywords = self._generate_keywords(name, brand, raw_data.get("category"))

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(
                raw_data, DataCollectionMethod.AUTOMATED_STEALTH
            )

            return Product(
                id=uuid4(),
                name=name,
                brand=brand,
                price=price,
                currency=Currency.CAD,
                size=size,
                size_unit=size_unit,
                store_id=store_id,
                sku=raw_data.get("sku"),
                category=self._clean_text(raw_data.get("category")),
                subcategory=self._clean_text(raw_data.get("subcategory")),
                description=self._clean_text(raw_data.get("description")),
                image_url=raw_data.get("image_url"),
                product_url=raw_data.get("product_url"),
                in_stock=raw_data.get("in_stock", True),
                on_sale=raw_data.get("on_sale", False),
                sale_price=self._parse_price(raw_data.get("sale_price")),
                keywords=keywords,
                collection_method=DataCollectionMethod.AUTOMATED_STEALTH,
                confidence_score=confidence,
                source_metadata={
                    "scraper_type": "stealth",
                    "original_data": raw_data,
                    "parsing_version": "1.0",
                },
            )

        except Exception as e:
            logger.error(f"Failed to normalize stealth scraper data: {e}")
            raise

    def normalize_human_browser_data(
        self, raw_data: Dict[str, Any], store_id: str
    ) -> Product:
        """Normalize data from human-assisted browser scraping.

        Args:
            raw_data: Raw scraped data with human interaction
            store_id: Store identifier

        Returns:
            Normalized Product object
        """
        try:
            # Human-browser data is typically more complete and accurate
            name = self._clean_text(raw_data.get("name", "Unknown Product"))
            brand = self._clean_text(raw_data.get("brand"))
            price = self._parse_price(raw_data.get("price", "0"))

            size, size_unit = self._parse_size(raw_data.get("size"))
            keywords = self._generate_keywords(name, brand, raw_data.get("category"))

            # Higher base confidence due to human verification
            confidence = self._calculate_confidence(
                raw_data, DataCollectionMethod.HUMAN_BROWSER
            )

            return Product(
                id=uuid4(),
                name=name,
                brand=brand,
                price=price,
                currency=Currency.CAD,
                size=size,
                size_unit=size_unit,
                store_id=store_id,
                sku=raw_data.get("sku"),
                category=self._clean_text(raw_data.get("category")),
                subcategory=self._clean_text(raw_data.get("subcategory")),
                description=self._clean_text(raw_data.get("description")),
                image_url=raw_data.get("image_url"),
                product_url=raw_data.get("product_url"),
                in_stock=raw_data.get("in_stock", True),
                on_sale=raw_data.get("on_sale", False),
                sale_price=self._parse_price(raw_data.get("sale_price")),
                keywords=keywords,
                collection_method=DataCollectionMethod.HUMAN_BROWSER,
                confidence_score=confidence,
                source_metadata={
                    "scraper_type": "human_browser",
                    "human_verified": True,
                    "original_data": raw_data,
                    "parsing_version": "1.0",
                },
            )

        except Exception as e:
            logger.error(f"Failed to normalize human browser data: {e}")
            raise

    def normalize_clipboard_data(
        self, clipboard_text: str, store_id: str = "unknown"
    ) -> Optional[Product]:
        """Normalize data from clipboard parsing.

        Args:
            clipboard_text: Raw clipboard text
            store_id: Store identifier (may be inferred from text)

        Returns:
            Normalized Product object or None if parsing fails
        """
        try:
            # Parse clipboard text using various patterns
            parsed_data = self._parse_clipboard_text(clipboard_text)

            if not parsed_data or not parsed_data.get("name"):
                logger.warning("Could not parse meaningful product data from clipboard")
                return None

            # Extract store from URL if available
            if parsed_data.get("url") and store_id == "unknown":
                store_id = self._extract_store_from_url(parsed_data["url"])

            name = self._clean_text(parsed_data["name"])
            brand = self._clean_text(parsed_data.get("brand"))
            price = self._parse_price(parsed_data.get("price", "0"))

            size, size_unit = self._parse_size(parsed_data.get("size"))
            keywords = self._generate_keywords(name, brand, parsed_data.get("category"))

            # Confidence depends on how much data was successfully parsed
            confidence = self._calculate_confidence(
                parsed_data, DataCollectionMethod.CLIPBOARD_MANUAL
            )

            return Product(
                id=uuid4(),
                name=name,
                brand=brand,
                price=price,
                currency=Currency.CAD,
                size=size,
                size_unit=size_unit,
                store_id=store_id,
                category=self._clean_text(parsed_data.get("category")),
                description=self._clean_text(parsed_data.get("description")),
                product_url=parsed_data.get("url"),
                in_stock=parsed_data.get("in_stock", True),
                keywords=keywords,
                collection_method=DataCollectionMethod.CLIPBOARD_MANUAL,
                confidence_score=confidence,
                source_metadata={
                    "scraper_type": "clipboard",
                    "raw_clipboard": clipboard_text,
                    "parsed_fields": list(parsed_data.keys()),
                    "parsing_version": "1.0",
                },
            )

        except Exception as e:
            logger.error(f"Failed to normalize clipboard data: {e}")
            return None

    def _parse_clipboard_text(self, text: str) -> Dict[str, Any]:
        """Parse structured data from clipboard text."""
        data = {}
        lines = text.strip().split("\n")

        # Common patterns for grocery product information
        price_pattern = r"\$?(\d+(?:\.\d{2})?)"
        size_pattern = r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract price
            if "$" in line and not data.get("price"):
                price_match = re.search(price_pattern, line)
                if price_match:
                    data["price"] = price_match.group(1)

            # Try to extract size
            size_match = re.search(size_pattern, line)
            if size_match and not data.get("size"):
                data["size"] = f"{size_match.group(1)} {size_match.group(2)}"

            # Check for URLs
            if "http" in line and not data.get("url"):
                data["url"] = line.strip()

            # Assume first substantial line is product name
            if not data.get("name") and len(line) > 5 and "$" not in line:
                data["name"] = line

        return data

    def _extract_store_from_url(self, url: str) -> str:
        """Extract store identifier from product URL."""
        url_lower = url.lower()

        if "metro.ca" in url_lower:
            return "metro_ca"
        elif "walmart.ca" in url_lower:
            return "walmart_ca"
        elif "freshco.com" in url_lower:
            return "freshco_com"
        else:
            return "unknown"

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text fields."""
        if not text:
            return None

        # Remove extra whitespace and normalize
        cleaned = re.sub(r"\s+", " ", text.strip())
        return cleaned if cleaned else None

    def _parse_price(self, price_str: Optional[str]) -> Decimal:
        """Parse price string to Decimal."""
        if not price_str:
            return Decimal("0.00")

        try:
            # Remove currency symbols and whitespace
            price_clean = re.sub(r"[^\d.]", "", str(price_str))
            if price_clean:
                return Decimal(price_clean)
        except (InvalidOperation, ValueError):
            logger.warning(f"Could not parse price: {price_str}")

        return Decimal("0.00")

    def _parse_size(
        self, size_str: Optional[str]
    ) -> tuple[Optional[float], Optional[UnitType]]:
        """Parse size string to size and unit."""
        if not size_str:
            return None, None

        try:
            # Extract number and unit
            match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", str(size_str))
            if match:
                size_value = float(match.group(1))
                unit_str = match.group(2).lower()

                # Map to standard unit
                unit = self.UNIT_MAPPINGS.get(unit_str)
                if unit:
                    return size_value, unit

        except (ValueError, AttributeError):
            logger.warning(f"Could not parse size: {size_str}")

        return None, None

    def _generate_keywords(
        self, name: Optional[str], brand: Optional[str], category: Optional[str]
    ) -> List[str]:
        """Generate search keywords from product information."""
        keywords = []

        if name:
            # Split name into words and add meaningful ones
            words = re.findall(r"\b\w+\b", name.lower())
            keywords.extend([w for w in words if len(w) > 2])

        if brand:
            keywords.append(brand.lower())

        if category:
            keywords.append(category.lower())

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def _calculate_confidence(
        self, data: Dict[str, Any], method: DataCollectionMethod
    ) -> float:
        """Calculate confidence score based on data completeness and method."""
        base_confidence = self.METHOD_BASE_CONFIDENCE[method]

        # Count available fields
        required_fields = ["name", "price"]
        optional_fields = ["brand", "size", "category", "description", "image_url"]

        required_score = sum(1 for field in required_fields if data.get(field))
        optional_score = sum(1 for field in optional_fields if data.get(field))

        # Calculate completeness ratio
        completeness = (
            required_score / len(required_fields) * 0.7
            + optional_score / len(optional_fields) * 0.3
        )

        # Final confidence is base confidence adjusted by completeness
        return min(1.0, base_confidence * (0.7 + 0.3 * completeness))
