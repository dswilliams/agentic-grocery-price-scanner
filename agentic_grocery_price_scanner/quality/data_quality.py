"""
Data quality framework with automated anomaly detection, validation,
and consistency checking for production-level reliability.
"""

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from decimal import Decimal, InvalidOperation
import re
import json

from ..data_models import Product, Ingredient
from ..data_models.base import UnitType

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """Types of data quality issues."""
    PRICE_ANOMALY = "price_anomaly"
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE_PRODUCT = "duplicate_product"
    INCONSISTENT_UNITS = "inconsistent_units"
    SUSPICIOUS_PRICE = "suspicious_price"
    INCOMPLETE_PRODUCT = "incomplete_product"
    UNREALISTIC_VALUE = "unrealistic_value"
    PARSING_ERROR = "parsing_error"
    BRAND_MISMATCH = "brand_mismatch"


class Severity(Enum):
    """Severity levels for quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Alert for a data quality issue."""
    
    issue_type: QualityIssue
    severity: Severity
    message: str
    product_id: Optional[str] = None
    store_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "product_id": self.product_id,
            "store_id": self.store_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "suggested_action": self.suggested_action
        }


@dataclass
class QualityMetrics:
    """Quality metrics for a data collection."""
    
    total_products: int = 0
    valid_products: int = 0
    products_with_issues: int = 0
    
    # Completeness metrics
    complete_products: int = 0
    missing_prices: int = 0
    missing_names: int = 0
    missing_brands: int = 0
    missing_images: int = 0
    
    # Consistency metrics
    price_anomalies: int = 0
    duplicate_products: int = 0
    unit_inconsistencies: int = 0
    
    # Quality scores (0-100)
    overall_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    
    @property
    def validity_rate(self) -> float:
        """Calculate validity rate as percentage."""
        return (self.valid_products / max(self.total_products, 1)) * 100
    
    @property
    def issue_rate(self) -> float:
        """Calculate issue rate as percentage."""
        return (self.products_with_issues / max(self.total_products, 1)) * 100


class PriceAnomalyDetector:
    """Detects price anomalies using statistical methods."""
    
    def __init__(self, z_score_threshold: float = 3.0):
        self.z_score_threshold = z_score_threshold
        self.price_history: Dict[str, List[float]] = {}  # product_category -> prices
        self.store_price_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}  # store -> category -> (min, max)
    
    def add_price_sample(self, store_id: str, category: str, price: float):
        """Add a price sample to historical data."""
        category_key = f"{store_id}:{category}"
        if category_key not in self.price_history:
            self.price_history[category_key] = []
        
        self.price_history[category_key].append(price)
        
        # Limit history size
        if len(self.price_history[category_key]) > 1000:
            self.price_history[category_key] = self.price_history[category_key][-500:]
        
        # Update price ranges
        if store_id not in self.store_price_ranges:
            self.store_price_ranges[store_id] = {}
        
        if category not in self.store_price_ranges[store_id]:
            self.store_price_ranges[store_id][category] = (price, price)
        else:
            current_min, current_max = self.store_price_ranges[store_id][category]
            self.store_price_ranges[store_id][category] = (
                min(current_min, price),
                max(current_max, price)
            )
    
    def detect_anomaly(self, store_id: str, category: str, price: float) -> Optional[QualityAlert]:
        """Detect if a price is anomalous."""
        category_key = f"{store_id}:{category}"
        
        if category_key not in self.price_history or len(self.price_history[category_key]) < 10:
            # Not enough historical data
            return None
        
        prices = self.price_history[category_key]
        
        try:
            mean_price = statistics.mean(prices)
            std_price = statistics.stdev(prices)
            
            if std_price == 0:
                return None
            
            z_score = abs(price - mean_price) / std_price
            
            if z_score > self.z_score_threshold:
                severity = Severity.HIGH if z_score > 5.0 else Severity.MEDIUM
                
                return QualityAlert(
                    issue_type=QualityIssue.PRICE_ANOMALY,
                    severity=severity,
                    message=f"Price ${price:.2f} is {z_score:.1f} standard deviations from mean ${mean_price:.2f}",
                    store_id=store_id,
                    metadata={
                        "price": price,
                        "mean_price": mean_price,
                        "std_price": std_price,
                        "z_score": z_score,
                        "category": category,
                        "sample_size": len(prices)
                    },
                    suggested_action="Verify price accuracy and check for data entry errors"
                )
        
        except statistics.StatisticsError:
            return None
        
        return None


class ProductValidator:
    """Validates individual product data for quality issues."""
    
    def __init__(self):
        self.required_fields = ["name", "price", "store_id"]
        self.optional_fields = ["brand", "image_url", "product_url", "description"]
        
        # Common price validation patterns
        self.unrealistic_prices = {
            "too_low": 0.01,  # Anything below 1 cent
            "too_high": 1000.0  # Anything above $1000 for groceries
        }
        
        # Brand normalization patterns
        self.brand_variations = {
            "coca cola": ["coca-cola", "coke", "coca cola", "cocacola"],
            "pepsi": ["pepsi", "pepsi-cola", "pepsi cola"],
            "nestle": ["nestle", "nestlé", "nestle®"],
            "kelloggs": ["kellogg's", "kelloggs", "kellogs", "kellogg"]
        }
    
    def validate_product(self, product: Product) -> List[QualityAlert]:
        """Validate a single product and return quality alerts."""
        alerts = []
        
        # Check required fields
        alerts.extend(self._check_required_fields(product))
        
        # Validate price
        alerts.extend(self._validate_price(product))
        
        # Validate name
        alerts.extend(self._validate_name(product))
        
        # Validate brand
        alerts.extend(self._validate_brand(product))
        
        # Check completeness
        alerts.extend(self._check_completeness(product))
        
        # Validate URLs
        alerts.extend(self._validate_urls(product))
        
        return alerts
    
    def _check_required_fields(self, product: Product) -> List[QualityAlert]:
        """Check for missing required fields."""
        alerts = []
        
        if not product.name or not product.name.strip():
            alerts.append(QualityAlert(
                issue_type=QualityIssue.MISSING_DATA,
                severity=Severity.CRITICAL,
                message="Product name is missing",
                product_id=str(product.id) if product.id else None,
                store_id=product.store_id,
                suggested_action="Reject product or flag for manual review"
            ))
        
        if product.price is None or product.price <= 0:
            alerts.append(QualityAlert(
                issue_type=QualityIssue.MISSING_DATA,
                severity=Severity.CRITICAL,
                message="Product price is missing or invalid",
                product_id=str(product.id) if product.id else None,
                store_id=product.store_id,
                suggested_action="Reject product or flag for manual review"
            ))
        
        if not product.store_id:
            alerts.append(QualityAlert(
                issue_type=QualityIssue.MISSING_DATA,
                severity=Severity.HIGH,
                message="Store ID is missing",
                product_id=str(product.id) if product.id else None,
                suggested_action="Assign to default store or reject"
            ))
        
        return alerts
    
    def _validate_price(self, product: Product) -> List[QualityAlert]:
        """Validate product price for realistic values."""
        alerts = []
        
        if product.price is not None:
            price_float = float(product.price)
            
            if price_float < self.unrealistic_prices["too_low"]:
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.UNREALISTIC_VALUE,
                    severity=Severity.HIGH,
                    message=f"Price ${price_float:.2f} is unrealistically low",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"price": price_float},
                    suggested_action="Verify price accuracy"
                ))
            
            elif price_float > self.unrealistic_prices["too_high"]:
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.UNREALISTIC_VALUE,
                    severity=Severity.MEDIUM,
                    message=f"Price ${price_float:.2f} is unusually high for groceries",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"price": price_float},
                    suggested_action="Verify this is not a bulk/specialty item"
                ))
        
        return alerts
    
    def _validate_name(self, product: Product) -> List[QualityAlert]:
        """Validate product name for quality issues."""
        alerts = []
        
        if product.name:
            name = product.name.strip()
            
            # Check for suspiciously short names
            if len(name) < 3:
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.INVALID_FORMAT,
                    severity=Severity.MEDIUM,
                    message=f"Product name '{name}' is suspiciously short",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    suggested_action="Verify name completeness"
                ))
            
            # Check for HTML tags or special characters
            if re.search(r'<[^>]+>', name):
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.PARSING_ERROR,
                    severity=Severity.MEDIUM,
                    message="Product name contains HTML tags",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"raw_name": name},
                    suggested_action="Clean HTML from name"
                ))
            
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\-\'.,&%]', name)) / len(name)
            if special_char_ratio > 0.3:
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.INVALID_FORMAT,
                    severity=Severity.LOW,
                    message="Product name has many special characters",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"name": name, "special_char_ratio": special_char_ratio},
                    suggested_action="Verify name parsing accuracy"
                ))
        
        return alerts
    
    def _validate_brand(self, product: Product) -> List[QualityAlert]:
        """Validate brand information for consistency."""
        alerts = []
        
        if product.brand:
            brand = product.brand.strip().lower()
            
            # Check for brand variations that should be normalized
            normalized_brand = None
            for standard_brand, variations in self.brand_variations.items():
                if brand in variations:
                    normalized_brand = standard_brand
                    break
            
            if normalized_brand and normalized_brand != brand:
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.BRAND_MISMATCH,
                    severity=Severity.LOW,
                    message=f"Brand '{product.brand}' could be normalized to '{normalized_brand}'",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"original_brand": product.brand, "suggested_brand": normalized_brand},
                    suggested_action="Consider normalizing brand name"
                ))
        
        return alerts
    
    def _check_completeness(self, product: Product) -> List[QualityAlert]:
        """Check completeness of optional but valuable fields."""
        alerts = []
        missing_fields = []
        
        if not product.brand:
            missing_fields.append("brand")
        
        if not product.image_url:
            missing_fields.append("image_url")
        
        if not product.description:
            missing_fields.append("description")
        
        if missing_fields:
            alerts.append(QualityAlert(
                issue_type=QualityIssue.INCOMPLETE_PRODUCT,
                severity=Severity.LOW,
                message=f"Product missing optional fields: {', '.join(missing_fields)}",
                product_id=str(product.id) if product.id else None,
                store_id=product.store_id,
                metadata={"missing_fields": missing_fields},
                suggested_action="Consider enhancing data collection for these fields"
            ))
        
        return alerts
    
    def _validate_urls(self, product: Product) -> List[QualityAlert]:
        """Validate URL fields for proper format."""
        alerts = []
        
        url_fields = [
            ("image_url", product.image_url),
            ("product_url", product.product_url)
        ]
        
        for field_name, url in url_fields:
            if url and not self._is_valid_url(url):
                alerts.append(QualityAlert(
                    issue_type=QualityIssue.INVALID_FORMAT,
                    severity=Severity.LOW,
                    message=f"Invalid {field_name}: {url}",
                    product_id=str(product.id) if product.id else None,
                    store_id=product.store_id,
                    metadata={"field": field_name, "url": url},
                    suggested_action="Verify URL format and accessibility"
                ))
        
        return alerts
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None


class DuplicateDetector:
    """Detects duplicate products across different sources."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        
    def find_duplicates(self, products: List[Product]) -> List[QualityAlert]:
        """Find potential duplicate products in a list."""
        alerts = []
        seen_products = {}  # normalized_key -> Product
        
        for product in products:
            normalized_key = self._normalize_product_key(product)
            
            if normalized_key in seen_products:
                existing_product = seen_products[normalized_key]
                
                # Check if this is a true duplicate or variation
                if self._are_likely_duplicates(product, existing_product):
                    alerts.append(QualityAlert(
                        issue_type=QualityIssue.DUPLICATE_PRODUCT,
                        severity=Severity.MEDIUM,
                        message=f"Potential duplicate: '{product.name}' similar to existing product",
                        product_id=str(product.id) if product.id else None,
                        store_id=product.store_id,
                        metadata={
                            "duplicate_candidate": {
                                "name": existing_product.name,
                                "price": float(existing_product.price) if existing_product.price else None,
                                "store": existing_product.store_id
                            }
                        },
                        suggested_action="Review for actual duplication or merge if appropriate"
                    ))
            else:
                seen_products[normalized_key] = product
        
        return alerts
    
    def _normalize_product_key(self, product: Product) -> str:
        """Create normalized key for duplicate detection."""
        name = product.name.lower().strip() if product.name else ""
        brand = product.brand.lower().strip() if product.brand else ""
        store = product.store_id.lower() if product.store_id else ""
        
        # Remove common words and normalize
        name = re.sub(r'\b(organic|natural|fresh|premium|select|choice)\b', '', name)
        name = re.sub(r'[^a-z0-9]', '', name)
        brand = re.sub(r'[^a-z0-9]', '', brand)
        
        return f"{store}:{brand}:{name}"
    
    def _are_likely_duplicates(self, product1: Product, product2: Product) -> bool:
        """Determine if two products are likely duplicates."""
        # Same store is required
        if product1.store_id != product2.store_id:
            return False
        
        # Price similarity (within 10%)
        if product1.price and product2.price:
            price1, price2 = float(product1.price), float(product2.price)
            price_diff = abs(price1 - price2) / max(price1, price2)
            if price_diff > 0.1:  # More than 10% difference
                return False
        
        # Name similarity
        name1 = product1.name.lower() if product1.name else ""
        name2 = product2.name.lower() if product2.name else ""
        
        # Simple similarity check
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= self.similarity_threshold


class DataQualityManager:
    """Main manager for data quality assessment and improvement."""
    
    def __init__(self):
        self.validator = ProductValidator()
        self.anomaly_detector = PriceAnomalyDetector()
        self.duplicate_detector = DuplicateDetector()
        
        self.quality_alerts: List[QualityAlert] = []
        self.quality_history: List[QualityMetrics] = []
        
        logger.info("Initialized DataQualityManager")
    
    async def assess_product_quality(
        self,
        products: List[Product],
        store_id: Optional[str] = None
    ) -> Tuple[QualityMetrics, List[QualityAlert]]:
        """Comprehensive quality assessment of product data."""
        
        logger.info(f"Starting quality assessment for {len(products)} products")
        
        alerts = []
        metrics = QualityMetrics(total_products=len(products))
        
        # Validate individual products
        valid_products = 0
        for product in products:
            product_alerts = self.validator.validate_product(product)
            alerts.extend(product_alerts)
            
            if not any(alert.severity == Severity.CRITICAL for alert in product_alerts):
                valid_products += 1
                metrics.valid_products += 1
                
                # Update price history for anomaly detection
                if product.price and store_id:
                    category = self._categorize_product(product)
                    self.anomaly_detector.add_price_sample(store_id, category, float(product.price))
            
            # Count issues by type
            if product_alerts:
                metrics.products_with_issues += 1
            
            # Count missing fields
            if not product.name:
                metrics.missing_names += 1
            if not product.price:
                metrics.missing_prices += 1
            if not product.brand:
                metrics.missing_brands += 1
            if not product.image_url:
                metrics.missing_images += 1
            
            # Check completeness
            required_fields = [product.name, product.price, product.store_id]
            if all(field for field in required_fields):
                metrics.complete_products += 1
        
        # Detect duplicates
        duplicate_alerts = self.duplicate_detector.find_duplicates(products)
        alerts.extend(duplicate_alerts)
        metrics.duplicate_products = len(duplicate_alerts)
        
        # Detect price anomalies
        if store_id:
            for product in products:
                if product.price:
                    category = self._categorize_product(product)
                    anomaly_alert = self.anomaly_detector.detect_anomaly(
                        store_id, category, float(product.price)
                    )
                    if anomaly_alert:
                        anomaly_alert.product_id = str(product.id) if product.id else None
                        alerts.append(anomaly_alert)
                        metrics.price_anomalies += 1
        
        # Calculate quality scores
        self._calculate_quality_scores(metrics)
        
        # Store alerts and metrics
        self.quality_alerts.extend(alerts)
        self.quality_history.append(metrics)
        
        # Limit history size
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-50:]
        
        logger.info(f"Quality assessment completed: {metrics.overall_quality_score:.1f}% overall quality")
        
        return metrics, alerts
    
    def _categorize_product(self, product: Product) -> str:
        """Categorize product for price anomaly detection."""
        name_lower = product.name.lower() if product.name else ""
        
        # Simple category detection based on keywords
        categories = {
            "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
            "meat": ["chicken", "beef", "pork", "fish", "salmon", "turkey"],
            "produce": ["apple", "banana", "orange", "lettuce", "tomato", "onion"],
            "bread": ["bread", "bun", "roll", "bagel"],
            "cereal": ["cereal", "oats", "granola"],
            "beverages": ["juice", "soda", "water", "coffee", "tea"],
            "snacks": ["chips", "crackers", "cookies", "nuts"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _calculate_quality_scores(self, metrics: QualityMetrics):
        """Calculate quality scores based on metrics."""
        total = max(metrics.total_products, 1)
        
        # Completeness score
        completeness_factors = [
            (total - metrics.missing_names) / total,
            (total - metrics.missing_prices) / total,
            (total - metrics.missing_brands) / total * 0.5,  # Brand less critical
            (total - metrics.missing_images) / total * 0.3   # Images least critical
        ]
        metrics.completeness_score = sum(completeness_factors) / len(completeness_factors) * 100
        
        # Consistency score
        consistency_penalties = [
            metrics.price_anomalies / total * 30,  # Price anomalies are serious
            metrics.duplicate_products / total * 20,
            metrics.unit_inconsistencies / total * 10
        ]
        consistency_penalty = min(sum(consistency_penalties), 100)
        metrics.consistency_score = max(0, 100 - consistency_penalty)
        
        # Accuracy score (based on validation errors)
        accuracy_penalty = (metrics.products_with_issues / total) * 50
        metrics.accuracy_score = max(0, 100 - accuracy_penalty)
        
        # Overall quality score (weighted average)
        weights = [0.4, 0.3, 0.3]  # completeness, consistency, accuracy
        scores = [metrics.completeness_score, metrics.consistency_score, metrics.accuracy_score]
        metrics.overall_quality_score = sum(w * s for w, s in zip(weights, scores))
    
    def get_quality_report(self, include_alerts: bool = True) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.quality_history:
            return {"message": "No quality data available"}
        
        latest_metrics = self.quality_history[-1]
        
        # Aggregate recent alerts by type and severity
        recent_alerts = self.quality_alerts[-100:]  # Last 100 alerts
        alert_summary = {}
        severity_summary = {}
        
        for alert in recent_alerts:
            issue_type = alert.issue_type.value
            severity = alert.severity.value
            
            alert_summary[issue_type] = alert_summary.get(issue_type, 0) + 1
            severity_summary[severity] = severity_summary.get(severity, 0) + 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "latest_metrics": {
                "total_products": latest_metrics.total_products,
                "overall_quality_score": latest_metrics.overall_quality_score,
                "completeness_score": latest_metrics.completeness_score,
                "consistency_score": latest_metrics.consistency_score,
                "accuracy_score": latest_metrics.accuracy_score,
                "validity_rate": latest_metrics.validity_rate,
                "issue_rate": latest_metrics.issue_rate
            },
            "issue_breakdown": {
                "missing_data": {
                    "missing_names": latest_metrics.missing_names,
                    "missing_prices": latest_metrics.missing_prices,
                    "missing_brands": latest_metrics.missing_brands,
                    "missing_images": latest_metrics.missing_images
                },
                "data_issues": {
                    "price_anomalies": latest_metrics.price_anomalies,
                    "duplicate_products": latest_metrics.duplicate_products,
                    "unit_inconsistencies": latest_metrics.unit_inconsistencies
                }
            },
            "alert_summary": {
                "by_type": alert_summary,
                "by_severity": severity_summary,
                "total_recent_alerts": len(recent_alerts)
            },
            "quality_trend": self._calculate_quality_trend()
        }
        
        if include_alerts:
            report["recent_alerts"] = [alert.to_dict() for alert in recent_alerts[-20:]]  # Last 20 alerts
        
        return report
    
    def _calculate_quality_trend(self) -> Dict[str, Any]:
        """Calculate quality trend over recent assessments."""
        if len(self.quality_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_scores = [m.overall_quality_score for m in self.quality_history[-10:]]
        
        if len(recent_scores) < 3:
            return {"trend": "insufficient_data"}
        
        # Simple linear trend
        x = list(range(len(recent_scores)))
        y = recent_scores
        
        # Calculate slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        trend_direction = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
        
        return {
            "trend": trend_direction,
            "slope": slope,
            "recent_average": statistics.mean(recent_scores),
            "best_recent_score": max(recent_scores),
            "worst_recent_score": min(recent_scores)
        }
    
    async def remediate_quality_issues(
        self,
        products: List[Product],
        auto_fix: bool = True
    ) -> List[Product]:
        """Attempt to automatically remediate quality issues."""
        
        logger.info(f"Starting quality remediation for {len(products)} products")
        
        remediated_products = []
        
        for product in products:
            remediated_product = product
            
            if auto_fix:
                # Clean product name
                if product.name:
                    cleaned_name = self._clean_product_name(product.name)
                    if cleaned_name != product.name:
                        remediated_product.name = cleaned_name
                
                # Normalize brand
                if product.brand:
                    normalized_brand = self._normalize_brand(product.brand)
                    if normalized_brand != product.brand:
                        remediated_product.brand = normalized_brand
                
                # Validate and clean price
                if product.price:
                    try:
                        price_decimal = Decimal(str(product.price))
                        if price_decimal <= 0:
                            logger.warning(f"Invalid price {product.price} for product {product.name}")
                            # Could set to None or use a default
                    except (InvalidOperation, ValueError):
                        logger.warning(f"Could not parse price {product.price} for product {product.name}")
            
            remediated_products.append(remediated_product)
        
        logger.info(f"Quality remediation completed for {len(remediated_products)} products")
        
        return remediated_products
    
    def _clean_product_name(self, name: str) -> str:
        """Clean product name of common issues."""
        if not name:
            return name
        
        # Remove HTML tags
        name = re.sub(r'<[^>]+>', '', name)
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove common parsing artifacts
        name = re.sub(r'\s*\|\s*', ' - ', name)  # Replace | with -
        name = re.sub(r'\s*&amp;\s*', ' & ', name)  # Replace HTML entities
        
        return name
    
    def _normalize_brand(self, brand: str) -> str:
        """Normalize brand name to standard form."""
        if not brand:
            return brand
        
        brand_lower = brand.lower().strip()
        
        # Check against known variations
        for standard_brand, variations in self.validator.brand_variations.items():
            if brand_lower in variations:
                return standard_brand.title()
        
        return brand.title()  # Default to title case


# Global instance
data_quality_manager = DataQualityManager()